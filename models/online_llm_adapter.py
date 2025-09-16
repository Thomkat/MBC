import os
import random
from typing import Optional
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from models.modules.aggregate import Aggregator, MLP
from models.modules.self_attention import TokenSelfAttend
from utils.misc import tqdm_distributed, decode_to_clean_text, exact_match, f1_score, lora_disabled
from models.modules.membank_comp import VectorQuantizer


def _get_batch_size(input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]) -> int:
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


class OnlineContextAdapter(nn.Module):
    def __init__(self, config, base_lm=None, enc_decoder=None, question_encoder=None):
        super().__init__()
        self.config = config
        self.base_lm = base_lm
        self._log_dir = config.log_dir if 'log_dir' in config else './logs'

        self.token_dim = self.config.model.token_dim
        self.set_base_lm(base_lm)

        self.output_size = self.token_dim
        self.num_actual_tokens = self.config.model.layer_num_virtual_tokens * self.base_num_layers * 2
        self.token_SA = TokenSelfAttend(self.token_dim, self.num_actual_tokens, self.config.model.num_virtual_tokens)
        self.enc_decoder = enc_decoder

        # Determine the hidden size for the amortization encoder-decoder
        if 't5' in self.config.model.amortization_network:
            enc_hidden_size = self.enc_decoder.encoder.config.hidden_size
        else:
            raise ValueError(f"Unsupported model type in config: {self.config.model.amortization_network}")

        self.mlps = nn.ModuleList([
            MLP(enc_hidden_size, self.token_dim)
            for _ in range(self.config.model.num_virtual_tokens)
        ])

        # Determine the hidden size for the question encoder
        if 't5' in self.config.model.amortization_network:
            q_enc_hidden_size = question_encoder.encoder.config.hidden_size
        else:
            raise ValueError(f"Unsupported model type in config: {self.config.model.amortization_network}")

        # Define aggregation network
        self.aggregator = Aggregator(
            config, question_encoder, self.token_dim,
            q_enc_hidden_size, self.config.model.num_virtual_tokens,
        )
        
        # Define the memory bank compressor
        self.vq = VectorQuantizer(
            num_codes=self.config.model.vq_num_codes,
            code_dim=self.token_dim,
            commitment_cost=self.config.model.vq_commitment_cost
        )

    def set_base_lm(self, base_lm):
        self.base_model_torch_dtype = base_lm.base_lm.dtype
        self.base_num_layers = base_lm.config.num_hidden_layers
        self.base_num_attention_heads = base_lm.config.num_attention_heads

    def freeze_param(self):
        # freeze everything in the base model except LoRA params
        for name, param in self.base_lm.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, update_batch, train=True):
        batch_size = _get_batch_size(update_batch['text_ids'], None)
        text_labels = update_batch['text_ids'].clone()
        text_labels[update_batch['text_attention'] != 1] = -100

        # Backpropagation dropout (only during training)
        if self.config.model.backprop_drop != 1.0 and train:
            update_batch["gen_q_ids_amort"] = update_batch["gen_q_ids_amort"][:int(batch_size * self.config.model.backprop_drop)]
            update_batch["gen_q_attn_mask_amort"] = update_batch["gen_q_attn_mask_amort"][:int(batch_size * self.config.model.backprop_drop)]
            update_batch['text_ids'] = update_batch['text_ids'][:int(batch_size * self.config.model.backprop_drop)]
            update_batch['text_attention'] = update_batch['text_attention'][:int(batch_size * self.config.model.backprop_drop)]
            text_labels = text_labels[:int(batch_size * self.config.model.backprop_drop)]
            update_batch['qa_ids'] = update_batch['qa_ids'][:int(batch_size * self.config.model.backprop_drop)]
            update_batch['qa_attention'] = update_batch['qa_attention'][:int(batch_size * self.config.model.backprop_drop)]
            update_batch['qa_target_ids'] = update_batch['qa_target_ids'][:int(batch_size * self.config.model.backprop_drop)]

        contexts_amort = self.context_amortize(update_batch["text_ids_amort"], update_batch["text_attention_amort"], train) # Amortize context
        quantized_contexts, vq_loss, vq_perplexity, codebook_indices = self.vq(contexts_amort) # Compress (quantize) it
        mod_latent = self.aggregator(update_batch["gen_q_ids_amort"], update_batch["gen_q_attn_mask_amort"], quantized_contexts) # Aggregate
        modulation = self.mod_latent_to_modulation(mod_latent) # Get modulation for current batch

        if self.training and self.config.drop_enable:
            self.base_lm.train()
        else:
            self.base_lm.eval()

        qa_output = self.base_lm(
            input_ids=update_batch['qa_ids'],
            attention_mask=update_batch['qa_attention'],
            labels=update_batch['qa_target_ids'],
            prompts=modulation
        )
        qa_loss = qa_output.loss

        # Calculate these metrics without gradients and without LoRA (pure base LLM)
        with torch.no_grad():
            with lora_disabled(self.base_lm):
                # initial text loss and qa outputs should be measured without prompts
                init_text_loss = self.base_lm(
                    input_ids=update_batch['text_ids'],
                    attention_mask=update_batch['text_attention'],
                    labels=text_labels,
                    prompts=None,
                ).loss
                init_qa_outputs = self.base_lm(
                    input_ids=update_batch['qa_ids'],
                    attention_mask=update_batch['qa_attention'],
                    labels=update_batch['qa_target_ids'],
                    prompts=None,
                )
                final_text_loss = self.base_lm(
                    input_ids=update_batch['text_ids'],
                    attention_mask=update_batch['text_attention'],
                    labels=text_labels,
                    prompts=modulation,
                ).loss

        metrics = {
            'text_loss': final_text_loss.item(),
            'text_gain_from_base': init_text_loss.item() - final_text_loss.item(),
            'qa_loss': qa_loss.item(),
            'qa_gain_from_base': init_qa_outputs.loss.item() - qa_loss.item(),
        }

        # Final total loss
        total_loss = qa_loss + self.config.model.vq_lambda * vq_loss

        metrics['vq_loss'] = vq_loss.item()
        metrics['vq_perplexity'] = vq_perplexity.item()
        metrics['total_loss'] = total_loss.item()
        
        # During online adaptation, return the codes (indices of the codebook) to be saved to the memory bank
        if not train:
            return total_loss, metrics, codebook_indices

        return total_loss, metrics

    def compute_qa_metrics(self, batch, context_summary_bank, top_k=1, no_adapt=False):
        em_correct = 0

        avg_f1s = []
        total_cnt = len(batch['gen_q_ids'])
        use_cache = False

        with torch.no_grad():
            kwargs = {}
            if not no_adapt:
                continuous_prompt = self.get_modulation_from_memorybank(
                    batch['gen_q_ids_amort'],
                    batch['gen_q_attn_mask_amort'],
                    context_summary_bank
                )

                if self.config.model.base_model in ['Llama2_7b']:
                    # continuous_prompt = DynamicCache.from_legacy_cache(continuous_prompt)
                    use_cache=True
                kwargs['prompts'] = continuous_prompt

            outs = self.base_lm.generate(
                input_ids=batch['gen_q_ids'],
                attention_mask=batch["gen_q_attn_mask"],
                use_cache=use_cache,
                max_length=batch['gen_q_ids'].shape[1] + 16,
                num_return_sequences=top_k,
                num_beams=1,
                peft_generation=not no_adapt,
                do_sample=False,
                early_stopping=False, # this is for beam search (not used for validation)
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

            dec = decode_to_clean_text(self.tokenizer, outs)
            texts = decode_to_clean_text(self.tokenizer, batch['gen_q_ids'])
            targets = decode_to_clean_text(self.tokenizer, batch['answer_ids'])
            for i in range(len(batch['gen_q_ids'])):
                answer = targets[i]

                predicted_answers = [dec[i * top_k + j][len(texts[i]):] for j in range(top_k)]

                em = 0
                f_1s = []
                for pred_ans in predicted_answers:
                    if exact_match(pred_ans, answer, match_length=False):
                        em = 1
                    f_1s.append(f1_score(pred_ans, answer))
                em_correct += em
                avg_f1s.append(np.mean(f_1s))

        return {'EM': em_correct / total_cnt, 'F1 Score': np.mean(avg_f1s).item()}

    def load(self, target_path=None):
        state = torch.load(target_path, map_location="cpu")
        self.load_state_dict(state['state_dict'], strict=False)

        # load LoRA into base LM if present
        if 'lora_state_dict' in state and (self.base_lm is not None):
            missing, unexpected = self.base_lm.load_state_dict(state['lora_state_dict'], strict=False)
            assert not any("lora_" in k for k in missing), f"Missing LoRA params: {missing}"
            assert not any("lora_" in k for k in unexpected), f"Unexpected LoRA params: {unexpected}"
            print(f'\nLoaded KV-LoRA params.')
            
        print(f'Loaded checkpoint {target_path}\n')


    def save(self, epoch, log_dir=None, file_name=None, main_process=False):
        if main_process:
            if log_dir is None:
                log_dir = self._log_dir
            file_name = file_name or f'state{epoch}.pt'
            os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

            # collect amortized side
            temp_base_lm = self.base_lm
            self.base_lm = None
            state_dict = self.state_dict()
            self.base_lm = temp_base_lm

            # collect only LoRA params from base_lm wrapper
            lora_state = {k: v for k, v in temp_base_lm.state_dict().items()
                        if ("lora_A" in k) or ("lora_B" in k)}
            assert len(lora_state) > 0, "No LoRA weights found to save!"

            torch.save(
                dict(state_dict=state_dict, lora_state_dict=lora_state),
                f'{os.path.join(log_dir, "checkpoints", file_name)}'
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def validate_amort_compress(self, val_loader, main_process=False):
        metrics_dic = defaultdict(lambda: [])

        context_bank = []
        for _, batch in tqdm_distributed(
                main_process, enumerate(val_loader), desc='Validation: Amortizing Context & Compressing Memory Bank',
                position=1, total=len(val_loader)
        ):
            _, metrics, context = self.forward(batch, train=False)
            context_bank.append(context)

            for k, v in metrics.items():
                metrics_dic[f'[VAL]{k}'].append(v)

            del batch  # free memory

        context_bank = torch.cat(context_bank, dim=0)
        return {k: np.mean(v) for k, v in metrics_dic.items()}, context_bank

    def validate_aggregate(self, val_gen_loader, context_bank=None, no_adapt=False, main_process=False):
        metrics_dic = defaultdict(lambda: [])

        for _, batch in tqdm_distributed(
                main_process, enumerate(val_gen_loader), desc='Validation: Aggregating context',
                position=2, total=len(val_gen_loader)
        ):

            # For large memory banks, enable hierarchical aggregation to reduce memory complexity to O(MT)
            if self.config.model.hierarchical_aggregation:
                for context_window in self.config.model.context_window_list:
                    modulation_hierarch = self.get_hierarchical_context(
                        batch, context_bank.clone().detach(), context_window
                    )
                    qa_metrics_hierarch = self.compute_qa_metrics(
                        batch, modulation_hierarch, no_adapt=no_adapt
                    )
                    for k, v in qa_metrics_hierarch.items():
                        metrics_dic[f'[VAL][Context{context_window}]{k}'].append(v)

                del modulation_hierarch
            # Else aggregate the whole bank at once
            else:
                qa_metrics = self.compute_qa_metrics(batch, context_bank, no_adapt=no_adapt)
                for k, v in qa_metrics.items():
                    metrics_dic[f'[VAL]{k}'].append(v)

            del batch

        return {k: np.mean(v) for k, v in metrics_dic.items()}

    def context_amortize(self, indices, text_attention, train):
        if self.config.model.backprop_drop != 1.0 and train:
            indices_lift = indices[:int(indices.shape[0] * self.config.model.backprop_drop)]
            text_attention_lift = text_attention[:int(text_attention.shape[0] * self.config.model.backprop_drop)]
            indices_no_lift = indices[int(indices.shape[0] * self.config.model.backprop_drop):]
            text_attention_no_lift = text_attention[int(text_attention.shape[0] * self.config.model.backprop_drop):]
            with torch.no_grad():
                hidden_state_no_lift = self.enc_decoder(
                    input_ids=indices_no_lift,
                    attention_mask=text_attention_no_lift,
                )
                context_vectors_no_lift = []
                for i, mlp in enumerate(self.mlps):
                    context_vectors_no_lift.append(mlp(hidden_state_no_lift[:, i:i + 1, :]))

                context_vectors_no_lift = torch.cat(context_vectors_no_lift, dim=1)
            hidden_state_lift = self.enc_decoder(
                input_ids=indices_lift,
                attention_mask=text_attention_lift,
            )
            context_vectors_lift = []
            for i, mlp in enumerate(self.mlps):
                context_vectors_lift.append(mlp(hidden_state_lift[:, i:i + 1, :]))

            context_vectors_lift = torch.cat(context_vectors_lift, dim=1)
            context_vectors = torch.cat([context_vectors_lift, context_vectors_no_lift], dim=0)
        else:
            hidden_state = self.enc_decoder(
                input_ids=indices,
                attention_mask=text_attention,
            )

            context_vectors = []
            for i, mlp in enumerate(self.mlps):
                context_vectors.append(mlp(hidden_state[:, i:i + 1, :]))

            context_vectors = torch.cat(context_vectors, dim=1)
        return context_vectors  # compress (quantize) those

    def get_hierarchical_context(self, batch, context_summary_bank, context_window):
        # Decompress the entire bank of indices at the start
        with torch.no_grad():
            context_bank = self.vq.embedding(context_summary_bank)

        while len(context_bank) > context_window:
            context_bank = self.hierarchical_aggregate(
                batch['gen_q_ids_amort'],
                batch['gen_q_attn_mask_amort'],
                context_bank,
                context_window
            )
        return context_bank # this is the modulation after hierarchical aggregation

    def hierarchical_aggregate(self, gen_q_ids_amort, gen_q_attn_mask_amort,
                               context_summary_bank, hierarchy_context_size):
        chunk_iter = range(0, gen_q_ids_amort.shape[1], hierarchy_context_size)
        aggregated_modulations = []
        for k in chunk_iter:
            context_chunk = context_summary_bank[k:k + hierarchy_context_size]
            predict = self.aggregator(
                gen_q_ids_amort, gen_q_attn_mask_amort, context_chunk
            )
            aggregated_modulations.append(predict)
        return torch.cat(aggregated_modulations, dim=0)

    def get_modulation_from_memorybank(self, gen_q_ids_amort, gen_q_attn_mask_amort, context_summary_bank):
        # Check if the bank is already decompressed (float) or still compressed (int)
        if torch.is_floating_point(context_summary_bank):
            # It's already vectors, so use it directly
            reconstructed_vectors = context_summary_bank
        else:
            # It's integer indices, so decompress it
            with torch.no_grad():
                reconstructed_vectors = self.vq.embedding(context_summary_bank)

        mod_latent = self.aggregator(gen_q_ids_amort, gen_q_attn_mask_amort, reconstructed_vectors)
        modulation = self.mod_latent_to_modulation(mod_latent)
        return modulation

    def mod_latent_to_modulation(self, mod_latent):
        repeat_token = self.token_SA(mod_latent)
        past_key_values = repeat_token.view(
            len(repeat_token),
            self.config.model.layer_num_virtual_tokens,
            self.base_num_layers * 2,
            self.base_num_attention_heads,
            self.token_dim // self.base_num_attention_heads,
        )
        modulation = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return modulation