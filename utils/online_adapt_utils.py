import torch
from tqdm import tqdm
import numpy as np
import time
from utils.misc import decode_to_clean_text, exact_match, f1_score
import json

def online_adapt_qa_eval(cfg, dataloader, model, tokenizer, compressed_memory_bank=None, device=None,
            top_k=1, diversity_penalty=10., num_beam_groups=4, num_beams=12, mbc_model=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    total_cnt = 0
    em_correct = 0
    avg_f1s = []
    max_f1s = []

    total_inference_time = 0

    if compressed_memory_bank is not None:
        compressed_memory_bank = compressed_memory_bank.to(device)

    use_cache = True if cfg.model.base_model in ['Llama2_7b'] else False

    detailed_results_data = []

    for batch in tqdm(dataloader):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            if isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    batch[k] = [x.to(device) for x in v]

        torch.cuda.empty_cache()
        kwargs = {}
        if compressed_memory_bank is not None:
            with torch.no_grad():
                modulation = mbc_model.get_modulation_from_memorybank(
                    batch['gen_q_ids_amort'],
                    batch['gen_q_attn_mask_amort'],
                    compressed_memory_bank
                )
            kwargs['peft_generation'] = True
            kwargs['prompts'] = modulation

        if batch['gen_q_ids'].shape[1] > 256:
            batch['gen_q_ids'] = batch['gen_q_ids'][:, -256:]
            batch['gen_q_attn_mask'] = batch['gen_q_attn_mask'][:, -256:]

        inference_start_time = time.time()
        with torch.no_grad():
            outs = model.generate(
                input_ids=batch['gen_q_ids'],
                attention_mask=batch["gen_q_attn_mask"],
                use_cache=use_cache,
                max_length=batch['gen_q_ids'].shape[1] + 16,
                num_return_sequences=top_k,
                num_beam_groups=num_beam_groups,
                num_beams=num_beams,
                diversity_penalty=diversity_penalty,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        inference_end_time = time.time()
        total_inference_time += (inference_end_time - inference_start_time)

        dec = decode_to_clean_text(tokenizer, outs)
        texts = decode_to_clean_text(tokenizer, batch['gen_q_ids'])
        targets = decode_to_clean_text(tokenizer, batch['answer_ids'])

        for i in range(len(batch['gen_q_ids'])):
            total_cnt += 1
            answer = targets[i]

            predicted_answers = [dec[i * top_k + j][len(texts[i]):] for j in range(top_k)]

            em = 0
            f1s = []
            for pred_ans in predicted_answers:
                if exact_match(pred_ans, answer, match_length=False):
                    em = 1
                f1s.append(f1_score(pred_ans, answer))
            em_correct += em
            
            detailed_results_data.append([
                texts[i],
                answer,
                json.dumps(predicted_answers, ensure_ascii=False),
                dec[i],
                json.dumps(f1s),
                em
            ])

            avg_f1s.append(np.mean(f1s))
            max_f1s.append(np.max(f1s))

    avg_inference_ms = (1000.0 * total_inference_time / total_cnt) if total_cnt else 0.0
    em_rate = (em_correct / total_cnt) if total_cnt else 0.0
    avg_f1_mean = float(np.mean(avg_f1s)) if avg_f1s else 0.0
    avg_f1_std  = float(np.std(avg_f1s))  if avg_f1s else 0.0
    max_f1_mean = float(np.mean(max_f1s)) if max_f1s else 0.0
    max_f1_std  = float(np.std(max_f1s))  if max_f1s else 0.0

    summary_metrics = {
        'queries_evaluated': total_cnt,
        'em_correct': em_correct,
        'em_rate': em_rate,
        'avg_f1_mean': avg_f1_mean,
        'avg_f1_std': avg_f1_std,
        'max_f1_mean': max_f1_mean,
        'max_f1_std': max_f1_std,
        'avg_inference_ms': avg_inference_ms,
    }
    
    return detailed_results_data, summary_metrics


def context_summarization_compression(dataloader, mbc_model, train=False):
    print('Summarizing Context & Compressing Memory Bank')
    prompts = []
    mbc_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()

    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            if isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    batch[k] = [x.to(device) for x in v]
        with torch.no_grad():
            contexts_amort = mbc_model.context_amortize(batch["text_ids_amort"], batch["text_attention_amort"], train=train)
            _, _, _, codebook_indices = mbc_model.vq(contexts_amort)
            
        prompts.append(codebook_indices.detach().cpu())
        # break
    
    end_time = time.time()
    total_adaptation_time = end_time - start_time
    num_documents = len(dataloader.dataset)
    avg_adaptation_time = total_adaptation_time / num_documents if num_documents > 0 else 0

    return torch.cat(prompts, dim=0), avg_adaptation_time