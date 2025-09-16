import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM
from models.ptuningv2_wrapper import BaseModelPTV2Wrapper
from hydra.utils import to_absolute_path
from models.t5_wrapper import T5ForwardWrapper
from models.online_llm_adapter import OnlineContextAdapter
from models.modules.kv_lora import apply_lora_kv
from omegaconf import OmegaConf


def prepare_prompt_learning_config(config, model_config):
    if config.model.token_dim is None:
        if hasattr(model_config, "hidden_size"):
            token_dim = model_config.hidden_size
        elif hasattr(model_config, "n_embd"):
            token_dim = model_config.n_embd
        elif hasattr(model_config, "d_model"):
            token_dim = model_config.d_model
        else:
            raise ValueError("Please specify `token_dim` in `config`")
    else:
        token_dim = config.model.token_dim

    return token_dim

def get_base_model(cfg, accelerator=None):
    kwargs = {}
    if cfg.model.base_model in ['Llama2_7b'] or cfg.quant_type is not None:
        if cfg.quant_type == 'nf4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=False,
            )
        elif cfg.quant_type == 'int8':
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"quant_type {cfg.quant_type} not supported")
        kwargs['quantization_config'] = bnb_config
        kwargs['torch_dtype'] = torch.float32

    if accelerator is not None and cfg.quant_type is not None:
            kwargs['device_map'] = {"": accelerator.device}

    if cfg.model.base_model == 'Llama2_7b':
        base_lm = LlamaForCausalLM.from_pretrained(cfg.model.llama_cache_dir, **kwargs)
    else:
        base_lm = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model, cache_dir=cfg.CACHE_DIR, **kwargs
        )

    if cfg.model.token_dim is None:
        cfg.model.token_dim = prepare_prompt_learning_config(cfg, base_lm.config)

    if cfg.model.base_model == 'Llama2_7b':
        model_type = 'llama'
    else:
        model_type = 'gpt2'

    # Inject LoRA BEFORE freezing
    if cfg.model.enable_lora:
        apply_lora_kv(
            base_lm,
            model_type=model_type,
            last_n_layers=cfg.model.lora_last_n_layers,
            r=cfg.model.lora_r,
            alpha=cfg.model.lora_alpha,
            dropout=cfg.model.lora_dropout,
        )

    base_lm = BaseModelPTV2Wrapper(base_lm, cfg)

    base_lm.train()
    return base_lm


def get_mbc_model(cfg, base_lm, tokenizer=None, tokenizer_amort=None, accelerator=None):
    # Determine which model class to use for the amortization network
    if 't5' in cfg.model.amortization_network:
        AmortModelClass = T5ForwardWrapper
    else:
        raise NameError(f"Unknown or unsupported amortization model type in '{cfg.model.amortization_network}'")

    if 't5' in cfg.model.amortization_network:
        kwargs = {}
        if cfg.quant_type is not None:
            if cfg.quant_type == 'nf4':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=False,
                )
            elif cfg.quant_type == 'int8':
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise ValueError(f"quant_type {cfg.quant_type} not supported")
            kwargs['quantization_config'] = bnb_config
            kwargs['torch_dtype'] = torch.float32

        if accelerator is not None and cfg.quant_type is not None:
            kwargs['device_map'] = {"": accelerator.device}

        # Load encoder-decoder model for amortization
        amort_encoder_decoder = AmortModelClass.from_pretrained(
            cfg.model.amortization_network, cache_dir=cfg.CACHE_DIR, **kwargs
        )
        
        # T5 models access the encoder directly via the .encoder attribute
        encoder_config_hidden_size = amort_encoder_decoder.encoder.config.hidden_size
        learnable_prompts = torch.randn(1, cfg.model.num_virtual_tokens, encoder_config_hidden_size) * .02
        amort_encoder_decoder.learnable_prompts = nn.Parameter(learnable_prompts)
        amort_encoder_decoder.num_virtual_tokens = cfg.model.num_virtual_tokens
        if hasattr(amort_encoder_decoder, 'lm_head'): del amort_encoder_decoder.lm_head

        # Load question encoder
        question_model_name = cfg.model.amortization_network if cfg.model.question_encoder is None else cfg.model.question_encoder
        question_encoder = AmortModelClass.from_pretrained(
            question_model_name, cache_dir=cfg.CACHE_DIR, **kwargs
        )
        
        # T5 models access the encoder directly via the .encoder attribute
        q_encoder_config_hidden_size = question_encoder.encoder.config.hidden_size
        learnable_prompts = torch.randn(1, cfg.model.num_virtual_tokens, q_encoder_config_hidden_size) * .02
        question_encoder.learnable_prompts = nn.Parameter(learnable_prompts)
        question_encoder.num_virtual_tokens = cfg.model.num_virtual_tokens
        if hasattr(question_encoder, 'lm_head'): del question_encoder.lm_head
        
        mbc_model = OnlineContextAdapter(
            cfg, base_lm=base_lm, enc_decoder=amort_encoder_decoder, question_encoder=question_encoder
        )
        mbc_model.tokenizer = tokenizer
        mbc_model.tokenizer_amort = tokenizer_amort
    else:
        raise NameError('Unknown model type')

    load_path = OmegaConf.select(cfg, "load_path", default=None)
    if load_path is not None:
        mbc_model.load(target_path=to_absolute_path(load_path))

    mbc_model.freeze_param() # Freeze base LLM except for LoRA

    return mbc_model