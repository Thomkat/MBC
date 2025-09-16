import math
import torch
import torch.nn as nn


class LoRALinearAdd(nn.Module):
    def __init__(self, orig_linear: nn.Module, in_features: int, out_features: int, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.enabled = True
        self.orig = orig_linear
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # start as no-op

        for p in self.orig.parameters():
            p.requires_grad = False

    def forward(self, x):
        if not self.enabled:
            return self.orig(x)
        return self.orig(x) + self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class LoRAKVForGPT2CAttn(nn.Module):
    # GPT-2 attn.c_attn projects to [Q||K||V] (3*H)
    # We add LoRA only to the K and V slices, leaving Q untouched
    def __init__(self, orig_c_attn: nn.Module, hidden_size: int, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.enabled = True
        self.orig = orig_c_attn
        self.hidden_size = hidden_size
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(hidden_size, r, bias=False)
        self.lora_B = nn.Linear(r, 2 * hidden_size, bias=False)  # only KV
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.orig.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.orig(x)
        if not self.enabled:
            return base_out
        H = self.hidden_size
        delta_kv = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        # pad a zero Q slice on the left
        zero_q = base_out.new_zeros(base_out.shape[:-1] + (H,))
        delta = torch.cat([zero_q, delta_kv], dim=-1)
        return base_out + delta


def apply_lora_kv_gpt2(model, last_n_layers=1, r=4, alpha=8, dropout=0.0):
    H = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size
    for block in model.transformer.h[-last_n_layers:]:
        block.attn.c_attn = LoRAKVForGPT2CAttn(block.attn.c_attn, H, r=r, alpha=alpha, dropout=dropout)


def apply_lora_kv_llama(model, last_n_layers=1, r=4, alpha=8, dropout=0.0):
    layers = model.model.layers
    for layer in layers[-last_n_layers:]:
        attn = layer.self_attn
        attn.k_proj = LoRALinearAdd(attn.k_proj, attn.k_proj.in_features, attn.k_proj.out_features,
                                    r=r, alpha=alpha, dropout=dropout)
        attn.v_proj = LoRALinearAdd(attn.v_proj, attn.v_proj.in_features, attn.v_proj.out_features,
                                    r=r, alpha=alpha, dropout=dropout)


def apply_lora_kv(model, model_type: str, last_n_layers=1, r=4, alpha=8, dropout=0.0):
    if model_type == "gpt2":
        apply_lora_kv_gpt2(model, last_n_layers, r, alpha, dropout)
    elif model_type == "llama":
        apply_lora_kv_llama(model, last_n_layers, r, alpha, dropout)
    else:
        raise ValueError(f"KV-LoRA not implemented for model_type={model_type}")