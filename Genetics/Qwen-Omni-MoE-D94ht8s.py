"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain
LinkedIn: https://www.linkedin.com/in/vonronge/

Licensed under the MIT License.
See the LICENSE file in the repository root for full license text.

This file is part of AEIOU Brain, a personal open-source project
for experimenting with hybrid autoregressive + diffusion architectures,
persistent memory graphs, and local multimodal training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- GUI METADATA ---
INFO = {
    "name": "Qwen-Omni-MoE (8k)",
    "desc": "Native Multimodal MoE. Uses Top-2 Routing & M-RoPE logic. 8192 ctx.",
    "vram_train": "12 GB",
    "vram_run": "8 GB"
}


# --- SHARED COMPONENTS (Local copy for self-containment) ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=16384):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, seq_len):
        return self.cos[:seq_len, :], self.sin[:seq_len, :]


def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    head_dim = x.shape[-1]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)


# --- ARCHITECTURE CONFIG ---
class NucleusConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.embed_dim = 768
        self.context_len = 1024  # Internal attention window
        self.n_layers = 12
        self.n_heads = 12

        # MoE Settings (DeepSeek-VL2 Style)
        self.num_experts = 8
        self.active_experts = 2

        # Modality Dimensions
        self.vis_dim = 768
        self.aud_dim = 128
        self.mot_dim = 64


# --- MOE LAYER (The Expert System) ---
class SwiGLUExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SparseMoE(nn.Module):
    def __init__(self, dim, num_experts=8, active_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Create 8 small experts instead of 1 giant MLP
        # Scale hidden dim down slightly per expert to keep param count sane
        expert_dim = int(dim * 1.5)
        self.experts = nn.ModuleList([SwiGLUExpert(dim, expert_dim) for _ in range(num_experts)])

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        # Router logits: [Batch*Seq, NumExperts]
        gate_logits = self.router(x_flat)

        # Select Top-K Experts
        weights, indices = torch.topk(gate_logits, self.active_experts, dim=-1)
        weights = F.softmax(weights, dim=-1)

        output = torch.zeros_like(x_flat)

        # Iterate over active experts and compute
        # Note: This is a naive sequential loop implementation (easier to read/debug than optimized kernels)
        for i in range(self.active_experts):
            expert_idx = indices[:, i]
            expert_weight = weights[:, i, None]

            for e_idx in range(self.num_experts):
                # Find tokens assigned to this expert at this rank
                mask = (expert_idx == e_idx)
                if mask.any():
                    tokens = x_flat[mask]
                    processed = self.experts[e_idx](tokens)
                    output[mask] += processed * expert_weight[mask]

        return output.view(B, T, C)


# --- MAIN BLOCK ---
class OmniBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.attn_norm = RMSNorm(config.embed_dim)

        # Attention
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.o_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        # MoE Feed Forward
        self.ffn_norm = RMSNorm(config.embed_dim)
        self.moe = SparseMoE(config.embed_dim, config.num_experts, config.active_experts)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        h = self.attn_norm(x)

        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim)

        # Apply RoPE
        # Safety slicing for variable sequence lengths
        curr_cos = rope_cos[:T, :]
        curr_sin = rope_sin[:T, :]

        q = apply_rope(q, curr_cos, curr_sin).transpose(1, 2)
        k = apply_rope(k, curr_cos, curr_sin).transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        x = x + self.o_proj(y)

        # Apply MoE
        x = x + self.moe(self.ffn_norm(x))
        return x


# --- THE MODEL ---
class Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None: config = NucleusConfig()
        self.config = config

        # Core Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)

        # Multimodal Adapters (Qwen2-VL style "Naive" projection)
        self.vis_proj = nn.Linear(config.vis_dim, config.embed_dim)
        self.aud_proj = nn.Linear(config.aud_dim, config.embed_dim)
        self.mot_proj = nn.Linear(config.mot_dim, config.embed_dim)

        # M-RoPE Simulators: Modality Type Embeddings
        # 0=Text, 1=Vis, 2=Aud, 3=Mot
        self.type_emb = nn.Embedding(4, config.embed_dim)

        self.layers = nn.ModuleList([OmniBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.embed_dim)

        # Dual Heads (NExT-OMNI style)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.velocity_head = nn.Linear(config.embed_dim, config.embed_dim, bias=False)  # For future Flow Matching

        # Match legacy saves (8192)
        self.rope = RotaryEmbedding(config.embed_dim // config.n_heads, max_seq_len=8192)

    def forward(self, v, a, t, c=None):
        B, T = t.shape
        device = t.device

        # 1. Embed Text
        x = self.tok_emb(t) + self.type_emb(torch.tensor(0, device=device))

        # 2. Early Fusion with Type Embeddings
        if a is not None and a.numel() > 0:
            a_emb = self.aud_proj(a) + self.type_emb(torch.tensor(2, device=device))
            x = torch.cat([a_emb, x], dim=1)

        if v is not None and v.numel() > 0:
            v_emb = self.vis_proj(v) + self.type_emb(torch.tensor(1, device=device))
            x = torch.cat([x, v_emb], dim=1)

        if c is not None and c.numel() > 0:
            c_emb = self.mot_proj(c) + self.type_emb(torch.tensor(3, device=device))
            x = torch.cat([x, c_emb], dim=1)

        # 3. RoPE Generation
        seq_len = x.shape[1]
        rope_cos, rope_sin = self.rope(seq_len)
        rope_cos, rope_sin = rope_cos.to(device), rope_sin.to(device)

        # 4. Deep Transformer Flow
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)

        x = self.norm(x)

        # 5. Multi-Head Output
        logits = self.lm_head(x)

        # We can optionally return velocity fields here if we were doing Flow Matching training
        # For now, we stick to AR logits
        return logits, None, None