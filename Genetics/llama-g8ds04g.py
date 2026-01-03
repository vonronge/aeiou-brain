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
from Genetics.common import RMSNorm, SwiGLU, RotaryEmbedding, apply_rope

# --- GUI METADATA ---
INFO = {
    "name": "Tetra-Llama",
    "desc": "Multimodal Llama (RoPE + SwiGLU). Standardized Fusion Order.",
    "vram_train": "8 GB",
    "vram_run": "4 GB"
}


class NucleusConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.embed_dim = 768
        self.context_len = 1024
        self.n_layers = 12
        self.n_heads = 12
        self.vis_dim = 768
        self.aud_dim = 128
        self.mot_dim = 64


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.attn_norm = RMSNorm(config.embed_dim)
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.o_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.ffn_norm = RMSNorm(config.embed_dim)
        self.ffn = SwiGLU(config.embed_dim, int(config.embed_dim * 2.6))

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim)

        cos = rope_cos[:T, :]
        sin = rope_sin[:T, :]

        q = apply_rope(q, cos, sin).transpose(1, 2)
        k = apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(y)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None: config = NucleusConfig()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.vis_proj = nn.Linear(config.vis_dim, config.embed_dim)
        self.aud_proj = nn.Linear(config.aud_dim, config.embed_dim)
        self.mot_proj = nn.Linear(config.mot_dim, config.embed_dim)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.rope = RotaryEmbedding(config.embed_dim // config.n_heads, max_seq_len=4096)

    def forward(self, v, a, t, c=None):
        B, T = t.shape
        device = t.device
        x = self.tok_emb(t)

        # STANDARDIZED FUSION ORDER: Visual -> Audio -> Motion -> Text
        if v is not None and v.numel() > 0:
            v_emb = self.vis_proj(v)
            x = torch.cat([v_emb, x], dim=1)

        if a is not None and a.numel() > 0:
            a_emb = self.aud_proj(a)
            x = torch.cat([x, a_emb], dim=1)

        if c is not None and c.numel() > 0:
            c_emb = self.mot_proj(c)
            x = torch.cat([x, c_emb], dim=1)

        seq_len = x.shape[1]
        rope_cos, rope_sin = self.rope(seq_len)
        rope_cos, rope_sin = rope_cos.to(device), rope_sin.to(device)

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)

        x = self.norm(x)
        return self.lm_head(x), None, None