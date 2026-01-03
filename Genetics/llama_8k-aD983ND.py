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
from Genetics.common import RMSNorm, SwiGLU, apply_rope
from Genetics.base_multimodal import MultimodalBase

INFO = {
    "name": "Llama-8k",
    "desc": "High Context Variant (8192). Refactored Base. (v23.1)",
    "vram_train": "10 GB",
    "vram_run": "6 GB"
}


class NucleusConfig:
    def __init__(self):
        self.vocab_size = 72000
        self.embed_dim = 768
        self.context_len = 1024  # Internal block context
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

        # 1. Attention
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim)

        # Apply RoPE (Passed from Base Class)
        # Slicing for safety if rope cache is larger than current T
        cos = rope_cos[:T, :]
        sin = rope_sin[:T, :]

        q = apply_rope(q, cos, sin).transpose(1, 2)
        k = apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(y)

        # 2. Feed Forward
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Model(MultimodalBase):
    def __init__(self, config=None):
        if config is None: config = NucleusConfig()
        super().__init__(config)

        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, v, a, t, c=None):
        # 1. Embed (Unified)
        x = self.embed_inputs(v, a, t, c)

        # 2. RoPE (Unified)
        rope_cos, rope_sin = self.get_positional_embeddings(x)

        # 3. Llama Blocks
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)

        x = self.norm(x)
        return self.lm_head(x), None, None