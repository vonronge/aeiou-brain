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
from Genetics.common import RMSNorm

# --- GUI METADATA ---
INFO = {
    "name": "Tetra-GPT2",
    "desc": "Classic Transformer. Standardized Fusion Order.",
    "vram_train": "6 GB",
    "vram_run": "2 GB"
}


class NucleusConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.embed_dim = 768
        self.context_len = 1024
        self.n_layers = 12
        self.n_heads = 12
        self.dropout = 0.1
        self.vis_dim = 768
        self.aud_dim = 128
        self.mot_dim = 64


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.n_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        B, T, C = x.shape
        attn_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x),
                                attn_mask=attn_mask,
                                is_causal=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None: config = NucleusConfig()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.context_len * 8, config.embed_dim)
        self.vis_proj = nn.Linear(config.vis_dim, config.embed_dim)
        self.aud_proj = nn.Linear(config.aud_dim, config.embed_dim)
        self.mot_proj = nn.Linear(config.mot_dim, config.embed_dim)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, v, a, t, c=None):
        B, T = t.shape
        device = t.device

        # Strict Order: V -> A -> C -> T
        parts = []
        if v is not None and v.numel() > 0: parts.append(self.vis_proj(v))
        if a is not None and a.numel() > 0: parts.append(self.aud_proj(a))
        if c is not None and c.numel() > 0: parts.append(self.mot_proj(c))
        parts.append(self.tok_emb(t))

        x = torch.cat(parts, dim=1)

        # GPT2 Absolute Positional Embeddings
        # We must extend positions to cover the new total length
        curr_len = x.shape[1]
        positions = torch.arange(0, curr_len, device=device)
        if curr_len > self.pos_emb.num_embeddings:
            # If too long, slice the input (this is a rough truncate, but prevents crash)
            x = x[:, -self.pos_emb.num_embeddings:, :]
            positions = positions[:self.pos_emb.num_embeddings]

        x = x + self.pos_emb(positions)

        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        return self.head(x), None, None