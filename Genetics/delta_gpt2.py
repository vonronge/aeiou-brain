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
from Genetics.common import RMSNorm, StrategicLinear, sinkhorn_knopp, DeltaOperator
from Genetics.base_multimodal import MultimodalBase

INFO = {
    "name": "Delta-mHC-Muon",
    "desc": "Deep Delta Learning + mHC. Refactored. (v23.1)",
    "vram_train": "7 GB",
    "vram_run": "3 GB"
}


class NucleusConfig:
    def __init__(self):
        self.vocab_size = 72000  # Standardized Safety Buffer
        self.embed_dim = 768
        self.context_len = 1024
        self.n_layers = 12
        self.n_heads = 12
        self.dropout = 0.1
        self.vis_dim = 768
        self.aud_dim = 128
        self.mot_dim = 64
        self.cost_coefficient = 1e-4
        self.num_streams = 4


class DeltaMHCBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n = config.num_streams
        self.dim = config.embed_dim
        self.stream_dim = self.dim // self.n

        # --- DELTA & MHC ---
        self.delta_op = DeltaOperator(self.dim)
        self.gate_proj = nn.Linear(self.dim, 3 * self.n)
        self.phi_res = nn.Parameter(torch.randn(self.n, self.n) * 0.02)

        self.ln_1 = RMSNorm(self.dim)
        self.ln_2 = RMSNorm(self.dim)

        # --- GAME THEORETIC MLP ---
        self.mlp = nn.Sequential(
            StrategicLinear(self.dim, 4 * self.dim, cost_coefficient=config.cost_coefficient),
            nn.GELU(),
            StrategicLinear(4 * self.dim, self.dim, cost_coefficient=config.cost_coefficient),
            nn.Dropout(config.dropout)
        )
        self.attn = nn.MultiheadAttention(self.dim, config.n_heads, batch_first=True)

    def forward(self, x, rope_cos, rope_sin):
        # 1. DEEP DELTA UPDATE
        x_delta = self.delta_op(x)

        # 2. mHC ROUTING
        x_state = x_delta.mean(dim=1)
        gates = self.gate_proj(x_state)
        g_res = torch.sigmoid(gates[:, 2 * self.n:]).unsqueeze(1).unsqueeze(-1)
        H_res = sinkhorn_knopp(self.phi_res, iterations=15)

        x_streams = x_delta.view(x.shape[0], x.shape[1], self.n, self.stream_dim)
        x_perm = (x_streams * g_res).permute(0, 1, 3, 2)
        res_act = (x_perm @ H_res.T).permute(0, 1, 3, 2)
        mhc_out = res_act.reshape(x.shape[0], x.shape[1], self.dim)

        # 3. ATTENTION
        normed = self.ln_1(x_delta)
        attn_out, _ = self.attn(normed, normed, normed, is_causal=True)

        # 4. MLP
        normed2 = self.ln_2(x_delta + attn_out)
        ffn_out = self.mlp(normed2)

        # 5. FUSION
        return x_delta + attn_out + ffn_out + mhc_out


class Model(MultimodalBase):
    def __init__(self, config=None):
        if config is None: config = NucleusConfig()
        super().__init__(config)

        # Hybrid Positional Embeddings (Standard AEIOU Pattern)
        self.pos_emb = nn.Embedding(config.context_len * 8, config.embed_dim)

        self.blocks = nn.ModuleList([DeltaMHCBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.embed_dim)

        self.head = StrategicLinear(config.embed_dim, config.vocab_size,
                                    cost_coefficient=config.cost_coefficient,
                                    bias=False)

    def forward(self, v, a, t, c=None):
        device = t.device

        # Base Class Embed
        x = self.embed_inputs(v, a, t, c)

        # Positional Add
        curr_len = x.shape[1]
        if curr_len > self.pos_emb.num_embeddings:
            x = x[:, -self.pos_emb.num_embeddings:, :]
        positions = torch.arange(0, x.shape[1], device=device)
        x = x + self.pos_emb(positions)

        # RoPE Gen
        rope_cos, rope_sin = self.get_positional_embeddings(x)

        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)

        x = self.ln_f(x)
        return self.head(x), None, None