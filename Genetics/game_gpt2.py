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
from Genetics.common import RMSNorm

# --- GUI METADATA ---
INFO = {
    "name": "Equilibrium-GPT2",
    "desc": "Game-Theoretic Pruning. Weights play a non-cooperative game to justify their existence.",
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
        self.cost_coefficient = 1e-4  # The cost of playing the game


class StrategicLinear(nn.Linear):
    """
    A Linear layer where weights must 'pay' to participate.
    Based on: 'Pruning as a Game: Equilibrium-Driven Sparsification' (2025)
    """

    def __init__(self, in_features, out_features, cost_coefficient=1e-4, **kwargs):
        super().__init__(in_features, out_features, **kwargs)

        # The "Participation Variable" (alpha).
        # Initialized to >1.0 (fully participating) using inverse sigmoid logic.
        # logits of 5.0 -> sigmoid(5.0) ~= 0.993
        self.alpha_logits = nn.Parameter(torch.randn(out_features) * 0.1 + 5.0)
        self.cost_coefficient = cost_coefficient

    def get_participation(self):
        # Sigmoid ensures alpha is strictly [0, 1]
        return torch.sigmoid(self.alpha_logits)

    def forward(self, input):
        alpha = self.get_participation()

        # Strategic Interaction: Mask the weights column-wise
        # alpha shape: [out_features] -> unsqueeze -> [out_features, 1] for broadcasting
        masked_weight = self.weight * alpha.unsqueeze(1)

        return F.linear(input, masked_weight, self.bias)

    def game_loss(self):
        # The Payoff: Minimize sum of alphas (Sparsity) scaled by cost
        alpha = self.get_participation()
        return self.cost_coefficient * torch.sum(alpha)


class StrategicBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.n_heads, batch_first=True)

        # We assume standard Attention doesn't need pruning logic for now,
        # or we could replace the internal QKV projections if we manually implemented Attention.
        # For simplicity, we apply the Game only to the MLP layers below.

        self.ln_2 = nn.LayerNorm(config.embed_dim)

        # Replace standard Linears with StrategicLinears
        self.mlp = nn.Sequential(
            StrategicLinear(config.embed_dim, 4 * config.embed_dim, cost_coefficient=config.cost_coefficient),
            nn.GELU(),
            StrategicLinear(4 * config.embed_dim, config.embed_dim, cost_coefficient=config.cost_coefficient),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        attn_mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float('-inf'), device=x.device), diagonal=1)

        # Standard Attention (Cooperative Phase)
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x),
                                attn_mask=attn_mask, is_causal=False)
        x = x + attn_out

        # Strategic MLP (Competitive Phase)
        x = x + self.mlp(self.ln_2(x))
        return x


class Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None: config = NucleusConfig()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.context_len * 8, config.embed_dim)

        # Adapters
        self.vis_proj = nn.Linear(config.vis_dim, config.embed_dim)
        self.aud_proj = nn.Linear(config.aud_dim, config.embed_dim)
        self.mot_proj = nn.Linear(config.mot_dim, config.embed_dim)

        # Strategic Blocks
        self.blocks = nn.ModuleList([StrategicBlock(config) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.embed_dim)
        # The Final Head is also a player in the game
        self.head = StrategicLinear(config.embed_dim, config.vocab_size, cost_coefficient=config.cost_coefficient,
                                    bias=False)

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

        curr_len = x.shape[1]
        positions = torch.arange(0, curr_len, device=device)
        if curr_len > self.pos_emb.num_embeddings:
            x = x[:, -self.pos_emb.num_embeddings:, :]
            positions = positions[:self.pos_emb.num_embeddings]

        x = x + self.pos_emb(positions)

        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        return self.head(x), None, None