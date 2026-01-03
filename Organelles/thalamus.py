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


class Organelle_Thalamus(nn.Module):
    def __init__(self, dim_visual=768, dim_text=768, max_keep=96):
        super().__init__()
        self.max_keep = max_keep

        self.scorer = nn.Sequential(
            nn.Linear(dim_visual + dim_text, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.register_token = nn.Parameter(torch.randn(1, 1, dim_visual))

    def forward(self, dense_visual, text_embedding=None):
        B, N, D = dense_visual.shape

        if text_embedding is not None:
            ctx = text_embedding.mean(dim=1, keepdim=True).expand(-1, N, -1)
            scorer_in = torch.cat([dense_visual, ctx], dim=-1)
        else:
            scorer_in = torch.cat([dense_visual, torch.zeros_like(dense_visual)], dim=-1)

        scores = self.scorer(scorer_in).squeeze(-1)

        num_keep = min(self.max_keep, N)
        num_keep = max(1, num_keep)

        top_k_idx = torch.topk(scores, num_keep, dim=1).indices
        top_k_idx, _ = torch.sort(top_k_idx, dim=1)

        gather_idx = top_k_idx.unsqueeze(-1).expand(-1, -1, D)
        sparse_visual = torch.gather(dense_visual, 1, gather_idx)

        register = self.register_token.expand(B, -1, -1)
        final_visual = torch.cat([sparse_visual, register], dim=1)

        return final_visual, top_k_idx