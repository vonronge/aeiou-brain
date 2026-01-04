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
    def __init__(self, dim_visual=768, dim_text=768, max_keep=96, golgi=None):
        super().__init__()
        self.max_keep = max_keep
        self.golgi = golgi
        self.call_counter = 0

        # The Gating Network (Lightweight Scorer)
        # Decides importance of each patch based on content + context
        self.scorer = nn.Sequential(
            nn.Linear(dim_visual + dim_text, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Register Tokens (Global Context)
        # We append these to the pruned sequence so the model always has
        # a "summary" view, even if specific patches are dropped.
        self.register_token = nn.Parameter(torch.randn(1, 1, dim_visual) * 0.02)

    def _log(self, msg, tag="INFO"):
        if self.golgi and self.call_counter % 100 == 0:  # Reduce spam
            method = getattr(self.golgi, tag.lower(), self.golgi.info)
            method(msg, source="Thalamus")

    def forward(self, dense_visual, text_embedding=None):
        """
        dense_visual: [Batch, N_patches, Dim] (e.g. 196 patches from ViT)
        text_embedding: [Batch, Seq, Dim] (Optional context for top-down attention)

        Returns:
        - sparse_visual: [Batch, max_keep + 1, Dim]
        - indices: The indices of kept patches (for visualization)
        """
        B, N, D = dense_visual.shape
        self.call_counter += 1

        # 1. Prepare Context
        if text_embedding is not None:
            # Mean pool text to get a global context vector
            ctx = text_embedding.mean(dim=1, keepdim=True).expand(-1, N, -1)
            scorer_in = torch.cat([dense_visual, ctx], dim=-1)
        else:
            # Zero context (Bottom-up saliency only)
            zeros = torch.zeros(B, N, 768, device=dense_visual.device)  # Assume text dim 768
            scorer_in = torch.cat([dense_visual, zeros], dim=-1)

        # 2. Score Patches
        # scores: [Batch, N]
        scores = self.scorer(scorer_in).squeeze(-1)

        # 3. Select Top-K
        num_keep = min(self.max_keep, N)
        num_keep = max(1, num_keep)  # Safety

        # Get top k indices
        # indices: [Batch, num_keep]
        top_k_scores, top_k_idx = torch.topk(scores, num_keep, dim=1)

        # Sort indices to preserve spatial ordering (mostly)
        top_k_idx, _ = torch.sort(top_k_idx, dim=1)

        # 4. Gather Features
        # Expand indices for gather: [Batch, num_keep, Dim]
        gather_idx = top_k_idx.unsqueeze(-1).expand(-1, -1, D)
        sparse_visual = torch.gather(dense_visual, 1, gather_idx)

        # 5. Append Register Token
        # Always keep the learnable register token at the end
        register = self.register_token.expand(B, -1, -1)
        final_visual = torch.cat([sparse_visual, register], dim=1)

        # Telemetry
        if self.call_counter % 50 == 0:
            compression = (1 - (num_keep / N)) * 100
            self._log(f"Gating Active: {N} -> {num_keep} patches ({compression:.1f}% compression)", "INFO")

        return final_visual, top_k_idx