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
from Genetics.common import RotaryEmbedding


class MultimodalBase(nn.Module):
    """
    The Foundation for all AEIOU Architectures.
    Enforces consistent sensory fusion order:
    Visual (V) -> Audio (A) -> Control (C) -> Text (T)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- UNIFIED PROJECTION LAYERS ---
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)

        # We check attributes to allow flexible configs, defaulting to standard AEIOU sizes
        vis_dim = getattr(config, 'vis_dim', 768)
        aud_dim = getattr(config, 'aud_dim', 128)
        mot_dim = getattr(config, 'mot_dim', 64)

        self.vis_proj = nn.Linear(vis_dim, config.embed_dim)
        self.aud_proj = nn.Linear(aud_dim, config.embed_dim)
        self.mot_proj = nn.Linear(mot_dim, config.embed_dim)

        # --- SHARED ROPE GENERATOR ---
        # Calculates the frequency matrix once for the maximum context
        # Assumes config has n_heads. If not, defaults to 12.
        n_heads = getattr(config, 'n_heads', 12)
        head_dim = config.embed_dim // n_heads

        # Max seq len 16384 covers standard 8k models + vision tokens
        self.rope = RotaryEmbedding(head_dim, max_seq_len=16384)

    def embed_inputs(self, v, a, t, c=None):
        """
        Fuses modalities into a single causal stream.
        """
        # Safety clamp for text tokens (prevents index errors if vocab size changes)
        if t.max() >= self.config.vocab_size:
            t = torch.clamp(t, 0, self.config.vocab_size - 1)

        parts = []

        # 1. Visual
        if v is not None and v.numel() > 0:
            parts.append(self.vis_proj(v))

        # 2. Audio
        if a is not None and a.numel() > 0:
            parts.append(self.aud_proj(a))

        # 3. Control / Motion
        if c is not None and c.numel() > 0:
            parts.append(self.mot_proj(c))

        # 4. Text (Always present)
        parts.append(self.tok_emb(t))

        # Concatenate along sequence dimension (dim=1)
        x = torch.cat(parts, dim=1)
        return x

    def get_positional_embeddings(self, x):
        """
        Generates RoPE cos/sin tables for the current sequence length.
        """
        device = x.device
        seq_len = x.shape[1]

        # Get pre-computed cos/sin from common.RotaryEmbedding
        rope_cos, rope_sin = self.rope(seq_len)

        return rope_cos.to(device), rope_sin.to(device)