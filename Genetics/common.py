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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192):
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


# --- GAME THEORETIC PRUNING LAYER ---
class StrategicLinear(nn.Linear):
    def __init__(self, in_features, out_features, cost_coefficient=1e-4, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.alpha_logits = nn.Parameter(torch.randn(out_features) * 0.1 + 5.0)
        self.cost_coefficient = cost_coefficient

    def get_participation(self):
        return torch.sigmoid(self.alpha_logits)

    def forward(self, input):
        alpha = self.get_participation()
        masked_weight = self.weight * alpha.unsqueeze(1)
        return F.linear(input, masked_weight, self.bias)

    def game_loss(self):
        return self.cost_coefficient * torch.sum(self.get_participation())


# --- MANIFOLD CONSTRAINED UTILS ---
def sinkhorn_knopp(log_matrix, iterations=5):
    # SAFETY CLAMP: Prevent exp() from exploding
    # 30.0 is safe for float32 (e^30 is large but finite)
    # 10.0 is safe for float16
    safe_log = torch.clamp(log_matrix, max=10.0)

    M = torch.exp(safe_log)
    for _ in range(iterations):
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-6)
        M = M / (M.sum(dim=-2, keepdim=True) + 1e-6)
    return M


# --- DEEP DELTA LEARNING OPERATOR ---
class DeltaOperator(nn.Module):
    def __init__(self, dim, beta_scale=2.0):
        super().__init__()
        self.beta_proj = nn.Linear(dim, 1)
        self.k_proj = nn.Linear(dim, dim)
        self.beta_scale = beta_scale

        nn.init.constant_(self.beta_proj.weight, 0)
        nn.init.constant_(self.beta_proj.bias, -2.0)

    def forward(self, x):
        beta = torch.sigmoid(self.beta_proj(x)) * self.beta_scale
        k_raw = self.k_proj(x)
        k_norm = torch.norm(k_raw, dim=-1, keepdim=True) + 1e-6
        k = k_raw / k_norm
        proj_scalar = torch.sum(k * x, dim=-1, keepdim=True)
        rank1_term = proj_scalar * k
        return (1 - beta) * x + beta * rank1_term