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
from Genetics.common import RMSNorm, StrategicLinear, sinkhorn_knopp, DeltaOperator
import math
import random

INFO = {
    "name": "MaskedDiffusion-mHC",
    "desc": "Bidirectional Discrete Diffusion. Vocab-aware masking. Ideal for Dreaming/Refinement.",
    "vram_train": "9 GB",
    "vram_run": "5 GB"
}


class NucleusConfig:
    def __init__(self):
        self.vocab_size = 72000
        self.embed_dim = 768
        self.n_layers = 12
        self.n_heads = 12
        self.dropout = 0.1
        self.vis_dim = 768
        self.aud_dim = 128
        self.mot_dim = 64
        self.cost_coefficient = 1e-4
        self.num_streams = 4

        # Diffusion Config
        self.inference_steps = 32
        self.mask_token_id = 71999

        # Vocab Boundaries (Must match Ribosome)
        self.vocab_img_start = 50257
        self.vocab_aud_start = 66641

        # Scheduling
        self.use_modality_masking = False
        self.use_cross_modal_masking = False
        self.cross_modal_prob = 0.25

        self.cross_modal_min_modalities = 1
        self.cross_modal_max_modalities = 2

        self.vis_mask_rate = 0.80
        self.aud_mask_rate = 0.60
        self.mot_mask_rate = 0.30
        self.text_mask_rate = 0.15


class DiffusionMHCBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.embed_dim
        self.n = config.num_streams
        self.stream_dim = self.dim // self.n

        self.time_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.delta_op = DeltaOperator(self.dim)
        self.gate_proj = nn.Linear(self.dim, 3 * self.n)
        self.phi_res = nn.Parameter(torch.randn(self.n, self.n) * 0.02)

        self.ln_1 = RMSNorm(self.dim)
        self.ln_2 = RMSNorm(self.dim)

        self.mlp = nn.Sequential(
            StrategicLinear(self.dim, 4 * self.dim, cost_coefficient=config.cost_coefficient),
            nn.GELU(),
            StrategicLinear(4 * self.dim, self.dim, cost_coefficient=config.cost_coefficient),
            nn.Dropout(config.dropout)
        )

        self.attn = nn.MultiheadAttention(self.dim, config.n_heads, batch_first=True)

    def forward(self, x, t_emb):
        if t_emb is not None:
            x = x + self.time_proj(t_emb)

        x_delta = self.delta_op(x)

        x_state = x_delta.mean(dim=1)
        gates = self.gate_proj(x_state)
        g_res = torch.sigmoid(gates[:, 2 * self.n:]).unsqueeze(1).unsqueeze(-1)
        H_res = sinkhorn_knopp(self.phi_res, iterations=15)

        x_streams = x_delta.view(x.shape[0], x.shape[1], self.n, self.stream_dim)
        x_perm = (x_streams * g_res).permute(0, 1, 3, 2)
        res_act = (x_perm @ H_res.T).permute(0, 1, 3, 2)
        mhc_out = res_act.reshape(x.shape[0], x.shape[1], self.dim)

        normed = self.ln_1(x_delta)
        attn_out, _ = self.attn(normed, normed, normed, is_causal=False)

        normed2 = self.ln_2(x_delta + attn_out)
        ffn_out = self.mlp(normed2)

        return x_delta + attn_out + ffn_out + mhc_out


class Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None: config = NucleusConfig()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size + 1, config.embed_dim)
        self.vis_proj = nn.Linear(config.vis_dim, config.embed_dim)
        self.aud_proj = nn.Linear(config.aud_dim, config.embed_dim)
        self.mot_proj = nn.Linear(config.mot_dim, config.embed_dim)

        self.timestep_emb = nn.Embedding(1001, config.embed_dim)
        self.pos_emb = nn.Embedding(4096, config.embed_dim)

        self.blocks = nn.ModuleList([DiffusionMHCBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.embed_dim)
        self.head = StrategicLinear(config.embed_dim, config.vocab_size,
                                    cost_coefficient=config.cost_coefficient, bias=False)

        self.mask_token_id = config.mask_token_id

    def _compute_modality_lengths_from_tokens(self, t):
        tokens = t.flatten()
        vis_mask = (tokens >= self.config.vocab_img_start) & (tokens < self.config.vocab_aud_start)
        vis_len = vis_mask.sum().item()
        aud_mask = (tokens >= self.config.vocab_aud_start)
        aud_len = aud_mask.sum().item()
        text_len = t.shape[1] - vis_len - aud_len
        mot_len = 0
        return vis_len, aud_len, mot_len, text_len

    def _get_section_starts(self, t, vis_len, aud_len, mot_len, text_len):
        starts = [0]
        starts.append(starts[-1] + vis_len)
        starts.append(starts[-1] + aud_len)
        starts.append(starts[-1] + mot_len)
        return starts

    def _apply_masking(self, seq, vis_len, aud_len, mot_len, text_len, mask_ratio=None):
        B, T = seq.shape
        device = seq.device
        mask = torch.zeros(T, device=device, dtype=torch.bool)
        masked_seq = seq.clone()

        section_starts = self._get_section_starts(seq, vis_len, aud_len, mot_len, text_len)

        if mask_ratio is not None:
            num_mask = int(T * mask_ratio)
            mask_idx = torch.randperm(T, device=device)[:num_mask]
            mask[mask_idx] = True

        elif self.config.use_cross_modal_masking and random.random() < self.config.cross_modal_prob:
            sections = [
                ('vis', section_starts[0], section_starts[1]),
                ('aud', section_starts[1], section_starts[2]),
                ('mot', section_starts[2], section_starts[3]),
                ('text', section_starts[3], T)
            ]
            present = [s for s in sections if s[2] - s[1] > 0]

            if len(present) >= 2:
                num_to_mask = random.randint(self.config.cross_modal_min_modalities,
                                             min(self.config.cross_modal_max_modalities, len(present) - 1))
                to_mask = random.sample(present, num_to_mask)
                for _, start, end in to_mask:
                    mask[start:end] = True
            else:
                num_mask = int(T * 0.4)
                mask_idx = torch.randperm(T, device=device)[:num_mask]
                mask[mask_idx] = True

        elif self.config.use_modality_masking:
            def mask_section(start, end, rate):
                length = end - start
                if length == 0: return
                num_mask = int(length * rate)
                section_idx = torch.randperm(length, device=device)[:num_mask]
                mask[start + section_idx] = True

            mask_section(section_starts[0], section_starts[1], self.config.vis_mask_rate)
            mask_section(section_starts[1], section_starts[2], self.config.aud_mask_rate)
            mask_section(section_starts[2], section_starts[3], self.config.mot_mask_rate)
            mask_section(section_starts[3], T, self.config.text_mask_rate)

        else:
            num_mask = int(T * 0.3)
            mask_idx = torch.randperm(T, device=device)[:num_mask]
            mask[mask_idx] = True

        # --- FIX: Ensure 2D mask for [B, T] ---
        # If mask is 1D [T], expand it to [B, T]
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(B, -1)

        masked_seq[mask] = self.mask_token_id
        return masked_seq, mask

    def forward(self, v, a, t, c=None, timestep=None, mask_ratio=None):
        device = t.device
        seq = t
        vis_len, aud_len, mot_len, text_len = self._compute_modality_lengths_from_tokens(seq)

        seq_masked, mask = self._apply_masking(seq, vis_len, aud_len, mot_len, text_len, mask_ratio)

        x = self.tok_emb(seq_masked)

        positions = torch.arange(0, x.shape[1], device=device)
        if x.shape[1] > self.pos_emb.num_embeddings:
            x = x[:, :self.pos_emb.num_embeddings]
            positions = positions[:self.pos_emb.num_embeddings]
        x = x + self.pos_emb(positions)

        if timestep is None: timestep = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        t_emb = self.timestep_emb(timestep).unsqueeze(1)

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits, mask, None

    @torch.no_grad()
    def generate(self, prompt_tokens=None, max_length=1024, steps=None, temperature=1.0, top_k=50,
                 force_sample_remaining=True):
        device = next(self.parameters()).device
        steps = steps or self.config.inference_steps

        # --- SHAPE SAFETY INITIALIZATION ---
        if prompt_tokens is not None:
            if isinstance(prompt_tokens, list):
                seq = torch.tensor(prompt_tokens, device=device).unsqueeze(0)
            elif isinstance(prompt_tokens, torch.Tensor):
                if prompt_tokens.ndim == 1:
                    seq = prompt_tokens.unsqueeze(0)
                else:
                    seq = prompt_tokens

            # Setup Padding and Mask
            known_mask = torch.zeros_like(seq, dtype=torch.bool)
            pad_len = max_length - seq.shape[1]
            if pad_len > 0:
                pad = torch.full((1, pad_len), self.mask_token_id, device=device)
                seq = torch.cat([seq, pad], dim=1)
                # Pad known_mask with Falses (unknowns)
                pad_mask = torch.zeros((1, pad_len), dtype=torch.bool, device=device)
                known_mask = torch.cat([known_mask, pad_mask], dim=1)
        else:
            seq = torch.full((1, max_length), self.mask_token_id, dtype=torch.long, device=device)
            known_mask = torch.zeros((1, max_length), dtype=torch.bool, device=device)

        # --- ITERATIVE REFINEMENT ---
        for step in range(steps):
            ratio = math.cos((step / steps) * math.pi / 2)
            ts_val = int(ratio * 1000)
            timestep = torch.tensor([ts_val], device=device)

            # Manual Forward Pass (Bypass masking logic)
            # 1. Embed
            x = self.tok_emb(seq)
            positions = torch.arange(0, x.shape[1], device=device)
            x = x + self.pos_emb(positions)
            t_emb = self.timestep_emb(timestep).unsqueeze(1)

            # 2. Denoise
            for block in self.blocks:
                x = block(x, t_emb)
            x = self.ln_f(x)
            logits = self.head(x)

            # 3. Sampling
            logits = logits / temperature
            if top_k > 0:
                v_top, _ = torch.topk(logits, top_k)
                logits[logits < v_top[..., [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            confidences, predicted = torch.max(probs, dim=-1)

            # Ignore what we already know (set confidence to -inf so we don't pick it)
            confidences = confidences.masked_fill(known_mask, -float('inf'))

            # Calculate how many to unmask
            num_remaining = (~known_mask).sum().item()
            num_unmask = max(1, int(num_remaining * (1 - ratio)))

            # Greedy Unmasking
            _, indices = torch.topk(confidences.flatten(), num_unmask)

            # Scatter updates
            # Create a flat mask for update
            flat_mask = torch.zeros_like(confidences.flatten(), dtype=torch.bool)
            flat_mask[indices] = True
            update_mask = flat_mask.view_as(known_mask)

            # Update sequence and known_mask
            seq[update_mask] = predicted[update_mask]
            known_mask = known_mask | update_mask

        # --- FORCE SAMPLE REMAINING (Anti-Blank Fix) ---
        if force_sample_remaining and not known_mask.all():
            remaining = ~known_mask
            # Final clean pass
            timestep = torch.zeros(1, device=device, dtype=torch.long)

            x = self.tok_emb(seq)
            positions = torch.arange(0, x.shape[1], device=device)
            x = x + self.pos_emb(positions)
            t_emb = self.timestep_emb(timestep).unsqueeze(1)
            for block in self.blocks:
                x = block(x, t_emb)
            x = self.ln_f(x)
            final_logits = self.head(x)

            # Sample
            final_logits = final_logits / temperature
            probs = F.softmax(final_logits[remaining], dim=-1)
            if top_k > 0:
                v, _ = torch.topk(probs, top_k)
                probs[probs < v[..., [-1]]] = 0
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)

            sampled = torch.multinomial(probs, 1).squeeze(1)
            seq[remaining] = sampled

        return seq[0].cpu().tolist()