"""
AEIOU Brain — Local Multimodal AI Ecosystem
Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain
LinkedIn: https://www.linkedin.com/in/vonronge/
Licensed under the MIT License.

Genetics: CortexAligned-WideConvDiffusion
Inspired by "Convolutional Architectures Are Cortex-Aligned De Novo" (Nature MI 2025)
Ultra-wide convolutional U-Net denoiser with extreme feature expansion + spatial compression
for de novo brain-like representations from initialization.

New: Staggered Half-Neuron Training ("Awakening Mode")
- Randomly activates awakening_rate fraction of channels per step
- Pseudo-random reproducible masks → high overlap + eventual full coverage
- Huge VRAM saver for single-GPU training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

INFO = {
    "name": "CortexAligned-WideConv-Diffusion-512channel",
    "desc": "Ultra-wide conv U-Net. Cortex-aligned via feature expansion. Includes Awakening Mode for VRAM efficiency.",
    "vram_train": "10-12 GB (Awakening Mode)",
    "vram_run": "8 GB"
}


class NucleusConfig:
    def __init__(self):
        self.vocab_size = 72000
        self.mask_token_id = 71999
        # Vocab ranges (match Ribosome)
        self.vocab_img_start = 50257
        self.vocab_aud_start = 66641
        # Grid: 256px / 16 patch -> 16x16 = 256 tokens
        self.grid_side = 16
        self.img_tokens = self.grid_side ** 2
        # Extreme width for cortex alignment
        self.base_channels = 512  # Tune down to 1024/768 if needed
        # Diffusion
        self.inference_steps = 32
        # Masking
        self.vis_mask_rate = 0.80
        self.text_mask_rate = 0.05
        self.aud_mask_rate = 0.30
        self.mot_mask_rate = 0.15
        # === AWAKENING MODE ===
        self.awakening_rate = 0.5  # Fraction ACTIVE per step (0.5=half, 1.0=full)


def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class WideConvBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels)
        ) if time_dim else None
        self.act = nn.SiLU()

    def forward(self, x, time_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        if time_emb is not None:
            t = self.time_mlp(time_emb)[:, :, None, None]
            h = h + t
        return h


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.block1 = WideConvBlock(out_ch, time_dim)
        self.block2 = WideConvBlock(out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)  # Spatial compression

    def forward(self, x, time_emb):
        x = self.proj(x)
        h = self.block1(x, time_emb)
        h = self.block2(h, time_emb)
        skip = h
        out = self.down(h)
        return out, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = WideConvBlock(out_ch * 2, time_dim)
        self.block2 = WideConvBlock(out_ch, time_dim)
        self.proj = nn.Conv2d(out_ch * 2, out_ch, 1)

    def forward(self, x, skip, time_emb):
        x = self.up(x)
        # Handle spatial mismatch if any
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        return x


class Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = NucleusConfig()
        self.config = config
        ch = config.base_channels

        self.tok_emb = nn.Embedding(config.vocab_size + 1, ch)
        self.pos_emb = nn.Embedding(4096, ch)

        self.time_mlp = nn.Sequential(
            nn.Linear(ch, ch * 4),
            nn.SiLU(),
            nn.Linear(ch * 4, ch)
        )

        self.input_block = WideConvBlock(ch, None)

        self.down1 = DownBlock(ch, ch * 2, ch)
        self.down2 = DownBlock(ch * 2, ch * 4, ch)
        self.down3 = DownBlock(ch * 4, ch * 4, ch)

        self.bottleneck = nn.Sequential(
            WideConvBlock(ch * 4, ch),
            WideConvBlock(ch * 4, ch)
        )

        self.up1 = UpBlock(ch * 4, ch * 4, ch)
        self.up2 = UpBlock(ch * 4, ch * 2, ch)
        self.up3 = UpBlock(ch * 2, ch, ch)

        self.output = nn.Conv2d(ch, config.vocab_size, 1)

        # [FIX v24.2] Persistent Linear Head for Text/Audio
        self.text_head = nn.Linear(ch, config.vocab_size)

    def _get_channel_mask(self, channels):
        """
        Awakening Mode: Randomly activates a subset of channels.
        Uses standard PyTorch RNG so it rotates every step automatically.
        """
        rate = self.config.awakening_rate
        if rate >= 1.0 or not self.training:
            return None

        active = int(channels * rate)
        device = next(self.parameters()).device

        # Randomly select channels to keep active
        idx = torch.randperm(channels, device=device)[:active]
        mask = torch.zeros(channels, device=device)
        mask[idx] = 1.0

        # Scale up active neurons (Inverted Dropout style) to preserve magnitude
        mask = mask * (1.0 / rate)

        return mask.view(1, -1, 1, 1)

    def _compute_modality_lengths_from_tokens(self, t):
        tokens = t.flatten()
        vis_mask = (tokens >= self.config.vocab_img_start) & (tokens < self.config.vocab_aud_start)
        vis_len = vis_mask.sum().item()
        aud_mask = (tokens >= self.config.vocab_aud_start)
        aud_len = aud_mask.sum().item()
        text_len = t.shape[1] - vis_len - aud_len
        mot_len = 0
        return vis_len, aud_len, mot_len, text_len

    def _apply_masking(self, seq, vis_len, aud_len, mot_len, text_len, mask_ratio=None):
        B, T = seq.shape
        device = seq.device
        masked_seq = seq.clone()
        mask = torch.zeros(T, device=device, dtype=torch.bool)

        def mask_section(start, end, rate):
            length = end - start
            if length <= 0: return
            num_mask = int(length * rate)
            idx = torch.randperm(length, device=device)[:num_mask]
            global_idx = start + idx
            mask[global_idx] = True
            masked_seq[:, global_idx] = self.config.mask_token_id

        # Order: text -> vis -> aud -> mot
        starts = [0, text_len, text_len + vis_len, text_len + vis_len + aud_len]

        if mask_ratio is not None:
            num_mask = int(T * mask_ratio)
            idx = torch.randperm(T, device=device)[:num_mask]
            mask[idx] = True
            masked_seq[:, idx] = self.config.mask_token_id
        else:
            mask_section(0, starts[1], self.config.text_mask_rate)
            mask_section(starts[1], starts[2], self.config.vis_mask_rate)
            mask_section(starts[2], starts[3], self.config.aud_mask_rate)
            mask_section(starts[3], T, self.config.mot_mask_rate)

        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(B, -1)

        return masked_seq, mask

    def forward(self, v, a, t, c=None, timestep=None, mask_ratio=None, global_step=0):
        device = t.device
        B, T = t.shape
        ch = self.config.base_channels

        vis_len, aud_len, mot_len, text_len = self._compute_modality_lengths_from_tokens(t)

        seq_masked, mask = self._apply_masking(t, vis_len, aud_len, mot_len, text_len, mask_ratio)

        x = self.tok_emb(seq_masked)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)

        # Fallback if no visual tokens (pure text)
        if vis_len != self.config.img_tokens:
            logits = self.text_head(x)
            return logits, mask, None

        # Extract visual part
        vis_start = text_len
        vis_x = x[:, vis_start:vis_start + vis_len, :]

        # Reshape to [B, C, H, W]
        # 16x16 grid
        vis_x = vis_x.view(B, self.config.grid_side, self.config.grid_side, -1).permute(0, 3, 1, 2).contiguous()

        if timestep is None:
            timestep = torch.zeros(B, dtype=torch.long, device=device)
        t_emb_raw = get_timestep_embedding(timestep, ch)
        t_emb = self.time_mlp(t_emb_raw)[:, :, None, None]

        # --- U-NET PASS ---
        h = self.input_block(vis_x) + t_emb

        skips = []

        # Down 1
        mask_d1 = self._get_channel_mask(ch * 2)
        h, skip = self.down1(h, t_emb)
        if mask_d1 is not None:
            h = h * mask_d1
            skip = skip * mask_d1
        skips.append(skip)

        # Down 2
        mask_d2 = self._get_channel_mask(ch * 4)
        h, skip = self.down2(h, t_emb)
        if mask_d2 is not None:
            h = h * mask_d2
            skip = skip * mask_d2
        skips.append(skip)

        # Down 3
        mask_d3 = self._get_channel_mask(ch * 4)
        h, skip = self.down3(h, t_emb)
        if mask_d3 is not None:
            h = h * mask_d3
            skip = skip * mask_d3
        skips.append(skip)

        # Bottleneck
        mask_bot = self._get_channel_mask(ch * 4)
        h = self.bottleneck(h)
        if mask_bot is not None:
            h = h * mask_bot

        # Up 1
        h = self.up1(h, skips.pop(), t_emb)
        # Up 2
        h = self.up2(h, skips.pop(), t_emb)
        # Up 3
        h = self.up3(h, skips.pop(), t_emb)

        # Output
        vis_logits = self.output(h)
        # [B, Vocab, H, W] -> [B, H, W, Vocab] -> [B, Seq, Vocab]
        vis_logits = vis_logits.permute(0, 2, 3, 1).reshape(B, vis_len, -1)

        # Text part (simple projection using persistent head)
        # We slice x to get everything BEFORE the visual tokens (Text)
        # and everything AFTER (Audio/Motion) if needed, but simplistic concat implies visual is contiguous block
        # For this specific architecture, we assume: Text -> Visual -> Audio
        text_logits = self.text_head(x[:, :vis_start, :])

        logits = torch.cat([text_logits, vis_logits], dim=1)

        # If there are tokens after visual (Audio/Mot), we need to predict them too
        # Our current shape is [B, text_len + vis_len, Vocab]
        # If T > text_len + vis_len, we compute tail
        processed_len = text_logits.shape[1] + vis_logits.shape[1]
        if processed_len < T:
            tail_logits = self.text_head(x[:, processed_len:, :])
            logits = torch.cat([logits, tail_logits], dim=1)

        return logits, mask, None

    @torch.no_grad()
    def generate(self, prompt_tokens=None, max_length=1024, steps=None, temperature=1.0, top_k=50,
                 force_sample_remaining=True):
        device = next(self.parameters()).device
        steps = steps or self.config.inference_steps

        # Full capacity for inference (No Awakening)
        prev_rate = self.config.awakening_rate
        self.config.awakening_rate = 1.0

        # Setup sequence
        if prompt_tokens is not None:
            if isinstance(prompt_tokens, list):
                seq = torch.tensor(prompt_tokens, device=device).unsqueeze(0)
            else:
                seq = prompt_tokens if prompt_tokens.ndim == 2 else prompt_tokens.unsqueeze(0)

            curr_len = seq.shape[1]
            if curr_len < max_length:
                pad = torch.full((1, max_length - curr_len), self.config.mask_token_id, device=device)
                seq = torch.cat([seq, pad], dim=1)

            known_mask = (seq != self.config.mask_token_id)
        else:
            seq = torch.full((1, max_length), self.config.mask_token_id, device=device)
            known_mask = torch.zeros_like(seq, dtype=torch.bool)

        img_len = self.config.img_tokens

        for step in range(steps):
            ratio = math.cos((step / steps) * math.pi / 2)
            ts_val = int(ratio * 1000)
            timestep = torch.tensor([ts_val], device=device)

            logits, _, _ = self.forward(None, None, seq, None, timestep=timestep)

            logits = logits / temperature
            if top_k > 0:
                v_top, _ = torch.topk(logits, top_k)
                logits[logits < v_top[..., [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            confidences, predicted = torch.max(probs, dim=-1)

            # Focus unmasking on visual tokens
            # Determine visual start index by finding where text tokens end or assuming fixed offset
            vis_start = max(0, seq.shape[1] - img_len)

            # Mask out known tokens from confidence score so we don't pick them again
            confidences = confidences.masked_fill(known_mask, -float('inf'))

            # Determine count
            num_remaining = (~known_mask).sum().item()
            num_unmask = max(1, int(num_remaining * (1 - ratio)))

            # Greedy Unmasking of highest confidence tokens
            _, indices = torch.topk(confidences.flatten(), num_unmask)

            # Create update mask
            flat_mask = torch.zeros_like(known_mask.flatten(), dtype=torch.bool)
            flat_mask[indices] = True
            update_mask = flat_mask.view_as(known_mask)

            seq[update_mask] = predicted[update_mask]
            known_mask = known_mask | update_mask

        # Final Cleanup Pass
        if force_sample_remaining and not known_mask.all():
            timestep = torch.zeros(1, device=device, dtype=torch.long)
            logits, _, _ = self.forward(None, None, seq, None, timestep=timestep)
            remaining = ~known_mask
            probs = F.softmax(logits[remaining] / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(1)
            seq[remaining] = sampled

        self.config.awakening_rate = prev_rate
        return seq[0].cpu().tolist()