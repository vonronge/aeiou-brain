"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Cytoplasm:
The active medium where learning occurs.
Consolidates training loops, gradient scaling, and nursery safety logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import math
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Dict, Any, List

# Import LobeHandle type for type hinting if available
try:
    from Organelles.lobe_manager import LobeHandle
except ImportError:
    LobeHandle = Any


@dataclass
class TrainConfig:
    """Configuration for a training run."""
    epochs: int = 1
    max_steps: Optional[int] = None
    autosave_interval: int = 100  # Steps

    # Nursery Settings (Safety Clamps)
    nursery_active: bool = True
    clip_grad_norm: float = 1.0
    use_fp32: bool = False

    # Loss Scaling (Min, Max)
    # If None, no clamping is applied
    loss_clamp_prediction: Optional[tuple] = (0.1, 2.5)
    loss_clamp_game: Optional[tuple] = (0.001, 15.0)
    loss_clamp_aux: Optional[tuple] = None


class Organelle_Cytoplasm:
    def __init__(self, device: str):
        self.device = device
        self.stop_requested = False
        self.is_paused = False

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "step": [],  # fn(step, loss_dict)
            "epoch": [],  # fn(epoch)
            "autosave": [],  # fn(step)
            "finished": [],  # fn()
            "error": []  # fn(exception)
        }

    def register_callback(self, event: str, fn: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(fn)

    def stop(self):
        self.stop_requested = True

    def pause(self):
        self.is_paused = not self.is_paused

    def _trigger(self, event: str, *args):
        for fn in self._callbacks.get(event, []):
            try:
                fn(*args)
            except Exception as e:
                print(f"[Cytoplasm] Callback Error ({event}): {e}")

    def train(self, config: TrainConfig, lobe: LobeHandle, dataset_iterator, mode: str = "ar"):
        """
        Main entry point. Runs in the CALLER'S thread (usually a worker thread).
        dataset_iterator: A generator yielding batches (v, a, t, c, target/mask)
        """
        self.stop_requested = False
        self.is_paused = False

        lobe.train()
        step_count = 0

        print(f"[Cytoplasm] Starting {mode.upper()} training on Lobe {lobe.id}...")

        try:
            for epoch in range(1, config.epochs + 1):
                if self.stop_requested: break

                self._trigger("epoch", epoch)

                # Iterate over the provided generator/loader
                for batch in dataset_iterator:
                    if self.stop_requested: break

                    # Pause handling
                    while self.is_paused: time.sleep(0.1)

                    # --- UNPACK BATCH ---
                    # Standardized format expected from plugins:
                    # AR: (v, a, t, c, targets)
                    # Diff: (v, a, t, c, mask_ratio) or similar
                    if mode == "ar":
                        v, a, t, c, targets = batch
                        loss_dict = self._step_ar(lobe, config, v, a, t, c, targets)
                    elif mode == "diffusion":
                        # Diffusion usually generates its own targets/masks internally or via wrapper
                        v, a, t, c, mask_ratio = batch
                        loss_dict = self._step_diffusion(lobe, config, v, a, t, c, mask_ratio)
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    if loss_dict is None: continue  # Skip step (e.g. NaN or empty)

                    step_count += 1
                    self._trigger("step", step_count, loss_dict)

                    # Autosave Trigger
                    if config.autosave_interval > 0 and step_count % config.autosave_interval == 0:
                        self._trigger("autosave", step_count)

                    if config.max_steps and step_count >= config.max_steps:
                        self.stop_requested = True
                        break

            self._trigger("finished")

        except Exception as e:
            self._trigger("error", e)
            raise e

    def _step_ar(self, lobe, config, v, a, t, c, targets):
        """Autoregressive Step"""
        optimizer = lobe.optimizer
        scaler = lobe.scaler
        use_amp = (scaler is not None) and (not config.use_fp32)

        # Move to device if not already (plugins usually handle this, but safety first)
        if t is not None: t = t.to(self.device)
        if targets is not None: targets = targets.to(self.device)

        # Zero Gradients
        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            # Forward
            try:
                logits, _, _ = lobe.model(v, a, t, c)
            except RuntimeError as e:
                # Fallback for models without Control channel support
                if "positional" in str(e):
                    logits, _, _ = lobe.model(v, a, t)
                else:
                    raise e

            # Compute Prediction Loss
            # Align logits and targets. Usually t is input, targets is t shifted by 1.
            # Here we assume the plugin provided aligned `t` (input) and `targets` (label).

            # Identify Text Portion (Shift logic varies by architecture, usually managed by plugin pre-calc)
            # Standard: Logits [B, T, V], Targets [B, T]

            # HACK: Handle multimodal concatenation offset if needed
            # For now, we assume simple alignment provided by plugin

            vocab_size = logits.size(-1)
            loss_pred = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=-100  # Standard ignore
            )

            # Game Loss (Equilibrium)
            loss_game = torch.tensor(0.0, device=self.device)
            if hasattr(lobe.model, "modules"):
                for module in lobe.model.modules():
                    if hasattr(module, 'game_loss'):
                        loss_game += module.game_loss()

            # --- NURSERY CLAMPING ---
            raw_pred = loss_pred.item()
            raw_game = loss_game.item()

            # NaN Check
            if math.isnan(raw_pred) or math.isinf(raw_pred):
                return None

            # Apply Clamps
            final_loss = self._apply_clamps(loss_pred, config.loss_clamp_prediction) + \
                         self._apply_clamps(loss_game, config.loss_clamp_game)

        # Backward
        if scaler:
            scaler.scale(final_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lobe.model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(lobe.model.parameters(), config.clip_grad_norm)
            optimizer.step()

        return {
            "total": final_loss.item(),
            "pred": raw_pred,
            "game": raw_game
        }

    def _step_diffusion(self, lobe, config, v, a, t, c, mask_ratio):
        """Discrete Diffusion Step"""
        optimizer = lobe.optimizer
        scaler = lobe.scaler
        use_amp = (scaler is not None) and (not config.use_fp32)

        # Generate Timestep
        B = t.shape[0]
        timestep = torch.randint(0, 1000, (B,), device=self.device)

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            # Model Forward (returns logits, mask_boolean, aux)
            # Note: Masking happens INSIDE model forward based on mask_ratio
            logits, mask_bool, _ = lobe.model(v, a, t, c, timestep=timestep, mask_ratio=mask_ratio)

            # Flatten
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = t.reshape(-1)
            flat_mask = mask_bool.reshape(-1)

            # Compute Cross Entropy only on MASKED tokens
            pred = flat_logits[flat_mask]
            target = flat_targets[flat_mask]

            if pred.numel() == 0: return None  # Nothing masked?

            loss_recon = F.cross_entropy(pred, target)

            # Game Loss
            loss_game = torch.tensor(0.0, device=self.device)
            for module in lobe.model.modules():
                if hasattr(module, 'game_loss'):
                    loss_game += module.game_loss()

            # --- NURSERY ---
            raw_recon = loss_recon.item()
            raw_game = loss_game.item()

            if math.isnan(raw_recon): return None

            final_loss = self._apply_clamps(loss_recon, config.loss_clamp_prediction) + \
                         self._apply_clamps(loss_game, config.loss_clamp_game)

        # Backward
        if scaler:
            scaler.scale(final_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lobe.model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(lobe.model.parameters(), config.clip_grad_norm)
            optimizer.step()

        return {
            "total": final_loss.item(),
            "recon": raw_recon,
            "game": raw_game
        }

    def _apply_clamps(self, loss_tensor, settings):
        """Applies Nursery scaling logic: (min, max)."""
        if not settings: return loss_tensor

        min_v, max_v = settings
        val = loss_tensor.item()

        # Dynamic Scaling Factor logic (soft clamp)
        if val < min_v:
            return loss_tensor * (min_v / (val + 1e-9))
        elif val > max_v:
            return loss_tensor * (max_v / (val + 1e-9))

        return loss_tensor