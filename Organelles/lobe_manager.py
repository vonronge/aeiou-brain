"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Lobe Manager:
Central authority for loading, saving, and managing neural lobes.
Abstracts away file I/O, dynamic imports, and hardware configuration.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import importlib.util
from dataclasses import dataclass
from typing import Optional, Any, Dict
from torch.cuda.amp import GradScaler
import gc
import threading
import time


class LobeNotFoundError(Exception): pass


class GeneticsNotFoundError(Exception): pass


class CorruptLobeError(Exception): pass


@dataclass
class LobeHandle:
    id: int
    genome: str
    model_type: str
    model: nn.Module
    optimizer: Optional[optim.Optimizer]
    scaler: Optional[GradScaler]
    device: str

    def train(self): self.model.train()

    def eval(self): self.model.eval()


class Organelle_LobeManager:
    def __init__(self, lobes_dir: str, genetics_dir: str, device: str, ribosome=None):
        self.lobes_dir = lobes_dir
        self.genetics_dir = genetics_dir
        self.device = device
        self.ribosome = ribosome
        self._active_lobes: Dict[int, LobeHandle] = {}
        self._genetics_registry: Dict[str, str] = {}
        self._save_lock = threading.Lock()  # Prevent concurrent saves of same file
        self.refresh_registry()

    def refresh_registry(self):
        self._genetics_registry = {}
        if not os.path.exists(self.genetics_dir): return
        files = [f for f in os.listdir(self.genetics_dir) if f.endswith(".py") and not f.startswith("__")]
        for f in files:
            try:
                path = os.path.join(self.genetics_dir, f)
                spec = importlib.util.spec_from_file_location("temp_dna_scan", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "INFO"):
                    name = module.INFO.get("name", f)
                    self._genetics_registry[name] = f
                else:
                    self._genetics_registry[f] = f
            except:
                pass

    def list_available_genetics(self):
        return sorted(list(self._genetics_registry.keys()))

    def get_lobe(self, lobe_id: int) -> Optional[LobeHandle]:
        return self._active_lobes.get(lobe_id)

    def unload_lobe(self, lobe_id: int):
        if lobe_id in self._active_lobes:
            try:
                self._active_lobes[lobe_id].model.cpu()
            except:
                pass
            del self._active_lobes[lobe_id]
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[LobeManager] Unloaded Lobe {lobe_id}")

    def load_lobe(self, lobe_id: int) -> LobeHandle:
        path = os.path.join(self.lobes_dir, f"brain_lobe_{lobe_id}.pt")
        if not os.path.exists(path):
            raise LobeNotFoundError(f"Lobe file not found: {path}")

        print(f"[LobeManager] Loading Lobe {lobe_id} from disk...")

        try:
            # Load to CPU first to prevent OOM
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as e:
            raise CorruptLobeError(f"Failed to load checkpoint: {e}")

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint)
            genome_name = checkpoint.get("genome", "gpt2")
            model_type = checkpoint.get("model_type")
        else:
            state_dict = checkpoint
            genome_name = "gpt2"
            model_type = "ar"

        if model_type is None:
            model_type = "diffusion" if "diffusion" in genome_name.lower() else "ar"

        module = self._import_genetics(genome_name)

        try:
            config = module.NucleusConfig()
            model = module.Model(config)
            model.load_state_dict(state_dict, strict=False)

            # VRAM Safety Check
            param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
            model_gb = param_size / (1024 ** 3)

            print(f"[LobeManager] Model Size: {model_gb:.2f} GB")

            # Safe limit for 3080 Ti (12GB) is roughly 10GB loaded
            if model_gb < 10.0 and self.device == "cuda":
                print("[LobeManager] Moving to GPU...")
                model = model.to(self.device)
            else:
                print(f"[LobeManager] ⚠️ Model large ({model_gb:.2f} GB). Keeping on CPU.")

        except Exception as e:
            raise CorruptLobeError(f"Architecture mismatch for {genome_name}: {e}")

        optimizer = None
        if "Muon" in genome_name or getattr(config, 'use_muon', False):
            from Genetics.muon import Muon
            optimizer = Muon(model.parameters(), lr=0.0005, momentum=0.95)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        scaler = GradScaler() if self.device == "cuda" else None

        if self.ribosome and hasattr(model, "tokenizer"):
            self.ribosome.set_tokenizer(model.tokenizer)

        handle = LobeHandle(
            id=lobe_id,
            genome=genome_name,
            model_type=model_type,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=self.device
        )

        self._active_lobes[lobe_id] = handle
        print(f"[LobeManager] Lobe {lobe_id} ({genome_name}) is online.")
        return handle

    def create_lobe(self, lobe_id: int, genome_name: str) -> LobeHandle:
        print(f"[LobeManager] Genesis: Creating Lobe {lobe_id} with {genome_name}...")
        module = self._import_genetics(genome_name)
        config = module.NucleusConfig()

        model = module.Model(config)

        # Move to GPU if fits
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        if (param_size / 1e9) < 10 and self.device == "cuda":
            model = model.to(self.device)

        model_type = "diffusion" if "diffusion" in genome_name.lower() else "ar"

        optimizer = None
        if "Muon" in genome_name:
            from Genetics.muon import Muon
            optimizer = Muon(model.parameters(), lr=0.0005, momentum=0.95)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        scaler = GradScaler() if self.device == "cuda" else None

        handle = LobeHandle(
            id=lobe_id,
            genome=genome_name,
            model_type=model_type,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=self.device
        )

        self._active_lobes[lobe_id] = handle
        # Blocking save for initial create is fine
        self._sync_save(lobe_id)
        return handle

    # --- ASYNC SAVE LOGIC ---
    def save_lobe(self, lobe_id: int, custom_path: str = None) -> None:
        """
        Non-blocking save.
        1. Clones weights to CPU (Fast).
        2. Spawns thread to write disk (Slow).
        """
        handle = self._active_lobes.get(lobe_id)
        if not handle: return

        save_path = custom_path if custom_path else os.path.join(self.lobes_dir, f"brain_lobe_{lobe_id}.pt")

        # 1. Fast Snapshot to CPU RAM
        # We iterate and .cpu().clone() to ensure thread safety against ongoing training updates
        try:
            cpu_state = {k: v.cpu().clone() for k, v in handle.model.state_dict().items()}
        except Exception as e:
            print(f"[LobeManager] Snapshot Failed: {e}")
            return

        payload = {
            "genome": handle.genome,
            "model_type": handle.model_type,
            "state_dict": cpu_state
        }

        # 2. Background Writer
        def _write_worker():
            with self._save_lock:
                try:
                    temp_path = save_path + ".tmp"
                    torch.save(payload, temp_path)
                    if os.path.exists(save_path): os.remove(save_path)
                    os.rename(temp_path, save_path)
                    # print(f"[LobeManager] Background Save Complete: Lobe {lobe_id}") # Optional: Reduce spam
                except Exception as e:
                    print(f"[LobeManager] Background Save Failed: {e}")

        # 3. Launch
        t = threading.Thread(target=_write_worker, daemon=True)
        t.start()
        print(f"[LobeManager] Snapshot taken. Saving Lobe {lobe_id} in background...")

    def _sync_save(self, lobe_id: int):
        """Blocking save for initial creation or exit."""
        handle = self._active_lobes.get(lobe_id)
        if not handle: return
        path = os.path.join(self.lobes_dir, f"brain_lobe_{lobe_id}.pt")
        payload = {
            "genome": handle.genome,
            "model_type": handle.model_type,
            "state_dict": handle.model.state_dict()
        }
        torch.save(payload, path)
        print(f"[LobeManager] Saved Lobe {lobe_id}.")

    def _import_genetics(self, genome_name: str):
        filename = self._genetics_registry.get(genome_name, f"{genome_name}.py")
        path = os.path.join(self.genetics_dir, filename)
        if not os.path.exists(path):
            found = [f for f in os.listdir(self.genetics_dir) if f.lower() == filename.lower()]
            if found:
                path = os.path.join(self.genetics_dir, found[0])
            else:
                raise GeneticsNotFoundError(f"Genetics '{genome_name}' not found.")
        try:
            spec = importlib.util.spec_from_file_location(genome_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            raise GeneticsNotFoundError(f"Failed to import '{genome_name}': {e}")