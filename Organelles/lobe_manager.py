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


# Define custom exceptions for cleaner error handling
class LobeNotFoundError(Exception): pass


class GeneticsNotFoundError(Exception): pass


class CorruptLobeError(Exception): pass


@dataclass
class LobeHandle:
    """
    A lightweight handle representing a loaded active lobe.
    Passed around to plugins so they don't need to know about files or imports.
    """
    id: int
    genome: str
    model_type: str  # "ar", "diffusion", etc.
    model: nn.Module
    optimizer: Optional[optim.Optimizer]
    scaler: Optional[GradScaler]
    device: str

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class Organelle_LobeManager:
    def __init__(self, lobes_dir: str, genetics_dir: str, device: str, ribosome=None):
        self.lobes_dir = lobes_dir
        self.genetics_dir = genetics_dir
        self.device = device
        self.ribosome = ribosome

        # Internal cache of loaded lobes: {int_id: LobeHandle}
        self._active_lobes: Dict[int, LobeHandle] = {}

    def get_lobe(self, lobe_id: int) -> Optional[LobeHandle]:
        """Returns the handle if loaded, else None."""
        return self._active_lobes.get(lobe_id)

    def list_active_lobes(self) -> Dict[int, LobeHandle]:
        """Returns a dict of all currently loaded lobes."""
        return self._active_lobes

    def unload_lobe(self, lobe_id: int):
        """Safely unloads a lobe and frees memory."""
        if lobe_id in self._active_lobes:
            del self._active_lobes[lobe_id]
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[LobeManager] Unloaded Lobe {lobe_id}")

    def load_lobe(self, lobe_id: int) -> LobeHandle:
        """
        The heavy lifter. Loads weights, resolves genetics, sets up optimizers.
        Returns a ready-to-use LobeHandle.
        """
        # 1. Path Resolution
        filename = f"brain_lobe_{lobe_id}.pt"
        path = os.path.join(self.lobes_dir, filename)

        if not os.path.exists(path):
            raise LobeNotFoundError(f"Lobe file not found: {path}")

        print(f"[LobeManager] Loading Lobe {lobe_id} from {filename}...")

        # 2. Load Weights & Metadata
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as e:
            raise CorruptLobeError(f"Failed to load checkpoint: {e}")

        # Handle legacy or dict-based checkpoints
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint)
            genome_name = checkpoint.get("genome", "gpt2")  # Default fallback
            model_type = checkpoint.get("model_type")
        else:
            # Very old format (raw state dict)
            state_dict = checkpoint
            genome_name = "gpt2"
            model_type = "ar"

        # Auto-detect model type if missing
        if model_type is None:
            model_type = "diffusion" if "diffusion" in genome_name.lower() else "ar"

        # 3. Resolve & Import Genetics
        module = self._import_genetics(genome_name)

        # 4. Instantiate Model
        try:
            config = module.NucleusConfig()
            model = module.Model(config).to(self.device)

            # Strict=False allows for architecture evolution (e.g. adding new layers)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise CorruptLobeError(f"Architecture mismatch for {genome_name}: {e}")

        # 5. Setup Optimizer (Muon vs AdamW)
        # We check the genome name or config for "Muon" preference
        optimizer = None
        if "Muon" in genome_name or getattr(config, 'use_muon', False):
            try:
                from Genetics.muon import Muon
                # Muon typically needs specific params, here we use safe defaults
                optimizer = Muon(model.parameters(), lr=0.0005, momentum=0.95)
                print(f"[LobeManager] Attached Muon Optimizer to Lobe {lobe_id}")
            except ImportError:
                print("[LobeManager] Muon optimizer missing, falling back to AdamW.")
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        # 6. Setup Scaler (AMP)
        scaler = None
        if self.device == "cuda":
            scaler = GradScaler()

        # 7. Register Tokenizer (if Ribosome exists)
        if self.ribosome and hasattr(model, "tokenizer"):
            self.ribosome.set_tokenizer(model.tokenizer)

        # 8. Create Handle & Cache
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
        """
        Initializes a FRESH lobe from genetics (random weights).
        """
        print(f"[LobeManager] Genesis: Creating Lobe {lobe_id} with {genome_name}...")

        module = self._import_genetics(genome_name)
        config = module.NucleusConfig()
        model = module.Model(config).to(self.device)

        # Determine Type
        model_type = "diffusion" if "diffusion" in genome_name.lower() else "ar"

        # Setup Optimizer
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
        self.save_lobe(lobe_id)  # Initial save
        return handle

    def save_lobe(self, lobe_id: int, custom_path: str = None) -> str:
        """
        Saves the lobe state to disk.
        """
        handle = self._active_lobes.get(lobe_id)
        if not handle:
            raise LobeNotFoundError(f"Cannot save Lobe {lobe_id}: Not loaded.")

        if custom_path:
            save_path = custom_path
        else:
            save_path = os.path.join(self.lobes_dir, f"brain_lobe_{lobe_id}.pt")

        payload = {
            "genome": handle.genome,
            "model_type": handle.model_type,
            "state_dict": handle.model.state_dict()
        }

        # Atomic save (write temp then rename) to prevent corruption
        temp_path = save_path + ".tmp"
        torch.save(payload, temp_path)

        if os.path.exists(save_path):
            os.remove(save_path)
        os.rename(temp_path, save_path)

        print(f"[LobeManager] Saved Lobe {lobe_id} to {os.path.basename(save_path)}")
        return save_path

    def _import_genetics(self, genome_name: str):
        """
        Dynamically imports the genetics module.
        Handles strict matching and fuzzy fallback.
        """
        # 1. Direct path check
        target_file = f"{genome_name}.py"
        path = os.path.join(self.genetics_dir, target_file)

        if not os.path.exists(path):
            # 2. Fuzzy Search (Case insensitive)
            found = [f for f in os.listdir(self.genetics_dir)
                     if f.lower() == target_file.lower()]
            if found:
                path = os.path.join(self.genetics_dir, found[0])
            else:
                raise GeneticsNotFoundError(f"Genetics module '{genome_name}' not found in {self.genetics_dir}")

        # 3. Import
        try:
            spec = importlib.util.spec_from_file_location(genome_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            raise GeneticsNotFoundError(f"Failed to import genetics '{genome_name}': {e}")