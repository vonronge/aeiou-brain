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

        # [NEW] Registry: Maps "Display Name" -> "filename.py"
        self._genetics_registry: Dict[str, str] = {}
        self.refresh_registry()

    def refresh_registry(self):
        """Scans Genetics folder and builds the name map."""
        self._genetics_registry = {}
        if not os.path.exists(self.genetics_dir): return

        files = [f for f in os.listdir(self.genetics_dir) if f.endswith(".py") and not f.startswith("__")]

        for f in files:
            try:
                # We import spec only to read INFO, not initialize the model
                path = os.path.join(self.genetics_dir, f)
                spec = importlib.util.spec_from_file_location("temp_dna_scan", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "INFO"):
                    # Map "MaskedDiffusion-mHC" -> "diffusion_mhc.py"
                    name = module.INFO.get("name", f)
                    self._genetics_registry[name] = f
                else:
                    # Fallback to filename
                    self._genetics_registry[f] = f

            except Exception as e:
                print(f"[LobeManager] Skipping {f}: {e}")

    def list_available_genetics(self):
        """Returns list of display names for the GUI."""
        return sorted(list(self._genetics_registry.keys()))

    def get_lobe(self, lobe_id: int) -> Optional[LobeHandle]:
        return self._active_lobes.get(lobe_id)

    def unload_lobe(self, lobe_id: int):
        if lobe_id in self._active_lobes:
            del self._active_lobes[lobe_id]
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[LobeManager] Unloaded Lobe {lobe_id}")

    def load_lobe(self, lobe_id: int) -> LobeHandle:
        path = os.path.join(self.lobes_dir, f"brain_lobe_{lobe_id}.pt")
        if not os.path.exists(path):
            raise LobeNotFoundError(f"Lobe file not found: {path}")

        print(f"[LobeManager] Loading Lobe {lobe_id}...")
        try:
            checkpoint = torch.load(path, map_location=self.device)
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

        # Import using the Registry
        module = self._import_genetics(genome_name)

        try:
            config = module.NucleusConfig()
            model = module.Model(config).to(self.device)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise CorruptLobeError(f"Architecture mismatch for {genome_name}: {e}")

        # Optimizer Setup
        optimizer = None
        if "Muon" in genome_name or getattr(config, 'use_muon', False):
            try:
                from Genetics.muon import Muon
                optimizer = Muon(model.parameters(), lr=0.0005, momentum=0.95)
            except ImportError:
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
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

        # Use Registry to resolve name -> file
        module = self._import_genetics(genome_name)

        config = module.NucleusConfig()
        model = module.Model(config).to(self.device)
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
        self.save_lobe(lobe_id)
        return handle

    def save_lobe(self, lobe_id: int, custom_path: str = None) -> str:
        handle = self._active_lobes.get(lobe_id)
        if not handle: raise LobeNotFoundError(f"Lobe {lobe_id} not loaded.")

        save_path = custom_path if custom_path else os.path.join(self.lobes_dir, f"brain_lobe_{lobe_id}.pt")

        payload = {
            "genome": handle.genome,  # Saves "MaskedDiffusion-mHC"
            "model_type": handle.model_type,
            "state_dict": handle.model.state_dict()
        }

        temp_path = save_path + ".tmp"
        torch.save(payload, temp_path)
        if os.path.exists(save_path): os.remove(save_path)
        os.rename(temp_path, save_path)

        print(f"[LobeManager] Saved Lobe {lobe_id} ({handle.genome}).")
        return save_path

    def _import_genetics(self, genome_name: str):
        """
        Resolves "MaskedDiffusion-mHC" -> "diffusion_mhc.py" using the registry.
        """
        # 1. Check Registry
        filename = self._genetics_registry.get(genome_name)

        # 2. If not in registry, try literal (fallback for old saves or direct filenames)
        if not filename:
            filename = f"{genome_name}.py"

        path = os.path.join(self.genetics_dir, filename)

        # 3. Final File Existence Check
        if not os.path.exists(path):
            # Try fuzzy match against directory as last resort
            found = [f for f in os.listdir(self.genetics_dir) if f.lower() == filename.lower()]
            if found:
                path = os.path.join(self.genetics_dir, found[0])
            else:
                raise GeneticsNotFoundError(f"Genetics '{genome_name}' not found (tried {filename}).")

        try:
            spec = importlib.util.spec_from_file_location(genome_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            raise GeneticsNotFoundError(f"Failed to import '{genome_name}' from {filename}: {e}")