"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Phagus:
Maintains homeostasis.
Consumes configuration files, digests paths, and regulates the system environment.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any

# Default Theme (Dark Mode)
DEFAULT_COLORS = {
    "BG_MAIN": "#0b0f19",
    "BG_CARD": "#131620",
    "FG_TEXT": "#E3E3E3",
    "FG_DIM": "#8e9198",
    "ACCENT": "#A8C7FA",
    "BTN": "#1E222D",
    "BTN_ACT": "#2B3042",
    "SUCCESS": "#81C995",
    "ERROR": "#F28B82",
    "WARN": "#FDD663",
    "BORDER": "#444444",
    "GRID": "#333333",
    "SCROLL": "#2B3042"
}


@dataclass
class HomeostasisState:
    """The complete state of the organism's environment."""
    system_name: str = "AEIOU_v23.2"
    ui_scale: float = 1.3
    colors: Dict[str, str] = field(default_factory=lambda: DEFAULT_COLORS.copy())

    # Paths (Can be absolute or relative)
    data_dir: str = "Training_Data"
    chaos_dir: str = "Training_Data/Chaos_Buffer"
    output_dir: str = "Training_Data/Comics_Output"

    # Window State
    last_active_lobe: int = 1
    window_geometry: str = "1600x900+50+50"
    window_state: str = "normal"
    sidebar_order: list = field(default_factory=list)


class Organelle_Phagus:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.config_path = os.path.join(self.root_dir, "settings.json")

        # Core Infrastructure Paths (Immutable / System Critical)
        self.core_paths = {
            "root": self.root_dir,
            "lobes": os.path.join(self.root_dir, "lobes"),
            "genetics": os.path.join(self.root_dir, "Genetics"),
            "plugins": os.path.join(self.root_dir, "Plugins"),
            "memories": os.path.join(self.root_dir, "memories"),
            "organelles": os.path.join(self.root_dir, "Organelles")
        }

        # Ensure core paths exist immediately
        for p in self.core_paths.values():
            if not os.path.exists(p):
                os.makedirs(p, exist_ok=True)

        # Load State
        self.state = self.load()

    def load(self) -> HomeostasisState:
        """Loads settings.json, merging with defaults to prevent crashes on missing keys."""
        if not os.path.exists(self.config_path):
            return HomeostasisState()

        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)

            # Start with defaults
            default = HomeostasisState()

            # Merge Colors (don't overwrite whole dict if keys missing)
            merged_colors = default.colors.copy()
            if "colors" in data:
                merged_colors.update(data["colors"])

            # Construct State
            state = HomeostasisState(
                system_name=data.get("system_name", default.system_name),
                ui_scale=data.get("ui_scale", default.ui_scale),
                colors=merged_colors,
                data_dir=data.get("data_dir", default.data_dir),
                chaos_dir=data.get("chaos_dir", default.chaos_dir),
                output_dir=data.get("output_dir", default.output_dir),
                last_active_lobe=data.get("last_active_lobe", default.last_active_lobe),
                window_geometry=data.get("window_geometry", default.window_geometry),
                window_state=data.get("window_state", default.window_state),
                sidebar_order=data.get("sidebar_order", default.sidebar_order)
            )
            return state

        except Exception as e:
            print(f"[Phagus] Config Load Error: {e}. Using defaults.")
            return HomeostasisState()

    def save(self):
        """Persists the current HomeostasisState to disk."""
        try:
            data = asdict(self.state)
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            print("[Phagus] Homeostasis Saved.")
        except Exception as e:
            print(f"[Phagus] Save Error: {e}")

    def get_paths(self) -> Dict[str, str]:
        """
        Returns the MASTER PATH DICTIONARY.
        Combines immutable core paths with user-configured data paths.
        Automatically resolves relative paths against the root directory.
        """
        paths = self.core_paths.copy()

        # Helper to resolve user paths
        def resolve(p):
            if os.path.isabs(p): return p
            return os.path.join(self.root_dir, p)

        paths["data"] = resolve(self.state.data_dir)
        paths["chaos"] = resolve(self.state.chaos_dir)
        paths["output"] = resolve(self.state.output_dir)

        # Ensure user paths exist (Lazy Creation)
        for k in ["data", "chaos", "output"]:
            if not os.path.exists(paths[k]):
                try:
                    os.makedirs(paths[k], exist_ok=True)
                except:
                    pass

        return paths

    def get_theme(self) -> Dict[str, str]:
        """Returns the active color palette."""
        return self.state.colors