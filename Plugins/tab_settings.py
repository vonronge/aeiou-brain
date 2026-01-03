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

import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
import json
import os


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "System Config"

        self.config_path = os.path.join(self.app.paths["root"], "settings.json")
        self.local_colors = self.app.colors.copy()

        # --- DEFINING THE 4 SCHEMES ---
        self.SCHEMES = {
            "NOSTROMO": {  # Default Sci-Fi
                "BG_MAIN": "#0b0f19", "BG_CARD": "#131620", "FG_TEXT": "#E3E3E3",
                "FG_DIM": "#8e9198", "ACCENT": "#A8C7FA", "BTN": "#1E222D",
                "BTN_ACT": "#2B3042", "SUCCESS": "#81C995", "ERROR": "#F28B82",
                "WARN": "#FDD663", "BORDER": "#444444", "GRID": "#333333", "SCROLL": "#2B3042"
            },
            "APERTURE": {  # Light/Lab
                "BG_MAIN": "#F0F2F5", "BG_CARD": "#FFFFFF", "FG_TEXT": "#1A1A1A",
                "FG_DIM": "#606060", "ACCENT": "#007ACC", "BTN": "#E1E4E8",
                "BTN_ACT": "#D1D5DA", "SUCCESS": "#2EA043", "ERROR": "#DA3633",
                "WARN": "#D29922", "BORDER": "#E1E4E8", "GRID": "#E1E1E1", "SCROLL": "#C0C4C8"
            },
            "MATRIX": {  # Hacker
                "BG_MAIN": "#000000", "BG_CARD": "#0D110D", "FG_TEXT": "#00FF41",
                "FG_DIM": "#008F11", "ACCENT": "#00FF41", "BTN": "#003B00",
                "BTN_ACT": "#005500", "SUCCESS": "#00FF41", "ERROR": "#FF0000",
                "WARN": "#FFFF00", "BORDER": "#003B00", "GRID": "#002200", "SCROLL": "#003B00"
            },
            "NEON": {  # Cyberpunk
                "BG_MAIN": "#120024", "BG_CARD": "#1F003D", "FG_TEXT": "#E0E0E0",
                "FG_DIM": "#B07CC6", "ACCENT": "#FF00FF", "BTN": "#2D0052",
                "BTN_ACT": "#45007A", "SUCCESS": "#00FFC8", "ERROR": "#FF0055",
                "WARN": "#FFE600", "BORDER": "#45007A", "GRID": "#2D0052", "SCROLL": "#45007A"
            }
        }

        self.color_vars = {}
        for key, val in self.local_colors.items():
            self.color_vars[key] = tk.StringVar(value=val)

        self._setup_ui()

    def _setup_ui(self):
        # 1. THEME GALLERY (2x2 Grid)
        gallery_frame = ttk.LabelFrame(self.parent, text="Visual Core Presets", padding=10)
        gallery_frame.pack(fill="x", padx=20, pady=10)

        col = 0
        row = 0

        for name, scheme in self.SCHEMES.items():
            # Card Frame
            card = tk.Frame(gallery_frame, bg=scheme["BG_CARD"], bd=1, relief="solid")
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            gallery_frame.columnconfigure(col, weight=1)

            # Preview Content
            header = tk.Label(card, text=name, bg=scheme["BG_CARD"], fg=scheme["ACCENT"], font=("Segoe UI", 12, "bold"))
            header.pack(pady=(10, 5))

            sub = tk.Label(card, text="System Interface", bg=scheme["BG_CARD"], fg=scheme["FG_DIM"],
                           font=("Segoe UI", 9))
            sub.pack(pady=(0, 10))

            # Activate Button
            btn = tk.Button(card, text="ACTIVATE", bg=scheme["BTN"], fg=scheme["FG_TEXT"], relief="flat",
                            activebackground=scheme["BTN_ACT"], activeforeground=scheme["FG_TEXT"],
                            command=lambda s=scheme: self._apply_live(s))
            btn.pack(fill="x", padx=20, pady=10)

            col += 1
            if col > 1:
                col = 0
                row += 1

        # 2. FINE TUNING
        tune_frame = ttk.LabelFrame(self.parent, text="Manual Override", padding=15)
        tune_frame.pack(fill="both", expand=True, padx=20, pady=5)

        r = 0
        c = 0
        keys = sorted(self.local_colors.keys())
        for key in keys:
            if key not in self.color_vars: continue

            ttk.Label(tune_frame, text=key).grid(row=r, column=c, sticky="w", padx=5)

            # Swatch
            btn = tk.Button(tune_frame, bg=self.color_vars[key].get(), width=4,
                            command=lambda k=key: self._pick_color(k))
            btn.grid(row=r, column=c + 1, padx=5, pady=2)

            setattr(self, f"btn_{key}", btn)

            r += 1
            if r > 6:
                r = 0
                c += 2

        # 3. SAVE
        ttk.Button(self.parent, text="SAVE CONFIGURATION", command=self._save_config).pack(fill="x", padx=20, pady=10)

    def _apply_live(self, theme):
        # 1. Update App State
        self.app.colors.update(theme)

        # 2. Update UI Vars
        for k, v in theme.items():
            if k in self.color_vars:
                self.color_vars[k].set(v)
                if hasattr(self, f"btn_{k}"):
                    getattr(self, f"btn_{k}").config(bg=v)

        # 3. Trigger Global Refresh
        self.app.apply_theme()
        self._save_config(silent=True)

    def _pick_color(self, key):
        curr = self.color_vars[key].get()
        color = colorchooser.askcolor(color=curr, title=f"Override {key}")
        if color[1]:
            hex_val = color[1]
            self.color_vars[key].set(hex_val)
            getattr(self, f"btn_{key}").config(bg=hex_val)
            # Apply immediately to see effect
            self.app.colors[key] = hex_val
            self.app.apply_theme()

    def _save_config(self, silent=False):
        # Sync vars to app state
        new_colors = {k: v.get() for k, v in self.color_vars.items()}
        self.app.colors.update(new_colors)

        data = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    pass

        data["colors"] = self.app.colors

        try:
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            if not silent:
                messagebox.showinfo("System Update", "Theme saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save settings: {e}")

    def on_theme_change(self):
        # Called when theme changes externally (to update swatches if needed)
        pass