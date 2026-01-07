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

# FILE: Plugins/tab_graphs.py

import tkinter as tk
from tkinter import ttk
import math
import numpy as np


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Telemetry"

        # Settings
        self.smoothing = tk.IntVar(value=15)
        self.auto_refresh = tk.BooleanVar(value=True)
        self.micro_window = tk.IntVar(value=1000)

        # Base Margins (Will scale slightly)
        self.MARGIN_LEFT_BASE = 0.1  # 10% of width
        self.MARGIN_TOP_BASE = 0.1  # 10% of height

        self._setup_ui()
        if self.parent:
            self.parent.after(2000, self._animate)

    def _setup_ui(self):
        if self.parent is None: return

        # 1. TOOLBAR
        bar = ttk.Frame(self.parent, padding=5)
        bar.pack(fill="x")

        ttk.Label(bar, text="Micro Window:").pack(side="left")
        ttk.Spinbox(bar, from_=100, to=10000, textvariable=self.micro_window, width=6).pack(side="left", padx=5)

        ttk.Label(bar, text="Smooth:").pack(side="left", padx=(10, 0))
        ttk.Scale(bar, from_=1, to=100, variable=self.smoothing, orient="horizontal", length=100).pack(side="left",
                                                                                                       padx=5)

        ttk.Checkbutton(bar, text="Auto-Refresh", variable=self.auto_refresh).pack(side="left", padx=10)
        ttk.Button(bar, text="⟳ REFRESH", command=self._update_graphs).pack(side="right", padx=5)

        # 2. UNIFIED CANVAS
        self.canvas = tk.Canvas(self.parent, bg=self.app.colors["BG_MAIN"], highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Resize listener
        self.canvas.bind("<Configure>", lambda e: self._update_graphs())

    def _animate(self):
        if self.auto_refresh.get() and self.parent.winfo_ismapped():
            self._update_graphs()
        self.parent.after(2000, self._animate)

    def _moving_average(self, data, window):
        if not data or window < 2: return data
        w = min(int(window), len(data))
        if w < 2: return data
        return np.convolve(data, np.ones(w) / w, 'valid').tolist()

    def _update_graphs(self):
        c = self.canvas
        c.delete("all")

        w = c.winfo_width()
        h = c.winfo_height()
        if w < 50 or h < 50: return

        # --- DYNAMIC SCALING ---
        # Calculate font size based on height (e.g., 1/25th of screen height)
        # Clamped between 10 and 24
        raw_fs = int(h / 25)
        fs_axis = max(10, min(24, raw_fs))
        fs_title = max(12, min(32, int(raw_fs * 1.2)))

        # Margins adapt to font size
        m_left = fs_axis * 5  # Room for "0.0" text
        m_right = 30
        m_top = fs_title * 3
        m_bottom = fs_title * 3

        # --- DATA FETCH ---
        macro_data = self.app.graph_macro.get('recon', []) if hasattr(self.app, 'graph_macro') else []

        raw_micro = []
        if self.app.graph_data:
            for ep in sorted(self.app.graph_data.keys()):
                raw_recon = self.app.graph_data[ep].get('raw_text', [])
                if raw_recon: raw_micro.extend(raw_recon)

        limit = self.micro_window.get()
        if raw_micro: raw_micro = raw_micro[-limit:]
        micro_data = self._moving_average(raw_micro, self.smoothing.get())

        if not macro_data and not micro_data:
            c.create_text(w / 2, h / 2, text="WAITING FOR DATA...", fill=self.app.colors["FG_DIM"],
                          font=("Segoe UI", fs_title))
            return

        # --- Y-RANGE ---
        all_vals = macro_data + micro_data
        if not all_vals: return

        min_y = max(0, min(all_vals) * 0.9)
        max_y = max(all_vals) * 1.1
        if max_y - min_y < 0.01: max_y += 0.1

        # Drawing Area
        draw_w = w - m_left - m_right
        draw_h = h - m_top - m_bottom
        origin_x = m_left
        origin_y = h - m_bottom

        # --- DRAW AXES ---
        # Y Labels
        for i in range(6):
            ratio = i / 5
            y = origin_y - (ratio * draw_h)
            val = min_y + (ratio * (max_y - min_y))

            c.create_line(origin_x, y, w - m_right, y, fill=self.app.colors["BORDER"], dash=(2, 4))
            c.create_text(origin_x - 10, y, text=f"{val:.1f}", anchor="e",
                          fill=self.app.colors["FG_TEXT"], font=("Consolas", fs_axis, "bold"))

        # Vertical Line
        c.create_line(origin_x, m_top, origin_x, origin_y, fill=self.app.colors["FG_DIM"], width=2)

        # --- MACRO PLOT ---
        if len(macro_data) > 1:
            color_macro = self.app.colors["ACCENT"]
            points = []
            x_step = draw_w / (len(macro_data) - 1)
            for i, val in enumerate(macro_data):
                px = origin_x + i * x_step
                norm_y = (val - min_y) / (max_y - min_y)
                py = origin_y - (norm_y * draw_h)
                points.extend([px, py])

            if points:
                c.create_line(points, fill=color_macro, width=3, smooth=True)

            c.create_text(w / 2, h - (m_bottom / 2), text=f"MACRO HISTORY ({len(macro_data)}k Steps)",
                          fill=color_macro, font=("Segoe UI", fs_title, "bold"))

        # --- MICRO PLOT ---
        if len(micro_data) > 1:
            color_micro = self.app.colors["ERROR"]
            points = []
            x_step = draw_w / (len(micro_data) - 1)
            for i, val in enumerate(micro_data):
                px = origin_x + i * x_step
                norm_y = (val - min_y) / (max_y - min_y)
                py = origin_y - (norm_y * draw_h)
                points.extend([px, py])

            if points:
                c.create_line(points, fill=color_micro, width=2)

            c.create_text(w / 2, m_top / 2, text=f"MICRO VIEW (Last {len(micro_data)} Steps)",
                          fill=color_micro, font=("Segoe UI", fs_title, "bold"))

    def on_theme_change(self):
        if hasattr(self, 'canvas'):
            self.canvas.config(bg=self.app.colors["BG_MAIN"])