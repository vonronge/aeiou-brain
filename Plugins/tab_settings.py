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
from tkinter import ttk, messagebox, filedialog, colorchooser
import os
import sys


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "System Config"

        # --- LOAD STATE FROM PHAGUS ---
        state = self.app.phagus.state

        # 1. Scaling
        self.scale_var = tk.DoubleVar(value=state.ui_scale)

        # 2. Paths (Load current values)
        # We display the raw values from config (which might be relative)
        # The user can browse to set absolute paths
        self.path_vars = {
            "data_dir": tk.StringVar(value=state.data_dir),
            "chaos_dir": tk.StringVar(value=state.chaos_dir),
            "output_dir": tk.StringVar(value=state.output_dir)
        }

        # 3. Colors (Theme)
        self.col_vars = {
            "ACCENT": tk.StringVar(value=state.colors.get("ACCENT", "#A8C7FA")),
            "BG_MAIN": tk.StringVar(value=state.colors.get("BG_MAIN", "#0b0f19")),
            "BG_CARD": tk.StringVar(value=state.colors.get("BG_CARD", "#131620")),
            "FG_TEXT": tk.StringVar(value=state.colors.get("FG_TEXT", "#E3E3E3")),
            "SUCCESS": tk.StringVar(value=state.colors.get("SUCCESS", "#81C995")),
            "ERROR": tk.StringVar(value=state.colors.get("ERROR", "#F28B82"))
        }

        self._setup_ui()

    def _setup_ui(self):
        if self.parent is None: return

        # Scrollable Container
        canvas = tk.Canvas(self.parent, borderwidth=0, highlightthickness=0, bg=self.app.colors["BG_MAIN"])
        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel
        def _on_mousewheel(event):
            if os.name == 'nt' or sys.platform == 'darwin':
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        # Bind only when hovering this frame
        self.parent.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.parent.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # --- SECTION 1: DISPLAY ---
        fr_disp = ttk.LabelFrame(scroll_frame, text="Display & Scaling", padding=15)
        fr_disp.pack(fill="x", pady=5)

        ttk.Label(fr_disp, text="UI Zoom Level (Requires Restart):").pack(anchor="w")

        scl_row = ttk.Frame(fr_disp)
        scl_row.pack(fill="x", pady=5)

        scl = ttk.Scale(scl_row, from_=0.8, to=3.0, variable=self.scale_var, orient="horizontal")
        scl.pack(side="left", fill="x", expand=True)

        lbl_scl = ttk.Label(scl_row, text=f"{self.scale_var.get():.1f}x", width=5)
        lbl_scl.pack(side="right", padx=10)
        scl.configure(command=lambda v: lbl_scl.configure(text=f"{float(v):.1f}x"))

        # --- SECTION 2: STORAGE PATHS ---
        fr_paths = ttk.LabelFrame(scroll_frame, text="Storage Configuration", padding=15)
        fr_paths.pack(fill="x", pady=10)

        def add_path_row(label, key):
            f = ttk.Frame(fr_paths)
            f.pack(fill="x", pady=5)
            ttk.Label(f, text=label, width=15, anchor="w").pack(side="left")
            ttk.Entry(f, textvariable=self.path_vars[key]).pack(side="left", fill="x", expand=True, padx=5)
            ttk.Button(f, text="📂", width=4,
                       command=lambda: self._browse_path(self.path_vars[key])).pack(side="right")

        add_path_row("Training Data:", "data_dir")
        add_path_row("Chaos Buffer:", "chaos_dir")
        add_path_row("Comics Output:", "output_dir")

        ttk.Label(fr_paths, text="* Paths are relative to application root unless absolute.",
                  font=("Segoe UI", 9, "italic"), foreground=self.app.colors["FG_DIM"]).pack(anchor="w", pady=(5, 0))

        # --- SECTION 3: THEME ---
        fr_theme = ttk.LabelFrame(scroll_frame, text="Neural Theme", padding=15)
        fr_theme.pack(fill="x", pady=10)

        def add_col_row(label, key):
            f = ttk.Frame(fr_theme)
            f.pack(fill="x", pady=5)
            ttk.Label(f, text=label, width=18, anchor="w").pack(side="left")

            # Preview box
            lbl_prev = tk.Label(f, bg=self.col_vars[key].get(), width=4, relief="solid", borderwidth=1)
            lbl_prev.pack(side="left", padx=5)

            entry = ttk.Entry(f, textvariable=self.col_vars[key])
            entry.pack(side="left", fill="x", expand=True, padx=5)

            def pick():
                c = colorchooser.askcolor(color=self.col_vars[key].get())[1]
                if c:
                    self.col_vars[key].set(c)
                    lbl_prev.config(bg=c)

            ttk.Button(f, text="Pick", width=6, command=pick).pack(side="right")

        add_col_row("Accent Color:", "ACCENT")
        add_col_row("Main Background:", "BG_MAIN")
        add_col_row("Card Background:", "BG_CARD")
        add_col_row("Text Color:", "FG_TEXT")
        add_col_row("Success Color:", "SUCCESS")
        add_col_row("Error Color:", "ERROR")

        # --- SAVE ---
        btn_save = ttk.Button(scroll_frame, text="SAVE ALL SETTINGS", command=self._save_settings)
        btn_save.pack(fill="x", pady=30)

    def _browse_path(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d: var.set(d)

    def _save_settings(self):
        try:
            # Update Phagus State
            state = self.app.phagus.state

            # 1. Scale
            state.ui_scale = round(self.scale_var.get(), 2)

            # 2. Paths
            state.data_dir = self.path_vars["data_dir"].get()
            state.chaos_dir = self.path_vars["chaos_dir"].get()
            state.output_dir = self.path_vars["output_dir"].get()

            # 3. Colors
            for k, v in self.col_vars.items():
                state.colors[k] = v.get()

            # Persist via Phagus
            self.app.phagus.save()

            # Log to Golgi
            self.app.golgi.save("Configuration Updated.", source="Settings")

            if messagebox.askyesno("Saved",
                                   "Settings saved successfully.\n\nRestart Application now to apply changes?"):
                self.app.destroy()
                # Re-launch logic could go here, but usually OS handles it
                sys.exit(0)

        except Exception as e:
            messagebox.showerror("Error", f"Could not save settings: {e}")
            self.app.golgi.error(f"Config Save Failed: {e}", source="Settings")

    def on_theme_change(self):
        # Settings tab doesn't dynamically reload its own theme to avoid glitches during editing
        pass