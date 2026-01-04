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
import json
import os


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "System Settings"

        # --- VARIABLES ---
        # Scaling
        self.scale_var = tk.DoubleVar(value=getattr(self.app, 'ui_scale', 1.3))

        # Paths (Load current absolute paths or defaults)
        self.path_vars = {
            "data_dir": tk.StringVar(value=self.app.paths.get("data", "")),
            "chaos_dir": tk.StringVar(value=self.app.paths.get("chaos", "")),
            "output_dir": tk.StringVar(value=self.app.paths.get("output", ""))
        }

        # Colors (Theme)
        self.col_vars = {
            "ACCENT": tk.StringVar(value=self.app.colors.get("ACCENT", "#A8C7FA")),
            "BG_MAIN": tk.StringVar(value=self.app.colors.get("BG_MAIN", "#0b0f19")),
            "BG_CARD": tk.StringVar(value=self.app.colors.get("BG_CARD", "#131620"))
        }

        self._setup_ui()

    def _setup_ui(self):
        # Scrollable container for settings
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

        # Hook mousewheel
        def _on_mousewheel(event):
            if os.name == 'nt' or sys.platform == 'darwin':
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        scroll_frame.bind_all("<MouseWheel>", _on_mousewheel)

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

        ttk.Label(fr_paths, text="* Relative paths are relative to the app folder. Absolute paths recommended.",
                  font=("Segoe UI", 9, "italic"), foreground=self.app.colors["FG_DIM"]).pack(anchor="w", pady=(5, 0))

        # --- SECTION 3: THEME ---
        fr_theme = ttk.LabelFrame(scroll_frame, text="Theme Customization", padding=15)
        fr_theme.pack(fill="x", pady=10)

        def add_col_row(label, key):
            f = ttk.Frame(fr_theme)
            f.pack(fill="x", pady=5)
            ttk.Label(f, text=label, width=15, anchor="w").pack(side="left")

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

        # --- SAVE ---
        btn_save = ttk.Button(scroll_frame, text="SAVE ALL SETTINGS", command=self._save_settings)
        btn_save.pack(fill="x", pady=30)

    def _browse_path(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d: var.set(d)

    def _save_settings(self):
        try:
            config_path = os.path.join(self.app.paths["root"], "settings.json")

            data = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)

            # 1. Update Scale
            data["ui_scale"] = round(self.scale_var.get(), 2)

            # 2. Update Paths
            # Note: We save them exactly as entered.
            # GUI.py handles relative vs absolute logic on load.
            data["data_dir"] = self.path_vars["data_dir"].get()
            data["chaos_dir"] = self.path_vars["chaos_dir"].get()
            data["output_dir"] = self.path_vars["output_dir"].get()

            # 3. Update Colors
            if "colors" not in data: data["colors"] = {}
            data["colors"]["ACCENT"] = self.col_vars["ACCENT"].get()
            data["colors"]["BG_MAIN"] = self.col_vars["BG_MAIN"].get()
            data["colors"]["BG_CARD"] = self.col_vars["BG_CARD"].get()

            # Write
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)

            if messagebox.askyesno("Saved",
                                   "Settings saved successfully.\n\nRestart Application now to apply changes?"):
                self.app.destroy()
                # Optional: self-restart logic could go here, but usually safer to just close

        except Exception as e:
            messagebox.showerror("Error", f"Could not save settings: {e}")

    def on_theme_change(self):
        pass