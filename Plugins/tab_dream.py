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
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import os
import time
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Dream Studio"

        self.is_dreaming = False
        self.update_queue = queue.Queue()
        self.last_image = None  # PIL Image

        # --- SETTINGS ---
        self.prompt_text = tk.StringVar(value="A futuristic bio-digital interface, glowing neon neurons")
        self.neg_prompt = tk.StringVar(value="blurry, low quality, distortion")

        self.cfg_scale = tk.DoubleVar(value=5.0)
        self.steps = tk.IntVar(value=30)
        self.seed = tk.IntVar(value=-1)  # -1 = Random
        self.resolution = tk.IntVar(value=256)  # Matches Ribosome default

        self.mode = tk.StringVar(value="Visual")  # Visual, Audio, Text

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        scale = getattr(self.app, 'ui_scale', 1.0)

        # Split: Controls (Left) | Preview (Right)
        panes = ttk.PanedWindow(self.parent, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=10, pady=10)

        # --- LEFT: CONTROLS ---
        fr_ctrl = ttk.Frame(panes)
        panes.add(fr_ctrl, weight=1)

        # 1. Prompt
        lbl_p = ttk.Label(fr_ctrl, text="Dream Prompt:", font=("Segoe UI", int(11 * scale), "bold"))
        lbl_p.pack(anchor="w", pady=(0, 5))

        self.txt_prompt = tk.Text(fr_ctrl, height=4, font=("Segoe UI", int(10 * scale)),
                                  bg=self.app.colors["BG_CARD"], fg=self.app.colors["FG_TEXT"],
                                  insertbackground=self.app.colors["ACCENT"], wrap="word")
        self.txt_prompt.insert("1.0", self.prompt_text.get())
        self.txt_prompt.pack(fill="x", pady=(0, 10))

        lbl_n = ttk.Label(fr_ctrl, text="Negative Prompt:", font=("Segoe UI", int(10 * scale)))
        lbl_n.pack(anchor="w")
        entry_neg = ttk.Entry(fr_ctrl, textvariable=self.neg_prompt)
        entry_neg.pack(fill="x", pady=(0, 15))

        # 2. Parameters
        fr_param = ttk.LabelFrame(fr_ctrl, text="Inference Parameters", padding=10)
        fr_param.pack(fill="x", pady=5)

        # Mode
        r0 = ttk.Frame(fr_param)
        r0.pack(fill="x", pady=5)
        ttk.Label(r0, text="Modality:").pack(side="left")
        ttk.Combobox(r0, textvariable=self.mode, values=["Visual", "Audio", "Narrative"],
                     state="readonly", width=15).pack(side="left", padx=10)

        # CFG
        r1 = ttk.Frame(fr_param)
        r1.pack(fill="x", pady=5)
        ttk.Label(r1, text="Guidance (CFG):").pack(side="left", width=15)
        s_cfg = ttk.Scale(r1, from_=1.0, to=20.0, variable=self.cfg_scale, orient="horizontal")
        s_cfg.pack(side="left", fill="x", expand=True)
        l_cfg = ttk.Label(r1, text=f"{self.cfg_scale.get():.1f}")
        l_cfg.pack(side="left", padx=5)
        s_cfg.configure(command=lambda v: l_cfg.configure(text=f"{float(v):.1f}"))

        # Steps
        r2 = ttk.Frame(fr_param)
        r2.pack(fill="x", pady=5)
        ttk.Label(r2, text="Steps:").pack(side="left", width=15)
        ttk.Spinbox(r2, from_=1, to=100, textvariable=self.steps, width=5).pack(side="left")

        # Seed
        r3 = ttk.Frame(fr_param)
        r3.pack(fill="x", pady=5)
        ttk.Label(r3, text="Seed (-1=Rnd):").pack(side="left", width=15)
        ttk.Entry(r3, textvariable=self.seed, width=10).pack(side="left")

        # 3. Action
        self.btn_dream = ttk.Button(fr_ctrl, text="✨ DREAM", command=self._start_dreaming)
        self.btn_dream.pack(fill="x", pady=20)

        self.lbl_status = ttk.Label(fr_ctrl, text="Ready.", foreground=self.app.colors["FG_DIM"], anchor="center")
        self.lbl_status.pack(fill="x")

        # --- RIGHT: PREVIEW ---
        fr_view = ttk.LabelFrame(panes, text="Manifestation", padding=10)
        panes.add(fr_view, weight=3)

        self.canvas = tk.Canvas(fr_view, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Toolbar
        fr_tool = ttk.Frame(fr_view)
        fr_tool.pack(fill="x", pady=5)
        ttk.Button(fr_tool, text="💾 SAVE", command=self._save_result).pack(side="right")
        ttk.Button(fr_tool, text="🗑️ CLEAR", command=self._clear_result).pack(side="right", padx=5)

    def _start_dreaming(self):
        # 1. Validate
        active_id = self.app.active_lobe.get()
        lobe = self.app.lobe_manager.get_lobe(active_id)
        if not lobe:
            messagebox.showerror("Void", "No Lobe Loaded. Please activate a model first.")
            return

        self.is_dreaming = True
        self.btn_dream.config(state="disabled", text="DREAMING...")
        self.lbl_status.config(text="Initializing Cytosis...", foreground=self.app.colors["ACCENT"])

        # Update prompt var from text box
        self.prompt_text.set(self.txt_prompt.get("1.0", tk.END).strip())

        # 2. Launch Thread
        threading.Thread(target=self._worker, args=(lobe,), daemon=True).start()

    def _worker(self, lobe):
        try:
            # Prepare Config
            from Organelles.cytosis import DreamConfig

            cfg = DreamConfig(
                prompt=self.prompt_text.get(),
                negative_prompt=self.neg_prompt.get(),
                cfg_scale=self.cfg_scale.get(),
                steps=self.steps.get(),
                seed=self.seed.get() if self.seed.get() != -1 else None,
                modality=self.mode.get().lower()
            )

            # Call Organelle
            self.update_queue.put(lambda: self.lbl_status.config(text="Inferencing..."))

            # Returns dict {'image': PIL, 'audio': path, 'text': str}
            result = self.app.cytosis.dream(lobe, cfg)

            self.update_queue.put(lambda: self._on_success(result))

        except Exception as e:
            self.update_queue.put(lambda: self._on_error(str(e)))

    def _on_success(self, result):
        self.is_dreaming = False
        self.btn_dream.config(state="normal", text="✨ DREAM")
        self.lbl_status.config(text="Manifestation Complete.", foreground=self.app.colors["SUCCESS"])

        # Handle Output
        if result.get("image"):
            self.last_image = result["image"]
            self._display_image(self.last_image)

        if result.get("audio"):
            # Auto-play or show path
            self.lbl_status.config(text=f"Audio Generated: {os.path.basename(result['audio'])}")

        if result.get("text"):
            # Show in a popup or separate box? For now, log it.
            self.app.golgi.info(f"Narrative: {result['text'][:50]}...", source="Dream")

    def _on_error(self, err_msg):
        self.is_dreaming = False
        self.btn_dream.config(state="normal", text="✨ DREAM")
        self.lbl_status.config(text="Nightmare (Error).", foreground=self.app.colors["WARN"])
        self.app.golgi.error(f"Dream Failed: {err_msg}", source="Dream")
        messagebox.showerror("Dream Failed", err_msg)

    def _display_image(self, pil_img):
        # Resize to fit canvas
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: return

        # Aspect Ratio
        w, h = pil_img.size
        ratio = min(cw / w, ch / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)  # Keep ref

        self.canvas.delete("all")
        # Center
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)

    def _save_result(self):
        if not self.last_image: return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        f = filedialog.asksaveasfilename(initialfile=f"dream_{ts}.png", defaultextension=".png")
        if f:
            self.last_image.save(f)
            self.app.golgi.success(f"Saved dream to {f}", source="Dream")

    def _clear_result(self):
        self.canvas.delete("all")
        self.last_image = None
        self.lbl_status.config(text="Canvas Cleared.")

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                fn = self.update_queue.get_nowait()
                fn()
            except:
                break
        if self.parent: self.parent.after(100, self._process_gui_queue)

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'txt_prompt'):
            self.txt_prompt.config(bg=c["BG_CARD"], fg=c["FG_TEXT"], insertbackground=c["ACCENT"])