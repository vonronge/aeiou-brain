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
"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Comic Architect:
A specialized sequencer for Cytosis.
Generates multi-panel visual narratives (Comics/Manga).
- Defines a 4-Panel Script.
- Generates consistency across panels (using Seed/Lobe).
- Stitches results into a page layout.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import queue
import os
import time
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Comic Architect"

        self.is_rendering = False
        self.stop_requested = False
        self.update_queue = queue.Queue()

        # Result Cache
        self.panel_images = [None] * 4
        self.final_page = None

        # --- SETTINGS ---
        self.style_prompt = tk.StringVar(value="black and white manga style, high contrast, ink lines")
        self.neg_prompt = tk.StringVar(value="color, photorealistic, 3d render, blurry")
        self.steps = tk.IntVar(value=25)
        self.cfg = tk.DoubleVar(value=7.0)
        self.seed = tk.IntVar(value=-1)  # Fixed seed ensures character consistency across panels

        # Panel Scripts
        self.script_p1 = tk.StringVar(value="Panel 1: A cybernetic detective stands in rain")
        self.script_p2 = tk.StringVar(value="Panel 2: Close up on his glowing red eye")
        self.script_p3 = tk.StringVar(value="Panel 3: He holds a datapad showing 'ERROR'")
        self.script_p4 = tk.StringVar(value="Panel 4: He looks up at a massive neon tower")

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return
        scale = getattr(self.app, 'ui_scale', 1.0)

        # Layout: Script (Left) | Preview (Right)
        split = ttk.PanedWindow(self.parent, orient="horizontal")
        split.pack(fill="both", expand=True, padx=10, pady=10)

        # --- LEFT: SCRIPTING ---
        fr_script = ttk.Frame(split)
        split.add(fr_script, weight=1)

        # Global Style
        fr_style = ttk.LabelFrame(fr_script, text="Art Direction", padding=10)
        fr_style.pack(fill="x", pady=5)

        ttk.Label(fr_style, text="Global Style:").pack(anchor="w")
        ttk.Entry(fr_style, textvariable=self.style_prompt).pack(fill="x", pady=(0, 5))

        ttk.Label(fr_style, text="Negative:").pack(anchor="w")
        ttk.Entry(fr_style, textvariable=self.neg_prompt).pack(fill="x")

        # Panels
        fr_panels = ttk.LabelFrame(fr_script, text="Storyboard", padding=10)
        fr_panels.pack(fill="x", pady=5)

        def add_panel_in(lbl, var):
            f = ttk.Frame(fr_panels)
            f.pack(fill="x", pady=2)
            ttk.Label(f, text=lbl, width=8, font=("Segoe UI", int(9 * scale), "bold")).pack(side="left")
            ttk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True)

        add_panel_in("Panel 1:", self.script_p1)
        add_panel_in("Panel 2:", self.script_p2)
        add_panel_in("Panel 3:", self.script_p3)
        add_panel_in("Panel 4:", self.script_p4)

        # Tech Settings
        fr_tech = ttk.LabelFrame(fr_script, text="Render Config", padding=10)
        fr_tech.pack(fill="x", pady=5)

        r1 = ttk.Frame(fr_tech)
        r1.pack(fill="x")
        ttk.Label(r1, text="Steps:").pack(side="left")
        ttk.Spinbox(r1, from_=10, to=100, textvariable=self.steps, width=5).pack(side="left", padx=5)

        ttk.Label(r1, text="CFG:").pack(side="left", padx=(10, 0))
        ttk.Spinbox(r1, from_=1.0, to=20.0, textvariable=self.cfg, width=5, increment=0.5).pack(side="left", padx=5)

        ttk.Label(r1, text="Seed:").pack(side="left", padx=(10, 0))
        ttk.Entry(r1, textvariable=self.seed, width=10).pack(side="left", padx=5)

        # Buttons
        self.btn_draw = ttk.Button(fr_script, text="🎨 DRAW PAGE", command=self._start_render)
        self.btn_draw.pack(fill="x", pady=20)

        self.lbl_status = ttk.Label(fr_script, text="Ready.", foreground=self.app.colors["FG_DIM"], anchor="center")
        self.lbl_status.pack(fill="x")

        # --- RIGHT: PAGE PREVIEW ---
        fr_view = ttk.LabelFrame(split, text="Page Layout", padding=10)
        split.add(fr_view, weight=3)

        self.canvas = tk.Canvas(fr_view, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Toolbar
        tb = ttk.Frame(fr_view)
        tb.pack(fill="x", pady=5)
        ttk.Button(tb, text="💾 SAVE COMIC", command=self._save_page).pack(side="right")

    # --- ACTIONS ---
    def _start_render(self):
        # Validate Lobe
        active_id = self.app.active_lobe.get()
        lobe = self.app.lobe_manager.get_lobe(active_id)
        if not lobe:
            messagebox.showerror("Void", "No Lobe Loaded.")
            return

        self.is_rendering = True
        self.stop_requested = False
        self.btn_draw.config(state="disabled", text="INKING...")

        # Clear old
        self.panel_images = [None] * 4
        self.final_page = None
        self.canvas.delete("all")

        threading.Thread(target=self._worker, args=(lobe,), daemon=True).start()

    def _worker(self, lobe):
        try:
            from Organelles.cytosis import DreamConfig

            prompts = [
                self.script_p1.get(),
                self.script_p2.get(),
                self.script_p3.get(),
                self.script_p4.get()
            ]
            style = self.style_prompt.get()
            neg = self.neg_prompt.get()
            seed_base = self.seed.get()

            # If seed is -1, generate a random base seed for the PAGE
            # This ensures the page has a consistent 'vibe' if we increment it per panel
            import random
            if seed_base == -1: seed_base = random.randint(0, 2 ** 32 - 1)

            for i, p_text in enumerate(prompts):
                if self.stop_requested: break

                # Combine Panel Prompt + Global Style
                full_prompt = f"{style}, {p_text}"

                # Update Status
                self.update_queue.put(lambda idx=i: self.lbl_status.config(text=f"Rendering Panel {idx + 1}/4..."))

                # Create Config
                # We increment seed slightly per panel to allow variation while keeping style
                cfg = DreamConfig(
                    prompt=full_prompt,
                    negative_prompt=neg,
                    cfg_scale=self.cfg.get(),
                    steps=self.steps.get(),
                    seed=seed_base + i,
                    modality="visual"
                )

                # Call Cytosis
                res = self.app.cytosis.dream(lobe, cfg)
                if res.get('image'):
                    self.panel_images[i] = res['image']
                    # Show progress on canvas (simple placeholder)
                    self.update_queue.put(lambda idx=i: self._draw_placeholder(idx))

            # Stitch
            if not self.stop_requested:
                self.update_queue.put(lambda: self.lbl_status.config(text="Assembling Page..."))
                self._assemble_page(prompts)
                self.update_queue.put(lambda: self._display_result())

        except Exception as e:
            self.update_queue.put(lambda: self._on_error(str(e)))

        self.is_rendering = False
        self.update_queue.put(lambda: self.btn_draw.config(state="normal", text="🎨 DRAW PAGE"))

    def _assemble_page(self, captions):
        """Combines 4 images into a 2x2 grid with captions."""
        # Assume 512x512 panels
        w, h = 512, 512
        # Page size: 2w x 2h + padding
        pad = 20
        page_w = (w * 2) + (pad * 3)
        page_h = (h * 2) + (pad * 3) + 100  # Extra space for bottom captions if needed

        bg_color = (255, 255, 255)  # White paper
        page = Image.new("RGB", (page_w, page_h), bg_color)
        draw = ImageDraw.Draw(page)

        # Positions
        positions = [
            (pad, pad), (pad + w + pad, pad),
            (pad, pad + h + pad), (pad + w + pad, pad + h + pad)
        ]

        try:
            # Try to load a font, or default
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for i, img in enumerate(self.panel_images):
            if img:
                x, y = positions[i]
                # Paste Image
                page.paste(img, (x, y))
                # Draw Border
                draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), width=3)

                # Draw Caption (Simple strip at bottom of panel)
                # cap = captions[i]
                # draw.text((x, y + h + 5), cap[:50], fill=(0,0,0), font=font)

        self.final_page = page

    def _draw_placeholder(self, idx):
        # Just to show user something happened
        pass

    def _display_result(self):
        if not self.final_page: return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        w, h = self.final_page.size
        ratio = min(cw / w, ch / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = self.final_page.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)

        self.lbl_status.config(text="Page Complete.", foreground=self.app.colors["SUCCESS"])

    def _save_page(self):
        if not self.final_page: return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        f = filedialog.asksaveasfilename(initialfile=f"comic_{ts}.png", defaultextension=".png")
        if f:
            self.final_page.save(f)
            self.app.golgi.success(f"Comic saved to {f}", source="Comic")

    def _on_error(self, msg):
        self.lbl_status.config(text="Render Error.", foreground=self.app.colors["WARN"])
        self.app.golgi.error(f"Comic Error: {msg}", source="Comic")

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                fn = self.update_queue.get_nowait()
                fn()
            except:
                break
        if self.parent: self.parent.after(100, self._process_gui_queue)

    def on_theme_change(self):
        pass