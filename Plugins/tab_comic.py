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
from tkinter import ttk, messagebox
import threading
import os
import time
import torch
from PIL import Image, ImageTk
import traceback
from datetime import datetime


# --- HEADLESS HELPER ---
class MockVar:
    def __init__(self, value=None): self._val = value

    def set(self, value): self._val = value

    def get(self): return self._val


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Comic Factory"
        self.is_running = False
        self.stop_requested = False
        self.image_refs = []  # Keep references to prevent GC

        # --- DYNAMIC PATHS ---
        default_data = self.app.paths.get("data", os.path.join(self.app.paths["root"], "Training_Data"))
        self.output_dir = os.path.join(default_data, "Comics_Output")
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except:
                pass

        # --- STATE ---
        if self.parent is None:
            # Headless Defaults
            self.seed_prompt = MockVar("A robot discovering a flower in a cyberpunk city")
            self.num_panels = MockVar(4)
            self.style_prompt = MockVar("comic book style, vibrant colors, detailed line art")
            self.status_var = MockVar("Ready.")
        else:
            # GUI Defaults
            self.seed_prompt = tk.StringVar(value="A robot discovering a flower in a cyberpunk city")
            self.num_panels = tk.IntVar(value=4)
            self.style_prompt = tk.StringVar(value="comic book style, vibrant colors, detailed line art")
            self.status_var = tk.StringVar(value="Ready.")

        self._setup_ui()

    def _setup_ui(self):
        if self.parent is None: return

        # Main Layout: Left (Controls) | Right (Comic Strip View)
        panel = ttk.PanedWindow(self.parent, orient="horizontal")
        panel.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(panel, width=350)
        right = ttk.Frame(panel)
        panel.add(left, weight=1)
        panel.add(right, weight=3)

        # --- LEFT: CONTROLS ---
        fr_input = ttk.LabelFrame(left, text="Story Settings", padding=15)
        fr_input.pack(fill="x", pady=5)

        ttk.Label(fr_input, text="Story Concept:").pack(anchor="w")
        self.txt_seed = tk.Text(fr_input, height=4, font=("Segoe UI", 10), wrap="word")
        self.txt_seed.insert("1.0", self.seed_prompt.get())
        self.txt_seed.pack(fill="x", pady=5)

        ttk.Label(fr_input, text="Art Style:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(fr_input, textvariable=self.style_prompt).pack(fill="x", pady=5)

        ttk.Label(fr_input, text="Panels:").pack(anchor="w", pady=(10, 0))
        ttk.Spinbox(fr_input, from_=1, to=10, textvariable=self.num_panels).pack(fill="x", pady=5)

        self.btn_gen = ttk.Button(left, text="GENERATE COMIC", command=self._start_generation)
        self.btn_gen.pack(fill="x", pady=20)

        self.lbl_status = ttk.Label(left, textvariable=self.status_var, foreground=self.app.colors["ACCENT"],
                                    wraplength=300)
        self.lbl_status.pack(pady=10)

        # --- RIGHT: PREVIEW AREA ---
        fr_view = ttk.LabelFrame(right, text="Comic Strip Preview", padding=10)
        fr_view.pack(fill="both", expand=True)

        # Scrollable Canvas for vertical strip
        self.canvas = tk.Canvas(fr_view, bg=self.app.colors["BG_CARD"], highlightthickness=0)
        sb = ttk.Scrollbar(fr_view, orient="vertical", command=self.canvas.yview)

        self.scroll_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=sb.set)
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _update_status(self, msg):
        if self.parent:
            self.status_var.set(msg)
            self.parent.update_idletasks()
        else:
            print(f"[Comic] {msg}")

    def _start_generation(self):
        if self.is_running: return

        # Get text from widget if GUI
        if self.parent:
            seed = self.txt_seed.get("1.0", tk.END).strip()
            self.seed_prompt.set(seed)

        active_id = self.app.active_lobe.get()
        brain = self.app.lobes[active_id]

        if not brain:
            messagebox.showerror("Error", "No Lobe Loaded.")
            return

        # Check Capabilities
        self.is_diffusion = hasattr(brain, 'timestep_emb') or (self.app.lobe_types.get(active_id) == "diffusion")

        self.is_running = True
        if self.parent: self.btn_gen.config(state="disabled")

        # Clear previous
        if self.parent:
            for w in self.scroll_frame.winfo_children(): w.destroy()
            self.image_refs = []

        threading.Thread(target=self._worker_comic, daemon=True).start()

    def _worker_comic(self):
        try:
            brain = self.app.lobes[self.app.active_lobe.get()]
            ribosome = self.app.ribosome

            prompt = self.seed_prompt.get()
            style = self.style_prompt.get()
            panels = self.num_panels.get()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            comic_folder = os.path.join(self.output_dir, f"Comic_{ts}")
            os.makedirs(comic_folder, exist_ok=True)

            self._update_status("Drafting Storyboard...")

            # 1. Generate Storyboard (Text)
            # If Diffusion model only, skip text gen or use internal prompt logic
            storyboard = []
            if not self.is_diffusion:
                # AR Model: Ask it to split story
                story_prompt = f"Create a {panels}-panel comic script about: {prompt}. Format: Panel 1: [Desc], Panel 2: [Desc]..."
                story_text = self._generate_text(brain, ribosome, story_prompt, max_new=200)

                # Naive parse
                parts = story_text.split("Panel")
                for p in parts:
                    if len(p.strip()) > 10:
                        storyboard.append(p.strip())

                # Fallback
                if len(storyboard) < panels:
                    storyboard = [f"{prompt}, scene {i + 1}" for i in range(panels)]
            else:
                # Diffusion Model: Just iterate variations
                storyboard = [f"{prompt}, sequence {i + 1}" for i in range(panels)]

            storyboard = storyboard[:panels]

            # 2. Generate Panels
            for i, panel_desc in enumerate(storyboard):
                self._update_status(f"Rendering Panel {i + 1}/{panels}...")

                full_prompt = f"{panel_desc}, {style}"

                img = self._generate_image(brain, ribosome, full_prompt)

                if img:
                    # Save
                    fname = f"panel_{i + 1}.png"
                    save_path = os.path.join(comic_folder, fname)
                    img.save(save_path)

                    # Display
                    if self.parent:
                        self.parent.after(0, lambda image=img, txt=panel_desc: self._add_panel_to_ui(image, txt))

                time.sleep(0.5)

            self._update_status(f"Comic Complete! Saved to {comic_folder}")

        except Exception as e:
            self._update_status(f"Error: {e}")
            traceback.print_exc()
        finally:
            self.is_running = False
            if self.parent: self.parent.after(0, lambda: self.btn_gen.config(state="normal"))

    def _generate_text(self, brain, ribosome, prompt, max_new=100):
        ids = ribosome._tokenize(prompt)
        t = torch.tensor(ids, device=self.app.device).unsqueeze(0)
        v = torch.zeros(1, 1, 768).to(self.app.device)
        a = torch.zeros(1, 1, 128).to(self.app.device)

        out_ids = []
        with self.app.gpu_lock:
            with torch.no_grad():
                for _ in range(max_new):
                    logits, _, _ = brain(v, a, t)
                    next_tok = torch.argmax(logits[:, -1, :], dim=-1)
                    t = torch.cat([t, next_tok.unsqueeze(0)], dim=1)
                    out_ids.append(next_tok.item())
                    if next_tok.item() == 50256: break

        return ribosome.decode(out_ids)

    def _generate_image(self, brain, ribosome, prompt):
        # Requires a model capable of image generation (Diffusion)
        if not hasattr(brain, 'timestep_emb'):
            return None  # AR model can't draw (yet)

        ids = ribosome._tokenize(prompt)

        with self.app.gpu_lock:
            with torch.no_grad():
                # Standard Diffusion Generation Call
                # (Assumes brain.generate returns tokens containing image blocks)
                tokens = brain.generate(
                    prompt_tokens=ids,
                    max_length=1024 + len(ids),  # Enough for 32x32 patches
                    steps=25,
                    temperature=1.0
                )

        # Extract Image from tokens
        # Find visual tokens range
        vis_tokens = [t for t in tokens if t >= ribosome.image_vocab_base and t < ribosome.audio_vocab_base]

        if not vis_tokens: return None

        try:
            return ribosome.decode_image_tokens(vis_tokens)
        except:
            return None

    def _add_panel_to_ui(self, pil_img, text):
        # Resize for display
        display_img = pil_img.copy()
        display_img.thumbnail((500, 500))
        photo = ImageTk.PhotoImage(display_img)
        self.image_refs.append(photo)

        # Container
        fr = ttk.Frame(self.scroll_frame, style="Card.TFrame", padding=10)
        fr.pack(fill="x", pady=10, padx=10)

        # Image
        lbl_img = ttk.Label(fr, image=photo, background=self.app.colors["BG_CARD"])
        lbl_img.pack()

        # Caption
        lbl_cap = ttk.Label(fr, text=text, wraplength=480, justify="center",
                            background=self.app.colors["BG_CARD"], foreground=self.app.colors["FG_TEXT"])
        lbl_cap.pack(pady=(5, 0))

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'canvas'): self.canvas.config(bg=c["BG_CARD"])