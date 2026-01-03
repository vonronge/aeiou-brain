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
from tkinter import ttk, messagebox, filedialog
import threading
import os
import time
import torch
import torch.nn.functional as F
from datetime import datetime
import traceback


# --- HEADLESS HELPER ---
class MockVar:
    def __init__(self, value=None): self._val = value

    def set(self, value): self._val = value

    def get(self): return self._val


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Dream State"
        self.is_dreaming = False
        self.stop_requested = False

        # --- DYNAMIC PATHS ---
        # Get configured data dir or default to local
        default_data = self.app.paths.get("data", os.path.join(self.app.paths["root"], "Training_Data"))

        # Chaos Buffer (Where dreams go to be sorted by Factory)
        self.chaos_dir = os.path.join(default_data, "Chaos_Buffer")
        if not os.path.exists(self.chaos_dir):
            try:
                os.makedirs(self.chaos_dir)
            except:
                pass

        # --- STATE ---
        if self.parent is None:
            # Headless Defaults
            self.temperature = MockVar(0.85)
            self.top_k = MockVar(50)
            self.max_length = MockVar(1024)
            self.refresh_rate = MockVar(1.0)
            self.autosave = MockVar(True)
            self.seed_prompt = MockVar("The nature of consciousness is")
            self.diffusion_steps = MockVar(20)
        else:
            # GUI Defaults
            self.temperature = tk.DoubleVar(value=0.85)
            self.top_k = tk.IntVar(value=50)
            self.max_length = tk.IntVar(value=1024)
            self.refresh_rate = tk.DoubleVar(value=0.5)
            self.autosave = tk.BooleanVar(value=True)
            self.seed_prompt = tk.StringVar(value="The nature of consciousness is")
            self.diffusion_steps = tk.IntVar(value=20)

        self._setup_ui()

    def _setup_ui(self):
        if self.parent is None: return

        # Layout
        panel = ttk.PanedWindow(self.parent, orient="horizontal")
        panel.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(panel, width=300)
        right = ttk.Frame(panel)
        panel.add(left, weight=1)
        panel.add(right, weight=3)

        # --- CONTROLS ---
        fr_cfg = ttk.LabelFrame(left, text="Dream Parameters", padding=10)
        fr_cfg.pack(fill="x", pady=5)

        ttk.Label(fr_cfg, text="Seed Prompt:").pack(anchor="w")
        ttk.Entry(fr_cfg, textvariable=self.seed_prompt).pack(fill="x", pady=(0, 10))

        def add_scale(lbl, var, min_v, max_v):
            ttk.Label(fr_cfg, text=lbl).pack(anchor="w")
            ttk.Scale(fr_cfg, from_=min_v, to=max_v, variable=var, orient="horizontal").pack(fill="x", pady=(0, 10))

        add_scale("Temperature (Creativity):", self.temperature, 0.1, 2.0)
        add_scale("Top-K (Diversity):", self.top_k, 1, 200)
        add_scale("Max Length:", self.max_length, 64, 4096)

        # Diffusion Specific
        self.fr_diff = ttk.Frame(fr_cfg)
        self.fr_diff.pack(fill="x")
        ttk.Label(self.fr_diff, text="Diffusion Steps:", foreground=self.app.colors["ACCENT"]).pack(anchor="w")
        ttk.Scale(self.fr_diff, from_=1, to=100, variable=self.diffusion_steps, orient="horizontal").pack(fill="x")

        ttk.Checkbutton(fr_cfg, text="Autosave to Chaos", variable=self.autosave).pack(anchor="w", pady=5)

        self.btn_start = ttk.Button(left, text="INITIATE DREAM STATE", command=self._toggle_dream)
        self.btn_start.pack(fill="x", pady=20)

        # --- VISUALIZATION ---
        fr_vis = ttk.LabelFrame(right, text="The Mind's Eye", padding=10)
        fr_vis.pack(fill="both", expand=True, pady=5)

        self.txt_out = tk.Text(fr_vis, font=("Consolas", 11), bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"], wrap="word", padx=15, pady=15, borderwidth=0)
        self.txt_out.pack(fill="both", expand=True)

        # Tags
        self.txt_out.tag_config("prompt", foreground="#888")
        self.txt_out.tag_config("new", foreground=self.app.colors["ACCENT"])
        self.txt_out.tag_config("diff", foreground=self.app.colors["SUCCESS"])

    def _log(self, text, tag="new"):
        if self.parent:
            self.txt_out.insert(tk.END, text, tag)
            self.txt_out.see(tk.END)
        else:
            print(text, end="", flush=True)

    def _toggle_dream(self):
        if self.is_dreaming:
            self.stop_requested = True
            if self.parent: self.btn_start.config(text="Stopping...")
        else:
            lobe_id = self.app.active_lobe.get()
            if self.app.lobes[lobe_id] is None:
                msg = "No Lobe Loaded."
                if self.parent:
                    messagebox.showerror("Error", msg)
                else:
                    print(msg)
                return

            self.is_dreaming = True
            self.stop_requested = False
            if self.parent:
                self.btn_start.config(text="WAKE UP")
                self.txt_out.delete("1.0", tk.END)

            threading.Thread(target=self._dream_loop, daemon=True).start()

    def _dream_loop(self):
        try:
            lobe_id = self.app.active_lobe.get()
            brain = self.app.lobes[lobe_id]
            ribosome = self.app.ribosome

            # Detect Type
            is_diffusion = hasattr(brain, 'timestep_emb') or (self.app.lobe_types.get(lobe_id) == "diffusion")

            if self.parent and is_diffusion:
                self.fr_diff.pack(fill="x")
            elif self.parent:
                self.fr_diff.pack_forget()

            while not self.stop_requested:
                prompt = self.seed_prompt.get()
                self._log(f"\n\n>>> SEED: {prompt}\n", "prompt")

                # Tokenize
                ids = ribosome._tokenize(prompt)

                generated_text = ""

                with self.app.gpu_lock:
                    brain.eval()
                    with torch.no_grad():
                        # --- DIFFUSION GENERATION ---
                        if is_diffusion:
                            steps = self.diffusion_steps.get()
                            # Generate
                            tokens = brain.generate(
                                prompt_tokens=ids,
                                max_length=self.max_length.get(),
                                steps=steps,
                                temperature=self.temperature.get()
                            )
                            generated_text = ribosome.decode(tokens)
                            # Remove prompt echo
                            if generated_text.startswith(prompt):
                                generated_text = generated_text[len(prompt):]

                            self._log(generated_text, "diff")

                        # --- AR GENERATION ---
                        else:
                            t = torch.tensor(ids, device=self.app.device).unsqueeze(0)
                            # Placeholder sensors
                            v = torch.zeros(1, 1, 768).to(self.app.device)
                            a = torch.zeros(1, 1, 128).to(self.app.device)

                            for _ in range(self.max_length.get()):
                                if self.stop_requested: break

                                logits, _, _ = brain(v, a, t)
                                next_logits = logits[:, -1, :] / self.temperature.get()
                                probs = F.softmax(next_logits, dim=-1)

                                # Top-K
                                k = self.top_k.get()
                                if k > 0:
                                    v_top, _ = torch.topk(probs, k)
                                    probs[probs < v_top[:, [-1]]] = 0
                                    probs = probs / probs.sum(dim=-1, keepdim=True)

                                next_tok = torch.multinomial(probs, 1)
                                t = torch.cat([t, next_tok], dim=1)

                                # Stream Decode
                                word = ribosome.decode([next_tok.item()])
                                self._log(word, "new")
                                generated_text += word

                                if next_tok.item() == 50256: break  # EOS
                                time.sleep(0.01)  # UI Breather

                # --- AUTOSAVE TO CHAOS ---
                if self.autosave.get() and generated_text.strip():
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"dream_{ts}.txt"
                    path = os.path.join(self.chaos_dir, fname)

                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(f"PROMPT: {prompt}\n\n")
                            f.write(generated_text)
                        if self.parent:
                            self._log(f"\n[Saved to Chaos Buffer: {fname}]", "prompt")
                        else:
                            print(f"\n[Saved: {fname}]")
                    except Exception as e:
                        print(f"Save Error: {e}")

                if self.stop_requested: break
                time.sleep(self.refresh_rate.get())

        except Exception as e:
            if self.parent:
                self._log(f"\nCRASH: {e}", "error")
            else:
                print(f"CRASH: {e}")
            traceback.print_exc()
        finally:
            self.is_dreaming = False
            if self.parent: self.btn_start.config(text="INITIATE DREAM STATE")

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'txt_out'):
            self.txt_out.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])