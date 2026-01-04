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
from tkinter import ttk
import threading
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
import os
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Symbiosis"

        self.is_running = False
        self.stop_requested = False

        # UI vars
        self.teacher_id = tk.IntVar(value=2)
        self.student_id = tk.IntVar(value=1)

        # Storage settings
        self.autosave_enabled = tk.BooleanVar(value=True)
        self.save_interval = tk.IntVar(value=50)
        self.harvest_enabled = tk.BooleanVar(value=True)
        self.auto_scroll = tk.BooleanVar(value=True)

        # Harvest Config
        self.harvest_dir = os.path.join(self.app.paths['memories'], "harvested")
        if not os.path.exists(self.harvest_dir): os.makedirs(self.harvest_dir)
        self.max_file_size = 10 * 1024 * 1024  # 10 MB limit

        self._setup_ui()

    def _setup_ui(self):
        split = ttk.PanedWindow(self.parent, orient="vertical")
        split.pack(fill="both", expand=True, padx=10, pady=10)

        # --- TOP PANEL: CONTROLS ---
        top_frame = ttk.Frame(split)
        split.add(top_frame, weight=0)

        # 1. Connection Panel
        panel = ttk.LabelFrame(top_frame, text="Neural Link", padding=15)
        panel.pack(fill="x", pady=(0, 10))

        f_link = ttk.Frame(panel)
        f_link.pack(fill="x", pady=5)

        # Selectors
        ttk.Label(f_link, text="TEACHER:", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Spinbox(f_link, from_=1, to=4, textvariable=self.teacher_id, width=3).pack(side="left", padx=5)
        ttk.Label(f_link, text=" >>> STREAMS TO >>> ", foreground=self.app.colors["ACCENT"]).pack(side="left", padx=5)
        ttk.Label(f_link, text="STUDENT:", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Spinbox(f_link, from_=1, to=4, textvariable=self.student_id, width=3).pack(side="left", padx=5)

        # Storage Row
        f_store = ttk.Frame(panel)
        f_store.pack(fill="x", pady=(10, 0))

        # Auto-Save Brain
        ttk.Checkbutton(f_store, text="Auto-Save Brain", variable=self.autosave_enabled).pack(side="left")
        ttk.Label(f_store, text="Every").pack(side="left", padx=2)
        ttk.Entry(f_store, textvariable=self.save_interval, width=4).pack(side="left")
        ttk.Label(f_store, text="Cycles").pack(side="left", padx=(2, 15))

        # Harvest Data
        ttk.Separator(f_store, orient="vertical").pack(side="left", fill="y", padx=5)
        ttk.Checkbutton(f_store, text="Harvest Output (Max 10MB/file)", variable=self.harvest_enabled).pack(side="left",
                                                                                                            padx=5)

        # 2. Action Button
        self.btn_start = ttk.Button(panel, text="INITIATE SYMBIOSIS", command=self._toggle_symbiosis)
        self.btn_start.pack(fill="x", pady=(10, 0))

        # --- BOTTOM PANEL: LOGS ---
        bot_frame = ttk.LabelFrame(split, text="Symbiosis Telemetry", padding=10)
        split.add(bot_frame, weight=1)

        tool_fr = ttk.Frame(bot_frame)
        tool_fr.pack(fill="x")
        ttk.Checkbutton(tool_fr, text="Autoscroll", variable=self.auto_scroll).pack(side="right")

        self.log_box = tk.Text(bot_frame, font=("Consolas", int(9 * getattr(self.app, 'ui_scale', 1.0))), height=15,
                               bg=self.app.colors["BG_MAIN"], fg=self.app.colors["FG_TEXT"], borderwidth=0)
        self.log_box.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(bot_frame, orient="vertical", command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self.log_box.tag_config("info", foreground=self.app.colors["FG_TEXT"])
        self.log_box.tag_config("gen", foreground=self.app.colors["ACCENT"])
        self.log_box.tag_config("save", foreground=self.app.colors["SUCCESS"], font=("Consolas", 9, "bold"))
        self.log_box.tag_config("harvest", foreground="#FDD663", font=("Consolas", 9, "italic"))
        self.log_box.tag_config("warn", foreground=self.app.colors["WARN"])
        self.log_box.tag_config("err", foreground=self.app.colors["ERROR"])

    def _log(self, msg, tag="info"):
        ts = datetime.now().strftime('%H:%M:%S')
        full_msg = f"[{ts}] {msg}\n"
        self.log_box.insert(tk.END, full_msg, tag)
        if self.auto_scroll.get(): self.log_box.see(tk.END)

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])

    def _toggle_symbiosis(self):
        if self.is_running:
            self.stop_requested = True
            self.btn_start.config(text="STOPPING...", state="disabled")
        else:
            t_id = self.teacher_id.get()
            s_id = self.student_id.get()

            if not self.app.lobes[t_id] or not self.app.lobes[s_id]:
                self._log("Error: Both Lobes must be loaded.", "err")
                return

            self.is_running = True
            self.stop_requested = False
            self.btn_start.config(text="SEVER LINK")
            self._log(f"Link Established: Teacher Lobe {t_id} -> Student Lobe {s_id}", "info")

            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        t_id = self.teacher_id.get()
        s_id = self.student_id.get()

        teacher = self.app.lobes[t_id]
        student = self.app.lobes[s_id]
        opt = self.app.optimizers[s_id]
        scaler = self.app.scalers[s_id]

        # --- SAFETY: FREEZE SENSES ---
        frozen_layers = []
        try:
            if hasattr(student, 'vis_emb'):
                for p in student.vis_emb.parameters(): p.requires_grad = False
                frozen_layers.append("Vision")
            if hasattr(student, 'aud_emb'):
                for p in student.aud_emb.parameters(): p.requires_grad = False
                frozen_layers.append("Audio")
        except:
            pass

        msg_freeze = f"Sensors Locked: {', '.join(frozen_layers)}" if frozen_layers else "No Sensors Found"
        self.parent.after(0, lambda: self._log(f"Safety Protocol: {msg_freeze}", "warn"))

        student.train()
        teacher.eval()

        try:
            student_vocab_limit = student.tok_emb.weight.shape[0]
        except:
            student_vocab_limit = 50257

        prompts = [
            "Explain the concept of", "The history of", "Why is", "How does",
            "Describe the function of", "A summary of", "Write a story about",
            "Define the term", "Compare and contrast", "What happens if"
        ]

        try:
            t_tok = getattr(teacher, 'tokenizer', self.app.ribosome.tokenizer)
            s_tok = self.app.ribosome.tokenizer
        except:
            self.parent.after(0, lambda: self._log("Error: Could not locate tokenizers.", "err"))
            self.is_running = False
            return

        cycles = 0

        # Init Harvest File
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        harvest_base = f"symbiosis_session_{timestamp}"
        harvest_part = 1
        harvest_path = os.path.join(self.harvest_dir, f"{harvest_base}_p{harvest_part}.txt")

        while not self.stop_requested:
            try:
                # 1. Generate
                import random
                prompt = random.choice(prompts)

                generated_text = ""
                if hasattr(teacher, "generate"):
                    generated_text = teacher.generate(prompt, max_new_tokens=64)
                else:
                    generated_text = "The neural network is learning to dream."

                if not generated_text: generated_text = "Empty response."

                # 2. Translate
                try:
                    s_tokens = s_tok.encode(generated_text)
                except:
                    if hasattr(s_tok, '__call__'):
                        s_tokens = s_tok(generated_text)['input_ids']
                    else:
                        continue

                # 3. Clamp
                safe_tokens = [t % student_vocab_limit for t in s_tokens]
                t_input = torch.tensor([safe_tokens]).to(self.app.device)

                if t_input.shape[1] < 2: continue

                # 4. Student Learns
                inp = t_input[:, :-1]
                tgt = t_input[:, 1:]

                v = torch.randn(1, 1, 768).to(self.app.device) * 0.01
                a = torch.randn(1, 1, 128).to(self.app.device) * 0.01
                c = torch.zeros(1, 1, 64).to(self.app.device)

                opt.zero_grad()

                with autocast():
                    logits, _, _ = student(v, a, inp, c)
                    offset = (v.shape[1]) + (a.shape[1]) + (c.shape[1])
                    logits_text = logits[:, offset: offset + inp.shape[1], :]
                    loss = F.cross_entropy(logits_text.reshape(-1, logits_text.size(-1)), tgt.reshape(-1))

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                cycles += 1

                # --- LOGGING ---
                clean_text = generated_text.replace('\n', ' ').strip()
                if len(clean_text) > 60: clean_text = clean_text[:60] + "..."
                self.parent.after(0, lambda l=loss.item(), t=clean_text: self._log(f"Loss: {l:.4f} | {t}", "gen"))

                # --- HARVEST & ROTATION ---
                if self.harvest_enabled.get():
                    try:
                        # Check file size (Rotation)
                        if os.path.exists(harvest_path) and os.path.getsize(harvest_path) > self.max_file_size:
                            harvest_part += 1
                            harvest_path = os.path.join(self.harvest_dir, f"{harvest_base}_p{harvest_part}.txt")
                            self.parent.after(0, lambda p=harvest_part: self._log(f"HARVEST: Rotating to Part {p}",
                                                                                  "harvest"))

                        with open(harvest_path, "a", encoding="utf-8") as f:
                            f.write(generated_text + "\n<|endoftext|>\n")
                    except Exception as e:
                        print(f"Harvest Error: {e}")

                # --- AUTO-SAVE BRAIN ---
                if self.autosave_enabled.get():
                    interval = self.save_interval.get()
                    if interval > 0 and cycles % interval == 0:
                        save_path = os.path.join(self.app.paths['lobes'], f"brain_lobe_{s_id}.pt")
                        genome = self.app.lobe_genomes.get(s_id, "Unknown")
                        torch.save({"genome": genome, "state_dict": student.state_dict()}, save_path)
                        self.parent.after(0, lambda c=cycles: self._log(f"AUTO-SAVE: Lobe {s_id} saved at cycle {c}",
                                                                        "save"))

                time.sleep(0.05)

            except Exception as e:
                print(e)
                self.parent.after(0, lambda: self._log("Sync Glitch (Skipping Cycle)", "warn"))
                time.sleep(1)

        # --- RESTORE SENSES ---
        try:
            if hasattr(student, 'vis_emb'):
                for p in student.vis_emb.parameters(): p.requires_grad = True
            if hasattr(student, 'aud_emb'):
                for p in student.aud_emb.parameters(): p.requires_grad = True
        except:
            pass

        self.is_running = False
        self.parent.after(0, lambda: self.btn_start.config(text="INITIATE SYMBIOSIS", state="normal"))
        self.parent.after(0, lambda: self._log("Link Severed. Senses Unlocked.", "info"))