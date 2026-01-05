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

The Council Chamber:
Mixture of Experts (MoE) Simulation.
Submits a query to multiple loaded Lobes and aggregates their responses.
Can optionally perform a "Synthesis" pass where one Lobe summarizes the others.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn.functional as F
import threading
import queue
import time


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "The Council"

        self.is_running = False
        self.stop_requested = False
        self.update_queue = queue.Queue()

        # Settings
        self.active_members = {
            1: tk.BooleanVar(value=True),
            2: tk.BooleanVar(value=False),
            3: tk.BooleanVar(value=False),
            4: tk.BooleanVar(value=False)
        }

        self.synthesis_enabled = tk.BooleanVar(value=False)
        self.synthesizer_id = tk.IntVar(value=1)

        # Generation Params
        self.temperature = tk.DoubleVar(value=0.7)
        self.max_tokens = tk.IntVar(value=150)

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        scale = getattr(self.app, 'ui_scale', 1.0)

        # 1. TOP: CONFIGURATION
        fr_config = ttk.LabelFrame(self.parent, text="Council Configuration", padding=10)
        fr_config.pack(fill="x", padx=10, pady=5)

        # Members
        f_mem = ttk.Frame(fr_config)
        f_mem.pack(side="left", fill="x", expand=True)
        ttk.Label(f_mem, text="Active Councilors:", font=("Segoe UI", int(10 * scale), "bold")).pack(anchor="w")

        r_m = ttk.Frame(f_mem)
        r_m.pack(anchor="w")
        for i in range(1, 5):
            ttk.Checkbutton(r_m, text=f"Lobe {i}", variable=self.active_members[i]).pack(side="left", padx=5)

        # Synthesis
        f_syn = ttk.Frame(fr_config)
        f_syn.pack(side="left", fill="x", expand=True, padx=20)

        ttk.Checkbutton(f_syn, text="Enable Synthesis", variable=self.synthesis_enabled).pack(anchor="w")

        r_s = ttk.Frame(f_syn)
        r_s.pack(anchor="w", pady=2)
        ttk.Label(r_s, text="Synthesizer:").pack(side="left")
        for i in range(1, 5):
            ttk.Radiobutton(r_s, text=f"{i}", variable=self.synthesizer_id, value=i).pack(side="left")

        # Params
        f_par = ttk.Frame(fr_config)
        f_par.pack(side="left")
        ttk.Label(f_par, text="Temp:").pack(side="left")
        ttk.Entry(f_par, textvariable=self.temperature, width=5).pack(side="left", padx=5)
        ttk.Label(f_par, text="Len:").pack(side="left")
        ttk.Entry(f_par, textvariable=self.max_tokens, width=5).pack(side="left", padx=5)

        # 2. MAIN: PROMPT & OUTPUT
        panes = ttk.PanedWindow(self.parent, orient="vertical")
        panes.pack(fill="both", expand=True, padx=10, pady=5)

        # Prompt Area
        fr_prompt = ttk.LabelFrame(panes, text="Topic for Debate", padding=10)
        panes.add(fr_prompt, weight=1)

        txt_font = ("Segoe UI", int(11 * scale))
        self.txt_prompt = tk.Text(fr_prompt, height=4, font=txt_font,
                                  bg=self.app.colors["BG_CARD"], fg=self.app.colors["FG_TEXT"],
                                  insertbackground=self.app.colors["ACCENT"])
        self.txt_prompt.pack(fill="both", expand=True)
        self.txt_prompt.insert("1.0", "Explain the concept of entropy in simple terms.")

        # Action Bar
        fr_act = ttk.Frame(fr_prompt)
        fr_act.pack(fill="x", pady=(5, 0))
        self.btn_run = ttk.Button(fr_act, text="CONVENE COUNCIL", command=self._start_council)
        self.btn_run.pack(side="right")

        # Output Grid
        # We create a flexible grid for outputs
        self.fr_outputs = ttk.Frame(panes)
        panes.add(self.fr_outputs, weight=4)

        # We will dynamically pack output boxes here
        self.output_boxes = {}  # Map lobe_id -> text_widget

    def _setup_output_grid(self):
        # Clear existing
        for w in self.fr_outputs.winfo_children(): w.destroy()
        self.output_boxes = {}

        active_ids = [i for i, v in self.active_members.items() if v.get()]
        if self.synthesis_enabled.get():
            active_ids.append(99)  # 99 = Synthesis ID

        count = len(active_ids)
        if count == 0: return

        scale = getattr(self.app, 'ui_scale', 1.0)
        out_font = ("Segoe UI", int(10 * scale))

        # Layout Logic: 1-2 items = 1 row; 3-4 items = 2 rows
        cols = 2 if count > 1 else 1

        for idx, lid in enumerate(active_ids):
            row = idx // cols
            col = idx % cols

            title = f"Lobe {lid} Opinion" if lid != 99 else "FINAL SYNTHESIS"
            color = self.app.colors["ACCENT"] if lid == 99 else self.app.colors["FG_TEXT"]

            fr = ttk.LabelFrame(self.fr_outputs, text=title)
            fr.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            self.fr_outputs.grid_columnconfigure(col, weight=1)
            self.fr_outputs.grid_rowconfigure(row, weight=1)

            txt = tk.Text(fr, font=out_font, wrap="word", height=5,
                          bg=self.app.colors["BG_MAIN"], fg=color)
            txt.pack(fill="both", expand=True)
            self.output_boxes[lid] = txt

    def _start_council(self):
        prompt = self.txt_prompt.get("1.0", tk.END).strip()
        if not prompt: return

        # Validate Members
        targets = [i for i, v in self.active_members.items() if v.get()]
        if not targets:
            messagebox.showwarning("Empty Chamber", "Select at least one active councilor.")
            return

        # Setup UI
        self._setup_output_grid()
        self.is_running = True
        self.stop_requested = False
        self.btn_run.config(state="disabled", text="DEBATING...")

        threading.Thread(target=self._worker, args=(prompt, targets), daemon=True).start()

    def _worker(self, prompt, targets):
        try:
            responses = {}

            # 1. GATHER OPINIONS
            for lid in targets:
                if self.stop_requested: break

                lobe = self.app.lobe_manager.get_lobe(lid)
                if not lobe:
                    self.update_queue.put(lambda l=lid: self._write(l, "[Lobe Not Loaded]"))
                    continue

                self.update_queue.put(lambda l=lid: self._write(l, "Thinking..."))

                # Inference
                res = self._generate(lobe, prompt)
                responses[lid] = res

                self.update_queue.put(lambda l=lid, r=res: self._write(l, r))

            # 2. SYNTHESIS
            if self.synthesis_enabled.get() and not self.stop_requested:
                syn_id = self.synthesizer_id.get()
                leader = self.app.lobe_manager.get_lobe(syn_id)

                if leader:
                    self.update_queue.put(lambda: self._write(99, "Synthesizing Consensus..."))

                    # Construct Meta-Prompt
                    meta_prompt = f"Topic: {prompt}\n\n"
                    for lid, resp in responses.items():
                        meta_prompt += f"Expert {lid} says: {resp}\n"
                    meta_prompt += "\nSynthesize a final answer based on these opinions:"

                    final_res = self._generate(leader, meta_prompt)
                    self.update_queue.put(lambda: self._write(99, final_res))
                else:
                    self.update_queue.put(lambda: self._write(99, "[Synthesizer Lobe Not Loaded]"))

        except Exception as e:
            self.app.golgi.error(f"Council Error: {e}", source="Council")

        self.is_running = False
        self.update_queue.put(lambda: self.btn_run.config(state="normal", text="CONVENE COUNCIL"))

    def _generate(self, lobe, text):
        device = self.app.device

        # Tokenize
        input_ids = self.app.ribosome._tokenize(text)
        t_in = torch.tensor([input_ids], device=device).long()

        # Empty multimodal slots
        v_in = torch.zeros(1, 1, 768).to(device)
        a_in = torch.zeros(1, 1, 128).to(device)
        c_in = torch.zeros(1, 1, 32).to(device)  # Control vector (if supported)

        max_new = self.max_tokens.get()
        temp = self.temperature.get()

        generated = []

        with torch.no_grad():
            curr_t = t_in
            for _ in range(max_new):
                if self.stop_requested: break

                try:
                    logits, _, _ = lobe.model(v_in, a_in, curr_t, c_in)
                except:
                    logits, _, _ = lobe.model(v_in, a_in, curr_t)

                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits / temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()
                if token_id == 50256: break  # EOS

                generated.append(token_id)
                curr_t = torch.cat([curr_t, next_token], dim=1)

        return self.app.ribosome.decode(generated)

    def _write(self, lid, text):
        if lid in self.output_boxes:
            w = self.output_boxes[lid]
            w.delete("1.0", tk.END)
            w.insert("1.0", text)

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