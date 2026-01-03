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
import itertools
import time
import torch


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "The Council"

        self.is_running = False
        self.members = {1: tk.BooleanVar(value=True), 2: tk.BooleanVar(value=True),
                        3: tk.BooleanVar(value=True), 4: tk.BooleanVar(value=True)}
        self.topic = tk.StringVar()

        self._setup_ui()

    def _setup_ui(self):
        # Controls
        ctrl = ttk.LabelFrame(self.parent, text="Chamber Controls", padding=15)
        ctrl.pack(fill="x", padx=20, pady=10)

        ttk.Label(ctrl, text="Topic:").pack(side="left")
        ttk.Entry(ctrl, textvariable=self.topic).pack(side="left", fill="x", expand=True, padx=10)

        self.btn = ttk.Button(ctrl, text="CONVENE", command=self._toggle)
        self.btn.pack(side="right")

        # Members
        mem_fr = ttk.Frame(self.parent)
        mem_fr.pack(fill="x", padx=25)
        ttk.Label(mem_fr, text="Speakers:").pack(side="left")
        for i in range(1, 5):
            ttk.Checkbutton(mem_fr, text=f"Lobe {i}", variable=self.members[i]).pack(side="left", padx=5)

        # Chat
        self.chat = tk.Text(self.parent, font=("Consolas", 11), wrap="word",
                            bg=self.app.colors["BG_MAIN"], fg=self.app.colors["FG_TEXT"],
                            borderwidth=0, padx=20, pady=20)
        self.chat.pack(fill="both", expand=True, padx=20, pady=10)

    def _toggle(self):
        if self.is_running:
            self.is_running = False
            self.btn.config(text="CONVENE")
        else:
            self.is_running = True
            self.btn.config(text="ADJOURN")
            threading.Thread(target=self._run_debate, daemon=True).start()

    def _run_debate(self):
        topic = self.topic.get() or "The nature of consciousness"
        history = f"TOPIC: {topic}\n\n"
        self.parent.after(0, lambda: self.chat.insert(tk.END, history))

        active_ids = [i for i in range(1, 5) if self.members[i].get() and self.app.lobes[i]]
        if not active_ids:
            self.parent.after(0, lambda: self.chat.insert(tk.END, "[System] No Lobes online.\n"))
            self.is_running = False
            return

        cycle = itertools.cycle(active_ids)

        # V/A/C zeros
        v = torch.zeros(1, 1, 768).to(self.app.device)
        a = torch.zeros(1, 1, 128).to(self.app.device)
        c = torch.zeros(1, 1, 64).to(self.app.device)

        while self.is_running:
            lobe_id = next(cycle)
            brain = self.app.lobes[lobe_id]
            brain.eval()

            # Generate
            t = torch.tensor(self.app.ribosome.tokenizer.encode(history)).unsqueeze(0).to(self.app.device)
            # Clip history if too long
            if t.shape[1] > 900: t = t[:, -900:]

            response_toks = []
            with torch.no_grad():
                for _ in range(64):
                    logits, _, _ = brain(v, a, t, c)
                    token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                    t = torch.cat([t, token], dim=1)
                    response_toks.append(token.item())
                    if token.item() == 50256: break

            text = self.app.ribosome.tokenizer.decode(response_toks).strip()
            history += f"Lobe {lobe_id}: {text}\n\n"

            self.parent.after(0, lambda l=lobe_id, t=text: self._append_msg(l, t))
            time.sleep(1)

    def _append_msg(self, lobe_id, text):
        self.chat.insert(tk.END, f"LOBE {lobe_id}: {text}\n\n")
        self.chat.see(tk.END)

        def on_theme_change(self):
            # Update Text Widgets which don't use ttk styles
            c = self.app.colors
            if hasattr(self, 'log_box'):
                self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"], insertbackground=c["ACCENT"])
            if hasattr(self, 'chat_box'):
                self.chat_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"], insertbackground=c["ACCENT"])
            # Update Canvas backgrounds if needed
            if hasattr(self, 'canvas'):
                self.canvas.config(bg=c["BG_MAIN"])