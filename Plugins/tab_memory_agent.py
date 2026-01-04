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
import time
import os
import torch
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
        self.name = "Memory Agent"
        self.is_running = False
        self.stop_requested = False

        # --- STATE ---
        if self.parent is None:
            # Headless Defaults
            self.interval_min = MockVar(15)  # Run every 15 mins
            self.consolidation_depth = MockVar(5)  # Number of items to merge
            self.auto_prune = MockVar(True)
            self.status_var = MockVar("Idle.")
        else:
            # GUI Defaults
            self.interval_min = tk.IntVar(value=15)
            self.consolidation_depth = tk.IntVar(value=5)
            self.auto_prune = tk.BooleanVar(value=True)
            self.status_var = tk.StringVar(value="Idle.")

        self._setup_ui()

    def _setup_ui(self):
        if self.parent is None: return

        # Layout
        panel = ttk.Frame(self.parent, padding=10)
        panel.pack(fill="both", expand=True)

        # 1. Controls
        fr_ctrl = ttk.LabelFrame(panel, text="Agent Configuration", padding=10)
        fr_ctrl.pack(fill="x", pady=5)

        ttk.Label(fr_ctrl, text="Wake Interval (mins):").pack(side="left")
        ttk.Spinbox(fr_ctrl, from_=1, to=1440, textvariable=self.interval_min, width=5).pack(side="left", padx=5)

        ttk.Label(fr_ctrl, text="Batch Depth:").pack(side="left", padx=(10, 0))
        ttk.Spinbox(fr_ctrl, from_=1, to=50, textvariable=self.consolidation_depth, width=5).pack(side="left", padx=5)

        ttk.Checkbutton(fr_ctrl, text="Auto-Prune Weak Links", variable=self.auto_prune).pack(side="left", padx=15)

        self.btn_toggle = ttk.Button(fr_ctrl, text="START AGENT", command=self._toggle_agent)
        self.btn_toggle.pack(side="right")

        # 2. Activity Log
        fr_log = ttk.LabelFrame(panel, text="Hippocampal Activity", padding=10)
        fr_log.pack(fill="both", expand=True, pady=5)

        self.log_box = tk.Text(fr_log, font=("Consolas", int(10 * getattr(self.app, 'ui_scale', 1.0))), bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"], state="disabled")
        self.log_box.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(fr_log, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # Tags
        self.log_box.tag_config("info", foreground=self.app.colors["FG_DIM"])
        self.log_box.tag_config("action", foreground=self.app.colors["ACCENT"])
        self.log_box.tag_config("error", foreground=self.app.colors["ERROR"])

        # Status Bar
        ttk.Label(panel, textvariable=self.status_var, foreground="#888").pack(anchor="w")

    def _log(self, msg, tag="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        text = f"[{ts}] {msg}"

        if self.parent:
            self.log_box.config(state="normal")
            self.log_box.insert(tk.END, text + "\n", tag)
            self.log_box.see(tk.END)
            self.log_box.config(state="disabled")
            self.status_var.set(msg)
        else:
            print(f"[Memory Agent] {msg}")

    def _toggle_agent(self):
        if self.is_running:
            self.stop_requested = True
            if self.parent: self.btn_toggle.config(text="Stopping...")
        else:
            if not self.app.hippocampus:
                self._log("Hippocampus not loaded. Agent aborting.", "error")
                return

            self.is_running = True
            self.stop_requested = False
            if self.parent: self.btn_toggle.config(text="STOP AGENT")
            threading.Thread(target=self._agent_loop, daemon=True).start()

    def _agent_loop(self):
        self._log("Agent started. Monitoring short-term memory...", "action")

        while not self.stop_requested:
            try:
                # 1. Sleep Phase (Wait interval minutes)
                # Check stop_requested every second to stay responsive
                interval_sec = self.interval_min.get() * 60
                for _ in range(int(interval_sec)):
                    if self.stop_requested: break
                    time.sleep(1)

                if self.stop_requested: break

                # 2. Consolidation Phase
                self._log("Waking up for consolidation cycle...", "action")
                self._perform_consolidation()

            except Exception as e:
                self._log(f"Cycle Error: {e}", "error")
                traceback.print_exc()
                time.sleep(60)  # Wait a bit before retry on error

        self.is_running = False
        if self.parent: self.btn_toggle.config(text="START AGENT")
        self._log("Agent stopped.", "info")

    def _perform_consolidation(self):
        """
        Reads recent/raw memories from Hippocampus, summarizes them using the active Lobe,
        and re-saves them as 'consolidated' nodes.
        """
        if not self.app.hippocampus: return

        # 1. Get Active Brain for summarization
        lobe_id = self.app.active_lobe.get()
        brain = self.app.lobes[lobe_id]

        if not brain:
            self._log("No Lobe loaded. Skipping cognitive tasks.", "error")
            return

        # 2. Fetch 'Raw' Memories
        # Simulating fetching recent additions. In a full graph implementation,
        # this would query nodes created > X minutes ago but not yet tagged 'consolidated'.
        memories = self.app.hippocampus.search("recent", limit=self.consolidation_depth.get())

        if not memories:
            self._log("No new memories to consolidate.")
            return

        # 3. Summarize / Reflect
        context_str = "\n".join([m.get('content', '') for m in memories])
        prompt = f"Consolidate these recent events into a single concise insight:\n\n{context_str}\n\nInsight:"

        self._log(f"Consolidating {len(memories)} items...", "action")

        try:
            insight = self._generate_thought(brain, prompt)

            # 4. Save back to Hippocampus
            if insight:
                # Add the new consolidated node
                self.app.hippocampus.add(insight, tags=["consolidated", "agent_generated"])
                self._log(f"Stored insight: {insight[:50]}...", "action")

                # Optional: Mark old ones as processed or prune
                if self.auto_prune.get():
                    self._log(f"Processed {len(memories)} raw inputs.", "info")
                    # Actual graph pruning logic would go here

            # 5. Save Graph to Disk
            self.app.hippocampus.save_memory()

        except Exception as e:
            self._log(f"Cognitive Failure: {e}", "error")

    def _generate_thought(self, brain, prompt):
        """Helper to run inference on the active lobe"""
        ribosome = self.app.ribosome
        ids = ribosome._tokenize(prompt)
        t = torch.tensor(ids, device=self.app.device).unsqueeze(0)

        # Placeholder sensors for multimodal models
        v = torch.zeros(1, 1, 768).to(self.app.device)
        a = torch.zeros(1, 1, 128).to(self.app.device)

        out_text = ""
        with self.app.gpu_lock:
            brain.eval()
            with torch.no_grad():
                # Simple AR generation loop (limit 100 tokens for brevity)
                for _ in range(100):
                    # Handle both AR and Diffusion (though AR is preferred for text summarization)
                    if hasattr(brain, 'timestep_emb'):
                        # Diffusion models aren't great at pure text summarization
                        # without specific training, but we try anyway:
                        return "[Diffusion model cannot summarize text reliably yet]"

                    logits, _, _ = brain(v, a, t)
                    next_tok = torch.argmax(logits[:, -1, :], dim=-1)
                    t = torch.cat([t, next_tok.unsqueeze(0)], dim=1)

                    word = ribosome.decode([next_tok.item()])
                    out_text += word
                    if next_tok.item() == 50256: break

        return out_text.strip()

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'):
            self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])