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
import time
import random
import yaml


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Memory Agent"

        self.is_running = False
        self.stop_requested = False
        self.stats = {"scanned": 0, "cleaned": 0, "errors": 0}

        # Settings
        self.agent_lobe = tk.IntVar(value=1)
        self.auto_prune = tk.BooleanVar(value=False)
        self.consolidation_mode = tk.StringVar(value="Summarize")  # Summarize, Verify

        self._setup_ui()

    def _setup_ui(self):
        if self.parent is None: return
        scale = getattr(self.app, 'ui_scale', 1.0)

        # 1. CONFIG
        fr_conf = ttk.LabelFrame(self.parent, text="Gardener Configuration", padding=15)
        fr_conf.pack(fill="x", padx=10, pady=10)

        r1 = ttk.Frame(fr_conf)
        r1.pack(fill="x", pady=5)
        ttk.Label(r1, text="Agent Lobe (The Cleaner):").pack(side="left")
        for i in range(1, 5):
            ttk.Radiobutton(r1, text=f"Lobe {i}", variable=self.agent_lobe, value=i).pack(side="left", padx=5)

        r2 = ttk.Frame(fr_conf)
        r2.pack(fill="x", pady=5)
        ttk.Label(r2, text="Operation Mode:").pack(side="left")
        modes = ["Sanity Check (Read-Only)", "Summarize Verbose Nodes", "Prune Orphans"]
        ttk.Combobox(r2, textvariable=self.consolidation_mode, values=modes, state="readonly", width=25).pack(
            side="left", padx=5)
        ttk.Checkbutton(r2, text="Auto-Save Changes", variable=self.auto_prune).pack(side="left", padx=15)

        # 2. CONTROLS
        fr_run = ttk.Frame(self.parent, padding=10)
        fr_run.pack(fill="x")

        self.btn_run = ttk.Button(fr_run, text="START SLEEP CONSOLIDATION", command=self._toggle_run)
        self.btn_run.pack(fill="x", pady=5)

        self.lbl_status = ttk.Label(fr_run, text="Agent Sleeping.", foreground=self.app.colors["FG_DIM"],
                                    anchor="center")
        self.lbl_status.pack(fill="x")

        # 3. LOG
        fr_log = ttk.LabelFrame(self.parent, text="Maintenance Log", padding=10)
        fr_log.pack(fill="both", expand=True, padx=10, pady=5)

        log_font = ("Consolas", int(10 * scale))
        self.log_box = tk.Text(fr_log, font=log_font, height=10, bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"])
        self.log_box.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(fr_log, command=self.log_box.yview)
        self.log_box.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

    def _log(self, msg):
        self.log_box.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_box.see(tk.END)
        # Also send to Golgi
        if self.app.golgi: self.app.golgi.info(msg, source="Gardener")

    def _toggle_run(self):
        if self.is_running:
            self.stop_requested = True
            self.btn_run.config(text="STOPPING...")
        else:
            # Validate Lobe
            lid = self.agent_lobe.get()
            if not self.app.lobe_manager.get_lobe(lid):
                self._log(f"Error: Agent Lobe {lid} is not loaded.")
                return

            self.is_running = True
            self.stop_requested = False
            self.btn_run.config(text="WAKE AGENT (STOP)")
            self.lbl_status.config(text="Agent Active: Consolidating Memories...", foreground=self.app.colors["ACCENT"])

            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        mode = self.consolidation_mode.get()
        nodes = list(self.app.hippocampus.nodes.keys())
        random.shuffle(nodes)  # Random walk

        self._log(f"Started '{mode}' on {len(nodes)} nodes.")

        for name in nodes:
            if self.stop_requested: break

            node = self.app.hippocampus.nodes[name]

            # --- OPERATION 1: SANITY CHECK ---
            if "Sanity" in mode:
                # Check for broken relations
                rels = node.data.get("relations", [])
                clean_rels = []
                dirty = False

                for r in rels:
                    if "->" in r:
                        target = r.split("->")[-1].strip()
                        if target in self.app.hippocampus.nodes:
                            clean_rels.append(r)
                        else:
                            self._log(f"Found broken link in '{name}': -> {target}")
                            dirty = True
                    else:
                        clean_rels.append(r)

                if dirty and self.auto_prune.get():
                    node.data["relations"] = clean_rels
                    self._log(f"Fixed links in '{name}'.")

            # --- OPERATION 2: SUMMARIZE ---
            elif "Summarize" in mode:
                # Mock logic: If description is too long, we would use the Lobe to shorten it.
                # Since we don't have a standardized Lobe API for "instruct" yet, we stick to metadata.
                desc = node.data.get("core_summary", "")
                if len(desc) > 500:
                    self._log(f"Node '{name}' is verbose ({len(desc)} chars). Needs summarization.")
                    # Future: self.app.ribosome.generate(f"Summarize: {desc}")
                else:
                    pass  # Good size

            time.sleep(0.1)  # Simulate thought/processing time

        self.is_running = False
        if self.parent:
            self.parent.after(0, lambda: self.btn_run.config(text="START SLEEP CONSOLIDATION"))
            self.parent.after(0, lambda: self.lbl_status.config(text="Agent Sleeping.",
                                                                foreground=self.app.colors["FG_DIM"]))
        self._log("Cycle Complete.")

        if self.auto_prune.get():
            self.app.hippocampus.save_memory()

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])