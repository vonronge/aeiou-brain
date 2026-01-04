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
import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
import importlib.util
import gc
import traceback
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Cortex Control"

        self.available_genetics = {}
        self.genome_var = tk.StringVar(value="Choose Genetics")
        self.info_text = tk.StringVar(value="Select a genetic structure to initialize.")
        self.status_labels = {}

        self._setup_ui()
        self.parent.bind("<Visibility>", lambda e: self._refresh_ui())
        self._scan_genetics()
        self._refresh_ui()

    def _scan_genetics(self):
        self.available_genetics = {}
        g_dir = self.app.paths['genetics']
        if not os.path.exists(g_dir): return

        files = [f for f in os.listdir(g_dir) if f.endswith(".py") and not f.startswith("__")]
        print(f"[Cortex] Scanning genetics in {g_dir}...")

        for f in files:
            try:
                path = os.path.join(g_dir, f)
                spec = importlib.util.spec_from_file_location("dynamic_dna", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "INFO"):
                    name = module.INFO.get("name", f)
                    self.available_genetics[name] = module
                    print(f"   > Loaded: {name}")
                else:
                    print(f"   ! Skipped {f}: No INFO dict found.")
            except Exception as e:
                print(f"   ! FAILED to load {f}: {e}")

        if hasattr(self, 'combo'):
            self.combo['values'] = list(self.available_genetics.keys())

    def _setup_ui(self):
        # SCALE CALCULATION
        scale = getattr(self.app, 'ui_scale', 1.0)
        font_size = int(10 * scale)

        frame_status = ttk.LabelFrame(self.parent, text="Cortex Status", padding=15)
        frame_status.pack(fill="x", padx=20, pady=10)

        for i in range(1, 5):
            row = ttk.Frame(frame_status)
            row.pack(fill="x", pady=2)
            # FIX: Dynamic Font Size
            lbl = ttk.Label(row, text=f"Lobe {i}: Checking...", font=("Segoe UI", font_size))
            lbl.pack(side="left")
            self.status_labels[i] = lbl

        frame_mgmt = ttk.LabelFrame(self.parent, text="Genetic Engineering", padding=15)
        frame_mgmt.pack(fill="both", expand=True, padx=20, pady=10)

        ttk.Label(frame_mgmt, text="Target Genetics:").pack(anchor="w")
        self.combo = ttk.Combobox(frame_mgmt, textvariable=self.genome_var, values=list(self.available_genetics.keys()),
                                  state="readonly")
        self.combo.pack(fill="x", pady=5)
        self.combo.bind("<<ComboboxSelected>>", self._on_select)

        info_lbl = ttk.Label(frame_mgmt, textvariable=self.info_text, foreground=self.app.colors["ACCENT"],
                             justify="left")
        info_lbl.pack(pady=10, anchor="w")

        frame_pwr = ttk.LabelFrame(frame_mgmt, text="Power & Memory", padding=10)
        frame_pwr.pack(fill="x", pady=5)
        ttk.Button(frame_pwr, text="⚡ ACTIVATE LOBE", command=self._activate_brain).pack(side="left", fill="x",
                                                                                         expand=True, padx=5)
        ttk.Button(frame_pwr, text="💤 DEACTIVATE (Unload)", command=self._deactivate_brain).pack(side="left", fill="x",
                                                                                                 expand=True, padx=5)

        frame_store = ttk.LabelFrame(frame_mgmt, text="Storage Operations", padding=10)
        frame_store.pack(fill="x", pady=5)
        ttk.Button(frame_store, text="INITIALIZE NEW", command=self._init_brain).pack(side="left", fill="x",
                                                                                      expand=True, padx=5)
        ttk.Button(frame_store, text="SAVE AS...", command=self._save_brain).pack(side="left", fill="x", expand=True,
                                                                                  padx=5)
        ttk.Button(frame_store, text="🛡️ BACKUP", command=self._backup_brain).pack(side="left", fill="x", expand=True,
                                                                                   padx=5)
        ttk.Button(frame_store, text="LOAD FILE", command=self._load_brain).pack(side="left", fill="x", expand=True,
                                                                                 padx=5)

        ttk.Button(frame_mgmt, text="🔄 Rescan Genetics Folder", command=self._scan_genetics).pack(fill="x", pady=10)

    def _on_select(self, event):
        name = self.genome_var.get()
        if name in self.available_genetics:
            info = self.available_genetics[name].INFO
            self.info_text.set(f"{info.get('desc', '')}\nEst. VRAM: {info.get('vram_train', '?')}")

    def _refresh_ui(self):
        # FIX: Re-calculate scale for updates
        scale = getattr(self.app, 'ui_scale', 1.0)
        font_norm = ("Segoe UI", int(10 * scale))
        font_bold = ("Segoe UI", int(10 * scale), "bold")

        for i in range(1, 5):
            brain = self.app.lobes[i]
            lbl = self.status_labels[i]
            if brain:
                g_name = self.app.lobe_genomes.get(i, "Unknown")
                m_type = self.app.lobe_types.get(i, "Unknown")
                is_active = (self.app.active_lobe.get() == i)
                prefix = "➤ " if is_active else "   "
                color = self.app.colors["SUCCESS"] if is_active else self.app.colors["FG_TEXT"]
                f_style = font_bold if is_active else font_norm

                opt_name = "AdamW"
                if hasattr(self.app.optimizers[i], 'adamw'): opt_name = "Muon"
                lbl.config(text=f"{prefix}Lobe {i}: ONLINE ({g_name} | {m_type}) [{opt_name}]", foreground=color,
                           font=f_style)
            else:
                path = os.path.join(self.app.paths['lobes'], f"brain_lobe_{i}.pt")
                prefix = "➤ " if (self.app.active_lobe.get() == i) else "   "
                if os.path.exists(path):
                    try:
                        meta = torch.load(path, map_location="cpu")
                        g_name = meta.get("genome", "GPT2") if isinstance(meta, dict) else "GPT2"
                        lbl.config(text=f"{prefix}Lobe {i}: OFFLINE (Ready: {g_name})",
                                   foreground=self.app.colors["WARN"], font=font_norm)
                    except:
                        lbl.config(text=f"{prefix}Lobe {i}: OFFLINE (Corrupt File)",
                                   foreground=self.app.colors["ERROR"], font=font_norm)
                else:
                    lbl.config(text=f"{prefix}Lobe {i}: EMPTY", foreground=self.app.colors["FG_DIM"],
                               font=font_norm)

    def on_theme_change(self):
        self._refresh_ui()

    def _deactivate_brain(self):
        active_id = self.app.active_lobe.get()
        if self.app.lobes[active_id] is None: return
        self.app.lobes[active_id] = None
        self.app.optimizers[active_id] = None
        self.app.scalers[active_id] = None
        gc.collect();
        torch.cuda.empty_cache()
        self._refresh_ui()
        self.app.save_state()
        self.app.refresh_header()

    def _activate_brain(self):
        active_id = self.app.active_lobe.get()
        if self.app.lobes[active_id] is not None:
            messagebox.showinfo("Info", f"Lobe {active_id} is already active.")
            return
        path = os.path.join(self.app.paths['lobes'], f"brain_lobe_{active_id}.pt")
        if os.path.exists(path):
            self.app._load_single_lobe(active_id, path)
            self._refresh_ui()
            self.app.save_state()
        else:
            if messagebox.askyesno("Activate", f"No file for Lobe {active_id}.\nInitialize new?"):
                self._init_brain()

    def _backup_brain(self):
        active_id = self.app.active_lobe.get()
        path = os.path.join(self.app.paths['lobes'], f"brain_lobe_{active_id}.pt")
        if not os.path.exists(path): return
        backup_dir = os.path.join(self.app.paths['lobes'], "backups")
        if not os.path.exists(backup_dir): os.makedirs(backup_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = os.path.join(backup_dir, f"lobe_{active_id}_{ts}.pt")
        import shutil
        shutil.copy2(path, bak)
        messagebox.showinfo("Backup", f"Saved to backups folder.")

    def _init_brain(self):
        active_id = self.app.active_lobe.get()
        genome = self.genome_var.get()
        if genome not in self.available_genetics:
            messagebox.showerror("Error", "Invalid Genetics")
            return
        if self.app.lobes[active_id] and not messagebox.askyesno("Overwrite", "Lobe active. Overwrite?"): return

        try:
            module = self.available_genetics[genome]
            config = module.NucleusConfig()
            brain = module.Model(config).to(self.app.device)
            self.app.lobes[active_id] = brain
            self.app.lobe_genomes[active_id] = genome

            # --- MODEL TYPE INFERENCE ---
            model_type = "diffusion" if "diffusion" in genome.lower() else "ar"
            self.app.lobe_types[active_id] = model_type

            if "Muon" in genome:
                from Genetics.muon import Muon
                self.app.optimizers[active_id] = Muon(brain.parameters(), lr=0.0005, momentum=0.95)
                print(f"[SYS] Initialized with Muon Optimizer (Safe Mode LR=0.0005)")
            else:
                self.app.optimizers[active_id] = optim.AdamW(brain.parameters(), lr=2e-5)

            self.app.scalers[active_id] = GradScaler()

            if hasattr(brain, "tokenizer"):
                self.app.ribosome.set_tokenizer(brain.tokenizer)
            else:
                import tiktoken
                self.app.ribosome.set_tokenizer(tiktoken.get_encoding("gpt2"))

            path = os.path.join(self.app.paths['lobes'], f"brain_lobe_{active_id}.pt")
            torch.save({
                "genome": genome,
                "model_type": model_type,  # Save type
                "state_dict": brain.state_dict()
            }, path)

            self._refresh_ui()
            self.app.refresh_header()
            self.app.save_state()
        except Exception as e:
            messagebox.showerror("Init Failed", str(e))
            print(e)

    def _save_brain(self):
        active_id = self.app.active_lobe.get()
        if not self.app.lobes[active_id]: return
        f = filedialog.asksaveasfilename(initialdir=self.app.paths['lobes'], defaultextension=".pt")
        if f:
            torch.save({
                "genome": self.app.lobe_genomes[active_id],
                "model_type": self.app.lobe_types.get(active_id, "ar"),
                "state_dict": self.app.lobes[active_id].state_dict()
            }, f)

    def _load_brain(self):
        f = filedialog.askopenfilename(initialdir=self.app.paths['lobes'], filetypes=[("Brain Files", "*.pt")])
        if f:
            self.app._load_single_lobe(self.app.active_lobe.get(), f)
            self._refresh_ui()
            self.app.save_state()