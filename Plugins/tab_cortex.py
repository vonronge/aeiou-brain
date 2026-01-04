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
import shutil
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Cortex Control"

        self.genome_var = tk.StringVar(value="Choose Genetics")
        self.info_text = tk.StringVar(value="Select a genetic structure to initialize.")
        self.status_labels = {}

        self._setup_ui()
        self.parent.bind("<Visibility>", lambda e: self._refresh_ui())
        self._scan_genetics()
        self._refresh_ui()

    def _scan_genetics(self):
        """
        Asks the Lobe Manager to refresh its registry and returns the list
        of human-readable names.
        """
        self.app.lobe_manager.refresh_registry()
        names = self.app.lobe_manager.list_available_genetics()

        if hasattr(self, 'combo'):
            self.combo['values'] = names

    def _on_select(self, event):
        """
        Uses Lobe Manager to resolve the display name to the actual module
        to fetch metadata (Description, VRAM est).
        """
        name = self.genome_var.get()
        try:
            # We access the internal import method to ensure we use the
            # exact same file resolution logic as the Manager.
            module = self.app.lobe_manager._import_genetics(name)

            if hasattr(module, "INFO"):
                info = module.INFO
                self.info_text.set(f"{info.get('desc', '')}\nEst. VRAM: {info.get('vram_train', '?')}")
            else:
                self.info_text.set("No metadata available for this module.")
        except Exception as e:
            self.info_text.set(f"Could not read genetics info: {e}")

    def _setup_ui(self):
        scale = getattr(self.app, 'ui_scale', 1.0)
        font_size = int(10 * scale)
        font_norm = ("Segoe UI", font_size)
        font_bold = ("Segoe UI", font_size, "bold")

        # 1. STATUS PANEL
        frame_status = ttk.LabelFrame(self.parent, text="Cortex Status", padding=15)
        frame_status.pack(fill="x", padx=20, pady=10)

        for i in range(1, 5):
            row = ttk.Frame(frame_status)
            row.pack(fill="x", pady=2)
            lbl = ttk.Label(row, text=f"Lobe {i}: Checking...", font=font_norm)
            lbl.pack(side="left")
            self.status_labels[i] = lbl

        # 2. GENETICS PANEL
        frame_mgmt = ttk.LabelFrame(self.parent, text="Genetic Engineering", padding=15)
        frame_mgmt.pack(fill="both", expand=True, padx=20, pady=10)

        ttk.Label(frame_mgmt, text="Target Genetics:").pack(anchor="w")
        self.combo = ttk.Combobox(frame_mgmt, textvariable=self.genome_var, values=[], state="readonly")
        self.combo.pack(fill="x", pady=5)
        self.combo.bind("<<ComboboxSelected>>", self._on_select)

        info_lbl = ttk.Label(frame_mgmt, textvariable=self.info_text, foreground=self.app.colors["ACCENT"],
                             justify="left")
        info_lbl.pack(pady=10, anchor="w")

        # 3. CONTROLS
        frame_pwr = ttk.LabelFrame(frame_mgmt, text="Power & Memory", padding=10)
        frame_pwr.pack(fill="x", pady=5)

        ttk.Button(frame_pwr, text="⚡ ACTIVATE LOBE", command=self._activate_brain).pack(side="left", fill="x",
                                                                                         expand=True, padx=5)
        ttk.Button(frame_pwr, text="💤 DEACTIVATE", command=self._deactivate_brain).pack(side="left", fill="x",
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

    def _refresh_ui(self):
        scale = getattr(self.app, 'ui_scale', 1.0)
        font_norm = ("Segoe UI", int(10 * scale))
        font_bold = ("Segoe UI", int(10 * scale), "bold")

        for i in range(1, 5):
            handle = self.app.lobe_manager.get_lobe(i)
            lbl = self.status_labels[i]

            is_active_slot = (self.app.active_lobe.get() == i)
            prefix = "➤ " if is_active_slot else "   "
            f_style = font_bold if is_active_slot else font_norm

            if handle:
                opt_str = "Muon" if "Muon" in str(handle.optimizer) else "AdamW"
                lbl.config(text=f"{prefix}Lobe {i}: ONLINE ({handle.genome} | {handle.model_type}) [{opt_str}]",
                           foreground=self.app.colors["SUCCESS"], font=f_style)
            else:
                path = os.path.join(self.app.paths['lobes'], f"brain_lobe_{i}.pt")
                if os.path.exists(path):
                    lbl.config(text=f"{prefix}Lobe {i}: OFFLINE (Ready)",
                               foreground=self.app.colors["WARN"], font=font_norm)
                else:
                    lbl.config(text=f"{prefix}Lobe {i}: EMPTY",
                               foreground=self.app.colors["FG_DIM"], font=font_norm)

    def _activate_brain(self):
        active_id = self.app.active_lobe.get()
        if self.app.lobe_manager.get_lobe(active_id):
            messagebox.showinfo("Info", f"Lobe {active_id} is already active.")
            return

        try:
            self.app.lobe_manager.load_lobe(active_id)
            self._refresh_ui()
            self.app.refresh_header()
            self.app.golgi.success(f"Lobe {active_id} Activated.", source="Cortex")
        except FileNotFoundError:
            if messagebox.askyesno("Missing", f"Lobe {active_id} file not found.\nInitialize new?"):
                self._init_brain()
        except Exception as e:
            self.app.golgi.error(f"Activation Failed: {e}", source="Cortex")
            messagebox.showerror("Error", str(e))

    def _deactivate_brain(self):
        active_id = self.app.active_lobe.get()
        self.app.lobe_manager.unload_lobe(active_id)
        self._refresh_ui()
        self.app.refresh_header()
        self.app.golgi.info(f"Lobe {active_id} Deactivated.", source="Cortex")

    def _init_brain(self):
        active_id = self.app.active_lobe.get()
        genome = self.genome_var.get()
        if not genome or genome == "Choose Genetics":
            messagebox.showerror("Error", "Please select a valid genetics module.")
            return

        if self.app.lobe_manager.get_lobe(active_id):
            if not messagebox.askyesno("Overwrite",
                                       f"Lobe {active_id} is active in RAM.\nOverwrite with fresh weights?"):
                return

        try:
            self.app.lobe_manager.create_lobe(active_id, genome)
            self._refresh_ui()
            self.app.refresh_header()
            self.app.golgi.success(f"Lobe {active_id} Initialized ({genome}).", source="Cortex")
        except Exception as e:
            self.app.golgi.error(f"Genesis Failed: {e}", source="Cortex")
            messagebox.showerror("Error", str(e))

    def _save_brain(self):
        active_id = self.app.active_lobe.get()
        if not self.app.lobe_manager.get_lobe(active_id):
            messagebox.showerror("Error", "Lobe not loaded.")
            return

        f = filedialog.asksaveasfilename(initialdir=self.app.paths['lobes'], defaultextension=".pt")
        if f:
            try:
                self.app.lobe_manager.save_lobe(active_id, f)
                self.app.golgi.save(f"Lobe {active_id} saved to {os.path.basename(f)}", source="Cortex")
            except Exception as e:
                messagebox.showerror("Save Failed", str(e))

    def _load_brain(self):
        active_id = self.app.active_lobe.get()
        f = filedialog.askopenfilename(initialdir=self.app.paths['lobes'], filetypes=[("Brain Files", "*.pt")])
        if f:
            try:
                if messagebox.askyesno("Import", f"Import this file into Lobe Slot {active_id}?"):
                    target = os.path.join(self.app.paths['lobes'], f"brain_lobe_{active_id}.pt")
                    shutil.copy2(f, target)
                    self.app.lobe_manager.load_lobe(active_id)
                    self._refresh_ui()
                    self.app.refresh_header()
                    self.app.golgi.success(f"Imported {os.path.basename(f)} to Lobe {active_id}", source="Cortex")
            except Exception as e:
                messagebox.showerror("Import Failed", str(e))

    def _backup_brain(self):
        active_id = self.app.active_lobe.get()
        path = os.path.join(self.app.paths['lobes'], f"brain_lobe_{active_id}.pt")
        if not os.path.exists(path): return

        backup_dir = os.path.join(self.app.paths['lobes'], "backups")
        if not os.path.exists(backup_dir): os.makedirs(backup_dir)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = os.path.join(backup_dir, f"lobe_{active_id}_{ts}.pt")

        try:
            shutil.copy2(path, bak)
            messagebox.showinfo("Backup", f"Backup created:\n{os.path.basename(bak)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_theme_change(self):
        self._refresh_ui()