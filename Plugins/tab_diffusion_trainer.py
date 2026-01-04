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
from tkinter import ttk, filedialog, messagebox
import threading
import os
import time
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from datetime import datetime
import random
import queue
import math
import numpy as np
import json
from collections import deque
import traceback

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print(" ! Diffusion Trainer: Install 'pymupdf' for PDF support.")

# --- HEADLESS HELPER ---
class MockVar:
    def __init__(self, value=None): self._val = value
    def set(self, value): self._val = value
    def get(self): return self._val


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Diffusion Director"
        self.is_training = False
        self.is_paused = False
        self.stop_requested = False

        # PIPELINE
        self.data_queue = queue.Queue(maxsize=50)
        self.task_queue = queue.Queue(maxsize=100)
        self.ram_cache = {}
        self.MAX_CACHE_SIZE = 25000

        # STATE
        self.training_queue = []
        self.all_scanned_packets = []
        self.train_ext_vars = {}
        self.train_type_vars = {}
        self.history_file = os.path.join(self.app.paths["root"], "diffusion_history.json")
        self.recent_folders = self._load_json(self.history_file, [])
        default_folder = self.recent_folders[0] if self.recent_folders else "D:/Training_Data"
        
        # Telepathy/Config override
        if hasattr(self.app.paths, "data") and os.path.exists(self.app.paths["data"]):
             default_folder = self.app.paths["data"]

        self.processed_count = 0
        self.total_items = 0
        self.loss_history = {'recon': deque(maxlen=10000), 'game': deque(maxlen=10000)}

        # --- INITIALIZE VARIABLES (GUI vs HEADLESS) ---
        if self.parent is None:
            # Headless Mode
            self.folder_path = MockVar(default_folder)
            self.use_uniform = MockVar(True)
            self.uniform_ratio_min = MockVar(0.15)
            self.uniform_ratio_max = MockVar(0.60)
            self.use_cross_modal = MockVar(False)
            self.cross_modal_prob = MockVar(0.25)
            self.use_modality_specific = MockVar(False)
            self.nursery_autofit = MockVar(True)
            self.nurse_recon = [MockVar(0.1), MockVar(8.0), MockVar(True)]
            self.nurse_game = [MockVar(0.001), MockVar(10.0), MockVar(True)]
            self.auto_scroll = MockVar(False)
            self.autosave_enabled = MockVar(True)
            self.autosave_interval = MockVar(100)
            self.target_epochs = MockVar(1)
            self.narrative_mode = MockVar(True)
            self.num_workers = MockVar(4)
        else:
            # GUI Mode
            self.folder_path = tk.StringVar(value=default_folder)
            self.use_uniform = tk.BooleanVar(value=True)
            self.uniform_ratio_min = tk.DoubleVar(value=0.15)
            self.uniform_ratio_max = tk.DoubleVar(value=0.60)
            self.use_cross_modal = tk.BooleanVar(value=False)
            self.cross_modal_prob = tk.DoubleVar(value=0.25)
            self.use_modality_specific = tk.BooleanVar(value=False)
            self.nursery_autofit = tk.BooleanVar(value=True)
            self.nurse_recon = [tk.DoubleVar(value=0.1), tk.DoubleVar(value=8.0), tk.BooleanVar(value=True)]
            self.nurse_game = [tk.DoubleVar(value=0.001), tk.DoubleVar(value=10.0), tk.BooleanVar(value=True)]
            self.auto_scroll = tk.BooleanVar(value=True)
            self.autosave_enabled = tk.BooleanVar(value=True)
            self.autosave_interval = tk.IntVar(value=100)
            self.target_epochs = tk.IntVar(value=1)
            self.narrative_mode = tk.BooleanVar(value=True)
            self.num_workers = tk.IntVar(value=4)

        self._setup_ui()

    def _setup_ui(self):
        if self.parent is None: return

        style = ttk.Style()
        style.map('TCombobox', fieldbackground=[('readonly', self.app.colors['BG_CARD'])],
                  selectbackground=[('readonly', self.app.colors['BG_CARD'])],
                  selectforeground=[('readonly', self.app.colors['FG_TEXT'])],
                  foreground=[('readonly', self.app.colors['FG_TEXT'])])

        split = ttk.PanedWindow(self.parent, orient="horizontal")
        split.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(split)
        right = ttk.Frame(split)
        split.add(left, weight=3)
        split.add(right, weight=1)

        # 1. DATA SOURCE
        fr_src = ttk.LabelFrame(left, text="Data Feed", padding=10)
        fr_src.pack(fill="x", pady=5)

        row_src = ttk.Frame(fr_src)
        row_src.pack(fill="x", expand=True)
        self.cmb_folder = ttk.Combobox(row_src, textvariable=self.folder_path, values=self.recent_folders)
        self.cmb_folder.pack(side="left", fill="x", expand=True)
        self.cmb_folder.bind("<<ComboboxSelected>>", lambda e: self._scan_files())
        ttk.Button(row_src, text="📂", width=3, command=self._browse_folder).pack(side="left", padx=2)

        row_btns = ttk.Frame(fr_src)
        row_btns.pack(fill="x", pady=5)
        ttk.Button(row_btns, text="SCAN FOLDER", command=self._scan_files).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(row_btns, text="CLEAR QUEUE", command=self._clear_queue).pack(side="left", fill="x", expand=True, padx=2)

        # 2. MASKING
        fr_mask = ttk.LabelFrame(left, text="Masking Curriculum", padding=10)
        fr_mask.pack(fill="x", pady=5)

        f1 = ttk.Frame(fr_mask)
        f1.pack(fill="x", pady=2)
        ttk.Checkbutton(f1, text="Phase 1: Uniform", variable=self.use_uniform).pack(side="left")
        ttk.Label(f1, text="Ratio:").pack(side="left", padx=5)
        ttk.Entry(f1, textvariable=self.uniform_ratio_min, width=5).pack(side="left")
        ttk.Label(f1, text="-").pack(side="left")
        ttk.Entry(f1, textvariable=self.uniform_ratio_max, width=5).pack(side="left")

        f2 = ttk.Frame(fr_mask)
        f2.pack(fill="x", pady=2)
        ttk.Checkbutton(f2, text="Phase 2: Cross-Modal", variable=self.use_cross_modal).pack(side="left")
        ttk.Label(f2, text="Prob:").pack(side="left", padx=5)
        ttk.Scale(f2, from_=0.0, to=1.0, variable=self.cross_modal_prob, length=100).pack(side="left")

        f3 = ttk.Frame(fr_mask)
        f3.pack(fill="x", pady=2)
        ttk.Checkbutton(f3, text="Phase 3: Fine-Grained", variable=self.use_modality_specific).pack(side="left")

        # 3. NURSERY
        fr_nurse = ttk.LabelFrame(left, text="Nursery", padding=10)
        fr_nurse.pack(fill="x", pady=5)
        ttk.Checkbutton(fr_nurse, text="Auto-Fit", variable=self.nursery_autofit).pack(anchor="w")

        grid = ttk.Frame(fr_nurse)
        grid.pack(fill="x", pady=5)
        ttk.Label(grid, text="Channel", font=("Segoe UI", 8, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(grid, text="Active", font=("Segoe UI", 8)).grid(row=0, column=1)
        ttk.Label(grid, text="Min", font=("Segoe UI", 8)).grid(row=0, column=2)
        ttk.Label(grid, text="Max", font=("Segoe UI", 8)).grid(row=0, column=3)

        def add_row(r, label, vars):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="e", padx=5)
            ttk.Checkbutton(grid, variable=vars[2]).grid(row=r, column=1)
            ttk.Entry(grid, textvariable=vars[0], width=8).grid(row=r, column=2, padx=2)
            ttk.Entry(grid, textvariable=vars[1], width=8).grid(row=r, column=3, padx=2)

        add_row(1, "Reconstruction:", self.nurse_recon)
        add_row(2, "Game Penalty:", self.nurse_game)

        # 4. OPERATIONS
        fr_ctrl = ttk.LabelFrame(left, text="Operations", padding=10)
        fr_ctrl.pack(fill="x", pady=5)

        row_sets = ttk.Frame(fr_ctrl)
        row_sets.pack(fill="x", pady=5)
        ttk.Label(row_sets, text="Epochs:").pack(side="left")
        ttk.Spinbox(row_sets, from_=1, to=999, textvariable=self.target_epochs, width=5).pack(side="left", padx=5)
        ttk.Label(row_sets, text="CPU Workers:").pack(side="left", padx=(10, 0))
        ttk.Scale(row_sets, from_=1, to=16, variable=self.num_workers, orient="horizontal", length=100).pack(
            side="left", padx=5)
        ttk.Checkbutton(row_sets, text="Narrative Mode", variable=self.narrative_mode).pack(side="left", padx=10)

        row_btn = ttk.Frame(fr_ctrl)
        row_btn.pack(fill="x")
        self.btn_start = ttk.Button(row_btn, text="START TRAINING", command=self._start_training)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=2)
        self.btn_pause = ttk.Button(row_btn, text="PAUSE", command=self._toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", fill="x", expand=True, padx=2)

        # 5. LOGS
        fr_log = ttk.LabelFrame(left, text="Logs", padding=10)
        fr_log.pack(fill="both", expand=True, pady=5)
        self.log_box = tk.Text(fr_log, font=("Consolas", 9), bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"])
        self.log_box.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(fr_log, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self.log_box.tag_config("info", foreground=self.app.colors["ACCENT"])
        self.log_box.tag_config("prog", foreground=self.app.colors["FG_DIM"])
        self.log_box.tag_config("save", foreground=self.app.colors["SUCCESS"], font=("Consolas", 9, "bold"))
        self.log_box.tag_config("warn", foreground=self.app.colors["WARN"])
        self.log_box.tag_config("error", foreground=self.app.colors["ERROR"])

        # 6. CENSUS
        fr_census = ttk.LabelFrame(right, text="Census (Filter)", padding=10)
        fr_census.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(fr_census, width=200, bg=self.app.colors["BG_MAIN"], highlightthickness=0)
        scr = ttk.Scrollbar(fr_census, orient="vertical", command=self.canvas.yview)
        self.scroll_fr = ttk.Frame(self.canvas, style="Card.TFrame")
        self.scroll_fr.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_fr, anchor="nw")
        self.canvas.configure(yscrollcommand=scr.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scr.pack(side="right", fill="y")

    def _load_json(self, path, default):
        if os.path.exists(path):
            try:
                return json.load(open(path, 'r'))
            except:
                pass
        return default

    def _save_history(self):
        curr = self.folder_path.get()
        if curr in self.recent_folders: self.recent_folders.remove(curr)
        self.recent_folders.insert(0, curr)
        try:
            json.dump(self.recent_folders[:10], open(self.history_file, 'w'))
        except:
            pass

    def _log_threadsafe(self, msg, tag="info"):
        if self.parent is None:
            # Headless logging
            print(f"[{tag.upper()}] {msg}")
        else:
            # GUI logging with LAG PREVENTION
            def _update():
                if not hasattr(self, 'log_box'): return
                
                # --- OPTIMIZATION: PRUNE LOGS ---
                try:
                    num_lines = int(self.log_box.index('end-1c').split('.')[0])
                    if num_lines > 1000:
                        self.log_box.delete("1.0", "100.0")
                except: pass

                self.log_box.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n", tag)
                if self.auto_scroll.get(): self.log_box.see(tk.END)
            
            try:
                self.parent.after(0, _update)
            except: pass

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.parent:
            self.btn_pause.config(text="RESUME" if self.is_paused else "PAUSE")

    def _browse_folder(self):
        d = filedialog.askdirectory()
        if d: self.folder_path.set(d); self._save_history(); self._scan_files()

    def _clear_queue(self):
        self.training_queue = []
        self.all_scanned_packets = []
        self.ram_cache = {}
        while not self.data_queue.empty(): self.data_queue.get()
        while not self.task_queue.empty(): self.task_queue.get()

        if self.parent:
            for w in self.scroll_fr.winfo_children(): w.destroy()
        
        self._log_threadsafe("Queue Cleared.", "warn")

    def _scan_files(self):
        folder = self.folder_path.get()
        if not os.path.exists(folder):
            self._log_threadsafe("Folder not found.", "error")
            return

        self._clear_queue()
        self._log_threadsafe(f"Scanning {folder}...", "info")

        ext_map = {
            'v': {'.png', '.jpg', '.jpeg', '.bmp'},
            'a': {'.mp3', '.wav', '.flac'},
            'c': {'.json', '.csv'},
            't': {'.txt', '.md', '.json', '.pdf', '.epub', '.mobi', '.rtf', '.doc', '.docx', '.srt', '.vtt', '.ass'},
            'vid': {'.mp4', '.mkv', '.avi'}
        }
        all_valid_exts = set().union(*ext_map.values())
        file_sets = {}
        ext_counts = {}

        for root, _, fs in os.walk(folder):
            for f in fs:
                base, ext = os.path.splitext(f)
                lext = ext.lower()
                if lext in all_valid_exts:
                    key = os.path.join(root, base).lower()
                    if key not in file_sets: file_sets[key] = {}

                    if lext in ext_map['v']:
                        file_sets[key]['v'] = os.path.join(root, f)
                    elif lext in ext_map['a']:
                        file_sets[key]['a'] = os.path.join(root, f)
                    elif lext in ext_map['c']:
                        file_sets[key]['c'] = os.path.join(root, f)
                    elif lext in ext_map['t']:
                        file_sets[key]['t'] = os.path.join(root, f)
                    elif lext in ext_map['vid']:
                        file_sets[key]['vid'] = os.path.join(root, f)

        q, tr, p, s = 0, 0, 0, 0
        sorted_keys = sorted(file_sets.keys())

        for key in sorted_keys:
            packet = file_sets[key].copy()
            
            # --- FIXED: EMPTY FILE DETECTION ---
            if 't' in packet:
                try:
                    if os.path.getsize(packet['t']) < 10:
                        del packet['t']
                except:
                    pass

            has_v, has_a, has_t, has_c = 'v' in packet, 'a' in packet, 't' in packet, 'c' in packet
            has_vid = 'vid' in packet

            # --- 1. TRIPLET / QUAD (Narrative Data) ---
            if (has_vid and has_t) or (has_v and has_a and has_t):
                packet['type'] = 'triplet'
                self.all_scanned_packets.append(packet)
                tr += 1
                continue 

            if has_v and has_a and has_t and has_c:
                packet['type'] = 'quad'
                self.all_scanned_packets.append(packet)
                q += 1
                continue

            # --- 2. PAIR (Partial Narrative) ---
            if (has_v and has_t) or (has_a and has_t) or (has_v and has_a):
                packet['type'] = 'pair'
                self.all_scanned_packets.append(packet)
                p += 1
                continue

            # --- 3. SINGLES (Orphaned Data) ---
            if has_vid:
                if 'vid' in packet:
                    path = packet['vid']
                    _, e = os.path.splitext(path)
                    self.all_scanned_packets.append({'type': 'single', 'vid': path, 'ext': e.lower()})
                    ext_counts[e.lower()] = ext_counts.get(e.lower(), 0) + 1
                    s += 1
                continue

            for k, path in file_sets[key].items():
                _, e = os.path.splitext(path)
                lext = e.lower()
                self.all_scanned_packets.append({'type': 'single', k: path, 'ext': lext})
                ext_counts[lext] = ext_counts.get(lext, 0) + 1
                s += 1

        # GUI Update Logic
        if self.parent:
            row = 0
            def add_chk(text, var_key, container=self.train_type_vars):
                var = tk.BooleanVar(value=True)
                container[var_key] = var
                ttk.Checkbutton(self.scroll_fr, text=text, variable=var).grid(row=row, column=0, sticky="w")

            if q > 0: add_chk(f"Quadruplets ({q})", 'quad'); row += 1
            if tr > 0: add_chk(f"Triplets ({tr})", 'triplet'); row += 1
            if p > 0: add_chk(f"Pairs ({p})", 'pair'); row += 1
            if row > 0: ttk.Separator(self.scroll_fr, orient='horizontal').grid(row=row, column=0, sticky="ew",
                                                                                pady=5); row += 1

            for ext in sorted(ext_counts.keys()):
                add_chk(f"{ext} ({ext_counts[ext]})", ext, self.train_ext_vars);
                row += 1

            self.scroll_fr.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        else:
            # Headless: Enable all found types
            self.train_type_vars['quad'] = MockVar(True)
            self.train_type_vars['triplet'] = MockVar(True)
            self.train_type_vars['pair'] = MockVar(True)
            for ext in ext_counts.keys():
                self.train_ext_vars[ext] = MockVar(True)

        self._log_threadsafe(f"Scan: {s} Single, {p} Pair, {tr} Trip, {q} Quad.", "success")
        if s + p + tr + q > 0: self._save_history()

    def _start_training(self):
        if self.is_training:
            self.stop_requested = True
            return

        active_id = self.app.active_lobe.get()
        brain = self.app.lobes[active_id]

        if brain is None:
            if self.parent:
                messagebox.showerror("Error", "No Brain Loaded.\nPlease load 'MaskedDiffusion-mHC'.")
            else:
                print("Error: No Brain Loaded.")
            return

        # --- SAFETY CHECK: MODEL TYPE ---
        lobe_type = self.app.lobe_types.get(active_id)
        if lobe_type != "diffusion":
            if self.parent:
                messagebox.showwarning("Mismatch", "Diffusion Trainer requires a Diffusion lobe.\nThis appears to be a standard Transformer.")
            else:
                print("Error: Model type mismatch. Requires diffusion.")
            return

        if not hasattr(brain, 'timestep_emb'):
            if self.parent:
                messagebox.showerror("Error", "Wrong Architecture.\nActive Lobe is not a Diffusion model.")
            else:
                print("Error: Architecture mismatch.")
            return

        self.training_queue = []
        # Handle MockVar vs BooleanVar
        # Helper to safely get value
        def safe_get_bool(v):
            return v.get() if v else False

        active_exts = {e for e, v in self.train_ext_vars.items() if safe_get_bool(v)}
        t_quad = safe_get_bool(self.train_type_vars.get('quad'))
        t_trip = safe_get_bool(self.train_type_vars.get('triplet'))
        t_pair = safe_get_bool(self.train_type_vars.get('pair'))

        for p in self.all_scanned_packets:
            pt = p['type']
            if pt == 'quad' and t_quad:
                self.training_queue.append(p)
            elif pt == 'triplet' and t_trip:
                self.training_queue.append(p)
            elif pt == 'pair' and t_pair:
                self.training_queue.append(p)
            elif pt == 'single' and p.get('ext') in active_exts:
                self.training_queue.append(p)

        if not self.training_queue:
            if self.parent:
                messagebox.showinfo("Info", "Queue is empty. Check your filters in the Census panel.")
            else:
                print("Queue is empty.")
            return

        if self.narrative_mode.get():
            self._log_threadsafe("Narrative Mode: Enforcing strict filename sort.", "info")
            self.training_queue.sort(key=lambda x: x.get('t', x.get('vid', x.get('v', x.get('a', '')))))

        self.total_items = len(self.training_queue) * self.target_epochs.get()
        self.is_training = True
        self.stop_requested = False
        
        if self.parent:
            self.btn_start.config(text="STOP")
            self.btn_pause.config(state="normal")

        while not self.data_queue.empty(): self.data_queue.get()
        while not self.task_queue.empty(): self.task_queue.get()

        threading.Thread(target=self._manager_worker, daemon=True).start()

        n_workers = self.num_workers.get()
        for i in range(n_workers):
            threading.Thread(target=self._loader_thread, args=(i,), daemon=True).start()

        threading.Thread(target=self._diffusion_worker, daemon=True).start()

    def _manager_worker(self):
        epochs = self.target_epochs.get()
        for _ in range(epochs):
            if self.narrative_mode.get():
                order = list(range(len(self.training_queue)))
            else:
                order = random.sample(range(len(self.training_queue)), len(self.training_queue))

            for idx in order:
                if self.stop_requested: break
                self.task_queue.put(self.training_queue[idx])

        for _ in range(self.num_workers.get() * 2): self.task_queue.put(None)

    def _loader_thread(self, worker_id):
        while not self.stop_requested:
            try:
                packet = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue

            if packet is None: return

            pkey = packet.get('t', packet.get('vid', packet.get('v', packet.get('a'))))
            if pkey in self.ram_cache:
                self.data_queue.put((packet, self.ram_cache[pkey]))
                time.sleep(0.001)
                continue

            data = None
            if packet['type'] == 'single' and 't' in packet:
                path = packet['t']
                _, ext = os.path.splitext(path)
                if ext.lower() in ['.pdf', '.epub', '.mobi'] and HAS_FITZ:
                    try:
                        doc = fitz.open(path)
                        text = ""
                        for page in doc: text += page.get_text()

                        ribo = self.app.ribosome
                        toks = ribo._tokenize(text)

                        t_tensor = torch.tensor(toks).unsqueeze(0).to(self.app.device)
                        v = torch.zeros(1, 1, 768).to(self.app.device)
                        a = torch.zeros(1, 1, 128).to(self.app.device)
                        c = torch.zeros(1, 1, 64).to(self.app.device)
                        data = (v, a, t_tensor, c, None)
                    except:
                        pass

            if data is None:
                try:
                    data = self.app.ribosome.ingest_packet(packet)
                except:
                    continue

            if data:
                if len(self.ram_cache) < self.MAX_CACHE_SIZE:
                    self.ram_cache[pkey] = data
                self.data_queue.put((packet, data))

            time.sleep(0.005)

    def _diffusion_worker(self):
        try:
            active_id = self.app.active_lobe.get()
            brain = self.app.lobes[active_id]
            opt = self.app.optimizers[active_id]
            scaler = self.app.scalers[active_id]
            brain.train()

            while True:
                if self.stop_requested: break
                item = self.data_queue.get()
                packet, (_, _, t, _, _) = item
                if t is None or t.size(1) < 2: continue

                while self.is_paused: time.sleep(0.1)

                mask_ratio = None
                if self.use_uniform.get():
                    mask_ratio = random.uniform(self.uniform_ratio_min.get(), self.uniform_ratio_max.get())

                brain.config.use_cross_modal_masking = self.use_cross_modal.get()
                brain.config.cross_modal_prob = self.cross_modal_prob.get()
                brain.config.use_modality_masking = self.use_modality_specific.get()

                timestep = torch.randint(0, 1000, (t.shape[0],), device=self.app.device)

                with self.app.gpu_lock:
                    opt.zero_grad()
                    with autocast('cuda'):
                        logits, mask_bool, _ = brain(None, None, t, None, timestep=timestep, mask_ratio=mask_ratio)

                        if mask_bool.ndim == 1:
                            mask_bool = mask_bool.unsqueeze(0).expand(logits.shape[0], -1)

                        flat_logits = logits.reshape(-1, logits.shape[-1])
                        flat_targets = t.reshape(-1)
                        flat_mask = mask_bool.reshape(-1)

                        pred = flat_logits[flat_mask]
                        target = flat_targets[flat_mask]

                        loss_recon = F.cross_entropy(pred, target)

                        game_penalty = torch.tensor(0.0, device=self.app.device)
                        has_game = False
                        for module in brain.modules():
                            if hasattr(module, 'game_loss'):
                                game_penalty += module.game_loss()
                                has_game = True

                        raw_recon = loss_recon.item()
                        raw_game = game_penalty.item() if has_game else 0.0

                        if math.isnan(raw_recon) or math.isinf(raw_recon):
                            self._log_threadsafe("NaN detected. Skipping.", "warn")
                            continue

                        self.loss_history['recon'].append(raw_recon)
                        if has_game: self.loss_history['game'].append(raw_game)

                        if self.nursery_autofit.get() and self.processed_count % 100 == 0:
                            if len(self.loss_history['recon']) > 50:
                                max_r = np.percentile(self.loss_history['recon'], 99)
                                cur_max = self.nurse_recon[1].get()
                                new_max = cur_max * 0.9 + max_r * 2.0 * 0.1
                                self.nurse_recon[1].set(round(new_max, 4))

                        def clamp(val_tensor, setting):
                            if not setting[2].get(): return val_tensor
                            limit = torch.tensor(setting[1].get(), device=self.app.device)
                            return torch.minimum(val_tensor, limit)

                        clamped_recon = clamp(loss_recon, self.nurse_recon)
                        clamped_game = clamp(game_penalty, self.nurse_game)
                        loss = clamped_recon + clamped_game

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()

                self.processed_count += 1

                q_len = len(self.training_queue)
                current_epoch = (self.processed_count // q_len) + 1 if q_len > 0 else 1

                if current_epoch not in self.app.graph_data:
                    self.app.graph_data[current_epoch] = {
                        'total': [], 'text': [], 'vis': [], 'aud': [],
                        'raw_total': [], 'raw_text': [], 'raw_vis': [], 'raw_aud': []
                    }

                val_recon = clamped_recon.item() if isinstance(clamped_recon, torch.Tensor) else clamped_recon
                val_game = clamped_game.item() if isinstance(clamped_game, torch.Tensor) else clamped_game

                self.app.graph_data[current_epoch]['total'].append(val_recon + val_game)
                self.app.graph_data[current_epoch]['text'].append(val_recon)
                self.app.graph_data[current_epoch]['vis'].append(val_game)
                self.app.graph_data[current_epoch]['aud'].append(0)

                self.app.graph_data[current_epoch]['raw_total'].append(raw_recon + raw_game)
                self.app.graph_data[current_epoch]['raw_text'].append(raw_recon)
                self.app.graph_data[current_epoch]['raw_vis'].append(raw_game)
                self.app.graph_data[current_epoch]['raw_aud'].append(0)

                if self.processed_count % 10 == 0:
                    pct = int((self.processed_count / self.total_items) * 100) if self.total_items else 0
                    game_str = f" | Game:{raw_game:.2f}" if has_game else ""
                    name = "Unknown"
                    if 't' in packet:
                        name = os.path.basename(packet['t'])
                    elif 'vid' in packet:
                        name = os.path.basename(packet['vid'])

                    self._log_threadsafe(f"[{self.processed_count}] ({pct}%) {name} | Recon:{raw_recon:.3f}{game_str}",
                                         "prog")

                if self.autosave_enabled.get() and self.processed_count % self.autosave_interval.get() == 0:
                    path = os.path.join(self.app.paths['lobes'], f"brain_lobe_{active_id}.pt")
                    try:
                        torch.save({
                            "genome": self.app.lobe_genomes.get(active_id, "Unknown"), 
                            "model_type": "diffusion", # Explicitly save type
                            "state_dict": brain.state_dict()
                        }, path)
                        self._log_threadsafe(f"AUTO-SAVE at {self.processed_count}", "save")
                    except Exception as e:
                        print(f"Save failed: {e}")

            self._log_threadsafe("Training Complete.", "info")

        except Exception as e:
            self._log_threadsafe(f"CRASH: {e}", "error")
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False
            if self.parent:
                self.btn_start.config(text="START DIFFUSION TRAINING")
                self.btn_pause.config(state="disabled")

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])
