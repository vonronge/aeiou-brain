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
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from datetime import datetime
import random
import queue
import json
from collections import deque
import math
import numpy as np

# --- SAFE IMPORTS ---
try:
    import fitz  # PyMuPDF

    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print(" ! Transformer Trainer: Install 'pymupdf' for PDF support.")


# --- HEADLESS HELPER ---
class MockVar:
    def __init__(self, value=None): self._val = value

    def set(self, value): self._val = value

    def get(self): return self._val


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Transformer Trainer"
        self.is_training = False
        self.is_paused = False
        self.stop_requested = False
        self.training_queue = []
        self.all_scanned_packets = []
        self.train_ext_vars = {}
        self.train_type_vars = {}
        self.data_queue = queue.Queue(maxsize=5)

        self.state_file = os.path.join(self.app.paths["root"], "trainer_state.json")
        self.history_file = os.path.join(self.app.paths["root"], "trainer_history.json")
        self.recent_folders = self._load_json(self.history_file, [])
        default_folder = self.recent_folders[0] if self.recent_folders else "D:/Training_Data"
        # Telepathy/Config override
        if hasattr(self.app.paths, "data") and os.path.exists(self.app.paths["data"]):
            default_folder = self.app.paths["data"]

        # --- INITIALIZE VARIABLES (GUI vs HEADLESS) ---
        if self.parent is None:
            # Headless Mode
            self.folder_path = MockVar(default_folder)
            self.target_epochs = MockVar(1)
            self.current_epoch_var = MockVar("0")
            self.auto_scroll = MockVar(False)
            self.narrative_mode = MockVar(True)
            self.autosave_enabled = MockVar(True)
            self.autosave_interval = MockVar(100)
            self.nursery_autofit = MockVar(True)
            self.use_game_loss = MockVar(False)
            self.use_fp32 = MockVar(False)
            self.nurse_pred = [MockVar(0.1), MockVar(2.5), MockVar(True)]
            self.nurse_game = [MockVar(0.001), MockVar(15.0), MockVar(True)]
            self.nurse_aux = [MockVar(1.0), MockVar(5.0), MockVar(False)]
            self.num_workers = MockVar(4)
        else:
            # GUI Mode
            self.folder_path = tk.StringVar(value=default_folder)
            self.target_epochs = tk.IntVar(value=1)
            self.current_epoch_var = tk.StringVar(value="0")
            self.auto_scroll = tk.BooleanVar(value=True)
            self.narrative_mode = tk.BooleanVar(value=True)
            self.autosave_enabled = tk.BooleanVar(value=True)
            self.autosave_interval = tk.IntVar(value=100)
            self.nursery_autofit = tk.BooleanVar(value=True)
            self.use_game_loss = tk.BooleanVar(value=False)
            self.use_fp32 = tk.BooleanVar(value=False)
            self.nurse_pred = [tk.DoubleVar(value=0.1), tk.DoubleVar(value=2.5), tk.BooleanVar(value=True)]
            self.nurse_game = [tk.DoubleVar(value=0.001), tk.DoubleVar(value=15.0), tk.BooleanVar(value=True)]
            self.nurse_aux = [tk.DoubleVar(value=1.0), tk.DoubleVar(value=5.0), tk.BooleanVar(value=False)]
            self.num_workers = tk.IntVar(value=4)

        self.processed_count = 0
        self.total_items = 0
        self.loss_history = {
            'pred': deque(maxlen=10000),
            'game': deque(maxlen=10000),
            'aux': deque(maxlen=10000)
        }

        self._setup_ui()
        if self.parent:
            self.parent.after(1000, self._load_session_state)

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

        # --- DATA FEED ---
        fr_src = ttk.LabelFrame(left, text="Data Feed", padding=10)
        fr_src.pack(fill="x", pady=5)

        row_src = ttk.Frame(fr_src)
        row_src.pack(fill="x", expand=True)
        self.cmb_folder = ttk.Combobox(row_src, textvariable=self.folder_path, values=self.recent_folders)
        self.cmb_folder.pack(side="left", fill="x", expand=True)
        self.cmb_folder.bind("<<ComboboxSelected>>", lambda e: self._scan_files())
        ttk.Button(row_src, text="📂", width=3, command=self._browse_folder).pack(side="left", padx=2)

        row_btns_src = ttk.Frame(fr_src, padding=(0, 5, 0, 0))
        row_btns_src.pack(fill="x")
        ttk.Button(row_btns_src, text="SCAN FOLDER", command=self._scan_files).pack(side="left", padx=2, fill="x",
                                                                                    expand=True)
        ttk.Button(row_btns_src, text="CLEAR QUEUE", command=self._clear_queue).pack(side="left", padx=2, fill="x",
                                                                                     expand=True)
        ttk.Button(row_btns_src, text="RESET SESSION", command=self._reset_session).pack(side="left", padx=2, fill="x",
                                                                                         expand=True)

        # --- CONTROLS ---
        fr_ctrl = ttk.LabelFrame(left, text="Training Operations", padding=10)
        fr_ctrl.pack(fill="x", pady=5)

        row_btns = ttk.Frame(fr_ctrl)
        row_btns.pack(fill="x", pady=(0, 5))
        self.btn_start = ttk.Button(row_btns, text="START TRAINING", command=self._start_training)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=2)
        self.btn_pause = ttk.Button(row_btns, text="PAUSE", command=self._toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", fill="x", expand=True, padx=2)

        row_sets = ttk.Frame(fr_ctrl)
        row_sets.pack(fill="x", pady=2)
        ttk.Label(row_sets, text="Epochs:").pack(side="left")
        ttk.Spinbox(row_sets, from_=1, to=999, textvariable=self.target_epochs, width=4).pack(side="left")
        ttk.Label(row_sets, text="Workers:").pack(side="left", padx=(10, 0))
        ttk.Scale(row_sets, from_=1, to=16, variable=self.num_workers, orient="horizontal", length=80).pack(side="left",
                                                                                                            padx=5)
        ttk.Checkbutton(row_sets, text="Narrative Mode", variable=self.narrative_mode).pack(side="left", padx=15)
        ttk.Label(row_sets, textvariable=self.current_epoch_var, foreground=self.app.colors["ACCENT"]).pack(
            side="right", padx=10)

        # --- NURSERY ---
        fr_nurse = ttk.LabelFrame(fr_ctrl, text="Nursery (Safety Corridor)", padding=5)
        fr_nurse.pack(fill="x", pady=5)

        h_row = ttk.Frame(fr_nurse)
        h_row.pack(fill="x")
        ttk.Checkbutton(h_row, text="Auto-Fit", variable=self.nursery_autofit).pack(side="left", padx=5)
        ttk.Checkbutton(h_row, text="Game Loss", variable=self.use_game_loss).pack(side="left", padx=10)
        ttk.Checkbutton(h_row, text="Force FP32", variable=self.use_fp32).pack(side="left", padx=10)

        grid = ttk.Frame(fr_nurse)
        grid.pack(fill="x", pady=5)
        ttk.Label(grid, text="Channel", font=("Segoe UI", 8, "bold")).grid(row=0, column=0, padx=5)
        ttk.Label(grid, text="Active", font=("Segoe UI", 8)).grid(row=0, column=1, padx=2)
        ttk.Label(grid, text="Min Loss", font=("Segoe UI", 8)).grid(row=0, column=2, padx=2)
        ttk.Label(grid, text="Max Loss", font=("Segoe UI", 8)).grid(row=0, column=3, padx=2)

        def add_row(r, label, vars):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="e", padx=5)
            ttk.Checkbutton(grid, variable=vars[2]).grid(row=r, column=1, padx=2)
            ttk.Entry(grid, textvariable=vars[0], width=6).grid(row=r, column=2, padx=2)
            ttk.Entry(grid, textvariable=vars[1], width=6).grid(row=r, column=3, padx=2)

        add_row(1, "Prediction:", self.nurse_pred)
        add_row(2, "Game / Vis:", self.nurse_game)
        add_row(3, "Aux / Aud:", self.nurse_aux)

        # --- LOGS ---
        fr_log = ttk.LabelFrame(left, text="Neural Logs", padding=10)
        fr_log.pack(fill="both", expand=True, pady=5)

        log_head = ttk.Frame(fr_log)
        log_head.pack(fill="x")
        ttk.Checkbutton(log_head, text="Autoscroll", variable=self.auto_scroll).pack(side="right")

        self.log_box = tk.Text(fr_log, font=("Consolas", 9), height=10, bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"], borderwidth=0)
        self.log_box.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(fr_log, orient="vertical", command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self.log_box.tag_config("info", foreground=self.app.colors["ACCENT"])
        self.log_box.tag_config("error", foreground=self.app.colors["ERROR"])
        self.log_box.tag_config("warn", foreground=self.app.colors["WARN"])
        self.log_box.tag_config("success", foreground=self.app.colors["SUCCESS"])
        self.log_box.tag_config("prog", foreground=self.app.colors["FG_DIM"])
        self.log_box.tag_config("scale", foreground="#FDD663")
        self.log_box.tag_config("save", foreground="#81C995", font=("Consolas", 9, "bold"))

        # --- CENSUS ---
        fr_census = ttk.LabelFrame(right, text="Census", padding=10)
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

    def _save_session_state(self):
        state = {
            "processed_count": self.processed_count,
            "total_items": self.total_items,
            "loss_history": {k: list(v) for k, v in self.loss_history.items()},
            "graph_data": self.app.graph_data,
            "folder": self.folder_path.get(),
            "target_epochs": self.target_epochs.get()
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"State Save Error: {e}")

    def _load_session_state(self):
        if not os.path.exists(self.state_file): return
        try:
            state = json.load(open(self.state_file, 'r'))
            self.folder_path.set(state.get("folder", "D:/Training_Data"))
            self.target_epochs.set(state.get("target_epochs", 1))
            self.processed_count = state.get("processed_count", 0)
            self.total_items = state.get("total_items", 0)
            hist = state.get("loss_history", {})
            for k, v in hist.items():
                if k in self.loss_history: self.loss_history[k] = deque(v, maxlen=10000)
            g_data = state.get("graph_data", {})
            self.app.graph_data = {int(k): v for k, v in g_data.items()}
            self._log(f"Session Restored: {self.processed_count}/{self.total_items} items.", "info")
            if 'tab_graphs' in self.app.plugins: self.app.plugins['tab_graphs']._update_graphs()
        except Exception as e:
            self._log(f"Session Restore Failed: {e}", "error")

    def _reset_session(self):
        if messagebox.askyesno("Reset Session", "Clear training progress and graphs?"):
            self.processed_count = 0
            self.total_items = 0
            self.loss_history['pred'].clear()
            self.loss_history['game'].clear()
            self.loss_history['aux'].clear()
            self.app.graph_data = {}
            if os.path.exists(self.state_file): os.remove(self.state_file)
            self._log("Session Cleared.", "warn")
            if 'tab_graphs' in self.app.plugins: self.app.plugins['tab_graphs']._update_graphs()

    def _clear_queue(self):
        self.training_queue = []
        self.all_scanned_packets = []
        self.train_ext_vars = {}
        self.train_type_vars = {}
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                break

        if self.parent:
            for w in self.scroll_fr.winfo_children(): w.destroy()

        self._log("Queue and Buffer Cleared.", "warn")

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])
        if hasattr(self, 'canvas'): self.canvas.config(bg=c["BG_MAIN"])

    def _browse_folder(self):
        d = filedialog.askdirectory()
        if d: self.folder_path.set(d); self._save_history(); self._scan_files()

    def _log(self, msg, tag="info"):
        if self.parent is None:
            # Headless logging
            print(f"[{tag.upper()}] {msg}")
        else:
            # GUI logging
            def _update():
                self.log_box.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n", tag)
                if self.auto_scroll.get(): self.log_box.see(tk.END)

            self.parent.after(0, _update)

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.parent:
            self.btn_pause.config(text="RESUME" if self.is_paused else "PAUSE")

    def _scan_files(self):
        folder = self.folder_path.get()
        if not os.path.exists(folder): self._log("Folder not found.", "error"); return

        self._clear_queue()
        self._log(f"Scanning {folder}...", "info")

        # --- EXTENSION MAP ---
        ext_map = {
            'v': {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'},
            'a': {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'},
            'c': {'.json', '.csv', '.ctl'},
            't': {'.pdf', '.txt', '.md', '.doc', '.docx', '.html', '.xml', '.py', '.js', '.c', '.cpp', '.h', '.srt',
                  '.vtt', '.ass'},
            'vid': {'.mp4', '.mkv', '.avi', '.mov'}
        }
        all_valid_exts = set().union(*ext_map.values())
        file_sets = {}
        ext_counts = {}

        for root, _, fs in os.walk(folder):
            for f in fs:
                base, ext = os.path.splitext(f)
                lext = ext.lower()
                if lext in all_valid_exts:
                    key = os.path.join(root, base)
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

            # --- EMPTY TEXT CHECK ---
            if 't' in packet:
                try:
                    if os.path.getsize(packet['t']) < 10:
                        del packet['t']
                except:
                    pass

            has_v, has_a, has_t, has_c = 'v' in packet, 'a' in packet, 't' in packet, 'c' in packet
            has_vid = 'vid' in packet

            if (has_vid and has_t) or (has_v and has_a and has_t):
                packet['type'] = 'triplet'
                self.all_scanned_packets.append(packet)
                tr += 1;
                continue

            if has_v and has_a and has_t and has_c:
                packet['type'] = 'quad'
                self.all_scanned_packets.append(packet)
                q += 1;
                continue

            if (has_v and has_t) or (has_a and has_t) or (has_v and has_a):
                packet['type'] = 'pair'
                self.all_scanned_packets.append(packet)
                p += 1;
                continue

            # Singles logic
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
                var = tk.BooleanVar(value=True);
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
            # Headless: Enable all found
            self.train_type_vars['quad'] = MockVar(True)
            self.train_type_vars['triplet'] = MockVar(True)
            self.train_type_vars['pair'] = MockVar(True)
            for ext in ext_counts.keys():
                self.train_ext_vars[ext] = MockVar(True)

        self._log(f"Scan: {s} Single, {p} Pair, {tr} Trip, {q} Quad.", "success")
        if s + p + tr + q > 0: self._save_history()

    def _start_training(self):
        if self.is_training: self.stop_requested = True; return
        active = self.app.active_lobe.get()
        if self.app.lobes[active] is None: self._log("No Lobe Loaded for training.", "error"); return

        # --- SAFETY CHECK: MODEL TYPE ---
        lobe_type = self.app.lobe_types.get(active)
        if lobe_type == "diffusion":
            if self.parent:
                messagebox.showwarning("Mismatch",
                                       "Transformer Trainer cannot train Diffusion lobes.\nPlease switch to the Diffusion Director plugin.")
            else:
                print("Error: Model type mismatch (Diffusion in AR trainer)")
            return

        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                break

        self.training_queue = []

        # Handle MockVar vs BooleanVar for headless safety
        def get_val(v):
            return v.get() if v else False

        active_exts = {e for e, v in self.train_ext_vars.items() if get_val(v)}
        t_quad = get_val(self.train_type_vars.get('quad'))
        t_trip = get_val(self.train_type_vars.get('triplet'))
        t_pair = get_val(self.train_type_vars.get('pair'))

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

        if not self.training_queue: self._log("Queue Empty.", "error"); return

        if self.narrative_mode.get():
            # Narrative Sort
            self.training_queue.sort(key=lambda x: x.get('t', x.get('vid', x.get('v', x.get('a', '')))))

        self.total_items = len(self.training_queue) * self.target_epochs.get()
        self.is_training = True;
        self.stop_requested = False

        if self.parent:
            self.btn_start.config(text="STOP")
            self.btn_pause.config(state="normal")

        threading.Thread(target=self._prefetch_worker, daemon=True).start()
        threading.Thread(target=self._training_worker, daemon=True).start()

    def _prefetch_worker(self):
        epochs = self.target_epochs.get()
        for _ in range(epochs):
            if not self.narrative_mode.get(): random.shuffle(self.training_queue)
            for packet in self.training_queue:
                if self.stop_requested: return
                try:
                    data = self.app.ribosome.ingest_packet(packet)
                    self.data_queue.put((packet, data))
                except:
                    continue
        self.data_queue.put(None)

    def _training_worker(self):
        def dampen_value(current, target, limit=0.1):
            delta = target - current
            max_delta = abs(current * limit)
            safe_delta = max(-max_delta, min(max_delta, delta))
            return current + safe_delta

        try:
            active_id = self.app.active_lobe.get()
            brain = self.app.lobes[active_id]
            opt = self.app.optimizers[active_id]
            scaler = self.app.scalers[active_id]
            brain.train()

            if self.processed_count >= self.total_items: self.processed_count = 0

            while True:
                if self.stop_requested: break
                item = self.data_queue.get()
                if item is None: break
                packet, (v, a, t, c, _) = item
                if t is None or t.size(1) < 2: continue

                if self.parent:
                    self.parent.update_idletasks()
                while self.is_paused: time.sleep(0.1);
                if self.parent: self.parent.update()

                input_t = t[:, :-1]
                labels = t[:, 1:]

                # Noise Injection
                if c is not None: c = c + torch.randn_like(c) * 1e-5
                if a is not None: a = a + torch.randn_like(a) * 1e-5
                if v is not None: v = v + torch.randn_like(v) * 1e-5

                with self.app.gpu_lock:
                    opt.zero_grad()
                    use_amp = not self.use_fp32.get()

                    with autocast('cuda', enabled=use_amp):
                        try:
                            logits, _, _ = brain(v, a, input_t, c)
                        except:
                            logits, _, _ = brain(v, a, input_t)

                        offset = (v.shape[1] if v is not None else 0) + (a.shape[1] if a is not None else 0) + (
                            c.shape[1] if c is not None else 0)
                        if logits.shape[1] == input_t.shape[1]: offset = 0
                        logits_text = logits[:, offset: offset + input_t.shape[1], :]
                        loss_txt = F.cross_entropy(logits_text.reshape(-1, logits_text.size(-1)), labels.reshape(-1),
                                                   ignore_index=50256)

                        game_penalty = torch.tensor(0.0, device=self.app.device)
                        has_game = False
                        if self.use_game_loss.get():
                            for module in brain.modules():
                                if hasattr(module, 'game_loss'):
                                    game_penalty += module.game_loss()
                                    has_game = True

                        raw_pred = loss_txt.item()
                        raw_b = game_penalty.item() if has_game else 0.0

                        if any(math.isnan(x) or math.isinf(x) for x in [raw_pred, raw_b]):
                            fname = os.path.basename(packet.get('t', 'Unknown'))
                            self._log(f"NaN Loss in {fname}. Skipping.", "warn")
                            continue

                        self.loss_history['pred'].append(raw_pred)
                        if raw_b > 0: self.loss_history['game'].append(raw_b)

                        # Nursery Auto-Fit
                        if self.nursery_autofit.get() and self.processed_count % 100 == 0:
                            p_vals = [x for x in self.loss_history['pred'] if not math.isnan(x)]
                            if len(p_vals) > 50:
                                p_max = dampen_value(self.nurse_pred[1].get(), np.percentile(p_vals, 99) * 2.0)
                                self.nurse_pred[1].set(round(p_max, 4))

                            g_vals = [x for x in self.loss_history['game'] if not math.isnan(x)]
                            if len(g_vals) > 50:
                                g_max = dampen_value(self.nurse_game[1].get(), np.percentile(g_vals, 99) * 1.5)
                                self.nurse_game[1].set(round(g_max, 4))

                        def get_scaler(raw, settings):
                            low, high, is_active = settings[0].get(), settings[1].get(), settings[2].get()
                            if not is_active: return 1.0, raw
                            if abs(raw) < 1e-9: return 1.0, raw
                            if raw < low:
                                return (low / raw), low
                            elif raw > high:
                                return (high / raw), high
                            return 1.0, raw

                        s_pred, graph_pred = get_scaler(raw_pred, self.nurse_pred)
                        s_game, graph_b = get_scaler(raw_b, self.nurse_game)

                        loss = (loss_txt * s_pred)
                        if has_game: loss += (game_penalty * s_game)

                    # Backward
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)

                    # Gradient Check
                    total_norm = torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
                    if math.isnan(total_norm) or math.isinf(total_norm):
                        self._log("Warn: Gradient Explosion (clipped)", "warn")
                        # Zero out broken grads
                        for p in brain.parameters():
                            if p.grad is not None:
                                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)

                    scaler.step(opt)
                    scaler.update()

                self.processed_count += 1
                val = loss.item()
                ep = (self.processed_count // len(self.training_queue)) + 1 if len(self.training_queue) > 0 else 1

                if ep not in self.app.graph_data:
                    self.app.graph_data[ep] = {'total': [], 'text': [], 'vis': [], 'aud': [], 'raw_total': [],
                                               'raw_text': [], 'raw_vis': [], 'raw_aud': []}

                self.app.graph_data[ep]['total'].append(val)
                self.app.graph_data[ep]['text'].append(graph_pred)
                self.app.graph_data[ep]['vis'].append(graph_b)
                self.app.graph_data[ep]['raw_total'].append(raw_pred + raw_b)
                self.app.graph_data[ep]['raw_text'].append(raw_pred)
                self.app.graph_data[ep]['raw_vis'].append(raw_b)

                if self.autosave_enabled.get() and self.processed_count % self.autosave_interval.get() == 0:
                    self._save_session_state()
                    save_path = os.path.join(self.app.paths['lobes'], f"brain_lobe_{active_id}.pt")
                    try:
                        torch.save({
                            "genome": self.app.lobe_genomes.get(active_id, "Unknown"),
                            "model_type": self.app.lobe_types.get(active_id, "ar"),  # Save Type
                            "state_dict": brain.state_dict()
                        }, save_path)
                        # Only update GUI log if parent exists, else print
                        msg = f"[SAVE] Auto-saved at step {self.processed_count}"
                        if self.parent:
                            self.parent.after(0, lambda: self._log(msg, "save"))
                        else:
                            print(msg)
                    except:
                        pass

                if self.processed_count % 10 == 0:
                    pct = int((self.processed_count / self.total_items) * 100) if self.total_items > 0 else 0
                    name = os.path.basename(packet.get('t', 'Unknown'))
                    game_stat = f" | Game:{graph_b:.2f}" if has_game else ""
                    msg = f"[{self.processed_count}] ({pct}%) {name} | Tot:{val:.2f} | Pred:{graph_pred:.2f}{game_stat}"
                    if self.parent:
                        self.parent.after(0, lambda m=msg: self._log(m, "prog"))
                    else:
                        print(msg)

            if self.parent:
                self.parent.after(0, lambda: self._log("Training Complete.", "success"))
            else:
                print("Training Complete.")
            self._save_session_state()

        except Exception as e:
            if self.parent:
                self.parent.after(0, lambda m=f"CRASH: {e}": self._log(m, "error"))
            else:
                print(f"CRASH: {e}")
            import traceback;
            traceback.print_exc()
        finally:
            self.is_training = False
            if self.parent:
                self.parent.after(0, lambda: self.btn_start.config(text="START TRAINING"))
                self.parent.after(0, lambda: self.btn_pause.config(state="disabled"))

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])
        if hasattr(self, 'canvas'): self.canvas.config(bg=c["BG_MAIN"])