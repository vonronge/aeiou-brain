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
import random
import queue
import traceback
from collections import deque
from datetime import datetime


# --- HEADLESS COMPATIBILITY ---
class MockVar:
    def __init__(self, value=None): self._val = value

    def set(self, value): self._val = value

    def get(self): return self._val


# --- DATASET ITERATOR ---
class DiffusionDataset:
    """
    An iterable that yields one epoch of data.
    Refreshes order on every iteration if shuffling is enabled.
    """

    def __init__(self, packet_list, app, settings):
        self.packets = packet_list
        self.app = app
        self.settings = settings

    def __iter__(self):
        indices = list(range(len(self.packets)))
        if not self.settings.get("narrative", False):
            random.shuffle(indices)

        for idx in indices:
            if getattr(self.app.cytoplasm, "stop_requested", False): break

            packet = self.packets[idx]
            try:
                # Membrane returns (v, a, t, c, meta)
                data = self.app.ribosome.ingest_packet(packet)
            except:
                continue

            v, a, t, c, _ = data
            if t is None or t.size(1) < 1: continue

            mask_ratio = None
            if self.settings["use_uniform"]:
                low = self.settings["uniform_min"]
                high = self.settings["uniform_max"]
                mask_ratio = random.uniform(low, high)

            yield (v, a, t, c, mask_ratio)


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Diffusion Director"

        self.is_training = False
        self.training_queue = []
        self.all_scanned_packets = []

        self.update_queue = queue.Queue()
        self.processed_count = 0
        self.total_items = 0

        self.train_ext_vars = {}
        self.train_type_vars = {}
        self.recent_folders = ["D:/Training_Data"]

        # --- UI VARIABLES ---
        if self.parent is None:
            self.folder_path = MockVar(self.app.paths["data"])
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
        else:
            self.folder_path = tk.StringVar(value=self.app.paths["data"])
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

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        # === SCALING FIX ===
        scale = getattr(self.app, 'ui_scale', 1.0)
        log_font = ("Consolas", int(10 * scale))

        split = ttk.PanedWindow(self.parent, orient="horizontal")
        split.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(split)
        right = ttk.Frame(split)
        split.add(left, weight=3)
        split.add(right, weight=1)

        # 1. Source
        fr_src = ttk.LabelFrame(left, text="Data Source", padding=10)
        fr_src.pack(fill="x", pady=5)

        r1 = ttk.Frame(fr_src)
        r1.pack(fill="x")
        self.cmb_folder = ttk.Combobox(r1, textvariable=self.folder_path, values=self.recent_folders)
        self.cmb_folder.pack(side="left", fill="x", expand=True)
        ttk.Button(r1, text="📂", width=3, command=self._browse).pack(side="left", padx=2)
        ttk.Button(r1, text="SCAN", command=self._scan).pack(side="left", padx=2)

        # 2. Masking
        fr_mask = ttk.LabelFrame(left, text="Masking Strategy", padding=10)
        fr_mask.pack(fill="x", pady=5)

        r_uni = ttk.Frame(fr_mask)
        r_uni.pack(fill="x")
        ttk.Checkbutton(r_uni, text="Uniform Random", variable=self.use_uniform).pack(side="left")
        ttk.Label(r_uni, text="Ratio:").pack(side="left", padx=(10, 0))
        ttk.Entry(r_uni, textvariable=self.uniform_ratio_min, width=5).pack(side="left", padx=2)
        ttk.Label(r_uni, text="-").pack(side="left")
        ttk.Entry(r_uni, textvariable=self.uniform_ratio_max, width=5).pack(side="left", padx=2)

        r_cross = ttk.Frame(fr_mask)
        r_cross.pack(fill="x", pady=5)
        ttk.Checkbutton(r_cross, text="Cross-Modal Drop", variable=self.use_cross_modal).pack(side="left")
        ttk.Label(r_cross, text="Prob:").pack(side="left", padx=(10, 0))
        ttk.Scale(r_cross, from_=0.0, to=1.0, variable=self.cross_modal_prob, orient="horizontal").pack(side="left",
                                                                                                        fill="x",
                                                                                                        expand=True)

        # 3. Controls
        fr_ctrl = ttk.LabelFrame(left, text="Training Control", padding=10)
        fr_ctrl.pack(fill="x", pady=5)

        r_run = ttk.Frame(fr_ctrl)
        r_run.pack(fill="x")
        self.btn_start = ttk.Button(r_run, text="START DIFFUSION TRAINING", command=self._start_training)
        self.btn_start.pack(side="left", fill="x", expand=True)
        self.btn_pause = ttk.Button(r_run, text="PAUSE", command=self._toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", fill="x", expand=True, padx=5)

        r_set = ttk.Frame(fr_ctrl)
        r_set.pack(fill="x", pady=5)
        ttk.Label(r_set, text="Epochs:").pack(side="left")
        ttk.Spinbox(r_set, from_=1, to=1000, textvariable=self.target_epochs, width=5).pack(side="left", padx=5)
        ttk.Checkbutton(r_set, text="Narrative Mode (No Shuffle)", variable=self.narrative_mode).pack(side="left",
                                                                                                      padx=10)

        # 4. Logs (Applied Font Here)
        fr_log = ttk.LabelFrame(left, text="Director Logs", padding=10)
        fr_log.pack(fill="both", expand=True, pady=5)

        self.log_box = tk.Text(fr_log, font=log_font, height=10, bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"])
        self.log_box.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(fr_log, command=self.log_box.yview)
        sb.pack(side="right", fill="y")
        self.log_box.config(yscrollcommand=sb.set)

        # 5. Census
        fr_census = ttk.LabelFrame(right, text="Census (Filters)", padding=10)
        fr_census.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(fr_census, width=200, bg=self.app.colors["BG_MAIN"], highlightthickness=0)
        scr = ttk.Scrollbar(fr_census, command=self.canvas.yview)
        self.scroll_fr = ttk.Frame(self.canvas, style="Card.TFrame")

        self.scroll_fr.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_fr, anchor="nw")
        self.canvas.configure(yscrollcommand=scr.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scr.pack(side="right", fill="y")

    # --- HELPERS ---
    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.folder_path.get())
        if d: self.folder_path.set(d)

    def _log(self, msg):
        self.update_queue.put(lambda: self._write_log(msg))

    def _write_log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n")
        if self.auto_scroll.get(): self.log_box.see(tk.END)

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                fn = self.update_queue.get_nowait()
                fn()
            except:
                break
        if self.parent: self.parent.after(100, self._process_gui_queue)

    def _toggle_pause(self):
        self.app.cytoplasm.pause()
        paused = self.app.cytoplasm.is_paused
        self.btn_pause.config(text="RESUME" if paused else "PAUSE")

    # --- SCANNING ---
    def _scan(self):
        folder = self.folder_path.get()
        if not os.path.exists(folder):
            self._log("Folder not found.")
            return

        self._log(f"Scanning {folder}...")
        self.all_scanned_packets = []
        self.train_type_vars = {}
        self.train_ext_vars = {}

        ext_map = {
            'v': {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'},
            'a': {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'},
            'c': {'.json', '.csv', '.ctl'},
            't': {'.txt', '.md', '.json', '.pdf', '.epub', '.mobi', '.rtf', '.doc', '.docx', '.srt', '.vtt', '.ass'},
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
                    key = os.path.join(root, base).lower()
                    if key not in file_sets: file_sets[key] = {}
                    full_path = os.path.join(root, f)
                    if lext in ext_map['v']:
                        file_sets[key]['v'] = full_path
                    elif lext in ext_map['a']:
                        file_sets[key]['a'] = full_path
                    elif lext in ext_map['c']:
                        file_sets[key]['c'] = full_path
                    elif lext in ext_map['t']:
                        file_sets[key]['t'] = full_path
                    elif lext in ext_map['vid']:
                        file_sets[key]['vid'] = full_path

        q, tr, p, s = 0, 0, 0, 0
        sorted_keys = sorted(file_sets.keys())

        for key in sorted_keys:
            packet = file_sets[key].copy()
            if 't' in packet:
                try:
                    if os.path.getsize(packet['t']) < 10: del packet['t']
                except:
                    pass

            has_v = 'v' in packet
            has_a = 'a' in packet
            has_t = 't' in packet
            has_c = 'c' in packet
            has_vid = 'vid' in packet

            if has_v and has_a and has_t and has_c:
                packet['type'] = 'quad'
                self.all_scanned_packets.append(packet)
                q += 1
                continue
            if (has_vid and has_t) or (has_v and has_a and has_t):
                packet['type'] = 'triplet'
                self.all_scanned_packets.append(packet)
                tr += 1
                continue
            if (has_v and has_t) or (has_a and has_t) or (has_v and has_a):
                packet['type'] = 'pair'
                self.all_scanned_packets.append(packet)
                p += 1
                continue
            if has_vid:
                path = packet['vid']
                _, e = os.path.splitext(path)
                lext = e.lower()
                self.all_scanned_packets.append({'type': 'single', 'vid': path, 'ext': lext})
                ext_counts[lext] = ext_counts.get(lext, 0) + 1
                s += 1
                continue
            for k, path in file_sets[key].items():
                _, e = os.path.splitext(path)
                lext = e.lower()
                self.all_scanned_packets.append({'type': 'single', k: path, 'ext': lext})
                ext_counts[lext] = ext_counts.get(lext, 0) + 1
                s += 1

        self._log(f"Scan: {s} Single, {p} Pair, {tr} Trip, {q} Quad.", )

        if self.parent:
            for w in self.scroll_fr.winfo_children(): w.destroy()
            row = 0

            def add_chk(text, var_key, container=self.train_type_vars):
                var = tk.BooleanVar(value=True)
                container[var_key] = var
                ttk.Checkbutton(self.scroll_fr, text=text, variable=var).grid(row=row, column=0, sticky="w")

            if q > 0: add_chk(f"Quadruplets ({q})", 'quad'); row += 1
            if tr > 0: add_chk(f"Triplets ({tr})", 'triplet'); row += 1
            if p > 0: add_chk(f"Pairs ({p})", 'pair'); row += 1
            if row > 0:
                ttk.Separator(self.scroll_fr, orient='horizontal').grid(row=row, column=0, sticky="ew", pady=5)
                row += 1
            for ext in sorted(ext_counts.keys()):
                add_chk(f"{ext} ({ext_counts[ext]})", ext, self.train_ext_vars)
                row += 1
            self.scroll_fr.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # --- TRAINING START ---
    def _start_training(self):
        active_id = self.app.active_lobe.get()
        handle = self.app.lobe_manager.get_lobe(active_id)
        if not handle:
            messagebox.showerror("Error", "No Lobe Loaded.")
            return
        if handle.model_type != "diffusion":
            messagebox.showwarning("Mismatch", "Active lobe is not a Diffusion model.")
            return

        self.training_queue = []

        def safe_get(v):
            return v.get() if v else False

        t_quad = safe_get(self.train_type_vars.get('quad'))
        t_trip = safe_get(self.train_type_vars.get('triplet'))
        t_pair = safe_get(self.train_type_vars.get('pair'))
        active_exts = {e for e, v in self.train_ext_vars.items() if safe_get(v)}

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
            self._log("Queue empty.")
            return

        if self.narrative_mode.get():
            self.training_queue.sort(key=lambda x: x.get('t', x.get('vid', x.get('v', ''))))

        if self.is_training:
            self.app.cytoplasm.stop()
            self.btn_start.config(text="STOPPING...")
            return

        self.is_training = True
        self.btn_start.config(text="STOP")
        self.btn_pause.config(state="normal")

        from Organelles.cytoplasm import TrainConfig
        iter_settings = {
            "narrative": self.narrative_mode.get(),
            "use_uniform": self.use_uniform.get(),
            "uniform_min": self.uniform_ratio_min.get(),
            "uniform_max": self.uniform_ratio_max.get()
        }
        dataset = DiffusionDataset(self.training_queue, self.app, iter_settings)

        conf = TrainConfig(
            epochs=self.target_epochs.get(),
            autosave_interval=self.autosave_interval.get(),
            nursery_active=self.nursery_autofit.get(),
            loss_clamp_prediction=(self.nurse_recon[0].get(), self.nurse_recon[1].get()) if self.nurse_recon[
                2].get() else None,
            loss_clamp_game=(self.nurse_game[0].get(), self.nurse_game[1].get()) if self.nurse_game[2].get() else None
        )

        self.app.cytoplasm.clear_callbacks()
        self.app.cytoplasm.register_callback("step", self._on_step)
        self.app.cytoplasm.register_callback("epoch", lambda e: self._log(f"Epoch {e} started"))
        self.app.cytoplasm.register_callback("autosave", self._on_autosave)
        self.app.cytoplasm.register_callback("finished", self._on_finished)
        self.app.cytoplasm.register_callback("error", lambda e: self._log(f"Error: {e}"))

        threading.Thread(target=self.app.cytoplasm.train,
                         args=(conf, handle, dataset, "diffusion"),
                         daemon=True).start()

    def _on_step(self, step, loss_dict):
        if step % 10 == 0:
            recon = loss_dict.get("recon", 0)
            game = loss_dict.get("game", 0)
            self._log(f"Step {step} | Recon: {recon:.4f} | Game: {game:.4f}")
            ep = 1
            if ep not in self.app.graph_data:
                self.app.graph_data[ep] = {'total': [], 'text': [], 'vis': [], 'raw_total': [], 'raw_text': [],
                                           'raw_vis': []}
            self.app.graph_data[ep]['total'].append(loss_dict['total'])
            self.app.graph_data[ep]['raw_total'].append(loss_dict['total'])
            self.app.graph_data[ep]['raw_text'].append(recon)
            self.app.graph_data[ep]['raw_vis'].append(game)

    def _on_autosave(self, step):
        active_id = self.app.active_lobe.get()
        self.app.lobe_manager.save_lobe(active_id)
        self._log(f"Auto-saved Lobe {active_id}")

    def _on_finished(self):
        self.is_training = False
        self.update_queue.put(lambda: self.btn_start.config(text="START DIFFUSION TRAINING"))
        self._log("Training Session Ended.")

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])