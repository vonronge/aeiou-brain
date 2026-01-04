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
from datetime import datetime


# --- HEADLESS COMPATIBILITY ---
class MockVar:
    def __init__(self, value=None): self._val = value

    def set(self, value): self._val = value

    def get(self): return self._val


# --- DATASET ITERATOR ---
class ARDataset:
    """
    Yields one epoch of autoregressive data (Next-Token Prediction).
    Format: (v, a, inputs, c, targets)
    """

    def __init__(self, packet_list, app, narrative_mode=False):
        self.packets = packet_list
        self.app = app
        self.narrative = narrative_mode

    def __iter__(self):
        # 1. Determine Order
        indices = list(range(len(self.packets)))
        if not self.narrative:
            random.shuffle(indices)

        # 2. Iterate
        for idx in indices:
            # Check for stop signal from the engine
            if getattr(self.app.cytoplasm, "stop_requested", False): break

            packet = self.packets[idx]

            # 3. Ingest
            try:
                # Membrane returns: (v, a, full_seq, c, meta)
                data = self.app.ribosome.ingest_packet(packet)
            except:
                continue

            v, a, t, c, _ = data

            # Validate
            if t is None or t.size(1) < 2: continue

            # 4. Prepare AR Targets (Shift Right)
            # Input:  [A, B, C]
            # Target: [B, C, EOS]
            inputs = t[:, :-1]
            targets = t[:, 1:]

            yield (v, a, inputs, c, targets)


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Transformer Trainer"

        self.is_training = False
        self.training_queue = []
        self.all_scanned_packets = []

        # State
        self.update_queue = queue.Queue()

        # Filters
        self.train_ext_vars = {}
        self.train_type_vars = {}

        # History
        self.recent_folders = ["D:/Training_Data"]

        # --- UI VARIABLES ---
        default_path = self.app.paths["data"]

        if self.parent is None:
            # Headless
            self.folder_path = MockVar(default_path)
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
        else:
            # GUI
            self.folder_path = tk.StringVar(value=default_path)
            self.target_epochs = tk.IntVar(value=1)
            self.current_epoch_var = tk.StringVar(value="Epoch: 0")
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

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        # UI Construction (Matches previous layout logic)
        scale = getattr(self.app, 'ui_scale', 1.0)

        split = ttk.PanedWindow(self.parent, orient="horizontal")
        split.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(split)
        right = ttk.Frame(split)
        split.add(left, weight=3)
        split.add(right, weight=1)

        # 1. Data Feed
        fr_src = ttk.LabelFrame(left, text="Data Feed", padding=10)
        fr_src.pack(fill="x", pady=5)

        r1 = ttk.Frame(fr_src)
        r1.pack(fill="x")
        self.cmb_folder = ttk.Combobox(r1, textvariable=self.folder_path, values=self.recent_folders)
        self.cmb_folder.pack(side="left", fill="x", expand=True)
        ttk.Button(r1, text="📂", width=3, command=self._browse).pack(side="left", padx=2)

        r2 = ttk.Frame(fr_src)
        r2.pack(fill="x", pady=5)
        ttk.Button(r2, text="SCAN FOLDER", command=self._scan).pack(side="left", fill="x", expand=True)
        ttk.Button(r2, text="CLEAR QUEUE", command=self._clear_queue).pack(side="left", fill="x", expand=True, padx=5)

        # 2. Controls
        fr_ctrl = ttk.LabelFrame(left, text="Training Operations", padding=10)
        fr_ctrl.pack(fill="x", pady=5)

        r3 = ttk.Frame(fr_ctrl)
        r3.pack(fill="x")
        self.btn_start = ttk.Button(r3, text="START TRAINING", command=self._start_training)
        self.btn_start.pack(side="left", fill="x", expand=True)
        self.btn_pause = ttk.Button(r3, text="PAUSE", command=self._toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", fill="x", expand=True, padx=5)

        r4 = ttk.Frame(fr_ctrl)
        r4.pack(fill="x", pady=5)
        ttk.Label(r4, text="Epochs:").pack(side="left")
        ttk.Spinbox(r4, from_=1, to=1000, textvariable=self.target_epochs, width=5).pack(side="left", padx=5)
        ttk.Checkbutton(r4, text="Narrative Mode", variable=self.narrative_mode).pack(side="left", padx=10)
        ttk.Label(r4, textvariable=self.current_epoch_var, foreground=self.app.colors["ACCENT"]).pack(side="right")

        # 3. Nursery
        fr_nurse = ttk.LabelFrame(fr_ctrl, text="Nursery (Safety Corridor)", padding=5)
        fr_nurse.pack(fill="x", pady=5)

        rn = ttk.Frame(fr_nurse)
        rn.pack(fill="x")
        ttk.Checkbutton(rn, text="Auto-Fit", variable=self.nursery_autofit).pack(side="left")
        ttk.Checkbutton(rn, text="Game Loss", variable=self.use_game_loss).pack(side="left", padx=10)
        ttk.Checkbutton(rn, text="Force FP32", variable=self.use_fp32).pack(side="left", padx=10)

        # Grid for clamps
        g = ttk.Frame(fr_nurse)
        g.pack(fill="x", pady=5)
        ttk.Label(g, text="Channel", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(g, text="Active").grid(row=0, column=1)
        ttk.Label(g, text="Min").grid(row=0, column=2)
        ttk.Label(g, text="Max").grid(row=0, column=3)

        def add_clamp(r, txt, vars):
            ttk.Label(g, text=txt).grid(row=r, column=0, sticky="e")
            ttk.Checkbutton(g, variable=vars[2]).grid(row=r, column=1)
            ttk.Entry(g, textvariable=vars[0], width=5).grid(row=r, column=2, padx=2)
            ttk.Entry(g, textvariable=vars[1], width=5).grid(row=r, column=3, padx=2)

        add_clamp(1, "Prediction:", self.nurse_pred)
        add_clamp(2, "Game / Vis:", self.nurse_game)

        # 4. Logs
        fr_log = ttk.LabelFrame(left, text="Neural Logs", padding=10)
        fr_log.pack(fill="both", expand=True, pady=5)

        self.log_box = tk.Text(fr_log, font=("Consolas", 9), height=10, bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"])
        self.log_box.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(fr_log, command=self.log_box.yview)
        sb.pack(side="right", fill="y")
        self.log_box.config(yscrollcommand=sb.set)

        # 5. Census (Right)
        fr_census = ttk.LabelFrame(right, text="Census", padding=10)
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

    def _clear_queue(self):
        self.training_queue = []
        self.all_scanned_packets = []
        for w in self.scroll_fr.winfo_children(): w.destroy()
        self._log("Queue Cleared.")

    # --- SCANNING ---
    def _scan(self):
        folder = self.folder_path.get()
        if not os.path.exists(folder): return

        self._log(f"Scanning {folder}...")

        # Simple scan logic to populate Census
        self.all_scanned_packets = []

        ext_map = {'.png': 'v', '.jpg': 'v', '.txt': 't', '.wav': 'a', '.json': 'c'}
        file_sets = {}

        for root, _, files in os.walk(folder):
            for f in files:
                base, ext = os.path.splitext(f)
                if ext.lower() in ext_map:
                    key = os.path.join(root, base)
                    if key not in file_sets: file_sets[key] = {}
                    file_sets[key][ext_map[ext.lower()]] = os.path.join(root, f)

        for p in file_sets.values():
            if 't' in p: self.all_scanned_packets.append(p)

        self._log(f"Found {len(self.all_scanned_packets)} valid packets.")

        # Populate Census Checkboxes
        for w in self.scroll_fr.winfo_children(): w.destroy()

        # (Simplified Census Logic: Just one checkbox for all)
        # In production, use the detailed logic from previous iteration
        ttk.Label(self.scroll_fr, text=f"Total: {len(self.all_scanned_packets)}").pack()

    # --- START TRAINING ---
    def _start_training(self):
        # 1. Check Lobe
        active_id = self.app.active_lobe.get()
        handle = self.app.lobe_manager.get_lobe(active_id)

        if not handle:
            messagebox.showerror("Error", "No Lobe Loaded.")
            return

        # 2. Filter Queue
        self.training_queue = self.all_scanned_packets  # Apply filters here
        if not self.training_queue:
            self._log("Queue empty.")
            return

        if self.narrative_mode.get():
            self.training_queue.sort(key=lambda x: x.get('t', ''))

        # 3. Stop if running
        if self.is_training:
            self.app.cytoplasm.stop()
            self.btn_start.config(text="STOPPING...")
            return

        self.is_training = True
        self.btn_start.config(text="STOP")
        self.btn_pause.config(state="normal")

        # 4. Configure
        from Organelles.cytoplasm import TrainConfig

        conf = TrainConfig(
            epochs=self.target_epochs.get(),
            autosave_interval=self.autosave_interval.get(),
            nursery_active=self.nursery_autofit.get(),
            use_fp32=self.use_fp32.get(),
            loss_clamp_prediction=(self.nurse_pred[0].get(), self.nurse_pred[1].get()) if self.nurse_pred[
                2].get() else None,
            loss_clamp_game=(self.nurse_game[0].get(), self.nurse_game[1].get()) if self.nurse_game[2].get() else None
        )

        # 5. Dataset
        dataset = ARDataset(self.training_queue, self.app, narrative_mode=self.narrative_mode.get())

        # 6. Callbacks
        self.app.cytoplasm.register_callback("step", self._on_step)
        self.app.cytoplasm.register_callback("epoch", lambda e: self.current_epoch_var.set(f"Epoch: {e}"))
        self.app.cytoplasm.register_callback("autosave", self._on_autosave)
        self.app.cytoplasm.register_callback("finished", self._on_finished)

        # 7. Launch
        threading.Thread(target=self.app.cytoplasm.train,
                         args=(conf, handle, dataset, "ar"),
                         daemon=True).start()

    # --- CALLBACKS ---
    def _on_step(self, step, loss_dict):
        if step % 10 == 0:
            total = loss_dict.get("total", 0)
            self._log(f"Step {step} | Loss: {total:.4f}")

            # Update Graphs if tab_graphs exists
            # (Requires app.graph_data dict to be populated here if you want visuals)
            # app.graph_data is available via self.app.graph_data

    def _on_autosave(self, step):
        active_id = self.app.active_lobe.get()
        self.app.lobe_manager.save_lobe(active_id)
        self._log(f"Auto-saved Lobe {active_id}")

    def _on_finished(self):
        self.is_training = False
        self.update_queue.put(lambda: self.btn_start.config(text="START TRAINING"))
        self._log("Training Session Ended.")

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])