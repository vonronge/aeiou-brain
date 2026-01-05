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

# FILE: Plugins/tab_video_factory.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
import threading
import subprocess
import uuid
import queue
import time
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Video Factory"

        # State
        self.is_processing = False
        self.stop_requested = False
        self.video_queue = []
        self.update_queue = queue.Queue()

        # Config Defaults
        default_in = self.app.paths.get("data", "")
        default_out = os.path.join(default_in, "Processed_Video")

        self.input_dir = tk.StringVar(value=default_in)
        self.output_dir = tk.StringVar(value=default_out)
        self.chunk_len = tk.IntVar(value=10)  # Seconds
        self.resize_dim = tk.IntVar(value=512)  # px
        self.skip_existing = tk.BooleanVar(value=True)

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        # 1. IO CONFIG
        fr_io = ttk.LabelFrame(self.parent, text="Pipeline Configuration", padding=10)
        fr_io.pack(fill="x", padx=10, pady=5)

        # Input
        r1 = ttk.Frame(fr_io)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Video Source:", width=12).pack(side="left")
        ttk.Entry(r1, textvariable=self.input_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂", width=4, command=lambda: self._browse(self.input_dir)).pack(side="left")

        # Output
        r2 = ttk.Frame(fr_io)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Output Root:", width=12).pack(side="left")
        ttk.Entry(r2, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r2, text="📂", width=4, command=lambda: self._browse(self.output_dir)).pack(side="left")

        # Settings
        r3 = ttk.Frame(fr_io)
        r3.pack(fill="x", pady=5)
        ttk.Label(r3, text="Chunk Size (s):").pack(side="left")
        ttk.Spinbox(r3, from_=1, to=60, textvariable=self.chunk_len, width=5).pack(side="left", padx=5)

        ttk.Label(r3, text="Resize (px):").pack(side="left", padx=(10, 0))
        ttk.Spinbox(r3, from_=128, to=1024, increment=64, textvariable=self.resize_dim, width=5).pack(side="left",
                                                                                                      padx=5)

        ttk.Checkbutton(r3, text="Skip Existing", variable=self.skip_existing).pack(side="left", padx=15)

        # 2. QUEUE VIEW
        fr_list = ttk.Frame(self.parent)
        fr_list.pack(fill="both", expand=True, padx=10, pady=5)

        # Tools
        tb = ttk.Frame(fr_list)
        tb.pack(fill="x", pady=2)
        ttk.Button(tb, text="SCAN FOR VIDEOS", command=self._scan).pack(side="left", fill="x", expand=True)
        ttk.Label(tb, text=" | ", foreground=self.app.colors["FG_DIM"]).pack(side="left")
        self.lbl_count = ttk.Label(tb, text="0 files found", foreground=self.app.colors["ACCENT"])
        self.lbl_count.pack(side="left")

        # Tree
        cols = ("File", "Size", "Status")
        self.tree = ttk.Treeview(fr_list, columns=cols, show="headings", height=8)
        self.tree.heading("File", text="Filename")
        self.tree.heading("Size", text="Size")
        self.tree.heading("Status", text="Status")

        self.tree.column("File", width=400)
        self.tree.column("Size", width=80, anchor="center")
        self.tree.column("Status", width=150, anchor="center")

        sb = ttk.Scrollbar(fr_list, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 3. ACTIONS
        fr_run = ttk.Frame(self.parent, padding=10)
        fr_run.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(fr_run, text="START PROCESSING", command=self._toggle_run, state="disabled")
        self.btn_run.pack(side="left", fill="x", expand=True, padx=5)

        self.progress = ttk.Progressbar(fr_run, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=5)

        # Status Bar
        self.lbl_status = ttk.Label(self.parent, text="Ready.", foreground=self.app.colors["FG_DIM"], anchor="w")
        self.lbl_status.pack(fill="x", padx=15, pady=(0, 10))

    # --- HELPERS ---
    def _browse(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d: var.set(d)

    def _log(self, msg, tag="INFO"):
        self.update_queue.put(lambda: self.lbl_status.config(text=msg))
        if self.app.golgi:
            # Route to Golgi
            level = "INFO"
            if tag == "ERROR":
                level = "ERROR"
            elif tag == "SUCCESS":
                level = "SUCCESS"
            self.app.golgi._dispatch(level, msg, source="VideoFactory")

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                fn = self.update_queue.get_nowait()
                fn()
            except:
                break
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _update_tree(self, iid, col, val):
        self.update_queue.put(lambda: self.tree.set(iid, col, val))

    # --- SCANNING ---
    def _scan(self):
        folder = self.input_dir.get()
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Input folder does not exist.")
            return

        for i in self.tree.get_children(): self.tree.delete(i)
        self.video_queue = []

        valid_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

        for root, _, files in os.walk(folder):
            for f in files:
                _, ext = os.path.splitext(f)
                if ext.lower() in valid_exts:
                    full = os.path.join(root, f)
                    sz = f"{os.path.getsize(full) / (1024 * 1024):.1f} MB"

                    self.tree.insert("", "end", iid=full, values=(f, sz, "Queued"))
                    self.video_queue.append(full)

        self.lbl_count.config(text=f"{len(self.video_queue)} files found")
        if self.video_queue:
            self.btn_run.config(state="normal")
        self._log(f"Scan complete. Found {len(self.video_queue)} videos.")

    # --- PROCESSING ---
    def _toggle_run(self):
        if self.is_processing:
            self.stop_requested = True
            self.btn_run.config(text="STOPPING...")
        else:
            self.stop_requested = False
            self.is_processing = True
            self.btn_run.config(text="STOP PROCESSING")
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        out_root = self.output_dir.get()
        if not os.path.exists(out_root):
            try:
                os.makedirs(out_root)
            except:
                pass

        chunk_s = self.chunk_len.get()
        dim = self.resize_dim.get()

        total = len(self.video_queue)

        for idx, vid_path in enumerate(self.video_queue):
            if self.stop_requested: break

            fname = os.path.basename(vid_path)
            self._update_tree(vid_path, "Status", "Processing...")
            self._log(f"Processing {fname} ({idx + 1}/{total})...")

            # Create subfolder for this video? Or flat?
            # Let's do flat but prefixed
            base_name = os.path.splitext(fname)[0].replace(" ", "_")

            try:
                cap = cv2.VideoCapture(vid_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps

                # Iterate chunks
                num_chunks = int(duration // chunk_s)

                for i in range(num_chunks):
                    if self.stop_requested: break

                    start_t = i * chunk_s
                    # Naming scheme: video_T0000.jpg
                    chunk_id = f"{base_name}_T{int(start_t):04d}"
                    out_img = os.path.join(out_root, f"{chunk_id}.jpg")
                    out_aud = os.path.join(out_root, f"{chunk_id}.wav")

                    if self.skip_existing.get() and os.path.exists(out_img) and os.path.exists(out_aud):
                        continue

                    # 1. VISUAL: Grab frame at start_t
                    frame_id = int(start_t * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, frame = cap.read()

                    if ret:
                        # Resize
                        h, w = frame.shape[:2]
                        scale = dim / min(h, w)
                        nh, nw = int(h * scale), int(w * scale)
                        frame = cv2.resize(frame, (nw, nh))

                        # Center crop to square (optional, but good for training)
                        # For now, just save resized
                        cv2.imwrite(out_img, frame)

                    # 2. AUDIO: Extract segment using ffmpeg
                    # ffmpeg -ss {start} -t {dur} -i {in} -ac 1 -ar 24000 {out}
                    cmd = [
                        "ffmpeg", "-y", "-v", "error",
                        "-ss", str(start_t),
                        "-t", str(chunk_s),
                        "-i", vid_path,
                        "-ac", "1",  # Mono
                        "-ar", "24000",  # Ribosome standard
                        out_aud
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    # Update Progress
                    pct = ((idx + (i / num_chunks)) / total) * 100
                    self.update_queue.put(lambda v=pct: self.progress.configure(value=v))

                cap.release()
                self._update_tree(vid_path, "Status", "Done")

            except Exception as e:
                self._update_tree(vid_path, "Status", "Failed")
                self._log(f"Error on {fname}: {e}", "ERROR")

        self.is_processing = False
        self.update_queue.put(lambda: self.btn_run.config(text="START PROCESSING"))
        self._log("Batch Processing Complete.", "SUCCESS")
        self.update_queue.put(lambda: self.progress.configure(value=0))

    def on_theme_change(self):
        pass