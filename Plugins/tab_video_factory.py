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
"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Video Factory (v23.3 "The Director"):
Extracts training Triplets (Frame, Audio, Text) from video files.
- Mode A (Semantic): If .srt exists, slices based on dialogue timestamps.
- Mode B (Raw): If no .srt, slices into fixed-length chunks (10s).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
import threading
import subprocess
import queue
import re
import math
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
        self.chunk_len = tk.IntVar(value=10)  # For Raw Mode
        self.resize_dim = tk.IntVar(value=512)
        self.skip_existing = tk.BooleanVar(value=True)
        self.use_srt = tk.BooleanVar(value=True)

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        # 1. IO CONFIG
        fr_io = ttk.LabelFrame(self.parent, text="Pipeline Configuration", padding=10)
        fr_io.pack(fill="x", padx=10, pady=5)

        # Input
        r1 = ttk.Frame(fr_io);
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Video Source:", width=12).pack(side="left")
        ttk.Entry(r1, textvariable=self.input_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂", width=4, command=lambda: self._browse(self.input_dir)).pack(side="left")

        # Output
        r2 = ttk.Frame(fr_io);
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Output Root:", width=12).pack(side="left")
        ttk.Entry(r2, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r2, text="📂", width=4, command=lambda: self._browse(self.output_dir)).pack(side="left")

        # Settings
        fr_set = ttk.LabelFrame(self.parent, text="Slicing Logic", padding=10)
        fr_set.pack(fill="x", padx=10, pady=5)

        ttk.Checkbutton(fr_set, text="Prefer SRT Subtitles (Semantic Slicing)", variable=self.use_srt).pack(side="left",
                                                                                                            padx=5)

        ttk.Label(fr_set, text="|  Fallback Chunk Size (s):").pack(side="left", padx=(15, 5))
        ttk.Spinbox(fr_set, from_=1, to=60, textvariable=self.chunk_len, width=5).pack(side="left")

        ttk.Label(fr_set, text="|  Resize (px):").pack(side="left", padx=(15, 5))
        ttk.Spinbox(fr_set, from_=128, to=1024, increment=64, textvariable=self.resize_dim, width=5).pack(side="left")

        ttk.Checkbutton(fr_set, text="Skip Existing", variable=self.skip_existing).pack(side="right", padx=5)

        # 2. QUEUE VIEW
        fr_list = ttk.Frame(self.parent)
        fr_list.pack(fill="both", expand=True, padx=10, pady=5)

        # Tools
        tb = ttk.Frame(fr_list)
        tb.pack(fill="x", pady=2)
        ttk.Button(tb, text="SCAN FOR VIDEOS", command=self._scan).pack(side="left", fill="x", expand=True)
        self.lbl_count = ttk.Label(tb, text="0 files found", foreground=self.app.colors["ACCENT"])
        self.lbl_count.pack(side="left", padx=10)

        # Tree
        cols = ("File", "Mode", "Status")
        self.tree = ttk.Treeview(fr_list, columns=cols, show="headings", height=8)
        self.tree.heading("File", text="Filename")
        self.tree.heading("Mode", text="Detected Mode")
        self.tree.heading("Status", text="Status")

        self.tree.column("File", width=350)
        self.tree.column("Mode", width=100, anchor="center")
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

        self.lbl_status = ttk.Label(self.parent, text="Ready.", foreground=self.app.colors["FG_DIM"], anchor="w")
        self.lbl_status.pack(fill="x", padx=15, pady=(0, 10))

    # --- HELPERS ---
    def _browse(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d: var.set(d)

    def _log(self, msg, tag="INFO"):
        self.update_queue.put(lambda: self.lbl_status.config(text=msg))
        if self.app.golgi:
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

    def _parse_timestamp(self, ts_str):
        """Converts '00:00:05,123' to seconds (float)."""
        try:
            ts_str = ts_str.strip().replace(',', '.')
            h, m, s = ts_str.split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
        except:
            return 0.0

    def _load_srt(self, srt_path):
        """Parses SRT into list of (start, end, text)."""
        segments = []
        if not os.path.exists(srt_path): return []

        try:
            with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Regex for SRT block
            # 1
            # 00:00:01,000 --> 00:00:04,000
            # Text
            pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\n\n).)*)',
                                 re.DOTALL)
            matches = pattern.findall(content)

            for _, start_str, end_str, text in matches:
                start = self._parse_timestamp(start_str)
                end = self._parse_timestamp(end_str)
                clean_text = text.replace('\n', ' ').strip()
                # Filter noise
                if clean_text and not clean_text.startswith('[') and (end - start > 0.5):
                    segments.append((start, end, clean_text))

            return segments
        except Exception as e:
            self._log(f"SRT Parse Error: {e}", "ERROR")
            return []

    # --- SCANNING ---
    def _scan(self):
        folder = self.input_dir.get()
        if not os.path.exists(folder): return

        for i in self.tree.get_children(): self.tree.delete(i)
        self.video_queue = []

        valid_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

        for root, _, files in os.walk(folder):
            for f in files:
                base, ext = os.path.splitext(f)
                if ext.lower() in valid_exts:
                    full_vid = os.path.join(root, f)
                    full_srt = os.path.join(root, base + ".srt")

                    mode = "Raw (Time)"
                    if self.use_srt.get() and os.path.exists(full_srt):
                        mode = "Semantic (SRT)"

                    self.tree.insert("", "end", iid=full_vid, values=(f, mode, "Queued"))
                    self.video_queue.append((full_vid, full_srt, mode))

        self.lbl_count.config(text=f"{len(self.video_queue)} files")
        if self.video_queue: self.btn_run.config(state="normal")
        self._log(f"Scan complete. {len(self.video_queue)} items.")

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

        chunk_len = self.chunk_len.get()
        target_dim = self.resize_dim.get()
        skip = self.skip_existing.get()

        total_files = len(self.video_queue)

        for f_idx, (vid_path, srt_path, mode) in enumerate(self.video_queue):
            if self.stop_requested: break

            fname = os.path.basename(vid_path)
            base_name = os.path.splitext(fname)[0].replace(" ", "_")
            self._update_tree(vid_path, "Status", "Processing...")
            self._log(f"Processing {fname} ({f_idx + 1}/{total_files})...")

            try:
                # 1. Determine Segments
                segments = []  # (start, end, text_content)

                if "SRT" in mode:
                    srt_data = self._load_srt(srt_path)
                    if srt_data:
                        segments = srt_data
                        self._log(f" > Loaded {len(segments)} subtitles.")
                    else:
                        self._log(" > SRT empty/invalid. Fallback to Raw.")
                        mode = "Raw"  # Fallback logic below

                if "Raw" in mode or not segments:
                    # Get Duration using CV2
                    cap = cv2.VideoCapture(vid_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0: fps = 24.0  # Safety
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frames / fps
                    cap.release()

                    # Create fixed chunks
                    curr = 0
                    while curr < duration:
                        end = min(curr + chunk_len, duration)
                        if end - curr > 1.0:  # Ignore tiny tails
                            segments.append((curr, end, ""))
                        curr += chunk_len

                # 2. Process Segments
                cap = cv2.VideoCapture(vid_path)
                fps = cap.get(cv2.CAP_PROP_FPS)

                seg_total = len(segments)

                for s_idx, (start, end, text) in enumerate(segments):
                    if self.stop_requested: break

                    # Naming: vidname_p0001 (Sequential for Narrative Mode)
                    seq_id = str(s_idx + 1).zfill(4)
                    file_base = f"{base_name}_p{seq_id}"

                    path_img = os.path.join(out_root, f"{file_base}.jpg")
                    path_aud = os.path.join(out_root, f"{file_base}.wav")
                    path_txt = os.path.join(out_root, f"{file_base}.txt")

                    if skip and os.path.exists(path_img) and os.path.exists(path_aud):
                        continue

                    # A. Extract Visual (Midpoint Frame)
                    midpoint = start + (end - start) / 2
                    frame_num = int(midpoint * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()

                    if ret:
                        # Resize
                        h, w = frame.shape[:2]
                        s_fac = target_dim / min(h, w)
                        nh, nw = int(h * s_fac), int(w * s_fac)
                        frame = cv2.resize(frame, (nw, nh))
                        cv2.imwrite(path_img, frame)
                    else:
                        continue  # Skip malformed segments

                    # B. Extract Audio (FFmpeg)
                    dur = end - start
                    cmd = [
                        "ffmpeg", "-y", "-v", "error",
                        "-ss", str(start),
                        "-t", str(dur),
                        "-i", vid_path,
                        "-ac", "1", "-ar", "24000",  # Ribosome Standard
                        path_aud
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    # C. Save Text (Subtitle)
                    if text:
                        with open(path_txt, 'w', encoding='utf-8') as f:
                            f.write(text)

                    # Update Bar
                    pct = ((s_idx / seg_total) * 100)
                    self.update_queue.put(lambda v=pct: self.progress.configure(value=v))

                cap.release()
                self._update_tree(vid_path, "Status", f"Done ({seg_total})")

            except Exception as e:
                self._update_tree(vid_path, "Status", "Failed")
                self._log(f"Error: {e}", "ERROR")
                traceback.print_exc()

        self.is_processing = False
        self.update_queue.put(lambda: self.btn_run.config(text="START PROCESSING"))
        self._log("Factory Batch Complete.", "SUCCESS")
        self.update_queue.put(lambda: self.progress.configure(value=0))

    def on_theme_change(self):
        pass