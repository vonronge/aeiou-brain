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
import threading
import os
from datetime import datetime
import cv2
import numpy as np
import torch
import torchaudio
from PIL import Image
import subprocess
import json
import traceback

# --- DEPENDENCY CHECK ---
try:
    from moviepy.editor import AudioFileClip

    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
    print(" ! VideoFactory: Install 'moviepy' for audio extraction.")

try:
    import imageio_ffmpeg

    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = "ffmpeg"

try:
    from pysrt import SubRipFile

    HAS_PYSRT = True
except ImportError:
    HAS_PYSRT = False
    print(" ! VideoFactory: Install 'pysrt' for subtitle syncing.")


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Video Factory"

        # State
        self.is_running = False
        self.stop_requested = False
        self.queue = []

        # Paths
        default_data = self.app.paths.get("data", "D:/Training_Data")
        self.source_path = tk.StringVar(value=default_data)

        # Settings
        self.fps_var = tk.DoubleVar(value=2.0)  # Extract 2 frames per second
        self.audio_len_var = tk.DoubleVar(value=2.0)  # 2 seconds of audio per frame
        self.skip_existing = tk.BooleanVar(value=True)
        self.auto_scroll = tk.BooleanVar(value=True)
        self.gen_physics = tk.BooleanVar(value=True)  # Generate Optical Flow (Control)

        self._setup_ui()

    def _setup_ui(self):
        # 1. HEADER & CONTROLS
        top = ttk.LabelFrame(self.parent, text="Timeline Slicer Configuration", padding=15)
        top.pack(fill="x", padx=10, pady=10)

        # Path Row
        r1 = ttk.Frame(top)
        r1.pack(fill="x", pady=5)
        ttk.Label(r1, text="Source Folder:", width=12).pack(side="left")
        ttk.Entry(r1, textvariable=self.source_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂", width=4, command=self._browse).pack(side="left")
        ttk.Button(r1, text="SCAN", command=self._scan).pack(side="left", padx=(5, 0))

        # Config Row
        r2 = ttk.Frame(top)
        r2.pack(fill="x", pady=10)

        ttk.Label(r2, text="Extraction FPS:").pack(side="left")
        ttk.Spinbox(r2, from_=0.1, to=30.0, increment=0.5, textvariable=self.fps_var, width=5).pack(side="left", padx=5)

        ttk.Label(r2, text="Audio Window (s):").pack(side="left", padx=(15, 0))
        ttk.Spinbox(r2, from_=0.1, to=10.0, increment=0.5, textvariable=self.audio_len_var, width=5).pack(side="left",
                                                                                                          padx=5)

        ttk.Checkbutton(r2, text="Calc Optical Flow (Physics)", variable=self.gen_physics).pack(side="left", padx=20)
        ttk.Checkbutton(r2, text="Skip Existing", variable=self.skip_existing).pack(side="left", padx=5)

        # 2. ACTION BUTTON
        self.btn_run = ttk.Button(top, text="START PRODUCTION LINE", command=self._toggle_run, state="disabled")
        self.btn_run.pack(fill="x", pady=(10, 0))

        # 3. LOG OUTPUT
        log_fr = ttk.LabelFrame(self.parent, text="Factory Telemetry", padding=10)
        log_fr.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_box = tk.Text(log_fr, font=("Consolas", int(9 * getattr(self.app, 'ui_scale', 1.0))),
                               bg=self.app.colors["BG_MAIN"], fg=self.app.colors["FG_TEXT"], borderwidth=0)
        self.log_box.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(log_fr, command=self.log_box.yview)
        sb.pack(side="right", fill="y")
        self.log_box.config(yscrollcommand=sb.set)

        # Tags
        self.log_box.tag_config("info", foreground=self.app.colors["FG_DIM"])
        self.log_box.tag_config("proc", foreground=self.app.colors["ACCENT"])
        self.log_box.tag_config("success", foreground=self.app.colors["SUCCESS"])
        self.log_box.tag_config("warn", foreground=self.app.colors["WARN"])
        self.log_box.tag_config("err", foreground=self.app.colors["ERROR"])

    def _browse(self):
        d = filedialog.askdirectory()
        if d: self.source_path.set(d)

    def _log(self, msg, tag="info"):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n", tag)
        if self.auto_scroll.get(): self.log_box.see(tk.END)

    def _scan(self):
        folder = self.source_path.get()
        if not os.path.exists(folder): return

        self.queue = []
        valid_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}

        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    self.queue.append(os.path.join(root, f))

        self._log(f"Scanner found {len(self.queue)} video files.", "success")
        if self.queue:
            self.btn_run.config(state="normal")

    def _toggle_run(self):
        if self.is_running:
            self.stop_requested = True
            self.btn_run.config(text="STOPPING...")
        else:
            if not self.queue: self._scan()
            if not self.queue: return

            if not HAS_MOVIEPY:
                messagebox.showerror("Dependency Error", "Please install 'moviepy' to slice audio.")
                return

            self.is_running = True
            self.stop_requested = False
            self.btn_run.config(text="HALT FACTORY")
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            total_vids = len(self.queue)

            for i, vid_path in enumerate(self.queue):
                if self.stop_requested: break

                vid_name = os.path.basename(vid_path)
                base_name = os.path.splitext(vid_name)[0]
                parent_dir = os.path.dirname(vid_path)

                # Output folder: adjacent to video, named "VideoName_timeline"
                out_dir = os.path.join(parent_dir, f"{base_name}_timeline")
                if not os.path.exists(out_dir): os.makedirs(out_dir)

                self.parent.after(0, lambda m=f"Processing {i + 1}/{total_vids}: {vid_name}": self._log(m, "proc"))

                # --- PRE-PROCESSING ---

                # 1. Audio Extraction (Full track to RAM/Temp)
                temp_wav = os.path.join(parent_dir, "_temp_audio.wav")
                has_audio = False
                try:
                    clip = AudioFileClip(vid_path)
                    clip.write_audiofile(temp_wav, fps=24000, nbytes=2, verbose=False, logger=None)
                    clip.close()
                    waveform, sr = torchaudio.load(temp_wav)
                    has_audio = True
                except Exception as e:
                    self.parent.after(0, lambda: self._log(f"Audio extraction failed (silent mode): {e}", "warn"))
                    waveform = None
                    sr = 24000

                # 2. Subtitle Extraction
                subs = None
                sidecar = os.path.splitext(vid_path)[0] + ".srt"
                if os.path.exists(sidecar) and HAS_PYSRT:
                    try:
                        subs = SubRipFile.open(sidecar, encoding='utf-8')
                        self.parent.after(0, lambda: self._log("Loaded sidecar subtitles.", "info"))
                    except:
                        pass

                # --- SLICING LOOP ---
                cap = cv2.VideoCapture(vid_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 24.0

                target_interval = int(fps / self.fps_var.get())
                if target_interval < 1: target_interval = 1

                frame_idx = 0
                slice_idx = 0
                prev_gray = None  # For Optical Flow

                while cap.isOpened():
                    if self.stop_requested: break

                    ret, frame = cap.read()
                    if not ret: break

                    # Optical Flow Calculation (Physics/Context Channel)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    control_vec = [0.0, 0.0]

                    if self.gen_physics.get() and prev_gray is not None:
                        # Farneback Dense Flow
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        # Average motion vector for the whole frame (Simplified Physics)
                        avg_flow = np.mean(flow, axis=(0, 1))
                        # Normalize/Scale for embedding
                        control_vec = [float(avg_flow[0]) / 5.0, float(avg_flow[1]) / 5.0]

                    # Capture Slice
                    if frame_idx % target_interval == 0:
                        timestamp = frame_idx / fps

                        # File Naming (Narrative Order)
                        # e.g., vid_name_p0001.png
                        fname_base = f"{base_name}_p{slice_idx:05d}"
                        out_path_base = os.path.join(out_dir, fname_base)

                        if self.skip_existing.get() and os.path.exists(f"{out_path_base}.png"):
                            pass  # Skip saving, but update state
                        else:
                            # A. SAVE VISUAL (V)
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(rgb).resize((256, 256))
                            img.save(f"{out_path_base}.png")

                            # B. SAVE AUDIO (A)
                            if has_audio:
                                center_sample = int(timestamp * sr)
                                half_win = int((self.audio_len_var.get() / 2) * sr)
                                start = max(0, center_sample - half_win)
                                end = min(waveform.shape[1], center_sample + half_win)

                                audio_slice = waveform[:, start:end]
                                # Pad if too short (at edges)
                                target_len = int(self.audio_len_var.get() * sr)
                                if audio_slice.shape[1] < target_len:
                                    pad = target_len - audio_slice.shape[1]
                                    audio_slice = torch.cat([audio_slice, torch.zeros(audio_slice.shape[0], pad)],
                                                            dim=1)

                                torchaudio.save(f"{out_path_base}.wav", audio_slice, sr)

                            # C. SAVE TEXT (T)
                            txt_content = ""
                            if subs:
                                cur_ms = timestamp * 1000
                                # Find active sub
                                # Simple linear search (can be optimized, but fine for factory speed)
                                for sub in subs:
                                    if sub.start.ordinal <= cur_ms <= sub.end.ordinal:
                                        txt_content = sub.text.replace('\n', ' ')
                                        break

                            with open(f"{out_path_base}.txt", "w", encoding='utf-8') as f:
                                f.write(txt_content)

                            # D. SAVE CONTROL/PHYSICS (C)
                            # Future-proofing: Saving as JSON for flexible metadata loading
                            c_data = {
                                "control_vec": control_vec,
                                "timestamp": timestamp,
                                "source": vid_name
                            }
                            with open(f"{out_path_base}.json", "w", encoding='utf-8') as f:
                                json.dump(c_data, f)

                        slice_idx += 1
                        if slice_idx % 50 == 0:
                            self.parent.after(0, lambda c=slice_idx: self._log(f" -> Generated {c} engrams...", "info"))

                    # Update state
                    prev_gray = gray
                    frame_idx += 1

                cap.release()

                # Cleanup temp audio
                if os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                    except:
                        pass

                self.parent.after(0, lambda n=vid_name, c=slice_idx: self._log(f"Completed {n}: {c} Quadruplets.",
                                                                               "success"))

        except Exception as e:
            traceback.print_exc()
            self.parent.after(0, lambda err=str(e): self._log(f"CRITICAL ERROR: {err}", "err"))

        finally:
            self.is_running = False
            self.parent.after(0, lambda: self.btn_run.config(text="START PRODUCTION LINE"))

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'):
            self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])