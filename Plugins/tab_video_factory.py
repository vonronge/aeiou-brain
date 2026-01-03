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
import asyncio
from datetime import datetime
import cv2
import numpy as np
import torch
import torchaudio
from PIL import Image
import subprocess

# --- ROBUST IMPORTS ---
try:
    # MoviePy handles audio extraction safely on Windows
    from moviepy.editor import AudioFileClip

    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
    print(" ! Install: pip install moviepy")

try:
    # We use imageio's ffmpeg binary because we know it exists if moviepy is installed
    import imageio_ffmpeg

    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = "ffmpeg"  # Hope it's in PATH

try:
    from pysrt import SubRipFile

    HAS_PYSRT = True
except ImportError:
    HAS_PYSRT = False
    print(" ! Install: pip install pysrt")


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Video Timeline Factory"
        self.factory_running = False
        self.factory_stop = False
        self.factory_queue = []
        self.folder_path = tk.StringVar(value="D:/Training_Data")
        self.skip_existing = tk.BooleanVar(value=True)
        self.auto_scroll = tk.BooleanVar(value=True)

        # Slicing Configuration
        self.target_fps = tk.DoubleVar(value=2.0)
        self.audio_seconds = tk.DoubleVar(value=2.0)

        self._setup_ui()

    def _setup_ui(self):
        top = ttk.LabelFrame(self.parent, text="Video Timeline Production (v3 - Embedded Subs)", padding=15)
        top.pack(fill="x", padx=20, pady=10)

        row1 = ttk.Frame(top)
        row1.pack(fill="x", pady=5)
        ttk.Entry(row1, textvariable=self.folder_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row1, text="📂 Choose Folder", command=self._browse).pack(side="left", padx=2)
        ttk.Button(row1, text="🔍 Scan Videos", command=self._scan_videos).pack(side="left", padx=2)

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=5)
        ttk.Label(row2, text="Extract FPS:").pack(side="left")
        ttk.Spinbox(row2, from_=0.5, to=30.0, increment=0.5, textvariable=self.target_fps, width=5).pack(side="left",
                                                                                                         padx=5)

        ttk.Label(row2, text="| Audio Context (Sec):").pack(side="left", padx=10)
        ttk.Spinbox(row2, from_=0.5, to=5.0, increment=0.5, textvariable=self.audio_seconds, width=5).pack(side="left",
                                                                                                           padx=5)

        row3 = ttk.Frame(top)
        row3.pack(fill="x", pady=5)
        self.btn_start = ttk.Button(row3, text="START PRODUCTION", command=self._start_factory, state="disabled")
        self.btn_start.pack(side="left", fill="x", expand=True, padx=2)
        ttk.Checkbutton(row3, text="Skip Existing Slices", variable=self.skip_existing).pack(side="right", padx=10)

        log_frame = ttk.LabelFrame(self.parent, text="Factory Log", padding=15)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)

        tool_fr = ttk.Frame(log_frame)
        tool_fr.pack(fill="x")
        ttk.Checkbutton(tool_fr, text="Autoscroll", variable=self.auto_scroll).pack(side="right")

        self.log_box = tk.Text(log_frame, font=("Consolas", 9), bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"], borderwidth=0)
        self.log_box.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self.log_box.tag_config("info", foreground=self.app.colors["ACCENT"])
        self.log_box.tag_config("success", foreground=self.app.colors["SUCCESS"])
        self.log_box.tag_config("error", foreground=self.app.colors["ERROR"])
        self.log_box.tag_config("warn", foreground=self.app.colors["WARN"])

    def _browse(self):
        d = filedialog.askdirectory()
        if d: self.folder_path.set(d)

    def _log(self, msg, tag="info"):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n", tag)
        if self.auto_scroll.get(): self.log_box.see(tk.END)

    def _scan_videos(self):
        folder = self.folder_path.get()
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Folder not found")
            return

        videos = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                    videos.append(os.path.join(root, f))

        self.factory_queue = videos
        self._log(f"Found {len(videos)} video files.", "success")
        if videos: self.btn_start.config(state="normal")

    def _start_factory(self):
        if self.factory_running:
            self.factory_stop = True
            self.btn_start.config(text="STOPPING...")
            return

        if not self.factory_queue: self._scan_videos()
        if not self.factory_queue: return

        if not HAS_MOVIEPY:
            self._log("ERROR: 'moviepy' not installed. Audio extraction will fail.", "error")
            return

        self.factory_running = True
        self.factory_stop = False
        self.btn_start.config(text="STOP PRODUCTION")
        threading.Thread(target=self._factory_worker, daemon=True).start()

    def _factory_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_pipeline())
        loop.close()

        self.factory_running = False
        self.parent.after(0, lambda: self.btn_start.config(text="START PRODUCTION"))
        self.parent.after(0, lambda: self._log("Production Run Complete.", "success"))

    async def _async_pipeline(self):
        semaphore = asyncio.Semaphore(1)
        for i, video_path in enumerate(self.factory_queue):
            if self.factory_stop: break
            name = os.path.basename(video_path)
            self.parent.after(0, lambda m=f"Processing {i + 1}/{len(self.factory_queue)}: {name}": self._log(m))
            await self._process_video(video_path, semaphore)

    async def _process_video(self, video_path, semaphore):
        async with semaphore:
            name = os.path.basename(video_path)
            temp_wav = os.path.join(os.path.dirname(video_path), "temp_extract_audio.wav")
            temp_srt = os.path.join(os.path.dirname(video_path), "temp_extract_subs.srt")

            try:
                # 1. Setup Output
                base_name = os.path.splitext(name)[0]
                output_dir = os.path.join(os.path.dirname(video_path), f"{base_name}_timeline")
                if not os.path.exists(output_dir): os.makedirs(output_dir)

                # --- 2. SUBTITLE EXTRACTION LOGIC ---
                subs = None
                sidecar_srt = os.path.splitext(video_path)[0] + ".srt"

                # A. Try Sidecar
                if os.path.exists(sidecar_srt):
                    try:
                        subs = SubRipFile.open(sidecar_srt, encoding='utf-8')
                        self.parent.after(0, lambda: self._log(f" > Found Sidecar Subtitles.", "info"))
                    except:
                        pass

                # B. Try Embedded (FFMPEG Rip)
                if not subs:
                    try:
                        # -map 0:s:0 selects the first subtitle stream
                        cmd = [FFMPEG_EXE, '-i', video_path, '-map', '0:s:0', temp_srt, '-y']
                        # Run quietly
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)

                        if os.path.exists(temp_srt) and os.path.getsize(temp_srt) > 0:
                            subs = SubRipFile.open(temp_srt, encoding='utf-8')
                            self.parent.after(0, lambda: self._log(f" > Extracted Embedded Subtitles.", "info"))
                        else:
                            self.parent.after(0, lambda: self._log(f" > No text track found.", "warn"))
                    except Exception as e:
                        self.parent.after(0, lambda: self._log(f" > Sub Extraction failed: {e}", "warn"))

                # --- 3. AUDIO EXTRACTION LOGIC ---
                self.parent.after(0, lambda: self._log(f" > Extracting full audio track...", "info"))
                try:
                    audioclip = AudioFileClip(video_path)
                    audioclip.write_audiofile(temp_wav, fps=24000, nbytes=2, verbose=False, logger=None)
                    audioclip.close()
                    waveform, sr = torchaudio.load(temp_wav)
                except Exception as ae:
                    self.parent.after(0, lambda m=str(ae): self._log(f" > Audio Extract Failed: {m}", "error"))
                    cap_check = cv2.VideoCapture(video_path)
                    fps_check = cap_check.get(cv2.CAP_PROP_FPS)
                    frames_check = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap_check.release()
                    waveform = torch.zeros(1, int(frames_check / fps_check * 24000) if fps_check > 0 else 24000)
                    sr = 24000

                # --- 4. SLICING LOOP ---
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 24.0

                extract_rate = self.target_fps.get()
                frame_interval = max(1, int(fps / extract_rate))
                audio_window = self.audio_seconds.get()

                frame_cursor = 0
                slice_count = 0

                while cap.isOpened():
                    if self.factory_stop: break
                    ret, frame = cap.read()
                    if not ret: break

                    if frame_cursor % frame_interval == 0:
                        timestamp_sec = frame_cursor / fps
                        slice_id = slice_count + 1
                        out_base = os.path.join(output_dir, f"{base_name}_p{slice_id:05d}")

                        # IMAGE
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb).resize((256, 256))
                        img.save(f"{out_base}.png")

                        # AUDIO SLICE
                        center_sample = int(timestamp_sec * sr)
                        half_window = int((audio_window / 2) * sr)
                        start_s = max(0, center_sample - half_window)
                        end_s = min(waveform.shape[1], center_sample + half_window)
                        clip = waveform[:, start_s:end_s]

                        target_samples = int(audio_window * sr)
                        if clip.shape[1] < target_samples:
                            pad = target_samples - clip.shape[1]
                            clip = torch.cat([clip, torch.zeros(clip.shape[0], pad)], dim=1)

                        torchaudio.save(f"{out_base}.wav", clip, sr)

                        # TEXT SYNC (Robust PySRT check)
                        txt_out = ""
                        if subs:
                            # pysrt uses ordinal (ms) objects
                            # We check if current timestamp is within any sub's duration
                            current_ms = timestamp_sec * 1000
                            # Simple linear search (for short videos this is fine, for movies binary search is better but complex)
                            # Optimizing: search only recent subs
                            matches = []
                            for sub in subs:
                                if sub.start.ordinal <= current_ms <= sub.end.ordinal:
                                    matches.append(sub.text)
                                elif sub.start.ordinal > current_ms:
                                    break  # Sorted assumption

                            if matches:
                                txt_out = " ".join(matches).replace('\n', ' ')

                        with open(f"{out_base}.txt", 'w', encoding='utf-8') as f:
                            f.write(txt_out)

                        # CONTROL
                        with open(f"{out_base}.json", 'w', encoding='utf-8') as f:
                            f.write("{}")

                        slice_count += 1
                        if slice_count % 10 == 0:
                            self.parent.after(0, lambda n=slice_count, t=timestamp_sec:
                            self._log(f" > Slice {n} @ {t:.1f}s"))

                    frame_cursor += 1

                cap.release()

                # Cleanup
                for tmp in [temp_wav, temp_srt]:
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except:
                            pass

                self.parent.after(0, lambda: self._log(f"Completed {name}: {slice_count} slices.", "success"))

            except Exception as e:
                self.parent.after(0, lambda m=str(e): self._log(f"CRITICAL ERROR {name}: {m}", "error"))
                import traceback
                traceback.print_exc()

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])