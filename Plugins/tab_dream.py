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
from tkinter import ttk, filedialog
import threading
import os
import time
import torch
import torch.nn.functional as F
from torch.amp import autocast
from datetime import datetime
import random
import json
import uuid
from PIL import ImageTk, Image

try:
    import pygame

    pygame.mixer.init()
    HAS_AUDIO = True
except:
    HAS_AUDIO = False

try:
    import cv2

    HAS_OPENCV = True
except:
    HAS_OPENCV = False


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Dream State"

        self.is_dreaming = False
        self.stop_requested = False
        self.noise_folder = tk.StringVar(value="D:/Training_Data")
        self.image_buffer = []
        self.audio_buffer = []
        self.video_buffer = []

        self._setup_ui()

    def _setup_ui(self):
        config = ttk.LabelFrame(self.parent, text="REM Cycle Configuration", padding=15)
        config.pack(fill="x", padx=20, pady=10)

        row1 = ttk.Frame(config);
        row1.pack(fill="x", pady=5)
        ttk.Label(row1, text="Diverse Data Source:").pack(side="left")
        ttk.Entry(row1, textvariable=self.noise_folder).pack(side="left", fill="x", expand=True, padx=10)
        ttk.Button(row1, text="Browse", command=self._browse).pack(side="left")

        ctrl = ttk.Frame(self.parent)
        ctrl.pack(fill="x", padx=20, pady=10)
        self.btn_dream = ttk.Button(ctrl, text="INITIATE SLEEP MODE", command=self._toggle_dream)
        self.btn_dream.pack(side="left", fill="x", expand=True)

        self.vis_frame = ttk.LabelFrame(self.parent, text="Dream Visualizer", padding=10)
        self.vis_frame.pack(fill="both", expand=True, padx=20, pady=5)

        self.stream_box = tk.Text(self.vis_frame, font=("Consolas", 10), bg="#000000", fg="#00FF00", height=8)
        self.stream_box.pack(fill="x", side="top")
        self.stream_box.tag_config("mem", foreground="#A8C7FA")
        self.stream_box.tag_config("sys", foreground="#8e9198")
        self.stream_box.tag_config("vid", foreground="#FF69B4")

        self.img_canvas = tk.Canvas(self.vis_frame, bg="#111", height=256)
        self.img_canvas.pack(fill="both", expand=True, side="bottom")

    def _browse(self):
        d = filedialog.askdirectory()
        if d: self.noise_folder.set(d)

    def _toggle_dream(self):
        if self.is_dreaming:
            self.stop_requested = True
            self.btn_dream.config(text="WAKING UP...", state="disabled")
        else:
            self.is_dreaming = True
            self.stop_requested = False
            self.btn_dream.config(text="WAKE UP (STOP)")
            threading.Thread(target=self._dream_loop, daemon=True).start()
            # Start the memory consolidation worker in parallel
            threading.Thread(target=self._consolidation_worker, daemon=True).start()

    def _consolidation_worker(self):
        """
        Background process that reads recent episodic logs and cements them
        into the Hippocampus vector index.
        """
        mem_file = os.path.join(self.app.paths["memories"], "episodic_chat_log.jsonl")

        while not self.stop_requested:
            if os.path.exists(mem_file):
                # Read last few lines
                lines = []
                try:
                    with open(mem_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()[-10:]  # Process batch of 10 recent
                except:
                    pass

                for line in lines:
                    try:
                        entry = json.loads(line)
                        conv = entry.get("full_text", "")
                        img_path = entry.get("image_path", "")

                        # Extract Engram
                        if img_path and os.path.exists(img_path):
                            v_vec = self.app.ribosome.get_engram(img_path)
                            if v_vec is not None:
                                self.app.hippocampus.consolidate(v_vec, None, conv)
                    except:
                        pass

            time.sleep(10)  # Run every 10 seconds

    def _dream_loop(self):
        brain = self.app.lobes[self.app.active_lobe.get()]
        if not brain: return
        brain.eval()
        ribosome = self.app.ribosome

        self._log("Entering REM Sleep...", "sys")
        files = self._scan_noise_files()

        while not self.stop_requested:
            if not files: break
            f_path = random.choice(files)

            # Ingest
            packet = {'t': f_path}
            base = os.path.splitext(f_path)[0]
            if os.path.exists(base + ".png"): packet['v'] = base + ".png"
            if os.path.exists(base + ".mp4"): packet['vid'] = base + ".mp4"

            v, a, t, c, _ = ribosome.ingest_packet(packet)
            if t is None: continue

            input_t = t[:, :-1]
            labels = t[:, 1:]

            # Forward (Locked)
            loss_val = 0.0
            with self.app.gpu_lock:
                with torch.no_grad():
                    with autocast('cuda'):
                        try:
                            logits, _, _ = brain(v, a, input_t, c)
                        except:
                            logits, _, _ = brain(v, a, input_t)

                        # Calc Loss for "Surprise" metric
                        offset = (v.shape[1] if v is not None else 0) + (a.shape[1] if a is not None else 0) + (
                            c.shape[1] if c is not None else 0)
                        logits_text = logits[:, offset: offset + input_t.shape[1], :]
                        loss = F.cross_entropy(logits_text.reshape(-1, logits_text.size(-1)), labels.reshape(-1),
                                               ignore_index=50256)
                        loss_val = loss.item()

            # Visualize
            self.video_buffer = []
            has_vid = False

            seq = input_t[0].tolist()
            for token in seq:
                if token >= ribosome.image_vocab_base and token < ribosome.audio_vocab_base:
                    self.video_buffer.append(token)
                    if len(self.video_buffer) == 256:
                        try:
                            pil_img = ribosome.decode_image_tokens(self.video_buffer)
                            self._update_canvas(pil_img)
                        except:
                            pass

                    if len(self.video_buffer) >= 4096 and len(self.video_buffer) % 256 == 0:
                        try:
                            clip = self.video_buffer[-4096:]
                            vid_path = ribosome.decode_video_tokens(clip)
                            if vid_path and HAS_OPENCV:
                                self._play_video_on_canvas(vid_path)
                                has_vid = True
                                self.video_buffer = []
                        except:
                            pass

            tag = "vid" if has_vid else "mem"
            msg = f"Dreamt: {os.path.basename(f_path)} (L:{loss_val:.2f})"
            if has_vid: msg += " [VIDEO]"
            self.parent.after(0, lambda m=msg, t=tag: self._log(m, t))
            time.sleep(0.5)

        self.is_dreaming = False
        self.parent.after(0, lambda: self.btn_dream.config(text="INITIATE SLEEP MODE", state="normal"))

    def _update_canvas(self, pil_img):
        pil_img = pil_img.resize((256, 256))
        self.photo = ImageTk.PhotoImage(pil_img)
        self.parent.after(0, lambda: self.img_canvas.create_image(128, 128, image=self.photo))

    def _play_video_on_canvas(self, vid_path):
        if not HAS_OPENCV: return
        try:
            cap = cv2.VideoCapture(vid_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame).resize((256, 256))
                self._update_canvas(pil_img)
                time.sleep(0.125)
            cap.release()
        except:
            pass

    def _log(self, msg, tag):
        self.stream_box.insert(tk.END, f"{msg}\n", tag)
        self.stream_box.see(tk.END)

    def _scan_noise_files(self):
        path = self.noise_folder.get()
        if not os.path.exists(path): return []
        valid = {'.txt', '.mp4'}
        files = []
        for root, _, fs in os.walk(path):
            for f in fs:
                if os.path.splitext(f)[1].lower() in valid:
                    files.append(os.path.join(root, f))
        random.shuffle(files)
        return files

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'stream_box'): self.stream_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])