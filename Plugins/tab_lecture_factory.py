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
import fitz  # PyMuPDF
import os
import threading
import re
import queue
import random
import time
import asyncio
import traceback

# --- NEURAL TTS ENGINE ---
try:
    import edge_tts

    HAS_EDGE = True
except ImportError:
    HAS_EDGE = False

# Curated high-quality voices for variety
NEURAL_VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural",
    "en-US-ChristopherNeural", "en-US-EricNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-IE-EmilyNeural", "en-IE-ConnorNeural"
]


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Lecture Factory"

        self.is_processing = False
        self.is_paused = False
        self.stop_requested = False
        self.pdf_queue = []
        self.update_queue = queue.Queue()

        # --- SETTINGS ---
        default_root = self.app.paths.get("data", os.path.abspath("Training_Data"))
        default_out = os.path.join(default_root, "Lectures")

        self.input_folder = tk.StringVar(value=default_root)
        self.output_folder = tk.StringVar(value=default_out)
        self.target_width = tk.IntVar(value=1024)
        self.generate_audio = tk.BooleanVar(value=True)
        self.random_voice = tk.BooleanVar(value=True)
        self.selected_voice = tk.StringVar(value="en-US-GuyNeural")  # Default if not random

        self._setup_ui()
        if not HAS_EDGE:
            self._log("⚠️ 'edge-tts' library missing. Install via pip.", "ERROR")

        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return
        scale = getattr(self.app, 'ui_scale', 1.0)

        # 1. PATHS
        fr_io = ttk.LabelFrame(self.parent, text="Library Configuration", padding=10)
        fr_io.pack(fill="x", padx=10, pady=5)

        r1 = ttk.Frame(fr_io)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Source Folder:", width=15).pack(side="left")
        ttk.Entry(r1, textvariable=self.input_folder).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂", width=4, command=lambda: self._browse(self.input_folder)).pack(side="left")

        r2 = ttk.Frame(fr_io)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Output Root:", width=15).pack(side="left")
        ttk.Entry(r2, textvariable=self.output_folder).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r2, text="📂", width=4, command=lambda: self._browse(self.output_folder)).pack(side="left")

        # 2. SETTINGS
        fr_scan = ttk.Frame(self.parent)
        fr_scan.pack(fill="both", expand=True, padx=10, pady=5)

        ctrl_row = ttk.Frame(fr_scan)
        ctrl_row.pack(fill="x", pady=5)

        ttk.Button(ctrl_row, text="1. SCAN FOR PDFS", command=self._scan_folder).pack(side="left", fill="x",
                                                                                      expand=True, padx=(0, 5))

        ttk.Label(ctrl_row, text="Target Width (px):").pack(side="left")
        ttk.Spinbox(ctrl_row, from_=512, to=2048, increment=128, textvariable=self.target_width, width=5).pack(
            side="left", padx=5)

        ttk.Checkbutton(ctrl_row, text="Neural Audio", variable=self.generate_audio).pack(side="left", padx=10)
        ttk.Checkbutton(ctrl_row, text="Randomize Voices", variable=self.random_voice).pack(side="left", padx=5)

        # Treeview
        style = ttk.Style()
        row_h = int(25 * scale)
        style.configure("Lecture.Treeview", rowheight=row_h, font=("Segoe UI", int(10 * scale)))
        style.configure("Lecture.Treeview.Heading", font=("Segoe UI", int(11 * scale), "bold"))

        cols = ("Filename", "Pages", "Status")
        self.tree = ttk.Treeview(fr_scan, columns=cols, show="headings", height=8, style="Lecture.Treeview")
        self.tree.heading("Filename", text="Filename")
        self.tree.heading("Pages", text="Pages")
        self.tree.heading("Status", text="Status")
        self.tree.column("Filename", width=int(400 * scale))
        self.tree.column("Pages", width=int(80 * scale), anchor="center")
        self.tree.column("Status", width=int(150 * scale), anchor="center")

        sb = ttk.Scrollbar(fr_scan, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 3. ACTIONS
        fr_run = ttk.Frame(self.parent, padding=10)
        fr_run.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(fr_run, text="▶ START FACTORY", command=self._start_processing, state="disabled")
        self.btn_run.pack(side="left", fill="x", expand=True, padx=5)
        self.btn_pause = ttk.Button(fr_run, text="⏸ PAUSE", command=self._toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", padx=5)
        self.btn_stop = ttk.Button(fr_run, text="⏹ STOP", command=self._stop_processing, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(fr_run, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=5)

        self.log_lbl = ttk.Label(self.parent, text="Ready.", foreground=self.app.colors["FG_DIM"], anchor="w")
        self.log_lbl.pack(fill="x", padx=15, pady=(0, 10))

    # --- ASYNC BRIDGE ---
    def _run_edge_tts(self, text, voice, rate, out_path):
        """Runs the async Edge-TTS coroutine in a synchronous wrapper."""

        async def _gen():
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(out_path)

        try:
            # Create a fresh loop for this thread if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_gen())
            loop.close()
            return True
        except Exception as e:
            self._log(f"TTS Error: {e}", "ERROR")
            return False

    # --- HELPERS ---
    def _browse(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d: var.set(d)

    def _log(self, msg, tag="INFO"):
        self.update_queue.put(lambda: self.log_lbl.config(text=msg))
        if self.app.golgi and "Page" not in msg:
            level = "INFO"
            if tag == "SUCCESS":
                level = "SUCCESS"
            elif tag == "ERROR":
                level = "ERROR"
            self.app.golgi._dispatch(level, msg, source="LectureFactory")

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                fn = self.update_queue.get_nowait()
                fn()
            except:
                break
        if self.parent: self.parent.after(100, self._process_gui_queue)

    def _update_tree(self, iid, col, val):
        self.update_queue.put(lambda: self.tree.set(iid, col, val))

    def _update_progress(self, val):
        self.update_queue.put(lambda: self.progress.configure(value=val))

    # --- ACTIONS ---
    def _scan_folder(self):
        folder = self.input_folder.get()
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Source folder not found.")
            return

        for item in self.tree.get_children(): self.tree.delete(item)
        self.pdf_queue = []
        count = 0

        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    full_path = os.path.join(root, f)
                    try:
                        doc = fitz.open(full_path)
                        pages = len(doc)
                        doc.close()
                        self.tree.insert("", "end", iid=full_path, values=(f, pages, "Queued"))
                        self.pdf_queue.append(full_path)
                        count += 1
                    except:
                        pass

        self._log(f"Found {count} PDFs.", "SUCCESS")
        if count > 0: self.btn_run.config(state="normal")

    def _start_processing(self):
        if not self.pdf_queue: return
        self.is_processing = True
        self.stop_requested = False
        self.is_paused = False

        self.btn_run.config(state="disabled")
        self.btn_pause.config(state="normal", text="⏸ PAUSE")
        self.btn_stop.config(state="normal")

        threading.Thread(target=self._worker, daemon=True).start()

    def _toggle_pause(self):
        if self.is_paused:
            self.is_paused = False
            self.btn_pause.config(text="⏸ PAUSE")
            self._log("Resuming...")
        else:
            self.is_paused = True
            self.btn_pause.config(text="▶ RESUME")
            self._log("Paused.")

    def _stop_processing(self):
        self.stop_requested = True
        self.btn_stop.config(text="STOPPING...")

    # --- WORKER ---
    def _worker(self):
        out_root = self.output_folder.get()
        if not os.path.exists(out_root):
            try:
                os.makedirs(out_root)
            except:
                pass

        if self.generate_audio.get() and not HAS_EDGE:
            self._log("Skipping Audio (Edge-TTS not installed).", "ERROR")

        total_files = len(self.pdf_queue)
        target_w = self.target_width.get()

        for idx, pdf_path in enumerate(self.pdf_queue):
            if self.stop_requested: break

            while self.is_paused:
                if self.stop_requested: break
                time.sleep(0.5)

            filename = os.path.basename(pdf_path)
            book_name = os.path.splitext(filename)[0]
            safe_book_name = re.sub(r'[^a-zA-Z0-9]', '_', book_name)
            book_dir = os.path.join(out_root, f"{safe_book_name}_lecture")
            if not os.path.exists(book_dir): os.makedirs(book_dir)

            self._update_tree(pdf_path, "Status", "Processing...")
            self._log(f"Processing {filename} ({idx + 1}/{total_files})...")

            # Voice Selection for this Book
            voice = self.selected_voice.get()
            rate = "+0%"
            if self.random_voice.get():
                voice = random.choice(NEURAL_VOICES)
                # Slight speed jitter for natural variety
                r_val = random.randint(-5, 10)
                rate = f"{r_val:+d}%"

            try:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)

                for i, page in enumerate(doc):
                    if self.stop_requested: break
                    while self.is_paused: time.sleep(0.1)

                    fname = f"{safe_book_name}_p{str(i + 1).zfill(4)}"
                    base_path = os.path.join(book_dir, fname)

                    # Implicit "Skip Done"
                    if os.path.exists(base_path + ".txt") and os.path.exists(base_path + ".png"):
                        continue

                    # 1. SMART VISUAL
                    zoom = target_w / page.rect.width
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    pix.save(base_path + ".png")

                    # 2. TEXT
                    text = page.get_text("text").strip()
                    if len(text) < 10: text = "Visual content."
                    with open(base_path + ".txt", "w", encoding="utf-8") as f:
                        f.write(text)

                    # 3. NEURAL AUDIO
                    if HAS_EDGE and self.generate_audio.get() and len(text) > 5:
                        clean_text = " ".join(text.split())
                        # Remove citations [1], URLs
                        clean_text = re.sub(r'\[\d+\]|http\S+', '', clean_text)

                        # Async Call (Blocking this thread only)
                        self._run_edge_tts(clean_text, voice, rate, base_path + ".wav")

                    # Pulse UI
                    if i % 2 == 0:
                        pct = ((i + 1) / total_pages) * 100
                        self._update_progress(pct)
                        msg = f"Reading {filename} (pg {i + 1}) - Voice: {voice.split('-')[2]}"
                        self.parent.after(0, lambda m=msg: self.log_lbl.config(text=m))

                doc.close()
                self._update_tree(pdf_path, "Status", "Done")

            except Exception as e:
                self._log(f"Error on {filename}: {e}", "ERROR")
                self._update_tree(pdf_path, "Status", "Failed")
                traceback.print_exc()

        self.is_processing = False
        self.update_queue.put(lambda: self.btn_run.config(state="normal"))
        self.update_queue.put(lambda: self.btn_pause.config(state="disabled"))
        self.update_queue.put(lambda: self.btn_stop.config(state="disabled", text="⏹ STOP"))
        self._update_progress(0)

        if not self.stop_requested:
            self._log("Factory Run Complete.", "SUCCESS")
        else:
            self._log("Factory Stopped.", "INFO")

    def on_theme_change(self):
        pass