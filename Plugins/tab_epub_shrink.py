"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The EPUB Shrinker (v25.4):
Reduces file size of EPUBs by downscaling and compressing internal images.
Essential for optimizing training data storage.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import zipfile
from PIL import Image
import io
import threading
import queue
import time


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "EPUB Shrinker"

        self.is_running = False
        self.stop_requested = False
        self.queue = []
        self.update_queue = queue.Queue()

        # Settings
        self.input_path = tk.StringVar(value=self.app.paths["data"])
        self.target_dim = tk.IntVar(value=1024)  # Max width/height
        self.jpeg_quality = tk.IntVar(value=75)
        self.convert_png_to_jpg = tk.BooleanVar(value=False)  # Risky (transparency), default off

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return
        scale = getattr(self.app, 'ui_scale', 1.0)

        # 1. IO
        fr_io = ttk.LabelFrame(self.parent, text="Select Books", padding=10)
        fr_io.pack(fill="x", padx=10, pady=5)

        r1 = ttk.Frame(fr_io)
        r1.pack(fill="x")
        ttk.Entry(r1, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂 File", width=6, command=lambda: self._browse(False)).pack(side="left", padx=1)
        ttk.Button(r1, text="📂 Folder", width=8, command=lambda: self._browse(True)).pack(side="left", padx=1)

        # 2. SETTINGS
        fr_set = ttk.LabelFrame(self.parent, text="Compression Rules", padding=10)
        fr_set.pack(fill="x", padx=10, pady=5)

        ttk.Label(fr_set, text="Max Dimension (px):").pack(side="left")
        ttk.Spinbox(fr_set, from_=512, to=2048, increment=128, textvariable=self.target_dim, width=5).pack(side="left",
                                                                                                           padx=5)

        ttk.Label(fr_set, text="| JPEG Quality:").pack(side="left", padx=(15, 0))
        ttk.Spinbox(fr_set, from_=10, to=100, increment=5, textvariable=self.jpeg_quality, width=5).pack(side="left",
                                                                                                         padx=5)

        ttk.Checkbutton(fr_set, text="Convert PNG to JPG (Aggressive)", variable=self.convert_png_to_jpg).pack(
            side="right")

        # 3. ACTIONS
        fr_run = ttk.Frame(self.parent, padding=10)
        fr_run.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(fr_run, text="START COMPRESSION", command=self._start)
        self.btn_run.pack(side="left", fill="x", expand=True)

        self.progress = ttk.Progressbar(fr_run, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        # 4. LOG
        self.log_box = tk.Text(self.parent, height=12, font=("Consolas", int(9 * scale)),
                               bg=self.app.colors["BG_MAIN"], fg=self.app.colors["FG_TEXT"])
        self.log_box.pack(fill="both", expand=True, padx=10, pady=5)

    def _browse(self, is_folder):
        if is_folder:
            d = filedialog.askdirectory(initialdir=self.input_path.get())
            if d: self.input_path.set(d)
        else:
            f = filedialog.askopenfilename(filetypes=[("EPUB files", "*.epub")])
            if f: self.input_path.set(f)

    def _log(self, msg):
        self.update_queue.put(lambda: self.log_box.insert(tk.END, f"{msg}\n"))
        self.update_queue.put(lambda: self.log_box.see(tk.END))

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                self.update_queue.get_nowait()()
            except:
                break
        if self.parent: self.parent.after(100, self._process_gui_queue)

    def _start(self):
        if self.is_running:
            self.stop_requested = True
            self.btn_run.config(text="STOPPING...")
            return

        path = self.input_path.get()
        if not os.path.exists(path):
            messagebox.showerror("Error", "Path does not exist.")
            return

        self.queue = []
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(".epub") and "_shrunk" not in f:
                        self.queue.append(os.path.join(root, f))
        else:
            self.queue.append(path)

        if not self.queue:
            self._log("No valid EPUBs found.")
            return

        self.is_running = True
        self.stop_requested = False
        self.btn_run.config(text="STOP")
        self.progress["maximum"] = len(self.queue)

        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        max_d = self.target_dim.get()
        qual = self.jpeg_quality.get()
        conv_png = self.convert_png_to_jpg.get()

        processed = 0
        saved_bytes = 0

        for idx, epub_path in enumerate(self.queue):
            if self.stop_requested: break

            fname = os.path.basename(epub_path)
            self._log(f"[{idx + 1}/{len(self.queue)}] Processing: {fname}")

            out_path = epub_path.replace(".epub", "_shrunk.epub")

            try:
                original_size = os.path.getsize(epub_path)

                with zipfile.ZipFile(epub_path, 'r') as zin:
                    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
                        # 1. Handle mimetype FIRST (Must be uncompressed)
                        if "mimetype" in zin.namelist():
                            zout.writestr("mimetype", zin.read("mimetype"), compress_type=zipfile.ZIP_STORED)

                        # 2. Process other files
                        for item in zin.infolist():
                            if item.filename == "mimetype": continue

                            data = zin.read(item.filename)

                            # Check if image
                            ext = os.path.splitext(item.filename)[1].lower()
                            if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                                try:
                                    img = Image.open(io.BytesIO(data))

                                    # Resize logic
                                    w, h = img.size
                                    if max(w, h) > max_d:
                                        ratio = max_d / max(w, h)
                                        new_size = (int(w * ratio), int(h * ratio))
                                        img = img.resize(new_size, Image.Resampling.LANCZOS)

                                    # Compression
                                    out_buf = io.BytesIO()
                                    target_fmt = img.format if img.format else "JPEG"

                                    # Force JPG if requested and safe
                                    if conv_png and ext == '.png' and img.mode != 'RGBA':
                                        img = img.convert("RGB")
                                        target_fmt = "JPEG"
                                        # Update filename in zip? No, complexity high (manifest rewrite).
                                        # We keep extension but write JPEG bytes if we assume reader handles mime sniffing.
                                        # ACTUALLY: Changing format without changing manifest breaks EPUB.
                                        # Safe fallback: Don't change format, just compress.
                                        target_fmt = "PNG"

                                    if target_fmt in ["JPEG", "JPG"]:
                                        if img.mode != "RGB": img = img.convert("RGB")
                                        img.save(out_buf, format="JPEG", quality=qual, optimize=True)
                                    elif target_fmt == "PNG":
                                        # Optimize PNG (Pillow uses zlib level)
                                        img.save(out_buf, format="PNG", optimize=True)
                                    else:
                                        # WebP etc
                                        img.save(out_buf, format=target_fmt)

                                    zout.writestr(item.filename, out_buf.getvalue())

                                except Exception as e:
                                    # If image fail, copy original
                                    self._log(f"  > Img Fail {item.filename}: {e}")
                                    zout.writestr(item, data)
                            else:
                                # Not image, copy
                                zout.writestr(item, data)

                new_size = os.path.getsize(out_path)
                diff = (original_size - new_size) / (1024 * 1024)
                saved_bytes += (original_size - new_size)

                self._log(f"  > Done. Reduced by {diff:.2f} MB")
                self.update_queue.put(lambda v=idx + 1: self.progress.configure(value=v))

            except Exception as e:
                self._log(f"CRITICAL ERROR on {fname}: {e}")
                if os.path.exists(out_path): os.remove(out_path)

        self.is_running = False
        self.update_queue.put(lambda: self.btn_run.config(text="START COMPRESSION"))
        total_saved_mb = saved_bytes / (1024 * 1024)
        self._log(f"--- Finished. Total space saved: {total_saved_mb:.2f} MB ---")

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])