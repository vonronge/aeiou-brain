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

"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The General Factory:
Standardizes loose raw data (Images & Text) for the training pipeline.
- Images: Converted to PNG, resized to standard dimensions, stripped of metadata.
- Text: Converted to UTF-8, chunked into context-window-friendly blocks.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps
import os
import shutil
import threading
import queue
import traceback
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "General Factory"

        self.is_processing = False
        self.stop_requested = False
        self.work_queue = []
        self.update_queue = queue.Queue()

        # --- SETTINGS ---
        default_in = self.app.paths.get("data", "")
        default_out = os.path.join(default_in, "Processed_General")

        self.input_dir = tk.StringVar(value=default_in)
        self.output_dir = tk.StringVar(value=default_out)

        self.target_dim = tk.IntVar(value=512)
        self.chunk_size = tk.IntVar(value=1024)  # Characters per text chunk
        self.overlap = tk.IntVar(value=100)
        self.skip_existing = tk.BooleanVar(value=True)
        self.convert_grayscale = tk.BooleanVar(value=False)

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        scale = getattr(self.app, 'ui_scale', 1.0)

        # 1. PATH CONFIG
        fr_io = ttk.LabelFrame(self.parent, text="Assembly Line Configuration", padding=10)
        fr_io.pack(fill="x", padx=10, pady=5)

        # Input
        r1 = ttk.Frame(fr_io)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Raw Source:", width=12).pack(side="left")
        ttk.Entry(r1, textvariable=self.input_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂", width=4, command=lambda: self._browse(self.input_dir)).pack(side="left")

        # Output
        r2 = ttk.Frame(fr_io)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Output Dest:", width=12).pack(side="left")
        ttk.Entry(r2, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r2, text="📂", width=4, command=lambda: self._browse(self.output_dir)).pack(side="left")

        # 2. PROCESSING RULES
        fr_rules = ttk.LabelFrame(self.parent, text="Standardization Rules", padding=10)
        fr_rules.pack(fill="x", padx=10, pady=5)

        # Image Rules
        r_img = ttk.Frame(fr_rules)
        r_img.pack(fill="x", pady=2)
        ttk.Label(r_img, text="[Images] Max Dimension:", width=22).pack(side="left")
        ttk.Spinbox(r_img, from_=256, to=2048, increment=64, textvariable=self.target_dim, width=6).pack(side="left")
        ttk.Checkbutton(r_img, text="Force Grayscale", variable=self.convert_grayscale).pack(side="left", padx=15)

        # Text Rules
        r_txt = ttk.Frame(fr_rules)
        r_txt.pack(fill="x", pady=2)
        ttk.Label(r_txt, text="[Text] Chunk Size (chars):", width=22).pack(side="left")
        ttk.Spinbox(r_txt, from_=100, to=10000, increment=100, textvariable=self.chunk_size, width=6).pack(side="left")
        ttk.Label(r_txt, text="Overlap:").pack(side="left", padx=(10, 5))
        ttk.Entry(r_txt, textvariable=self.overlap, width=5).pack(side="left")

        # Global Rules
        r_glob = ttk.Frame(fr_rules)
        r_glob.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(r_glob, text="Skip Existing Files (Incremental Update)", variable=self.skip_existing).pack(
            side="left")

        # 3. QUEUE
        fr_list = ttk.Frame(self.parent)
        fr_list.pack(fill="both", expand=True, padx=10, pady=5)

        tb = ttk.Frame(fr_list)
        tb.pack(fill="x", pady=2)
        ttk.Button(tb, text="SCAN FOLDER", command=self._scan).pack(side="left", fill="x", expand=True)
        self.lbl_count = ttk.Label(tb, text="0 files queued", foreground=self.app.colors["ACCENT"])
        self.lbl_count.pack(side="left", padx=10)

        # Treeview
        cols = ("File", "Type", "Status")
        style = ttk.Style()
        row_h = int(25 * scale)
        style.configure("Factory.Treeview", rowheight=row_h, font=("Segoe UI", int(10 * scale)))
        style.configure("Factory.Treeview.Heading", font=("Segoe UI", int(11 * scale), "bold"))

        self.tree = ttk.Treeview(fr_list, columns=cols, show="headings", height=8, style="Factory.Treeview")
        self.tree.heading("File", text="Filename")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Status", text="Status")

        self.tree.column("File", width=int(400 * scale))
        self.tree.column("Type", width=int(80 * scale), anchor="center")
        self.tree.column("Status", width=int(150 * scale), anchor="center")

        sb = ttk.Scrollbar(fr_list, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 4. ACTION
        fr_run = ttk.Frame(self.parent, padding=10)
        fr_run.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(fr_run, text="START PRODUCTION", command=self._toggle_run, state="disabled")
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
            # Route to Golgi
            level = "INFO"
            if tag == "ERROR":
                level = "ERROR"
            elif tag == "SUCCESS":
                level = "SUCCESS"
            self.app.golgi._dispatch(level, msg, source="Factory")

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
        self.work_queue = []

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        txt_exts = {".txt", ".md", ".json", ".csv", ".py", ".html", ".xml"}

        count = 0
        for root, _, files in os.walk(folder):
            for f in files:
                _, ext = os.path.splitext(f)
                lext = ext.lower()
                full_path = os.path.join(root, f)

                ftype = None
                if lext in img_exts:
                    ftype = "Image"
                elif lext in txt_exts:
                    ftype = "Text"

                if ftype:
                    self.tree.insert("", "end", iid=full_path, values=(f, ftype, "Queued"))
                    self.work_queue.append((full_path, ftype))
                    count += 1

        self.lbl_count.config(text=f"{count} files queued")
        if count > 0:
            self.btn_run.config(state="normal")
        self._log(f"Scan complete. Found {count} items.")

    # --- PROCESSING ---
    def _toggle_run(self):
        if self.is_processing:
            self.stop_requested = True
            self.btn_run.config(text="STOPPING...")
        else:
            self.stop_requested = False
            self.is_processing = True
            self.btn_run.config(text="STOP PRODUCTION")
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        out_root = self.output_dir.get()
        if not os.path.exists(out_root):
            try:
                os.makedirs(out_root)
            except:
                pass

        total = len(self.work_queue)
        target_s = self.target_dim.get()
        chunk_s = self.chunk_size.get()
        ovlp = self.overlap.get()
        skip = self.skip_existing.get()
        gray = self.convert_grayscale.get()

        for idx, (path, ftype) in enumerate(self.work_queue):
            if self.stop_requested: break

            fname = os.path.basename(path)
            self._update_tree(path, "Status", "Processing...")
            self._log(f"Processing {fname} ({idx + 1}/{total})...")

            try:
                if ftype == "Image":
                    # Image Processing Logic
                    dest_path = os.path.join(out_root, os.path.splitext(fname)[0] + ".png")

                    if skip and os.path.exists(dest_path):
                        self._update_tree(path, "Status", "Skipped")
                    else:
                        with Image.open(path) as img:
                            # Convert to RGB (remove alpha)
                            if img.mode in ("RGBA", "P"): img = img.convert("RGB")

                            # Grayscale check
                            if gray: img = ImageOps.grayscale(img)

                            # Resize (Downscale only)
                            w, h = img.size
                            if max(w, h) > target_s:
                                ratio = target_s / max(w, h)
                                new_size = (int(w * ratio), int(h * ratio))
                                img = img.resize(new_size, Image.Resampling.LANCZOS)

                            img.save(dest_path, "PNG", optimize=True)
                        self._update_tree(path, "Status", "Done")

                elif ftype == "Text":
                    # Text Chunking Logic
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            raw = f.read()
                    except:
                        # Fallback for weird encodings
                        with open(path, 'r', encoding='latin-1') as f:
                            raw = f.read()

                    # Clean whitespace
                    clean = " ".join(raw.split())

                    if len(clean) < 50:
                        self._update_tree(path, "Status", "Ignored (Tiny)")
                    else:
                        base_name = os.path.splitext(fname)[0]
                        chunks = []
                        start = 0
                        while start < len(clean):
                            end = min(start + chunk_s, len(clean))
                            chunks.append(clean[start:end])
                            start += (chunk_s - ovlp)

                        # Save chunks
                        for i, c in enumerate(chunks):
                            c_name = f"{base_name}_part{str(i).zfill(3)}.txt"
                            c_path = os.path.join(out_root, c_name)
                            if skip and os.path.exists(c_path): continue

                            with open(c_path, 'w', encoding='utf-8') as f:
                                f.write(c)

                        self._update_tree(path, "Status", f"Done ({len(chunks)} chunks)")

                # Update Progress
                pct = ((idx + 1) / total) * 100
                self.update_queue.put(lambda v=pct: self.progress.configure(value=v))

            except Exception as e:
                self._update_tree(path, "Status", "Failed")
                self._log(f"Error on {fname}: {e}", "ERROR")

        self.is_processing = False
        self.update_queue.put(lambda: self.btn_run.config(text="START PRODUCTION"))
        self._log("Factory Run Complete.", "SUCCESS")
        self.update_queue.put(lambda: self.progress.configure(value=0))

    def on_theme_change(self):
        pass