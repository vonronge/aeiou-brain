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
import pyttsx3
import re
import queue
from datetime import datetime
import traceback


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Lecture Factory"

        self.is_processing = False
        self.stop_requested = False
        self.pdf_queue = []

        # Queue for thread-safe UI updates
        self.update_queue = queue.Queue()

        # --- PATHS (via Phagus) ---
        # Default to the main data directory if set
        default_root = self.app.paths.get("data", os.path.abspath("Training_Data"))
        default_out = os.path.join(default_root, "Lectures")

        # Config Vars
        self.input_folder = tk.StringVar(value=default_root)
        self.output_folder = tk.StringVar(value=default_out)
        self.dpi = tk.IntVar(value=150)
        self.generate_audio = tk.BooleanVar(value=True)
        self.skip_existing = tk.BooleanVar(value=True)

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        # 1. PATH CONFIGURATION
        fr_io = ttk.LabelFrame(self.parent, text="Library Configuration", padding=10)
        fr_io.pack(fill="x", padx=10, pady=5)

        # Source
        r1 = ttk.Frame(fr_io)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Source Folder:", width=12).pack(side="left")
        ttk.Entry(r1, textvariable=self.input_folder).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂", width=4, command=lambda: self._browse(self.input_folder)).pack(side="left")

        # Output
        r2 = ttk.Frame(fr_io)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Output Root:", width=12).pack(side="left")
        ttk.Entry(r2, textvariable=self.output_folder).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r2, text="📂", width=4, command=lambda: self._browse(self.output_folder)).pack(side="left")

        # 2. SCANNER & QUEUE
        fr_scan = ttk.Frame(self.parent)
        fr_scan.pack(fill="both", expand=True, padx=10, pady=5)

        # Controls
        ctrl_row = ttk.Frame(fr_scan)
        ctrl_row.pack(fill="x", pady=5)

        ttk.Button(ctrl_row, text="1. SCAN FOR PDFS", command=self._scan_folder).pack(side="left", fill="x",
                                                                                      expand=True, padx=(0, 5))

        # Settings
        ttk.Label(ctrl_row, text="Quality (DPI):").pack(side="left")
        ttk.Spinbox(ctrl_row, from_=72, to=300, textvariable=self.dpi, width=5).pack(side="left", padx=5)

        ttk.Checkbutton(ctrl_row, text="TTS Audio", variable=self.generate_audio).pack(side="left", padx=10)
        ttk.Checkbutton(ctrl_row, text="Skip Done", variable=self.skip_existing).pack(side="left", padx=5)

        # Treeview
        cols = ("Filename", "Pages", "Status")
        self.tree = ttk.Treeview(fr_scan, columns=cols, show="headings", height=8)
        self.tree.heading("Filename", text="Filename")
        self.tree.heading("Pages", text="Pages")
        self.tree.heading("Status", text="Status")

        self.tree.column("Filename", width=400)
        self.tree.column("Pages", width=80, anchor="center")
        self.tree.column("Status", width=150, anchor="center")

        sb = ttk.Scrollbar(fr_scan, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 3. ACTION PANEL
        fr_run = ttk.Frame(self.parent, padding=10)
        fr_run.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(fr_run, text="2. START FACTORY", command=self._start_processing, state="disabled")
        self.btn_run.pack(side="left", fill="x", expand=True, padx=5)

        self.progress = ttk.Progressbar(fr_run, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=5)

        # Local Status Label
        self.log_lbl = ttk.Label(self.parent, text="Ready.", foreground=self.app.colors["FG_DIM"], anchor="w")
        self.log_lbl.pack(fill="x", padx=15, pady=(0, 10))

    # --- HELPERS ---
    def _browse(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d: var.set(d)

    def _log(self, msg, tag="INFO"):
        # 1. Update Local Label via Queue
        self.update_queue.put(lambda: self.log_lbl.config(text=msg))

        # 2. Update Golgi (System Log) - Filter high frequency messages
        if self.app.golgi and "Page" not in msg:
            if tag == "INFO":
                self.app.golgi.info(msg, source="LectureFactory")
            elif tag == "SUCCESS":
                self.app.golgi.success(msg, source="LectureFactory")
            elif tag == "ERROR":
                self.app.golgi.error(msg, source="LectureFactory")

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

    def _update_progress(self, val):
        self.update_queue.put(lambda: self.progress.configure(value=val))

    # --- WORKERS ---
    def _scan_folder(self):
        folder = self.input_folder.get()
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Source folder not found.")
            return

        # Clear Tree
        for item in self.tree.get_children():
            self.tree.delete(item)

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
                        pass  # Corrupt PDF logic handled by repair tool

        self._log(f"Found {count} PDFs.", "SUCCESS")
        if count > 0:
            self.btn_run.config(state="normal")

    def _start_processing(self):
        if self.is_processing:
            self.stop_requested = True
            self.btn_run.config(text="STOPPING...")
        else:
            if not self.pdf_queue: return
            self.is_processing = True
            self.stop_requested = False
            self.btn_run.config(text="STOP FACTORY")
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        out_root = self.output_folder.get()
        if not os.path.exists(out_root):
            try:
                os.makedirs(out_root)
            except:
                pass

        # Init TTS Engine locally (not an organelle yet)
        engine = None
        if self.generate_audio.get():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
            except Exception as e:
                self._log(f"TTS Init Failed: {e}", "ERROR")

        total_files = len(self.pdf_queue)

        for idx, pdf_path in enumerate(self.pdf_queue):
            if self.stop_requested: break

            filename = os.path.basename(pdf_path)
            book_name = os.path.splitext(filename)[0]
            # Sanitize
            safe_book_name = re.sub(r'[^a-zA-Z0-9]', '_', book_name)

            # Destination
            book_dir = os.path.join(out_root, f"{safe_book_name}_lecture")
            if not os.path.exists(book_dir): os.makedirs(book_dir)

            self._update_tree(pdf_path, "Status", "Processing...")
            self._log(f"Processing {filename} ({idx + 1}/{total_files})...")

            try:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)

                for i, page in enumerate(doc):
                    if self.stop_requested: break

                    # Naming: book_p0001
                    fname = f"{safe_book_name}_p{str(i + 1).zfill(4)}"
                    base_path = os.path.join(book_dir, fname)

                    # Skip Logic
                    if self.skip_existing.get():
                        if os.path.exists(base_path + ".txt") and os.path.exists(base_path + ".png"):
                            continue

                    # 1. EXTRACT TEXT
                    text = page.get_text("text").strip()
                    # Filter junk (page numbers, empty pages)
                    if len(text) < 10:
                        text = "Visual content."  # Placeholder for visual-only pages

                    with open(base_path + ".txt", "w", encoding="utf-8") as f:
                        f.write(text)

                    # 2. EXTRACT IMAGE (Quadruplet Visual)
                    pix = page.get_pixmap(dpi=self.dpi.get())
                    pix.save(base_path + ".png")

                    # 3. GENERATE AUDIO (Quadruplet Audio)
                    if engine and text:
                        clean_text = " ".join(text.split())
                        # Remove citations [1], URLs, etc.
                        clean_text = re.sub(r'\[\d+\]', '', clean_text)

                        if len(clean_text) > 5:
                            try:
                                engine.save_to_file(clean_text, base_path + ".wav")
                                engine.runAndWait()
                            except:
                                pass

                    # UI Pulse
                    if i % 2 == 0:
                        pct = ((i + 1) / total_pages) * 100
                        self._update_progress(pct)
                        self.parent.after(0, lambda
                            m=f"Reading {filename}: Page {i + 1}/{total_pages}": self.log_lbl.config(text=m))

                doc.close()
                self._update_tree(pdf_path, "Status", "Done")

            except Exception as e:
                self._log(f"Error on {filename}: {e}", "ERROR")
                self._update_tree(pdf_path, "Status", "Failed")
                traceback.print_exc()

        self.is_processing = False
        self.update_queue.put(lambda: self.btn_run.config(text="2. START FACTORY"))
        self._update_progress(0)
        self._log("Factory Run Complete.", "SUCCESS")

    def on_theme_change(self):
        pass