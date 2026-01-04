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
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Lecture Factory"

        # State
        self.is_processing = False
        self.stop_requested = False
        self.pdf_queue = []

        # Config
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar(value="D:/Training_Data/Lectures")
        self.dpi = tk.IntVar(value=150)
        self.generate_audio = tk.BooleanVar(value=True)

        self._setup_ui()

    def _setup_ui(self):
        # --- TOP: PATHS ---
        fr_io = ttk.LabelFrame(self.parent, text="Library Paths", padding=10)
        fr_io.pack(fill="x", padx=10, pady=5)

        # Source Folder
        r1 = ttk.Frame(fr_io);
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Source Folder:", width=12).pack(side="left")
        ttk.Entry(r1, textvariable=self.input_folder).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂", width=4, command=self._browse_input).pack(side="left")

        # Output Folder
        r2 = ttk.Frame(fr_io);
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Output Root:", width=12).pack(side="left")
        ttk.Entry(r2, textvariable=self.output_folder).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r2, text="📂", width=4, command=self._browse_output).pack(side="left")

        # --- MID: SCANNER & LIST ---
        fr_scan = ttk.Frame(self.parent)
        fr_scan.pack(fill="both", expand=True, padx=10, pady=5)

        # Controls Row
        ctrl_row = ttk.Frame(fr_scan)
        ctrl_row.pack(fill="x", pady=5)

        ttk.Button(ctrl_row, text="1. SCAN FOR PDFS", command=self._scan_folder).pack(side="left", fill="x",
                                                                                      expand=True, padx=(0, 5))

        ttk.Label(ctrl_row, text="Quality (DPI):").pack(side="left")
        ttk.Spinbox(ctrl_row, from_=72, to=300, textvariable=self.dpi, width=5).pack(side="left", padx=5)

        ttk.Checkbutton(ctrl_row, text="Generate Audio (TTS)", variable=self.generate_audio).pack(side="left", padx=10)

        # Treeview for Queue
        cols = ("Filename", "Pages", "Status")
        self.tree = ttk.Treeview(fr_scan, columns=cols, show="headings", height=8)
        self.tree.heading("Filename", text="Filename")
        self.tree.heading("Pages", text="Pages")
        self.tree.heading("Status", text="Status")
        self.tree.column("Filename", width=300)
        self.tree.column("Pages", width=80, anchor="center")
        self.tree.column("Status", width=120, anchor="center")

        sb = ttk.Scrollbar(fr_scan, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # --- BOTTOM: ACTION ---
        fr_run = ttk.Frame(self.parent, padding=10)
        fr_run.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(fr_run, text="2. START FACTORY", command=self._start_processing, state="disabled")
        self.btn_run.pack(side="left", fill="x", expand=True, padx=5)

        self.progress = ttk.Progressbar(fr_run, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=5)

        # Log
        self.log_lbl = ttk.Label(self.parent, text="Ready.", foreground=self.app.colors["FG_DIM"], anchor="w")
        self.log_lbl.pack(fill="x", padx=15, pady=(0, 10))

    # --- ACTIONS ---
    def _browse_input(self):
        d = filedialog.askdirectory()
        if d: self.input_folder.set(d)

    def _browse_output(self):
        d = filedialog.askdirectory()
        if d: self.output_folder.set(d)

    def _log(self, msg):
        self.log_lbl.config(text=msg)
        self.parent.update_idletasks()

    def _scan_folder(self):
        folder = self.input_folder.get()
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Source folder not found.")
            return

        self.tree.delete(*self.tree.get_children())
        self.pdf_queue = []

        count = 0
        for f in os.listdir(folder):
            if f.lower().endswith(".pdf"):
                full_path = os.path.join(folder, f)
                try:
                    # Quick open to count pages
                    doc = fitz.open(full_path)
                    pages = len(doc)
                    doc.close()

                    self.tree.insert("", "end", iid=full_path, values=(f, pages, "Queued"))
                    self.pdf_queue.append(full_path)
                    count += 1
                except:
                    pass

        self._log(f"Found {count} PDFs.")
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
        if not os.path.exists(out_root): os.makedirs(out_root)

        # Setup TTS Engine
        engine = None
        if self.generate_audio.get():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
            except Exception as e:
                print(f"TTS Init Error: {e}")

        total_files = len(self.pdf_queue)

        for idx, pdf_path in enumerate(self.pdf_queue):
            if self.stop_requested: break

            filename = os.path.basename(pdf_path)
            book_name = os.path.splitext(filename)[0]
            # Sanitize folder name
            safe_book_name = re.sub(r'[^a-zA-Z0-9]', '_', book_name)

            # Create Book Subfolder
            book_dir = os.path.join(out_root, f"{safe_book_name}_lecture")
            if not os.path.exists(book_dir): os.makedirs(book_dir)

            self.parent.after(0, lambda p=pdf_path: self.tree.set(p, "Status", "Processing..."))
            self._log(f"Processing book {idx + 1}/{total_files}: {filename}")

            try:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)

                for i, page in enumerate(doc):
                    if self.stop_requested: break

                    # Naming Scheme: BookName_p0001
                    fname = f"{safe_book_name}_p{str(i + 1).zfill(4)}"
                    base_path = os.path.join(book_dir, fname)

                    # 0. Check Exists (Skip Logic)
                    if os.path.exists(base_path + ".txt") and \
                            os.path.exists(base_path + ".png"):
                        continue

                    # 1. EXTRACT TEXT
                    text = page.get_text("text").strip()
                    with open(base_path + ".txt", "w", encoding="utf-8") as f:
                        f.write(text)

                    # 2. EXTRACT IMAGE
                    pix = page.get_pixmap(dpi=self.dpi.get())
                    pix.save(base_path + ".png")

                    # 3. GENERATE AUDIO
                    if engine and text:
                        clean_text = " ".join(text.split())
                        if len(clean_text) > 5:
                            try:
                                # Note: pyttsx3 saves as WAV by default
                                engine.save_to_file(clean_text, base_path + ".wav")
                                engine.runAndWait()
                            except Exception as e:
                                print(f"Audio Error: {e}")

                    # UI Pulse
                    if i % 5 == 0:
                        pct = ((i + 1) / total_pages) * 100
                        self.parent.after(0, lambda v=pct: self.progress.configure(value=v))
                        self.parent.after(0, lambda
                            m=f"Book {idx + 1}/{total_files} | Page {i + 1}/{total_pages}": self._log(m))

                doc.close()
                self.parent.after(0, lambda p=pdf_path: self.tree.set(p, "Status", "Done"))

            except Exception as e:
                print(f"Error on {filename}: {e}")
                self.parent.after(0, lambda p=pdf_path: self.tree.set(p, "Status", "Error"))

        self.is_processing = False
        self.parent.after(0, lambda: self.btn_run.config(text="2. START FACTORY"))
        self.parent.after(0, lambda: self._log("Factory Run Complete."))
        self.parent.after(0, lambda: self.progress.configure(value=0))

    def on_theme_change(self):
        pass