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
from PIL import Image, ImageTk
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Lecture Factory"  # Canonical Name

        # State
        self.is_processing = False
        self.stop_requested = False
        self.queue = []

        # Config
        self.input_file = tk.StringVar()
        self.output_folder = tk.StringVar(value="D:/Training_Data/Lectures")
        self.dpi = tk.IntVar(value=150)  # Quality of page render
        self.generate_audio = tk.BooleanVar(value=True)
        self.prefix = tk.StringVar(value="")

        self._setup_ui()

    def _setup_ui(self):
        # --- INPUT / OUTPUT ---
        fr_io = ttk.LabelFrame(self.parent, text="Source Material (PDF Textbooks)", padding=10)
        fr_io.pack(fill="x", padx=10, pady=5)

        # Input PDF
        r1 = ttk.Frame(fr_io);
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="PDF File:").pack(side="left", width=10)
        ttk.Entry(r1, textvariable=self.input_file).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="Browse", width=8, command=self._browse_input).pack(side="left")

        # Output Folder
        r2 = ttk.Frame(fr_io);
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Output:").pack(side="left", width=10)
        ttk.Entry(r2, textvariable=self.output_folder).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r2, text="Browse", width=8, command=self._browse_output).pack(side="left")

        # Prefix
        r3 = ttk.Frame(fr_io);
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Prefix:").pack(side="left", width=10)
        ttk.Entry(r3, textvariable=self.prefix, placeholder="e.g. physics_101").pack(side="left", fill="x", expand=True,
                                                                                     padx=5)

        # --- SETTINGS ---
        fr_set = ttk.LabelFrame(self.parent, text="Processing Options", padding=10)
        fr_set.pack(fill="x", padx=10, pady=5)

        ttk.Checkbutton(fr_set, text="Generate Audio (TTS Narration)", variable=self.generate_audio).pack(side="left",
                                                                                                          padx=10)

        ttk.Label(fr_set, text="| Image Quality (DPI):").pack(side="left", padx=(20, 5))
        ttk.Spinbox(fr_set, from_=72, to=300, textvariable=self.dpi, width=5).pack(side="left")

        # --- CONTROLS ---
        fr_run = ttk.Frame(self.parent, padding=10)
        fr_run.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(fr_run, text="START EXTRACTION", command=self._start_processing)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=5)

        self.progress = ttk.Progressbar(fr_run, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=5)

        # --- LOG ---
        fr_log = ttk.LabelFrame(self.parent, text="Factory Log", padding=10)
        fr_log.pack(fill="both", expand=True, padx=10, pady=10)

        # Autoscroll toggle
        self.auto_scroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr_log, text="Autoscroll", variable=self.auto_scroll).pack(anchor="e")

        self.log_box = tk.Text(fr_log, height=10, font=("Consolas", 9),
                               bg=self.app.colors["BG_MAIN"], fg=self.app.colors["FG_TEXT"],
                               borderwidth=0)
        self.log_box.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(fr_log, orient="vertical", command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # Tags
        self.log_box.tag_config("info", foreground=self.app.colors["ACCENT"])
        self.log_box.tag_config("success", foreground=self.app.colors["SUCCESS"])
        self.log_box.tag_config("error", foreground=self.app.colors["ERROR"])

    # --- ACTIONS ---
    def _browse_input(self):
        f = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if f:
            self.input_file.set(f)
            # Auto-set prefix based on filename
            name = os.path.splitext(os.path.basename(f))[0]
            name = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
            self.prefix.set(name)

    def _browse_output(self):
        d = filedialog.askdirectory()
        if d: self.output_folder.set(d)

    def _log(self, msg, tag="info"):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n", tag)
        if self.auto_scroll.get():
            self.log_box.see(tk.END)

    def _start_processing(self):
        if self.is_processing:
            self.stop_requested = True
            self.btn_run.config(text="STOPPING...")
        else:
            if not os.path.exists(self.input_file.get()):
                messagebox.showerror("Error", "Input file not found.")
                return

            self.is_processing = True
            self.stop_requested = False
            self.btn_run.config(text="STOP")
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        pdf_path = self.input_file.get()
        out_root = self.output_folder.get()
        prefix = self.prefix.get()

        try:
            doc = fitz.open(pdf_path)
            total = len(doc)
            self.progress["maximum"] = total

            # Setup TTS Engine
            engine = None
            if self.generate_audio.get():
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                except Exception as e:
                    self.parent.after(0, lambda: self._log(f"TTS Init Error: {e}", "error"))

            if not os.path.exists(out_root): os.makedirs(out_root)

            self.parent.after(0, lambda: self._log(f"Starting extraction of {total} pages...", "info"))

            for i, page in enumerate(doc):
                if self.stop_requested: break

                # Filename scheme: prefix_p0001
                fname = f"{prefix}_p{str(i + 1).zfill(4)}"
                base_path = os.path.join(out_root, fname)

                # 1. EXTRACT TEXT
                text = page.get_text("text").strip()

                # Save Text
                with open(base_path + ".txt", "w", encoding="utf-8") as f:
                    f.write(text)

                # 2. EXTRACT IMAGE (Visual)
                pix = page.get_pixmap(dpi=self.dpi.get())
                pix.save(base_path + ".png")

                # 3. GENERATE AUDIO (Audio)
                if engine and text:
                    clean_text = " ".join(text.split())
                    if len(clean_text) > 5:
                        try:
                            # Note: pyttsx3 is blocking. In a thread this is fine,
                            # but extensive use might lag if not careful.
                            engine.save_to_file(clean_text, base_path + ".wav")
                            engine.runAndWait()
                        except Exception as e:
                            print(f"Audio Gen Fail: {e}")

                # Update UI
                self.parent.after(0, lambda v=i + 1: self.progress.configure(value=v))

                if i % 5 == 0:
                    self.parent.after(0, lambda n=i + 1: self._log(f"Processed Page {n}/{total}", "info"))

            self.parent.after(0, lambda: self._log("Extraction Complete.", "success"))

        except Exception as e:
            self.parent.after(0, lambda err=str(e): self._log(f"CRITICAL ERROR: {err}", "error"))
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False
            self.parent.after(0, lambda: self.btn_run.config(text="START EXTRACTION"))

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'):
            self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])