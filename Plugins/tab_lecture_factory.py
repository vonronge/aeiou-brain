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
import textwrap
import tempfile
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont

# --- OCR SUPPORT ---
try:
    import pytesseract

    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# --- PLAYWRIGHT SUPPORT ---
try:
    from playwright.sync_api import sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

NEURAL_VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural",
    "en-US-ChristopherNeural", "en-US-EricNeural", "en-GB-SoniaNeural"
]

WEB_FONTS = [
    "Arial, sans-serif", "Times New Roman, serif", "Courier New, monospace",
    "Georgia, serif", "Verdana, sans-serif", "Trebuchet MS, sans-serif",
    "Comic Sans MS, cursive", "Impact, sans-serif", "Segoe UI, sans-serif", "Tahoma, sans-serif"
]

POSSIBLE_FONTS_PIL = [
    "arial.ttf", "times.ttf", "cour.ttf", "georgia.ttf",
    "verdana.ttf", "calibri.ttf", "segoeui.ttf", "tahoma.ttf"
]


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Lecture Factory"

        self.is_processing = False
        self.stop_requested = False
        self.doc_queue = []
        self.update_queue = queue.Queue()

        # Settings
        default_root = self.app.paths.get("data", os.path.abspath("Training_Data"))
        default_out = os.path.join(default_root, "Lectures")

        self.input_folder = tk.StringVar(value=default_root)
        self.output_folder = tk.StringVar(value=default_out)
        self.render_dpi = tk.IntVar(value=300)  # Increased for OCR
        self.generate_audio = tk.BooleanVar(value=True)
        self.random_voice = tk.BooleanVar(value=True)
        self.selected_voice = tk.StringVar(value="en-US-GuyNeural")

        self.render_txt = tk.BooleanVar(value=True)
        self.use_playwright = tk.BooleanVar(value=HAS_PLAYWRIGHT)

        # OCR Settings
        self.use_ocr = tk.BooleanVar(value=False)
        self.tesseract_path = tk.StringVar(value=r"C:\Program Files\Tesseract-OCR\tesseract.exe")

        # Auto-detect Tesseract
        if HAS_OCR:
            if os.path.exists(self.tesseract_path.get()):
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path.get()
                self.use_ocr.set(True)
            else:
                # Try pathless
                try:
                    pytesseract.get_tesseract_version()
                    self.use_ocr.set(True)
                except:
                    pass

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return
        scale = getattr(self.app, 'ui_scale', 1.0)

        f_header = ("Segoe UI", int(12 * scale), "bold")
        f_normal = ("Segoe UI", int(10 * scale))

        # 1. PATHS
        fr_io = ttk.LabelFrame(self.parent, text="Library Configuration", padding=10)
        fr_io.pack(fill="x", padx=10, pady=5)

        r1 = ttk.Frame(fr_io);
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Source Folder:", width=15, font=f_header).pack(side="left")
        ttk.Entry(r1, textvariable=self.input_folder, font=f_normal).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂", width=4, command=lambda: self._browse(self.input_folder)).pack(side="left")

        r2 = ttk.Frame(fr_io);
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Output Root:", width=15, font=f_header).pack(side="left")
        ttk.Entry(r2, textvariable=self.output_folder, font=f_normal).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r2, text="📂", width=4, command=lambda: self._browse(self.output_folder)).pack(side="left")

        # 2. SETTINGS
        fr_scan = ttk.Frame(self.parent)
        fr_scan.pack(fill="both", expand=True, padx=10, pady=5)

        ctrl_row = ttk.Frame(fr_scan)
        ctrl_row.pack(fill="x", pady=5)

        ttk.Button(ctrl_row, text="1. FAST SCAN", command=self._scan_folder).pack(side="left", fill="x", expand=True,
                                                                                  padx=(0, 5))
        ttk.Label(ctrl_row, text="DPI:", font=f_normal).pack(side="left")
        ttk.Spinbox(ctrl_row, from_=72, to=600, increment=24, textvariable=self.render_dpi, width=5,
                    font=f_normal).pack(side="left", padx=5)
        ttk.Checkbutton(ctrl_row, text="Neural Audio (MP3)", variable=self.generate_audio).pack(side="left", padx=10)

        # OCR Controls
        chk_ocr = ttk.Checkbutton(ctrl_row, text="OCR (Scans)", variable=self.use_ocr)
        chk_ocr.pack(side="left", padx=5)
        if not HAS_OCR: chk_ocr.config(state="disabled", text="OCR Missing")

        ttk.Button(ctrl_row, text="OCR Path", width=8, command=self._set_tesseract_path).pack(side="left", padx=2)

        # Playwright
        chk_pw = ttk.Checkbutton(ctrl_row, text="Playwright Fonts", variable=self.use_playwright)
        chk_pw.pack(side="left", padx=5)
        if not HAS_PLAYWRIGHT: chk_pw.config(state="disabled", text="PW Missing")

        self.lbl_summary = ttk.Label(fr_scan, text="Ready to Scan.", font=("Consolas", int(11 * scale)),
                                     background=self.app.colors["BG_CARD"], anchor="center", relief="sunken")
        self.lbl_summary.pack(fill="both", expand=True, pady=5, ipady=10)

        # 3. ACTIONS
        fr_run = ttk.Frame(self.parent, padding=10)
        fr_run.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(fr_run, text="▶ START FACTORY", command=self._start_processing, state="disabled")
        self.btn_run.pack(side="left", fill="x", expand=True, padx=5)
        self.btn_stop = ttk.Button(fr_run, text="⏹ STOP", command=self._stop_processing, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(fr_run, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=5)

        self.log_lbl = ttk.Label(self.parent, text="Ready.", foreground=self.app.colors["FG_DIM"], anchor="w",
                                 font=f_normal)
        self.log_lbl.pack(fill="x", padx=15, pady=(0, 10))

    def _browse(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d: var.set(d)

    def _set_tesseract_path(self):
        f = filedialog.askopenfilename(title="Find tesseract.exe", filetypes=[("Executables", "*.exe")])
        if f:
            self.tesseract_path.set(f)
            if HAS_OCR: pytesseract.pytesseract.tesseract_cmd = f

    def _log(self, msg):
        self.update_queue.put(lambda: self.log_lbl.config(text=msg))

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                self.update_queue.get_nowait()()
            except:
                break
        if self.parent: self.parent.after(100, self._process_gui_queue)

    def _run_edge_tts(self, text, voice, out_path):
        """CLI Wrapper for MP3 generation"""
        clean_text = re.sub(r'[\r\n]+', ' ', text).replace('"', '').replace("'", "").strip()
        if not clean_text: return False

        # Output to MP3
        cmd = [
            "edge-tts",
            "--voice", voice,
            "--text", clean_text,
            "--write-media", out_path
        ]

        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           startupinfo=startupinfo)
            return True
        except:
            return False

    def _scan_folder(self):
        folder = self.input_folder.get()
        if not os.path.exists(folder): return

        self.doc_queue = []
        valid_exts = {'.pdf', '.epub', '.mobi', '.txt', '.fb2', '.cbz', '.xps'}
        counts = {ext: 0 for ext in valid_exts}

        for root, _, files in os.walk(folder):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    self.doc_queue.append(os.path.join(root, f))
                    counts[ext] += 1

        total = len(self.doc_queue)
        summary = f"Found {total} Documents:\n"
        for ext, count in counts.items():
            if count > 0: summary += f"  {ext.upper()}: {count}  "

        self.lbl_summary.config(text=summary)
        if total > 0: self.btn_run.config(state="normal")

    def _start_processing(self):
        if not self.doc_queue: return
        self.is_processing = True
        self.stop_requested = False
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        threading.Thread(target=self._worker, daemon=True).start()

    def _stop_processing(self):
        self.stop_requested = True
        self.btn_stop.config(text="STOPPING...")

    def _render_with_playwright(self, doc_path, book_dir, safe_name, dpi):
        if not HAS_PLAYWRIGHT: return
        try:
            doc = fitz.open(doc_path)
            total = len(doc)

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={'width': 800, 'height': 1100})

                for i, pdf_page in enumerate(doc):
                    if self.stop_requested: break

                    fname = f"{safe_name}_p{str(i + 1).zfill(4)}"
                    base_path = os.path.join(book_dir, fname)

                    missing_img = not os.path.exists(base_path + ".png")
                    missing_aud = self.generate_audio.get() and not os.path.exists(base_path + ".mp3")

                    if not missing_img and not missing_aud: continue

                    if missing_img:
                        html_content = pdf_page.get_text("html")
                        rand_font = random.choice(WEB_FONTS)
                        injected_html = html_content.replace(
                            "<style>", f"<style>\nbody, p, div, span, td {{ font-family: {rand_font} !important; }}\n")
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False,
                                                         encoding='utf-8') as tmp:
                            tmp.write(injected_html)
                            tmp_path = tmp.name
                        try:
                            page.goto(f"file://{tmp_path}")
                            page.screenshot(path=base_path + ".png", full_page=True)
                        finally:
                            try:
                                os.remove(tmp_path)
                            except:
                                pass

                    # Text
                    text_content = pdf_page.get_text("text").strip()
                    with open(base_path + ".txt", "w", encoding="utf-8") as f:
                        f.write(text_content or "Visual.")

                    if missing_aud and len(text_content) > 20:
                        voice = random.choice(NEURAL_VOICES) if self.random_voice.get() else self.selected_voice.get()
                        self._run_edge_tts(text_content[:4000], voice, base_path + ".mp3")

                    if i % 5 == 0:
                        self.update_queue.put(lambda v=((i + 1) / total) * 100: self.progress.configure(value=v))
                browser.close()
            doc.close()
        except Exception as e:
            self._log(f"PW Failed: {e}")

    def _render_txt_to_images(self, txt_path, out_dir, safe_name):
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return
        chunk_size = 2000
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if self.stop_requested: break
            fname = f"{safe_name}_p{str(i + 1).zfill(4)}"
            base = os.path.join(out_dir, fname)

            missing_img = not os.path.exists(base + ".png")
            missing_aud = self.generate_audio.get() and not os.path.exists(base + ".mp3")

            if not missing_img and not missing_aud:
                self.update_queue.put(lambda v=((i + 1) / total) * 100: self.progress.configure(value=v))
                continue

            if missing_img:
                font_name = random.choice(POSSIBLE_FONTS_PIL)
                try:
                    font = ImageFont.truetype(font_name, random.randint(14, 24))
                except:
                    font = ImageFont.load_default()
                W, H = 800, 1000
                img = Image.new('RGB', (W, H), color=(255, 255, 255))
                draw = ImageDraw.Draw(img)
                lines = textwrap.wrap(chunk, width=int((W - 100) / (20 * 0.6)))
                y = 50
                for line in lines:
                    draw.text((50, y), line, font=font, fill=0)
                    y += 30
                    if y > H - 50: break
                img.save(base + ".png")

            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(chunk)

            if missing_aud:
                voice = random.choice(NEURAL_VOICES)
                self._run_edge_tts(chunk[:1000], voice, base + ".mp3")

            self.update_queue.put(lambda v=((i + 1) / total) * 100: self.progress.configure(value=v))

    def _worker(self):
        out_root = self.output_folder.get()
        if not os.path.exists(out_root): os.makedirs(out_root, exist_ok=True)
        dpi = self.render_dpi.get()
        total_files = len(self.doc_queue)

        for idx, doc_path in enumerate(self.doc_queue):
            if self.stop_requested: break

            filename = os.path.basename(doc_path)
            ext = os.path.splitext(filename)[1].lower()
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', os.path.splitext(filename)[0])
            book_dir = os.path.join(out_root, f"{safe_name}_lecture")
            if not os.path.exists(book_dir): os.makedirs(book_dir)
            self._log(f"Processing {filename} ({idx + 1}/{total_files})...")

            if ext == '.txt' and self.render_txt.get():
                self._render_txt_to_images(doc_path, book_dir, safe_name)
                continue

            if self.use_playwright.get() and HAS_PLAYWRIGHT:
                self._render_with_playwright(doc_path, book_dir, safe_name, dpi)
            else:
                try:
                    doc = fitz.open(doc_path)
                    total = len(doc)
                    for i, page in enumerate(doc):
                        if self.stop_requested: break
                        fname = f"{safe_name}_p{str(i + 1).zfill(4)}"
                        base_path = os.path.join(book_dir, fname)

                        missing_img = not os.path.exists(base_path + ".png")
                        missing_aud = self.generate_audio.get() and not os.path.exists(base_path + ".mp3")

                        if not missing_img and not missing_aud: continue

                        if missing_img:
                            pix = page.get_pixmap(dpi=dpi)
                            pix.save(base_path + ".png")

                        # TEXT & OCR
                        text = page.get_text("text").strip()

                        # Fallback to OCR if empty
                        if len(text) < 10 and self.use_ocr.get() and HAS_OCR:
                            try:
                                # Preprocess: Grayscale + Tesseract
                                img = Image.open(base_path + ".png").convert("L")
                                text = pytesseract.image_to_string(img, lang='eng')
                            except:
                                pass

                        clean_text = re.sub(r'\s+', ' ', text).strip()
                        with open(base_path + ".txt", "w", encoding="utf-8") as f:
                            f.write(clean_text or "Visual.")

                        if missing_aud and len(clean_text) > 20:
                            voice = random.choice(
                                NEURAL_VOICES) if self.random_voice.get() else self.selected_voice.get()
                            self._run_edge_tts(clean_text[:4000], voice, base_path + ".mp3")

                        if i % 5 == 0:
                            self.update_queue.put(lambda v=((i + 1) / total) * 100: self.progress.configure(value=v))
                    doc.close()
                except Exception as e:
                    self._log(f"Error {filename}: {e}")

        self.is_processing = False
        self.update_queue.put(lambda: self.btn_run.config(state="normal"))
        self.update_queue.put(lambda: self.btn_stop.config(state="disabled", text="⏹ STOP"))
        self._log("Factory Run Complete.")