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
import threading
import os
import fitz  # PyMuPDF
from datetime import datetime
import shutil
import concurrent.futures
import queue
import time


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "PDF Health Scanner"
        self.running = False
        self.stop_requested = False

        # --- QUEUE INFRASTRUCTURE ---
        self.update_queue = queue.Queue()

        # Settings
        self.folder_path = tk.StringVar(value=self.app.paths.get("data", "D:/Training_Data"))
        self.auto_repair = tk.BooleanVar(value=True)
        self.delete_original = tk.BooleanVar(value=False)
        self.move_bad = tk.BooleanVar(value=True)
        self.num_threads = tk.IntVar(value=8)
        self.auto_scroll = tk.BooleanVar(value=True)

        self.max_log_lines = 1000

        self._setup_ui()

        # Start GUI Poller
        if self.parent:
            self.parent.after(50, self._process_gui_queue)

    def _setup_ui(self):
        # 1. Controls
        top = ttk.LabelFrame(self.parent, text="PDF Repair Controls", padding=15)
        top.pack(fill="x", padx=20, pady=10)

        row1 = ttk.Frame(top)
        row1.pack(fill="x", pady=5)
        ttk.Entry(row1, textvariable=self.folder_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row1, text="📂 Choose Folder", command=self._browse).pack(side="left", padx=2)

        self.btn_scan = ttk.Button(row1, text="SCAN & REPAIR", command=self._start_scan)
        self.btn_scan.pack(side="left", padx=5)

        opts = ttk.Frame(top)
        opts.pack(fill="x", pady=10)
        ttk.Checkbutton(opts, text="Auto-Repair Corrupt PDFs", variable=self.auto_repair).pack(side="left")
        ttk.Checkbutton(opts, text="Delete Original", variable=self.delete_original).pack(side="left", padx=10)
        ttk.Checkbutton(opts, text="Move Unfixable", variable=self.move_bad).pack(side="left", padx=10)

        ttk.Label(opts, text="|  Threads:").pack(side="left", padx=(10, 5))
        ttk.Spinbox(opts, from_=1, to=32, textvariable=self.num_threads, width=3).pack(side="left")

        # 2. Log
        log_frame = ttk.LabelFrame(self.parent, text="Repair Log", padding=15)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Toolbar
        tool_fr = ttk.Frame(log_frame)
        tool_fr.pack(fill="x")
        ttk.Checkbutton(tool_fr, text="Autoscroll", variable=self.auto_scroll).pack(side="left")
        ttk.Button(tool_fr, text="Clear Log", command=self._clear_log, width=10).pack(side="right")

        self.log_box = tk.Text(log_frame, font=("Consolas", 9), bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"])
        self.log_box.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self.log_box.tag_config("info", foreground=self.app.colors["FG_TEXT"])
        self.log_box.tag_config("success", foreground=self.app.colors["SUCCESS"])
        self.log_box.tag_config("fixed", foreground="#FFD700")  # Gold
        self.log_box.tag_config("error", foreground=self.app.colors["ERROR"])
        self.log_box.tag_config("warn", foreground=self.app.colors["WARN"])

    # --- QUEUE PROCESSING (CRITICAL FIX) ---
    def _process_gui_queue(self):
        """Executes queued updates. Rate limited to prevent GUI freeze."""
        MAX_UPDATES_PER_TICK = 50  # Process max 50 items per frame

        updates_processed = 0
        while not self.update_queue.empty() and updates_processed < MAX_UPDATES_PER_TICK:
            try:
                func = self.update_queue.get_nowait()
                func()
                updates_processed += 1
            except queue.Empty:
                break
            except Exception:
                pass

        # Reschedule immediately
        if self.parent:
            self.parent.after(50, self._process_gui_queue)

    def _log(self, msg, tag="info"):
        """Queues a log message"""
        if self.parent:
            self.update_queue.put(lambda: self._internal_log(msg, tag))
        else:
            print(f"[{tag.upper()}] {msg}")

    def _internal_log(self, msg, tag):
        """Actual text widget update (Main Thread Only)"""
        if not hasattr(self, 'log_box'): return

        ts = datetime.now().strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n", tag)

        if self.auto_scroll.get():
            self.log_box.see(tk.END)

        # Trim aggressively to prevent memory leak
        try:
            count = int(self.log_box.index('end-1c').split('.')[0])
            if count > self.max_log_lines + 50:
                self.log_box.delete("1.0", f"{count - self.max_log_lines}.0")
        except:
            pass

    def _clear_log(self):
        self.log_box.delete("1.0", tk.END)

    def _browse(self):
        d = filedialog.askdirectory()
        if d: self.folder_path.set(d)

    def _start_scan(self):
        if self.running:
            self.stop_requested = True
            self.btn_scan.config(text="STOPPING...")
            return

        folder = self.folder_path.get()
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Folder not found")
            return

        self.running = True
        self.stop_requested = False
        self.btn_scan.config(text="STOP SCAN")

        # Start Manager
        threading.Thread(target=self._manager_thread, args=(folder,), daemon=True).start()

    def _task_process_pdf(self, pdf_path, bad_dir):
        """
        Thread-safe worker function.
        """
        res = {"path": pdf_path, "status": "ok", "msg": "", "tag": "info"}
        name = os.path.basename(pdf_path)
        is_corrupt = False

        # 0. Pre-check: Zero Byte File
        try:
            if os.path.getsize(pdf_path) == 0:
                is_corrupt = True
                res["msg"] = "Zero Byte File (Empty)"
        except OSError as e:
            return {"path": pdf_path, "status": "skip", "msg": f"OS Error: {e}", "tag": "warn"}

        # 1. Integrity Check (MuPDF)
        if not is_corrupt:
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                is_corrupt = True
                res["msg"] = f"Header Invalid: {str(e)[:50]}"

        if not is_corrupt:
            try:
                # Stress test pages
                for i, page in enumerate(doc):
                    try:
                        # Fast render test (low DPI is sufficient to trigger stream errors)
                        pix = page.get_pixmap(dpi=36)
                    except Exception as e:
                        is_corrupt = True
                        res["msg"] = f"Pg {i + 1} Render Fail: {str(e)[:50]}"
                        break  # Fast fail
            except Exception as e:
                is_corrupt = True
                res["msg"] = f"Structure Invalid: {str(e)[:50]}"
            finally:
                if 'doc' in locals(): doc.close()

        # 2. Repair Logic
        if is_corrupt:
            res["status"] = "corrupt"
            res["tag"] = "error"

            if self.auto_repair.get():
                try:
                    # Attempt Repair
                    doc = fitz.open(pdf_path)
                    clean_path = pdf_path.replace(".pdf", "_REPAIRED.pdf")

                    # garbage=4: deduplicate + remove unused objects
                    # clean=True: sanitize syntax
                    # deflate=True: re-compress streams (fixes bad zlib blocks)
                    doc.save(clean_path, garbage=4, deflate=True, clean=True)
                    doc.close()

                    # Verify Repair
                    try:
                        check = fitz.open(clean_path)
                        for p in check: p.get_pixmap(dpi=36)
                        check.close()

                        # Apply
                        if self.delete_original.get():
                            os.remove(pdf_path)
                            os.rename(clean_path, pdf_path)
                            res["msg"] += " -> FIXED & Replaced."
                        else:
                            res["msg"] += f" -> SAVED CLEAN COPY."

                        res["status"] = "fixed"
                        res["tag"] = "fixed"

                    except:
                        # Repair generated bad file
                        if os.path.exists(clean_path): os.remove(clean_path)
                        raise Exception("Repair verification failed (Still corrupt)")

                except Exception as e:
                    res["msg"] += f" -> Repair Failed: {str(e)[:50]}"

                    if self.move_bad.get() and os.path.exists(bad_dir):
                        try:
                            dest = os.path.join(bad_dir, name)
                            shutil.move(pdf_path, dest)
                            res["msg"] += " -> Quarantined."
                        except:
                            res["msg"] += " -> Move Failed."

        return res

    def _manager_thread(self, folder):
        bad_dir = os.path.join(folder, "Bad_PDFs")
        if self.move_bad.get() and not os.path.exists(bad_dir):
            try:
                os.makedirs(bad_dir)
            except:
                pass

        # 1. Collect
        self._log("Scanning directory tree...", "info")
        pdfs = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith('.pdf'):
                    pdfs.append(os.path.join(root, f))

        total = len(pdfs)
        self._log(f"Found {total} PDFs. Launching {self.num_threads.get()} worker threads...", "info")

        fixed_count = 0
        corrupt_count = 0

        # 2. CHUNKED EXECUTION (Prevent memory flood)
        # We process in batches of 100 so we don't spawn 7000 futures at once
        chunk_size = 100

        for i in range(0, total, chunk_size):
            if self.stop_requested: break

            chunk = pdfs[i: i + chunk_size]

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads.get()) as executor:
                futures = {executor.submit(self._task_process_pdf, p, bad_dir): p for p in chunk}

                for future in concurrent.futures.as_completed(futures):
                    if self.stop_requested:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    try:
                        res = future.result()

                        # Only log if interesting (not "ok")
                        if res["status"] != "ok":
                            name = os.path.basename(res["path"])
                            self._log(f"{name}: {res['msg']}", res["tag"])

                            if res["status"] == "fixed":
                                fixed_count += 1
                            elif res["status"] == "corrupt":
                                corrupt_count += 1

                    except Exception as e:
                        self._log(f"Thread Error: {e}", "error")

            # Batch Progress Log
            progress = min(i + chunk_size, total)
            self._log(f"Progress: {progress}/{total} checked...", "info")

        self.running = False
        self._log(f"DONE. Scanned {total}. Fixed: {fixed_count}. Unfixable: {corrupt_count}.", "success")

        # Reset Button via Queue
        if self.parent:
            self.update_queue.put(lambda: self.btn_scan.config(text="SCAN & REPAIR"))

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])