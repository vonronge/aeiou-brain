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
import queue
import json
import time
import subprocess
import traceback


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "PDF Health Scanner"
        self.running = False
        self.stop_requested = False

        # --- PATHS (via Phagus) ---
        self.data_dir = self.app.paths["data"]
        self.db_path = os.path.join(self.data_dir, "pdf_health_map.json")
        self.bad_dir = os.path.join(self.data_dir, "Bad_PDFs")

        # --- SETTINGS ---
        self.folder_path = tk.StringVar(value=self.data_dir)
        self.auto_repair = tk.BooleanVar(value=True)
        self.delete_orig = tk.BooleanVar(value=False)
        self.move_bad = tk.BooleanVar(value=True)
        self.auto_scroll = tk.BooleanVar(value=True)

        # --- STATE ---
        self.db = {}
        self.stats = {"OK": 0, "NEW": 0, "CORRUPT": 0, "REPAIRED": 0, "CRASH": 0, "FAIL": 0}

        # Local UI Update Queue
        self.update_queue = queue.Queue()
        self.log_lines = 0
        self.MAX_LOG = 1000

        self._setup_ui()
        self._load_db()
        self._recover_crashes()

        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        # 1. CONTROLS
        top = ttk.LabelFrame(self.parent, text="Database & Controls", padding=15)
        top.pack(fill="x", padx=20, pady=10)

        # Path Row
        r1 = ttk.Frame(top)
        r1.pack(fill="x", pady=5)
        ttk.Entry(r1, textvariable=self.folder_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="📂", width=4, command=self._browse).pack(side="left")

        # Action Buttons
        r2 = ttk.Frame(top)
        r2.pack(fill="x", pady=10)

        self.btn_inventory = ttk.Button(r2, text="1. UPDATE INVENTORY", command=self._start_inventory)
        self.btn_inventory.pack(side="left", fill="x", expand=True, padx=5)

        self.btn_process = ttk.Button(r2, text="2. PROCESS QUEUE", command=self._start_processing)
        self.btn_process.pack(side="left", fill="x", expand=True, padx=5)

        self.btn_retry = ttk.Button(r2, text="RETRY FAILED", command=self._retry_failures)
        self.btn_retry.pack(side="left", fill="x", expand=True, padx=5)

        self.btn_stop = ttk.Button(r2, text="STOP", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        # Toggles & Tools
        r3 = ttk.Frame(top)
        r3.pack(fill="x", pady=5)
        ttk.Checkbutton(r3, text="Auto-Repair", variable=self.auto_repair).pack(side="left", padx=5)
        ttk.Checkbutton(r3, text="Delete Originals", variable=self.delete_orig).pack(side="left", padx=10)
        ttk.Checkbutton(r3, text="Quarantine Bad Files", variable=self.move_bad).pack(side="left", padx=10)

        # Tools
        ttk.Button(r3, text="📂 BAD PDFS", command=self._open_quarantine).pack(side="right", padx=5)
        ttk.Button(r3, text="🔓 FORCE UNLOCK", command=self._force_unlock).pack(side="right", padx=5)

        # 2. STATS DASHBOARD
        stat_fr = ttk.Frame(self.parent)
        stat_fr.pack(fill="x", padx=20)

        self.lbl_stats = ttk.Label(stat_fr, text="DB Loaded. Ready.", font=("Segoe UI", 10, "bold"),
                                   foreground=self.app.colors["ACCENT"])
        self.lbl_stats.pack(anchor="w")

        # 3. LOG
        log_fr = ttk.LabelFrame(self.parent, text="Activity Log", padding=10)
        log_fr.pack(fill="both", expand=True, padx=20, pady=10)

        # Log Toolbar
        log_tools = ttk.Frame(log_fr)
        log_tools.pack(fill="x")
        ttk.Checkbutton(log_tools, text="Autoscroll", variable=self.auto_scroll).pack(side="right")
        ttk.Label(log_tools, text="Log History:", font=("Segoe UI", 9, "bold")).pack(side="left")

        self.log_box = tk.Text(log_fr, font=("Consolas", 9), bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"])
        self.log_box.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(log_fr, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # Log Coloring
        tags = {
            "info": self.app.colors["FG_TEXT"],
            "ok": self.app.colors["SUCCESS"],
            "new": self.app.colors["ACCENT"],
            "bad": self.app.colors["ERROR"],
            "warn": self.app.colors["WARN"],
            "fix": "#FFD700"
        }
        for tag, col in tags.items():
            self.log_box.tag_config(tag, foreground=col)

    # --- ACTIONS ---
    def _open_quarantine(self):
        if not os.path.exists(self.bad_dir):
            os.makedirs(self.bad_dir)
        try:
            if os.name == 'nt':
                os.startfile(self.bad_dir)
            else:
                subprocess.call(["xdg-open", self.bad_dir])
        except Exception as e:
            self._log(f"Could not open folder: {e}", "warn")

    def _force_unlock(self):
        """Kills common PDF viewers to release file handles"""
        if messagebox.askyesno("Force Unlock",
                               "This will terminate Acrobat, Edge, and Chrome processes to unlock files.\nSave your work elsewhere first.\nProceed?"):
            try:
                if os.name == 'nt':
                    targets = ["Acrobat.exe", "AcroRd32.exe", "msedge.exe", "chrome.exe"]
                    for t in targets:
                        subprocess.call(f"taskkill /IM {t} /F", shell=True, stderr=subprocess.DEVNULL)
                    self._log("Force Unlock command sent.", "warn")
                else:
                    self._log("Force Unlock is Windows-only.", "info")
            except Exception as e:
                self._log(f"Unlock Failed: {e}", "bad")

    # --- DATABASE LOGIC ---
    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.db = json.load(f)
                self._update_stats()
                self._log(f"Database loaded: {len(self.db)} records.", "info")
            except:
                self.db = {}
                self._log("Database corrupt or missing. Starting fresh.", "warn")
        else:
            self._log("No database found. Please run Inventory.", "info")

    def _save_db(self):
        """Atomic save to prevent corruption on crash"""
        temp = self.db_path + ".tmp"
        try:
            with open(temp, 'w') as f:
                json.dump(self.db, f, indent=1)
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            os.rename(temp, self.db_path)
        except Exception as e:
            print(f"DB Save Error: {e}")

    def _recover_crashes(self):
        """Check if we died while processing a file last time"""
        recovered = 0
        for path, data in self.db.items():
            if data['status'] in ["PROCESSING", "REPAIRING"]:
                self.db[path]['status'] = "CRASH_SUSPECT"
                self.db[path]['log'] = "Process crashed while handling this file."
                recovered += 1

        if recovered > 0:
            self._save_db()
            self._log(f"⚠️ RECOVERY: Marked {recovered} files as CRASH SUSPECTS. Skipping them.", "bad")
            self._update_stats()

    def _retry_failures(self):
        """Resets FAIL, CRASH_SUSPECT, and QUARANTINED to NEW"""
        if self.running: return
        count = 0
        for path in self.db:
            if self.db[path]['status'] in ["FAIL", "CRASH_SUSPECT", "CORRUPT", "QUARANTINED"]:
                self.db[path]['status'] = "NEW"
                self.db[path]['log'] = "Manual Retry Requested"
                count += 1

        self._save_db()
        self._update_stats()
        self._log(f"Reset {count} items to NEW. Click 'Process Queue' to try again.", "info")

    def _update_stats(self):
        counts = {"OK": 0, "NEW": 0, "CORRUPT": 0, "REPAIRED": 0, "CRASH": 0, "FAIL": 0}
        for d in self.db.values():
            s = d.get('status', 'NEW')
            if "CRASH" in s: s = "CRASH"
            if s in counts: counts[s] += 1

        txt = f"TOTAL: {len(self.db)}  |  ✅ OK: {counts['OK']}  |  🆕 NEW: {counts['NEW']}  |  ❌ BAD: {counts['CORRUPT']}  |  🔧 FIXED: {counts['REPAIRED']}  |  ⚠️ FAIL/CRASH: {counts['FAIL'] + counts['CRASH']}"
        self.update_queue.put(lambda: self.lbl_stats.config(text=txt))

    # --- INVENTORY WORKER ---
    def _start_inventory(self):
        if self.running: return
        self.running = True
        self.stop_requested = False
        self._toggle_ui(False)
        threading.Thread(target=self._worker_inventory, daemon=True).start()

    def _worker_inventory(self):
        root_dir = self.folder_path.get()
        self._log(f"Scanning directory: {root_dir}...", "info")

        found = 0
        new = 0

        current_files = set()
        for root, _, files in os.walk(root_dir):
            if self.stop_requested: break
            for f in files:
                if f.lower().endswith(".pdf"):
                    path = os.path.join(root, f).replace("\\", "/")
                    current_files.add(path)

                    if path not in self.db:
                        self.db[path] = {
                            "status": "NEW",
                            "last_checked": None,
                            "log": "Discovered"
                        }
                        new += 1
                    found += 1

                    if found % 500 == 0:
                        self._log(f"Scanned {found} files...", "info")

        # Cleanup Orphans
        to_remove = [k for k in self.db if k not in current_files]
        for k in to_remove:
            del self.db[k]

        self._save_db()
        self._update_stats()
        self._log(f"Inventory Complete. Found {found}, Added {new}, Removed {len(to_remove)}.", "ok")
        self.running = False
        self.update_queue.put(lambda: self._toggle_ui(True))

    # --- PROCESSING WORKER ---
    def _start_processing(self):
        if self.running: return
        self.running = True
        self.stop_requested = False
        self._toggle_ui(False)

        if self.move_bad.get() and not os.path.exists(self.bad_dir):
            try:
                os.makedirs(self.bad_dir)
            except:
                pass

        threading.Thread(target=self._worker_process, daemon=True).start()

    def _worker_process(self):
        tasks = []
        for path, data in self.db.items():
            s = data['status']
            if s in ["NEW", "CORRUPT"]:
                tasks.append(path)

        total = len(tasks)
        self._log(f"Processing Queue: {total} files.", "info")

        for i, path in enumerate(tasks):
            if self.stop_requested: break

            # --- CRITICAL: MARK PROCESSING ---
            self.db[path]['status'] = "PROCESSING"
            self._save_db()

            name = os.path.basename(path)
            if i % 10 == 0: self.update_queue.put(
                lambda p=i, t=total: self.lbl_stats.config(text=f"Processing {p}/{t}..."))

            # --- CHECK HEALTH ---
            is_valid = False
            error_msg = ""

            if not os.path.exists(path):
                del self.db[path]
                continue

            try:
                # 1. Zero Byte Check
                if os.path.getsize(path) == 0:
                    raise Exception("Zero Byte File")

                # 2. MuPDF Open & Render Test
                doc = fitz.open(path)
                # Fast Check: First and Last page render
                for p_idx in [0, len(doc) - 1]:
                    if p_idx >= 0: doc[p_idx].get_pixmap(dpi=20)
                doc.close()
                is_valid = True

            except Exception as e:
                error_msg = str(e)
                is_valid = False

            # --- HANDLE RESULTS ---
            if is_valid:
                self.db[path]['status'] = "OK"
                self.db[path]['last_checked'] = datetime.now().isoformat()
                self._log(f"OK: {name}", "ok")
            else:
                self._log(f"BAD: {name} ({error_msg})", "bad")

                if self.auto_repair.get():
                    self._attempt_repair(path, name)
                else:
                    self.db[path]['status'] = "CORRUPT"
                    self.db[path]['log'] = error_msg

            # Save every 5 files
            if i % 5 == 0:
                self._save_db()
                self._update_stats()

        self._save_db()
        self._update_stats()
        self.running = False
        self._log("Processing Queue Complete.", "ok")
        self.update_queue.put(lambda: self._toggle_ui(True))

    def _attempt_repair(self, path, name):
        self.db[path]['status'] = "REPAIRING"

        try:
            # Re-save Strategy
            doc = fitz.open(path)
            clean_path = path.replace(".pdf", "_REPAIRED.pdf")
            doc.save(clean_path, garbage=4, deflate=True, clean=True)
            doc.close()

            # Verify
            try:
                check = fitz.open(clean_path)
                check[0].get_pixmap(dpi=20)
                check.close()
            except:
                if os.path.exists(clean_path): os.remove(clean_path)
                raise Exception("Repair failed verification")

            # Apply
            if self.delete_orig.get():
                try:
                    os.remove(path)
                    os.rename(clean_path, path)
                    self.db[path]['status'] = "REPAIRED"
                    self.db[path]['log'] = "Repaired in-place."
                    self._log(f" -> REPAIRED (Replaced): {name}", "fix")
                except OSError as e:
                    self._log(f" -> Replace Failed (Locked?): {e}", "warn")
                    self.db[path]['status'] = "FAIL"
            else:
                self.db[path]['status'] = "CORRUPT"
                self.db[clean_path] = {"status": "REPAIRED", "log": "Created from corrupt source"}
                self._log(f" -> REPAIRED (Saved Copy): {name}", "fix")

        except Exception as e:
            self.db[path]['status'] = "FAIL"
            self.db[path]['log'] = f"Repair Failed: {e}"
            self._log(f" -> Repair Failed: {e}", "bad")

            if self.move_bad.get():
                try:
                    dest = os.path.join(self.bad_dir, name)
                    shutil.move(path, dest)
                    self.db[path]['status'] = "QUARANTINED"
                    self._log(f" -> Moved to Quarantine.", "warn")
                except OSError as e:
                    self._log(f" -> MOVE FAILED (Locked?): {e}", "bad")

    # --- GUI HELPERS ---
    def _browse(self):
        d = filedialog.askdirectory()
        if d: self.folder_path.set(d)

    def _stop(self):
        self.stop_requested = True
        self.btn_stop.config(text="STOPPING...")

    def _toggle_ui(self, enable):
        state = "normal" if enable else "disabled"
        self.btn_inventory.config(state=state)
        self.btn_process.config(state=state)
        self.btn_retry.config(state=state)
        self.btn_stop.config(state="normal" if not enable else "disabled")
        if enable: self.btn_stop.config(text="STOP")

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                func = self.update_queue.get_nowait()
                func()
            except:
                break

        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _log(self, msg, tag="info"):
        """Logs to local UI + System Golgi."""
        # 1. System Log (for CLI visibility)
        if self.app.golgi:
            # Map GUI tag to Golgi level
            level = "INFO"
            if tag in ["bad", "fail", "error"]:
                level = "ERROR"
            elif tag in ["warn"]:
                level = "WARN"
            elif tag in ["ok", "fix"]:
                level = "SUCCESS"

            # Avoid spamming the Golgi with every single file check, only failures/fixes
            if level != "INFO" or "Queue" in msg:
                self.app.golgi._dispatch(level, msg, source="PDFMedic")

        # 2. Local GUI Log (Detailed)
        self.update_queue.put(lambda: self._internal_log(msg, tag))

    def _internal_log(self, msg, tag):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n", tag)
        if self.auto_scroll.get(): self.log_box.see(tk.END)

        # Trim
        if self.log_lines > self.MAX_LOG:
            self.log_box.delete("1.0", "2.0")
        else:
            self.log_lines += 1

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])