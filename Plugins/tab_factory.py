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
from tkinter import ttk, messagebox, filedialog
import os
import threading
import shutil
import time
import queue
import traceback


# --- HEADLESS HELPER ---
class MockVar:
    def __init__(self, value=None): self._val = value

    def set(self, value): self._val = value

    def get(self): return self._val


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Data Factory"
        self.running = False

        # --- DYNAMIC PATHS ---
        # Get default from GUI.py logic (configurable via settings.json)
        default_data = self.app.paths.get("data", os.path.join(self.app.paths["root"], "Training_Data"))

        # Default Source: "Chaos_Buffer" inside Training_Data
        chaos_default = os.path.join(default_data, "Chaos_Buffer")
        if not os.path.exists(chaos_default):
            try:
                os.makedirs(chaos_default)
            except:
                pass

        # --- INITIALIZE VARIABLES ---
        if self.parent is None:
            # Headless Mode
            self.chaos_folder = MockVar(chaos_default)
            self.target_folder = MockVar(default_data)
            self.status_var = MockVar("Ready.")
            self.progress_var = MockVar(0.0)
        else:
            # GUI Mode
            self.chaos_folder = tk.StringVar(value=chaos_default)
            self.target_folder = tk.StringVar(value=default_data)
            self.status_var = tk.StringVar(value="Ready.")
            self.progress_var = tk.DoubleVar(value=0.0)

        self._setup_ui()

    def _setup_ui(self):
        if self.parent is None: return

        # 1. Source (Chaos)
        fr_src = ttk.LabelFrame(self.parent, text="Incoming Data (Source)", padding=15)
        fr_src.pack(fill="x", padx=10, pady=10)

        ttk.Label(fr_src, text="Source Folder (The Chaos):").pack(anchor="w")
        row_src = ttk.Frame(fr_src)
        row_src.pack(fill="x", pady=5)

        e_chaos = ttk.Entry(row_src, textvariable=self.chaos_folder)
        e_chaos.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(row_src, text="📂", width=4, command=lambda: self._browse(self.chaos_folder)).pack(side="left")

        # 2. Target (Organized)
        fr_tgt = ttk.LabelFrame(self.parent, text="Structured Storage (Target)", padding=15)
        fr_tgt.pack(fill="x", padx=10, pady=10)

        ttk.Label(fr_tgt, text="Destination Folder:").pack(anchor="w")
        row_tgt = ttk.Frame(fr_tgt)
        row_tgt.pack(fill="x", pady=5)

        e_tgt = ttk.Entry(row_tgt, textvariable=self.target_folder)
        e_tgt.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(row_tgt, text="📂", width=4, command=lambda: self._browse(self.target_folder)).pack(side="left")

        # 3. Operations
        fr_ops = ttk.LabelFrame(self.parent, text="Factory Operations", padding=15)
        fr_ops.pack(fill="both", expand=True, padx=10, pady=10)

        btn_grid = ttk.Frame(fr_ops)
        btn_grid.pack(fill="x", pady=10)

        self.btn_sort = ttk.Button(btn_grid, text="✨ SORT & CLEAN", command=self._start_sort)
        self.btn_sort.pack(side="left", fill="x", expand=True, padx=5)

        self.btn_flat = ttk.Button(btn_grid, text="🔨 FLATTEN DIRECTORY", command=self._start_flatten)
        self.btn_flat.pack(side="left", fill="x", expand=True, padx=5)

        # Status & Progress
        self.pb = ttk.Progressbar(fr_ops, variable=self.progress_var, maximum=100)
        self.pb.pack(fill="x", pady=(20, 5))

        lbl_stat = ttk.Label(fr_ops, textvariable=self.status_var, foreground=self.app.colors["ACCENT"],
                             font=("Segoe UI", 10))
        lbl_stat.pack()

    def _browse(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d: var.set(d)

    def _toggle_ui(self, state):
        if self.parent is None: return
        s = "normal" if state else "disabled"
        self.btn_sort.config(state=s)
        self.btn_flat.config(state=s)

    # --- SORT LOGIC ---
    def _start_sort(self):
        if self.running: return

        src = self.chaos_folder.get()
        tgt = self.target_folder.get()

        if not os.path.exists(src):
            if self.parent:
                messagebox.showerror("Error", "Source folder does not exist.")
            else:
                print("Error: Source folder missing.")
            return

        self.running = True
        self._toggle_ui(False)
        self.status_var.set("Initializing Sorter...")
        self.progress_var.set(0)

        threading.Thread(target=self._worker_sort, args=(src, tgt), daemon=True).start()

    def _worker_sort(self, src, tgt):
        try:
            if not os.path.exists(tgt): os.makedirs(tgt)

            # Definitions
            categories = {
                'Images': ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'],
                'Audio': ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'],
                'Text': ['.txt', '.md', '.pdf', '.epub', '.json', '.csv', '.doc', '.docx', '.srt', '.vtt', '.ass'],
                'Video': ['.mp4', '.mkv', '.avi', '.mov', '.webm'],
                'Code': ['.py', '.js', '.html', '.css', '.cpp', '.h', '.java'],
                'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz']
            }

            # Create subfolders
            for cat in categories:
                p = os.path.join(tgt, cat)
                if not os.path.exists(p): os.makedirs(p)

            # Scan
            files_to_move = []
            for root, _, files in os.walk(src):
                for f in files:
                    files_to_move.append(os.path.join(root, f))

            total = len(files_to_move)
            if total == 0:
                if self.parent:
                    self.parent.after(0, lambda: self.status_var.set("Source is empty."))
                    self.parent.after(0, lambda: self._toggle_ui(True))
                else:
                    print("Source is empty.")
                self.running = False
                return

            moved_count = 0

            for i, file_path in enumerate(files_to_move):
                _, ext = os.path.splitext(file_path)
                lext = ext.lower()

                dest_cat = "Misc"
                for cat, exts in categories.items():
                    if lext in exts:
                        dest_cat = cat
                        break

                # Setup destination
                dest_dir = os.path.join(tgt, dest_cat)
                if not os.path.exists(dest_dir): os.makedirs(dest_dir)

                fname = os.path.basename(file_path)
                dest_path = os.path.join(dest_dir, fname)

                # Handle collision
                if os.path.exists(dest_path):
                    base, ex = os.path.splitext(fname)
                    ts = int(time.time() * 1000)
                    dest_path = os.path.join(dest_dir, f"{base}_{ts}{ex}")

                try:
                    shutil.move(file_path, dest_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Move failed for {fname}: {e}")

                # UI Update
                if self.parent and i % 5 == 0:
                    pct = (i / total) * 100
                    self.parent.after(0, lambda p=pct, m=moved_count: [
                        self.progress_var.set(p),
                        self.status_var.set(f"Sorting... ({m}/{total})")
                    ])

            # Cleanup Empty Dirs in Source
            for root, dirs, files in os.walk(src, topdown=False):
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except:
                        pass

            if self.parent:
                self.parent.after(0, lambda m=moved_count: self.status_var.set(f"Done! Organized {m} files."))
                self.parent.after(0, lambda: self.progress_var.set(100))
            else:
                print(f"Done! Organized {moved_count} files.")

        except Exception as e:
            traceback.print_exc()
            if self.parent:
                self.parent.after(0, lambda err=str(e): self.status_var.set(f"Error: {err}"))
        finally:
            self.running = False
            if self.parent:
                self.parent.after(0, lambda: self._toggle_ui(True))

    # --- FLATTEN LOGIC ---
    def _start_flatten(self):
        if self.running: return

        tgt = self.target_folder.get()
        if not os.path.exists(tgt): return

        if self.parent:
            if not messagebox.askyesno("Confirm Flatten",
                                       f"This will move ALL files from subfolders of:\n{tgt}\n\nInto the root of that folder.\nAre you sure?"):
                return

        self.running = True
        self._toggle_ui(False)
        self.status_var.set("Flattening...")
        self.progress_var.set(0)

        threading.Thread(target=self._worker_flatten, args=(tgt,), daemon=True).start()

    def _worker_flatten(self, root_dir):
        try:
            files_to_move = []
            # Walk topdown=False so we can delete dirs after
            for root, dirs, files in os.walk(root_dir, topdown=False):
                if root == root_dir: continue  # Skip root files, they are fine

                for f in files:
                    files_to_move.append(os.path.join(root, f))

            total = len(files_to_move)
            moved = 0

            for i, src_path in enumerate(files_to_move):
                fname = os.path.basename(src_path)
                dest_path = os.path.join(root_dir, fname)

                # Collision handling
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(fname)
                    # Use parent folder name to disambiguate
                    parent_name = os.path.basename(os.path.dirname(src_path))
                    dest_path = os.path.join(root_dir, f"{base}_{parent_name}{ext}")

                    # Double check
                    if os.path.exists(dest_path):
                        ts = int(time.time())
                        dest_path = os.path.join(root_dir, f"{base}_{ts}{ext}")

                try:
                    shutil.move(src_path, dest_path)
                    moved += 1
                except:
                    pass

                if self.parent and i % 10 == 0:
                    pct = (i / total) * 100
                    self.parent.after(0, lambda p=pct: self.progress_var.set(p))

            # Cleanup Empty Dirs
            cleaned = 0
            for root, dirs, files in os.walk(root_dir, topdown=False):
                if root == root_dir: continue
                try:
                    os.rmdir(root)
                    cleaned += 1
                except:
                    pass

            if self.parent:
                self.parent.after(0, lambda m=moved, c=cleaned: self.status_var.set(
                    f"Flattened {m} files. Removed {c} folders."))
                self.parent.after(0, lambda: self.progress_var.set(100))
            else:
                print(f"Flattened {moved} files.")

        except Exception as e:
            if self.parent:
                self.parent.after(0, lambda err=str(e): self.status_var.set(f"Error: {err}"))
        finally:
            self.running = False
            if self.parent:
                self.parent.after(0, lambda: self._toggle_ui(True))

    def on_theme_change(self):
        pass