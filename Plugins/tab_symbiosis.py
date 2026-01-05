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
from tkinter import ttk, messagebox
import threading
import queue
import time
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Neural Symbiosis"

        # --- STATE ---
        self.is_running = False
        self.stop_requested = False
        self.update_queue = queue.Queue()

        # Config Vars
        self.teacher_id = tk.IntVar(value=1)
        self.student_id = tk.IntVar(value=2)

        # Distillation Params
        self.temperature = tk.DoubleVar(value=2.0)
        self.alpha = tk.DoubleVar(value=0.5)  # 0.5 = Equal mix of Hard/Soft loss
        self.harvest_memories = tk.BooleanVar(value=True)  # Use Hippocampus replay?
        self.auto_save = tk.BooleanVar(value=True)

        self.status_msg = tk.StringVar(value="Ready to Link.")

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        scale = getattr(self.app, 'ui_scale', 1.0)

        # 1. LINK CONFIGURATION
        fr_link = ttk.LabelFrame(self.parent, text="Neural Link Configuration", padding=15)
        fr_link.pack(fill="x", padx=10, pady=10)

        # Teacher Selection
        f_teach = ttk.Frame(fr_link)
        f_teach.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(f_teach, text="Teacher Lobe (Source):", font=("Segoe UI", int(10 * scale), "bold")).pack(anchor="w")

        for i in range(1, 5):
            r = ttk.Radiobutton(f_teach, text=f"Lobe {i}", variable=self.teacher_id, value=i,
                                command=self._validate_selection)
            r.pack(anchor="w", pady=2)

        # Separator (Arrow)
        ttk.Label(fr_link, text="➔", font=("Segoe UI", int(20 * scale))).pack(side="left", padx=10)

        # Student Selection
        f_stud = ttk.Frame(fr_link)
        f_stud.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(f_stud, text="Student Lobe (Target):", font=("Segoe UI", int(10 * scale), "bold")).pack(anchor="w")

        for i in range(1, 5):
            r = ttk.Radiobutton(f_stud, text=f"Lobe {i}", variable=self.student_id, value=i,
                                command=self._validate_selection)
            r.pack(anchor="w", pady=2)

        # 2. HYPERPARAMETERS
        fr_param = ttk.LabelFrame(self.parent, text="Distillation Parameters", padding=15)
        fr_param.pack(fill="x", padx=10, pady=5)

        # Temperature
        r1 = ttk.Frame(fr_param)
        r1.pack(fill="x", pady=5)
        ttk.Label(r1, text="Temperature (Softness):").pack(side="left", width=20)
        s_temp = ttk.Scale(r1, from_=1.0, to=10.0, variable=self.temperature, orient="horizontal")
        s_temp.pack(side="left", fill="x", expand=True)
        l_temp = ttk.Label(r1, text=f"{self.temperature.get():.1f}")
        l_temp.pack(side="left", padx=5)
        s_temp.configure(command=lambda v: l_temp.configure(text=f"{float(v):.1f}"))

        # Alpha (Balance)
        r2 = ttk.Frame(fr_param)
        r2.pack(fill="x", pady=5)
        ttk.Label(r2, text="Alpha (Teacher Weight):").pack(side="left", width=20)
        s_alp = ttk.Scale(r2, from_=0.0, to=1.0, variable=self.alpha, orient="horizontal")
        s_alp.pack(side="left", fill="x", expand=True)
        l_alp = ttk.Label(r2, text=f"{self.alpha.get():.2f}")
        l_alp.pack(side="left", padx=5)
        s_alp.configure(command=lambda v: l_alp.configure(text=f"{float(v):.2f}"))

        # Toggles
        r3 = ttk.Frame(fr_param)
        r3.pack(fill="x", pady=10)
        ttk.Checkbutton(r3, text="Harvest Hippocampus (Replay Memories)", variable=self.harvest_memories).pack(
            side="left", padx=5)
        ttk.Checkbutton(r3, text="Auto-Save Student", variable=self.auto_save).pack(side="left", padx=20)

        # 3. CONTROLS & LOGS
        fr_ctrl = ttk.Frame(self.parent, padding=10)
        fr_ctrl.pack(fill="both", expand=True)

        self.btn_start = ttk.Button(fr_ctrl, text="INITIATE SYMBIOSIS", command=self._toggle_run)
        self.btn_start.pack(fill="x", pady=5)

        # Status Bar
        self.lbl_status = ttk.Label(fr_ctrl, textvariable=self.status_msg, foreground=self.app.colors["ACCENT"],
                                    anchor="center")
        self.lbl_status.pack(fill="x", pady=5)

        # Log Box
        log_font = ("Consolas", int(10 * scale))
        self.log_box = tk.Text(fr_ctrl, font=log_font, height=8, bg=self.app.colors["BG_MAIN"],
                               fg=self.app.colors["FG_TEXT"])
        self.log_box.pack(fill="both", expand=True)

        # Attach Golgi Sink for this tab
        self.app.golgi.attach_sink("symbiosis_tab", self._on_golgi_message)

    def _validate_selection(self):
        t = self.teacher_id.get()
        s = self.student_id.get()

        if t == s:
            self.status_msg.set("⚠️ Error: Teacher and Student cannot be the same Lobe.")
            self.btn_start.config(state="disabled")
        else:
            self.status_msg.set(f"Ready: Lobe {t} ➔ Lobe {s}")
            self.btn_start.config(state="normal")

    def _toggle_run(self):
        if self.is_running:
            self.app.symbiont.stop()
            self.btn_start.config(text="STOPPING...")
            self.status_msg.set("Severing link...")
        else:
            self._start_symbiosis()

    def _start_symbiosis(self):
        t_id = self.teacher_id.get()
        s_id = self.student_id.get()

        # 1. Get Handles
        teacher = self.app.lobe_manager.get_lobe(t_id)
        student = self.app.lobe_manager.get_lobe(s_id)

        if not teacher:
            messagebox.showerror("Error", f"Teacher Lobe {t_id} is not loaded.")
            return
        if not student:
            messagebox.showerror("Error", f"Student Lobe {s_id} is not loaded.")
            return

        # 2. Configure Symbiont
        from Organelles.symbiont import SymbiosisConfig

        config = SymbiosisConfig(
            temperature=self.temperature.get(),
            alpha=self.alpha.get(),
            harvest_enabled=self.harvest_memories.get(),
            save_interval=50 if self.auto_save.get() else 0
        )

        # 3. Link & Launch
        try:
            self.app.symbiont.link(teacher, student, config)
            self.app.symbiont.start()

            self.is_running = True
            self.btn_start.config(text="SEVER CONNECTION")
            self.status_msg.set("Symbiosis Active: Knowledge Transfer in Progress...")

            # Start monitoring thread
            threading.Thread(target=self._monitor_loop, daemon=True).start()

        except Exception as e:
            self.app.golgi.error(f"Symbiosis Failed: {e}", source="Symbiosis")

    def _monitor_loop(self):
        while self.app.symbiont.is_active:
            time.sleep(0.5)
            # You could pull stats from symbiont here if exposed

        self.is_running = False
        self.update_queue.put(lambda: self.btn_start.config(text="INITIATE SYMBIOSIS"))
        self.update_queue.put(lambda: self.status_msg.set("Link Severed."))

    def _on_golgi_message(self, record):
        """Receives log messages from the Golgi apparatus."""
        if record.source == "Symbiont":
            self.update_queue.put(lambda: self._log(f"[{record.timestamp}] {record.message}"))

    def _log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                fn = self.update_queue.get_nowait()
                fn()
            except:
                break
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])