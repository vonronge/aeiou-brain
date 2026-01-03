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
import re
import traceback
import torch
from datetime import datetime

# --- SAFE IMPORTS ---
try:
    from RestrictedPython import compile_restricted, safe_globals

    HAS_SANDBOX = True
except ImportError:
    HAS_SANDBOX = False


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Local RLM"

        self.is_running = False
        self.stop_requested = False
        self.context_data = ""
        self.history = []

        # Configuration
        self.manager_id = tk.IntVar(value=1)
        self.worker_id = tk.IntVar(value=2)
        self.max_iters = tk.IntVar(value=12)
        self.temp = tk.DoubleVar(value=0.7)

        self._setup_ui()
        if not HAS_SANDBOX:
            self._log("CRITICAL: Install 'RestrictedPython' via pip.", "error")

    def _setup_ui(self):
        pane = ttk.PanedWindow(self.parent, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(pane, width=400)
        right = ttk.Frame(pane)
        pane.add(left, weight=1)
        pane.add(right, weight=3)

        # --- LEFT PANEL ---
        # 1. Config
        conf = ttk.LabelFrame(left, text="Agent Architecture", padding=10)
        conf.pack(fill="x", pady=5)

        # Manager Selector
        ttk.Label(conf, text="Manager (Code/Plan):", foreground=self.app.colors["ACCENT"]).pack(anchor="w")
        f_mgr = ttk.Frame(conf)
        f_mgr.pack(fill="x", pady=2)
        for i in range(1, 5):
            ttk.Radiobutton(f_mgr, text=f"L{i}", variable=self.manager_id, value=i).pack(side="left", padx=4)

        # Worker Selector
        ttk.Label(conf, text="Worker (Read/Scan):", foreground=self.app.colors["SUCCESS"]).pack(anchor="w",
                                                                                                pady=(10, 0))
        f_wrk = ttk.Frame(conf)
        f_wrk.pack(fill="x", pady=2)
        for i in range(1, 5):
            ttk.Radiobutton(f_wrk, text=f"L{i}", variable=self.worker_id, value=i).pack(side="left", padx=4)

        ttk.Label(conf, text="Max Iterations:").pack(anchor="w", pady=(10, 0))
        ttk.Spinbox(conf, from_=1, to=50, textvariable=self.max_iters).pack(fill="x", pady=2)

        ttk.Label(conf, text="Temperature:").pack(anchor="w")
        ttk.Scale(conf, from_=0.1, to=2.0, variable=self.temp, orient="horizontal").pack(fill="x")

        # 2. Context
        loader = ttk.LabelFrame(left, text="Context Source", padding=10)
        loader.pack(fill="both", expand=True, pady=5)

        btn_row = ttk.Frame(loader)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Load Text File", command=self._load_file).pack(side="left", fill="x", expand=True)
        ttk.Button(btn_row, text="X", width=3, command=self._clear_context).pack(side="right")

        self.lbl_context = ttk.Label(loader, text="No context.", foreground="#888", wraplength=200)
        self.lbl_context.pack(pady=5)

        # 3. Query
        q_fr = ttk.LabelFrame(left, text="Goal / Query", padding=10)
        q_fr.pack(fill="x", pady=5)
        self.txt_query = tk.Text(q_fr, height=6, font=("Segoe UI", 10))
        self.txt_query.pack(fill="x")

        self.btn_run = ttk.Button(left, text="START HIERARCHICAL REASONING", command=self._start_loop)
        self.btn_run.pack(fill="x", pady=10)

        # --- RIGHT PANEL ---
        self.log_box = tk.Text(right, font=("Consolas", 10), bg="#1E1E1E", fg="#D4D4D4", borderwidth=0)
        self.log_box.pack(fill="both", expand=True)

        self.log_box.tag_config("sys", foreground="#569CD6")
        self.log_box.tag_config("llm", foreground="#CE9178")
        self.log_box.tag_config("code", foreground="#4EC9B0")
        self.log_box.tag_config("exec", foreground="#DCDCAA")
        self.log_box.tag_config("error", foreground="#F44747")
        self.log_box.tag_config("final", foreground="#B5CEA8", font=("Consolas", 11, "bold"))

    def _log(self, msg, tag="sys"):
        self.log_box.insert(tk.END, f"\n[{tag.upper()}] {msg}\n", tag)
        self.log_box.see(tk.END)

    def _load_file(self):
        f = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if f:
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    self.context_data = file.read()
                self.lbl_context.config(text=f"{os.path.basename(f)} ({len(self.context_data)} chars)")
                self._log(f"Loaded context: {len(self.context_data)} chars.", "sys")
            except Exception as e:
                self._log(f"Load failed: {e}", "error")

    def _clear_context(self):
        self.context_data = ""
        self.lbl_context.config(text="No context.")

    def _start_loop(self):
        if self.is_running:
            self.stop_requested = True
            self.btn_run.config(text="Stopping...")
            return

        mgr_brain = self.app.lobes[self.manager_id.get()]
        wrk_brain = self.app.lobes[self.worker_id.get()]

        if not mgr_brain:
            messagebox.showerror("Error", f"Manager Lobe {self.manager_id.get()} is not loaded.")
            return
        if not wrk_brain:
            messagebox.showerror("Error", f"Worker Lobe {self.worker_id.get()} is not loaded.")
            return

        if not self.context_data:
            messagebox.showerror("Error", "Load text context first.")
            return

        self.is_running = True
        self.stop_requested = False
        self.btn_run.config(text="STOP LOOP")
        self.log_box.delete("1.0", tk.END)

        threading.Thread(target=self._rlm_worker, daemon=True).start()

    def _generate(self, lobe_id, prompt, max_new=256):
        """Generic generation helper"""
        try:
            brain = self.app.lobes[lobe_id]
            ribosome = self.app.ribosome

            p_ids = ribosome._tokenize(prompt)
            p_tensor = torch.tensor(p_ids, device=self.app.device).unsqueeze(0)

            with self.app.gpu_lock:
                brain.eval()
                with torch.no_grad():
                    if hasattr(brain, 'timestep_emb'):
                        # Diffusion
                        tokens = brain.generate(
                            prompt_tokens=p_ids,
                            max_length=len(p_ids) + max_new,
                            steps=24,
                            temperature=self.temp.get()
                        )
                    else:
                        # AR
                        t = p_tensor
                        v = torch.zeros(1, 1, 768).to(self.app.device)
                        a = torch.zeros(1, 1, 128).to(self.app.device)
                        for _ in range(max_new):
                            logits, _, _ = brain(v, a, t)
                            next_tok = torch.multinomial(
                                torch.nn.functional.softmax(logits[:, -1, :] / self.temp.get(), dim=-1), 1)
                            t = torch.cat([t, next_tok], dim=1)
                            if next_tok.item() == 50256: break
                        tokens = t[0].tolist()

            text = ribosome.decode(tokens)
            return text[len(prompt):].strip()
        except Exception as e:
            return f"Gen Error: {e}"

    def _rlm_worker(self):
        try:
            query = self.txt_query.get("1.0", tk.END).strip()
            if not query: query = "Summarize the context."

            # --- SYSTEM PROMPT (For Manager) ---
            sys_prompt = f"""
You are the MANAGER Agent.
You have a massive text file in variable `CONTEXT`.
You cannot read it directly. You must use the `worker` tool to read chunks.

TOOLS:
1. `print(val)`: Log thoughts/results.
2. `worker(slice, question)`: Ask the WORKER agent to read a specific text slice and answer a question.

GOAL: {query}

INSTRUCTIONS:
- Write valid Python code wrapped in ```python ... ```.
- Inspect `len(CONTEXT)`.
- Use string slicing to send manageable chunks to the worker.
- When you have the answer, print "FINAL: [Answer]".
"""

            # Safe Tool for Worker
            def _worker_tool(text_slice, question):
                # Worker sees only the slice + specific question
                # Limit slice size to prevent context overflow on worker
                chunk = text_slice[:3000]
                p = f"Text Chunk: {chunk}\n\nQuestion: {question}\n\nAnswer concisely:"
                self._log(f"[Worker L{self.worker_id.get()}] Reading {len(chunk)} chars...", "sys")
                return self._generate(self.worker_id.get(), p, max_new=128)

            env = safe_globals.copy()
            env['CONTEXT'] = self.context_data
            env['worker'] = _worker_tool
            env['_print_buffer'] = []
            env['print'] = lambda *args: env['_print_buffer'].append(" ".join(map(str, args)))

            current_prompt = sys_prompt

            for i in range(self.max_iters.get()):
                if self.stop_requested: break

                self._log(f"--- Cycle {i + 1} (Manager L{self.manager_id.get()}) ---", "sys")

                # 1. Manager Thinks (Code Generation)
                full_p = current_prompt + "\n\nWrite Python code for the next step:"
                llm_text = self._generate(self.manager_id.get(), full_p, max_new=300)
                self._log(llm_text, "llm")

                if "FINAL:" in llm_text:
                    self._log(f"\n{llm_text}", "final")
                    break

                # 2. Extract Code
                code_match = re.search(r"```python\n(.*?)\n```", llm_text, re.DOTALL)
                if not code_match:
                    code_match = re.search(r"```\n(.*?)\n```", llm_text, re.DOTALL)

                if code_match:
                    code = code_match.group(1)
                    self._log(f"Executing:\n{code}", "code")

                    env['_print_buffer'] = []
                    try:
                        byte_code = compile_restricted(code, '<string>', 'exec')
                        exec(byte_code, env)
                        output = "\n".join(env['_print_buffer']) or "[Done]"

                        self._log(f"Result:\n{output}", "exec")
                        current_prompt = f"Previous Code Output:\n{output}\n\nNext step?"

                    except Exception as e:
                        self._log(f"Runtime Error: {e}", "error")
                        current_prompt = f"Error: {e}. Fix the code."
                else:
                    self._log("No code block. Prompting again...", "warn")
                    current_prompt = "Please output valid Python code in ```python blocks."

        except Exception as e:
            self._log(f"Critical Failure: {e}", "error")
            traceback.print_exc()
        finally:
            self.is_running = False
            self.parent.after(0, lambda: self.btn_run.config(text="START HIERARCHICAL REASONING"))

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])