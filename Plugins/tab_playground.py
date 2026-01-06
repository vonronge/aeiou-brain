"""
AEIOU Brain — Local Multimodal AI Ecosystem
Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Playground (v25.12):
Debug Edition. Prints detailed lifecycle of a generation request.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn.functional as F
import threading
import queue
import os
import json
import uuid
import traceback
from datetime import datetime
from PIL import Image, ImageTk

class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Playground"

        self.is_generating = False
        self.stop_requested = False
        self.update_queue = queue.Queue()

        self.history = []
        self.attached_image_path = None
        self.attached_dense = None
        self.attached_tokens = None

        self.memory_file = os.path.join(self.app.paths["memories"], "episodic_chat_log.jsonl")

        # --- SETTINGS ---
        self.temperature = tk.DoubleVar(value=0.8)
        self.steps = tk.IntVar(value=30)
        self.top_k = tk.IntVar(value=50)
        self.max_tokens = tk.IntVar(value=100)
        self.do_sample = tk.BooleanVar(value=True)
        self.use_recall = tk.BooleanVar(value=True)
        self.verbose_log = tk.BooleanVar(value=True)

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return
        scale = getattr(self.app, 'ui_scale', 1.0)
        c = self.app.colors

        fr_main = ttk.Frame(self.parent)
        fr_main.pack(fill="both", expand=True, padx=10, pady=10)

        fr_hist = ttk.Frame(fr_main)
        fr_hist.pack(fill="both", expand=True)

        chat_font = ("Segoe UI", int(11 * scale))
        self.chat_box = tk.Text(fr_hist, font=chat_font, wrap="word",
                                bg=c["BG_MAIN"], fg=c["FG_TEXT"],
                                state="disabled", padx=10, pady=10)
        self.chat_box.pack(side="left", fill="both", expand=True)

        self.chat_box.tag_config("user", foreground=c["ACCENT"], font=("Segoe UI", int(11 * scale), "bold"))
        self.chat_box.tag_config("ai", foreground=c["SUCCESS"], font=("Segoe UI", int(11 * scale), "bold"))
        self.chat_box.tag_config("sys", foreground=c["FG_DIM"], font=("Segoe UI", int(10 * scale), "italic"))
        self.chat_box.tag_config("error", foreground=c["ERROR"], font=("Consolas", int(10 * scale)))
        self.chat_box.tag_config("debug", foreground=c["WARN"], font=("Consolas", int(9 * scale)))
        self.chat_box.tag_config("content", foreground=c["FG_TEXT"])

        sb = ttk.Scrollbar(fr_hist, command=self.chat_box.yview)
        sb.pack(side="right", fill="y")
        self.chat_box.config(yscrollcommand=sb.set)

        fr_input = ttk.LabelFrame(fr_main, text="Input Signal", padding=10)
        fr_input.pack(fill="x", pady=(10, 0))

        self.txt_input = tk.Text(fr_input, height=3, font=chat_font,
                                 bg=c["BG_CARD"], fg=c["FG_TEXT"],
                                 insertbackground=c["ACCENT"])
        self.txt_input.pack(fill="x", side="top", pady=(0, 5))
        self.txt_input.bind("<Shift-Return>", lambda e: "break")
        self.txt_input.bind("<Return>", self._on_enter)

        fr_ctrl = ttk.Frame(fr_input)
        fr_ctrl.pack(fill="x")

        # Attachments
        self.btn_img = ttk.Button(fr_ctrl, text="📷 Attach", command=self._attach_image)
        self.btn_img.pack(side="left")
        self.btn_clear_img = ttk.Button(fr_ctrl, text="❌", width=3, command=self._clear_attachment, state="disabled")
        self.btn_clear_img.pack(side="left", padx=2)
        self.lbl_img = ttk.Label(fr_ctrl, text="", foreground=c["FG_DIM"])
        self.lbl_img.pack(side="left", padx=5)

        # Settings
        ttk.Label(fr_ctrl, text="Temp:").pack(side="left", padx=(15, 0))
        ttk.Entry(fr_ctrl, textvariable=self.temperature, width=4).pack(side="left")

        # Steps Slider
        ttk.Label(fr_ctrl, text="Steps:").pack(side="left", padx=(10, 0))
        self.lbl_steps = ttk.Label(fr_ctrl, text=str(self.steps.get()), width=3)
        self.lbl_steps.pack(side="left")

        s_steps = tk.Scale(fr_ctrl, from_=1, to=100, variable=self.steps, orient="horizontal",
                           length=200, width=15, showvalue=0, highlightthickness=0,
                           bg=c["BG_CARD"], fg=c["FG_TEXT"], troughcolor=c["BG_MAIN"], activebackground=c["ACCENT"],
                           command=lambda v: self.lbl_steps.config(text=str(int(float(v)))))
        s_steps.pack(side="left", padx=2)

        ttk.Checkbutton(fr_ctrl, text="Recall", variable=self.use_recall).pack(side="left", padx=(10, 5))
        ttk.Checkbutton(fr_ctrl, text="Debug", variable=self.verbose_log).pack(side="left", padx=5)

        # Actions
        self.btn_send = ttk.Button(fr_ctrl, text="SEND ➤", command=self._send_message)
        self.btn_send.pack(side="right")
        self.btn_stop = ttk.Button(fr_ctrl, text="STOP", command=self._stop_gen, state="disabled")
        self.btn_stop.pack(side="right", padx=5)

    def _save_to_memory(self, user_text, ai_text):
        try:
            full_conversation = f"User: {user_text}\nLobe: {ai_text}"
            entry = {
                "timestamp": str(datetime.now()),
                "user_text": user_text,
                "ai_text": ai_text,
                "full_text": full_conversation
            }
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except: pass

    def _attach_image(self):
        f = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp")])
        if f:
            self.attached_image_path = f
            self.lbl_img.config(text=f"[{os.path.basename(f)}]")
            self.btn_clear_img.config(state="normal")
            try:
                packet = {'v': f, 'type': 'single'}
                v_feat, _, full_seq, _, _ = self.app.ribosome.ingest_packet(packet)
                self.attached_dense = v_feat
                self.attached_tokens = full_seq
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {e}")
                self._clear_attachment()

    def _clear_attachment(self):
        self.attached_image_path = None
        self.attached_dense = None
        self.attached_tokens = None
        self.lbl_img.config(text="")
        self.btn_clear_img.config(state="disabled")

    def _on_enter(self, event):
        if not self.is_generating:
            self._send_message()
        return "break"

    def _send_message(self):
        text = self.txt_input.get("1.0", tk.END).strip()
        if not text and not self.attached_image_path: return

        active_id = self.app.active_lobe.get()
        lobe = self.app.lobe_manager.get_lobe(active_id)
        if not lobe:
            self._append_chat("System", "No Lobe Loaded.", "sys")
            return

        self.txt_input.delete("1.0", tk.END)
        self._append_chat("User", text, "user")
        if self.attached_image_path:
            self._append_chat("System", f"[Image Attached]", "sys")

        self.is_generating = True
        self.stop_requested = False
        self.btn_send.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_clear_img.config(state="disabled")

        threading.Thread(target=self._worker, args=(lobe, text), daemon=True).start()

    def _stop_gen(self):
        self.stop_requested = True
        self.btn_stop.config(text="STOPPING...")

    def _worker(self, lobe, prompt_text):
        try:
            device = self.app.device
            ribosome = self.app.ribosome

            # 1. RECALL
            full_input_text = prompt_text

            # 2. TOKENIZE
            text_tokens = ribosome._tokenize(full_input_text)
            t_text = torch.tensor(text_tokens, device=device).unsqueeze(0)

            # 3. GENERATE
            is_diffusion = hasattr(lobe.model, "generate") and (lobe.model_type == "diffusion" or "Diffusion" in lobe.genome)

            self.update_queue.put(lambda: self._append_chat("AI", "", "ai"))

            if is_diffusion:
                self.update_queue.put(lambda: self._append_chat(None, "(Thinking/Diffusing...)", "sys"))
                parts = []
                if self.attached_tokens is not None:
                    parts.append(self.attached_tokens.to(device))
                parts.append(t_text)
                t_in = torch.cat(parts, dim=1)

                if self.verbose_log.get():
                     msg = f"[Debug] Diffusion Input Shape: {t_in.shape} | Steps: {self.steps.get()}"
                     print(msg)
                     self.update_queue.put(lambda m=msg: self._append_chat(None, m, "debug"))

                with torch.no_grad():
                    gen_tokens = lobe.model.generate(
                        prompt_tokens=t_in,
                        max_length=self.max_tokens.get() + t_in.shape[1],
                        steps=self.steps.get(),
                        temperature=self.temperature.get()
                    )

                # DEBUG RAW
                if self.verbose_log.get():
                     raw_summary = f"[Debug] Raw Generated Tokens (First 20): {gen_tokens[:20]}"
                     print(raw_summary)
                     self.update_queue.put(lambda m=raw_summary: self._append_chat(None, m, "debug"))

                full_text = ribosome.decode(gen_tokens)

                if self.verbose_log.get():
                     raw_txt = f"[Debug] Full Decoded Text: '{full_text}'"
                     print(raw_txt)
                     self.update_queue.put(lambda m=raw_txt: self._append_chat(None, m, "debug"))

                # Naive strip with lstrip() to handle spaces
                if full_text.lstrip().startswith(prompt_text.lstrip()):
                    # Find where prompt ends roughly
                    final_output = full_text[len(prompt_text):].strip()
                else:
                    final_output = full_text

                if not final_output.strip():
                    final_output = "[...] (Model produced no text)"

                self.update_queue.put(lambda: self.chat_box.insert(tk.END, final_output, "content"))
            else:
                # AR Mode
                t_in = t_text
                v_in = self.attached_dense if self.attached_dense is not None else torch.zeros(1, 1, 768).to(device)
                a_in = torch.zeros(1, 1, 128).to(device); c_in = torch.zeros(1, 1, 32).to(device)

                with torch.no_grad():
                    curr_t = t_in
                    for _ in range(self.max_tokens.get()):
                        if self.stop_requested: break
                        try: logits, _, _ = lobe.model(v_in, a_in, curr_t, c_in)
                        except: logits, _, _ = lobe.model(v_in, a_in, curr_t)
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        word = ribosome.decode([next_token.item()])
                        self.update_queue.put(lambda w=word: self.chat_box.insert(tk.END, w, "content"))
                        self.update_queue.put(lambda: self.chat_box.see(tk.END))
                        curr_t = torch.cat([curr_t, next_token], dim=1)

            self.update_queue.put(lambda: self.chat_box.insert(tk.END, "\n\n"))
            self._save_to_memory(prompt_text, "Response Generated")
            self._clear_attachment()

        except Exception as e:
            err_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.update_queue.put(lambda: self._append_chat("Runtime Error", err_msg, "error"))
            print(err_msg)

        self.is_generating = False
        self.update_queue.put(lambda: self.btn_send.config(state="normal"))
        self.update_queue.put(lambda: self.btn_stop.config(state="disabled", text="STOP"))

    def _append_chat(self, sender, text, tag):
        self.chat_box.config(state="normal")
        if sender: self.chat_box.insert(tk.END, f"{sender}: ", tag)
        self.chat_box.insert(tk.END, f"{text}\n", "content")
        self.chat_box.config(state="disabled")
        self.chat_box.see(tk.END)

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try: self.update_queue.get_nowait()()
            except: break
        if self.parent: self.parent.after(50, self._process_gui_queue)

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'chat_box'): self.chat_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])