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
import torch
import torch.nn.functional as F
import threading
import queue
import os
from PIL import Image, ImageTk


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Playground"

        self.is_generating = False
        self.stop_requested = False
        self.update_queue = queue.Queue()

        # Conversation History
        self.history = []
        self.attached_image_path = None
        self.attached_image_tensor = None

        # --- SETTINGS ---
        self.temperature = tk.DoubleVar(value=0.8)
        self.top_k = tk.IntVar(value=50)
        self.max_tokens = tk.IntVar(value=100)
        self.do_sample = tk.BooleanVar(value=True)

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        scale = getattr(self.app, 'ui_scale', 1.0)

        # Layout: Chat History (Top), Controls (Bottom)
        fr_main = ttk.Frame(self.parent)
        fr_main.pack(fill="both", expand=True, padx=10, pady=10)

        # 1. CHAT HISTORY
        fr_hist = ttk.Frame(fr_main)
        fr_hist.pack(fill="both", expand=True)

        chat_font = ("Segoe UI", int(11 * scale))
        self.chat_box = tk.Text(fr_hist, font=chat_font, wrap="word",
                                bg=self.app.colors["BG_MAIN"], fg=self.app.colors["FG_TEXT"],
                                state="disabled", padx=10, pady=10)
        self.chat_box.pack(side="left", fill="both", expand=True)

        # Tags for coloring
        self.chat_box.tag_config("user", foreground=self.app.colors["ACCENT"],
                                 font=("Segoe UI", int(11 * scale), "bold"))
        self.chat_box.tag_config("ai", foreground=self.app.colors["SUCCESS"],
                                 font=("Segoe UI", int(11 * scale), "bold"))
        self.chat_box.tag_config("sys", foreground=self.app.colors["FG_DIM"],
                                 font=("Segoe UI", int(10 * scale), "italic"))
        self.chat_box.tag_config("content", foreground=self.app.colors["FG_TEXT"])

        sb = ttk.Scrollbar(fr_hist, command=self.chat_box.yview)
        sb.pack(side="right", fill="y")
        self.chat_box.config(yscrollcommand=sb.set)

        # 2. INPUT AREA
        fr_input = ttk.LabelFrame(fr_main, text="Input Signal", padding=10)
        fr_input.pack(fill="x", pady=(10, 0))

        # Text Input
        self.txt_input = tk.Text(fr_input, height=3, font=chat_font,
                                 bg=self.app.colors["BG_CARD"], fg=self.app.colors["FG_TEXT"],
                                 insertbackground=self.app.colors["ACCENT"])
        self.txt_input.pack(fill="x", side="top", pady=(0, 5))
        self.txt_input.bind("<Shift-Return>", lambda e: "break")  # Newline
        self.txt_input.bind("<Return>", self._on_enter)

        # Controls Row
        fr_ctrl = ttk.Frame(fr_input)
        fr_ctrl.pack(fill="x")

        # Attachments
        self.btn_img = ttk.Button(fr_ctrl, text="📷 Attach Image", command=self._attach_image)
        self.btn_img.pack(side="left")

        self.lbl_img = ttk.Label(fr_ctrl, text="", foreground=self.app.colors["FG_DIM"])
        self.lbl_img.pack(side="left", padx=5)

        # Params
        ttk.Label(fr_ctrl, text="Temp:").pack(side="left", padx=(15, 0))
        ttk.Entry(fr_ctrl, textvariable=self.temperature, width=5).pack(side="left")

        ttk.Label(fr_ctrl, text="Len:").pack(side="left", padx=(5, 0))
        ttk.Entry(fr_ctrl, textvariable=self.max_tokens, width=5).pack(side="left")

        # Buttons
        self.btn_send = ttk.Button(fr_ctrl, text="SEND ➤", command=self._send_message)
        self.btn_send.pack(side="right")

        self.btn_stop = ttk.Button(fr_ctrl, text="STOP", command=self._stop_gen, state="disabled")
        self.btn_stop.pack(side="right", padx=5)

    # --- ACTIONS ---
    def _attach_image(self):
        f = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp")])
        if f:
            self.attached_image_path = f
            self.lbl_img.config(text=f"[{os.path.basename(f)}]")
            # Pre-process immediately via Ribosome
            try:
                # We create a dummy packet to use Membrane/Ribosome logic
                packet = {'v': f, 'type': 'single'}
                # Returns (v_feat, a_feat, tokens, c_emb, meta)
                v, _, _, _, _ = self.app.ribosome.ingest_packet(packet)
                self.attached_image_tensor = v  # Keep the dense features
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {e}")
                self._clear_attachment()

    def _clear_attachment(self):
        self.attached_image_path = None
        self.attached_image_tensor = None
        self.lbl_img.config(text="")

    def _on_enter(self, event):
        if not self.is_generating:
            self._send_message()
        return "break"

    def _send_message(self):
        text = self.txt_input.get("1.0", tk.END).strip()
        if not text and not self.attached_image_path: return

        # 1. Verify Lobe
        active_id = self.app.active_lobe.get()
        lobe = self.app.lobe_manager.get_lobe(active_id)
        if not lobe:
            self._append_chat("System", "No Lobe Loaded. Please activate a model.", "sys")
            return

        # 2. Update UI
        self.txt_input.delete("1.0", tk.END)
        self._append_chat("User", text, "user")
        if self.attached_image_path:
            self._append_chat("System", f"[Image Attached: {os.path.basename(self.attached_image_path)}]", "sys")

        # 3. Start Inference
        self.is_generating = True
        self.stop_requested = False
        self.btn_send.config(state="disabled")
        self.btn_stop.config(state="normal")

        threading.Thread(target=self._worker, args=(lobe, text), daemon=True).start()

    def _stop_gen(self):
        self.stop_requested = True
        self.btn_stop.config(text="STOPPING...")

    # --- INFERENCE WORKER ---
    def _worker(self, lobe, prompt_text):
        try:
            device = self.app.device

            # 1. Tokenize Text (Ribosome)
            # The ribosome gives us a list of ints. We need a tensor.
            input_ids = self.app.ribosome._tokenize(prompt_text)
            t_in = torch.tensor([input_ids], device=device).long()

            # 2. Prepare Visuals
            v_in = self.attached_image_tensor
            if v_in is None:
                # Empty visual tensor (Batch, 1, 768)
                v_in = torch.zeros(1, 1, 768).to(device)

            # 3. Prepare Audio (Empty for now)
            a_in = torch.zeros(1, 1, 128).to(device)

            # 4. Control Vector (Neutral)
            c_in = torch.zeros(1, 1, 32).to(device)

            # 5. Generation Loop
            max_new = self.max_tokens.get()
            temp = self.temperature.get()
            top_k = self.top_k.get()

            generated = []

            self.update_queue.put(lambda: self._append_chat("AI", "", "ai"))  # Start AI line

            with torch.no_grad():
                # Autoregressive Loop
                curr_t = t_in

                for _ in range(max_new):
                    if self.stop_requested: break

                    # Forward
                    # model(v, a, t, c)
                    try:
                        logits, _, _ = lobe.model(v_in, a_in, curr_t, c_in)
                    except RuntimeError:
                        # Fallback for models without Control Vector
                        logits, _, _ = lobe.model(v_in, a_in, curr_t)

                    # Get last token logits
                    next_token_logits = logits[:, -1, :]

                    # Sample
                    if self.do_sample.get():
                        # Top-K
                        v, i = torch.topk(next_token_logits, top_k)
                        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                        probs = F.softmax(next_token_logits / temp, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    # Decode Token
                    token_id = next_token.item()
                    word = self.app.ribosome.decode([token_id])

                    # Stream to UI
                    self.update_queue.put(lambda w=word: self.chat_box.insert(tk.END, w, "content"))
                    self.update_queue.put(lambda: self.chat_box.see(tk.END))

                    # Append
                    generated.append(token_id)
                    curr_t = torch.cat([curr_t, next_token], dim=1)

                    # Stop Token Check (50256 is GPT-2 EOS)
                    if token_id == 50256: break

            self.update_queue.put(lambda: self.chat_box.insert(tk.END, "\n\n"))
            self._clear_attachment()

        except Exception as e:
            self.update_queue.put(lambda: self._append_chat("Error", str(e), "sys"))
            print(e)

        self.is_generating = False
        self.update_queue.put(lambda: self.btn_send.config(state="normal"))
        self.update_queue.put(lambda: self.btn_stop.config(state="disabled", text="STOP"))

    def _append_chat(self, sender, text, tag):
        self.chat_box.config(state="normal")
        if sender:
            self.chat_box.insert(tk.END, f"{sender}: ", tag)
        self.chat_box.insert(tk.END, f"{text}\n", "content")
        self.chat_box.config(state="disabled")
        self.chat_box.see(tk.END)

    def _process_gui_queue(self):
        while not self.update_queue.empty():
            try:
                fn = self.update_queue.get_nowait()
                fn()
            except:
                break
        if self.parent:
            self.parent.after(50, self._process_gui_queue)

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'chat_box'):
            self.chat_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])
            self.chat_box.tag_config("content", foreground=c["FG_TEXT"])
        if hasattr(self, 'txt_input'):
            self.txt_input.config(bg=c["BG_CARD"], fg=c["FG_TEXT"], insertbackground=c["ACCENT"])