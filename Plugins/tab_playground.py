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
import json
import uuid
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

        # Conversation History
        self.history = []
        self.attached_image_path = None
        self.attached_image_tensor = None

        # Memory Config
        self.memory_file = os.path.join(self.app.paths["memories"], "episodic_chat_log.jsonl")

        # --- SETTINGS ---
        self.temperature = tk.DoubleVar(value=0.8)
        self.top_k = tk.IntVar(value=50)
        self.max_tokens = tk.IntVar(value=100)
        self.do_sample = tk.BooleanVar(value=True)

        # Memory Settings
        self.use_recall = tk.BooleanVar(value=True)
        self.recall_threshold = tk.DoubleVar(value=0.65)

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        scale = getattr(self.app, 'ui_scale', 1.0)

        # Layout
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

        self.txt_input = tk.Text(fr_input, height=3, font=chat_font,
                                 bg=self.app.colors["BG_CARD"], fg=self.app.colors["FG_TEXT"],
                                 insertbackground=self.app.colors["ACCENT"])
        self.txt_input.pack(fill="x", side="top", pady=(0, 5))
        self.txt_input.bind("<Shift-Return>", lambda e: "break")
        self.txt_input.bind("<Return>", self._on_enter)

        # Controls
        fr_ctrl = ttk.Frame(fr_input)
        fr_ctrl.pack(fill="x")

        self.btn_img = ttk.Button(fr_ctrl, text="📷 Attach", command=self._attach_image)
        self.btn_img.pack(side="left")
        self.lbl_img = ttk.Label(fr_ctrl, text="", foreground=self.app.colors["FG_DIM"])
        self.lbl_img.pack(side="left", padx=5)

        ttk.Label(fr_ctrl, text="Temp:").pack(side="left", padx=(15, 0))
        ttk.Entry(fr_ctrl, textvariable=self.temperature, width=5).pack(side="left")

        ttk.Checkbutton(fr_ctrl, text="Visual Recall", variable=self.use_recall).pack(side="left", padx=(15, 5))

        ttk.Label(fr_ctrl, text="Strength:").pack(side="left", padx=(0, 5))
        self.lbl_thresh = ttk.Label(fr_ctrl, text=f"{self.recall_threshold.get():.2f}", width=4)
        self.lbl_thresh.pack(side="left")

        scale_rec = ttk.Scale(fr_ctrl, from_=0.3, to=0.99, variable=self.recall_threshold,
                              orient="horizontal", length=80,
                              command=lambda v: self.lbl_thresh.config(text=f"{float(v):.2f}"))
        scale_rec.pack(side="left", padx=5)

        self.btn_send = ttk.Button(fr_ctrl, text="SEND ➤", command=self._send_message)
        self.btn_send.pack(side="right")
        self.btn_stop = ttk.Button(fr_ctrl, text="STOP", command=self._stop_gen, state="disabled")
        self.btn_stop.pack(side="right", padx=5)

    def _save_to_memory(self, user_text, ai_text):
        full_conversation = f"User: {user_text}\nLobe: {ai_text}"
        try:
            if hasattr(self.app.ribosome, "render_text_to_image"):
                img = self.app.ribosome.render_text_to_image(full_conversation)
            else:
                from PIL import ImageDraw
                img = Image.new('RGB', (512, 512), color=(20, 20, 20))
                d = ImageDraw.Draw(img)
                d.text((10, 10), full_conversation[:500], fill=(200, 200, 200))

            if not os.path.exists(self.app.paths["memories"]):
                os.makedirs(self.app.paths["memories"])

            img_filename = f"memory_{uuid.uuid4()}.png"
            img_path = os.path.join(self.app.paths["memories"], img_filename)
            img.save(img_path)

            entry = {
                "timestamp": str(datetime.now()),
                "user_text": user_text,
                "ai_text": ai_text,
                "full_text": full_conversation,
                "image_path": img_path
            }

            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            if self.app.hippocampus:
                v_feat, _, _, _, _ = self.app.ribosome.ingest_packet({'v': img_path})
                if v_feat is not None:
                    if v_feat.ndim == 3: v_feat = v_feat.mean(dim=1)
                    self.app.hippocampus.add_vector(full_conversation[:50], v_feat)

        except Exception as e:
            print(f"Memory Save Error: {e}")

    def _attach_image(self):
        f = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp")])
        if f:
            self.attached_image_path = f
            self.lbl_img.config(text=f"[{os.path.basename(f)}]")
            try:
                packet = {'v': f, 'type': 'single'}
                v, _, _, _, _ = self.app.ribosome.ingest_packet(packet)
                self.attached_image_tensor = v
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

        threading.Thread(target=self._worker, args=(lobe, text), daemon=True).start()

    def _stop_gen(self):
        self.stop_requested = True
        self.btn_stop.config(text="STOPPING...")

    def _worker(self, lobe, prompt_text):
        try:
            device = self.app.device
            ribosome = self.app.ribosome

            # --- 1. VISUAL EPISODIC RECALL ---
            recalled_context_text = ""
            if self.use_recall.get() and self.app.hippocampus:
                try:
                    if hasattr(ribosome, "render_text_to_image"):
                        prompt_img = ribosome.render_text_to_image(prompt_text)
                    else:
                        from PIL import Image
                        prompt_img = Image.new('RGB', (256, 256))

                    temp_path = os.path.join(self.app.paths["memories"], f"temp_prompt_{uuid.uuid4()}.png")
                    prompt_img.save(temp_path)
                    v_prompt = ribosome.get_engram(temp_path)
                    os.remove(temp_path)

                    thresh = self.recall_threshold.get()
                    hits = self.app.hippocampus.search(v_prompt, top_k=2, threshold=thresh)

                    if hits:
                        self.update_queue.put(
                            lambda: self._append_chat("System", f"Recalling {len(hits)} visual memories...", "sys"))
                        for score, entity in hits:
                            text_content = entity if isinstance(entity, str) else entity.entity
                            short_recall = text_content[:200].replace("\n", " ")
                            recalled_context_text += f"[Memory (Sim:{score:.2f})]: {short_recall}...\n"
                except Exception as e:
                    print(f"Recall Error: {e}")

            # --- 2. CONSTRUCT INPUT ---
            full_input_text = ""
            if recalled_context_text:
                full_input_text += f"{recalled_context_text}\n--- CURRENT ---\n"
            full_input_text += prompt_text

            input_ids = ribosome._tokenize(full_input_text)

            # --- 3. GENERATION BRANCH ---
            # Detect architecture type
            is_diffusion = hasattr(lobe.model, "generate") and (
                        lobe.model_type == "diffusion" or "Diffusion" in lobe.genome)

            self.update_queue.put(lambda: self._append_chat("AI", "", "ai"))
            final_output = ""

            if is_diffusion:
                # --- DIFFUSION MODE (Block Generation) ---
                self.update_queue.put(lambda: self._append_chat(None, "(Thinking/Diffusing...)", "sys"))

                # Diffusion generate() returns tokens list
                # We need to pass tensor to it if prompt is tokenized
                t_in = torch.tensor([input_ids], device=device).long()

                with torch.no_grad():
                    gen_tokens = lobe.model.generate(
                        prompt_tokens=t_in,
                        max_length=self.max_tokens.get() + len(input_ids),  # extend length
                        steps=30,  # fixed steps for speed
                        temperature=self.temperature.get()
                    )

                # Diffusion returns full sequence usually (Prompt + New)
                # Decode all
                full_text = ribosome.decode(gen_tokens)

                # Strip original prompt to find new content
                # This is heuristic; diffusion models might rewrite the prompt
                # Simple check:
                if full_text.startswith(full_input_text):
                    final_output = full_text[len(full_input_text):].strip()
                else:
                    final_output = full_text  # Return everything if it diverged

                self.update_queue.put(lambda: self.chat_box.insert(tk.END, final_output, "content"))

            else:
                # --- AUTOREGRESSIVE MODE (Streaming) ---
                t_in = torch.tensor([input_ids], device=device).long()
                v_in = self.attached_image_tensor if self.attached_image_tensor is not None else torch.zeros(1, 1,
                                                                                                             768).to(
                    device)
                a_in = torch.zeros(1, 1, 128).to(device)
                c_in = torch.zeros(1, 1, 32).to(device)

                max_new = self.max_tokens.get()
                temp = self.temperature.get()
                top_k = self.top_k.get()
                generated_ids = []

                with torch.no_grad():
                    curr_t = t_in
                    for _ in range(max_new):
                        if self.stop_requested: break
                        try:
                            logits, _, _ = lobe.model(v_in, a_in, curr_t, c_in)
                        except:
                            logits, _, _ = lobe.model(v_in, a_in, curr_t)

                        next_logits = logits[:, -1, :]
                        if self.do_sample.get():
                            v, i = torch.topk(next_logits, top_k)
                            next_logits[next_logits < v[:, [-1]]] = -float('Inf')
                            probs = F.softmax(next_logits / temp, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                        else:
                            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

                        token_id = next_token.item()
                        word = ribosome.decode([token_id])
                        self.update_queue.put(lambda w=word: self.chat_box.insert(tk.END, w, "content"))
                        self.update_queue.put(lambda: self.chat_box.see(tk.END))

                        generated_ids.append(token_id)
                        curr_t = torch.cat([curr_t, next_token], dim=1)
                        if token_id == 50256: break

                final_output = ribosome.decode(generated_ids)

            self.update_queue.put(lambda: self.chat_box.insert(tk.END, "\n\n"))
            self._save_to_memory(prompt_text, final_output)
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