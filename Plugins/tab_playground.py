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
from tkinter import ttk, font
import threading
import torch
import torch.nn.functional as F
from PIL import ImageTk, Image
import os
import traceback
import time

try:
    import pygame

    pygame.mixer.init()
    HAS_AUDIO = True
except:
    HAS_AUDIO = False


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Playground"
        self.is_generating = False
        self.image_refs = []  # Keep references to prevent GC

        # --- PARAMS ---
        self.temp = tk.DoubleVar(value=0.9)
        self.top_k = tk.IntVar(value=100)
        self.max_tokens = tk.IntVar(value=1024)
        self.steps = tk.IntVar(value=18)

        # --- UI STATE ---
        self.progress_var = tk.StringVar(value="")

        self._setup_ui()

    def _setup_ui(self):
        # Main split
        panel = ttk.PanedWindow(self.parent, orient="vertical")
        panel.pack(fill="both", expand=True, padx=5, pady=5)

        # 1. CHAT AREA
        chat_frame = ttk.Frame(panel)
        panel.add(chat_frame, weight=1)

        sb = ttk.Scrollbar(chat_frame)
        sb.pack(side="right", fill="y")

        self.chat_box = tk.Text(chat_frame, font=("Consolas", 11), bg=self.app.colors["BG_CARD"],
                                fg=self.app.colors["FG_TEXT"], wrap="word", padx=10, pady=10,
                                yscrollcommand=sb.set, borderwidth=0, highlightthickness=0)
        self.chat_box.pack(side="left", fill="both", expand=True)
        sb.config(command=self.chat_box.yview)

        # Tags
        self.chat_box.tag_config("user", foreground="#A8C7FA", justify="right", rmargin=10)
        self.chat_box.tag_config("brain", foreground="#E3E3E3", justify="left", lmargin1=10, lmargin2=10)
        self.chat_box.tag_config("system", foreground="#8e9198", font=("Segoe UI", 9, "italic"), justify="center")
        self.chat_box.tag_config("error", foreground="#F28B82", justify="center")
        self.chat_box.tag_config("media_placeholder", foreground="#FDD663", font=("Segoe UI", 9, "bold"),
                                 justify="center")
        self.chat_box.tag_config("media", justify="center")

        # 2. CONTROLS
        ctrl = ttk.Frame(panel, padding=5)
        panel.add(ctrl, weight=0)

        # Input
        input_frame = ttk.Frame(ctrl)
        input_frame.pack(fill="x", pady=(0, 5))

        self.input_var = tk.StringVar()
        self.entry = ttk.Entry(input_frame, textvariable=self.input_var, font=("Segoe UI", 10))
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry.bind("<Return>", lambda e: self._on_send())

        self.btn_send = ttk.Button(input_frame, text="SEND", command=self._on_send)
        self.btn_send.pack(side="left")

        # Settings
        sets = ttk.Frame(ctrl)
        sets.pack(fill="x")

        def add_slider(lbl, var, min_v, max_v):
            f = ttk.Frame(sets)
            f.pack(side="left", padx=5)
            ttk.Label(f, text=lbl, font=("Segoe UI", 8)).pack(side="left")
            ttk.Scale(f, from_=min_v, to=max_v, variable=var, orient="horizontal", length=80).pack(side="left", padx=5)

        add_slider("Temp:", self.temp, 0.1, 2.0)
        add_slider("Top-K:", self.top_k, 0, 200)
        add_slider("MaxLen:", self.max_tokens, 256, 4096)

        self.diff_frame = ttk.Frame(sets)
        add_slider("Steps:", self.steps, 4, 64)
        self.diff_frame.pack_forget()

        # Status Bar
        self.lbl_status = ttk.Label(ctrl, textvariable=self.progress_var, foreground=self.app.colors["ACCENT"],
                                    font=("Segoe UI", 9))
        self.lbl_status.pack(pady=(5, 0))

    def _detect_diffusion(self):
        brain = self.app.lobes.get(self.app.active_lobe.get())
        is_diff = (brain is not None and hasattr(brain, 'timestep_emb'))
        if is_diff:
            self.diff_frame.pack(side="left", padx=10)
        else:
            self.diff_frame.pack_forget()

    # --- OUTPUT HELPERS ---
    def _print(self, text, tag="brain"):
        self.chat_box.insert(tk.END, f"\n{text}\n", tag)
        self.chat_box.see(tk.END)

    def _print_image(self, pil_img):
        try:
            max_w, max_h = 512, 512
            pil_img.thumbnail((max_w, max_h))
            photo = ImageTk.PhotoImage(pil_img)
            self.image_refs.append(photo)

            self.chat_box.insert(tk.END, "\n", "media")
            self.chat_box.image_create(tk.END, image=photo, padx=10, pady=10)
            self.chat_box.insert(tk.END, "\n", "media")
            self.chat_box.see(tk.END)
        except Exception as e:
            self._print(f"[Image Render Error: {e}]", "error")

    # --- GENERATION ---
    def _on_send(self):
        if self.is_generating: return
        prompt = self.input_var.get().strip()
        if not prompt: return

        self.input_var.set("")
        self._print(f"> {prompt}", "user")

        self.is_generating = True
        self.btn_send.config(state="disabled")
        self._detect_diffusion()
        self.progress_var.set("Waiting for GPU...")

        threading.Thread(target=self._run_generation, args=(prompt,), daemon=True).start()

    def _run_generation(self, prompt):
        try:
            lobe_id = self.app.active_lobe.get()
            brain = self.app.lobes[lobe_id]
            ribosome = self.app.ribosome

            if not brain: raise Exception("No Brain Loaded.")

            prompt_ids = ribosome._tokenize(prompt)

            brain.eval()
            tokens = []

            with self.app.gpu_lock:
                with torch.no_grad():
                    if hasattr(brain, 'timestep_emb'):
                        steps = self.steps.get()
                        for i in range(1, steps + 1):
                            tokens = brain.generate(
                                prompt_tokens=prompt_ids,
                                max_length=self.max_tokens.get(),
                                steps=i,
                                temperature=self.temp.get(),
                                top_k=self.top_k.get()
                            )
                            if i % 2 == 0 or i == steps:
                                preview = ribosome.decode(tokens[:50]) + "..."
                                self.parent.after(0, lambda s=i, p=preview: self.progress_var.set(
                                    f"Diffusing {s}/{steps}: {p}"))
                    else:
                        t = torch.tensor(prompt_ids, device=self.app.device).unsqueeze(0)
                        v = torch.zeros(1, 1, 768).to(self.app.device)
                        a = torch.zeros(1, 1, 128).to(self.app.device)

                        for i in range(200):
                            self.parent.after(0, lambda c=i: self.progress_var.set(f"Generating token {c}..."))
                            logits, _, _ = brain(v, a, t)
                            next_tok = torch.multinomial(F.softmax(logits[:, -1, :] / self.temp.get(), dim=-1), 1)
                            t = torch.cat([t, next_tok], dim=1)
                            if next_tok.item() == 50256: break
                        tokens = t[0].tolist()

            self.parent.after(0, lambda: self.progress_var.set("Decoding content..."))

            # Text
            text_out = ribosome.decode(tokens)
            if text_out.startswith(prompt): text_out = text_out[len(prompt):].strip()
            if text_out.strip():
                self.parent.after(0, lambda t=text_out: self._print(t, "brain"))

            # Multimedia
            self._scan_and_render_media(tokens, ribosome)

        except Exception as e:
            traceback.print_exc()
            self.parent.after(0, lambda m=str(e): self._print(f"Error: {m}", "error"))
        finally:
            self.is_generating = False
            self.parent.after(0, lambda: self.btn_send.config(state="normal"))
            self.parent.after(0, lambda: self.progress_var.set("Ready."))

    def _scan_and_render_media(self, tokens, ribosome):
        """Scans tokens for media blocks and attempts to render them."""
        img_buffer = []
        aud_buffer = []

        side = ribosome.config.image_size // ribosome.config.patch_size
        img_tokens_needed = side ** 2

        for tok in tokens:
            val = tok.item() if torch.is_tensor(tok) else tok

            # --- VISUAL TOKENS ---
            if val >= ribosome.image_vocab_base and val < ribosome.audio_vocab_base:
                img_buffer.append(val)
                # Feedback for accumulating buffer
                if len(img_buffer) % 50 == 0:
                    self.parent.after(0, lambda: self.progress_var.set("Receiving Visual Data..."))

                # Full Image Check
                if len(img_buffer) == img_tokens_needed:
                    try:
                        img = ribosome.decode_image_tokens(img_buffer)
                        if img:
                            self.parent.after(0, lambda i=img: self._print_image(i))
                        else:
                            raise Exception("Empty image")
                    except Exception as e:
                        # RENDER FAILED: Show placeholder
                        self.parent.after(0, lambda: self._print(f"[Image Generation Failed: {len(img_buffer)} tokens]",
                                                                 "media_placeholder"))
                    img_buffer = []

            # --- AUDIO TOKENS ---
            elif val >= ribosome.audio_vocab_base:
                aud_buffer.append(val)
                if len(aud_buffer) % 100 == 0:
                    self.parent.after(0, lambda: self.progress_var.set("Receiving Audio Data..."))

                if len(aud_buffer) >= 1000:
                    self._flush_audio(aud_buffer, ribosome)
                    aud_buffer = []

        # Flush leftovers
        if aud_buffer:
            self._flush_audio(aud_buffer, ribosome)

        # If image buffer has stragglers (incomplete image)
        if len(img_buffer) > 0:
            self.parent.after(0, lambda: self._print(
                f"[Incomplete Image Data: {len(img_buffer)}/{img_tokens_needed} tokens]", "media_placeholder"))

    def _flush_audio(self, buffer, ribosome):
        try:
            wav = ribosome.decode_audio_tokens(buffer)
            if wav and HAS_AUDIO:
                self.parent.after(0, lambda: self._print("[Audio Clip Playing...]", "system"))
                threading.Thread(target=self._play_audio, args=(wav,), daemon=True).start()
            else:
                raise Exception("Audio decode failed")
        except:
            self.parent.after(0, lambda: self._print("[Audio Generation Failed]", "media_placeholder"))

    def _play_audio(self, wav_path):
        try:
            while pygame.mixer.music.get_busy(): time.sleep(0.1)
            pygame.mixer.music.load(wav_path)
            pygame.mixer.music.play()
        except:
            pass

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'chat_box'):
            self.chat_box.config(bg=c["BG_CARD"], fg=c["FG_TEXT"])