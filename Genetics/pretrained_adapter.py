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

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import tkinter as tk
from tkinter import ttk

# --- GUI METADATA ---
INFO = {
    "name": "HuggingFace Adapter",
    "desc": "Universal Loader. Select model variant upon initialization.",
    "vram_train": "Varies",
    "vram_run": "Varies"
}


class NucleusConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.embed_dim = 4096
        self.context_len = 8192


class Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        # --- INTERACTIVE MODEL SELECTOR ---
        self.model_id = self._ask_user_for_model()
        if not self.model_id:
            raise ValueError("No model selected.")

        print(f"[Adapter] Loading {self.model_id} in 4-bit...")

        # 4-Bit Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load Real Model
        # device_map="auto" AUTOMATICALLY puts this on the GPU
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.tok_emb = nn.Identity()

    def _ask_user_for_model(self):
        """ Pops up a dialog to select the HF model """
        selector = tk.Toplevel()
        selector.title("Select Pretrained Model")
        selector.geometry("400x350")
        selector.transient()
        selector.grab_set()

        selected_model = tk.StringVar(value="Qwen/Qwen2.5-7B-Instruct")

        ttk.Label(selector, text="Choose a Hugging Face Model:", font=("Segoe UI", 10, "bold")).pack(pady=10)

        options = [
            ("Qwen 2.5 (14B) - Best for 3080 Ti", "Qwen/Qwen2.5-14B-Instruct"),
            ("Qwen 2.5 (7B) - Fast & Light", "Qwen/Qwen2.5-7B-Instruct"),
            ("Qwen 2.5 (32B) - Requires 24GB+ VRAM", "Qwen/Qwen2.5-32B-Instruct"),
            ("Llama 3.1 (8B) - Solid Generalist", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            ("Mistral Nemo (12B) - High Context", "mistralai/Mistral-Nemo-Instruct-2407"),
            ("Phi-3.5 Mini (3.8B) - Very Fast", "microsoft/Phi-3.5-mini-instruct")
        ]

        for label, mid in options:
            ttk.Radiobutton(selector, text=label, variable=selected_model, value=mid).pack(anchor="w", padx=20, pady=2)

        ttk.Separator(selector, orient="horizontal").pack(fill="x", padx=10, pady=10)
        ttk.Label(selector, text="Or type custom HF ID:").pack()
        ttk.Entry(selector, textvariable=selected_model).pack(fill="x", padx=20)

        def confirm():
            selector.destroy()

        ttk.Button(selector, text="LOAD MODEL", command=confirm).pack(pady=15)

        selector.wait_window()
        return selected_model.get()

    def forward(self, v, a, t, c=None):
        outputs = self.hf_model(input_ids=t, output_hidden_states=True)
        return outputs.logits, None, None

    def generate(self, prompt, max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
        outputs = self.hf_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- CRITICAL FIX FOR 4-BIT MODELS ---
    def to(self, device):
        """
        Override .to() to prevent bitsandbytes error.
        The HF model is already on GPU via device_map='auto', so we ignore this call.
        """
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {"model_id": self.model_id}

    def load_state_dict(self, state_dict, strict=True):
        pass