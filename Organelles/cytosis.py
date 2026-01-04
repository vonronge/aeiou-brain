"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Cytosis:
Active transport of imagination.
Runs the Dream State loop, generating content from neural noise
and excreting it into the Chaos Buffer.
"""

import threading
import time
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List


@dataclass
class DreamConfig:
    temperature: float = 0.9
    top_k: int = 50
    max_length: int = 1024
    diffusion_steps: int = 20
    refresh_rate: float = 0.5  # Seconds between dreams
    autosave: bool = True


class Organelle_Cytosis:
    def __init__(self, device: str, ribosome, phagus, golgi):
        self.device = device
        self.ribosome = ribosome
        self.phagus = phagus
        self.golgi = golgi

        self.is_dreaming = False
        self.stop_requested = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "sample": [],  # fn(text, is_image_path)
            "autosave": [],  # fn(filename)
            "finished": [],  # fn()
            "error": []  # fn(exception)
        }

    def register_callback(self, event: str, fn: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(fn)

    def _trigger(self, event: str, *args):
        for fn in self._callbacks.get(event, []):
            try:
                fn(*args)
            except:
                pass

    def start_dream(self, lobe, prompt: str, config: DreamConfig):
        if self.is_dreaming: return
        self.is_dreaming = True
        self.stop_requested = False

        self.golgi.info(f"Entering Dream State. Prompt: '{prompt}'", source="Cytosis")

        self._thread = threading.Thread(
            target=self._worker,
            args=(lobe, prompt, config),
            daemon=True
        )
        self._thread.start()

    def stop(self):
        if self.is_dreaming:
            self.stop_requested = True
            self.golgi.warn("Waking up...", source="Cytosis")

    def _worker(self, lobe, prompt, config):
        try:
            model = lobe.model
            model.eval()

            # Detect Architecture
            is_diffusion = (lobe.model_type == "diffusion")

            while not self.stop_requested:
                # 1. Tokenize Prompt
                prompt_ids = self.ribosome._tokenize(prompt)

                generated_content = ""

                # 2. Generate
                with torch.no_grad():
                    if is_diffusion:
                        tokens = model.generate(
                            prompt_tokens=prompt_ids,
                            max_length=config.max_length,
                            steps=config.diffusion_steps,
                            temperature=config.temperature
                        )
                        # Diffusion usually returns full sequence including prompt
                        generated_content = self.ribosome.decode(tokens)
                    else:
                        # Autoregressive Loop
                        t = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
                        # Null sensors
                        v = torch.zeros(1, 1, 768).to(self.device)
                        a = torch.zeros(1, 1, 128).to(self.device)

                        # Simple AR generation
                        for _ in range(config.max_length):
                            if self.stop_requested: break

                            try:
                                logits, _, _ = model(v, a, t)
                            except:
                                # Fallback if model signature differs
                                logits, _, _ = model(v, a, t, None)

                            next_logits = logits[:, -1, :] / config.temperature
                            probs = F.softmax(next_logits, dim=-1)

                            # Top-K
                            if config.top_k > 0:
                                val, _ = torch.topk(probs, config.top_k)
                                probs[probs < val[:, [-1]]] = 0
                                probs = probs / probs.sum(dim=-1, keepdim=True)

                            next_tok = torch.multinomial(probs, 1)
                            t = torch.cat([t, next_tok], dim=1)

                            if next_tok.item() == 50256: break  # EOS

                        tokens = t[0].tolist()
                        generated_content = self.ribosome.decode(tokens)

                # 3. Process Output (Clean prompt echo)
                if generated_content.startswith(prompt):
                    clean_text = generated_content[len(prompt):].strip()
                else:
                    clean_text = generated_content.strip()

                if clean_text:
                    self._trigger("sample", clean_text)

                    # 4. Autosave to Chaos Buffer
                    if config.autosave:
                        self._save_to_chaos(prompt, clean_text)

                time.sleep(config.refresh_rate)

        except Exception as e:
            self._trigger("error", e)
            self.golgi.error(f"Nightmare (Crash): {e}", source="Cytosis")
        finally:
            self.is_dreaming = False
            self._trigger("finished")

    def _save_to_chaos(self, prompt, content):
        chaos_dir = self.phagus.state.chaos_dir
        if not os.path.exists(chaos_dir):
            try:
                os.makedirs(chaos_dir)
            except:
                return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"dream_{ts}.txt"
        path = os.path.join(chaos_dir, fname)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"PROMPT: {prompt}\n\n")
                f.write(content)
            self._trigger("autosave", fname)
        except Exception as e:
            self.golgi.error(f"Failed to save dream: {e}", source="Cytosis")