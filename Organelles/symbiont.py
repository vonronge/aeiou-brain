"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Symbiont:
Manages the symbiotic relationship between two lobes (Teacher -> Student).
Handles distillation, sensory isolation (freezing), and knowledge harvesting.
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import threading
import time
import os
import random
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List
from datetime import datetime

# Prompts for the Teacher to generate curriculum
DEFAULT_PROMPTS = [
    "Explain the concept of", "The history of", "Why is", "How does",
    "Describe the function of", "A summary of", "Write a story about",
    "Define the term", "Compare and contrast", "What happens if",
    "The connection between", "A detailed analysis of"
]


@dataclass
class SymbiosisConfig:
    """Settings for the distillation session."""
    save_interval: int = 50  # Cycles between autosaves
    harvest_enabled: bool = True  # Write synthetic data to disk
    max_file_size: int = 10 * 1024 * 1024  # 10 MB harvest chunks
    safety_freeze: bool = True  # Freeze visual/audio embeddings


class Organelle_Symbiont:
    def __init__(self, device: str, ribosome, golgi, memories_path: str):
        self.device = device
        self.ribosome = ribosome
        self.golgi = golgi

        self.harvest_dir = os.path.join(memories_path, "harvested")
        if not os.path.exists(self.harvest_dir):
            os.makedirs(self.harvest_dir, exist_ok=True)

        self.is_running = False
        self.stop_requested = False
        self._thread: Optional[threading.Thread] = None

        # State
        self.teacher_handle = None
        self.student_handle = None
        self.config = SymbiosisConfig()

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "cycle": [],  # fn(cycle_count, loss, generated_text)
            "status": [],  # fn(msg, tag) - for UI updates
            "finished": []  # fn()
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

    def _log(self, msg, tag="INFO"):
        # Log to Golgi (System) AND trigger local callback (UI specific)
        if self.golgi:
            # Map simple tags to Golgi methods
            if tag == "INFO":
                self.golgi.info(msg, source="Symbiont")
            elif tag == "WARN":
                self.golgi.warn(msg, source="Symbiont")
            elif tag == "ERROR":
                self.golgi.error(msg, source="Symbiont")
            elif tag == "SAVE":
                self.golgi.save(msg, source="Symbiont")
            elif tag == "HARVEST":
                self.golgi.harvest(msg, source="Symbiont")

        self._trigger("status", msg, tag)

    def link(self, teacher, student, config: SymbiosisConfig):
        """Prepares the connection but does not start loop."""
        if not teacher or not student:
            raise ValueError("Both Teacher and Student lobes must be loaded.")

        self.teacher_handle = teacher
        self.student_handle = student
        self.config = config
        self._log(f"Link Established: Lobe {teacher.id} (T) -> Lobe {student.id} (S)", "INFO")

    def start(self):
        if self.is_running: return
        self.is_running = True
        self.stop_requested = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        if self.is_running:
            self._log("Severing link...", "WARN")
            self.stop_requested = True

    def _worker(self):
        teacher = self.teacher_handle.model
        student = self.student_handle.model
        opt = self.student_handle.optimizer
        scaler = self.student_handle.scaler

        # 1. SAFETY: Freeze Senses (Lobotomy)
        frozen_layers = []
        if self.config.safety_freeze:
            try:
                if hasattr(student, 'vis_emb'):
                    for p in student.vis_emb.parameters(): p.requires_grad = False
                    frozen_layers.append("Vision")
                if hasattr(student, 'aud_emb'):
                    for p in student.aud_emb.parameters(): p.requires_grad = False
                    frozen_layers.append("Audio")
            except:
                pass

        if frozen_layers:
            self._log(f"Safety Protocol: Frozen {', '.join(frozen_layers)}", "WARN")

        student.train()
        teacher.eval()

        cycles = 0

        # Harvest Setup
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        harvest_part = 1
        harvest_path = os.path.join(self.harvest_dir, f"symbiosis_{ts}_p{harvest_part}.txt")

        # Get vocab limit to prevent OOB errors
        try:
            vocab_limit = student.tok_emb.weight.shape[0]
        except:
            vocab_limit = 50257

        try:
            while not self.stop_requested:
                # A. TEACHER GENERATES
                prompt = random.choice(DEFAULT_PROMPTS)

                # Check generation capability
                gen_text = ""
                if hasattr(teacher, "generate"):
                    # We assume teacher.generate returns tokens or text.
                    # If it returns tokens, we decode. If text, we use it.
                    # Adapt based on genetics implementation.
                    # Standard AEIOU genetics return tokens list usually.
                    try:
                        # Quick generation
                        # We use the ribosome to tokenize the prompt first if needed
                        # But `generate` usually handles prompt strings or token lists.
                        # Let's assume high-level generate accepts string or we tokenize.
                        p_ids = self.ribosome._tokenize(prompt)

                        with torch.no_grad():
                            # Generate tokens
                            # Note: using raw generate method from model
                            tokens = teacher.generate(prompt_tokens=p_ids, max_length=64, temperature=0.8)
                            gen_text = self.ribosome.decode(tokens)
                    except Exception as e:
                        # Fallback for models without generate or errors
                        gen_text = f"The nature of {prompt} is complex."
                else:
                    gen_text = "Neural activity detected."

                if not gen_text: gen_text = "Silence."

                # Clean prompt echo
                if gen_text.startswith(prompt):
                    clean_content = gen_text[len(prompt):].strip()
                else:
                    clean_content = gen_text

                if len(clean_content) < 5: continue

                # B. STUDENT LEARNS
                # Retokenize for Student (in case vocab differs slightly, though Ribosome unifies)
                s_ids = self.ribosome._tokenize(gen_text)

                # Clamp for safety
                s_ids = [t if t < vocab_limit else 0 for t in s_ids]

                if len(s_ids) < 2: continue

                t_tensor = torch.tensor([s_ids]).to(self.device)

                inp = t_tensor[:, :-1]
                tgt = t_tensor[:, 1:]

                # Null sensors for pure text learning
                v = torch.zeros(1, 1, 768).to(self.device)
                a = torch.zeros(1, 1, 128).to(self.device)
                c = torch.zeros(1, 1, 64).to(self.device)

                opt.zero_grad()

                with autocast('cuda' if 'cuda' in self.device else 'cpu'):
                    try:
                        logits, _, _ = student(v, a, inp, c)
                        # Calculate offset for text in the stream
                        # Standard AEIOU order: V->A->C->T
                        # Check model config if available, else assume standard dims
                        offset = v.shape[1] + a.shape[1] + c.shape[1]

                        # Extract text logits
                        if logits.shape[1] >= offset + inp.shape[1]:
                            logits_txt = logits[:, offset: offset + inp.shape[1], :]
                        else:
                            # Fallback if model doesn't concat sensors in output
                            logits_txt = logits

                        loss = F.cross_entropy(logits_txt.reshape(-1, logits_txt.size(-1)), tgt.reshape(-1))
                    except RuntimeError:
                        continue  # Skip bad batches

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                cycles += 1

                # C. FEEDBACK & LOGGING
                loss_val = loss.item()
                short_text = clean_content[:50].replace('\n', ' ') + "..."
                self._trigger("cycle", cycles, loss_val, short_text)

                if cycles % 10 == 0:
                    self._log(f"Cycle {cycles}: Loss {loss_val:.3f} | {short_text}", "INFO")

                # D. HARVEST
                if self.config.harvest_enabled:
                    try:
                        # Rotate file if too big
                        if os.path.exists(harvest_path) and os.path.getsize(harvest_path) > self.config.max_file_size:
                            harvest_part += 1
                            harvest_path = os.path.join(self.harvest_dir, f"symbiosis_{ts}_p{harvest_part}.txt")
                            self._log(f"Harvest rotated to Part {harvest_part}", "HARVEST")

                        with open(harvest_path, "a", encoding="utf-8") as f:
                            f.write(f"{gen_text}\n<|endoftext|>\n")
                    except:
                        pass

                # E. AUTOSAVE
                if self.config.save_interval > 0 and cycles % self.config.save_interval == 0:
                    # Use direct pytorch save here or call LobeManager via callback if preferred
                    # Ideally, Symbiont shouldn't access LobeManager directly to keep decoupling strict,
                    # but we have the handles.
                    # HOWEVER, LobeManager has the `save_lobe` method. We can't easily call it without a ref.
                    # Best practice: Emit 'autosave' callback, let Plugin call manager.
                    self._trigger("autosave", cycles)  # Plugin handles actual save

                time.sleep(0.1)  # Breathe

        except Exception as e:
            self._log(f"Symbiosis Crash: {e}", "ERROR")
            import traceback
            traceback.print_exc()

        finally:
            # 3. RESTORE SENSES
            if self.config.safety_freeze:
                try:
                    if hasattr(student, 'vis_emb'):
                        for p in student.vis_emb.parameters(): p.requires_grad = True
                    if hasattr(student, 'aud_emb'):
                        for p in student.aud_emb.parameters(): p.requires_grad = True
                except:
                    pass
                self._log("Senses Unlocked.", "INFO")

            self.is_running = False
            self._trigger("finished")