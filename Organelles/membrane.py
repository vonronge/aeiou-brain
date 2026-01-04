"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Membrane:
Selective permeability for data.
Handles physical I/O: Loading images, audio, and physics data from disk
and converting them into raw tensors on the correct device.
"""

import os
import torch
import torchaudio
import json
import numpy as np
from PIL import Image, ImageFile
import traceback

# Corrupt JPEG handling
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Organelle_Membrane:
    def __init__(self, device: str, golgi=None, magvit=None, encodec=None, thalamus=None, transform=None):
        self.device = device
        self.golgi = golgi

        # Hardware Backbones (Usually passed from Ribosome init)
        self.magvit = magvit
        self.encodec = encodec
        self.thalamus = thalamus
        self.visual_transform = transform

        # Config
        self.image_size = 256
        self.patch_size = 16
        self.audio_sr = 24000

        # Vocab Offsets (Hardcoded for stability, move to config later)
        self.vocab_img_base = 50257
        self.vocab_aud_base = 66641

    def _log(self, level, msg):
        if self.golgi:
            getattr(self.golgi, level.lower())(msg, source="Membrane")
        else:
            print(f"[Membrane:{level}] {msg}")

    # --- VISUAL INGESTION ---
    def ingest_image(self, path: str):
        """
        Loads image -> MagViT Tokens AND Thalamus Features.
        Returns: (tokens [1, 256], features [1, N, D]) or (None, None)
        """
        if not self.magvit or not os.path.exists(path): return None, None

        try:
            # 1. Load & Preprocess
            img = Image.open(path).convert('RGB')
            img_resized = img.resize((self.image_size, self.image_size))

            # Tensor for MagViT ([-1, 1])
            tens = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float()
            tens = tens.div(127.5).sub(1).unsqueeze(0).unsqueeze(2).to(self.device)

            # 2. Tokenize (MagViT)
            with torch.no_grad():
                if hasattr(self.magvit, 'tokenize'):
                    indices = self.magvit.tokenize(tens)
                else:
                    fmap = self.magvit.encode(tens)
                    indices = self.magvit.quantize(fmap)[2]

            img_tokens = indices.flatten() + self.vocab_img_base

            # 3. Feature Extraction (Thalamus/ViT)
            vis_features = None
            if self.thalamus and self.visual_transform:
                # Transform for ViT (Standard ImageNet norm)
                px = self.visual_transform(img).unsqueeze(0).to(self.device)

                # We assume the retina is accessible or passed.
                # Ideally, Thalamus should own the Retina (ViT).
                # For now, we return None if retina isn't tightly bound here,
                # but Membrane usually focuses on TOKENS.
                # Features are a "nice to have" for the routing mechanism.
                pass

            return img_tokens.unsqueeze(0), None

        except Exception as e:
            self._log("WARN", f"Image Ingest Failed ({os.path.basename(path)}): {e}")
            return None, None

    # --- AUDIO INGESTION ---
    def ingest_audio(self, path: str):
        """
        Loads audio -> EnCodec Tokens.
        Returns: tokens [1, N] or None
        """
        if not self.encodec or not os.path.exists(path): return None

        try:
            wav, sr = torchaudio.load(path)

            # Resample
            if sr != self.audio_sr:
                wav = torchaudio.functional.resample(wav, sr, self.audio_sr)

            # Mono & Batch
            wav = wav.mean(0, keepdim=True).unsqueeze(0).to(self.device)

            with torch.no_grad():
                encoded_frames = self.encodec.encode(wav)
                codes = encoded_frames[0][0]  # [K, T]
                flat_codes = codes.permute(1, 0).flatten()  # Interleaved

            aud_tokens = flat_codes + self.vocab_aud_base
            return aud_tokens.unsqueeze(0)

        except Exception as e:
            self._log("WARN", f"Audio Ingest Failed ({os.path.basename(path)}): {e}")
            return None

    # --- CONTROL / PHYSICS INGESTION ---
    def ingest_control(self, path: str):
        """
        Loads JSON physics data -> Tensor [1, 1, 64].
        """
        c_tensor = torch.zeros(1, 1, 64).to(self.device)

        if not os.path.exists(path): return c_tensor

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            vec = data.get("control_vec", [0.0, 0.0])

            # Simple embedding strategy:
            # pos 0 = x velocity, pos 1 = y velocity
            # Future expansion: pos 2 = acceleration, etc.
            c_tensor[0, 0, 0] = float(vec[0])
            c_tensor[0, 0, 1] = float(vec[1])

            return c_tensor
        except Exception as e:
            self._log("WARN", f"Control Ingest Failed: {e}")
            return c_tensor

    # --- PACKET ASSEMBLER ---
    def build_packet(self, files_dict: dict):
        """
        Orchestrates the loading of a raw file packet into tensors.
        files_dict example: {'v': 'path.png', 'a': 'path.wav', 't': 'path.txt', 'c': 'path.json'}
        Returns: (v_tokens, a_tokens, t_tokens, c_tensor)
        """
        # Placeholders
        v_tok, a_tok, t_tok = None, None, None

        # 1. Visual
        if 'v' in files_dict:
            v_tok, _ = self.ingest_image(files_dict['v'])

        # 2. Audio
        if 'a' in files_dict:
            a_tok = self.ingest_audio(files_dict['a'])

        # 3. Control
        if 'c' in files_dict:
            c_emb = self.ingest_control(files_dict['c'])
        else:
            c_emb = torch.zeros(1, 1, 64).to(self.device)

        # 4. Text (Still handled by Ribosome mostly, but we can read the string here)
        # We return the RAW string for text, as tokenization is the Ribosome's job.
        text_content = ""
        if 't' in files_dict and os.path.exists(files_dict['t']):
            try:
                with open(files_dict['t'], 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
            except:
                pass

        return v_tok, a_tok, text_content, c_emb