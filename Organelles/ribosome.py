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

import fitz
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import tiktoken
from PIL import Image, ImageFile
import warnings
import os
import io
import random
import glob
import re
import zipfile
import json
import csv
import librosa
import torchaudio
import numpy as np
from .thalamus import Organelle_Thalamus

# --- CRITICAL FIX FOR CORRUPT JPEGS ---
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CODEC IMPORTS ---
try:
    from magvit2_pytorch import MagViT2

    HAS_MAGVIT = True
except ImportError:
    HAS_MAGVIT = False
    print(" ! Install: pip install magvit2-pytorch")

try:
    from torchaudio.models import encodec_model_24khz

    HAS_ENCODEC = True
except ImportError:
    HAS_ENCODEC = False

try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    fitz.TOOLS.mupdf_display_errors(False)
except:
    pass

warnings.filterwarnings("ignore")


class RibosomeConfig:
    """Central configuration for sensory resolutions"""

    def __init__(self):
        self.image_size = 256
        self.clip_frames = 16
        self.patch_size = 16


class Organelle_Ribosome:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.config = RibosomeConfig()

        print(f" > Ribosome v24.1 (Physics-Aware) initializing on {device}...")

        # Visual Cortex (feature extractor)
        weights = ViT_B_16_Weights.DEFAULT
        self.retina = vit_b_16(weights=weights).to(device)

        def forward_features(x):
            x = self.retina._process_input(x)
            n = x.shape[0]
            batch_class_token = self.retina.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.retina.encoder(x)
            return x

        self.retina.forward = forward_features
        self.retina.eval()
        self.visual_transform = weights.transforms()
        self.thalamus = Organelle_Thalamus(max_keep=96).to(device)

        self.tokenizer = tiktoken.get_encoding("gpt2")

        # --- VOCABULARY MAP ---
        self.text_vocab_base = 50257
        self.image_vocab_size = 16384
        self.image_vocab_base = self.text_vocab_base
        self.audio_vocab_size = 8192
        self.audio_vocab_base = self.text_vocab_base + self.image_vocab_size

        # --- VIDEO/IMAGE TOKENIZER ---
        if HAS_MAGVIT:
            try:
                spatial_tokens = (self.config.image_size // self.config.patch_size) ** 2
                self.magvit = MagViT2(
                    num_codes=self.image_vocab_size,  # FIXED ARGUMENT NAME
                    image_size=self.config.image_size,
                    dim=512,
                    depth=8,
                    num_tokens_per_block=spatial_tokens,
                    channels=3,
                    use_3d=True
                ).to(device)
                self.magvit.eval()
                print(f" > MAGViT-v2 Loaded (Res: {self.config.image_size}px)")
            except Exception as e:
                print(f" ! MAGViT Init Error: {e}")
                self.magvit = None
        else:
            self.magvit = None

        # --- AUDIO TOKENIZER ---
        if HAS_ENCODEC:
            try:
                self.encodec = encodec_model_24khz().to(device)
                self.encodec.set_target_bandwidth(6.0)
                self.encodec.eval()
                print(" > EnCodec Loaded")
            except:
                self.encodec = None
        else:
            self.encodec = None

        print(" > Ribosome Online.")

    def set_tokenizer(self, new_tokenizer):
        self.tokenizer = new_tokenizer

    def _tokenize(self, text):
        text = text[:10000]
        try:
            if hasattr(self.tokenizer, "encode") and not isinstance(self.tokenizer, tiktoken.Encoding):
                toks = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                toks = self.tokenizer.encode(text)

            if hasattr(toks, "ids"): toks = toks.ids
            if isinstance(toks, torch.Tensor): toks = toks.tolist()
            return toks[:1024]
        except:
            return [50256]

    # --- DECODERS ---
    def decode(self, tokens):
        text_tokens = []
        for t in tokens:
            if isinstance(t, torch.Tensor): t = t.item()
            if t < self.text_vocab_base:
                text_tokens.append(t)
        try:
            return self.tokenizer.decode(text_tokens)
        except:
            return ""

    def decode_image_tokens(self, tokens):
        if not self.magvit: return None
        indices = torch.tensor(tokens) - self.image_vocab_base
        indices = indices.clamp(0, self.image_vocab_size - 1).to(self.device)

        side_tokens = self.config.image_size // self.config.patch_size
        target_len = side_tokens ** 2

        if len(indices) < target_len:
            indices = torch.cat([indices, torch.zeros(target_len - len(indices), device=self.device).long()])

        indices = indices[:target_len].view(1, side_tokens, side_tokens)

        with torch.no_grad():
            if hasattr(self.magvit, 'decode_from_codes'):
                rec = self.magvit.decode_from_codes(indices)
            else:
                rec = self.magvit.decode_from_indices(indices)

        if rec.ndim == 5:
            rec = rec[:, :, 0, :, :]

        rec = (rec.clamp(-1, 1) + 1) / 2
        rec = rec[0].permute(1, 2, 0).cpu().numpy() * 255
        return Image.fromarray(rec.astype(np.uint8))

    def decode_audio_tokens(self, tokens):
        if not self.encodec: return None
        codes = torch.tensor(tokens) - self.audio_vocab_base
        codes = codes.clamp(0, 1023)

        num_books = 8
        if len(codes) % num_books != 0:
            pad = num_books - (len(codes) % num_books)
            codes = torch.cat([codes, torch.zeros(pad).long()])

        T = len(codes) // num_books
        codes = codes.view(T, num_books).permute(1, 0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            waveform = self.encodec.decode([codes])

        temp_path = os.path.abspath("temp_dream_audio.wav")
        torchaudio.save(temp_path, waveform[0].cpu(), 24000)
        return temp_path

    # --- CONTROL LOADER (PHYSICS INJECTION) ---
    def _load_control(self, path):
        """
        Reads a JSON file containing control vectors (e.g. Optical Flow)
        and pads it to the standard [1, 1, 64] embedding shape.
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            vec = data.get("control_vec", [0.0, 0.0])

            t = torch.zeros(1, 1, 64)
            t[0, 0, 0] = float(vec[0])
            t[0, 0, 1] = float(vec[1])

            return t.to(self.device)
        except Exception as e:
            return torch.zeros(1, 1, 64).to(self.device)

    # --- INGESTION ---
    def ingest_packet(self, packet):
        v, a, t, c, kept_idx = None, None, None, None, None

        if 't' in packet:
            with open(packet['t'], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            toks = self._tokenize(content)
            t = torch.tensor(toks).unsqueeze(0).to(self.device)

        if 'c' in packet and packet['c']:
            c = self._load_control(packet['c'])
        else:
            c = torch.zeros(1, 1, 64).to(self.device)

        if 'v' in packet and packet['v'] and self.magvit:
            try:
                img = Image.open(packet['v']).convert('RGB')
                img = img.resize((self.config.image_size, self.config.image_size))

                tens = torch.tensor(np.array(img)).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).unsqueeze(
                    2).to(self.device)

                with torch.no_grad():
                    if hasattr(self.magvit, 'tokenize'):
                        indices = self.magvit.tokenize(tens)
                    else:
                        fmap = self.magvit.encode(tens)
                        indices = self.magvit.quantize(fmap)[2]

                img_tokens = indices.flatten() + self.image_vocab_base

                if t is not None:
                    t = torch.cat([t, img_tokens.unsqueeze(0)], dim=1)
                else:
                    t = img_tokens.unsqueeze(0)

                with torch.no_grad():
                    px = self.visual_transform(img).unsqueeze(0).to(self.device)
                    full_vis = self.retina(px)
                    v, kept_idx = self.thalamus(full_vis)

            except Exception as e:
                pass

        audio_src = packet.get('a')
        if audio_src and self.encodec:
            try:
                wav, sr = torchaudio.load(audio_src)
                if sr != 24000: wav = torchaudio.functional.resample(wav, sr, 24000)
                wav = wav.mean(0, keepdim=True).to(self.device).unsqueeze(0)

                with torch.no_grad():
                    encoded_frames = self.encodec.encode(wav)
                    codes = encoded_frames[0][0]
                    flat_codes = codes.permute(1, 0).flatten()
                    aud_tokens = flat_codes + self.audio_vocab_base

                    if t is not None:
                        t = torch.cat([t, aud_tokens.unsqueeze(0)], dim=1)
                    else:
                        t = aud_tokens.unsqueeze(0)
            except:
                pass

        if v is None: v = torch.zeros(1, 1, 768).to(self.device)
        if a is None: a = torch.zeros(1, 1, 128).to(self.device)
        if t is None: t = torch.tensor([[50256]]).to(self.device)

        return v, a, t, c, kept_idx

    def render_text_to_image(self, text):
        return Image.new('RGB', (100, 100))

    def get_engram(self, p):
        return None

    def ingest_doc(self, p):
        return self.ingest_packet({'t': p})

    def ingest_cbz(self, p):
        return self.ingest_packet({'t': p})

    def ingest(self, p):
        return self.ingest_packet({'t': p})