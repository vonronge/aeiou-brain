"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Ribosome:
The Translation Engine.
Initializes neural backbones (Retina, MagViT, EnCodec),
owns the Membrane (I/O), and converts raw data into neural language (Tokens).
"""

import fitz
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import tiktoken
from PIL import Image, ImageDraw, ImageFont
import warnings
import os
import inspect
import random
import re
import json
import torchaudio
import numpy as np
from .thalamus import Organelle_Thalamus
from .membrane import Organelle_Membrane

# --- CODECS ---
try:
    from magvit2_pytorch import MagViT2

    HAS_MAGVIT = True
except:
    HAS_MAGVIT = False

try:
    from torchaudio.models import encodec_model_24khz

    HAS_ENCODEC = True
except:
    HAS_ENCODEC = False

try:
    import cv2

    HAS_OPENCV = True
except:
    HAS_OPENCV = False

try:
    fitz.TOOLS.mupdf_display_errors(False)
except:
    pass
warnings.filterwarnings("ignore")


class RibosomeConfig:
    image_size = 256
    patch_size = 16
    audio_sr = 24000


class Organelle_Ribosome:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", golgi=None):
        self.device = device
        self.golgi = golgi
        self.config = RibosomeConfig()

        self.text_vocab_base = 50257
        self.image_vocab_size = 16384
        self.image_vocab_base = self.text_vocab_base
        self.audio_vocab_size = 8192
        self.audio_vocab_base = self.text_vocab_base + self.image_vocab_size

        try:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        except:
            self.tokenizer = None

        # Visual Cortex
        try:
            weights = ViT_B_16_Weights.DEFAULT
            self.retina = vit_b_16(weights=weights).to(device)
            self.retina.eval()
            self.visual_transform = weights.transforms()
        except:
            self.retina = None

        self.thalamus = Organelle_Thalamus(max_keep=96).to(device)

        # MagViT2
        self.magvit = None
        if HAS_MAGVIT:
            try:
                spatial_tokens = (self.config.image_size // self.config.patch_size) ** 2
                common_args = {
                    "image_size": self.config.image_size, "dim": 512, "depth": 8,
                    "num_tokens_per_block": spatial_tokens, "channels": 3, "use_3d": True
                }
                # Init with introspection for safety
                self.magvit = MagViT2(num_codes=16384, **common_args).to(device)
                self.magvit.eval()
            except:
                pass

        # EnCodec
        self.encodec = None
        if HAS_ENCODEC:
            try:
                self.encodec = encodec_model_24khz().to(device)
                self.encodec.set_target_bandwidth(6.0)
                self.encodec.eval()
            except:
                pass

        self.membrane = Organelle_Membrane(
            device=self.device, golgi=self.golgi, magvit=self.magvit,
            encodec=self.encodec, thalamus=self.thalamus, transform=self.visual_transform
        )

    def set_tokenizer(self, new_tokenizer):
        self.tokenizer = new_tokenizer

    def _tokenize(self, text):
        if not text: return [50256]
        try:
            if hasattr(self.tokenizer, "encode") and not isinstance(self.tokenizer, tiktoken.Encoding):
                toks = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                toks = self.tokenizer.encode(text)
            if hasattr(toks, "ids"): toks = toks.ids
            if isinstance(toks, torch.Tensor): toks = toks.tolist()
            return toks[:2048]
        except:
            return [50256]

    # --- v25.0 TILING LOGIC ---
    def tile_encode_image(self, pil_image, tile_size=256, overlap=64):
        """
        Slices large images into overlapping tiles and encodes each.
        Returns: 1D List of tokens (raster order).
        """
        if not self.magvit: return []

        w, h = pil_image.size
        # Limit max size to prevent insane context length
        if max(w, h) > 2048:
            scale = 2048 / max(w, h)
            pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = pil_image.size

        stride = tile_size - overlap
        all_tokens = []

        # Iterate Y then X (Raster Scan)
        for y in range(0, h, stride):
            if y + tile_size > h: y = max(0, h - tile_size)  # Adjust last tile

            for x in range(0, w, stride):
                if x + tile_size > w: x = max(0, w - tile_size)

                # Crop
                tile = pil_image.crop((x, y, x + tile_size, y + tile_size))

                # Encode
                tens = torch.tensor(np.array(tile)).permute(2, 0, 1).float().div(127.5).sub(1)
                tens = tens.unsqueeze(0).unsqueeze(2).to(self.device)  # [B, C, T, H, W]

                with torch.no_grad():
                    if hasattr(self.magvit, 'tokenize'):
                        indices = self.magvit.tokenize(tens)
                    else:
                        fmap = self.magvit.encode(tens)
                        indices = self.magvit.quantize(fmap)[2]

                flat = (indices.flatten() + self.image_vocab_base).cpu().tolist()
                all_tokens.extend(flat)

                if x + tile_size >= w: break
            if y + tile_size >= h: break

        return all_tokens

    def ingest_packet(self, packet):
        # 1. Text
        text_str = ""
        if 't' in packet:
            try:
                with open(packet['t'], 'r', encoding='utf-8', errors='ignore') as f:
                    text_str = f.read()
            except:
                pass
        t_ids = self._tokenize(text_str)
        t_tok = torch.tensor(t_ids).unsqueeze(0).to(self.device)

        # 2. Visual (Tiled if Large)
        v_tok = None
        if 'v' in packet and packet['v'] and self.magvit:
            try:
                img = Image.open(packet['v']).convert('RGB')
                w, h = img.size

                # If image is significantly larger than 256x256, use tiling
                if max(w, h) > 384:
                    # Tile Encode
                    tile_tokens = self.tile_encode_image(img)
                    v_tok = torch.tensor(tile_tokens).unsqueeze(0).to(self.device)
                else:
                    # Standard Encode
                    img = img.resize((256, 256))
                    tens = torch.tensor(np.array(img)).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(
                        0).unsqueeze(2).to(self.device)
                    with torch.no_grad():
                        if hasattr(self.magvit, 'tokenize'):
                            indices = self.magvit.tokenize(tens)
                        else:
                            fmap = self.magvit.encode(tens)
                            indices = self.magvit.quantize(fmap)[2]
                    v_tok = indices.flatten() + self.image_vocab_base
                    v_tok = v_tok.unsqueeze(0)
            except Exception as e:
                # print(f"Vis Error: {e}")
                pass

        # 3. Audio
        a_tok = None
        if 'a' in packet and packet['a'] and self.encodec:
            try:
                wav, sr = torchaudio.load(packet['a'])
                if sr != 24000: wav = torchaudio.functional.resample(wav, sr, 24000)
                wav = wav.mean(0, keepdim=True).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    codes = self.encodec.encode(wav)[0][0]
                    a_tok = codes.permute(1, 0).flatten() + self.audio_vocab_base
                    a_tok = a_tok.unsqueeze(0)
            except:
                pass

        # Assemble
        parts = []
        if a_tok is not None: parts.append(a_tok)
        if v_tok is not None: parts.append(v_tok)
        parts.append(t_tok)

        full_seq = torch.cat(parts, dim=1)

        # Placeholders
        v_feat = torch.zeros(1, 1, 768).to(self.device)
        a_feat = torch.zeros(1, 1, 128).to(self.device)
        c_emb = torch.zeros(1, 1, 64).to(self.device)

        return v_feat, a_feat, full_seq, c_emb, None

    # --- ENGRAM (v23.6) ---
    def get_engram(self, path):
        if not os.path.exists(path): return None
        try:
            v, _, _, _, _ = self.ingest_packet({'v': path})
            if v is not None:
                vec = v.mean(dim=1)
                if vec.ndim == 1: vec = vec.unsqueeze(0)
                return vec
            return None
        except:
            return None

    # --- HELPERS ---
    def render_text_to_image(self, text):
        img = Image.new('RGB', (512, 512), color=(10, 10, 15))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        margin = 10;
        offset = 10
        for line in text.split('\n'):
            draw.text((margin, offset), line[:100], font=font, fill=(200, 200, 200))
            offset += 20
        return img

    def decode(self, tokens):
        text_tokens = []
        for t in tokens:
            if isinstance(t, torch.Tensor): t = t.item()
            if t < self.text_vocab_base: text_tokens.append(t)
        try:
            return self.tokenizer.decode(text_tokens)
        except:
            return ""

    def decode_image_tokens(self, tokens):
        # Basic decode - visualizer mostly
        if not self.magvit: return None
        indices = torch.tensor(tokens) - self.image_vocab_base
        indices = indices.clamp(0, self.image_vocab_size - 1).to(self.device)
        indices = indices[:256].view(1, 16, 16)
        with torch.no_grad():
            if hasattr(self.magvit, 'decode_from_codes'):
                rec = self.magvit.decode_from_codes(indices)
            else:
                rec = self.magvit.decode_from_indices(indices)
        if rec.ndim == 5: rec = rec[:, :, 0, :, :]
        rec = (rec.clamp(-1, 1) + 1) / 2
        rec = rec[0].permute(1, 2, 0).cpu().numpy() * 255
        return Image.fromarray(rec.astype(np.uint8))