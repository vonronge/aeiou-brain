"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Ribosome:
The Translation Engine.
Initializes neural backbones (Retina, MagViT, EnCodec),
owns the Membrane (I/O), and converts raw data into neural language (Tokens).
"""


import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import tiktoken
import numpy as np
from PIL import Image
import os
import torchaudio
import inspect

# Internal Organelles
try:
    from Organelles.thalamus import Organelle_Thalamus
    from Organelles.membrane import Organelle_Membrane
except ImportError:
    pass

# Optional Neural Codecs
try:
    from magvit2_pytorch import MagViT2

    HAS_MAGVIT = True
except ImportError:
    HAS_MAGVIT = False

try:
    from torchaudio.models import encodec_model_24khz

    HAS_ENCODEC = True
except ImportError:
    HAS_ENCODEC = False


class RibosomeConfig:
    image_size = 256
    patch_size = 16
    audio_sr = 24000


class Organelle_Ribosome:
    def __init__(self, device: str, golgi=None):
        self.device = device
        self.golgi = golgi
        self.config = RibosomeConfig()

        self._log("Initializing Neural Backbones...", "INFO")

        # 1. Text Tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        except:
            self.tokenizer = None
            self._log("Tiktoken failed.", "ERROR")

        # 2. Visual Cortex
        self.retina = None
        self.visual_transform = None
        try:
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
        except Exception as e:
            self._log(f"Retina init failed: {e}", "WARN")

        # 3. Thalamus
        self.thalamus = Organelle_Thalamus(max_keep=96).to(device)

        # 4. MagViT-v2 (Auto-Detect Args)
        self.magvit = None
        if HAS_MAGVIT:
            try:
                spatial_tokens = (self.config.image_size // self.config.patch_size) ** 2
                common_args = {
                    "image_size": self.config.image_size,
                    "dim": 512,
                    "depth": 8,
                    "num_tokens_per_block": spatial_tokens,
                    "channels": 3,
                    "use_3d": True
                }

                # Introspect
                sig = inspect.signature(MagViT2.__init__)
                params = sig.parameters

                if 'num_codes' in params:
                    self.magvit = MagViT2(num_codes=16384, **common_args).to(device)
                elif 'num_tokens' in params:
                    self.magvit = MagViT2(num_tokens=16384, **common_args).to(device)
                elif 'codebook_size' in params:
                    self.magvit = MagViT2(codebook_size=16384, **common_args).to(device)
                else:
                    self._log(f"Unknown MagViT params: {list(params.keys())}", "WARN")
                    # Last ditch effort
                    self.magvit = MagViT2(num_codes=16384, **common_args).to(device)

                self.magvit.eval()
                self._log("MagViT-v2 Online.", "SUCCESS")
            except Exception as e:
                self._log(f"MagViT Error: {e}", "ERROR")

        # 5. EnCodec
        self.encodec = None
        if HAS_ENCODEC:
            try:
                self.encodec = encodec_model_24khz().to(device)
                self.encodec.set_target_bandwidth(6.0)
                self.encodec.eval()
                self._log("EnCodec Online.", "SUCCESS")
            except Exception as e:
                self._log(f"EnCodec Error: {e}", "ERROR")

        self.membrane = Organelle_Membrane(
            device=self.device,
            golgi=self.golgi,
            magvit=self.magvit,
            encodec=self.encodec,
            thalamus=self.thalamus,
            transform=self.visual_transform
        )

        self.text_vocab_base = 50257
        self.image_vocab_base = 50257
        self.image_vocab_size = 16384
        self.audio_vocab_base = 50257 + 16384

    def _log(self, msg, tag="INFO"):
        if self.golgi:
            getattr(self.golgi, tag.lower(), self.golgi.info)(msg, source="Ribosome")
        else:
            print(f"[Ribosome:{tag}] {msg}")

    def set_tokenizer(self, new_tokenizer):
        self.tokenizer = new_tokenizer

    def ingest_packet(self, packet):
        v_tok, a_tok, text_str, c_emb = self.membrane.build_packet(packet)
        t_ids = self._tokenize(text_str)
        t_tok = torch.tensor(t_ids).unsqueeze(0).to(self.device)
        parts = []
        if a_tok is not None: parts.append(a_tok)
        if v_tok is not None: parts.append(v_tok)
        parts.append(t_tok)
        full_seq = torch.cat(parts, dim=1)
        v_feat = torch.zeros(1, 1, 768).to(self.device)
        a_feat = torch.zeros(1, 1, 128).to(self.device)
        return v_feat, a_feat, full_seq, c_emb, None

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
        if not self.magvit: return None
        indices = torch.tensor(tokens) - self.image_vocab_base
        indices = indices.clamp(0, self.image_vocab_size - 1).to(self.device)
        side = self.config.image_size // self.config.patch_size
        target_len = side ** 2
        if len(indices) < target_len:
            pad = torch.zeros(target_len - len(indices), device=self.device).long()
            indices = torch.cat([indices, pad])
        else:
            indices = indices[:target_len]
        indices = indices.view(1, side, side)
        with torch.no_grad():
            if hasattr(self.magvit, 'decode_from_codes'):
                rec = self.magvit.decode_from_codes(indices)
            else:
                rec = self.magvit.decode_from_indices(indices)
        if rec.ndim == 5: rec = rec[:, :, 0, :, :]
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
            wav = self.encodec.decode([codes])
        path = os.path.abspath("temp_decoded_audio.wav")
        torchaudio.save(path, wav[0].cpu(), 24000)
        return path