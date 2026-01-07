"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
The Ribosome (v26.49 "Efficient Grid"):
- Vision: image_size=128 → default patch gives ~256 tokens (16x16 grid).
- Resize input to 128 for stability.
- No extra args (compatible with current library).
- Audio: Robust load + mono + padding.
- Safe vocab.
"""

import torch
import torch.nn as nn
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORTS ---
try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    from magvit2_pytorch import VideoTokenizer
    HAS_MAGVIT = True
except ImportError:
    HAS_MAGVIT = False

try:
    from transformers import EncodecModel, AutoProcessor, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

class Organelle_Ribosome(nn.Module):
    def __init__(self, device="cuda", golgi=None):
        super().__init__()
        self.device = device
        self.golgi = golgi
        print(f"[Ribosome] Initializing on {self.device}...")

        self.image_vocab_base = 40000
        self.audio_vocab_base = 60000

        # Dense Vision
        self.clip_model = None
        if HAS_CLIP:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
                print("[Ribosome] ✓ CLIP Ready")
            except Exception as e:
                print(f"[Ribosome] CLIP Failed: {e}")

        # Discrete Vision – Efficient 128x128
        self.magvit = None
        if HAS_MAGVIT:
            try:
                print("[Ribosome] Loading Efficient VideoTokenizer (image_size=128)...")
                self.magvit = VideoTokenizer(
                    image_size=128,          # → ~256 tokens with default patch
                    codebook_size=16384,
                    flash_attn=False
                ).to(device).eval()
                print("[Ribosome] ✓ VideoTokenizer Ready (~256 Tokens/Image)")
            except Exception as e:
                print(f"[Ribosome] Tokenizer Failed: {e}")
                import traceback
                traceback.print_exc()

        # Audio
        self.encodec = None
        self.audio_processor = None
        if HAS_TRANSFORMERS:
            try:
                self.encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
                self.audio_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
                print("[Ribosome] ✓ EnCodec Ready")
            except Exception as e:
                print(f"[Ribosome] EnCodec Failed: {e}")

        # Text
        self.tokenizer = None
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
                print("[Ribosome] ✓ T5 Ready")
            except Exception as e:
                print(f"[Ribosome] Tokenizer Failed: {e}")

    def _load_audio_robust(self, path: str):
        try:
            import torchaudio
            wav, sr = torchaudio.load(path)
            return wav, sr
        except Exception as e1:
            if HAS_SOUNDFILE:
                try:
                    data, sr = sf.read(path)
                    wav = torch.from_numpy(data).float().t()
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    return wav, sr
                except Exception as e2:
                    raise RuntimeError(f"Audio load failed: {e1}, soundfile {e2}")
            raise e1

    def _tokenize(self, text: str):
        if not self.tokenizer:
            return []
        return self.tokenizer.encode(text, truncation=True, max_length=1024)

    def ingest_packet(self, packet: dict):
        dense_v = None
        dense_a = None
        seq_parts = []

        # Visual
        if 'v' in packet and os.path.exists(packet['v']):
            if self.clip_model:
                try:
                    from PIL import Image
                    img = Image.open(packet['v']).convert("RGB")
                    pre = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        dense_v = self.clip_model.encode_image(pre).float().unsqueeze(1)
                except Exception as e:
                    if self.golgi:
                        self.golgi.warn(f"CLIP Fail: {e}", source="Ribosome")

            if self.magvit:
                try:
                    from torchvision import transforms
                    from PIL import Image

                    img = Image.open(packet['v']).convert("RGB")
                    tf = transforms.Compose([
                        transforms.Resize((128, 128)),  # Match tokenizer
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
                    ])
                    tens = tf(img).unsqueeze(0).unsqueeze(2).to(self.device)

                    with torch.no_grad():
                        codes = self.magvit.tokenize(tens)
                        print(f"[Ribosome] Raw codes shape: {codes.shape}")

                        v_tokens = codes.reshape(1, -1)
                        v_tokens = v_tokens + self.image_vocab_base

                        seq_parts.append(v_tokens)
                        print(f"[Ribosome] V({v_tokens.shape[1]}) tokens | Max ID: {v_tokens.max().item()}")
                except Exception as e:
                    print(f"[Ribosome] Visual Error: {e}")
                    import traceback
                    traceback.print_exc()

        # Audio
        if 'a' in packet and os.path.exists(packet['a']):
            if self.encodec and self.audio_processor:
                try:
                    wav, sr = self._load_audio_robust(packet['a'])

                    if sr != 24000:
                        import torchaudio
                        resampler = torchaudio.transforms.Resample(sr, 24000)
                        wav = resampler(wav)

                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)

                    inputs = self.audio_processor(
                        raw_audio=wav.squeeze(0).numpy(),
                        sampling_rate=24000,
                        return_tensors="pt",
                        padding=True
                    )

                    input_values = inputs["input_values"].to(self.device)
                    padding_mask = inputs.get("padding_mask")
                    if padding_mask is not None:
                        padding_mask = padding_mask.to(self.device)

                    with torch.no_grad():
                        outputs = self.encodec.encode(input_values, padding_mask=padding_mask)
                        codes = outputs.audio_codes[0][0]
                        a_tokens = codes + self.audio_vocab_base
                        seq_parts.append(a_tokens.unsqueeze(0))

                except Exception as e:
                    print(f"[Ribosome] Audio Error: {e}")

        # Text
        text_content = ""
        if 't' in packet and os.path.exists(packet['t']):
            try:
                with open(packet['t'], 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
            except Exception as e:
                print(f"Text Error: {e}")

        if text_content:
            t_ids = self._tokenize(text_content)
            if t_ids:
                t_seq = torch.tensor(t_ids, device=self.device).unsqueeze(0)
                seq_parts.append(t_seq)

        full_seq = None
        if seq_parts:
            try:
                full_seq = torch.cat(seq_parts, dim=1)
            except Exception as e:
                print(f"[Ribosome] Concat Error: {e}")

        return dense_v, dense_a, full_seq, None, None