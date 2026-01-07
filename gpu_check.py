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

"""
AEIOU Deep Diagnostic Tool (v26.27)
Run this after rebuilding the environment.
"""
import sys
import os
import subprocess

print("\n" + "=" * 60)
print("      AEIOU SYSTEM DIAGNOSTIC & INTEGRITY CHECK")
print("=" * 60 + "\n")

# --- 1. CORE PYTHON & SYSTEM ---
print(f"[SYSTEM] Python: {sys.version.split()[0]}")
try:
    import numpy

    print(f"[SYSTEM] NumPy:  {numpy.__version__} (Should be < 2.0.0)")
except ImportError:
    print("[SYSTEM] NumPy:  MISSING (Critical)")

# --- 2. THE ENGINE (PyTorch) ---
try:
    import torch

    print(f"[ENGINE] PyTorch: {torch.__version__}")
    print(f"[ENGINE] CUDA:    {torch.version.cuda}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[ENGINE] GPU:     {gpu_name} ({vram:.1f} GB VRAM)")

        # Simple Tensor Test
        x = torch.tensor([1.0]).cuda()
        print("[ENGINE] Tensor:  OK (Read/Write to VRAM successful)")
    else:
        print("[ENGINE] GPU:     NOT DETECTED (System will be blind/slow)")
except ImportError:
    print("[ENGINE] PyTorch: CRITICAL FAILURE (Not installed)")

print("-" * 30)

# --- 3. THE ORGANS (Model Libraries) ---
# We try to initialize them to prove they aren't just installed, but working.

# A. MagViT2 (Vision Tokenizer)
try:
    from magvit2_pytorch import MagViT2

    print("[ORGAN]  MagViT2: INSTALLED")
    # Stress Test
    try:
        if torch.cuda.is_available():
            model = MagViT2(image_size=64, init_dim=32, stages=(1,), dim_mults=(1,), flash_attn=False).cuda()
            dummy = torch.randn(1, 3, 64, 64).cuda()
            out = model(dummy)
            print("[ORGAN]  MagViT2: OPERATIONAL (Test Pass on GPU)")
        else:
            print("[ORGAN]  MagViT2: UNTESTED (No GPU)")
    except Exception as e:
        print(f"[ORGAN]  MagViT2: BROKEN ({e})")
except ImportError:
    print("[ORGAN]  MagViT2: MISSING (Vision will fail/fallback)")

# B. Torchaudio / EnCodec (Audio Tokenizer)
try:
    import torchaudio

    print(f"[ORGAN]  Audio:   INSTALLED ({torchaudio.__version__})")
    try:
        from torchaudio.models import encodec_model_24khz

        model = encodec_model_24khz()
        print("[ORGAN]  EnCodec: OPERATIONAL (Model definition found)")
    except Exception as e:
        print(f"[ORGAN]  EnCodec: BROKEN ({e})")
except ImportError:
    print("[ORGAN]  Audio:   MISSING (Hearing will fail)")

# C. Transformers (Text Tokenizer)
try:
    import transformers

    print(f"[ORGAN]  Transf:  INSTALLED ({transformers.__version__})")
except ImportError:
    print("[ORGAN]  Transf:  MISSING (Text processing will fail)")

# D. Playwright (Lecture Factory)
try:
    from playwright.sync_api import sync_playwright

    print("[TOOL]   Playwrt: INSTALLED")
except ImportError:
    print("[TOOL]   Playwrt: MISSING (Lecture Factory fonts disabled)")

# E. OCR
try:
    import pytesseract

    path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(path):
        print(f"[TOOL]   OCR:     FOUND at {path}")
    else:
        print("[TOOL]   OCR:     WARNING (Binary not found at default path)")
except ImportError:
    print("[TOOL]   OCR:     MISSING Lib")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("If all status checks above are OK/OPERATIONAL, you may launch GUI.py")
print("=" * 60 + "\n")
input("Press Enter to exit...")