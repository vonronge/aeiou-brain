# AEIOU Brain 🧠

**Train your own multimodal AI at home — no cloud, no limits.**

AEIOU is a complete local AI ecosystem for experimenting with cutting-edge hybrid architectures on your own hardware. Runs on **Windows, macOS, or Linux** — with full GUI or headless mode.

### Key Features
- **True multimodality**: Unified V→A→C→T token stream (vision → audio → control → text) with MAGViT2 video/image tokenization + EnCodec audio.
- **Architectural playground**: Swap between autoregressive transformers, bidirectional discrete diffusion for dreaming/refinement, game-theoretic pruning (Nash-equilibrium-inspired), delta operators, manifold-constrained hyper-connections (mHC), RoPE + SwiGLU, and more.
- **Persistent memory**: Semantic knowledge graph (Hippocampus) with vector search and associative recall.
- **Evolution tools**: Symbiosis (teacher-student distillation), Dream State (offline consolidation), Council (multi-lobe debate), RLM (restricted Python agents), and factories for comics, video timelines, and data structuring.
- **Cross-platform & headless**: Polished Tkinter GUI + CLI (`telepathy.py`) for servers.
- **No external APIs**: Everything runs locally on your GPU (CUDA/MPS/CPU fallback).

### Author
Created and maintained by **Frederick von Rönge**  
- GitHub: [@vonronge](https://github.com/vonronge)  
- LinkedIn: [frederick-von-rönge](https://www.linkedin.com/in/vonronge/)  

Built solo in 2025-2026 on a single RTX 3080 Ti.

### Quick Start
```bash
git clone https://github.com/vonronge/aeiou-brain.git
cd aeiou-brain
python genesis.py          # Sets up folders/structure
python GUI.py              # Launch the full cortex
# Or headless:
python telepathy.py --headless --mode=train --lobe=1 --data=./Training_Data
