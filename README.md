```markdown
# AEIOU Brain 🧠

**Finetune your own multimodal AI on local files — textbooks, ebooks, comics, videos, images, audio — all with an easy GUI.**

AEIOU is a complete, fully local AI ecosystem for training and experimenting with hybrid architectures on your home hardware. No cloud, no APIs, no data leaving your machine. Runs on **Windows, macOS, or Linux** (full GUI or headless CLI).

### Why AEIOU?
- **Local finetuning on your data**: Drop your personal files (PDFs, ebooks, comics CBZ, videos, images, audio, text) into `Training_Data` and train directly on them.
- **Rich multimodal support**: Handles mixed data automatically:
  - **Quad**: Image + Audio + Text + Control (full narrative richness)
  - **Triplet**: Video + Text, or Image + Audio + Text (great for illustrated books or lectures)
  - **Pair**: Image + Text, Audio + Text, Image + Audio
  - **Single**: Any lone file (images, audio clips, plain text, PDFs)
- **Hybrid architectures**: Mix autoregressive transformers with bidirectional discrete diffusion, game-theoretic pruning (Nash-inspired weight competition), delta learning, manifold routing (mHC), RoPE + SwiGLU, and more.
- **Persistent memory graph**: Hippocampus stores knowledge semantically with vector search and associative recall.
- **Creative factories**:
  - **Lecture Factory**: Turn PDFs/textbooks into sliced slides + narrated audio + paired text (perfect for study decks)
  - **Video Timeline Factory**: Slice videos into frame + audio clip + subtitle/text triplets (great for training on lectures/talks)
  - **Comic Scanner**: Extract pages from CBZ comics into image + text pairs
- **Evolution tools**: Dream State (offline consolidation), Symbiosis (distillation), Council (multi-lobe debate), RLM agents.
- **Headless mode**: Train on servers via `telepathy.py`.

Built solo in 2025–2026 on a single RTX 3080 Ti.

### How to Use
1. **Prepare Data**
   - Create folders inside `Training_Data` (e.g., `MyBooks`, `Lectures`, `Comics`).
   - Use matching filenames for multimodal pairs:
     - `book.pdf` + `book.mp3` → audio + text
     - `page.png` + `page.txt` → image + text
     - `video.mp4` + `video.srt` → video + subtitles (auto-sliced)
   - AEIOU detects quads, triplets, pairs, and singles automatically.

2. **Creating & Loading Lobes (Brains)**
   - Open the GUI (`python GUI.py`)
   - Go to **Cortex Control** tab (default on launch)
   - Select a lobe slot (LOBE 1–4) in the header
   - **Create a new lobe**:
     - Choose "Target Genetics" from dropdown (e.g., "MaskedDiffusion-mHC" for diffusion dreaming, "Tetra-Llama" for strong reasoning)
     - Click **INITIALIZE NEW** → creates fresh weights with selected architecture
   - **Load an existing lobe**:
     - Place `.pt` files in the `lobes/` folder (or use LOAD FILE button)
     - Select lobe slot → click **ACTIVATE LOBE** (or LOAD FILE to browse)
   - Save with **SAVE AS...** or auto-save during training
   - Tip: Start with diffusion genetics for creative tasks, AR for reasoning/text.

3. **Train/Finetune**
   - Go to **Transformer Trainer** (AR models) or **Diffusion Director** (diffusion models)
   - Click "SCAN FOLDER" → select your data folder
   - Filter by type (quad/triplet/pair/single) or extension in Census panel
   - Choose active lobe → START TRAINING
   - Monitor loss/telemetry in **Telemetry** tab

4. **Create Lectures (Lecture Factory)**
   - Put PDFs/textbooks in a folder
   - Open **Lecture Factory**
   - Scan folder → START PRODUCTION
   - Outputs structured slices (PNG slides + MP3 narration + TXT) ready for training

5. **Create Video Timelines (Video Timeline Factory)**
   - Put videos (optional .srt subtitles) in folder
   - Open **Video Timeline Factory**
   - Set extract FPS/audio seconds → START PRODUCTION
   - Outputs timed triplets (PNG frame + WAV clip + TXT subtitle)

6. **Generate Comics/Dreams**
   - **Comic Factory**: Story prompt → multi-panel generation
   - **Dream State**: Free-run consolidation/hallucination on chaos buffer
   - **Playground**: Chat/generate with active lobe

### Requirements
- Python 3.10+
- PyTorch (with CUDA/MPS for GPU)
- Other libs: tkinter, torchvision, torchaudio, pymupdf, magvit2-pytorch, etc.
Install via: `pip install torch torchvision torchaudio pymupdf magvit2-pytorch`

### Author
Created and maintained by **Frederick von Rönge**  
- GitHub: [@vonronge](https://github.com/vonronge)  
- LinkedIn: [frederick-von-rönge](https://www.linkedin.com/in/vonronge/)

### Quick Start
```bash
git clone https://github.com/vonronge/aeiou-brain.git
cd aeiou-brain
python genesis.py          # Setup folders
python GUI.py              # Launch GUI
# Headless training:
python telepathy.py --headless --mode=train --lobe=1 --data="./Training_Data/MyFolder"

MIT Licensed — experiment freely, share improvements.

Feedback and PRs welcome! This is an ongoing personal research project.
```
