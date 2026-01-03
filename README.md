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
   - Create a folder inside `Training_Data` (e.g., `MyBooks`, `Lectures`, `Comics`).
   - Drop files with matching filenames:
     - `lecture.pdf` + `lecture.mp3` → audio + text pair
     - `page.png` + `page.txt` → image + text pair
     - `video.mp4` + `video.srt` → video + subtitles (auto-sliced into triplets)
     - Full books: PDF + optional audio narration → Lecture Factory processes into quads/triplets
   - AEIOU automatically detects quads, triplets, pairs, and singles.

2. **Train/Finetune**
   - Launch `GUI.py`
   - Go to **Transformer Trainer** or **Diffusion Director**
   - Click "SCAN FOLDER" → select your data folder
   - Filter by type (quad/triplet/pair/single) or extension in the Census panel
   - Choose lobe/genetics → START TRAINING
   - Watch telemetry graphs for loss curves

3. **Create Lectures (Lecture Factory)**
   - Put PDFs/textbooks in a folder
   - Open **Lecture Factory** plugin
   - Scan folder → START PRODUCTION
   - Outputs: sliced PNG slides + MP3 narration + TXT transcripts (perfect paired data for training)

4. **Create Video Timelines (Video Timeline Factory)**
   - Put videos (with optional .srt subtitles) in a folder
   - Open **Video Timeline Factory**
   - Set FPS/audio context → START PRODUCTION
   - Outputs: timed PNG frames + WAV clips + TXT subtitles (triplets ready for training)

5. **Generate Comics/Dreams**
   - Use **Comic Factory** for story → panel generation
   - **Dream State** for free-running consolidation/hallucination

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
