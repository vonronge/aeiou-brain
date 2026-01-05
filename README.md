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

#### 1. Prepare Data
- Create folders inside `Training_Data` (e.g., `MyBooks`, `Lectures`, `Comics`).
- Use matching filenames for multimodal pairs:
  - `book.pdf` + `book.mp3` → audio + text
  - `page.png` + `page.txt` → image + text
  - `video.mp4` + `video.srt` → video + subtitles (auto-sliced)
- AEIOU detects quads, triplets, pairs, and singles automatically.

#### 2. Creating & Loading Lobes (Brains)
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
- *Tip: Start with diffusion genetics for creative tasks, AR for reasoning/text.*

#### 3. Train/Finetune
- Go to **Transformer Trainer** (AR models) or **Diffusion Director** (diffusion models)
- Click "SCAN FOLDER" → select your data folder
- Filter by type (quad/triplet/pair/single) or extension in Census panel
- Choose active lobe → START TRAINING
- Monitor loss/telemetry in **Telemetry** tab

#### 4. Create Lectures (Lecture Factory)
- Put PDFs/textbooks in a folder
- Open **Lecture Factory**
- Scan folder → START PRODUCTION
- Outputs structured slices (PNG slides + MP3 narration + TXT) ready for training

#### 5. Create Video Timelines (Video Timeline Factory)
- Put videos (optional .srt subtitles) in folder
- Open **Video Timeline Factory**
- Set extract FPS/audio seconds → START PRODUCTION
- Outputs timed triplets (PNG frame + WAV clip + TXT subtitle)

#### 6. Generate Comics/Dreams
- **Comic Factory**: Story prompt → multi-panel generation
- **Dream State**: Free-run consolidation/hallucination on chaos buffer
- **Playground**: Chat/generate with active lobe

### Spotlight: MaskedDiffusion-mHC Genetics

This architecture ("MaskedDiffusion-mHC") is one of the more advanced built-in options and showcases several experimental techniques:

- **Bidirectional discrete diffusion**: Unlike autoregressive models that generate left-to-right, this uses masked diffusion on discrete tokens, allowing parallel refinement across the entire sequence. Ideal for "dreaming" (free generation) and iterative improvement.

- **Modality-aware masking curriculum**:
  - Uniform random masking (early training)
  - Cross-modal masking (randomly drop entire modalities)
  - Fine-grained per-modality rates (high for vision, lower for text)
  - Enables robust multimodal understanding by forcing the model to reconstruct missing senses.

- **mHC routing (manifold-constrained hyper-connections)**:
  - Splits embeddings into streams
  - Learned gating + Sinkhorn-Knopp normalization for efficient residual mixing
  - Adds dynamic routing without full Mixture-of-Experts overhead

- **Deep Delta learning**: Rank-1 updates via projected residuals, helping stabilize training and improve representation efficiency.

- **Game-theoretic pruning**: StrategicLinear layers where weights "compete" via learned participation (alpha) penalized in loss—emergent sparsity without explicit pruning steps.

- **Confidence-driven generation**: Iterative denoising with greedy unmasking of highest-confidence tokens, plus final forced sampling to avoid blanks.

- **Time-conditioned blocks**: Standard diffusion timestep embedding injected per layer.

These combine to make a strong refiner/hallucinator—great for creative tasks or consolidating chaotic data in Dream State.

*Try initializing a lobe with this genetics in **Cortex Control** to experiment!*

### Adding New Genetics (Model Architectures)
AEIOU is an experimental playground—adding new "genetics" (architectures) is easy and encouraged.

1. Create a new `.py` file in `Genetics/` (e.g., `my_custom_arch.py`).
2. Include:
   - `INFO` dict (for GUI dropdown)
   - `NucleusConfig` class (hyperparams)
   - `Model` class inheriting from `MultimodalBase`

   **Minimal template:**
   ```python
   INFO = {
       "name": "My Custom Arch",
       "desc": "Short description",
       "vram_train": "8 GB",
       "vram_run": "4 GB"
   }

   class NucleusConfig:
       def __init__(self):
           self.vocab_size = 72000
           self.embed_dim = 768
           self.n_layers = 12
           self.n_heads = 12
           # Add custom params

   class Model(MultimodalBase):
       def __init__(self, config=None):
           if config is None: config = NucleusConfig()
           super().__init__(config)
           # Your layers here

       def forward(self, v, a, t, c=None):
           x = self.embed_inputs(v, a, t, c)
           # Custom processing
           return self.head(x), None, None
   ```

3. Restart GUI—new option appears in Cortex Control.

**Pro Tip: Use an AI to Generate Code** Paste this into Grok, Gemini, Claude, or any strong LLM to brainstorm a full genetics file:

```text
I'm adding a new model architecture to AEIOU Brain, a local multimodal PyTorch project.
The base class is MultimodalBase (handles V→A→C→T fusion with projections and RoPE).

Implement [your idea here, e.g. "a hybrid of Llama RoPE/SwiGLU with discrete diffusion masking for better long-context reasoning"].

Constraints:
- embed_dim = 768
- 12 layers, 12 heads
- Inherit from MultimodalBase
- Use existing utils (RMSNorm, RotaryEmbedding, apply_rope, etc.)

Output:
1. INFO dict
2. NucleusConfig class
3. Full Model class with __init__ and forward()
```
Iterate by pasting back output and asking for fixes.

PRs with new genetics welcome!

### Author
Created and maintained by **Frederick von Rönge** - GitHub: [@vonronge](https://github.com/vonronge)  
- LinkedIn: [frederick-von-rönge](https://www.linkedin.com/in/vonronge/)

### Quick Start
```bash
git clone [https://github.com/vonronge/aeiou-brain.git](https://github.com/vonronge/aeiou-brain.git)
cd aeiou-brain
python genesis.py          # Setup folders
python GUI.py              # Launch GUI
# Headless training:
python telepathy.py --headless --mode=train --lobe=1 --data="./Training_Data/MyFolder"
```


# 🧠 System Architecture

The AEIOU Brain is architected as a synthetic organism. The **Organelles** (Backend) handle the biological functions (processing, learning, memory), while the **Tabs** (Frontend) act as the Cortex, allowing you to consciously direct those functions.

---

## 🧬 Part 1: The Organelles (Backend Systems)
*Located in the `Organelles/` folder. These internal organs run the machinery.*

* **The Golgi (Apparatus)**
    * **Function:** Central Nervous System & Messaging.
    * **Role:** Captures logs, errors, and status updates from all other organs and routes them to the correct destination (GUI logs, terminal, or log files). If an organ fails, the Golgi reports the "pain."

* **The Phagus**
    * **Function:** Homeostasis & Environment.
    * **Role:** Manages the `settings.json` file. Knows where folders are, stores UI theme preferences, and remembers window size/position. Ensures the environment is stable for the brain to live in.

* **The Ribosome**
    * **Function:** The Universal Translator.
    * **Role:** Converts raw matter (Images, Audio, Text) into "Proteins" (Tokens/Tensors) that the neural network can understand.
    * **Sub-Systems:**
        * **Membrane:** Physical barrier handling file I/O.
        * **Retina:** Extracts visual features from images.
        * **MagViT:** Compresses images into discrete visual tokens.
        * **EnCodec:** Compresses audio into discrete sound tokens.

* **The Lobe Manager**
    * **Function:** Stem Cell Differentiation & Management.
    * **Role:** Responsible for the life cycle of AI models ("Lobes"). Creates new blank models, loads existing ones into VRAM, saves them to disk, and tracks which "Genetics" (Model Architecture) each Lobe uses.

* **The Cytoplasm**
    * **Function:** The Medium of Growth (Training).
    * **Role:** The engine room. Takes Tensors from the Ribosome and the Model from the Lobe Manager to run training loops. Handles backpropagation, loss calculation, and optimization (weight updates).

* **The Hippocampus**
    * **Function:** Long-Term Semantic Memory.
    * **Role:** A graph database storing concepts as "Engrams" (Nodes). Links related ideas together (e.g., "Fire" -> is hot -> "Burn"), allowing the AI to remember facts beyond its context window.

* **The Reticulum**
    * **Function:** Waste Disposal & Repair.
    * **Role:** Scans the Hippocampus for "dead" memories (broken links, empty nodes) and prunes them. Ensures the knowledge graph remains healthy and efficient.

* **The Symbiont**
    * **Function:** Knowledge Transfer Agent.
    * **Role:** Runs "Distillation" sessions where a large, mature "Teacher" Lobe transfers its knowledge to a smaller, faster "Student" Lobe, compressing intelligence into a more efficient form.

* **The Cytosis**
    * **Function:** The Hallucination Engine (Inference).
    * **Role:** The opposite of the Cytoplasm. Instead of learning, it *generates*. Handles complex sampling loops required to produce images (Diffusion) or text (Autoregression).

---

## 🖥️ Part 2: The Cortex (Frontend / Tabs)
*Located in the `Plugins/` folder. These are the user interfaces that control the Organelles.*

### **Core Controls**
* **Cortex Control (`tab_cortex`):** The main dashboard. Load/unload Lobes into the 4 available memory slots, create new brains, and view genetic stats.
* **Settings (`tab_settings`):** The control panel for the Phagus. Change UI themes, scaling, and file paths.
* **Visual Cortex (`tab_graphs`):** A real-time telemetry monitor showing training loss curves. Essential for verifying if the AI is learning or diverging.

### **Training Centers**
* **Transformer Trainer (`tab_trainer`):** The classroom for text and multimodal understanding. Teaches the AI to *read* and *predict* (Next-Token Prediction).
* **Diffusion Director (`tab_diffusion_trainer`):** The art school. Teaches the AI to *draw* images by denoising static (Generative Diffusion).
* **RL Gymnasium (`tab_rlm`):** The gym. Connects the AI to simulated environments (like games or physics sims) and rewards it for good behavior (Reinforcement Learning).
* **Neural Symbiosis (`tab_symbiosis`):** The distillation lab. Connect a Teacher Lobe and a Student Lobe here to compress knowledge.

### **Creative Studios**
* **Playground (`tab_playground`):** The chat room. A direct interface to talk to the AI, send images, and observe responses.
* **Dream Studio (`tab_dream`):** The art studio. Ask the AI to generate images, audio, or stories based on a prompt.
* **Comic Architect (`tab_comic`):** A specialized sequencer that forces the AI to generate 4 consistent panels and stitches them into a comic page layout.
* **The Council (`tab_council`):** A "Mixture of Experts" simulator. Submits a prompt to *all* active Lobes simultaneously, allowing them to debate the answer or synthesize a consensus.

### **Data Factories (The Digestive System)**
* **General Factory (`tab_factory`):** The chew toy. Grinds loose images and text files to standardize them (resize images, fix text encoding) for the Ribosome.
* **Video Factory (`tab_video_factory`):** The butcher. Chops video files into "visual-audio pairs" (10-second chunks) for temporal training.
* **Lecture Factory (`tab_lecture_factory`):** The scholar. Reads PDF textbooks (OCR + TTS) and creates a synchronized dataset of Text + Audio + Visuals.
* **PDF Repair (`tab_pdf_repair`):** A utility tool to fix broken headers in corrupt PDF files.

### **Memory Management**
* **Memory Graph (`tab_memory`):** The file explorer for the Hippocampus. View, search, and manually edit the AI's long-term memories.
* **Memory Agent (`tab_memory_agent`):** The gardener. Runs the Reticulum's cleaning cycles or asks an active Lobe to summarize and condense old memories ("Sleep Consolidation").



MIT Licensed — experiment freely, share improvements.

Feedback and PRs welcome! This is an ongoing personal research project.
