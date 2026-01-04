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

import argparse
import os
import sys
import threading
import time
import torch
import traceback

# --- ORGANELLIZATION IMPORTS ---
try:
    from Organelles.golgi import Organelle_Golgi
    from Organelles.phagus import Organelle_Phagus
    from Organelles.ribosome import Organelle_Ribosome
    from Organelles.lobe_manager import Organelle_LobeManager
    from Organelles.cytoplasm import Organelle_Cytoplasm, TrainConfig
    from Organelles.cytosis import Organelle_Cytosis, DreamConfig
    from Organelles.symbiont import Organelle_Symbiont, SymbiosisConfig
except ImportError as e:
    print(f"CRITICAL ERROR: Organelle failure: {e}")
    sys.exit(1)


class HeadlessApp:
    """
    A lightweight container that mimics the BrainApp GUI class structure
    so that organelles can be initialized identically.
    """

    def __init__(self, lobe_id: int, data_path_override: str = None):
        # 1. Initialize Golgi (Logging)
        self.golgi = Organelle_Golgi()

        # Attach a CLI sink to Golgi so we see output
        self.golgi.attach_sink("cli", self._cli_logger)
        self.golgi.info("Telepathy Link Established.", source="Telepathy")

        # 2. Initialize Phagus (Config & Environment)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.phagus = Organelle_Phagus(root_dir)

        # Apply override if provided (runtime only, no save)
        if data_path_override:
            if not os.path.exists(data_path_override):
                self.golgi.warn(f"Override path not found: {data_path_override}", source="Telepathy")
            else:
                self.phagus.state.data_dir = data_path_override

        self.paths = self.phagus.get_paths()

        # 3. Hardware
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.golgi.info(f"Hardware: {self.device.upper()}", source="Telepathy")

        # 4. Initialize Ribosome (Data Translation)
        self.ribosome = Organelle_Ribosome(self.device, golgi=self.golgi)

        # 5. Initialize Lobe Manager (Model Lifecycle)
        self.lobe_manager = Organelle_LobeManager(
            lobes_dir=self.paths["lobes"],
            genetics_dir=self.paths["genetics"],
            device=self.device,
            ribosome=self.ribosome
        )

        # 6. Initialize Cytoplasm (Training Engine)
        self.cytoplasm = Organelle_Cytoplasm(self.device)

        # 7. Initialize Specialized Agents
        self.cytosis = Organelle_Cytosis(
            device=self.device,
            ribosome=self.ribosome,
            phagus=self.phagus,
            golgi=self.golgi
        )

        self.symbiont = Organelle_Symbiont(
            device=self.device,
            ribosome=self.ribosome,
            golgi=self.golgi,
            memories_path=self.paths["memories"]
        )

        # Load the requested lobe immediately
        try:
            self.lobe_handle = self.lobe_manager.load_lobe(lobe_id)
        except Exception as e:
            self.golgi.error(f"Failed to load Lobe {lobe_id}: {e}", source="Telepathy")
            sys.exit(1)

    def _cli_logger(self, record):
        """Simple stdout formatting for CLI users."""
        color = ""
        reset = ""
        # Basic ANSI colors if on Linux/Mac, or Windows Terminal
        if os.name != 'nt':
            reset = "\033[0m"
            if record.level == "ERROR":
                color = "\033[91m"  # Red
            elif record.level == "WARN":
                color = "\033[93m"  # Yellow
            elif record.level == "SUCCESS":
                color = "\033[92m"  # Green
            elif record.level == "SAVE":
                color = "\033[96m"  # Cyan

        print(f"{color}[{record.timestamp}] [{record.level}] {record.message}{reset}")


# --- WORKER FUNCTIONS ---

def run_training(app, args):
    """Executes a training run using Cytoplasm."""
    app.golgi.info(f"Scanning training data in: {app.paths['data']}...", source="Telepathy")

    # 1. Scan for files (Simple recursive scan for CLI)
    files = []
    valid_exts = {'.png', '.jpg', '.txt', '.wav', '.mp3', '.json'}
    for root, _, fs in os.walk(app.paths['data']):
        for f in fs:
            if os.path.splitext(f)[1].lower() in valid_exts:
                files.append(os.path.join(root, f))

    if not files:
        app.golgi.error("No training data found.", source="Telepathy")
        return

    # Sort for narrative consistency
    files.sort()

    # 2. Build Generator
    def batch_generator():
        # Simple single-item batcher for headless safety
        # In a real scenario, you'd replicate the detailed logic from tab_trainer or membrane
        for path in files:
            # Detect type based on ext
            ext = os.path.splitext(path)[1].lower()
            packet = {}
            if ext in ['.png', '.jpg']:
                packet['v'] = path
            elif ext in ['.txt', '.md']:
                packet['t'] = path
            elif ext in ['.wav', '.mp3']:
                packet['a'] = path
            elif ext == '.json':
                packet['c'] = path  # Physics/Control

            # Use Membrane to ingest
            try:
                # Membrane returns (v_feat, a_feat, full_seq, c_emb, meta)
                data = app.ribosome.ingest_packet(packet)
                v, a, t, c, _ = data

                if t is None or t.size(1) < 2: continue

                # AR targets: Shift right
                inputs = t[:, :-1]
                targets = t[:, 1:]

                yield (v, a, inputs, c, targets)
            except:
                continue

    # 3. Setup Config
    config = TrainConfig(
        epochs=args.epochs,
        autosave_interval=100,
        nursery_active=True
    )

    # 4. Attach Callbacks
    def on_step(step, loss):
        if step % 10 == 0:
            app.golgi.info(f"Step {step} | Loss: {loss['total']:.4f}", source="Trainer")

    def on_autosave(step):
        app.lobe_manager.save_lobe(args.lobe)
        app.golgi.save(f"Auto-saved Lobe {args.lobe} at step {step}", source="Trainer")

    app.cytoplasm.register_callback("step", on_step)
    app.cytoplasm.register_callback("autosave", on_autosave)

    # 5. Run
    app.cytoplasm.train(config, app.lobe_handle, batch_generator(), mode="ar")


def run_dream(app, args):
    """Executes a creative generation loop using Cytosis."""
    config = DreamConfig(
        max_length=512,
        refresh_rate=1.0,
        autosave=True
    )

    def on_sample(text):
        print(f"\n--- DREAM SAMPLE ---\n{text}\n--------------------\n")

    app.cytosis.register_callback("sample", on_sample)

    app.golgi.info(f"Starting Dream State. Seed: '{args.prompt}'", source="Telepathy")
    app.cytosis.start_dream(app.lobe_handle, args.prompt, config)

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        app.cytosis.stop()


def run_symbiosis(app, args):
    """Runs the teacher-student distillation loop."""
    # We need a second lobe for this
    student_id = args.lobe
    teacher_id = args.teacher if args.teacher else (1 if student_id != 1 else 2)

    app.golgi.info(f"Loading Teacher Lobe {teacher_id}...", source="Telepathy")
    try:
        teacher_handle = app.lobe_manager.load_lobe(teacher_id)
    except Exception as e:
        app.golgi.error(f"Could not load teacher: {e}", source="Telepathy")
        return

    config = SymbiosisConfig(
        save_interval=50,
        harvest_enabled=True
    )

    app.symbiont.link(teacher_handle, app.lobe_handle, config)
    app.symbiont.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        app.symbiont.stop()


# --- ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEIOU Brain Telepathy Interface (v23.2)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (Default for this script)")
    parser.add_argument("--mode", choices=["train", "dream", "symbiosis"], required=True, help="Operation mode")
    parser.add_argument("--lobe", type=int, default=1, help="Lobe ID to load (Target/Student)")
    parser.add_argument("--teacher", type=int, help="Teacher Lobe ID (Symbiosis only)")
    parser.add_argument("--data", type=str, help="Override training data path")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--prompt", type=str, default="The nature of reality is", help="Seed prompt for Dream")

    args = parser.parse_args()

    print("--- AEIOU TELEPATHY INITIALIZING ---")
    app = HeadlessApp(args.lobe, args.data)

    try:
        if args.mode == "train":
            run_training(app, args)
        elif args.mode == "dream":
            run_dream(app, args)
        elif args.mode == "symbiosis":
            run_symbiosis(app, args)

    except KeyboardInterrupt:
        print("\n[Telepathy] Interrupted by user. Shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Telepathy] CRASH: {e}")
        traceback.print_exc()
        sys.exit(1)