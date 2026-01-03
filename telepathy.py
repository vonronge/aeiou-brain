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
import torch
import time
import importlib.util

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# --- HEADLESS INFRASTRUCTURE ---
class MockVar:
    """Replaces tk.StringVar/IntVar when running headless"""

    def __init__(self, value=None): self._val = value

    def set(self, value): self._val = value

    def get(self): return self._val


class HeadlessApp:
    """Mocks the BrainApp class for plugins"""

    def __init__(self, lobe_id, data_path):
        self.paths = {
            "root": current_dir,
            "lobes": os.path.join(current_dir, "lobes"),
            "genetics": os.path.join(current_dir, "Genetics"),
            "plugins": os.path.join(current_dir, "Plugins"),
            "memories": os.path.join(current_dir, "memories"),
            "data": data_path,
            # Configured Subdirs
            "chaos": os.path.join(data_path, "Chaos_Buffer"),
            "output": os.path.join(data_path, "Comics_Output")
        }

        # Ensure dirs
        for p in self.paths.values():
            if not os.path.exists(p):
                try:
                    os.makedirs(p)
                except:
                    pass

        # Dummy colors to prevent plugin crashes
        self.colors = {k: "" for k in ["BG_MAIN", "FG_TEXT", "ACCENT", "BG_CARD", "FG_DIM", "SUCCESS", "WARN", "ERROR"]}

        # --- DEVICE DETECTION ---
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.gpu_lock = threading.Lock()

        # Components
        from Organelles.ribosome import Organelle_Ribosome
        self.ribosome = Organelle_Ribosome(self.device)

        # Optional Hippocampus for Symbiosis
        try:
            from Organelles.hippocampus import Organelle_Hippocampus
            self.hippocampus = Organelle_Hippocampus(self.paths["memories"], self.device)
        except:
            self.hippocampus = None

        self.lobes = {1: None, 2: None, 3: None, 4: None}
        self.lobe_genomes = {}
        self.lobe_types = {}
        self.optimizers = {1: None, 2: None, 3: None, 4: None}
        self.scalers = {1: None, 2: None, 3: None, 4: None}

        self.active_lobe = MockVar(lobe_id)
        self.graph_data = {}
        self.plugins = {}  # For cross-plugin calls

        self._load_lobe(lobe_id)

    def _load_lobe(self, lobe_id):
        path = os.path.join(self.paths['lobes'], f"brain_lobe_{lobe_id}.pt")
        if not os.path.exists(path):
            print(f"[Telepathy] Error: Lobe {lobe_id} not found at {path}")
            sys.exit(1)

        print(f"[Telepathy] Loading Lobe {lobe_id} from disk...")
        try:
            data = torch.load(path, map_location=self.device)
            genome = data.get("genome", "gpt2")
            m_type = data.get("model_type", "ar")

            # Dynamic Import Genetics
            gen_path = os.path.join(self.paths['genetics'], f"{genome}.py")
            if not os.path.exists(gen_path):
                # Fallback search
                found = [f for f in os.listdir(self.paths['genetics']) if f.lower() == f"{genome.lower()}.py"]
                if found: gen_path = os.path.join(self.paths['genetics'], found[0])

            spec = importlib.util.spec_from_file_location(genome, gen_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            config = module.NucleusConfig()
            brain = module.Model(config).to(self.device)
            brain.load_state_dict(data["state_dict"], strict=False)

            self.lobes[lobe_id] = brain
            self.lobe_genomes[lobe_id] = genome
            self.lobe_types[lobe_id] = m_type

            if "Muon" in genome:
                from Genetics.muon import Muon
                self.optimizers[lobe_id] = Muon(brain.parameters(), lr=0.0005, momentum=0.95)
            else:
                self.optimizers[lobe_id] = torch.optim.AdamW(brain.parameters(), lr=2e-5)

            # Scaler handling
            if self.device == "cuda":
                self.scalers[lobe_id] = torch.cuda.amp.GradScaler()
            else:
                self.scalers[lobe_id] = None

            if hasattr(brain, "tokenizer"):
                self.ribosome.set_tokenizer(brain.tokenizer)

            print(f"[Telepathy] Lobe {lobe_id} Online ({genome} | {m_type}) on {self.device.upper()}")

        except Exception as e:
            print(f"[Telepathy] Load Failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


# --- ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEIOU Brain Telepathy Interface")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--mode", choices=["train", "diffusion", "dream", "symbiosis"], help="Operation mode")
    parser.add_argument("--lobe", type=int, default=1, help="Lobe ID to load (1-4)")
    parser.add_argument("--data", default=os.path.join(current_dir, "Training_Data"), help="Path to training data")
    parser.add_argument("--epochs", type=int, default=10, help="Target epochs (Train) or Iterations (Dream)")
    parser.add_argument("--prompt", type=str, default="The nature of reality is", help="Seed prompt for Dreaming")

    args = parser.parse_args()

    if not args.headless:
        # GUI Mode
        from GUI import BrainApp

        app = BrainApp()
        app.mainloop()
    else:
        # Headless Mode
        print("--- AEIOU TELEPATHY (HEADLESS) ---")
        if not args.mode:
            print("Error: --mode is required for headless operation.")
            sys.exit(1)

        app = HeadlessApp(args.lobe, args.data)

        try:
            # --- TRAINING MODES ---
            if args.mode == "train":
                from Plugins.tab_trainer import Plugin as Trainer

                print(f"[Telepathy] Initializing Transformer Trainer on {args.data}...")

                trainer = Trainer(None, app)
                trainer.folder_path.set(args.data)
                trainer.target_epochs.set(args.epochs)
                trainer.nursery_autofit.set(True)
                trainer.autosave_enabled.set(True)

                trainer._scan_files()
                if len(trainer.training_queue) > 0:
                    print(f"[Telepathy] Starting training loop for {args.epochs} epochs...")
                    trainer._start_training()
                    while trainer.is_training: time.sleep(1)
                else:
                    print("[Telepathy] Queue empty. Exiting.")

            elif args.mode == "diffusion":
                from Plugins.tab_diffusion_trainer import Plugin as DiffTrainer

                print(f"[Telepathy] Initializing Diffusion Director on {args.data}...")

                trainer = DiffTrainer(None, app)
                trainer.folder_path.set(args.data)
                trainer.target_epochs.set(args.epochs)

                trainer._scan_files()
                if len(trainer.training_queue) > 0:
                    print(f"[Telepathy] Diffusing...")
                    trainer._start_training()
                    while trainer.is_training: time.sleep(1)

            # --- GENERATION MODES ---
            elif args.mode == "dream":
                from Plugins.tab_dream import Plugin as Dreamer

                print(f"[Telepathy] Entering Dream State...")
                print(f"[Prompt] '{args.prompt}'")

                dreamer = Dreamer(None, app)
                dreamer.seed_prompt.set(args.prompt)
                dreamer.autosave.set(True)  # Force save to Chaos Buffer

                # Trick: Using args.epochs as 'iterations' isn't natively supported
                # by tab_dream (it runs forever). We can just start it and let it run
                # until user interrupt.

                dreamer._toggle_dream()

                print("[Telepathy] Dreaming... (Press Ctrl+C to stop)")
                print(f"[Output] Saving to: {app.paths['chaos']}")

                while dreamer.is_dreaming:
                    time.sleep(1)

            elif args.mode == "symbiosis":
                from Plugins.tab_symbiosis import Plugin as Symbiote

                print(f"[Telepathy] Initializing Symbiosis (Autonomous Loop)...")

                sym = Symbiote(None, app)
                sym.autonomous_mode.set(True)
                sym.auto_train.set(True)

                # Symbiosis needs access to a Trainer instance usually,
                # but in headless we rely on its internal calls.
                # Note: Symbiosis plugin might expect 'tab_trainer' in app.plugins.
                # Let's mock it if needed.
                from Plugins.tab_trainer import Plugin as Trainer

                app.plugins['tab_trainer'] = Trainer(None, app)

                sym._toggle_autonomy()

                print("[Telepathy] System is Alive. (Press Ctrl+C to kill)")
                while sym.is_autonomous:
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n[Telepathy] Interrupted by user. Shutting down...")
            sys.exit(0)
        except Exception as e:
            print(f"\n[Telepathy] CRASH: {e}")
            import traceback

            traceback.print_exc()