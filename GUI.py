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

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import json
import torch
import importlib.util
import traceback
import threading

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- CONFIG ---
BRAIN_DIR = os.path.join(current_dir, "lobes")
GENETICS_DIR = os.path.join(current_dir, "Genetics")
PLUGINS_DIR = os.path.join(current_dir, "Plugins")
ORGANELLES_DIR = os.path.join(current_dir, "Organelles")
MEMORIES_DIR = os.path.join(current_dir, "memories")
DATA_DIR = os.path.join(current_dir, "Training_Data") 
CONFIG_FILE = os.path.join(current_dir, "settings.json")

# Ensure core dirs exist
for d in [BRAIN_DIR, GENETICS_DIR, PLUGINS_DIR, ORGANELLES_DIR, MEMORIES_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# --- FALLBACK DATA DIR CREATION ---
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR)
        print(f"[SYS] Created default Training_Data folder at: {DATA_DIR}")
    except Exception as e:
        print(f"[ERR] Failed to create Data Dir: {e}")

# --- IMPORTS ---
try:
    from Organelles.ribosome import Organelle_Ribosome
except ImportError as e:
    messagebox.showerror("Critical Error", f"Could not import Organelles/ribosome.py\n\nError: {e}")
    sys.exit()

try:
    from Organelles.hippocampus import Organelle_Hippocampus
except ImportError as e:
    print(f"Warning: Hippocampus disabled. ({e})")
    Organelle_Hippocampus = None


class DraggableButton(tk.Button):
    def __init__(self, parent, app, text, command):
        super().__init__(parent, text=text, command=command, 
                         font=("Segoe UI", 11), relief="flat", anchor="w", padx=15, pady=8, cursor="hand2")
        self.parent = parent
        self.app = app
        self.text_val = text
        self.command_func = command
        
        self._drag_data = {"y": 0}
        
        self.bind("<Button-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag_motion)
        self.bind("<ButtonRelease-1>", self._on_drag_stop)

    def _on_drag_start(self, event):
        self._drag_data["y"] = event.y
        if self.command_func: self.command_func()

    def _on_drag_motion(self, event):
        delta = event.y - self._drag_data["y"]
        if abs(delta) < 10: return
        
        x, y = self.winfo_pointerxy()
        target = self.winfo_containing(x, y)
        
        if isinstance(target, DraggableButton) and target != self:
            try:
                my_idx = self.app.sidebar_order.index(self)
                target_idx = self.app.sidebar_order.index(target)
                self.app.sidebar_order[my_idx], self.app.sidebar_order[target_idx] = \
                    self.app.sidebar_order[target_idx], self.app.sidebar_order[my_idx]
                self.app._repack_sidebar()
            except ValueError:
                pass

    def _on_drag_stop(self, event):
        self.app._save_sidebar_order()


class BrainApp(tk.Tk):
    def __init__(self):
        # --- HIGH DPI FIX (Windows) ---
        if os.name == 'nt':
            try:
                import ctypes
                # Tell Windows: "We are DPI Aware, scale us properly"
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except: pass

        super().__init__()
        self.title("AEIOU Brain - The Unified Cortex (v24.7 Scaled UI)")

        # --- WINDOWS DARK MODE BAR ---
        if os.name == 'nt':
            try:
                self.update()
                import ctypes
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    ctypes.windll.user32.GetParent(self.winfo_id()), 20, ctypes.byref(ctypes.c_int(2)), 4
                )
            except: pass

        # --- PATHS ---
        self.paths = {
            "root": current_dir, "lobes": BRAIN_DIR, "genetics": GENETICS_DIR,
            "plugins": PLUGINS_DIR, "memories": MEMORIES_DIR, "data": DATA_DIR,
            "chaos": os.path.join(DATA_DIR, "Chaos_Buffer"), 
            "output": os.path.join(DATA_DIR, "Comics_Output") 
        }

        # --- STATE ---
        self.plugins = {}
        self.plugin_frames = {} 
        self.sidebar_buttons = {} 
        self.sidebar_order = []   
        
        # Device Detection
        if torch.cuda.is_available(): self.device = "cuda"
        elif torch.backends.mps.is_available(): self.device = "mps"
        else: self.device = "cpu"
            
        self.gpu_lock = threading.Lock()

        # Organelles
        self.ribosome = Organelle_Ribosome(self.device)
        self.hippocampus = Organelle_Hippocampus(MEMORIES_DIR, self.device) if Organelle_Hippocampus else None

        # Lobes
        self.lobes = {1: None, 2: None, 3: None, 4: None}
        self.lobe_genomes = {1: "gpt2", 2: "gpt2", 3: "gpt2", 4: "gpt2"}
        self.lobe_types = {1: None, 2: None, 3: None, 4: None}
        self.optimizers = {1: None, 2: None, 3: None, 4: None}
        self.scalers = {1: None, 2: None, 3: None, 4: None}
        self.active_lobe = tk.IntVar(value=1)
        self.graph_data = {}
        self.lobe_btns = {}

        # Colors
        self.colors = {
            "BG_MAIN": "#0b0f19", "BG_CARD": "#131620", "FG_TEXT": "#E3E3E3",
            "FG_DIM": "#8e9198", "ACCENT": "#A8C7FA", "BTN": "#1E222D",
            "BTN_ACT": "#2B3042", "SUCCESS": "#81C995", "ERROR": "#F28B82",
            "WARN": "#FDD663", "BORDER": "#444444", "GRID": "#333333", "SCROLL": "#2B3042"
        }
        
        self._load_config()
        self._ensure_paths()
        
        self.configure(bg=self.colors["BG_MAIN"])

        w, h = 1600, 900
        x = (self.winfo_screenwidth() - w) // 2
        y = 50
        self.geometry(f'{w}x{h}+{int(x)}+{int(y)}')
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- APPLY SCALING & LAYOUT ---
        # Try to auto-detect scaling if on Windows/HighDPI
        try:
            # 1.3 is a safe boost for 1080p/1440p. 
            # If you are on 4K, change this to 2.0 manually if needed.
            self.tk.call('tk', 'scaling', 1.3) 
        except: pass

        self._setup_layout()
        self.apply_theme()
        self._load_plugins()
        self.load_state()

    def _ensure_paths(self):
        for key, path in self.paths.items():
            if key in ["root", "lobes", "genetics", "plugins", "memories"]: continue 
            if not os.path.exists(path):
                try: os.makedirs(path)
                except: pass

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self.colors.update(data.get("colors", {}))
                    
                    custom_data = data.get("data_dir")
                    if custom_data:
                        if os.path.isabs(custom_data): self.paths["data"] = custom_data
                        else: self.paths["data"] = os.path.join(self.paths["root"], custom_data)

                    custom_chaos = data.get("chaos_dir")
                    if custom_chaos:
                         if os.path.isabs(custom_chaos): self.paths["chaos"] = custom_chaos
                         else: self.paths["chaos"] = os.path.join(self.paths["data"], custom_chaos)
                    else:
                        self.paths["chaos"] = os.path.join(self.paths["data"], "Chaos_Buffer")

                    custom_out = data.get("output_dir")
                    if custom_out:
                         if os.path.isabs(custom_out): self.paths["output"] = custom_out
                         else: self.paths["output"] = os.path.join(self.paths["data"], custom_out)
                    else:
                        self.paths["output"] = os.path.join(self.paths["data"], "Comics_Output")
            except: pass

    def _setup_layout(self):
        self.main_split = tk.Frame(self, bg=self.colors["BG_MAIN"])
        self.main_split.pack(fill="both", expand=True)

        self.sidebar_container = tk.Frame(self.main_split, bg=self.colors["BG_CARD"], width=260) # Widened sidebar
        self.sidebar_container.pack(side="left", fill="y")
        self.sidebar_container.pack_propagate(False)

        self.sb_scroll = ttk.Scrollbar(self.sidebar_container, orient="vertical")
        self.sb_scroll.pack(side="right", fill="y")

        self.sb_canvas = tk.Canvas(self.sidebar_container, bg=self.colors["BG_CARD"], 
                                   highlightthickness=0, yscrollcommand=self.sb_scroll.set)
        self.sb_canvas.pack(side="left", fill="both", expand=True)
        self.sb_scroll.config(command=self.sb_canvas.yview)

        self.sidebar_frame = tk.Frame(self.sb_canvas, bg=self.colors["BG_CARD"])
        self.sb_window = self.sb_canvas.create_window((0, 0), window=self.sidebar_frame, anchor="nw")

        self.sidebar_frame.bind("<Configure>", self._on_sb_configure)
        self.sb_canvas.bind("<Configure>", self._on_sb_canvas_configure)
        
        if os.name == 'nt' or sys.platform == 'darwin':
            self.sidebar_frame.bind_all("<MouseWheel>", self._on_mousewheel)
        else:
            self.sidebar_frame.bind_all("<Button-4>", self._on_mousewheel)
            self.sidebar_frame.bind_all("<Button-5>", self._on_mousewheel)

        lbl = tk.Label(self.sidebar_frame, text="NEURAL CORE", bg=self.colors["BG_CARD"], 
                       fg=self.colors["FG_DIM"], font=("Segoe UI", 11, "bold"), pady=15)
        lbl.pack(fill="x")

        self.content_area = tk.Frame(self.main_split, bg=self.colors["BG_MAIN"])
        self.content_area.pack(side="right", fill="both", expand=True)

        self._setup_header()

        self.plugin_container = tk.Frame(self.content_area, bg=self.colors["BG_MAIN"])
        self.plugin_container.pack(fill="both", expand=True)
        self.plugin_container.grid_rowconfigure(0, weight=1)
        self.plugin_container.grid_columnconfigure(0, weight=1)

    def _on_sb_configure(self, event):
        self.sb_canvas.configure(scrollregion=self.sb_canvas.bbox("all"))

    def _on_sb_canvas_configure(self, event):
        self.sb_canvas.itemconfig(self.sb_window, width=event.width)

    def _on_mousewheel(self, event):
        x, y = self.winfo_pointerxy()
        widget = self.winfo_containing(x, y)
        if str(widget).startswith(str(self.sidebar_container)):
            if sys.platform == 'darwin':
                self.sb_canvas.yview_scroll(int(-1 * event.delta), "units")
            elif hasattr(event, 'num') and event.num == 4:
                self.sb_canvas.yview_scroll(-1, "units")
            elif hasattr(event, 'num') and event.num == 5:
                self.sb_canvas.yview_scroll(1, "units")
            else:
                self.sb_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _setup_header(self):
        header = tk.Frame(self.content_area, bg=self.colors["BG_MAIN"], height=60)
        header.pack(fill="x", side="top", pady=(0, 5))
        
        tk.Label(header, text="ACTIVE LOBE:", bg=self.colors["BG_MAIN"], fg=self.colors["ACCENT"], 
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=15)

        for i in range(1, 5):
            btn = ttk.Radiobutton(header, text=f"LOBE {i}", variable=self.active_lobe, value=i, style="Lobe.TButton")
            btn.pack(side="left", padx=5)
            self.lobe_btns[i] = btn

        tk.Label(header, text=f"Running on: {self.device.upper()}", bg=self.colors["BG_MAIN"], 
                 fg=self.colors["FG_DIM"], font=("Segoe UI", 10)).pack(side="right", padx=20)

    def _add_plugin(self, name, plugin_class):
        frame = ttk.Frame(self.plugin_container)
        frame.grid(row=0, column=0, sticky="nsew")
        instance = plugin_class(frame, self)
        display_name = instance.name
        self.plugins[name] = instance
        self.plugin_frames[name] = frame
        btn = DraggableButton(self.sidebar_frame, self, text=display_name, command=lambda: self._show_plugin(name))
        self.sidebar_buttons[name] = btn
        self.sidebar_order.append(btn)
        return instance

    def _show_plugin(self, name):
        frame = self.plugin_frames.get(name)
        if frame: frame.tkraise()
        self.notebook = type('MockNotebook', (), {})()
        self.notebook.select = lambda: name
        self.notebook.tab = lambda x, option: self.plugins[name].name if option == "text" else None
        for n, btn in self.sidebar_buttons.items():
            if n == name:
                btn.config(bg=self.colors["BG_MAIN"], fg=self.colors["ACCENT"], relief="sunken")
            else:
                btn.config(bg=self.colors["BG_CARD"], fg=self.colors["FG_TEXT"], relief="flat")

    def _repack_sidebar(self):
        for btn in self.sidebar_buttons.values(): btn.pack_forget()
        for btn in self.sidebar_order: btn.pack(fill="x", pady=1)

    def _load_plugins(self):
        order = [
            "tab_cortex.py", "tab_playground.py", "tab_memory.py", "tab_memory_agent.py",
            "tab_rlm.py", "tab_trainer.py", "tab_diffusion_trainer.py", "tab_dream.py", 
            "tab_factory.py", "tab_video_factory.py", "tab_comic.py", "tab_symbiosis.py", 
            "tab_council.py", "tab_graphs.py", "tab_settings.py"
        ]
        found_files = [f for f in os.listdir(PLUGINS_DIR) if f.startswith("tab_") and f.endswith(".py")]
        found_files.sort(key=lambda x: order.index(x) if x in order else 999)
        first_plugin = None
        for filename in found_files:
            try:
                module_name = filename[:-3]
                path = os.path.join(PLUGINS_DIR, filename)
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "Plugin"):
                    self._add_plugin(module_name, module.Plugin)
                    if not first_plugin: first_plugin = module_name
                    print(f"[SYS] Loaded Plugin: {module_name}")
            except Exception as e:
                print(f"[ERR] Failed to load {filename}: {e}")
                traceback.print_exc()
        self._restore_sidebar_order()
        self._repack_sidebar()
        if first_plugin: self._show_plugin(first_plugin)

    def _save_sidebar_order(self):
        try:
            data = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f: data = json.load(f)
            ordered_names = []
            btn_to_name = {v: k for k, v in self.sidebar_buttons.items()}
            for btn in self.sidebar_order:
                name = btn_to_name.get(btn)
                if name: ordered_names.append(name)
            data["sidebar_order"] = ordered_names
            with open(CONFIG_FILE, 'w') as f: json.dump(data, f, indent=2)
        except Exception as e: print(f"Sidebar Save Error: {e}")

    def _restore_sidebar_order(self):
        try:
            if not os.path.exists(CONFIG_FILE): return
            with open(CONFIG_FILE, 'r') as f: data = json.load(f)
            saved = data.get("sidebar_order", [])
            if not saved: return
            new_order = []
            for name in saved:
                if name in self.sidebar_buttons:
                    new_order.append(self.sidebar_buttons[name])
            for btn in self.sidebar_order:
                if btn not in new_order: new_order.append(btn)
            self.sidebar_order = new_order
        except: pass

    def _load_single_lobe(self, lobe_id, path, silent=False):
        try:
            data = torch.load(path, map_location=self.device)
            genome_name = data.get("genome", "gpt2") if isinstance(data, dict) else "gpt2"
            model_type = data.get("model_type") if isinstance(data, dict) else None
            if model_type is None:
                model_type = "diffusion" if "diffusion" in genome_name.lower() else "ar"
            self.lobe_types[lobe_id] = model_type
            cortex = self.plugins.get('tab_cortex')
            if cortex and genome_name in cortex.available_genetics:
                module = cortex.available_genetics[genome_name]
            else:
                if not silent: messagebox.showerror("Error", f"Genetics '{genome_name}' not found.")
                return
            state_dict = data["state_dict"] if isinstance(data, dict) else data
            self.lobe_genomes[lobe_id] = genome_name
            brain = module.Model(module.NucleusConfig()).to(self.device)
            brain.load_state_dict(state_dict, strict=False)
            self.lobes[lobe_id] = brain
            if "Muon" in genome_name:
                from Genetics.muon import Muon
                self.optimizers[lobe_id] = Muon(brain.parameters(), lr=0.0005, momentum=0.95)
            else:
                self.optimizers[lobe_id] = torch.optim.AdamW(brain.parameters(), lr=2e-5)
            if self.device == "cuda":
                self.scalers[lobe_id] = torch.cuda.amp.GradScaler()
            else:
                self.scalers[lobe_id] = None
            if hasattr(brain, "tokenizer"): self.ribosome.set_tokenizer(brain.tokenizer)
            self.refresh_header()
            if not silent: print(f"[SYS] Loaded Lobe {lobe_id} ({genome_name} | {model_type})")
        except Exception as e:
            self.lobes[lobe_id] = None
            self.refresh_header()
            if not silent: messagebox.showerror("Load Failed", str(e))

    def save_state(self):
        data = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f: data = json.load(f)
            except: pass
        data["last_active_lobe"] = self.active_lobe.get()
        try:
            with open(CONFIG_FILE, 'w') as f: json.dump(data, f, indent=2)
        except: pass
        self._save_sidebar_order()

    def load_state(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f: data = json.load(f)
                self.active_lobe.set(data.get("last_active_lobe", 1))
            except: pass
        self.refresh_header()

    def refresh_header(self):
        for i in range(1, 5):
            style = "LobeLoaded.TButton" if self.lobes[i] else "Lobe.TButton"
            self.lobe_btns[i].configure(style=style)

    def _on_close(self):
        try:
            self.save_state()
            if self.hippocampus: self.hippocampus.save_memory()
        except: pass
        self.destroy()

    def apply_theme(self):
        style = ttk.Style()
        style.theme_use('clam')
        c = self.colors
        
        # --- INCREASED FONT SIZES FOR READABILITY ---
        style.configure(".", background=c["BG_MAIN"], foreground=c["FG_TEXT"], borderwidth=0)
        style.configure("TLabel", background=c["BG_MAIN"], foreground=c["FG_TEXT"], font=("Segoe UI", 11))
        style.configure("Card.TLabel", background=c["BG_CARD"], foreground=c["FG_TEXT"], font=("Segoe UI", 11))
        
        style.configure("TButton", background=c["BTN"], foreground=c["FG_TEXT"], borderwidth=0, padding=(15, 8), font=("Segoe UI", 11, "bold"))
        style.map("TButton", background=[("active", c["BTN_ACT"]), ("pressed", c["ACCENT"])], foreground=[("pressed", c["BG_MAIN"])])
        
        style.configure("Lobe.TButton", background=c["BG_CARD"], foreground=c["FG_DIM"], font=("Segoe UI", 12, "bold"), bordercolor=c["BORDER"], borderwidth=1)
        style.map("Lobe.TButton", background=[("selected", c["ACCENT"]), ("active", c["BTN_ACT"])], foreground=[("selected", c["BG_MAIN"])])
        style.configure("LobeLoaded.TButton", background=c["BG_CARD"], foreground=c["SUCCESS"], font=("Segoe UI", 12, "bold"), bordercolor=c["SUCCESS"], borderwidth=1)
        style.map("LobeLoaded.TButton", background=[("selected", c["ACCENT"]), ("active", c["BTN_ACT"])], foreground=[("selected", c["BG_MAIN"])])
        
        style.configure("TEntry", fieldbackground=c["BG_CARD"], foreground=c["FG_TEXT"], insertcolor=c["ACCENT"], borderwidth=1, bordercolor=c["BORDER"], padding=5)
        
        style.configure("TLabelframe", background=c["BG_CARD"], borderwidth=1, relief="solid", bordercolor=c["BORDER"])
        style.configure("TLabelframe.Label", background=c["BG_CARD"], foreground=c["ACCENT"], font=("Segoe UI", 11, "bold"))
        style.configure("Card.TFrame", background=c["BG_CARD"])
        
        style.configure("Treeview", background=c["BG_MAIN"], foreground=c["FG_TEXT"], fieldbackground=c["BG_MAIN"], borderwidth=1, bordercolor=c["BORDER"], font=("Consolas", 10))
        style.configure("Treeview.Heading", background=c["BG_CARD"], foreground=c["FG_TEXT"], borderwidth=1, bordercolor=c["BORDER"], padding=10, relief="flat")
        style.map("Treeview.Heading", background=[("active", c["BTN_ACT"])], foreground=[("active", c["ACCENT"])])
        
        style.configure("Vertical.TScrollbar", gripcount=0, background=c["SCROLL"], troughcolor=c["BG_CARD"], bordercolor=c["BG_CARD"], lightcolor=c["BG_CARD"], darkcolor=c["BG_CARD"], arrowsize=0)
        
        style.configure("TCheckbutton", background=c["BG_CARD"], foreground=c["FG_TEXT"], focuscolor=c["BG_CARD"], font=("Segoe UI", 11))
        style.map("TCheckbutton", indicatorcolor=[("selected", c["ACCENT"])], background=[("active", c["BG_CARD"])])
        style.configure("TSpinbox", fieldbackground=c["BG_MAIN"], foreground=c["FG_TEXT"], background=c["BTN"], arrowcolor=c["FG_TEXT"], borderwidth=1, bordercolor=c["BORDER"], font=("Segoe UI", 11))
        style.configure("TCombobox", fieldbackground=c["BG_CARD"], background=c["BTN"], foreground=c["FG_TEXT"], arrowcolor=c["FG_TEXT"], borderwidth=1)
        style.configure("TSeparator", background=c["BORDER"])

        for name, plugin in self.plugins.items():
            if hasattr(plugin, "on_theme_change"):
                try: plugin.on_theme_change()
                except: pass
        
        if hasattr(self, 'sidebar_frame'):
            self.sidebar_frame.config(bg=c["BG_CARD"])
            self.sidebar_container.config(bg=c["BG_CARD"])
            self.sb_canvas.config(bg=c["BG_CARD"])
            for btn in self.sidebar_buttons.values():
                btn.config(bg=c["BG_CARD"], fg=c["FG_TEXT"], activebackground=c["BTN_ACT"])


if __name__ == "__main__":
    if not hasattr(sys, 'frozen'):
        app = BrainApp()
        app.mainloop()
