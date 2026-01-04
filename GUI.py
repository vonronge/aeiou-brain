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
import threading
import time
import importlib.util
import traceback
import torch

# --- ORGANELLIZATION IMPORTS ---
try:
    from Organelles.golgi import Organelle_Golgi
    from Organelles.phagus import Organelle_Phagus
    from Organelles.ribosome import Organelle_Ribosome
    from Organelles.lobe_manager import Organelle_LobeManager
    from Organelles.cytoplasm import Organelle_Cytoplasm
    from Organelles.hippocampus import Organelle_Hippocampus
    from Organelles.reticulum import Organelle_Reticulum
    from Organelles.symbiont import Organelle_Symbiont
    from Organelles.cytosis import Organelle_Cytosis
except ImportError as e:
    messagebox.showerror("Organ Failure", f"Critical Organelle Missing: {e}\n\nCheck 'Organelles' folder.")
    sys.exit(1)


# --- UI HELPERS ---
class DraggableButton(tk.Button):
    def __init__(self, parent, app, text, command):
        # Scale padding/font with UI scale
        scale = getattr(app, 'ui_scale', 1.0)
        font_size = int(11 * scale)
        padx = int(15 * scale)
        pady = int(8 * scale)

        super().__init__(parent, text=text, command=command,
                         font=("Segoe UI", font_size), relief="flat", anchor="w",
                         padx=padx, pady=pady, cursor="hand2")
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
        if os.name == 'nt':
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass

        super().__init__()
        self.title("AEIOU Brain - The Unified Cortex (v24.15 Font Fix)")

        if os.name == 'nt':
            try:
                self.update()
                import ctypes
                # Enable dark title bar
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    ctypes.windll.user32.GetParent(self.winfo_id()), 20, ctypes.byref(ctypes.c_int(2)), 4
                )
            except:
                pass

        # 2. INITIALIZE ORGANELLES
        self.golgi = Organelle_Golgi()
        self.golgi.info("Brain Stem Initializing...", source="Cortex")

        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.phagus = Organelle_Phagus(root_dir)

        self.paths = self.phagus.get_paths()
        self.colors = self.phagus.get_theme()
        self.ui_scale = self.phagus.state.ui_scale

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.gpu_lock = threading.Lock()
        self.golgi.info(f"Hardware: {self.device.upper()}", source="Cortex")

        self.ribosome = Organelle_Ribosome(self.device, golgi=self.golgi)

        self.lobe_manager = Organelle_LobeManager(
            lobes_dir=self.paths["lobes"],
            genetics_dir=self.paths["genetics"],
            device=self.device,
            ribosome=self.ribosome
        )

        self.cytoplasm = Organelle_Cytoplasm(self.device)

        try:
            self.hippocampus = Organelle_Hippocampus(self.paths["memories"], self.device, golgi=self.golgi)
        except Exception as e:
            self.golgi.error(f"Hippocampus failure: {e}", source="Cortex")
            self.hippocampus = None

        self.reticulum = Organelle_Reticulum(self.hippocampus, self.golgi)

        self.symbiont = Organelle_Symbiont(
            device=self.device,
            ribosome=self.ribosome,
            golgi=self.golgi,
            memories_path=self.paths["memories"]
        )

        self.cytosis = Organelle_Cytosis(
            device=self.device,
            ribosome=self.ribosome,
            phagus=self.phagus,
            golgi=self.golgi
        )

        # 3. STATE COMPATIBILITY
        self.active_lobe = tk.IntVar(value=self.phagus.state.last_active_lobe)
        self.graph_data = {}
        self.plugins = {}
        self.plugin_frames = {}
        self.sidebar_buttons = {}
        self.sidebar_order = []
        self.lobe_btns = {}

        # 4. WINDOW CONFIG
        self.configure(bg=self.colors["BG_MAIN"])

        if self.phagus.state.window_geometry:
            self.geometry(self.phagus.state.window_geometry)
        else:
            w, h = 1600, 900
            x = (self.winfo_screenwidth() - w) // 2
            y = 50
            self.geometry(f'{w}x{h}+{int(x)}+{int(y)}')

        if self.phagus.state.window_state == 'zoomed':
            try:
                self.state('zoomed')
            except:
                pass

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            self.tk.call('tk', 'scaling', self.ui_scale)
        except:
            pass

        self._setup_layout()
        self.apply_theme()
        self._load_plugins()
        self.refresh_header()

    # --- PROPERTIES ---
    @property
    def lobes(self):
        return {i: (self.lobe_manager.get_lobe(i).model if self.lobe_manager.get_lobe(i) else None) for i in
                range(1, 5)}

    @property
    def optimizers(self):
        return {i: (self.lobe_manager.get_lobe(i).optimizer if self.lobe_manager.get_lobe(i) else None) for i in
                range(1, 5)}

    @property
    def scalers(self):
        return {i: (self.lobe_manager.get_lobe(i).scaler if self.lobe_manager.get_lobe(i) else None) for i in
                range(1, 5)}

    @property
    def lobe_genomes(self):
        return {i: (self.lobe_manager.get_lobe(i).genome if self.lobe_manager.get_lobe(i) else None) for i in
                range(1, 5)}

    @property
    def lobe_types(self):
        return {i: (self.lobe_manager.get_lobe(i).model_type if self.lobe_manager.get_lobe(i) else None) for i in
                range(1, 5)}

    # --- LAYOUT ---
    def _setup_layout(self):
        self.main_split = tk.Frame(self, bg=self.colors["BG_MAIN"])
        self.main_split.pack(fill="both", expand=True)

        # Dynamic Sidebar Width
        sb_width = int(280 * self.ui_scale)

        self.sidebar_container = tk.Frame(self.main_split, bg=self.colors["BG_CARD"], width=sb_width)
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

        self.sidebar_frame.bind("<Configure>",
                                lambda e: self.sb_canvas.configure(scrollregion=self.sb_canvas.bbox("all")))
        self.sb_canvas.bind("<Configure>", lambda e: self.sb_canvas.itemconfig(self.sb_window, width=e.width))

        if os.name == 'nt' or sys.platform == 'darwin':
            self.sidebar_frame.bind_all("<MouseWheel>", self._on_mousewheel)
        else:
            self.sidebar_frame.bind_all("<Button-4>", self._on_mousewheel)
            self.sidebar_frame.bind_all("<Button-5>", self._on_mousewheel)

        lbl = tk.Label(self.sidebar_frame, text="NEURAL CORE", bg=self.colors["BG_CARD"],
                       fg=self.colors["FG_DIM"], font=("Segoe UI", int(11 * self.ui_scale), "bold"),
                       pady=int(15 * self.ui_scale))
        lbl.pack(fill="x")

        self.content_area = tk.Frame(self.main_split, bg=self.colors["BG_MAIN"])
        self.content_area.pack(side="right", fill="both", expand=True)

        self._setup_header()

        self.plugin_container = tk.Frame(self.content_area, bg=self.colors["BG_MAIN"])
        self.plugin_container.pack(fill="both", expand=True)
        self.plugin_container.grid_rowconfigure(0, weight=1)
        self.plugin_container.grid_columnconfigure(0, weight=1)

    def _on_mousewheel(self, event):
        x, y = self.winfo_pointerxy()
        widget = self.winfo_containing(x, y)
        # Only scroll sidebar if mouse is over it
        if str(widget).startswith(str(self.sidebar_container)):
            if sys.platform == 'darwin':
                self.sb_canvas.yview_scroll(int(-1 * event.delta), "units")
            elif hasattr(event, 'num') and event.num == 4:
                self.sb_canvas.yview_scroll(-1, "units")
            elif hasattr(event, 'num') and event.num == 5:
                self.sb_canvas.yview_scroll(1, "units")
            else:
                self.sb_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _setup_header(self):
        h_height = int(60 * self.ui_scale)
        header = tk.Frame(self.content_area, bg=self.colors["BG_MAIN"], height=h_height)
        header.pack(fill="x", side="top", pady=(0, 5))

        tk.Label(header, text="ACTIVE LOBE:", bg=self.colors["BG_MAIN"], fg=self.colors["ACCENT"],
                 font=("Segoe UI", int(12 * self.ui_scale), "bold")).pack(side="left", padx=15)

        for i in range(1, 5):
            btn = ttk.Radiobutton(header, text=f"LOBE {i}", variable=self.active_lobe, value=i, style="Lobe.TButton")
            btn.pack(side="left", padx=5)
            self.lobe_btns[i] = btn

        tk.Label(header, text=f"Running on: {self.device.upper()}", bg=self.colors["BG_MAIN"],
                 fg=self.colors["FG_DIM"], font=("Segoe UI", int(10 * self.ui_scale))).pack(side="right", padx=20)

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
        # Mock Notebook for compatibility with old plugins accessing self.notebook.select()
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
        # Order logic
        order = [
            "tab_cortex.py", "tab_playground.py", "tab_memory.py", "tab_memory_agent.py",
            "tab_rlm.py", "tab_trainer.py", "tab_diffusion_trainer.py", "tab_dream.py",
            "tab_factory.py", "tab_video_factory.py",
            "tab_pdf_repair.py",
            "tab_comic.py", "tab_symbiosis.py",
            "tab_council.py", "tab_graphs.py", "tab_settings.py"
        ]

        plugins_dir = self.paths["plugins"]
        found_files = [f for f in os.listdir(plugins_dir) if f.startswith("tab_") and f.endswith(".py")]
        found_files.sort(key=lambda x: order.index(x) if x in order else 999)

        first_plugin = None
        for filename in found_files:
            try:
                module_name = filename[:-3]
                path = os.path.join(plugins_dir, filename)
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "Plugin"):
                    self._add_plugin(module_name, module.Plugin)
                    if not first_plugin: first_plugin = module_name
                    self.golgi.info(f"Loaded Plugin: {module_name}", source="GUI")
            except Exception as e:
                self.golgi.error(f"Failed to load {filename}: {e}", source="GUI")
                traceback.print_exc()

        self._restore_sidebar_order()
        self._repack_sidebar()
        if first_plugin: self._show_plugin(first_plugin)

    def _save_sidebar_order(self):
        try:
            ordered_names = []
            btn_to_name = {v: k for k, v in self.sidebar_buttons.items()}
            for btn in self.sidebar_order:
                name = btn_to_name.get(btn)
                if name: ordered_names.append(name)
            self.phagus.state.sidebar_order = ordered_names
            self.phagus.save()
        except Exception as e:
            print(f"Sidebar Save Error: {e}")

    def _restore_sidebar_order(self):
        try:
            saved_order = self.phagus.state.sidebar_order
            if not saved_order: return
            new_order_btns = []
            for name in saved_order:
                if name in self.sidebar_buttons:
                    new_order_btns.append(self.sidebar_buttons[name])

            # Append any new plugins not in saved order
            for name, btn in self.sidebar_buttons.items():
                if btn not in new_order_btns:
                    new_order_btns.append(btn)

            self.sidebar_order = new_order_btns
        except Exception as e:
            print(f"Sidebar Restore Error: {e}")

    def refresh_header(self):
        for i in range(1, 5):
            handle = self.lobe_manager.get_lobe(i)
            style = "LobeLoaded.TButton" if handle else "Lobe.TButton"
            self.lobe_btns[i].configure(style=style)

    def _on_close(self):
        self.golgi.warn("Shutdown Initiated...", source="System")
        if self.cytoplasm: self.cytoplasm.stop()
        if self.symbiont: self.symbiont.stop()
        if self.cytosis: self.cytosis.stop()
        for name, plugin in self.plugins.items():
            if hasattr(plugin, "stop_requested"): plugin.stop_requested = True
            if hasattr(plugin, "is_running"): plugin.is_running = False

        self.save_state()
        if self.hippocampus: self.hippocampus.save_memory()

        self.destroy()

        deadline = time.time() + 10
        while time.time() < deadline:
            active = [t for t in threading.enumerate() if t is not threading.current_thread() and not t.daemon]
            if not active: break
            time.sleep(1)
        sys.exit(0)

    def apply_theme(self):
        style = ttk.Style()
        style.theme_use('clam')
        c = self.colors

        # Scale Factors
        scale = self.ui_scale
        base_font_size = int(11 * scale)
        small_font_size = int(10 * scale)
        header_font_size = int(12 * scale)
        arrow_size = int(14 * scale)

        # --- GLOBAL FONT OVERRIDE ---
        self.option_add("*font", ("Segoe UI", base_font_size))

        # --- DROPDOWN LISTBOX FONT SCALING ---
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", base_font_size))
        self.option_add("*TCombobox*Listbox.background", c["BG_CARD"])
        self.option_add("*TCombobox*Listbox.foreground", c["FG_TEXT"])
        self.option_add("*TCombobox*Listbox.selectBackground", c["ACCENT"])
        self.option_add("*TCombobox*Listbox.selectForeground", c["BG_MAIN"])

        style.configure(".", background=c["BG_MAIN"], foreground=c["FG_TEXT"], borderwidth=0)
        style.configure("TLabel", background=c["BG_MAIN"], foreground=c["FG_TEXT"], font=("Segoe UI", base_font_size))
        style.configure("Card.TLabel", background=c["BG_CARD"], foreground=c["FG_TEXT"],
                        font=("Segoe UI", base_font_size))

        style.configure("TButton", background=c["BTN"], foreground=c["FG_TEXT"], borderwidth=0,
                        padding=(int(15 * scale), int(8 * scale)), font=("Segoe UI", base_font_size, "bold"))
        style.map("TButton", background=[("active", c["BTN_ACT"]), ("pressed", c["ACCENT"])],
                  foreground=[("pressed", c["BG_MAIN"])])

        style.configure("Lobe.TButton", background=c["BG_CARD"], foreground=c["FG_DIM"],
                        font=("Segoe UI", header_font_size, "bold"), bordercolor=c["BORDER"], borderwidth=1)
        style.map("Lobe.TButton", background=[("selected", c["ACCENT"]), ("active", c["BTN_ACT"])],
                  foreground=[("selected", c["BG_MAIN"])])
        style.configure("LobeLoaded.TButton", background=c["BG_CARD"], foreground=c["SUCCESS"],
                        font=("Segoe UI", header_font_size, "bold"), bordercolor=c["SUCCESS"], borderwidth=1)
        style.map("LobeLoaded.TButton", background=[("selected", c["ACCENT"]), ("active", c["BTN_ACT"])],
                  foreground=[("selected", c["BG_MAIN"])])

        style.configure("TEntry", fieldbackground=c["BG_CARD"], foreground=c["FG_TEXT"], insertcolor=c["ACCENT"],
                        borderwidth=1, bordercolor=c["BORDER"], padding=5, font=("Segoe UI", base_font_size))

        style.configure("TLabelframe", background=c["BG_CARD"], borderwidth=1, relief="solid", bordercolor=c["BORDER"])
        style.configure("TLabelframe.Label", background=c["BG_CARD"], foreground=c["ACCENT"],
                        font=("Segoe UI", base_font_size, "bold"))
        style.configure("Card.TFrame", background=c["BG_CARD"])

        style.configure("Treeview", background=c["BG_MAIN"], foreground=c["FG_TEXT"], fieldbackground=c["BG_MAIN"],
                        borderwidth=1, bordercolor=c["BORDER"], font=("Consolas", small_font_size))
        style.configure("Treeview.Heading", background=c["BG_CARD"], foreground=c["FG_TEXT"], borderwidth=1,
                        bordercolor=c["BORDER"], padding=10, relief="flat")
        style.map("Treeview.Heading", background=[("active", c["BTN_ACT"])], foreground=[("active", c["ACCENT"])])

        style.configure("Vertical.TScrollbar", gripcount=0, background=c["SCROLL"], troughcolor=c["BG_CARD"],
                        bordercolor=c["BG_CARD"], lightcolor=c["BG_CARD"], darkcolor=c["BG_CARD"], arrowsize=0)

        # --- FIXED CHECKBUTTON SCALING ---
        # Explicitly set indicatorsize based on scale
        style.configure("TCheckbutton", background=c["BG_MAIN"], foreground=c["FG_TEXT"], focuscolor=c["BG_MAIN"],
                        font=("Segoe UI", base_font_size), indicatorsize=int(16 * scale))
        style.map("TCheckbutton", indicatorcolor=[("selected", c["ACCENT"])], background=[("active", c["BG_MAIN"])])

        # --- FIXED SPINBOX/COMBOBOX ARROWS & FONT ---
        # Arrowsize must be explicit or it stays tiny
        style.configure("TSpinbox", fieldbackground=c["BG_CARD"], foreground=c["FG_TEXT"], background=c["BTN"],
                        arrowcolor=c["FG_TEXT"], borderwidth=1, bordercolor=c["BORDER"],
                        font=("Segoe UI", base_font_size), arrowsize=arrow_size)

        style.map("TSpinbox", fieldbackground=[("readonly", c["BG_CARD"]), ("disabled", c["BG_MAIN"])],
                  foreground=[("readonly", c["FG_TEXT"]), ("disabled", c["FG_DIM"])])

        style.configure("TCombobox", fieldbackground=c["BG_CARD"], background=c["BTN"], foreground=c["FG_TEXT"],
                        arrowcolor=c["FG_TEXT"], borderwidth=1,
                        font=("Segoe UI", base_font_size), arrowsize=arrow_size)

        style.map("TCombobox",
                  fieldbackground=[("readonly", c["BG_CARD"]), ("disabled", c["BG_MAIN"])],
                  selectbackground=[("readonly", c["BG_CARD"]), ("!readonly", c["ACCENT"])],
                  selectforeground=[("readonly", c["FG_TEXT"]), ("!readonly", c["BG_MAIN"])],
                  foreground=[("readonly", c["FG_TEXT"]), ("disabled", c["FG_DIM"])])

        style.configure("TSeparator", background=c["BORDER"])

        # Sliders
        style.configure("Horizontal.TScale", background=c["BG_MAIN"], troughcolor=c["BG_CARD"],
                        sliderlength=int(20 * scale), sliderthickness=int(10 * scale), bordercolor=c["BG_MAIN"])

        for name, plugin in self.plugins.items():
            if hasattr(plugin, "on_theme_change"):
                try:
                    plugin.on_theme_change()
                except:
                    pass

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