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

# FILE: Plugins/tab_graphs.py
import tkinter as tk
from tkinter import ttk
import math
import numpy as np


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Telemetry"

        # --- VIEW SETTINGS ---
        self.smoothing = tk.IntVar(value=10)
        self.show_power_law = tk.BooleanVar(value=True)
        self.auto_refresh = tk.BooleanVar(value=True)
        self.max_history = tk.IntVar(value=50)  # View window size

        self._setup_ui()

        # Start Animation Loop
        if self.parent:
            self.parent.after(2000, self._animate)

    def _setup_ui(self):
        if self.parent is None: return

        # 1. TOOLBAR
        bar = ttk.Frame(self.parent, padding=5)
        bar.pack(fill="x")

        # History Limit
        ttk.Label(bar, text="View Width:").pack(side="left")
        ttk.Spinbox(bar, from_=10, to=5000, textvariable=self.max_history, width=5,
                    command=self._refresh_if_auto).pack(side="left", padx=5)

        # Smoothing Slider
        ttk.Label(bar, text="Smooth:").pack(side="left", padx=(10, 0))
        scale = ttk.Scale(bar, from_=1, to=100, variable=self.smoothing, orient="horizontal", length=100,
                          command=lambda v: self._refresh_if_auto())
        scale.pack(side="left", padx=5)

        # Toggles
        ttk.Checkbutton(bar, text="Auto-Refresh (2s)", variable=self.auto_refresh).pack(side="left", padx=10)
        ttk.Checkbutton(bar, text="Show Power Law", variable=self.show_power_law,
                        command=self._refresh_if_auto).pack(side="left", padx=5)

        # Manual Refresh
        ttk.Button(bar, text="⟳ REFRESH", command=self._update_graphs).pack(side="right", padx=5)

        # 2. MAIN CANVAS (Step Loss)
        self.canv_main = tk.Canvas(self.parent, bg=self.app.colors["BG_MAIN"], height=350, highlightthickness=0)
        self.canv_main.pack(fill="both", expand=True, padx=10, pady=(5, 5))

        # 3. AUX CANVAS (Epoch Trends)
        self.canv_aux = tk.Canvas(self.parent, bg=self.app.colors["BG_CARD"], height=200, highlightthickness=0)
        self.canv_aux.pack(fill="x", padx=10, pady=(0, 10))

    def _refresh_if_auto(self, *args):
        if self.auto_refresh.get():
            self._update_graphs()

    def _animate(self):
        if self.auto_refresh.get():
            try:
                # Only redraw if visible to save CPU
                if self.parent.winfo_ismapped():
                    self._update_graphs()
            except:
                pass
        self.parent.after(2000, self._animate)

    # --- MATH HELPERS ---
    def _moving_average(self, data, window_size):
        if not data: return []
        data_arr = np.array(data)
        eff_window = min(int(window_size), len(data_arr))
        if eff_window < 2: return data_arr.tolist()

        window = np.ones(eff_window) / float(eff_window)
        smoothed = np.convolve(data_arr, window, 'valid')

        # Pad start to maintain alignment (right-aligned)
        pad = len(data_arr) - len(smoothed)
        return np.pad(smoothed, (pad, 0), mode='edge').tolist()

    def _fit_power_law(self, x_vals, y_vals):
        """Fits L = a * x^b"""
        try:
            x = np.array(x_vals)
            y = np.array(y_vals)

            # Filter valid
            mask = (x > 0) & (y > 0)
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) < 5: return None, None

            # Linear regression on log-log plot
            x_log = np.log(x_clean)
            y_log = np.log(y_clean)

            b, ln_a = np.polyfit(x_log, y_log, 1)
            a = np.exp(ln_a)

            def predict(val):
                return a * (val ** b)

            return (a, b), predict
        except:
            return None, None

    # --- DRAWING HELPERS ---
    def _get_coords(self, data, width, height, min_val, max_val, offset_x=40):
        if len(data) < 2: return []

        count = len(data)
        x_step = (width - offset_x - 10) / (count - 1)
        y_range = max_val - min_val if (max_val - min_val) > 1e-6 else 1.0
        draw_h = height - 40  # Padding

        coords = []
        for i, val in enumerate(data):
            x = offset_x + i * x_step
            # Normalize Y (0 at bottom)
            norm_y = (val - min_val) / y_range
            # Flip for Canvas (0 at top)
            y = height - 20 - (norm_y * draw_h)
            coords.extend([x, y])

        return coords

    def _draw_axes(self, canvas, width, height, min_val, max_val, h_lines=6, offset_x=40):
        canvas.delete("grid")
        y_range = max_val - min_val
        draw_h = height - 40

        font_spec = ("Consolas", int(8 * getattr(self.app, 'ui_scale', 1.0)))

        # Horizontal Grid Lines
        for i in range(h_lines + 1):
            ratio = i / h_lines
            val = min_val + (ratio * y_range)
            y = height - 20 - (ratio * draw_h)

            color = self.app.colors["BORDER"]
            canvas.create_line(offset_x, y, width, y, fill=color, dash=(2, 2), tags="grid")
            canvas.create_text(offset_x - 5, y, text=f"{val:.2f}", fill=self.app.colors["FG_DIM"],
                               anchor="e", font=font_spec, tags="grid")

        # Vertical Axes
        canvas.create_line(offset_x, 20, offset_x, height - 20, fill=self.app.colors["FG_DIM"], width=1, tags="grid")
        canvas.create_line(offset_x, height - 20, width, height - 20, fill=self.app.colors["FG_DIM"], width=1,
                           tags="grid")

    # --- MAIN UPDATE LOOP ---
    def _update_graphs(self):
        # Data is stored in app.graph_data: {epoch_int: {'total': [], 'text': [], ...}}
        if not self.app.graph_data: return

        # 1. Filter Data (Recent History)
        limit = self.max_history.get()
        all_epochs = sorted(self.app.graph_data.keys())
        active_epochs = all_epochs[-limit:]

        raw_recon = []  # Text Loss (Red)
        raw_game = []  # Game/Vis Loss (Blue)

        epoch_avgs = []
        epoch_indices = []

        for ep in active_epochs:
            ep_data = self.app.graph_data[ep]
            # Use raw unscaled metrics if available
            r = ep_data.get('raw_text', ep_data.get('text', []))
            g = ep_data.get('raw_vis', ep_data.get('vis', []))

            raw_recon.extend(r)
            raw_game.extend(g)

            if r:
                avg = sum(r) / len(r)
                epoch_avgs.append(avg)
                epoch_indices.append(ep)

        if not raw_recon: return

        # 2. DRAW MAIN GRAPH (Step-wise)
        w = self.canv_main.winfo_width()
        h = self.canv_main.winfo_height()

        if w > 10 and h > 10:
            self.canv_main.delete("line")
            self.canv_main.delete("text")

            # Smoothing
            smooth = self.smoothing.get()
            plot_recon = self._moving_average(raw_recon, smooth)
            plot_game = self._moving_average(raw_game, smooth)

            # Dynamic Range
            all_vals = np.array(plot_recon + plot_game)
            if len(all_vals) > 0:
                min_y = max(0, np.min(all_vals) * 0.9)
                max_y = np.max(all_vals) * 1.1
            else:
                min_y, max_y = 0, 1

            self._draw_axes(self.canv_main, w, h, min_y, max_y, h_lines=8)

            # Lines
            coords_r = self._get_coords(plot_recon, w, h, min_y, max_y)
            if len(coords_r) > 2:
                self.canv_main.create_line(coords_r, fill=self.app.colors["ERROR"], width=2, tags="line")

            coords_g = self._get_coords(plot_game, w, h, min_y, max_y)
            if len(coords_g) > 2:
                self.canv_main.create_line(coords_g, fill=self.app.colors["ACCENT"], width=2, tags="line")

            # Legend
            font_spec = ("Segoe UI", int(9 * getattr(self.app, 'ui_scale', 1.0)), "bold")
            info = f"Last {len(active_epochs)} Epochs"
            self.canv_main.create_text(w - 10, 10, text=f"Reconstruction (Red)\nGame/Vis (Blue)\n{info}",
                                       fill=self.app.colors["FG_TEXT"], anchor="ne", font=font_spec, tags="text")

        # 3. DRAW AUX GRAPH (Power Law Projection)
        w_aux = self.canv_aux.winfo_width()
        h_aux = self.canv_aux.winfo_height()

        if w_aux > 10 and h_aux > 10:
            self.canv_aux.delete("line")
            self.canv_aux.delete("text")

            if epoch_avgs:
                min_e = min(epoch_avgs) * 0.9
                max_e = max(epoch_avgs) * 1.1

                self._draw_axes(self.canv_aux, w_aux, h_aux, min_e, max_e, h_lines=5)

                # Forecasting
                future_ep = epoch_indices[-1]
                pl_coords = []
                pl_label = ""

                if self.show_power_law.get() and len(epoch_avgs) > 5:
                    params, predict_fn = self._fit_power_law(epoch_indices, epoch_avgs)
                    if params:
                        a, b = params
                        # Forecast 50% into future
                        future_ep = int(epoch_indices[-1] * 1.5)

                        trend_x = np.linspace(epoch_indices[0], future_ep, 50)
                        trend_y = predict_fn(trend_x)

                        # Project coords onto canvas space
                        # X-axis range: [start_epoch, future_epoch]
                        total_x_range = future_ep - epoch_indices[0]
                        if total_x_range == 0: total_x_range = 1

                        x_step_pl = (w_aux - 50) / total_x_range
                        y_range_pl = max_e - min_e
                        if y_range_pl < 1e-6: y_range_pl = 1.0
                        draw_h_pl = h_aux - 40

                        for tx, ty in zip(trend_x, trend_y):
                            cx = 40 + (tx - epoch_indices[0]) * x_step_pl
                            # Clamp Y to view
                            ty_c = max(min_e, min(max_e, ty))
                            cy = h_aux - 20 - ((ty_c - min_e) / y_range_pl * draw_h_pl)
                            pl_coords.extend([cx, cy])

                        pl_label = f"L ≈ {a:.3f}·e^({b:.3f})"

                # Draw Projection Line
                if pl_coords:
                    self.canv_aux.create_line(pl_coords, fill=self.app.colors["WARN"], width=2, dash=(4, 2),
                                              tags="line")
                    self.canv_aux.create_text(w_aux - 10, 20, text=pl_label, fill=self.app.colors["WARN"],
                                              anchor="ne", font=("Consolas", 9), tags="text")

                # Draw Actual Epoch Points (Overlay on top of projection)
                # Recalculate X-scale based on the Future projection (so they align)
                total_x_range = future_ep - epoch_indices[0]
                if total_x_range == 0: total_x_range = 1
                x_step_act = (w_aux - 50) / total_x_range
                y_range_act = max_e - min_e
                draw_h_act = h_aux - 40

                act_coords = []
                for ep, val in zip(epoch_indices, epoch_avgs):
                    cx = 40 + (ep - epoch_indices[0]) * x_step_act
                    cy = h_aux - 20 - ((val - min_e) / y_range_act * draw_h_act)
                    act_coords.extend([cx, cy])

                if len(act_coords) > 2:
                    self.canv_aux.create_line(act_coords, fill=self.app.colors["SUCCESS"], width=2, tags="line")
                    # Dots
                    for i in range(0, len(act_coords), 2):
                        cx, cy = act_coords[i], act_coords[i + 1]
                        self.canv_aux.create_oval(cx - 2, cy - 2, cx + 2, cy + 2,
                                                  fill=self.app.colors["BG_MAIN"], outline=self.app.colors["SUCCESS"],
                                                  tags="line")

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'canv_main'): self.canv_main.config(bg=c["BG_MAIN"])
        if hasattr(self, 'canv_aux'): self.canv_aux.config(bg=c["BG_CARD"])