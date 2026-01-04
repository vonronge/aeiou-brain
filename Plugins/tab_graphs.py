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

        # --- Settings ---
        self.smoothing = tk.IntVar(value=10)
        self.show_power_law = tk.BooleanVar(value=True)
        self.auto_refresh = tk.BooleanVar(value=True)

        # LAG FIX: Only show this many recent epochs
        self.max_history = tk.IntVar(value=50)

        self._setup_ui()
        self.parent.after(2000, self._animate)

    def _setup_ui(self):
        # Top Control Bar
        bar = ttk.Frame(self.parent, padding=5)
        bar.pack(fill="x")

        # 1. View Window (The Purge)
        ttk.Label(bar, text="View Limit:").pack(side="left")
        ttk.Spinbox(bar, from_=10, to=5000, textvariable=self.max_history, width=5, command=self._refresh_if_auto).pack(
            side="left", padx=5)

        # 2. Smoothing
        ttk.Label(bar, text="Smooth:").pack(side="left", padx=(10, 0))
        scale = ttk.Scale(bar, from_=1, to=100, variable=self.smoothing, orient="horizontal", length=100,
                          command=lambda v: self._refresh_if_auto())
        scale.pack(side="left", padx=5)

        # 3. Toggles
        ttk.Checkbutton(bar, text="Auto (2s)", variable=self.auto_refresh).pack(side="left", padx=10)
        ttk.Checkbutton(bar, text="Power Law", variable=self.show_power_law, command=self._refresh_if_auto).pack(
            side="left", padx=5)

        # 4. Manual Trigger
        btn_force = ttk.Button(bar, text="⟳ REFRESH", command=self._update_graphs)
        btn_force.pack(side="right", padx=5)

        # Main Graph (Step Loss)
        self.canv_main = tk.Canvas(self.parent, bg=self.app.colors["BG_MAIN"], height=350, highlightthickness=0)
        self.canv_main.pack(fill="both", expand=True, padx=10, pady=(5, 5))

        # Aux Graph (Epoch Reconstruction Avg)
        self.canv_aux = tk.Canvas(self.parent, bg=self.app.colors["BG_CARD"], height=200, highlightthickness=0)
        self.canv_aux.pack(fill="x", padx=10, pady=(0, 10))

    def _refresh_if_auto(self, *args):
        if self.auto_refresh.get():
            self._update_graphs()

    def _animate(self):
        if self.auto_refresh.get():
            try:
                if self.parent.winfo_ismapped():
                    self._update_graphs()
            except:
                pass
        self.parent.after(2000, self._animate)

    def _get_coords(self, data, width, height, min_val, max_val, offset_x=40):
        if len(data) == 0: return []
        count = len(data)
        if count < 2: return []

        x_step = (width - offset_x - 10) / (count - 1)
        # Avoid division by zero
        y_range = max_val - min_val if (max_val - min_val) > 1e-6 else 1.0
        draw_h = height - 40

        coords = []
        for i, val in enumerate(data):
            x = offset_x + i * x_step
            norm_y = (val - min_val) / y_range
            y = height - 20 - (norm_y * draw_h)
            coords.append(x)
            coords.append(y)
        return coords

    def _moving_average(self, data, window_size):
        if not data: return []
        data_arr = np.array(data)
        effective_window = min(int(window_size), len(data_arr))
        if effective_window < 2: return data_arr.tolist()
        window = np.ones(effective_window) / float(effective_window)
        smoothed = np.convolve(data_arr, window, 'valid')
        # Pad start to align right
        pad = len(data_arr) - len(smoothed)
        return np.pad(smoothed, (pad, 0), mode='edge').tolist()

    def _fit_power_law(self, epochs, losses):
        """Calculates L = a * epoch^b"""
        try:
            x = np.array(epochs)
            y = np.array(losses)
            mask = (x > 0) & (y > 0)
            if np.sum(mask) < 5: return None, None

            x_log = np.log(x[mask])
            y_log = np.log(y[mask])

            b, ln_a = np.polyfit(x_log, y_log, 1)
            a = np.exp(ln_a)

            def predict(ep):
                return a * (ep ** b)

            return (a, b), predict
        except:
            return None, None

    def _draw_axes(self, canvas, width, height, min_val, max_val, h_lines=6, offset_x=40):
        canvas.delete("grid")
        y_range = max_val - min_val if (max_val - min_val) > 1e-6 else 1.0
        draw_h = height - 40

        # Horizontal Grid
        for i in range(h_lines + 1):
            ratio = i / h_lines
            val = min_val + (ratio * y_range)
            y = height - 20 - (ratio * draw_h)

            color = self.app.colors["BORDER"]
            canvas.create_line(offset_x, y, width, y, fill=color, dash=(2, 2), tags="grid")
            canvas.create_text(offset_x - 5, y, text=f"{val:.2f}", fill=self.app.colors["FG_DIM"], anchor="e",
                               font=("Consolas", int(8 * getattr(self.app, 'ui_scale', 1.0))), tags="grid")

        # Axes
        canvas.create_line(offset_x, 20, offset_x, height - 20, fill=self.app.colors["FG_DIM"], width=1, tags="grid")
        canvas.create_line(offset_x, height - 20, width, height - 20, fill=self.app.colors["FG_DIM"], width=1,
                           tags="grid")

    def _update_graphs(self):
        if not self.app.graph_data: return

        # 1. OPTIMIZED DATA COLLECTION
        # Only take the last N epochs
        limit = self.max_history.get()
        all_epochs = sorted(self.app.graph_data.keys())
        active_epochs = all_epochs[-limit:]  # <--- THIS FIXES THE LAG

        raw_recon = []
        raw_game = []
        epoch_recon_avgs = []
        epoch_indices = []

        for ep in active_epochs:
            ep_data = self.app.graph_data[ep]
            # Use raw unscaled data if available
            r_data = ep_data.get('raw_text', ep_data.get('text', []))
            g_data = ep_data.get('raw_vis', ep_data.get('vis', []))

            raw_recon.extend(r_data)
            raw_game.extend(g_data)

            if r_data:
                avg = sum(r_data) / len(r_data)
                epoch_recon_avgs.append(avg)
                epoch_indices.append(ep)

        if not raw_recon: return

        # 2. Draw Main Graph (Step Loss)
        w = self.canv_main.winfo_width()
        h = self.canv_main.winfo_height()

        if w > 10 and h > 10:
            self.canv_main.delete("all")

            smooth = self.smoothing.get()
            plot_recon = self._moving_average(raw_recon, smooth)
            plot_game = self._moving_average(raw_game, smooth)

            # Calculate dynamic range
            all_vals = np.array(plot_recon + plot_game)
            if len(all_vals) > 0:
                min_y = max(0, np.min(all_vals) * 0.9)
                max_y = np.max(all_vals) * 1.1
            else:
                min_y, max_y = 0, 1

            self._draw_axes(self.canv_main, w, h, min_y, max_y, h_lines=8)

            # Draw Lines
            coords_r = self._get_coords(plot_recon, w, h, min_y, max_y)
            if len(coords_r) > 2:
                self.canv_main.create_line(coords_r, fill=self.app.colors["ERROR"], width=2, tags="line")

            coords_g = self._get_coords(plot_game, w, h, min_y, max_y)
            if len(coords_g) > 2:
                self.canv_main.create_line(coords_g, fill=self.app.colors["ACCENT"], width=2, tags="line")

            # Legend
            info = f"Showing Last {len(active_epochs)} Epochs"
            self.canv_main.create_text(w - 10, 10, text=f"Reconstruction (Red)\nGame (Blue)\n{info}",
                                       fill=self.app.colors["FG_TEXT"], anchor="ne", font=("Segoe UI", 9, "bold"))

        # 3. Draw Aux Graph (Epoch Trend + Power Law)
        w_aux = self.canv_aux.winfo_width()
        h_aux = self.canv_aux.winfo_height()

        if w_aux > 10 and h_aux > 10:
            self.canv_aux.delete("all")

            if epoch_recon_avgs:
                min_e = min(epoch_recon_avgs) * 0.9
                max_e = max(epoch_recon_avgs) * 1.1

                self._draw_axes(self.canv_aux, w_aux, h_aux, min_e, max_e, h_lines=6)

                # Power Law Projection
                pl_coords = []
                pl_text = ""
                future_ep = epoch_indices[-1]

                if self.show_power_law.get() and len(epoch_recon_avgs) > 5:
                    # Note: We fit on the VISIBLE window history to capture recent trend
                    params, predict_fn = self._fit_power_law(epoch_indices, epoch_recon_avgs)
                    if params:
                        a, b = params
                        future_ep = int(epoch_indices[-1] * 1.5)  # Forecast 50% further

                        trend_x = np.linspace(epoch_indices[0], future_ep, 50)
                        trend_y = predict_fn(trend_x)

                        # Generate Coords
                        total_x_range = future_ep - epoch_indices[0]
                        if total_x_range == 0: total_x_range = 1
                        x_step_pl = (w_aux - 50) / total_x_range
                        y_range_pl = max_e - min_e
                        draw_h_pl = h_aux - 40

                        for tx, ty in zip(trend_x, trend_y):
                            cx = 40 + (tx - epoch_indices[0]) * x_step_pl
                            # Clamp Y
                            ty_clamped = max(min_e, min(max_e, ty))
                            norm_ty = (ty_clamped - min_e) / y_range_pl
                            cy = h_aux - 20 - (norm_ty * draw_h_pl)
                            pl_coords.append(cx)
                            pl_coords.append(cy)

                        pl_text = f"L ≈ {a:.3f}·e^({b:.3f})"

                # Draw Actual Epoch Points
                max_x = future_ep
                x_range_act = max_x - epoch_indices[0]
                if x_range_act == 0: x_range_act = 1

                x_step_act = (w_aux - 50) / x_range_act
                y_range_act = max_e - min_e
                draw_h_act = h_aux - 40

                act_coords = []
                for ep, val in zip(epoch_indices, epoch_recon_avgs):
                    cx = 40 + (ep - epoch_indices[0]) * x_step_act
                    norm_y = (val - min_e) / y_range_act
                    cy = h_aux - 20 - (norm_y * draw_h_act)
                    act_coords.append(cx)
                    act_coords.append(cy)

                # Render Lines
                if pl_coords:
                    self.canv_aux.create_line(pl_coords, fill=self.app.colors["WARN"], width=2, dash=(4, 2))
                    self.canv_aux.create_text(w_aux - 10, 20, text=pl_text, fill=self.app.colors["WARN"],
                                              anchor="ne", font=("Consolas", int(9 * getattr(self.app, 'ui_scale', 1.0))))

                if len(act_coords) > 2:
                    self.canv_aux.create_line(act_coords, fill=self.app.colors["SUCCESS"], width=2)
                    # Dots
                    for i in range(0, len(act_coords), 2):
                        self.canv_aux.create_oval(act_coords[i] - 2, act_coords[i + 1] - 2,
                                                  act_coords[i] + 2, act_coords[i + 1] + 2,
                                                  fill=self.app.colors["BG_MAIN"], outline=self.app.colors["SUCCESS"])

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'canv_main'): self.canv_main.config(bg=c["BG_MAIN"])
        if hasattr(self, 'canv_aux'): self.canv_aux.config(bg=c["BG_CARD"])