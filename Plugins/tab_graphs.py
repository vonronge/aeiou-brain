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

        # Settings
        self.smoothing = tk.IntVar(value=20)
        self.show_trend = tk.BooleanVar(value=True)
        self.show_power_law = tk.BooleanVar(value=True)
        self.auto_refresh = tk.BooleanVar(value=True)

        self._setup_ui()
        self.parent.after(2000, self._animate)

    def _setup_ui(self):
        # Top Bar
        bar = ttk.Frame(self.parent, padding=5)
        bar.pack(fill="x")

        ttk.Label(bar, text="Smoothing:").pack(side="left")
        scale = ttk.Scale(bar, from_=1, to=200, variable=self.smoothing, orient="horizontal", length=200,
                          command=lambda v: self._update_graphs())
        scale.pack(side="left", padx=5)

        ttk.Checkbutton(bar, text="Auto-Refresh", variable=self.auto_refresh).pack(side="left", padx=10)
        ttk.Checkbutton(bar, text="Power Law Projection", variable=self.show_power_law,
                        command=self._update_graphs).pack(side="left", padx=10)

        ttk.Button(bar, text="Force Refresh", command=self._update_graphs).pack(side="right")

        # Main Graph (Step Loss - Reconstruction + Game)
        self.canv_main = tk.Canvas(self.parent, bg="#1E1E1E", height=300, highlightthickness=0)
        self.canv_main.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        # Aux Graph (Epoch Reconstruction Avg + Power Law)
        self.canv_aux = tk.Canvas(self.parent, bg="#252526", height=200, highlightthickness=0)
        self.canv_aux.pack(fill="x", padx=10, pady=(0, 10))

    def _animate(self):
        if self.auto_refresh.get():
            try:
                current_tab_id = self.app.notebook.select()
                if current_tab_id:
                    current_tab_text = self.app.notebook.tab(current_tab_id, "text")
                    if current_tab_text == self.name:
                        self._update_graphs()
            except:
                pass
        self.parent.after(2000, self._animate)

    def _get_coords(self, data, width, height, min_val, max_val, offset_x=40):
        if len(data) == 0: return []
        count = len(data)
        if count < 2: return []

        x_step = (width - offset_x - 10) / (count - 1)
        y_range = max_val - min_val if max_val > min_val else 1.0
        # Drawing area height: total height - 40 (20 top padding, 20 bottom)
        draw_h = height - 40

        coords = []
        for i, val in enumerate(data):
            x = offset_x + i * x_step
            # Normalize Y (0 at bottom)
            norm_y = (val - min_val) / y_range
            # Invert for Canvas (0 at top)
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
        # Pad start to keep length consistent
        pad = len(data_arr) - len(smoothed)
        return np.pad(smoothed, (pad, 0), mode='edge').tolist()

    def _fit_power_law(self, epochs, losses):
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

    def _draw_axes_and_grid(self, canvas, width, height, min_val, max_val, h_lines=6, offset_x=40):
        # Clear previous text/lines
        canvas.delete("grid")

        y_range = max_val - min_val if max_val > min_val else 1.0
        draw_h = height - 40

        # Horizontal Grid + Labels
        for i in range(h_lines + 1):
            ratio = i / h_lines
            # Calculate value for this line
            val = min_val + (ratio * y_range)
            # Calculate Y position
            y = height - 20 - (ratio * draw_h)

            # Line
            canvas.create_line(offset_x, y, width, y, fill="#333333", dash=(2, 2), tags="grid")

            # Text Label
            canvas.create_text(offset_x - 5, y, text=f"{val:.2f}", fill="#888", anchor="e", font=("Consolas", 8),
                               tags="grid")

        # Vertical Line (Axis)
        canvas.create_line(offset_x, 20, offset_x, height - 20, fill="#666", width=1, tags="grid")
        # Horizontal Line (Axis)
        canvas.create_line(offset_x, height - 20, width, height - 20, fill="#666", width=1, tags="grid")

    def _update_graphs(self):
        # Collect Data
        raw_recon = []
        raw_game = []
        epoch_recon_avgs = []
        epoch_indices = []

        sorted_epochs = sorted(self.app.graph_data.keys())
        for ep in sorted_epochs:
            ep_data = self.app.graph_data[ep]
            r_data = ep_data.get('text', [])  # Smoothed/Scaled steps
            g_data = ep_data.get('vis', [])  # Smoothed/Scaled steps

            raw_recon.extend(r_data)
            raw_game.extend(g_data)

            # Epoch reconstruction average (Using RAW text loss, not scaled)
            recon_totals = ep_data.get('raw_text', [])
            if recon_totals:
                avg = sum(recon_totals) / len(recon_totals)
                epoch_recon_avgs.append(avg)
                epoch_indices.append(ep)

        if not raw_recon: return

        # === MAIN GRAPH (Step Loss) ===
        w = self.canv_main.winfo_width()
        h = self.canv_main.winfo_height()
        if w > 10 and h > 10:
            self.canv_main.delete("all")

            smooth = self.smoothing.get()
            plot_recon = self._moving_average(raw_recon, smooth)
            plot_game = self._moving_average(raw_game, smooth)

            all_vals = np.array(plot_recon + plot_game)
            min_y = max(0, np.min(all_vals) * 0.9)
            max_y = np.max(all_vals) * 1.1

            # Grid & Labels
            self._draw_axes_and_grid(self.canv_main, w, h, min_y, max_y, h_lines=8)

            # Lines
            coords_r = self._get_coords(plot_recon, w, h, min_y, max_y)
            if len(coords_r) > 2:
                self.canv_main.create_line(coords_r, fill="#FF5555", width=2.5, tags="line")

            coords_g = self._get_coords(plot_game, w, h, min_y, max_y)
            if len(coords_g) > 2:
                self.canv_main.create_line(coords_g, fill="#5555FF", width=2, tags="line")

            # Legend
            self.canv_main.create_text(w - 10, 10, text="Reconstruction (Red)\nGame Penalty (Blue)", fill="#AAA",
                                       anchor="ne", font=("Segoe UI", 9, "bold"))

        # === AUX GRAPH (Epoch Reconstruction Only + Power Law) ===
        w_aux = self.canv_aux.winfo_width()
        h_aux = self.canv_aux.winfo_height()
        if w_aux > 10 and h_aux > 10:
            self.canv_aux.delete("all")

            if epoch_recon_avgs:
                # Scale
                min_e = min(epoch_recon_avgs) * 0.9
                max_e = max(epoch_recon_avgs) * 1.1

                # Grid & Labels
                self._draw_axes_and_grid(self.canv_aux, w_aux, h_aux, min_e, max_e, h_lines=6)

                # Power Law
                pl_coords = []
                pl_text = ""
                if self.show_power_law.get() and len(epoch_recon_avgs) > 5:
                    params, predict_fn = self._fit_power_law(epoch_indices, epoch_recon_avgs)
                    if params:
                        a, b = params
                        future_ep = int(epoch_indices[-1] * 2)  # Project 2x current epoch
                        trend_x = np.linspace(epoch_indices[0], future_ep, 200)
                        trend_y = predict_fn(trend_x)

                        # Adjust scale for trend if it goes lower
                        # (We don't change grid here to keep actual data visible, but clamp trend)

                        # Calculate coords manually for trend to allow projection beyond current data width
                        # X Scale needs to account for future_ep
                        total_x_range = future_ep - epoch_indices[0]
                        x_step_pl = (w_aux - 50) / total_x_range
                        y_range_pl = max_e - min_e
                        draw_h_pl = h_aux - 40

                        for tx, ty in zip(trend_x, trend_y):
                            cx = 40 + (tx - epoch_indices[0]) * x_step_pl
                            # Clamp Y to view
                            ty_clamped = max(min_e, min(max_e, ty))
                            norm_ty = (ty_clamped - min_e) / y_range_pl
                            cy = h_aux - 20 - (norm_ty * draw_h_pl)
                            pl_coords.append(cx)
                            pl_coords.append(cy)

                        # Text
                        pred_1000 = predict_fn(1000)
                        pred_5000 = predict_fn(5000)
                        pl_text = f"Power Law: L = {a:.3f} × epoch^({b:.3f})\n"
                        pl_text += f"Pred @ 1k: {pred_1000:.3f}\n"
                        pl_text += f"Pred @ 5k: {pred_5000:.3f}"

                # Actual Epoch Avgs (Green)
                # Map X based on max range (either current or future if projection on)
                max_x_domain = future_ep if (self.show_power_law.get() and pl_coords) else epoch_indices[-1]
                x_range_act = max_x_domain - epoch_indices[0]
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
                    self.canv_aux.create_line(pl_coords, fill="#FFD700", width=2, dash=(4, 2))
                    self.canv_aux.create_text(w_aux - 10, 20, text=pl_text, fill="#FFD700", anchor="ne",
                                              font=("Consolas", 9))

                if len(act_coords) > 2:
                    self.canv_aux.create_line(act_coords, fill="#55FF55", width=3)
                    # Dots
                    for i in range(0, len(act_coords), 2):
                        self.canv_aux.create_oval(act_coords[i] - 3, act_coords[i + 1] - 3,
                                                  act_coords[i] + 3, act_coords[i + 1] + 3,
                                                  fill="#55FF55", outline="#000")

                # Labels
                last_avg = epoch_recon_avgs[-1]
                self.canv_aux.create_text(50, 10, text=f"Avg Loss: {last_avg:.4f}",
                                          fill="#55FF55", anchor="nw", font=("Consolas", 11, "bold"))
                self.canv_aux.create_text(50, h_aux - 10, text=f"Ep {epoch_indices[0]}", fill="#888", anchor="sw")
                self.canv_aux.create_text(w_aux - 10, h_aux - 10, text=f"Ep {max_x_domain}", fill="#888", anchor="se")

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'canv_main'): self.canv_main.config(bg="#1E1E1E")
        if hasattr(self, 'canv_aux'): self.canv_aux.config(bg="#252526")