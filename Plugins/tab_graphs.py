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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import threading

class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Telemetry"
        self.is_active = False
        
        self.style_cfg = {
            'bg': '#0b0f19',
            'fg': '#E3E3E3',
            'accent': '#A8C7FA',
            'line_total': '#81C995',
            'line_text': '#A8C7FA',
            'line_vis': '#FDD663'
        }
        
        self._setup_ui()

    def _setup_ui(self):
        # Master Frame
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Matplotlib Figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8), facecolor=self.style_cfg['bg'])
        self.fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, left=0.1, right=0.95)
        
        self._style_ax(self.ax1, "Total Loss")
        self._style_ax(self.ax2, "Modality Loss")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Animation Loop (Check every 1000ms instead of 200ms to save CPU)
        self.ani = animation.FuncAnimation(self.fig, self._update, interval=1000, cache_frame_data=False)

    def _style_ax(self, ax, title):
        c = self.style_cfg
        ax.set_facecolor(c['bg'])
        ax.set_title(title, color=c['fg'], fontsize=10)
        ax.tick_params(axis='x', colors=c['fg'], labelsize=8)
        ax.tick_params(axis='y', colors=c['fg'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.grid(True, color='#222222', linestyle='--')

    def _update(self, frame):
        # Only update if data exists
        if not self.app.graph_data: return

        # --- OPTIMIZATION: SLICE DATA (Last 50 points only) ---
        limit = 50
        
        epochs = sorted(self.app.graph_data.keys())
        recent_epochs = epochs[-limit:] # Take last N
        
        if not recent_epochs: return

        # Extract data slices
        # We calculate the mean of each epoch's list to get a single scalar per epoch
        def get_means(key):
            res = []
            for e in recent_epochs:
                data_list = self.app.graph_data[e].get(key, [])
                if data_list:
                    res.append(sum(data_list) / len(data_list))
                else:
                    res.append(0)
            return res

        y_total = get_means('total')
        y_text = get_means('text')
        y_vis = get_means('vis')
        
        # Clear and Redraw
        self.ax1.clear()
        self.ax2.clear()
        self._style_ax(self.ax1, f"Total Loss (Last {limit} Epochs)")
        self._style_ax(self.ax2, f"Component Loss (Last {limit} Epochs)")

        c = self.style_cfg
        
        # Plot 1: Total
        self.ax1.plot(recent_epochs, y_total, color=c['line_total'], linewidth=2, label="Total")
        self.ax1.fill_between(recent_epochs, y_total, color=c['line_total'], alpha=0.1)
        
        # Plot 2: Components
        self.ax2.plot(recent_epochs, y_text, color=c['line_text'], linewidth=1.5, label="Text/Recon")
        self.ax2.plot(recent_epochs, y_vis, color=c['line_vis'], linewidth=1.5, label="Vis/Game")
        
        self.ax1.legend(facecolor=c['bg'], labelcolor=c['fg'], fontsize=8)
        self.ax2.legend(facecolor=c['bg'], labelcolor=c['fg'], fontsize=8)

        

    def _update_graphs(self):
       
        self._update(None)
        self.canvas.draw()

    def on_theme_change(self):
        
        pass
