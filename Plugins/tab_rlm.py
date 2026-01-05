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
import threading
import queue
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk

# Try importing Gymnasium (modern Gym)
try:
    import gymnasium as gym

    HAS_GYM = True
except ImportError:
    try:
        import gym

        HAS_GYM = True
    except ImportError:
        HAS_GYM = False


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "RL Gymnasium"

        self.is_running = False
        self.stop_requested = False
        self.update_queue = queue.Queue()
        self.render_queue = queue.Queue()

        # --- SETTINGS ---
        self.env_name = tk.StringVar(value="CartPole-v1")
        self.episodes = tk.IntVar(value=500)
        self.max_steps = tk.IntVar(value=200)
        self.gamma = tk.DoubleVar(value=0.99)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.render_live = tk.BooleanVar(value=True)

        self._setup_ui()
        if self.parent:
            self.parent.after(100, self._process_gui_queue)

    def _setup_ui(self):
        if self.parent is None: return

        scale = getattr(self.app, 'ui_scale', 1.0)

        # Split: Controls (Left) | Visualization (Right)
        panes = ttk.PanedWindow(self.parent, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=10, pady=10)

        # --- LEFT: CONTROLS ---
        fr_ctrl = ttk.Frame(panes)
        panes.add(fr_ctrl, weight=1)

        # 1. Environment
        fr_env = ttk.LabelFrame(fr_ctrl, text="Simulation Config", padding=10)
        fr_env.pack(fill="x", pady=5)

        ttk.Label(fr_env, text="Environment ID:").pack(anchor="w")
        envs = ["CartPole-v1", "LunarLander-v2", "MountainCar-v0", "Acrobot-v1"]
        ttk.Combobox(fr_env, textvariable=self.env_name, values=envs).pack(fill="x", pady=5)

        # 2. Hyperparameters
        fr_hyp = ttk.LabelFrame(fr_ctrl, text="Hyperparameters", padding=10)
        fr_hyp.pack(fill="x", pady=5)

        def add_param(lbl, var, min_v, max_v, inc):
            r = ttk.Frame(fr_hyp)
            r.pack(fill="x", pady=2)
            ttk.Label(r, text=lbl, width=15).pack(side="left")
            ttk.Spinbox(r, from_=min_v, to=max_v, increment=inc, textvariable=var, width=8).pack(side="right")

        add_param("Episodes:", self.episodes, 10, 10000, 10)
        add_param("Max Steps:", self.max_steps, 50, 1000, 50)
        add_param("Gamma:", self.gamma, 0.8, 0.999, 0.01)
        add_param("LR:", self.learning_rate, 0.0001, 0.1, 0.0001)

        ttk.Checkbutton(fr_hyp, text="Live Render", variable=self.render_live).pack(anchor="w", pady=5)

        # 3. Actions
        self.btn_train = ttk.Button(fr_ctrl, text="START TRAINING", command=self._toggle_train)
        self.btn_train.pack(fill="x", pady=20)

        # Logs
        log_font = ("Consolas", int(9 * scale))
        self.log_box = tk.Text(fr_ctrl, font=log_font, height=10,
                               bg=self.app.colors["BG_MAIN"], fg=self.app.colors["FG_TEXT"])
        self.log_box.pack(fill="both", expand=True)

        # --- RIGHT: VISUALIZATION ---
        fr_vis = ttk.LabelFrame(panes, text="Agent Perception", padding=10)
        panes.add(fr_vis, weight=2)

        self.canvas = tk.Canvas(fr_vis, bg="#000000")
        self.canvas.pack(fill="both", expand=True)

    def _log(self, msg):
        self.update_queue.put(lambda: self._write_log(msg))
        if self.app.golgi:
            # Only log significant events to global system, keep step info local
            if "Episode" in msg or "Error" in msg:
                self.app.golgi.info(msg, source="RLM")

    def _write_log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def _process_gui_queue(self):
        # 1. UI Updates
        while not self.update_queue.empty():
            try:
                fn = self.update_queue.get_nowait()
                fn()
            except:
                break

        # 2. Render Updates (The Game Screen)
        if not self.render_queue.empty():
            try:
                img = self.render_queue.get_nowait()
                self._display_frame(img)
            except:
                pass

        if self.parent: self.parent.after(30, self._process_gui_queue)

    def _display_frame(self, img_array):
        # Resize to fit canvas
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10: return

        try:
            pil_img = Image.fromarray(img_array)
            # Aspect Ratio
            w, h = pil_img.size
            ratio = min(cw / w, ch / h)
            new_w, new_h = int(w * ratio), int(h * ratio)

            resized = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST)
            self.tk_img = ImageTk.PhotoImage(resized)  # Keep ref

            self.canvas.delete("all")
            x = (cw - new_w) // 2
            y = (ch - new_h) // 2
            self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)
        except:
            pass

    def _toggle_train(self):
        if self.is_running:
            self.stop_requested = True
            self.btn_train.config(text="STOPPING...")
        else:
            if not HAS_GYM:
                messagebox.showerror("Error", "Gymnasium library not installed.\npip install gymnasium[box2d]")
                return

            # Validate Lobe
            lid = self.app.active_lobe.get()
            lobe = self.app.lobe_manager.get_lobe(lid)
            if not lobe:
                messagebox.showerror("Error", "No Active Lobe to train.")
                return

            self.is_running = True
            self.stop_requested = False
            self.btn_train.config(text="STOP TRAINING")
            self._log(f"Initializing {self.env_name.get()}...")

            threading.Thread(target=self._worker, args=(lobe,), daemon=True).start()

    def _worker(self, lobe):
        try:
            env_id = self.env_name.get()
            # Initialize Environment
            # 'render_mode' is needed for getting the array
            env = gym.make(env_id, render_mode="rgb_array")

            # Reset Lobe Optimizer for RL
            # We might need a fresh optimizer if the main one is cluttered
            optimizer = torch.optim.Adam(lobe.model.parameters(), lr=self.learning_rate.get())

            gamma = self.gamma.get()
            max_ep = self.episodes.get()
            max_steps = self.max_steps.get()

            self._log(f"Env: {env.observation_space.shape} -> {env.action_space.n} Actions")

            for ep in range(max_ep):
                if self.stop_requested: break

                state, _ = env.reset()
                log_probs = []
                rewards = []

                ep_reward = 0

                for t in range(max_steps):
                    if self.stop_requested: break

                    # 1. VISUALIZE
                    if self.render_live.get() and t % 2 == 0:  # Cap framerate slightly
                        frame = env.render()
                        if frame is not None:
                            # Clear queue to prevent lag
                            while not self.render_queue.empty(): self.render_queue.get()
                            self.render_queue.put(frame)

                    # 2. POLICY STEP
                    # Convert state to tensor
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.app.device)

                    # Forward Pass
                    # Note: We assume the model can handle simple vector inputs
                    # If it's a Transformer, we might need to embed this specially.
                    # For compatibility, we try/catch generic forward
                    try:
                        # Assuming model(v, a, t) - pass state as Visual or Text?
                        # Hack: Pass as visual features if shape fits, or project
                        # Ideally, the model has a specific .policy() method.
                        # For now, we assume standard Forward returns logits

                        # Use a dummy forward or specific method if available
                        if hasattr(lobe.model, "policy_forward"):
                            logits = lobe.model.policy_forward(state_t)
                        else:
                            # Fallback: Treat state as 'visual' features (B, 1, D)
                            # Pad state to model dim
                            d_model = 768  # GPT-2 small
                            if state_t.shape[1] < d_model:
                                pad = torch.zeros(1, d_model - state_t.shape[1]).to(self.app.device)
                                inp = torch.cat([state_t, pad], dim=1).unsqueeze(1)
                            else:
                                inp = state_t[:, :d_model].unsqueeze(1)

                            # Forward(v, a, t)
                            # We pass empty audio/text
                            dummy_t = torch.zeros(1, 1).long().to(self.app.device)
                            logits, _, _ = lobe.model(v=inp, a=None, t=dummy_t)

                            # Logits are (B, Seq, Vocab). We need (B, Actions)
                            # Slice the first few vocab tokens to represent actions
                            n_actions = env.action_space.n
                            logits = logits[:, 0, :n_actions]

                        # Softmax -> Distribution
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)

                        action = dist.sample()
                        log_prob = dist.log_prob(action)

                        action_item = action.item()

                        # 3. ENVIRONMENT STEP
                        next_state, reward, done, trunc, _ = env.step(action_item)

                        log_probs.append(log_prob)
                        rewards.append(reward)
                        ep_reward += reward

                        state = next_state

                        if done or trunc:
                            break

                    except Exception as e:
                        self._log(f"Model Incompatible: {e}")
                        self.stop_requested = True
                        break

                # 4. UPDATE (REINFORCE)
                if not self.stop_requested:
                    R = 0
                    returns = []
                    for r in rewards[::-1]:
                        R = r + gamma * R
                        returns.insert(0, R)

                    returns = torch.tensor(returns).to(self.app.device)
                    # Normalize
                    if returns.shape[0] > 1:
                        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

                    policy_loss = []
                    for log_prob, R in zip(log_probs, returns):
                        policy_loss.append(-log_prob * R)

                    optimizer.zero_grad()
                    policy_loss = torch.cat(policy_loss).sum()
                    policy_loss.backward()
                    optimizer.step()

                    self._log(f"Episode {ep + 1}: Reward {ep_reward:.1f}")

            env.close()

        except Exception as e:
            self._log(f"RL Crash: {e}")
            import traceback
            traceback.print_exc()

        self.is_running = False
        self.update_queue.put(lambda: self.btn_train.config(text="START TRAINING"))
        self._log("Training Session Ended.")

    def on_theme_change(self):
        c = self.app.colors
        if hasattr(self, 'log_box'): self.log_box.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"])