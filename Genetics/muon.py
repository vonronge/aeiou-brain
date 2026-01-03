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

import torch
import torch.optim as optim


def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration with dimensional folding and NaN safety.
    """
    if G is None or torch.isnan(G).any():
        return torch.zeros_like(G)  # Fail gracefully

    assert G.ndim == 2

    # Folding for tall matrices
    if G.size(0) > G.size(1):
        return zeroth_power_via_newtonschulz5(G.T, steps, eps).T

    # Cast to float32 for stability
    orig_dtype = G.dtype
    X = G.float()

    # Normalization
    trace_est = torch.sum(X ** 2)
    if trace_est <= 0 or torch.isnan(trace_est):
        return torch.zeros_like(G)  # Catch empty/nan trace

    X = X / (torch.sqrt(trace_est) + eps)

    # Iteration
    for _ in range(steps):
        A = X @ X.T
        B = 3.0 * torch.eye(A.shape[0], device=X.device, dtype=X.dtype) - A
        X = 0.5 * B @ X

        # Mid-loop safety check
        if torch.isnan(X).any():
            return torch.zeros_like(G)

    return X.to(orig_dtype)


class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.95, nesterov=True, ns_steps=5, adamw_lr=0.0001, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, adamw_lr=adamw_lr,
                        weight_decay=weight_decay)

        muon_params = []
        adam_params = []

        for p in params:
            if p.ndim == 2 and p.size(0) > 32 and p.size(1) > 32:
                muon_params.append(p)
            else:
                adam_params.append(p)

        self.adamw = optim.AdamW(adam_params, lr=adamw_lr, weight_decay=weight_decay, betas=(0.95, 0.95))

        super().__init__(muon_params, defaults)

    def step(self):
        self.adamw.step()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None: continue

                # Input NaN Guard
                if torch.isnan(p.grad).any():
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.grad)

                buf = state['momentum_buffer']
                g = p.grad

                buf.mul_(mu).add_(g)

                if group['nesterov']:
                    update_m = g.add(buf, alpha=mu)
                else:
                    update_m = buf

                if p.ndim == 2:
                    O = zeroth_power_via_newtonschulz5(update_m, steps=group['ns_steps'])

                    # If O came back all zeros (failed), skip update
                    if (O == 0).all():
                        continue

                    scale_factor = max(p.size(0), p.size(1)) ** 0.5
                    scaled_update = O * scale_factor * 0.2

                    if wd > 0:
                        scaled_update.add_(p.data, alpha=wd)

                    p.data.add_(scaled_update, alpha=-lr)