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

print("System Check:")
print(f"   > PyTorch Version: {torch.__version__}")
print(f"   > CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   > GPU Detected:    {torch.cuda.get_device_name(0)}")
    print(f"   > VRAM Total:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Quick Math Test
    x = torch.rand(5, 3).cuda()
    print(f"   > Test Tensor:     Allocated successfully on GPU.")
else:
    print("   ❌ ERROR: You are running on CPU. The 3080 Ti is sleeping.")