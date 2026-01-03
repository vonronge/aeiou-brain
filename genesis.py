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

import os
import json


def create_file(path, content=""):
    if not os.path.exists(path):
        with open(path, "w", encoding='utf-8') as f:
            f.write(content)
        print(f"   + Created file: {path}")


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"   + Created folder: {path}")


def genesis():
    print("⚡ AEIOU SYSTEM REPAIR PROTOCOL INITIATED...")
    root = os.getcwd()

    # 1. ESSENTIAL DIRECTORIES
    dirs = [
        "lobes",
        "Genetics",
        "Organelles",
        "Plugins",
        "memories",  # <--- The lost folder
        "system/lobegrams",
        "system/genegrams",
        "library/engrams",
        "library/socialgrams"
    ]

    for d in dirs:
        create_folder(os.path.join(root, d))

    # 2. PYTHON PACKAGE MARKERS (Crucial for imports)
    pkgs = ["Genetics", "Organelles", "Plugins", "lobes"]
    for p in pkgs:
        create_file(os.path.join(root, p, "__init__.py"), "")

    # 3. RESTORE CONFIG (Colors)
    config_path = os.path.join(root, "settings.json")
    if not os.path.exists(config_path):
        print("   ! settings.json missing. Restoring default Dark Theme...")
        default_conf = {
            "system_name": "AEIOU_v16",
            "colors": {
                "BG_MAIN": "#0b0f19", "BG_CARD": "#131620",
                "FG_TEXT": "#E3E3E3", "FG_DIM": "#8e9198",
                "ACCENT": "#A8C7FA", "BTN": "#1E222D",
                "BTN_ACT": "#2B3042", "SUCCESS": "#81C995",
                "ERROR": "#F28B82", "WARN": "#FDD663",
                "BORDER": "#444444", "GRID": "#333333",
                "SCROLL": "#2B3042"
            },
            "evolution_settings": {
                "shadow_mode_enabled": False,
                "commando_protocol": True
            }
        }
        create_file(config_path, json.dumps(default_conf, indent=2))

    # 4. MEMORY INDEX PLACEHOLDER
    # Prevents Hippocampus from crashing on first load
    mem_index = os.path.join(root, "memories", "replay_buffer.jsonl")
    create_file(mem_index, "")

    print("==========================================")
    print("✨ SYSTEM REPAIRED. ALL SYSTEMS ONLINE.")
    print("==========================================")


if __name__ == "__main__":
    genesis()