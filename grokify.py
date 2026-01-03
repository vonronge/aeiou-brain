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
import datetime


def grokify(root_dir, output_file="codebase.txt"):
    """
    Scans the codebase and creates a single text file.
    Aggressively filters out large files and binaries.
    """

    # --- CONFIG ---
    MAX_FILE_SIZE_KB = 100  # Strict limit: Skip anything > 100KB (Likely logs/state)

    # Block specific directory names
    SKIP_DIRS = {
        '__pycache__', '.git', '.idea', '.vscode', 'venv', 'env',
        'lobes', 'memories', 'Training_Data', 'logs', 'backups',
        'training_data', 'scans'
    }

    # Block specific filenames (Lower case)
    SKIP_FILES = {
        'trainer_history.json',
        'trainer_state.json',
        'grok_codebase.txt',
        'grokify.py',
        '.ds_store',
        'thumbs.db'
    }

    # Block extensions
    SKIP_EXTENSIONS = {
        '.pt', '.pth', '.bin', '.ckpt', '.safetensors',  # Weights
        '.png', '.jpg', '.jpeg', '.gif', '.bmp',  # Images
        '.mp3', '.wav', '.flac', '.ogg',  # Audio
        '.mp4', '.avi', '.mkv', '.mov', '.webm',  # Video
        '.pyc', '.pyd',  # Compiled
        '.exe', '.dll', '.so', '.o',  # Binary
        '.zip', '.7z', '.rar', '.gz'  # Archives
    }

    print(f"Grokifying {root_dir}...")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Header
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        outfile.write(f"# CODEBASE DUMP - {root_dir}\n")
        outfile.write(f"# Generated: {timestamp}\n")
        outfile.write(f"# Max File Size: {MAX_FILE_SIZE_KB}KB\n")
        outfile.write("=" * 80 + "\n\n")

        for root, dirs, files in os.walk(root_dir):
            # 1. Filter Directories (Modify in-place)
            # Remove hidden dirs or skipped dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]

            for file in files:
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, root_dir)

                # A. Extension Check
                _, ext = os.path.splitext(file)
                if ext.lower() in SKIP_EXTENSIONS:
                    # print(f" . Skipped (Ext): {rel_path}") # Optional noise
                    continue

                # B. Filename Blacklist Check (Case Insensitive)
                if file.lower() in SKIP_FILES:
                    print(f" ! Skipped (Blacklist): {rel_path}")
                    continue

                # C. Size Check (The Safety Net)
                try:
                    size_kb = os.path.getsize(path) / 1024
                    if size_kb > MAX_FILE_SIZE_KB:
                        print(f" ! Skipped (Too Large {size_kb:.1f}KB): {rel_path}")
                        continue
                except:
                    continue

                # D. Write Content
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as infile:
                        content = infile.read()

                        outfile.write(f"--- FILE: {rel_path} ---\n")
                        outfile.write(content)
                        outfile.write("\n\n")

                        print(f" + Added: {rel_path}")
                except Exception as e:
                    print(f" ! Read Error {rel_path}: {e}")

    print(f"\nDone. Output saved to {output_file}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    grokify(current_dir)