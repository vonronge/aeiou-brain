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
import yaml
import time
from datetime import datetime


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Memory Graph"

        self.selected_node_id = None

        # UI Scaling
        self.scale = getattr(self.app, 'ui_scale', 1.0)

        self._setup_ui()
        # Delay load to let Hippocampus boot
        if self.parent:
            self.parent.after(500, self._refresh_list)

    def _setup_ui(self):
        if self.parent is None: return

        # 1. TOOLBAR
        fr_tools = ttk.Frame(self.parent)
        fr_tools.pack(fill="x", padx=10, pady=5)

        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._on_search)

        ttk.Label(fr_tools, text="🔍 Search Nodes:").pack(side="left")
        ttk.Entry(fr_tools, textvariable=self.search_var, width=30).pack(side="left", padx=5)

        ttk.Button(fr_tools, text="💾 SAVE CHANGES", command=self._save_changes).pack(side="right")
        ttk.Button(fr_tools, text="🔄 RELOAD", command=self._refresh_list).pack(side="right", padx=5)

        # 2. MAIN SPLIT
        panes = ttk.PanedWindow(self.parent, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=10, pady=5)

        # LEFT: Node List
        fr_list = ttk.Frame(panes)
        panes.add(fr_list, weight=1)

        # Custom Tree Style for Scaling
        style = ttk.Style()
        row_h = int(25 * self.scale)
        style.configure("Memory.Treeview", rowheight=row_h, font=("Segoe UI", int(10 * self.scale)))
        style.configure("Memory.Treeview.Heading", font=("Segoe UI", int(11 * self.scale), "bold"))

        cols = ("Entity", "Type")
        self.tree = ttk.Treeview(fr_list, columns=cols, show="headings", style="Memory.Treeview")
        self.tree.heading("Entity", text="Entity Name")
        self.tree.heading("Type", text="Type")
        self.tree.column("Entity", width=int(200 * self.scale))
        self.tree.column("Type", width=int(100 * self.scale))

        sb = ttk.Scrollbar(fr_list, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # RIGHT: Editor
        fr_edit = ttk.LabelFrame(panes, text="Engram Editor (YAML)", padding=10)
        panes.add(fr_edit, weight=2)

        editor_font = ("Consolas", int(11 * self.scale))
        self.txt_editor = tk.Text(fr_edit, font=editor_font, bg=self.app.colors["BG_MAIN"],
                                  fg=self.app.colors["FG_TEXT"], insertbackground=self.app.colors["ACCENT"])
        self.txt_editor.pack(side="left", fill="both", expand=True)

        sb2 = ttk.Scrollbar(fr_edit, orient="vertical", command=self.txt_editor.yview)
        self.txt_editor.configure(yscrollcommand=sb2.set)
        sb2.pack(side="right", fill="y")

    def _refresh_list(self):
        # Clear
        for i in self.tree.get_children(): self.tree.delete(i)

        if not self.app.hippocampus: return

        # Load from Organelle
        nodes = self.app.hippocampus.nodes  # Dict {name: NodeObj}

        for name, node in nodes.items():
            n_type = node.type
            self.tree.insert("", "end", iid=name, values=(name, n_type))

    def _on_search(self, *args):
        query = self.search_var.get().lower()
        if not self.app.hippocampus: return

        # Clear tree
        for i in self.tree.get_children(): self.tree.delete(i)

        nodes = self.app.hippocampus.nodes
        for name, node in nodes.items():
            if query in name.lower() or query in node.type.lower():
                self.tree.insert("", "end", iid=name, values=(name, node.type))

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel: return

        entity_name = sel[0]
        self.selected_node_id = entity_name

        if self.app.hippocampus and entity_name in self.app.hippocampus.nodes:
            node = self.app.hippocampus.nodes[entity_name]
            # Dump data to YAML string
            try:
                y_str = yaml.dump(node.data, sort_keys=False, allow_unicode=True)
                self.txt_editor.delete("1.0", tk.END)
                self.txt_editor.insert("1.0", y_str)
            except Exception as e:
                self.txt_editor.delete("1.0", tk.END)
                self.txt_editor.insert("1.0", f"# Error reading node:\n# {e}")

    def _save_changes(self):
        if not self.selected_node_id: return

        raw_yaml = self.txt_editor.get("1.0", tk.END)
        try:
            new_data = yaml.safe_load(raw_yaml)
            if not isinstance(new_data, dict):
                raise ValueError("Must be a dictionary.")

            # Update Organelle
            self.app.hippocampus.update_memory_node(self.selected_node_id, new_data)
            self.app.hippocampus.save_memory()

            self.app.golgi.save(f"Updated memory: {self.selected_node_id}", source="MemoryTab")

            # Refresh Type column if changed
            self.tree.set(self.selected_node_id, "Type", new_data.get("type", "Unknown"))

        except Exception as e:
            messagebox.showerror("YAML Error", f"Invalid YAML Format:\n{e}")

    def on_theme_change(self):
        # Refresh colors
        c = self.app.colors
        if hasattr(self, 'txt_editor'):
            self.txt_editor.config(bg=c["BG_MAIN"], fg=c["FG_TEXT"], insertbackground=c["ACCENT"])