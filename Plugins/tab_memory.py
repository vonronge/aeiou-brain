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
import threading
import torch


class Plugin:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.name = "Memory Graph"

        self.search_var = tk.StringVar()
        self._setup_ui()

    def _setup_ui(self):
        # Split: List vs Details
        pane = ttk.PanedWindow(self.parent, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=10, pady=10)

        # LEFT: List
        left = ttk.Frame(pane, width=300)
        pane.add(left, weight=1)

        # Search Bar
        row_search = ttk.Frame(left)
        row_search.pack(fill="x", pady=5)
        ttk.Entry(row_search, textvariable=self.search_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row_search, text="🔎", width=3, command=self._refresh_list).pack(side="left")

        # Treeview
        self.tree = ttk.Treeview(left, columns=("Type", "Updated"), show="headings")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Updated", text="Updated")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # RIGHT: Editor/Viewer
        right = ttk.LabelFrame(pane, text="Entity Node Viewer")
        pane.add(right, weight=3)

        self.txt_editor = tk.Text(right, font=("Consolas", int(10 * getattr(self.app, 'ui_scale', 1.0))), bg="#1E1E1E", fg="#A8C7FA", insertbackground="white")
        self.txt_editor.pack(fill="both", expand=True, padx=5, pady=5)

        # Controls
        ctrl = ttk.Frame(right)
        ctrl.pack(fill="x", pady=5)
        ttk.Button(ctrl, text="SAVE CHANGES", command=self._save_node).pack(side="right", padx=5)
        ttk.Button(ctrl, text="DELETE NODE", command=self._delete_node).pack(side="left", padx=5)
        ttk.Button(ctrl, text="REFRESH", command=self._refresh_list).pack(side="left", padx=5)

        self._refresh_list()

    def _refresh_list(self):
        self.tree.delete(*self.tree.get_children())
        query = self.search_var.get().lower()

        nodes = self.app.hippocampus.nodes
        for name, node in nodes.items():
            if query and query not in name.lower():
                continue

            data = node.data
            updated = str(data.get('last_updated', ''))
            ntype = str(data.get('type', 'Unknown'))
            self.tree.insert("", "end", iid=name, text=name, values=(ntype, updated))

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        entity = sel[0]

        node = self.app.hippocampus.nodes.get(entity)
        if node:
            # Dump YAML to text box
            try:
                y_str = yaml.dump(node.data, sort_keys=False, allow_unicode=True)
                self.txt_editor.delete("1.0", tk.END)
                self.txt_editor.insert("1.0", y_str)
            except Exception as e:
                self.txt_editor.delete("1.0", tk.END)
                self.txt_editor.insert("1.0", f"Error parsing YAML: {e}")

    def _save_node(self):
        sel = self.tree.selection()
        if not sel: return
        entity = sel[0]

        content = self.txt_editor.get("1.0", tk.END).strip()
        try:
            # Validate YAML
            new_data = yaml.safe_load(content)
            if not isinstance(new_data, dict): raise ValueError("Must be a dictionary")

            self.app.hippocampus.update_memory_node(entity, new_data)
            self.app.hippocampus.save_memory()
            messagebox.showinfo("Success", f"Updated {entity}")
        except Exception as e:
            messagebox.showerror("YAML Error", str(e))

    def _delete_node(self):
        sel = self.tree.selection()
        if not sel: return
        entity = sel[0]

        if messagebox.askyesno("Confirm", f"Delete memory of {entity}?"):
            del self.app.hippocampus.nodes[entity]
            self.app.hippocampus.save_memory()
            self._refresh_list()
            self.txt_editor.delete("1.0", tk.END)

    def on_theme_change(self):
        pass