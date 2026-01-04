"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Reticulum:
A structural support network for memory.
Provides safe, unified access (CRUD) to the Hippocampus Knowledge Graph.
"""

import yaml
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Assuming EntityNode is defined in Hippocampus, we treat it as Any here
# or import it if circular imports are managed.
try:
    from Organelles.hippocampus import EntityNode
except ImportError:
    EntityNode = Any


class Organelle_Reticulum:
    def __init__(self, hippocampus, golgi):
        self.hippocampus = hippocampus
        self.golgi = golgi

    def _log(self, msg, tag="INFO"):
        if self.golgi:
            getattr(self.golgi, tag.lower())(msg, source="Reticulum")
        else:
            print(f"[Reticulum:{tag}] {msg}")

    # --- CRUD OPERATIONS ---

    def list_nodes(self, query: str = None) -> Dict[str, Any]:
        """
        Returns a filtered dictionary of nodes.
        query: Case-insensitive search string.
        """
        if not self.hippocampus: return {}

        all_nodes = self.hippocampus.nodes
        if not query:
            return all_nodes

        q = query.lower()
        return {k: v for k, v in all_nodes.items() if q in k.lower()}

    def get_node(self, name: str) -> Optional[Any]:
        if not self.hippocampus: return None
        return self.hippocampus.nodes.get(name)

    def upsert_node(self, name: str, data: dict):
        """
        Create or Update a node safely.
        """
        if not self.hippocampus: return

        # Validation
        if not isinstance(data, dict):
            self._log(f"Invalid data type for {name}: {type(data)}", "ERROR")
            return

        # Ensure minimal schema
        if "type" not in data: data["type"] = "Concept"

        # Use Hippocampus internal logic if available, else manual
        if hasattr(self.hippocampus, "update_memory_node"):
            self.hippocampus.update_memory_node(name, data)
        else:
            # Fallback direct insertion
            from Organelles.hippocampus import EntityNode
            self.hippocampus.nodes[name] = EntityNode(name, data["type"], data)

        self.hippocampus.save_memory()
        self._log(f"Persisted node: {name}", "SAVE")

    def delete_node(self, name: str):
        if not self.hippocampus: return
        if name in self.hippocampus.nodes:
            del self.hippocampus.nodes[name]
            self.hippocampus.save_memory()
            self._log(f"Deleted node: {name}", "WARN")

    # --- SERIALIZATION UTILS ---

    def to_yaml(self, node_data: dict) -> str:
        try:
            return yaml.dump(node_data, sort_keys=False, allow_unicode=True)
        except Exception as e:
            self._log(f"YAML Encode Error: {e}", "ERROR")
            return ""

    def from_yaml(self, yaml_str: str) -> dict:
        try:
            data = yaml.safe_load(yaml_str)
            if not isinstance(data, dict):
                raise ValueError("Parsed YAML is not a dictionary.")
            return data
        except Exception as e:
            raise ValueError(f"YAML Parse Error: {e}")