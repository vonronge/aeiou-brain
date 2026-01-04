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
import yaml
import time
import uuid
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Optional, List, Tuple, Any

# --- VECTOR STORE OPTIMIZATION ---
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class EntityNode:
    """
    A single neuron in the semantic graph.
    """

    def __init__(self, entity_name, entity_type, content):
        self.id = str(uuid.uuid4())
        self.entity = entity_name
        self.type = entity_type
        self.data = content  # The full YAML dictionary
        self.embedding = None  # Tensor [1, D] (Transient, not saved in YAML)
        self.last_accessed = time.time()


class Organelle_Hippocampus:
    def __init__(self, memory_dir: str, device: str = "cuda", golgi=None):
        self.memory_dir = memory_dir
        self.device = device
        self.golgi = golgi

        # File Paths
        self.graph_path = os.path.join(memory_dir, "knowledge_graph.yaml")
        self.vector_path = os.path.join(memory_dir, "semantic_index.faiss")
        self.meta_path = os.path.join(memory_dir, "index_meta.json")

        # In-Memory Graph
        self.nodes = {}  # {entity_name: EntityNode}

        # Vector Index State
        self.vector_map = []  # Maps numeric ID -> entity_name
        self.faiss_index = None
        self.torch_index = None
        self.dimension = 768  # Standard embedding size

        # Initialize Systems
        self._init_vector_store()
        self.load_memory()

        # Bootstrap if empty
        if "My Knowledge Graph" not in self.nodes:
            self._bootstrap_meta_memory()

    def _log(self, msg, tag="INFO"):
        if self.golgi:
            # Map tag to Golgi method
            method = getattr(self.golgi, tag.lower(), self.golgi.info)
            method(msg, source="Hippocampus")
        else:
            print(f"[Hippocampus:{tag}] {msg}")

    def _init_vector_store(self):
        if HAS_FAISS:
            # Inner Product (Cosine Similarity if normalized)
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            if not self.golgi: print("[Hippocampus] FAISS Accelerated Index Active.")
        else:
            self.torch_index = torch.empty(0, self.dimension).to(self.device)
            self._log("FAISS not found. Using PyTorch fallback (slower).", "WARN")

    def _bootstrap_meta_memory(self):
        seed_yaml = {
            "entity": "My Knowledge Graph",
            "type": "System/Meta",
            "core_summary": "The central index of all persistent memories.",
            "key_facts": ["Tracks Entity Counts", "Maintains Global Timeline"],
            "relations": [],
            "timeline": [f"{datetime.now().strftime('%Y-%m-%d')}: System Genesis"],
            "last_updated": datetime.now().strftime('%Y-%m-%d'),
            "confidence": "high"
        }
        self.nodes["My Knowledge Graph"] = EntityNode("My Knowledge Graph", "System/Meta", seed_yaml)

    # --- I/O OPERATIONS ---

    def load_memory(self):
        # 1. Load Graph (YAML)
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    for name, node_data in data.items():
                        node = EntityNode(name, node_data.get('type', 'Unknown'), node_data)
                        self.nodes[name] = node
                self._log(f"Loaded {len(self.nodes)} semantic nodes.", "INFO")
            except Exception as e:
                self._log(f"Graph Load Error: {e}", "ERROR")

        # 2. Load Vector Index
        if HAS_FAISS and os.path.exists(self.vector_path):
            try:
                self.faiss_index = faiss.read_index(self.vector_path)
                if os.path.exists(self.meta_path):
                    with open(self.meta_path, 'r') as f:
                        self.vector_map = json.load(f)
            except Exception as e:
                self._log(f"FAISS Load Error: {e}", "ERROR")

        elif not HAS_FAISS and os.path.exists(self.vector_path + ".pt"):
            try:
                pkg = torch.load(self.vector_path + ".pt", map_location=self.device)
                self.torch_index = pkg['vectors']
                self.vector_map = pkg['map']
            except Exception as e:
                self._log(f"Torch Index Load Error: {e}", "ERROR")

    def save_memory(self):
        # 1. Save Graph
        export_data = {name: node.data for name, node in self.nodes.items()}
        try:
            with open(self.graph_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            self._log(f"Graph Save Error: {e}", "ERROR")

        # 2. Save Vectors
        try:
            if HAS_FAISS and self.faiss_index:
                faiss.write_index(self.faiss_index, self.vector_path)
                with open(self.meta_path, 'w') as f:
                    json.dump(self.vector_map, f)
            elif self.torch_index is not None:
                torch.save({'vectors': self.torch_index, 'map': self.vector_map}, self.vector_path + ".pt")
        except Exception as e:
            self._log(f"Vector Save Error: {e}", "ERROR")

        self._log("Memory Consolidated.", "SAVE")

    # --- CORE MEMORY FUNCTIONS ---

    def add_vector(self, entity_name: str, embedding_tensor: torch.Tensor):
        """Adds an embedding to the index and maps it to the entity name."""
        if embedding_tensor is None: return

        # Normalize for Cosine Similarity
        vector = F.normalize(embedding_tensor, p=2, dim=1)

        if HAS_FAISS:
            np_vec = vector.cpu().numpy().astype('float32')
            self.faiss_index.add(np_vec)
            self.vector_map.append(entity_name)
        else:
            self.torch_index = torch.cat([self.torch_index, vector.to(self.device)], dim=0)
            self.vector_map.append(entity_name)

    def update_memory_node(self, entity_name: str, new_yaml_dict: dict):
        """Updates graph data. Does NOT auto-update vectors (requires re-embedding)."""
        if entity_name not in self.nodes:
            # Create new
            self.nodes[entity_name] = EntityNode(entity_name, new_yaml_dict.get("type", "Unknown"), new_yaml_dict)
        else:
            # Update existing
            node = self.nodes[entity_name]
            node.data = new_yaml_dict
            node.last_updated = time.time()

    def search(self, query_embedding: torch.Tensor, top_k=3, threshold=0.4, hops=1) -> List[Tuple[float, Any]]:
        """
        Associative Recall:
        1. Finds similar vectors.
        2. Expands to 1-hop graph neighbors.
        Returns: List of (score, EntityNode)
        """
        if query_embedding is None: return []

        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        hits = []

        # 1. Vector Search
        if HAS_FAISS:
            q_np = query_embedding.cpu().numpy().astype('float32')
            scores, indices = self.faiss_index.search(q_np, top_k * 2)  # Over-fetch
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.vector_map) and score > threshold:
                    hits.append(self.vector_map[idx])
        else:
            if len(self.torch_index) > 0:
                sim = F.cosine_similarity(query_embedding, self.torch_index)
                scores, indices = torch.topk(sim, k=min(top_k * 2, len(self.torch_index)))
                for score, idx in zip(scores, indices):
                    if score > threshold:
                        hits.append(self.vector_map[idx.item()])

        # Deduplicate (Latest reference wins)
        unique_hits = list(set(hits))

        # 2. Graph Expansion (Associativity)
        final_set = set(unique_hits)

        if hops > 0:
            neighbors = set()
            for entity in unique_hits:
                node = self.nodes.get(entity)
                if not node: continue

                # Parse "Relations" field
                rels = node.data.get('relations', [])
                for rel in rels:
                    # Format: "Predicate -> Target"
                    if "->" in rel:
                        target = rel.split("->")[-1].strip()
                        if target in self.nodes:
                            neighbors.add(target)

            final_set.update(neighbors)

        # 3. Format Results
        results = []
        for name in final_set:
            if name in self.nodes:
                # Score is 1.0 for neighbors since we don't have vector scores for them all easily
                results.append((1.0, self.nodes[name]))

        return results[:top_k + 2]  # Limit context window load

    # --- LLM UTILITIES ---

    def get_training_corpus(self) -> List[str]:
        """Returns all memory nodes formatted for LLM training."""
        corpus = []
        for name, node in self.nodes.items():
            try:
                y_str = yaml.dump(node.data, sort_keys=False, allow_unicode=True)
                corpus.append(f"### MEMORY ENGRAM: {name} ###\n{y_str}\n<|endoftext|>")
            except:
                pass
        return corpus

    def format_context_block(self, memory_nodes: List[Tuple[float, Any]]) -> str:
        """Formats search results into a system prompt string."""
        if not memory_nodes: return ""

        out = "### RELEVANT LONG-TERM MEMORIES ###\n"
        for _, node in memory_nodes:
            d = node.data
            out += f"ENTITY: {node.entity} ({node.type})\n"
            out += f"  SUMMARY: {d.get('core_summary', '')}\n"

            rels = d.get('relations', [])
            if rels:
                out += f"  LINKS: {'; '.join(rels[:3])}\n"
            out += "\n"
        return out