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

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class EntityNode:
    def __init__(self, entity_name, entity_type, content):
        self.id = str(uuid.uuid4())
        self.entity = entity_name
        self.type = entity_type
        self.data = content
        self.last_accessed = time.time()


class Organelle_Hippocampus:
    def __init__(self, memory_dir: str, device: str = "cuda", golgi=None):
        self.memory_dir = memory_dir
        self.device = device
        self.golgi = golgi

        self.graph_path = os.path.join(memory_dir, "knowledge_graph.yaml")
        self.vector_path = os.path.join(memory_dir, "semantic_index.faiss")
        self.meta_path = os.path.join(memory_dir, "index_meta.json")

        self.nodes = {}
        self.vector_map = []
        self.faiss_index = None
        self.torch_index = None
        self.dimension = 768

        self._init_vector_store()
        self.load_memory()

    def _init_vector_store(self):
        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
        else:
            self.torch_index = torch.empty(0, self.dimension).to(self.device)

    # --- CORE OPERATIONS (v23.6 - Robust Shapes) ---

    def consolidate(self, v_vec, a_vec, text):
        """
        Takes raw engrams and a text description, creates a graph node,
        and indexes the vector for future recall.
        """
        if not text: return

        # 1. Create Node
        node_id = text[:50].replace("\n", " ").strip()
        data = {
            "type": "Episodic",
            "content": text,
            "timestamp": str(datetime.now())
        }
        self.update_memory_node(node_id, data)

        # 2. Index Vector (Visual priority)
        if v_vec is not None:
            self.add_vector(node_id, v_vec)

        self.save_memory()
        if self.golgi: self.golgi.save(f"Consolidated: {node_id}", source="Hippocampus")

    def add_vector(self, entity_name: str, embedding_tensor: torch.Tensor):
        if embedding_tensor is None: return

        # Dynamic Shape Handling: Force [1, D]
        if embedding_tensor.ndim == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)

        # Normalize along FEATURE dimension (dim=1)
        # (Prevents the bug where normalizing [1, D] on dim=0 destroys data)
        vector = F.normalize(embedding_tensor, p=2, dim=1)

        if HAS_FAISS:
            np_vec = vector.cpu().numpy().astype('float32')
            self.faiss_index.add(np_vec)
            self.vector_map.append(entity_name)
        else:
            self.torch_index = torch.cat([self.torch_index, vector.to(self.device)], dim=0)
            self.vector_map.append(entity_name)

    def update_memory_node(self, entity_name: str, new_yaml_dict: dict):
        if entity_name not in self.nodes:
            self.nodes[entity_name] = EntityNode(entity_name, new_yaml_dict.get("type", "Unknown"), new_yaml_dict)
        else:
            self.nodes[entity_name].data = new_yaml_dict

    def search(self, query_embedding: torch.Tensor, top_k=3, threshold=0.4) -> List[Tuple[float, Any]]:
        if query_embedding is None: return []

        # Dynamic Shape Handling
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)

        # Normalize Feature Dimension
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        hits = []

        if HAS_FAISS:
            q_np = query_embedding.cpu().numpy().astype('float32')
            scores, indices = self.faiss_index.search(q_np, top_k * 2)
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.vector_map) and score > threshold:
                    hits.append((float(score), self.nodes.get(self.vector_map[idx], self.vector_map[idx])))
        else:
            if len(self.torch_index) > 0:
                # Cosine Similarity against Torch Index
                sim = F.cosine_similarity(query_embedding, self.torch_index)
                scores, indices = torch.topk(sim, k=min(top_k * 2, len(self.torch_index)))
                for score, idx in zip(scores, indices):
                    if score > threshold:
                        hits.append(
                            (float(score), self.nodes.get(self.vector_map[idx.item()], self.vector_map[idx.item()])))

        return hits[:top_k]

    # --- I/O ---
    def load_memory(self):
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
                    for name, d in data.items():
                        self.nodes[name] = EntityNode(name, d.get('type', 'Unknown'), d)
            except:
                pass

        if HAS_FAISS and os.path.exists(self.vector_path):
            try:
                self.faiss_index = faiss.read_index(self.vector_path)
                with open(self.meta_path, 'r') as f:
                    self.vector_map = json.load(f)
            except:
                pass
        elif not HAS_FAISS and os.path.exists(self.vector_path + ".pt"):
            try:
                pkg = torch.load(self.vector_path + ".pt", map_location=self.device)
                self.torch_index = pkg['vectors']
                self.vector_map = pkg['map']
            except:
                pass

    def save_memory(self):
        try:
            export = {n: node.data for n, node in self.nodes.items()}
            with open(self.graph_path, 'w') as f:
                yaml.dump(export, f)

            if HAS_FAISS and self.faiss_index:
                faiss.write_index(self.faiss_index, self.vector_path)
                with open(self.meta_path, 'w') as f:
                    json.dump(self.vector_map, f)
            elif self.torch_index is not None:
                torch.save({'vectors': self.torch_index, 'map': self.vector_map}, self.vector_path + ".pt")
        except:
            pass