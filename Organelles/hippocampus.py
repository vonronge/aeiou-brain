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
import torch.nn.functional as F
import os
import json
import yaml
import time
import uuid
import numpy as np
from datetime import datetime

# --- OPTIONAL: FAISS FOR SCALE ---
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print(" ! Hippocampus: FAISS not found. Using PyTorch fallback (slower at >10k nodes).")
    print("   Run: pip install faiss-cpu")


class EntityNode:
    """
    Represents a single node in the Knowledge Graph.
    """

    def __init__(self, entity_name, entity_type, content):
        self.id = str(uuid.uuid4())
        self.entity = entity_name
        self.type = entity_type
        self.data = content  # The YAML dictionary
        self.embedding = None  # Tensor [1, D] (Kept for syncing)
        self.last_accessed = time.time()


class Organelle_Hippocampus:
    def __init__(self, memory_dir, device="cuda"):
        self.memory_dir = memory_dir
        self.device = device

        # Paths
        self.graph_path = os.path.join(memory_dir, "knowledge_graph.yaml")
        self.vector_path = os.path.join(memory_dir, "semantic_index.faiss")
        self.meta_path = os.path.join(memory_dir, "index_meta.json")

        # In-Memory Stores
        self.nodes = {}  # {entity_name: EntityNode}

        # Vector Store State
        self.vector_map = []  # Maps index ID -> entity_name
        self.faiss_index = None
        self.torch_index = None
        self.dimension = 768  # Default embedding size

        self._init_vector_store()
        self.load_memory()

        # Bootstrap
        if "My Knowledge Graph" not in self.nodes:
            self._bootstrap_meta_memory()

    def _init_vector_store(self):
        if HAS_FAISS:
            # Inner Product (Cosine sim if vectors are normalized)
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
        else:
            self.torch_index = torch.empty(0, self.dimension).to(self.device)

    def _bootstrap_meta_memory(self):
        seed_yaml = {
            "entity": "My Knowledge Graph",
            "type": "System/Meta",
            "core_summary": "The central index of all persistent memories stored by the AI.",
            "key_facts": ["Tracks Entity Counts", "Maintains Global Timeline"],
            "relations": [],
            "timeline": [f"{datetime.now().strftime('%Y-%m-%d')}: System Genesis"],
            "last_updated": datetime.now().strftime('%Y-%m-%d'),
            "confidence": "high"
        }
        # Create without embedding initially
        self.create_memory_node(seed_yaml, "My Knowledge Graph", "System/Meta", None)

    # --- IO OPERATIONS ---
    def load_memory(self):
        # 1. Load Graph (YAML)
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    for name, node_data in data.items():
                        node = EntityNode(name, node_data.get('type', 'Unknown'), node_data)
                        self.nodes[name] = node
                print(f" > Hippocampus: Loaded {len(self.nodes)} semantic nodes.")
            except Exception as e:
                print(f" ! Graph Load Error: {e}")

        # 2. Load Vector Index
        if HAS_FAISS and os.path.exists(self.vector_path):
            try:
                self.faiss_index = faiss.read_index(self.vector_path)
                if os.path.exists(self.meta_path):
                    with open(self.meta_path, 'r') as f:
                        self.vector_map = json.load(f)
            except Exception as e:
                print(f" ! FAISS Load Error: {e}")
        elif not HAS_FAISS and os.path.exists(self.vector_path + ".pt"):
            # Fallback for torch
            try:
                pkg = torch.load(self.vector_path + ".pt", map_location=self.device)
                self.torch_index = pkg['vectors']
                self.vector_map = pkg['map']
            except:
                pass

    def save_memory(self):
        # 1. Save Graph
        export_data = {name: node.data for name, node in self.nodes.items()}
        try:
            with open(self.graph_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            print(f" ! Graph Save Error: {e}")

        # 2. Save Vectors
        try:
            if HAS_FAISS and self.faiss_index:
                faiss.write_index(self.faiss_index, self.vector_path)
                with open(self.meta_path, 'w') as f:
                    json.dump(self.vector_map, f)
            elif self.torch_index is not None:
                torch.save({'vectors': self.torch_index, 'map': self.vector_map}, self.vector_path + ".pt")
        except Exception as e:
            print(f" ! Vector Save Error: {e}")

        print(" > Hippocampus: Consolidated.")

    # --- CORE FUNCTIONS ---

    def create_memory_node(self, yaml_dict, entity_name, entity_type, embedding_tensor):
        # Create Node
        node = EntityNode(entity_name, entity_type, yaml_dict)
        self.nodes[entity_name] = node

        # Update Vectors
        if embedding_tensor is not None:
            self._add_vector(entity_name, embedding_tensor)

        return f"Created: {entity_name}"

    def update_memory_node(self, entity_name, new_yaml_dict, new_embedding=None):
        if entity_name not in self.nodes: return "Node missing."

        node = self.nodes[entity_name]
        node.data = new_yaml_dict
        node.last_updated = time.time()

        if new_embedding is not None:
            self._add_vector(entity_name,
                             new_embedding)  # FAISS doesn't support easy update, we append and rely on map logic

        return f"Updated: {entity_name}"

    def _add_vector(self, name, tensor):
        # Normalize for Cosine Similarity
        tensor = F.normalize(tensor, p=2, dim=1)

        if HAS_FAISS:
            np_vec = tensor.cpu().numpy().astype('float32')
            self.faiss_index.add(np_vec)
            self.vector_map.append(name)
        else:
            self.torch_index = torch.cat([self.torch_index, tensor.to(self.device)], dim=0)
            self.vector_map.append(name)

    def retrieve_relevant(self, query_embedding, top_k=2, threshold=0.4, hops=1):
        """
        Retrieves nodes + 1-Hop Neighbors (Associative Recall).
        """
        if query_embedding is None: return []

        # Normalize Query
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

        # 1. SEARCH
        hits = []
        if HAS_FAISS:
            q_np = query_embedding.cpu().numpy().astype('float32')
            scores, indices = self.faiss_index.search(q_np, top_k * 2)  # Grab extra candidates
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.vector_map) and score > threshold:
                    hits.append(self.vector_map[idx])
        else:
            sim = F.cosine_similarity(query_embedding, self.torch_index)
            scores, indices = torch.topk(sim, k=min(top_k * 2, len(self.torch_index)))
            for score, idx in zip(scores, indices):
                if score > threshold:
                    hits.append(self.vector_map[idx.item()])

        # Deduplicate hits (latest version of entity wins)
        # (In a naive append-only log, the last index is the newest)
        unique_hits = list(set(hits))

        # 2. GRAPH EXPANSION (Associativity)
        final_set = set(unique_hits)

        if hops > 0:
            neighbors = set()
            for entity in unique_hits:
                node = self.nodes.get(entity)
                if not node: continue

                # Check relations
                # Format: "Type -> Target"
                rels = node.data.get('relations', [])
                for rel in rels:
                    if "->" in rel:
                        target = rel.split("->")[-1].strip()
                        if target in self.nodes:
                            neighbors.add(target)

            # Add valid neighbors
            final_set.update(neighbors)

        # 3. Retrieve Nodes
        results = []
        for name in final_set:
            if name in self.nodes:
                results.append((1.0, self.nodes[name]))  # Score is dummy for expanded nodes

        return results[:top_k + 2]  # Limit context window

    def get_training_corpus(self):
        corpus = []
        for name, node in self.nodes.items():
            try:
                y_str = yaml.dump(node.data, sort_keys=False, allow_unicode=True)
                corpus.append(f"### MEMORY ENGRAM: {name} ###\n{y_str}\n<|endoftext|>")
            except:
                pass
        return corpus

    # --- PROMPTS ---

    def format_memories_for_context(self, memory_nodes):
        if not memory_nodes: return ""
        out = "### RELEVANT LONG-TERM MEMORIES (SEMANTIC GRAPH) ###\n"
        for _, node in memory_nodes:
            d = node.data
            out += f"ENTITY: {node.entity} ({node.type})\n"
            out += f"  SUMMARY: {d.get('core_summary', '')}\n"

            # Show relations to prove associativity works
            rels = d.get('relations', [])
            if rels: out += f"  LINKS: {'; '.join(rels[:4])}\n"
            out += "\n"
        return out

    def get_extraction_prompt(self, text):
        return f"""
Analyze the text and extract the MAIN Entity into the Knowledge Graph format.
Be concise.

TEMPLATE:
entity: [Name]
type: [Person/Location/Concept]
core_summary: [1-sentence definition]
relations:
  - [Relation] -> [TargetEntity]
timeline:
  - [Date]: [Event]
confidence: high

TEXT:
{text[:2500]}
"""

    def get_merge_prompt(self, existing, new_info):
        return f"""
Merge the New Info into the Existing Node.
CRITICAL: If New Info contradicts Existing, verify context. Prioritize recent data but flag low confidence.

EXISTING:
{json.dumps(existing, indent=2)}

NEW INFO:
{new_info[:2000]}

OUTPUT ONLY UPDATED YAML.
"""