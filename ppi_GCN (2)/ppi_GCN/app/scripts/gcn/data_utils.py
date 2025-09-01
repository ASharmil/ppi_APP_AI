from __future__ import annotations
from typing import Dict, List, Tuple, Set
import json
import random
import pandas as pd
import torch

def load_pairs_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"protein_a", "protein_b"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {needed}. Found: {list(df.columns)}")
    # Optional label col; if missing, treat all as unlabeled (for inference CSVs)
    if "label" not in df.columns:
        df["label"] = -1
    return df

def build_node_index(df: pd.DataFrame) -> Dict[str, int]:
    proteins = pd.unique(pd.concat([df["protein_a"], df["protein_b"]], ignore_index=True)).tolist()
    return {p: i for i, p in enumerate(proteins)}

def df_to_edge_index(df_pos: pd.DataFrame, node_index: Dict[str, int]) -> torch.Tensor:
    edges = []
    seen: Set[Tuple[int, int]] = set()
    for _, row in df_pos.iterrows():
        u = node_index[row["protein_a"]]
        v = node_index[row["protein_b"]]
        if u == v:
            continue
        # undirected edges (u,v) and (v,u)
        if (u, v) not in seen and (v, u) not in seen:
            edges.append((u, v))
            edges.append((v, u))
            seen.add((u, v))
            seen.add((v, u))
    if not edges:
        raise ValueError("No edges were constructed from positive pairs.")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def sample_negative_pairs(num_samples: int, node_count: int, positive_set: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # positive_set contains both (u,v) and (v,u)
    negs: List[Tuple[int, int]] = []
    tried = 0
    while len(negs) < num_samples and tried < num_samples * 20:
        u = random.randint(0, node_count - 1)
        v = random.randint(0, node_count - 1)
        tried += 1
        if u == v:
            continue
        if (u, v) in positive_set or (v, u) in positive_set:
            continue
        negs.append((u, v))
    return negs

def build_positive_set(df_pos: pd.DataFrame, node_index: Dict[str, int]) -> Set[Tuple[int, int]]:
    s: Set[Tuple[int, int]] = set()
    for _, r in df_pos.iterrows():
        u = node_index[r["protein_a"]]
        v = node_index[r["protein_b"]]
        if u != v:
            s.add((u, v))
            s.add((v, u))
    return s

def save_artifacts(save_dir: str, node_index: Dict[str, int], train_edges: List[Tuple[int,int]]):
    inv = {int(v): k for k, v in node_index.items()}
    artifacts = {
        "node_index": node_index,
        "id_to_protein": inv,
        "train_edges": train_edges,  # directed list
    }
    with open(f"{save_dir}/artifacts.json", "w") as f:
        json.dump(artifacts, f)

def load_artifacts(path: str):
    with open(path, "r") as f:
        return json.load(f)

def pairs_to_tensor(pairs: List[Tuple[int, int]]) -> torch.Tensor:
    if len(pairs) == 0:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)
