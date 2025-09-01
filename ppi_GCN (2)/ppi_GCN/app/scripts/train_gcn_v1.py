# CSV schema: protein_a, protein_b, label (0/1). If additional columns exist, they are ignored here.
# Usage:
#   python scripts/train_gcn_v1.py --csv data/train.csv --epochs 30 --emb-dim 64 --hidden-dim 64 --save-dir ./artifacts
import argparse
import json
import os
import random
from typing import Tuple, List, Set
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from gcn.model_gcn import GCNLinkPredictor
from gcn.data_utils import (
    load_pairs_csv, build_node_index, df_to_edge_index, sample_negative_pairs,
    build_positive_set, save_artifacts, pairs_to_tensor
)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def split_df(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.1, seed=42):
    df_pos = df[df["label"] == 1].copy()
    df_neg = df[df["label"] == 0].copy()
    df_pos = df_pos.sample(frac=1.0, random_state=seed)
    n_pos = len(df_pos)
    n_val = int(n_pos * val_ratio)
    n_test = int(n_pos * test_ratio)
    val_pos = df_pos.iloc[:n_val]
    test_pos = df_pos.iloc[n_val:n_val+n_test]
    train_pos = df_pos.iloc[n_val+n_test:]

    # Negatives: stratify-ish by slicing
    df_neg = df_neg.sample(frac=1.0, random_state=seed)
    n_neg = len(df_neg)
    n_valn = int(n_neg * val_ratio)
    n_testn = int(n_neg * test_ratio)
    val_neg = df_neg.iloc[:n_valn]
    test_neg = df_neg.iloc[n_valn:n_valn+n_testn]
    train_neg = df_neg.iloc[n_valn+n_testn:]
    return (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg)

def make_batches(pos_pairs: List[Tuple[int,int]], node_count: int, batch_size: int, pos_set: Set[Tuple[int,int]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    # create batches of 50/50 pos/neg when possible
    batches = []
    pos_idx = 0
    while pos_idx < len(pos_pairs):
        chunk = pos_pairs[pos_idx:pos_idx + batch_size // 2]
        pos_idx += len(chunk)
        neg_needed = max(1, len(chunk))
        negs = sample_negative_pairs(neg_needed, node_count, pos_set)
        pairs = chunk + negs
        labels = [1]*len(chunk) + [0]*len(negs)
        t_pairs = pairs_to_tensor(pairs)
        t_labels = torch.tensor(labels, dtype=torch.float32)
        batches.append((t_pairs, t_labels))
    return batches

def evaluate(model, data, eval_pairs: torch.Tensor, labels: torch.Tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(data.edge_index.to(device), eval_pairs.to(device), device)
        probs = torch.sigmoid(logits).cpu()
        loss = F.binary_cross_entropy_with_logits(logits.cpu(), labels)
        preds = (probs >= 0.5).long()
        acc = (preds == labels.long()).float().mean().item()
    return float(loss.item()), float(acc)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV with protein_a, protein_b, label")
    p.add_argument("--save-dir", default="artifacts", help="Directory to save model and artifacts")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("[v0] Starting training with args:", vars(args))
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    df = load_pairs_csv(args.csv)
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = split_df(df, val_ratio=0.1, test_ratio=0.1, seed=args.seed)

    # Build node index across ALL proteins appearing in the dataset
    node_index = build_node_index(df)
    node_count = len(node_index)
    print(f"[v0] Node count: {node_count}")

    # Build training graph from positive edges only
    train_graph_df = pd.concat([train_pos], ignore_index=True)
    edge_index = df_to_edge_index(train_graph_df, node_index)
    data = Data(edge_index=edge_index, num_nodes=node_count)

    # Build positive set for negative sampling and batches
    pos_set = build_positive_set(train_graph_df, node_index)
    pos_pairs_idx = [(node_index[r["protein_a"]], node_index[r["protein_b"]]) for _, r in train_pos.iterrows()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNLinkPredictor(num_nodes=node_count, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    # Prepare validation/test tensors
    def df_to_pairs_labels(dfp: pd.DataFrame, dfn: pd.DataFrame):
        pairs = [(node_index[r["protein_a"]], node_index[r["protein_b"]]) for _, r in dfp.iterrows()]
        pairs += [(node_index[r["protein_a"]], node_index[r["protein_b"]]) for _, r in dfn.iterrows()]
        labels = [1]*len(dfp) + [0]*len(dfn)
        return pairs_to_tensor(pairs), torch.tensor(labels, dtype=torch.float32)

    val_pairs, val_labels = df_to_pairs_labels(val_pos, val_neg)
    test_pairs, test_labels = df_to_pairs_labels(test_pos, test_neg)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = make_batches(pos_pairs=pos_pairs_idx, node_count=node_count, batch_size=args.batch_size, pos_set=pos_set)
        epoch_loss = 0.0
        for t_pairs, t_labels in batches:
            opt.zero_grad()
            logits = model(data.edge_index.to(device), t_pairs.to(device), device)
            loss = F.binary_cross_entropy_with_logits(logits.cpu(), t_labels)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

        v_loss, v_acc = evaluate(model, data, val_pairs, val_labels, device)
        print(f"[v0] Epoch {epoch:03d} | train_loss={epoch_loss:.4f} val_loss={v_loss:.4f} val_acc={v_acc:.4f}")

        if v_acc > best_val:
            best_val = v_acc
            # Save best
            torch.save(model.state_dict(), os.path.join(args.save_dir, "gcn_lp.pt"))
            # Save artifacts: node index and train edges
            train_edges_directed = data.edge_index.t().cpu().tolist()
            save_artifacts(args.save_dir, node_index, train_edges_directed)

    t_loss, t_acc = evaluate(model, data, test_pairs, test_labels, device)
    print(f"[v0] Training done. Test loss={t_loss:.4f} test_acc={t_acc:.4f}")
    # Save final metrics
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump({"val_best_acc": best_val, "test_loss": t_loss, "test_acc": t_acc}, f)

if __name__ == "__main__":
    main()
