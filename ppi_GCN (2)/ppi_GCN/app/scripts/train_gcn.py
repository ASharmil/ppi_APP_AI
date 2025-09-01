# Requirements: torch, torch_geometric, pandas, numpy, scikit-learn
# Input CSV columns:
#   protein_a, protein_b, label, [optional feature_* columns...]
# Produces:
#   - model.pt (trained weights)
#   - public/data/predictions.sample.json (example predictions export)
import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:
    raise RuntimeError("Install torch-geometric to train the GCN: pip install torch torch_geometric -f https://data.pyg.org/whl/torch-$(python -c 'import torch,platform;print(torch.__version__.split(\"+\")[0])')+cpu.html") from e

def build_index(proteins: List[str]) -> Dict[str, int]:
    return {p: i for i, p in enumerate(sorted(set(proteins)))}

def build_graph(df: pd.DataFrame) -> Tuple[Data, Dict[str, int], np.ndarray]:
    # Map proteins to indices
    all_proteins = list(df["protein_a"].values) + list(df["protein_b"].values)
    pid = build_index(all_proteins)
    n = len(pid)

    # Edge index
    edges = np.array([[pid[a], pid[b]] for a, b in zip(df["protein_a"], df["protein_b"])], dtype=np.int64).T
    edge_index = torch.tensor(edges, dtype=torch.long)

    # Node features: aggregate from HADDOCK features if present, else identity
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    if feat_cols:
        # Create simple node features by averaging features where node participates (toy approach)
        X = np.zeros((n, len(feat_cols)), dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)
        for (_, row) in df.iterrows():
            a, b = pid[row["protein_a"]], pid[row["protein_b"]]
            feats = row[feat_cols].values.astype(np.float32)
            X[a] += feats
            X[b] += feats
            counts[a] += 1.0
            counts[b] += 1.0
        counts[counts == 0] = 1.0
        X = (X.T / counts).T
    else:
        X = np.eye(n, dtype=np.float32)
    x = torch.tensor(X, dtype=torch.float)

    # Labels on edges (link prediction)
    y = torch.tensor(df["label"].values.astype(np.float32), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data, pid, y.numpy()

class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, out_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z, edge_index):
        # predict for given edges
        src, dst = edge_index
        pair = torch.cat([z[src], z[dst]], dim=1)
        out = self.mlp(pair).squeeze(-1)
        return out

def train(data: Data, labels: torch.Tensor, epochs: int = 50, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    xdim = data.x.size(1)
    enc = GCNEncoder(xdim).to(device)
    pred = LinkPredictor(64).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=lr)

    y = labels.to(device)
    for ep in range(epochs):
        enc.train(); pred.train()
        opt.zero_grad()
        z = enc(data.x, data.edge_index)
        logits = pred(z, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred_label = (probs > 0.5).float()
                acc = (pred_label == y).float().mean().item()
            print(f"[v0] epoch {ep+1} loss={loss.item():.4f} acc={acc:.3f}")
    return enc, pred

def export_predictions(enc, pred, data: Data, pid: Dict[str, int], out_json: str, top_k: int = 5):
    inv = {v: k for k, v in pid.items()}
    enc.eval(); pred.eval()
    with torch.no_grad():
        z = enc(data.x, data.edge_index)
        # naive: for each node, score against all others, output top_k
        Z = z.cpu()
        out = []
        for i in range(Z.size(0)):
            # score i with all j
            src = torch.full((Z.size(0),), i, dtype=torch.long)
            dst = torch.arange(Z.size(0), dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
            logits = pred(Z, edge_index)
            probs = torch.sigmoid(logits).cpu().numpy()
            order = np.argsort(-probs)
            for j in order[:top_k]:
                if j == i: continue
                out.append({
                    "query_uniprot": inv[i],
                    "predicted_partner_uniprot": inv[j],
                    "score": float(probs[j])
                })
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[v0] exported predictions to {out_json}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to PPI CSV")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--export", default="public/data/predictions.sample.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    data, pid, y = build_graph(df)
    enc, pred_head = train(data, torch.tensor(y))
    # Save weights (optional)
    os.makedirs("models", exist_ok=True)
    torch.save({"enc": enc.state_dict(), "pred": pred_head.state_dict()}, "models/model.pt")
    # Export predictions
    export_predictions(enc, pred_head, data, pid, args.export)

if __name__ == "__main__":
  main()
