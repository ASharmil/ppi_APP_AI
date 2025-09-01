import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple

from train_gcn import GCNEncoder, LinkPredictor, build_graph

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--train_csv", required=True, help="CSV used to fit node features/index")
  ap.add_argument("--cand_csv", required=True, help="Candidate pairs CSV with protein_a, protein_b")
  ap.add_argument("--model", default="models/model.pt")
  ap.add_argument("--export", default="public/data/predictions.sample.json")
  args = ap.parse_args()

  # Rebuild graph and load features/index
  train_df = pd.read_csv(args.train_csv)
  data, pid, _ = build_graph(train_df)

  checkpoint = torch.load(args.model, map_location="cpu")
  enc = GCNEncoder(data.x.size(1))
  pred = LinkPredictor(64)
  enc.load_state_dict(checkpoint["enc"])
  pred.load_state_dict(checkpoint["pred"])
  enc.eval(); pred.eval()

  cand = pd.read_csv(args.cand_csv)
  out = []
  with torch.no_grad():
    z = enc(data.x, data.edge_index)
    for _, row in cand.iterrows():
      a = pid.get(row["protein_a"]); b = pid.get(row["protein_b"])
      if a is None or b is None:
        continue
      src = torch.tensor([a], dtype=torch.long)
      dst = torch.tensor([b], dtype=torch.long)
      edge_index = torch.stack([src, dst], dim=0)
      logit = pred(z, edge_index)
      prob = torch.sigmoid(logit).item()
      out.append({
        "query_uniprot": row["protein_a"],
        "predicted_partner_uniprot": row["protein_b"],
        "score": prob
      })
  os.makedirs(os.path.dirname(args.export), exist_ok=True)
  with open(args.export, "w") as f:
    json.dump(out, f, indent=2)
  print(f"[v0] exported predictions to {args.export}")

if __name__ == "__main__":
  main()
