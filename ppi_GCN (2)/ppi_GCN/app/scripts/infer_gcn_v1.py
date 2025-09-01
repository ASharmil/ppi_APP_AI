# Usage:
#   python scripts/infer_gcn_v1.py --csv data/pairs_to_score.csv --artifacts artifacts/artifacts.json --model artifacts/gcn_lp.pt --out scores.csv --drug "imatinib"
import argparse
import json
import os
from typing import List, Tuple, Dict
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from gcn.model_gcn import GCNLinkPredictor
from gcn.data_utils import load_pairs_csv, pairs_to_tensor
from drug_data import find_drug_targets

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Pairs to score: columns protein_a, protein_b")
    p.add_argument("--artifacts", required=True, help="Path to artifacts.json from training")
    p.add_argument("--model", required=True, help="Path to gcn_lp.pt")
    p.add_argument("--out", default="scores.csv", help="Output CSV path")
    p.add_argument("--drug", default=None, help="Optional drug name to fetch live targets from ChEMBL and annotate")
    args = p.parse_args()

    print("[v0] Loading artifacts and model...")
    with open(args.artifacts, "r") as f:
        art = json.load(f)
    node_index: Dict[str, int] = art["node_index"]
    id_to_protein: Dict[str, str] = {int(k): v for k, v in art["id_to_protein"].items()}
    train_edges: List[List[int]] = art["train_edges"]

    df = load_pairs_csv(args.csv)

    # Map pairs to ids (skip pairs with unknown nodes)
    unknown = set()
    pairs_idx: List[Tuple[int, int]] = []
    keep_rows = []
    for i, r in df.iterrows():
        a = r["protein_a"]; b = r["protein_b"]
        if a not in node_index or b not in node_index:
            unknown.add((a, b))
            continue
        pairs_idx.append((node_index[a], node_index[b]))
        keep_rows.append(i)
    if not pairs_idx:
        print("[v0] No scorable pairs (proteins not in training node index).")
        return

    edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    num_nodes = len(node_index)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNLinkPredictor(num_nodes=num_nodes)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(device).eval()

    t_pairs = pairs_to_tensor(pairs_idx)
    with torch.no_grad():
        logits = model(data.edge_index.to(device), t_pairs.to(device), device)
        probs = torch.sigmoid(logits).cpu().tolist()

    out_df = df.loc[keep_rows].copy().reset_index(drop=True)
    out_df["probability"] = probs

    if args.drug:
        print(f"[v0] Fetching live targets for drug: {args.drug}")
        targets = find_drug_targets(args.drug)
        # Build a quick lookup for UniProt accessions targeted by the drug
        accessions = set()
        for t in targets:
            for u in t.get("uniprot_accessions", []):
                accessions.add(u)
        out_df["drug_targets_overlap"] = out_df.apply(
            lambda r: (r["protein_a"] in accessions) or (r["protein_b"] in accessions), axis=1
        )
        # Save a sidecar JSON with details
        with open(os.path.splitext(args.out)[0] + ".drug_targets.json", "w") as f:
            json.dump(targets, f, indent=2)

    out_df.to_csv(args.out, index=False)
    print(f"[v0] Wrote predictions to {args.out}")
    if unknown:
        print(f"[v0] Skipped {len(unknown)} pairs with unknown proteins (not in training index).")

if __name__ == "__main__":
    main()
