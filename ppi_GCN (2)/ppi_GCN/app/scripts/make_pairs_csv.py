#!/usr/bin/env python3
"""
Generate an eval CSV (protein_a,protein_b) from inline pairs or a text file.

Usage:
  - Inline pairs (semicolon-separated, each pair comma-separated):
      python -m scripts.make_pairs_csv --pairs "P12345,Q9Y2Z2;P04637,P31749" --out data/my_eval.csv
  - From a file (each line: P12345,Q9Y2Z2 or space-separated):
      python -m scripts.make_pairs_csv --pairs-file pairs.txt --out data/my_eval.csv

Then run inference with your existing script:
  python -m scripts.infer_gcn_v1 --csv data/my_eval.csv --artifacts artifacts/artifacts.json --model artifacts/gcn_lp.pt --out outputs/scores.csv
"""
import argparse
import csv
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=None, help="Semicolon-separated pairs, each pair comma-separated. Example: 'P1,Q1;P2,Q2'")
    ap.add_argument("--pairs-file", type=str, default=None, help="Path to a text file containing pairs (comma or space separated per line)")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path (will create directories as needed)")
    return ap.parse_args()

def parse_inline(pairs: str):
    items = []
    for chunk in pairs.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "," in chunk:
            a, b = [x.strip() for x in chunk.split(",", 1)]
        else:
            parts = chunk.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid pair format: {chunk}")
            a, b = parts
        items.append((a, b))
    return items

def parse_file(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                a, b = [x.strip() for x in line.split(",", 1)]
            else:
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid pair line: {line}")
                a, b = parts
            items.append((a, b))
    return items

def main():
    args = parse_args()
    if not args.pairs and not args.pairs_file:
        raise SystemExit("Provide --pairs or --pairs-file")

    pairs = []
    if args.pairs:
        pairs.extend(parse_inline(args.pairs))
    if args.pairs_file:
        pairs.extend(parse_file(args.pairs_file))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protein_a", "protein_b"])
        for a, b in pairs:
            w.writerow([a, b])
    print(f"Wrote {len(pairs)} pairs to {out_path}")

if __name__ == "__main__":
    main()
