"""Merge multiple .npz label files into one combined dataset.

Useful when labeling several months of Lichess Elite separately to scale
up the SL training data beyond what fits in one Elite month (~280k positions).

Usage:
    python scripts/12_merge_labels.py \\
        --inputs data/labels_elite_nov.npz data/labels_elite_oct.npz \\
        --out data/labels_elite_combined.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    fens, idx, probs, vals = [], [], [], []
    for path in args.inputs:
        if not path.exists():
            print(f"[err] missing {path}")
            return 1
        d = np.load(path)
        n = len(d["fens"])
        fens.append(d["fens"])
        idx.append(d["move_indices"])
        probs.append(d["move_probs"])
        vals.append(d["value_targets"])
        print(f"[load] {path}: {n:>7} positions, {path.stat().st_size/1e6:.1f} MB")

    fens_all = np.concatenate(fens)
    idx_all = np.concatenate(idx, axis=0)
    probs_all = np.concatenate(probs, axis=0)
    vals_all = np.concatenate(vals)
    n_total = len(fens_all)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        fens=fens_all,
        move_indices=idx_all,
        move_probs=probs_all,
        value_targets=vals_all,
    )
    print(f"\n[ok] merged {n_total:,} positions -> {args.out}")
    print(f"[size] {args.out.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
