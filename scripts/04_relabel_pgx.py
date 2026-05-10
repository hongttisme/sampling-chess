"""Convert a labeled .npz from our 4288 action encoding to pgx 4672.

Plan A migration step: the original labeling pipeline used our custom
action encoding (4288 = 4096 from-to + 192 promo). To use mctx + pgx we
need the same labels in pgx's 4672-action AlphaZero encoding.

Output keeps FENs + value_targets identical and replaces move_indices
with pgx-format integers. move_probs are unchanged.

Usage:
    .venv/bin/python scripts/04_relabel_pgx.py \\
        --in data/labels_50k_random.npz \\
        --out data/labels_50k_random_pgx.npz
"""

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

from sampling_chess import board as B
from sampling_chess.pgx_bridge import chess_move_to_pgx_action, PGX_NUM_ACTIONS


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    print(f"[load] {args.input}")
    d = np.load(args.input)
    fens = d["fens"]
    our_idx = d["move_indices"]  # (N, k) in our 4288 space; -1 padded
    move_probs = d["move_probs"]
    value_targets = d["value_targets"]
    n, k = our_idx.shape
    print(f"[load] {n} positions, k={k}")

    print(f"[conv] mapping move indices our 4288 -> pgx {PGX_NUM_ACTIONS}")
    pgx_idx = np.full((n, k), -1, dtype=np.int32)
    t0 = time.time()
    bad = 0
    for i, fen in enumerate(fens):
        bd = chess.Board(str(fen))
        turn = bd.turn
        for j in range(k):
            our = int(our_idx[i, j])
            if our < 0:
                continue
            try:
                mv = B.index_to_move(our)
                pgx_idx[i, j] = chess_move_to_pgx_action(mv, turn)
            except Exception as e:
                bad += 1
                if bad <= 3:
                    print(f"  [warn] failed @ pos {i} move {j}: {e}")

        if (i + 1) % 5000 == 0 or (i + 1) == n:
            dt = time.time() - t0
            print(f"  [conv] {i+1}/{n} | {(i+1)/dt:.0f} pos/s")

    if bad:
        print(f"[warn] {bad} conversions failed (set to -1)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        fens=fens,
        move_indices=pgx_idx,
        move_probs=move_probs,
        value_targets=value_targets,
    )
    print(f"[save] {args.out}  ({args.out.stat().st_size/1024:.1f} KiB)")

    # Quick sanity: every non-padded label should be < PGX_NUM_ACTIONS
    valid = pgx_idx[pgx_idx >= 0]
    assert (valid < PGX_NUM_ACTIONS).all(), "some labels exceed pgx action space"
    print(f"[ok] all {len(valid)} non-padded labels in [0, {PGX_NUM_ACTIONS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
