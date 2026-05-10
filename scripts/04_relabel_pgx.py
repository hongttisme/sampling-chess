"""Convert a labeled .npz from our 4288 action encoding to pgx 4672 +
pre-encode pgx observations to avoid per-load pgx_from_fen overhead at train time.

Plan A migration step: the original labeling pipeline used our custom
action encoding (4288 = 4096 from-to + 192 promo). To use mctx + pgx we
need the same labels in pgx's 4672 AlphaZero encoding AND the (8,8,119)
pgx observations directly available, since pgx's _from_fen is ~30 ms/pos
in JAX-traced overhead — fine once but expensive at training start.

Output schema:
  fens          : (N,) <U100        original FEN strings
  observation   : (N, 8, 8, 119) float32   pgx observation
  masks         : (N, 4672) bool           pgx legal_action_mask
  move_indices  : (N, k) int32             top-k move indices in pgx 4672
  move_probs    : (N, k) float32           soft policy targets (unchanged)
  value_targets : (N,) float32             value targets (unchanged)

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
from sampling_chess.pgx_bridge import (
    chess_board_to_pgx_state,
    chess_move_to_pgx_action,
    PGX_NUM_ACTIONS,
)
from sampling_chess.net import PGX_OBSERVATION_CHANNELS


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

    print(f"[conv] action mapping (our 4288 -> pgx {PGX_NUM_ACTIONS}) + "
          f"pgx observation/mask precompute")
    pgx_idx = np.full((n, k), -1, dtype=np.int32)
    obs = np.zeros((n, 8, 8, PGX_OBSERVATION_CHANNELS), dtype=np.float32)
    masks = np.zeros((n, PGX_NUM_ACTIONS), dtype=bool)
    t0 = time.time()
    bad = 0
    for i, fen in enumerate(fens):
        bd = chess.Board(str(fen))
        turn = bd.turn

        # 1) Map move indices to pgx action labels.
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

        # 2) Pre-encode pgx observation + legal mask.
        state = chess_board_to_pgx_state(bd)
        obs[i] = np.asarray(state.observation, dtype=np.float32)
        masks[i] = np.asarray(state.legal_action_mask, dtype=bool)

        if (i + 1) % 1000 == 0 or (i + 1) == n:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (n - i - 1) / rate
            print(f"  [conv] {i+1}/{n} | {rate:.1f} pos/s | ETA {eta:.0f}s")

    if bad:
        print(f"[warn] {bad} conversions failed (set to -1)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        fens=fens,
        observation=obs,
        masks=masks,
        move_indices=pgx_idx,
        move_probs=move_probs,
        value_targets=value_targets,
    )
    print(f"[save] {args.out}  ({args.out.stat().st_size/(1024*1024):.1f} MiB)")

    valid = pgx_idx[pgx_idx >= 0]
    assert (valid < PGX_NUM_ACTIONS).all(), "some labels exceed pgx action space"
    print(f"[ok] all {len(valid)} non-padded labels in [0, {PGX_NUM_ACTIONS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
