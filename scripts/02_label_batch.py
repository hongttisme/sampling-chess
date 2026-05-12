"""Stage 1 / 2 labeling driver: generate positions, label via Stockfish, save .npz.

Output schema (.npz with allow_pickle=False):
  fens          : (N,) unicode (<U100), original FEN of each position
  move_indices  : (N, k) int32, top-k legal moves' indices in NUM_ACTIONS;
                  short rows padded with -1
  move_probs    : (N, k) float32, softmax(V/0.1) over the k moves;
                  padded with 0.0
  value_targets : (N,) float32, best line's V from side-to-move POV

Stage 1 (prototype, random self-play seeds, ~40min):
    .venv/bin/python scripts/02_label_batch.py \\
        --source random --n 50000 --out data/labels_50k_random.npz

Stage 2/3 (Lichess, ~6h / ~26h):
    .venv/bin/python scripts/02_label_batch.py \\
        --source pgn --pgn-path data/lichess_2024-01.pgn.zst \\
        --n 500000 --out data/labels_500k.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from sampling_chess import data as D
from sampling_chess.stockfish import StockfishPool


def _stream_positions(args):
    if args.source == "random":
        return D.iter_random_selfplay_positions(n=args.n, seed=args.seed)
    if args.source == "pgn":
        if not args.pgn_path:
            raise SystemExit("--pgn-path required for source=pgn")
        return D.iter_pgn_positions(
            args.pgn_path, n=args.n,
            min_rating=args.min_rating, seed=args.seed,
        )
    raise ValueError(f"Unknown source: {args.source}")


def _save(out: Path, fens, move_idx, move_probs, value_targets):
    np.savez_compressed(
        out,
        fens=np.array(fens, dtype="<U100"),
        move_indices=np.stack(move_idx),
        move_probs=np.stack(move_probs),
        value_targets=np.asarray(value_targets, dtype=np.float32),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["random", "pgn"], required=True)
    p.add_argument("--pgn-path", help="for source=pgn")
    p.add_argument("--n", type=int, required=True, help="target #positions")
    p.add_argument("--min-rating", type=int, default=2000,
                   help="rating filter for source=pgn")
    p.add_argument("--out", type=Path, required=True, help=".npz output path")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--multipv", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=5000,
                   help="save partial output every N positions; 0 disables")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[gen] generating {args.n} positions from source={args.source}")
    t_gen = time.time()
    positions = list(_stream_positions(args))
    print(f"[gen] got {len(positions)} positions in {time.time()-t_gen:.1f}s")

    if not positions:
        print("[err] no positions to label")
        return 1

    print(f"[lab] {args.workers} workers, depth {args.depth}, multipv {args.multipv}")
    t0 = time.time()
    fens, move_idx, move_probs, value_targets = [], [], [], []

    with StockfishPool(
        n_workers=args.workers, depth=args.depth, multipv=args.multipv
    ) as pool:
        for i, lp in enumerate(pool.label_batch_iter(positions, chunksize=8), start=1):
            fens.append(lp.fen)
            mi = np.full(args.multipv, -1, dtype=np.int32)
            mp_ = np.zeros(args.multipv, dtype=np.float32)
            k = len(lp.move_indices)
            mi[:k] = lp.move_indices
            mp_[:k] = lp.move_probs
            move_idx.append(mi)
            move_probs.append(mp_)
            value_targets.append(lp.value_target)

            if i % max(1, len(positions) // 50) == 0 or i == len(positions):
                dt = time.time() - t0
                rate = i / dt
                eta = (len(positions) - i) / rate
                print(f"[lab] {i}/{len(positions)} | {rate:.1f} pos/s | ETA {eta:5.0f}s")

            if args.checkpoint_every > 0 and i % args.checkpoint_every == 0 and i < len(positions):
                _save(args.out, fens, move_idx, move_probs, value_targets)
                print(f"[ckp] partial save at {i}")

    _save(args.out, fens, move_idx, move_probs, value_targets)
    dt = time.time() - t0
    print(f"\n[done] {len(fens)} positions in {dt:.0f}s "
          f"({len(fens)/dt:.1f} pos/s, {1000*dt/len(fens):.1f} ms/pos)")
    print(f"[out] {args.out}  ({args.out.stat().st_size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
