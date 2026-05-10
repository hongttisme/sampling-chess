"""Throughput benchmark for StockfishPool — calibrates Phase 1 wall-clock.

Generates `--n-positions` positions via random self-play (mid-game cuts), runs
them through the pool at the chosen depth/multipv, and reports ms/position
plus a projected wall-clock for the full Phase 1 2M labeling job.

Usage:
    .venv/bin/python scripts/01_benchmark_labeler.py \\
        --n-positions 100 --n-workers 6 --depth 12 --multipv 5
"""

import argparse
import json
import os
import sys
from pprint import pprint

from sampling_chess.stockfish import benchmark_pool


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-positions", type=int, default=100,
                   help="positions to label in the benchmark")
    p.add_argument("--n-workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 2),
                   help="parallel Stockfish processes (default: ncpu - 2)")
    p.add_argument("--depth", type=int, default=12,
                   help="Stockfish search depth (Phase 1 default: 12)")
    p.add_argument("--multipv", type=int, default=5,
                   help="MultiPV (Phase 1 default: 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true",
                   help="emit results as a single JSON line (for log scraping)")
    args = p.parse_args()

    print(f"Benchmarking labeler: {args.n_positions} positions, "
          f"{args.n_workers} workers, depth {args.depth}, multipv {args.multipv}")
    print("(spawn pool startup may take a few seconds)\n")

    stats = benchmark_pool(
        n_positions=args.n_positions,
        n_workers=args.n_workers,
        depth=args.depth,
        multipv=args.multipv,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(stats))
    else:
        print("Results:")
        pprint(stats, sort_dicts=False)
        print()
        print(f"  -> ms / position : {stats['ms_per_position']:>8.1f}")
        print(f"  -> pos / sec     : {stats['positions_per_sec']:>8.1f}")
        print(f"  -> 2M positions  : ~{stats['projected_2M_hours']:>5.1f}h "
              f"on this machine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
