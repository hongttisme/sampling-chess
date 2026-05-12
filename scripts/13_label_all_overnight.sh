#!/bin/bash
# Label all three Lichess Elite months (Sep + Oct + Nov 2025) sequentially
# and merge into a single combined .npz for SL training.
#
# Total wall-clock: ~15h on a 12-core CPU @ 47ms/pos. Run overnight.
# Re-runnable: each month overwrites its own .npz, partial saves preserved
# at every 5000 positions during a single label batch.
#
# Usage:
#     bash scripts/13_label_all_overnight.sh
#     # waits ~15h then prints a summary
#
# Prereq: data/lichess_elite_2025-{09,10,11}.pgn must exist
# (run scripts/10_download_data.py first if missing).

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY=.venv/bin/python
DATA=data
WORKERS=12
DEPTH=12
MULTIPV=5
N_MAX=500000  # caps per-month; Elite month typically yields 200-300k

months=(11 10 09)
out_files=()

for m in "${months[@]}"; do
    pgn="$DATA/lichess_elite_2025-${m}.pgn"
    out="$DATA/labels_elite_2025-${m}.npz"
    if [ ! -f "$pgn" ]; then
        echo "[err] missing $pgn — run scripts/10_download_data.py with --url for that month"
        exit 1
    fi
    if [ -f "$out" ]; then
        echo "[skip] $out already exists; delete it to relabel"
    else
        echo "[label] 2025-${m}: pgn=$pgn -> $out"
        $PY scripts/02_label_batch.py --source pgn \
            --pgn-path "$pgn" --n "$N_MAX" \
            --out "$out" \
            --workers "$WORKERS" --depth "$DEPTH" --multipv "$MULTIPV"
    fi
    out_files+=("$out")
done

merged="$DATA/labels_elite_combined.npz"
echo "[merge] -> $merged"
$PY scripts/12_merge_labels.py --inputs "${out_files[@]}" --out "$merged"

echo
echo "[done] all months labeled + merged"
echo "  next: upload $merged to Drive, then run notebooks/02_colab_sl_bot.ipynb"
