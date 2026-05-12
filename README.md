# Chess Bot via Supervised Learning

A supervised-learned chess transformer trained on Stockfish-labeled
positions from the Lichess Elite Database. Inspired by Ruoss et al. 2024
(*Grandmaster-Level Chess Without Search*) at smaller scale: target
~Elo 2200-2700, comfortable enough to beat any casual or intermediate
opponent.

## Layout

```
sampling_chess/
├── board.py       # python-chess wrappers: position->tensor, move->index
├── net.py         # Flax encoder-only ChessTransformer (policy + value heads)
├── stockfish.py   # Subprocess labeler + skill-capped opponent
├── data.py        # PGN streaming + position sampling with rating filter
├── train.py       # SL loss (policy CE + value MSE), optimizer, train_step
├── eval.py        # Match harness vs Stockfish (Wilson CI, Elo)
├── log.py         # Thin wandb wrapper (no-op when no API key)
└── __init__.py

scripts/
├── 02_label_batch.py    # Stockfish-label N positions from a PGN
├── 03_sl_train.py       # SL training driver with wandb + ckpt-every
├── 10_download_data.py  # Lichess Elite Database downloader + unzip
└── 11_play_vs_human.py  # Interactive UCI play loop vs the trained net

notebooks/
└── 02_colab_sl_bot.ipynb  # End-to-end Colab pipeline

tests/
├── test_board.py     test_data.py     test_eval.py
├── test_log.py       test_net.py      test_stockfish.py
└── test_train.py
```

## Pipeline

1. **Download data**:
   ```bash
   python scripts/10_download_data.py --out-dir data/
   # downloads + unzips lichess_elite_2025-11.zip -> data/lichess_elite_2025-11.pgn
   ```

2. **Label with Stockfish** (multipv=5, depth=12; ~5h on a 12-core CPU
   for 500k positions):
   ```bash
   python scripts/02_label_batch.py --source pgn \
       --pgn-path data/lichess_elite_2025-11.pgn --n 500000 \
       --out data/labels_elite_500k.npz \
       --workers 12 --depth 12 --multipv 5
   ```

3. **Train** (~5-8h on a Colab G4; doc-spec 16M-param transformer,
   100k steps, batch 1024):
   ```bash
   python scripts/03_sl_train.py \
       --data data/labels_elite_500k.npz --steps 100000 \
       --batch-size 1024 --lr 3e-4 --warmup 1000 \
       --ckpt-dir checkpoints/ --wandb
   ```
   See `notebooks/02_colab_sl_bot.ipynb` for the Colab-friendly version.

4. **Play**:
   ```bash
   python scripts/11_play_vs_human.py --ckpt checkpoints/ckpt_0100000.pkl
   # play as black:
   python scripts/11_play_vs_human.py --ckpt ... --human-color black
   # more variety in bot moves:
   python scripts/11_play_vs_human.py --ckpt ... --mode sample --temperature 0.5
   ```
   Type UCI moves at the prompt (e.g., `e2e4`, `e7e8q` for queen-promo,
   `e1g1` for white kingside castle). `quit` ends the game.

## Requirements

```bash
pip install -e ".[ml]"          # core + jax/flax/optax/wandb
apt install stockfish            # opponent + label engine (linux)
```

Tested on Python 3.14, JAX 0.10.

## Tests

```bash
pytest tests/    # 64 tests; ~90s with stockfish installed
```
