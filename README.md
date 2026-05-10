# Sampling-Based Policy Improvement vs MCTS in Self-Play Chess

Comparison of two policy improvement operators inside an AlphaZero-style self-play loop:

- **Arm A (baseline):** MCTS via DeepMind `mctx`.
- **Arm B (ours):** Importance-weighted Monte Carlo sampling of *k*-ply trajectories with truncated value-head bootstrap, inspired by *Reasoning with Sampling*.

Both arms share the same network and the same SL bootstrap; compared at equal total network forwards.

See `chess_sampling_vs_mcts_guide.pdf` for the full experimental design.

## Layout

```
sampling_chess/
├── board.py       # python-chess wrappers; position→tensor, move→index
├── net.py         # Flax encoder-only transformer with policy + value heads
├── search.py      # Arm A: MCTS via mctx
├── sampling.py    # Arm B: Algorithm 1 (importance-weighted sampling)
├── selfplay.py    # Self-play loop + replay buffer
├── train.py       # SL + RL training step, losses
├── stockfish.py   # Subprocess pool + non-blocking eval queue
└── eval.py        # Match harness vs Stockfish, win-rate, Elo with CI

tests/             # Unit tests (incl. §7.4: uniform π + zero V → uniform target)
scripts/           # Phase entry points
notebooks/         # Colab notebooks
```

## Status

Phase 0 — infrastructure.
