"""Phase 0 smoke test: random-policy self-play + eval vs Stockfish skill 0.

Does not touch any learned network — a uniform-random move policy plays the
role of the "random init net". Verifies that:

  1. Random vs random games complete with no illegal moves.
  2. Position encoding (planes / global / legal mask) round-trips on every
     visited position.
  3. The eval harness against Stockfish skill 0 returns a sensible result
     (random policy should lose nearly all games).

Run:
    .venv/bin/python scripts/00_smoke_random.py
"""

import sys
import time

import chess
import numpy as np

from sampling_chess import board as B
from sampling_chess import eval as E


def _verify_encoding(board: chess.Board) -> None:
    planes = B.board_to_planes(board)
    glob = B.board_to_global(board)
    mask = B.legal_action_mask(board)
    assert planes.shape == B.BOARD_SHAPE and planes.dtype == np.int8
    assert glob.shape == (B.NUM_GLOBAL_FEATURES,) and glob.dtype == np.float32
    assert mask.shape == (B.NUM_ACTIONS,)
    # Mask count must equal legal-move count.
    assert int(mask.sum()) == sum(1 for _ in board.legal_moves)
    # Every legal move's index round-trips.
    for mv in board.legal_moves:
        idx = B.move_to_index(mv)
        assert B.index_to_move(idx) == mv, f"round-trip failed for {mv}"


def smoke_self_play(n_games: int = 4, seed_base: int = 0) -> None:
    """Random-vs-random; verifies no illegal moves and encoding round-trips."""
    print(f"\n[1/2] Random-vs-random self-play ({n_games} games)")
    total_plies = 0
    outcomes = {chess.WHITE: 0, chess.BLACK: 0, None: 0}
    t0 = time.time()
    for i in range(n_games):
        white = E.make_random_policy(seed=seed_base + 2 * i)
        black = E.make_random_policy(seed=seed_base + 2 * i + 1)
        # Custom loop so we can verify encoding mid-game, not just at the end.
        board = chess.Board()
        plies = 0
        for _ in range(400):
            if board.is_game_over(claim_draw=True):
                break
            _verify_encoding(board)
            policy = white if board.turn == chess.WHITE else black
            mv = policy(board)
            assert mv in board.legal_moves
            board.push(mv)
            plies += 1
        winner = board.outcome(claim_draw=True).winner if board.outcome(claim_draw=True) else None
        outcomes[winner] += 1
        total_plies += plies
        result_str = {chess.WHITE: "white", chess.BLACK: "black", None: "draw"}[winner]
        print(f"  game {i}: {result_str} in {plies} plies")
    dt = time.time() - t0
    print(f"  -> {outcomes[chess.WHITE]}W/{outcomes[None]}D/{outcomes[chess.BLACK]}L (white POV), "
          f"avg {total_plies/n_games:.0f} plies, {dt:.2f}s")


def smoke_vs_stockfish(n_games: int = 4) -> bool:
    print(f"\n[2/2] Random vs Stockfish skill 0 ({n_games} games)")
    p = E.make_random_policy(seed=42)
    t0 = time.time()
    result = E.play_match(
        p, opponent_skill=0, n_games=n_games,
        opponent_time=0.05, max_plies=300, seed=0,
    )
    dt = time.time() - t0
    print(f"  {result.summary()}")
    print(f"  wall-clock: {dt:.1f}s for {n_games} games")

    # Sanity bound: random-uniform should lose almost everything to even skill 0.
    if result.losses < n_games // 2:
        print(f"  FAIL: expected losses >= {n_games // 2}, got {result.losses}")
        return False
    print("  OK: random policy loses majority as expected")
    return True


def main() -> int:
    print("=" * 60)
    print("Phase 0 smoke test: random-policy self-play + eval")
    print("=" * 60)
    smoke_self_play(n_games=4)
    ok = smoke_vs_stockfish(n_games=4)
    print("\n" + "=" * 60)
    print("DONE." if ok else "FAILED.")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
