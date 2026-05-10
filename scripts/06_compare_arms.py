"""Head-to-head smoke: Arm A (mctx) vs Arm B (sampling_pgx) at the same
positions, with the same random-init pgx net. Reports each arm's
improvement target on the top-5 legal moves and the v_plus estimate.

Usage:
    .venv/bin/python scripts/06_compare_arms.py
"""

import sys

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pgx

from sampling_chess.net import ChessTransformerPgx
from sampling_chess.pgx_bridge import (
    chess_board_to_pgx_state,
    pgx_action_to_chess_move,
)
from sampling_chess.sampling_pgx import sample_improved_policy_pgx
from sampling_chess.search import MctsArmA


def _init_random_net():
    model = ChessTransformerPgx()
    dummy = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    params = model.init(jax.random.key(0), dummy)["params"]
    return model, params


def _make_apply_fn(model, params):
    def apply_fn(states):
        obs = jnp.stack([s.observation for s in states])
        logits, values = model.apply({"params": params}, obs)
        return np.asarray(logits), np.asarray(values)
    return apply_fn


def _print_top5(name: str, weights: np.ndarray, mask: np.ndarray, turn):
    legal = np.where(mask)[0]
    top = sorted(legal, key=lambda i: -float(weights[i]))[:5]
    pieces = []
    for i in top:
        uci = pgx_action_to_chess_move(int(i), turn).uci()
        pieces.append(f"{uci}={weights[i]:.3f}")
    print(f"  {name:6s} top5: " + ", ".join(pieces))


def main() -> int:
    print("[init] pgx env + random pgx net")
    env = pgx.make("chess")
    model, params = _init_random_net()

    arm_a = MctsArmA(model=model, params=params, num_simulations=16)
    apply_fn = _make_apply_fn(model, params)
    rng = np.random.default_rng(0)

    bd_after_e4 = chess.Board()
    bd_after_e4.push_san("e4")

    test_positions = [
        ("startpos (white to move)", chess.Board()),
        ("after 1.e4 (black to move)", bd_after_e4),
    ]

    for name, bd in test_positions:
        print(f"\n=== {name} ===")
        state = chess_board_to_pgx_state(bd)
        mask = np.array(state.legal_action_mask)
        n_legal = int(mask.sum())
        print(f"  legal first moves: {n_legal}, turn: {'WHITE' if bd.turn else 'BLACK'}")

        a_out = arm_a.improve_at(bd)
        b_out = sample_improved_policy_pgx(
            root_state=state, apply_fn=apply_fn,
            K=16, k_plies=4, beta=1.0, rng=rng, stratified=True, env=env,
        )

        _print_top5("Arm A", a_out.pi_improved, mask, bd.turn)
        _print_top5("Arm B", b_out.pi_sample, mask, bd.turn)
        print(f"  v_plus:  A={a_out.v_plus:+.3f}   B={b_out.v_plus:+.3f}")
        print(f"  Arm A visit_counts.sum() = {int(a_out.visit_counts.sum())}")
        print(f"  Arm B legal-mass = {float(b_out.pi_sample[mask].sum()):.4f}")

    print("\n[done] both arms produced valid distributions on legal first moves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
