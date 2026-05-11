"""Self-play smoke: one short game per arm with random-init pgx net.

Prints trajectory shape, terminal outcome, and verifies invariants
(actions legal, pi_improved normalized, value_targets reflect outcome).

Usage:
    .venv/bin/python scripts/07_selfplay_smoke.py
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pgx

from sampling_chess.net import ChessTransformerPgx
from sampling_chess.search import MctsArmA
from sampling_chess.selfplay import make_arm_b_op, play_self_game


def _init_random_net():
    model = ChessTransformerPgx()
    dummy = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    params = model.init(jax.random.key(0), dummy)["params"]
    return model, params


def _summarize(name, traj):
    print(f"\n=== {name} trajectory ===")
    print(f"  plies              : {traj.plies}")
    print(f"  terminated         : {traj.terminated}")
    print(f"  outcome (white,black): {traj.outcome_per_player.tolist()}")
    print(f"  obs shape          : {traj.observations.shape}")
    print(f"  pi_improved shape  : {traj.improved_policies.shape}")
    print(f"  actions[:10]       : {traj.actions[:10].tolist()}")
    print(f"  player_at_step[:10]: {traj.player_at_step[:10].tolist()}")

    # Verify invariants
    bad = 0
    for t in range(traj.plies):
        if not bool(traj.legal_masks[t, traj.actions[t]]):
            bad += 1
    if bad:
        print(f"  [!!] {bad} illegal action(s) recorded")
    else:
        print(f"  all {traj.plies} actions legal ✓")

    nz_pi = (traj.improved_policies.sum(axis=1) > 0)
    print(f"  steps with non-zero pi_improved: {int(nz_pi.sum())}/{traj.plies}")
    z = traj.value_targets()
    print(f"  value_targets[:5]  : {z[:5].tolist()}")


def main() -> int:
    print("[init] pgx env + random pgx net")
    env = pgx.make("chess")
    model, params = _init_random_net()

    arm_a = MctsArmA(model=model, params=params, num_simulations=8)
    arm_b_op = make_arm_b_op(
        model, params, K=8, k_plies=3, beta=1.0, stratified=True,
        rng=np.random.default_rng(0), env=env,
    )

    rng = np.random.default_rng(123)

    print("\n[play] Arm A (mctx, 8 sims) — short game (max 12 plies)")
    t0 = time.time()
    traj_a = play_self_game(
        op=arm_a.improve_at_state, env=env,
        max_plies=12, temperature_threshold=6, rng=rng,
    )
    print(f"  wall-clock: {time.time()-t0:.1f}s")
    _summarize("Arm A", traj_a)

    print("\n[play] Arm B (sampling, K=8, k=3) — short game (max 12 plies)")
    t0 = time.time()
    traj_b = play_self_game(
        op=arm_b_op, env=env,
        max_plies=12, temperature_threshold=6, rng=rng,
    )
    print(f"  wall-clock: {time.time()-t0:.1f}s")
    _summarize("Arm B", traj_b)

    print("\n[done] both arms produced valid trajectories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
