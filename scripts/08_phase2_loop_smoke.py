"""Phase 2 mini-iteration smoke: self-play -> buffer -> train -> verify loss drop.

Runs the full Phase 2 inner loop on CPU with tiny knobs:
  1. Play 2 short self-play games per arm (random pgx net + JIT Arm B / Arm A)
  2. Push trajectories into a small ReplayBuffer
  3. Run 20 train_step_phase2 steps on uniform batches
  4. Print loss curve (should monotonically-ish drop)

Usage:
    .venv/bin/python scripts/08_phase2_loop_smoke.py
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pgx

from sampling_chess.buffer import ReplayBuffer
from sampling_chess.net import ChessTransformerPgx, count_params
from sampling_chess.search import MctsArmA
from sampling_chess.selfplay import make_arm_b_op, play_self_game
from sampling_chess.train import (
    init_train_state,
    make_optimizer,
    make_train_step_phase2,
)


def _init_random_net(seed: int = 0):
    model = ChessTransformerPgx(
        n_layers=2, d_model=128, n_heads=4, ffn_dim=256
    )  # tiny for CPU smoke
    dummy = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    params = model.init(jax.random.key(seed), dummy)["params"]
    return model, params


def main() -> int:
    print("[init] pgx env + tiny pgx net")
    env = pgx.make("chess")
    model, params = _init_random_net()
    print(f"[model] {count_params(params):,} params (tiny)")

    arm_a = MctsArmA(model=model, params=params, num_simulations=4)
    arm_b_op = make_arm_b_op(
        model, params, K=4, k_plies=2, beta=1.0, stratified=False,
        rng=np.random.default_rng(0), env=env,
    )

    rng = np.random.default_rng(123)
    buf = ReplayBuffer(capacity=2_000)

    print("\n[play] 1 short Arm A game (max 8 plies)")
    t0 = time.time()
    traj_a = play_self_game(
        op=arm_a.improve_at_state, env=env,
        max_plies=8, temperature_threshold=4, rng=rng,
    )
    added_a = buf.add_trajectory(traj_a)
    print(f"  +{added_a} transitions  |  {time.time()-t0:.1f}s")

    print("[play] 1 short Arm B game (max 8 plies)")
    t0 = time.time()
    traj_b = play_self_game(
        op=arm_b_op, env=env,
        max_plies=8, temperature_threshold=4, rng=rng,
    )
    added_b = buf.add_trajectory(traj_b)
    print(f"  +{added_b} transitions  |  {time.time()-t0:.1f}s")
    print(f"[buf] {len(buf)} transitions stored")

    print("\n[train] 20 train_step_phase2 steps on batch=8")
    optimizer = make_optimizer(lr=1e-3, warmup_steps=2, total_steps=20)
    state = init_train_state(model, params, optimizer)
    train_step = make_train_step_phase2(model, lambda_v=1.0)

    losses = []
    for step in range(20):
        batch = buf.sample(batch_size=8, rng=rng)
        jbatch = {k: jnp.asarray(v) for k, v in batch.items()}
        state, metrics = train_step(state, jbatch)
        losses.append(float(metrics["loss"]))
        if (step + 1) % 5 == 0:
            print(
                f"  step {step+1:3d} | loss {metrics['loss']:.4f} | "
                f"p {metrics['policy_loss']:.4f} | v {metrics['value_loss']:.4f} | "
                f"gnorm {metrics['grad_norm']:.3f}"
            )

    drop_pct = 100 * (losses[0] - losses[-1]) / losses[0]
    print(f"\n[done] loss {losses[0]:.4f} -> {losses[-1]:.4f}  ({drop_pct:+.1f}%)")
    if losses[-1] < losses[0]:
        print("  ✓ loss decreased")
    else:
        print("  [!!] loss did not decrease")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
