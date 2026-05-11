"""End-to-end Phase 2 mini smoke: 2 iters x (self-play + train + eval).

Exercises the complete inner loop on CPU with tiny knobs:
  - 2 iterations
  - 2 self-play games per iter (max 8 plies)
  - 15 train steps per iter (batch 4)
  - eval after each iteration: 2 games vs Stockfish skill 0

Usage:
    .venv/bin/python scripts/09_phase2_full_smoke.py
    .venv/bin/python scripts/09_phase2_full_smoke.py --arm a   # MCTS
    .venv/bin/python scripts/09_phase2_full_smoke.py --arm b   # sampling
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pgx

from sampling_chess.iter_driver import run_phase2
from sampling_chess.net import ChessTransformerPgx, count_params
from sampling_chess.search import MctsArmA, MctsArmABatched
from sampling_chess.selfplay import (
    make_arm_b_batched_op_builder,
    make_arm_b_op_builder,
)
from sampling_chess.train import make_optimizer


def _tiny_net():
    model = ChessTransformerPgx(
        n_layers=2, d_model=128, n_heads=4, ffn_dim=256
    )
    dummy = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    params = model.init(jax.random.key(0), dummy)["params"]
    return model, params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=["a", "b"], default="b")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--games-per-iter", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=15)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=2)
    parser.add_argument("--eval-skill", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="if set, save params per-iter; resume on next run if latest.pkl exists")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any existing checkpoint in --ckpt-dir")
    # Arm-specific knobs (default to doc-spec config; smoke can override).
    parser.add_argument("--K", type=int, default=100,
                        help="Arm B: number of trajectories per sample call")
    parser.add_argument("--k-plies", type=int, default=10,
                        help="Arm B: rollout depth per trajectory")
    parser.add_argument("--beta", type=float, default=5.0,
                        help="Arm B: SNIS sharpening parameter")
    parser.add_argument("--num-sims", type=int, default=100,
                        help="Arm A: mctx num_simulations per move")
    parser.add_argument("--no-batched", action="store_true",
                        help="disable vmap-over-games self-play (slower; for debugging)")
    args = parser.parse_args()

    print(f"[init] arm={args.arm}, iters={args.iters}, "
          f"games/iter={args.games_per_iter}, train_steps/iter={args.train_steps}")
    if args.arm == "a":
        print(f"[arm A] num_sims={args.num_sims}")
    else:
        print(f"[arm B] K={args.K}, k_plies={args.k_plies}, beta={args.beta}")
    env = pgx.make("chess")
    model, params = _tiny_net()
    print(f"[model] {count_params(params):,} params (tiny)")

    wandb_active = False
    if args.wandb:
        from sampling_chess import log as wandb_log
        wandb_active = wandb_log.init_run(
            project="sampling-chess",
            name=f"phase2-smoke-{args.arm}",
            group="phase2-mini-smoke",
            tags=["phase2", "smoke", f"arm-{args.arm}"],
            config=vars(args),
        )

    use_batched = not args.no_batched
    if use_batched:
        if args.arm == "a":
            arm_a_batched = MctsArmABatched(
                model=model, params=params,
                n_games=args.games_per_iter,
                num_simulations=args.num_sims, env=env,
            )
            def op_builder(p):
                arm_a_batched.params = p
                def op(states_batched, _key):
                    return arm_a_batched.improve_at_states(states_batched)
                return op
        else:
            op_builder = make_arm_b_batched_op_builder(
                model, K=args.K, k_plies=args.k_plies, beta=args.beta,
                n_games=args.games_per_iter,
                rng=np.random.default_rng(0), env=env,
            )
    else:
        if args.arm == "a":
            arm_a = MctsArmA(model=model, params=params,
                             num_simulations=args.num_sims)
            def op_builder(p):
                arm_a.params = p
                return arm_a.improve_at_state
        else:
            op_builder = make_arm_b_op_builder(
                model, K=args.K, k_plies=args.k_plies, beta=args.beta,
                rng=np.random.default_rng(0), env=env,
            )

    optimizer = make_optimizer(
        lr=1e-3, warmup_steps=2,
        total_steps=args.iters * args.train_steps,
    )

    t0 = time.time()
    state, history = run_phase2(
        op_builder, model, optimizer, params,
        n_iterations=args.iters,
        games_per_iter=args.games_per_iter,
        train_steps_per_iter=args.train_steps,
        batch_size=args.batch,
        buffer_capacity=10_000,
        env=env,
        eval_every=args.eval_every,
        eval_skills=(args.eval_skill,),
        eval_n_games=args.eval_games,
        eval_opponent_time=0.02,
        max_plies=args.max_plies,
        temperature_threshold=max(2, args.max_plies // 2),
        seed=0,
        wandb_active=wandb_active,
        ckpt_dir=args.ckpt_dir,
        resume=not args.no_resume,
        batched=use_batched,
    )
    total_t = time.time() - t0

    print(f"\n[done] {args.iters} iters in {total_t:.0f}s")
    losses = [h["train"]["loss"] for h in history]
    if len(losses) >= 2:
        drop = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"  loss across iters: {[f'{l:.3f}' for l in losses]} ({drop:+.1f}%)")

    if args.wandb and wandb_active:
        from sampling_chess import log as wandb_log
        wandb_log.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
