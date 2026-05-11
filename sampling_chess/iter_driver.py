"""Phase 2 iteration driver: alternates self-play, train, and periodic eval.

One iteration:
  1. Generate `games_per_iter` self-play games with the current net + arm.
  2. Append trajectories to a ReplayBuffer.
  3. Run `train_steps_per_iter` train_step_phase2 steps on uniform batches.
  4. Optionally eval the (greedy) updated net vs Stockfish at each opponent
     skill in `eval_skills` for `eval_n_games` games each.

The arm operator captures `params` at build time, so when params change after
training we rebuild the op via `op_builder(params)` for the next iteration.
For MctsArmA you can equivalently reassign `arm.params = new_params`.

Metrics are returned as a dict per iteration; pass `wandb_run=True` to also
log via sampling_chess.log.
"""

import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

import pgx

from sampling_chess import log as wandb_log
from sampling_chess.buffer import ReplayBuffer
from sampling_chess.eval import (
    make_pgx_greedy_policy,
    play_pgx_match,
)
from sampling_chess.selfplay import play_self_game
from sampling_chess.train import (
    init_train_state,
    make_train_step_phase2,
)


def phase2_iter(
    op_builder: Callable,            # params -> (state -> result)
    model,
    train_state,
    buffer: ReplayBuffer,
    rng: np.random.Generator,
    *,
    games_per_iter: int = 8,
    train_steps_per_iter: int = 20,
    batch_size: int = 8,
    env=None,
    max_plies: int = 200,
    temperature_threshold: int = 30,
) -> tuple:
    """One Phase 2 iteration. Returns (new_train_state, metrics_dict)."""
    if env is None:
        env = pgx.make("chess")

    # 1) Self-play with the CURRENT net (pre-train params).
    arm_op = op_builder(train_state.params)
    sp_t0 = time.time()
    n_white_wins = n_black_wins = n_draws = 0
    total_plies = 0
    for _ in range(games_per_iter):
        traj = play_self_game(
            arm_op, env=env,
            max_plies=max_plies, temperature_threshold=temperature_threshold,
            rng=rng,
        )
        buffer.add_trajectory(traj)
        total_plies += traj.plies
        rw = float(traj.outcome_per_player[0])
        if rw > 0:
            n_white_wins += 1
        elif rw < 0:
            n_black_wins += 1
        else:
            n_draws += 1
    sp_dt = time.time() - sp_t0

    # 2) Train.
    train_step = make_train_step_phase2(model, lambda_v=1.0)
    tr_t0 = time.time()
    acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "grad_norm": 0.0}
    n_train = 0
    if len(buffer) >= batch_size:
        for _ in range(train_steps_per_iter):
            batch = buffer.sample(batch_size, rng)
            jbatch = {k: jnp.asarray(v) for k, v in batch.items()}
            train_state, metrics = train_step(train_state, jbatch)
            for k, v in metrics.items():
                acc[k] += float(v)
            n_train += 1
    tr_dt = time.time() - tr_t0
    avg_train = {k: (v / max(1, n_train)) for k, v in acc.items()}

    metrics = {
        "selfplay": {
            "games": games_per_iter,
            "total_plies": total_plies,
            "avg_plies": total_plies / max(1, games_per_iter),
            "white_wins": n_white_wins,
            "black_wins": n_black_wins,
            "draws": n_draws,
            "wall_clock_sec": sp_dt,
        },
        "train": {
            **avg_train,
            "n_steps": n_train,
            "wall_clock_sec": tr_dt,
        },
        "buffer_size": len(buffer),
    }
    return train_state, metrics


def evaluate_vs_stockfish(
    model, params,
    skills: list,
    n_games_per_skill: int = 20,
    opponent_time: float = 0.05,
    max_plies: int = 200,
    env=None,
    seed: Optional[int] = None,
) -> dict:
    """Greedy-policy eval at each opponent skill. Returns {skill: MatchResult}."""
    if env is None:
        env = pgx.make("chess")
    policy_fn = make_pgx_greedy_policy(model, params)
    out = {}
    for skill in skills:
        result = play_pgx_match(
            policy_fn, opponent_skill=skill,
            n_games=n_games_per_skill,
            opponent_time=opponent_time,
            max_plies=max_plies,
            env=env, seed=seed,
        )
        out[skill] = result
    return out


def _emit_log(iter_idx: int, m: dict, eval_results: Optional[dict],
              wandb_active: bool) -> None:
    """Pretty-print + (optionally) wandb-log per-iter metrics."""
    sp = m["selfplay"]
    tr = m["train"]
    print(
        f"[iter {iter_idx:3d}] "
        f"selfplay {sp['games']}g/{sp['total_plies']}p "
        f"({sp['white_wins']}W/{sp['draws']}D/{sp['black_wins']}L) "
        f"in {sp['wall_clock_sec']:.0f}s | "
        f"train loss={tr['loss']:.4f} (p={tr['policy_loss']:.4f} v={tr['value_loss']:.4f}) "
        f"steps={tr['n_steps']} in {tr['wall_clock_sec']:.0f}s | "
        f"buf={m['buffer_size']}"
    )
    if wandb_active:
        flat = {
            "iter": iter_idx,
            "selfplay/games": sp["games"],
            "selfplay/avg_plies": sp["avg_plies"],
            "selfplay/white_wins": sp["white_wins"],
            "selfplay/black_wins": sp["black_wins"],
            "selfplay/draws": sp["draws"],
            "selfplay/wall_clock_sec": sp["wall_clock_sec"],
            "train/loss": tr["loss"],
            "train/policy_loss": tr["policy_loss"],
            "train/value_loss": tr["value_loss"],
            "train/grad_norm": tr["grad_norm"],
            "train/wall_clock_sec": tr["wall_clock_sec"],
            "buffer_size": m["buffer_size"],
        }
        wandb_log.log(flat, step=iter_idx)
    if eval_results:
        for skill, r in eval_results.items():
            print(
                f"  eval vs SF skill {skill}: {r.summary()}"
            )
            if wandb_active:
                wandb_log.log(
                    {
                        f"eval/skill{skill}/score": r.score,
                        f"eval/skill{skill}/elo": r.elo(),
                        f"eval/skill{skill}/wins": r.wins,
                        f"eval/skill{skill}/draws": r.draws,
                        f"eval/skill{skill}/losses": r.losses,
                    },
                    step=iter_idx,
                )


def run_phase2(
    op_builder: Callable,
    model,
    optimizer,
    init_params,
    *,
    n_iterations: int,
    games_per_iter: int = 8,
    train_steps_per_iter: int = 20,
    batch_size: int = 8,
    buffer_capacity: int = 10_000,
    env=None,
    eval_every: int = 5,
    eval_skills: tuple = (0, 3),
    eval_n_games: int = 8,
    eval_opponent_time: float = 0.05,
    max_plies: int = 200,
    temperature_threshold: int = 30,
    seed: int = 0,
    wandb_active: bool = False,
):
    """Run N Phase 2 iterations; return final TrainState + per-iter metrics list."""
    if env is None:
        env = pgx.make("chess")
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    state = init_train_state(model, init_params, optimizer)

    history = []
    for it in range(1, n_iterations + 1):
        state, m = phase2_iter(
            op_builder, model, state, buffer, rng,
            games_per_iter=games_per_iter,
            train_steps_per_iter=train_steps_per_iter,
            batch_size=batch_size,
            env=env, max_plies=max_plies,
            temperature_threshold=temperature_threshold,
        )
        eval_results = None
        if eval_every > 0 and it % eval_every == 0:
            eval_results = evaluate_vs_stockfish(
                model, state.params,
                skills=list(eval_skills),
                n_games_per_skill=eval_n_games,
                opponent_time=eval_opponent_time,
                max_plies=max_plies,
                env=env, seed=seed + it,
            )
        _emit_log(it, m, eval_results, wandb_active)
        history.append({"iter": it, **m, "eval": eval_results})
    return state, history
