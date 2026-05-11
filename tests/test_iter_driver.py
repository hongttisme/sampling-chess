"""Tests for the Phase 2 iteration driver."""

import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pgx")
import pgx  # noqa: E402

from sampling_chess.buffer import ReplayBuffer  # noqa: E402
from sampling_chess.iter_driver import phase2_iter, run_phase2  # noqa: E402
from sampling_chess.net import ChessTransformerPgx  # noqa: E402
from sampling_chess.search import MctsArmA  # noqa: E402
from sampling_chess.selfplay import make_arm_b_op_builder  # noqa: E402
from sampling_chess.train import init_train_state, make_optimizer  # noqa: E402


_NO_STOCKFISH = shutil.which("stockfish") is None and not any(
    shutil.which(p) for p in ("/usr/games/stockfish", "/usr/local/bin/stockfish")
)
no_stockfish = pytest.mark.skipif(_NO_STOCKFISH, reason="stockfish not on PATH")

_ENV = pgx.make("chess")


def _tiny_net(seed: int = 0):
    model = ChessTransformerPgx(
        n_layers=2, d_model=64, n_heads=4, ffn_dim=128
    )
    dummy = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    params = model.init(jax.random.key(seed), dummy)["params"]
    return model, params


# ----- Single iteration -----

def test_phase2_iter_arm_b_runs_end_to_end():
    """One iter with Arm B: trajectories generated, buffer grows, train_state updates."""
    model, params = _tiny_net()
    op_builder = make_arm_b_op_builder(
        model, K=4, k_plies=2, beta=1.0,
        rng=np.random.default_rng(0), env=_ENV,
    )
    optimizer = make_optimizer(lr=1e-3, warmup_steps=2, total_steps=20)
    state = init_train_state(model, params, optimizer)
    buf = ReplayBuffer(capacity=200)
    rng = np.random.default_rng(0)

    new_state, metrics = phase2_iter(
        op_builder, model, state, buf, rng,
        games_per_iter=2, train_steps_per_iter=5, batch_size=4,
        env=_ENV, max_plies=6, temperature_threshold=3,
    )
    assert metrics["selfplay"]["games"] == 2
    assert metrics["selfplay"]["total_plies"] > 0
    assert metrics["train"]["n_steps"] == 5
    assert metrics["buffer_size"] > 0
    # Train state moved (step counter incremented at least 5)
    assert int(new_state.step) >= 5


def test_phase2_iter_arm_a_runs():
    """One iter with Arm A. arm_a.params is reassigned per-iter via op_builder."""
    model, params = _tiny_net()
    arm_a = MctsArmA(model=model, params=params, num_simulations=4)

    def op_builder(p):
        arm_a.params = p
        return arm_a.improve_at_state

    optimizer = make_optimizer(lr=1e-3, warmup_steps=2, total_steps=10)
    state = init_train_state(model, params, optimizer)
    buf = ReplayBuffer(capacity=100)
    rng = np.random.default_rng(0)

    new_state, metrics = phase2_iter(
        op_builder, model, state, buf, rng,
        games_per_iter=1, train_steps_per_iter=3, batch_size=2,
        env=_ENV, max_plies=4, temperature_threshold=2,
    )
    assert metrics["selfplay"]["games"] == 1
    assert metrics["train"]["n_steps"] == 3


# ----- Multi-iteration loop -----

def test_run_phase2_two_iters_no_eval():
    """run_phase2 with eval_every=0 (skip eval) completes 2 iterations."""
    model, params = _tiny_net()
    op_builder = make_arm_b_op_builder(
        model, K=4, k_plies=2, beta=1.0,
        rng=np.random.default_rng(1), env=_ENV,
    )
    optimizer = make_optimizer(lr=1e-3, warmup_steps=2, total_steps=10)

    state, history = run_phase2(
        op_builder, model, optimizer, params,
        n_iterations=2,
        games_per_iter=1, train_steps_per_iter=3, batch_size=2,
        buffer_capacity=50, env=_ENV,
        eval_every=0,
        max_plies=4, temperature_threshold=2,
        seed=0,
    )
    assert len(history) == 2
    assert history[0]["iter"] == 1
    assert history[1]["iter"] == 2
    assert int(state.step) >= 6  # 2 iters x 3 steps each
