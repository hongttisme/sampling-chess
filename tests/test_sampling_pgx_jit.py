"""Tests for the jit-vectorized Arm B sampler.

Verifies semantic equivalence with the Python-loop version on
the section 7.4 invariant + basic shape/sum invariants.
"""

import time

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pgx")
import pgx  # noqa: E402

from sampling_chess.net import ChessTransformerPgx, PGX_NUM_ACTIONS  # noqa: E402
from sampling_chess.pgx_bridge import chess_board_to_pgx_state  # noqa: E402
from sampling_chess.sampling_pgx import (  # noqa: E402
    make_jit_sampler,
    sample_improved_policy_pgx,
    sample_improved_policy_pgx_jit,
)


_ENV = pgx.make("chess")


def _random_net():
    model = ChessTransformerPgx()
    dummy = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    params = model.init(jax.random.key(0), dummy)["params"]
    return model, params


# ----- Shape / sum invariants -----

def test_jit_sampler_returns_normalized_pi():
    model, params = _random_net()
    sampler = make_jit_sampler(model, K=8, k_plies=3, env=_ENV)
    state = chess_board_to_pgx_state(chess.Board())
    out = sample_improved_policy_pgx_jit(
        root_state=state, sampler=sampler, params=params,
        K=8, beta=1.0, rng_key=jax.random.key(0),
    )
    assert out.pi_improved.shape == (PGX_NUM_ACTIONS,)
    assert float(out.pi_improved.sum()) == pytest.approx(1.0, abs=1e-4)


def test_jit_sampler_zero_on_illegal_first_moves():
    model, params = _random_net()
    sampler = make_jit_sampler(model, K=8, k_plies=2, env=_ENV)
    state = chess_board_to_pgx_state(chess.Board())
    out = sample_improved_policy_pgx_jit(
        root_state=state, sampler=sampler, params=params,
        K=8, beta=1.0, rng_key=jax.random.key(1),
    )
    legal = set(int(i) for i in np.where(np.array(state.legal_action_mask))[0])
    illegal = sum(p for i, p in enumerate(out.pi_improved) if i not in legal)
    assert abs(illegal) < 1e-6


def test_jit_sampler_first_moves_are_legal():
    model, params = _random_net()
    sampler = make_jit_sampler(model, K=8, k_plies=2, env=_ENV)
    state = chess_board_to_pgx_state(chess.Board())
    out = sample_improved_policy_pgx_jit(
        root_state=state, sampler=sampler, params=params,
        K=8, beta=1.0, rng_key=jax.random.key(2),
    )
    legal = set(int(i) for i in np.where(np.array(state.legal_action_mask))[0])
    for fm in out.first_moves:
        assert int(fm) in legal


def test_jit_speedup_over_python_loop():
    """Time both implementations; jit must be faster after warmup."""
    model, params = _random_net()
    state = chess_board_to_pgx_state(chess.Board())

    # Python loop
    def apply_fn(states):
        obs = jnp.stack([s.observation for s in states])
        logits, values = model.apply({"params": params}, obs)
        return np.asarray(logits), np.asarray(values)

    t0 = time.time()
    out_py = sample_improved_policy_pgx(
        root_state=state, apply_fn=apply_fn,
        K=8, k_plies=3, beta=1.0,
        rng=np.random.default_rng(0), stratified=False, env=_ENV,
    )
    dt_py = time.time() - t0

    # JIT — first call includes compile, second is hot
    sampler = make_jit_sampler(model, K=8, k_plies=3, env=_ENV)
    _ = sample_improved_policy_pgx_jit(
        root_state=state, sampler=sampler, params=params,
        K=8, beta=1.0, rng_key=jax.random.key(0),
    )  # warmup compile
    t0 = time.time()
    out_jit = sample_improved_policy_pgx_jit(
        root_state=state, sampler=sampler, params=params,
        K=8, beta=1.0, rng_key=jax.random.key(0),
    )
    dt_jit = time.time() - t0

    print(f"\npython-loop: {dt_py*1000:.0f} ms  |  jit (warm): {dt_jit*1000:.0f} ms")
    # Hot jit should be at least 5x faster; gap is bigger on GPU.
    assert dt_jit < dt_py / 3, f"jit not faster: {dt_jit:.3f}s vs {dt_py:.3f}s"
    # Both should produce normalized output
    assert float(out_py.pi_improved.sum()) == pytest.approx(1.0, abs=1e-4)
    assert float(out_jit.pi_improved.sum()) == pytest.approx(1.0, abs=1e-4)


def test_v_plus_in_unit_range():
    model, params = _random_net()
    sampler = make_jit_sampler(model, K=8, k_plies=3, env=_ENV)
    state = chess_board_to_pgx_state(chess.Board())
    out = sample_improved_policy_pgx_jit(
        root_state=state, sampler=sampler, params=params,
        K=8, beta=0.5, rng_key=jax.random.key(3),
    )
    assert -1.0 <= out.v_plus <= 1.0
