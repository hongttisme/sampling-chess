"""Tests for sampling_pgx.py (Arm B with pgx state + 4672 action space)."""

import numpy as np
import pytest

pytest.importorskip("pgx")
import pgx  # noqa: E402

from sampling_chess import sampling_pgx as SP  # noqa: E402
from sampling_chess.pgx_bridge import (  # noqa: E402
    chess_board_to_pgx_state,
    PGX_NUM_ACTIONS,
)
import chess  # noqa: E402


_ENV = pgx.make("chess")


def uniform_apply_fn(states):
    n = len(states)
    return (
        np.zeros((n, PGX_NUM_ACTIONS), dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    )


def constant_value_apply_fn(value: float):
    def _fn(states):
        n = len(states)
        return (
            np.zeros((n, PGX_NUM_ACTIONS), dtype=np.float32),
            np.full(n, value, dtype=np.float32),
        )
    return _fn


# ----- Section 7.4 invariant -----

def test_uniform_pi_uniform_v_uniform_target_stratified():
    state = chess_board_to_pgx_state(chess.Board())
    legal = np.where(np.array(state.legal_action_mask))[0]
    n_legal = len(legal)
    K = 4 * n_legal

    rng = np.random.default_rng(0)
    out = SP.sample_improved_policy_pgx(
        root_state=state, apply_fn=uniform_apply_fn,
        K=K, k_plies=4, beta=1.0, rng=rng, stratified=True, env=_ENV,
    )
    expected = 1.0 / n_legal
    on_legal = out.pi_improved[legal]
    np.testing.assert_allclose(on_legal, expected, atol=1e-6)
    illegal_mass = out.pi_improved.sum() - on_legal.sum()
    assert abs(illegal_mass) < 1e-6


# ----- Output invariants -----

def test_pi_improved_sums_to_one():
    state = chess_board_to_pgx_state(chess.Board())
    rng = np.random.default_rng(0)
    out = SP.sample_improved_policy_pgx(
        root_state=state, apply_fn=uniform_apply_fn,
        K=20, k_plies=2, beta=0.5, rng=rng, stratified=True, env=_ENV,
    )
    assert out.pi_improved.sum() == pytest.approx(1.0, abs=1e-5)


def test_pi_improved_zero_on_illegal_first_moves():
    state = chess_board_to_pgx_state(chess.Board())
    rng = np.random.default_rng(0)
    out = SP.sample_improved_policy_pgx(
        root_state=state, apply_fn=uniform_apply_fn,
        K=20, k_plies=2, beta=1.0, rng=rng, stratified=True, env=_ENV,
    )
    legal = set(int(i) for i in np.where(np.array(state.legal_action_mask))[0])
    illegal_mass = sum(p for i, p in enumerate(out.pi_improved) if i not in legal)
    assert abs(illegal_mass) < 1e-6


def test_first_moves_are_legal():
    state = chess_board_to_pgx_state(chess.Board())
    rng = np.random.default_rng(0)
    out = SP.sample_improved_policy_pgx(
        root_state=state, apply_fn=uniform_apply_fn,
        K=10, k_plies=2, beta=1.0, rng=rng, stratified=True, env=_ENV,
    )
    legal = set(int(i) for i in np.where(np.array(state.legal_action_mask))[0])
    for fm in out.first_moves:
        assert int(fm) in legal


# ----- Side-to-move parity (uses real pgx step) -----

def test_v_plus_stratified_constant_value():
    """k_plies=2: leaf player == root player after 2 steps -> no flip.
    Constant value 0.4 should propagate to v_plus ~ 0.4."""
    state = chess_board_to_pgx_state(chess.Board())
    rng = np.random.default_rng(0)
    out = SP.sample_improved_policy_pgx(
        root_state=state, apply_fn=constant_value_apply_fn(0.4),
        K=10, k_plies=2, beta=0.0, rng=rng, stratified=True, env=_ENV,
    )
    assert out.v_plus == pytest.approx(0.4, abs=0.05)


def test_v_plus_one_ply_flips():
    """k_plies=1: leaf player != root player -> flip."""
    state = chess_board_to_pgx_state(chess.Board())
    rng = np.random.default_rng(0)
    out = SP.sample_improved_policy_pgx(
        root_state=state, apply_fn=constant_value_apply_fn(0.5),
        K=10, k_plies=1, beta=0.0, rng=rng, stratified=True, env=_ENV,
    )
    assert out.v_plus == pytest.approx(-0.5, abs=1e-5)


# ----- Edge cases -----

def test_invalid_K_raises():
    state = chess_board_to_pgx_state(chess.Board())
    with pytest.raises(ValueError):
        SP.sample_improved_policy_pgx(
            root_state=state, apply_fn=uniform_apply_fn,
            K=0, k_plies=2, beta=1.0, env=_ENV,
        )


def test_terminal_root_returns_immediately():
    """Stalemate at root: no legal moves, returns 0."""
    bd = chess.Board("8/8/8/8/8/kq6/8/K7 w - - 0 1")
    assert bd.is_stalemate()
    state = chess_board_to_pgx_state(bd)
    out = SP.sample_improved_policy_pgx(
        root_state=state, apply_fn=uniform_apply_fn,
        K=4, k_plies=2, beta=1.0, env=_ENV,
    )
    assert out.pi_improved.sum() == 0.0
