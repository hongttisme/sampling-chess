"""Tests for sampling_chess.mcts — verifies tree mechanics + invariants
work with the real ChessTransformer (random init, no training needed)."""

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sampling_chess import board as B
from sampling_chess.mcts import (
    MctsResult,
    benchmark_per_sim,
    make_eval_fn,
    mcts_search,
)
from sampling_chess.net import ChessTransformer


def _tiny_net_and_params():
    model = ChessTransformer(n_layers=2, d_model=64, n_heads=4, ffn_dim=128)
    pieces = jnp.zeros((1, 8, 8), dtype=jnp.int32)
    globals_ = jnp.zeros((1, 9), dtype=jnp.float32)
    params = model.init(jax.random.key(0), pieces, globals_)["params"]
    return model, params


# ----- Eval fn -----

def test_eval_fn_returns_correct_shapes():
    model, params = _tiny_net_and_params()
    eval_fn = make_eval_fn(model, params)
    masked, v = eval_fn(chess.Board())
    assert masked.shape == (B.NUM_ACTIONS,)
    assert isinstance(v, float)
    assert -1.0 <= v <= 1.0


def test_eval_fn_masks_illegals():
    model, params = _tiny_net_and_params()
    eval_fn = make_eval_fn(model, params)
    masked, _ = eval_fn(chess.Board())
    # Get legal action indices at startpos
    legal = {B.move_to_index(m) for m in chess.Board().legal_moves}
    for i in range(B.NUM_ACTIONS):
        if i not in legal:
            assert not np.isfinite(masked[i]) or masked[i] < -1e8


# ----- Search shape / sum invariants -----

def test_mcts_search_returns_valid_result():
    model, params = _tiny_net_and_params()
    out = mcts_search(chess.Board(), model, params, num_simulations=8)
    assert isinstance(out, MctsResult)
    assert out.action_probs.shape == (B.NUM_ACTIONS,)
    assert out.visit_counts.shape == (B.NUM_ACTIONS,)
    assert out.action_probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert out.num_simulations == 8


def test_mcts_best_move_is_legal():
    model, params = _tiny_net_and_params()
    out = mcts_search(chess.Board(), model, params, num_simulations=8)
    legal = list(chess.Board().legal_moves)
    assert out.best_move in legal


def test_mcts_total_visits_grows_with_n_sims():
    """visit_counts.sum() (root child visits) is bounded by num_simulations."""
    model, params = _tiny_net_and_params()
    eval_fn = make_eval_fn(model, params)
    out_4 = mcts_search(chess.Board(), model, params, num_simulations=4, eval_fn=eval_fn)
    out_32 = mcts_search(chess.Board(), model, params, num_simulations=32, eval_fn=eval_fn)
    assert int(out_4.visit_counts.sum()) >= 4
    assert int(out_32.visit_counts.sum()) >= 32
    assert int(out_32.visit_counts.sum()) > int(out_4.visit_counts.sum())


# ----- Terminal handling -----

def test_mcts_at_checkmate_position():
    """Anastasia mate: black-to-move and mated. Search should still return cleanly."""
    model, params = _tiny_net_and_params()
    bd = chess.Board("6k1/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert bd.is_checkmate()
    # No legal moves -> we shouldn't crash; mcts_search handles via _expand returning
    # terminal value. Best_move will be undefined but the call shouldn't throw.
    out = mcts_search(bd, model, params, num_simulations=4)
    # Visits at root may be 0 since there are no children to descend into.
    # action_probs falls back to uniform-over-legal; legal is empty -> all zero.
    assert out.action_probs.sum() == pytest.approx(0.0, abs=1e-5) or out.action_probs.sum() == pytest.approx(1.0, abs=1e-5)


# ----- After 1.e4: black to move, search runs from black's POV -----

def test_mcts_search_after_e4_black_to_move():
    model, params = _tiny_net_and_params()
    bd = chess.Board()
    bd.push_san("e4")
    out = mcts_search(bd, model, params, num_simulations=8)
    legal = list(bd.legal_moves)
    assert out.best_move in legal


# ----- Per-sim benchmark -----

def test_benchmark_per_sim_returns_stats():
    model, params = _tiny_net_and_params()
    stats = benchmark_per_sim(model, params, n_sims=8)
    assert stats["n_sims"] == 8
    assert stats["per_sim_ms"] > 0
    assert isinstance(stats["best_move"], str)
    assert -1.0 <= stats["root_value"] <= 1.0
