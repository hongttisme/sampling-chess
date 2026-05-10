"""Tests for search.py scaffold (Arm A)."""

import chess
import pytest

pgx_bridge = pytest.importorskip("sampling_chess.pgx_bridge")
pytest.importorskip("pgx")

if not pgx_bridge.is_available():
    pytest.skip("pgx not available", allow_module_level=True)

from sampling_chess.search import MctsArmA, MctsResult  # noqa: E402
from sampling_chess.net import ChessTransformerPgx, PGX_NUM_ACTIONS  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def _make_random_net():
    model = ChessTransformerPgx()
    # Initialize with a tiny dummy obs from startpos.
    dummy_obs = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    params = model.init(jax.random.key(0), dummy_obs)["params"]
    return model, params


def test_mcts_result_dataclass_fields():
    r = MctsResult(
        pi_improved=np.zeros(10, dtype=np.float32),
        v_plus=0.0,
        visit_counts=np.zeros(10, dtype=np.int32),
    )
    assert r.pi_improved.shape == (10,)
    assert r.visit_counts.dtype == np.int32


def test_mcts_arm_a_runs_at_startpos():
    """End-to-end: random-init net + tiny mctx search at startpos returns
    a valid improvement target."""
    model, params = _make_random_net()
    op = MctsArmA(model=model, params=params, num_simulations=8)
    out = op.improve_at(chess.Board())
    assert isinstance(out, MctsResult)
    assert out.pi_improved.shape == (PGX_NUM_ACTIONS,)
    assert out.visit_counts.shape == (PGX_NUM_ACTIONS,)
    # Visit counts on illegal actions should be zero.
    state = pgx_bridge.chess_board_to_pgx_state(chess.Board())
    illegal = ~np.array(state.legal_action_mask)
    assert int(out.visit_counts[illegal].sum()) == 0
    # Visit counts on legal actions sum to num_simulations
    assert int(out.visit_counts.sum()) == op.num_simulations
    # action_weights normalize
    assert float(out.pi_improved.sum()) == pytest.approx(1.0, abs=1e-3)


def test_mcts_arm_a_runs_after_e4():
    """Same on a black-to-move position to exercise the POV flip."""
    model, params = _make_random_net()
    op = MctsArmA(model=model, params=params, num_simulations=8)
    bd = chess.Board()
    bd.push_san("e4")
    out = op.improve_at(bd)
    state = pgx_bridge.chess_board_to_pgx_state(bd)
    illegal = ~np.array(state.legal_action_mask)
    assert int(out.visit_counts[illegal].sum()) == 0
    assert int(out.visit_counts.sum()) == op.num_simulations
