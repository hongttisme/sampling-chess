"""Tests for search.py scaffold (Arm A)."""

import chess
import pytest

pgx_bridge = pytest.importorskip("sampling_chess.pgx_bridge")
pytest.importorskip("pgx")

if not pgx_bridge.is_available():
    pytest.skip("pgx not available", allow_module_level=True)

from sampling_chess.search import MctsArmA, MctsResult  # noqa: E402


def test_mcts_arm_a_constructible():
    op = MctsArmA(num_simulations=8)
    assert op.num_simulations == 8
    assert op.c_puct > 0


def test_mcts_arm_a_improve_at_raises_not_implemented():
    """Until the architectural decision (Plan A/B/D) is made, calling
    improve_at must fail loudly so we don't ship a half-wired baseline."""
    op = MctsArmA(num_simulations=8)
    with pytest.raises(NotImplementedError, match="Plan A / B / D"):
        op.improve_at(chess.Board())


def test_mcts_result_dataclass_fields():
    """Sanity-check the public output type's shape."""
    import numpy as np
    r = MctsResult(
        pi_improved=np.zeros(10, dtype=np.float32),
        v_plus=0.0,
        visit_counts=np.zeros(10, dtype=np.int32),
    )
    assert r.pi_improved.shape == (10,)
    assert r.v_plus == 0.0
    assert r.visit_counts.dtype == np.int32
