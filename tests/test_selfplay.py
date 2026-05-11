"""Tests for selfplay.play_self_game with both Arm A and Arm B operators."""

import numpy as np
import pytest

pytest.importorskip("pgx")
import pgx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from sampling_chess.selfplay import (  # noqa: E402
    Trajectory,
    play_self_game,
    make_arm_b_op,
)
from sampling_chess.net import ChessTransformerPgx  # noqa: E402
from sampling_chess.search import MctsArmA  # noqa: E402
from sampling_chess.pgx_bridge import PGX_NUM_ACTIONS  # noqa: E402


_ENV = pgx.make("chess")


def _random_net():
    model = ChessTransformerPgx()
    dummy = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    params = model.init(jax.random.key(0), dummy)["params"]
    return model, params


# ----- Trajectory dataclass -----

def test_trajectory_value_targets_match_outcome_at_player():
    """value_targets() picks rewards[player] for each step."""
    traj = Trajectory(
        observations=np.zeros((4, 8, 8, 119), dtype=np.float32),
        improved_policies=np.zeros((4, PGX_NUM_ACTIONS), dtype=np.float32),
        legal_masks=np.zeros((4, PGX_NUM_ACTIONS), dtype=bool),
        actions=np.zeros(4, dtype=np.int32),
        player_at_step=np.array([0, 1, 0, 1], dtype=np.int32),
        outcome_per_player=np.array([1.0, -1.0], dtype=np.float32),
        plies=4, terminated=True,
    )
    z = traj.value_targets()
    assert z.tolist() == [1.0, -1.0, 1.0, -1.0]


# ----- Arm B end-to-end via play_self_game -----

def test_arm_b_self_game_short():
    """Random net + tiny K + low max_plies; verify trajectory invariants."""
    model, params = _random_net()
    op = make_arm_b_op(model, params, K=4, k_plies=2, beta=1.0,
                        stratified=True, rng=np.random.default_rng(0), env=_ENV)
    traj = play_self_game(op, env=_ENV, max_plies=10,
                          temperature_threshold=5,
                          rng=np.random.default_rng(0))
    assert isinstance(traj, Trajectory)
    assert traj.plies == traj.observations.shape[0]
    assert traj.plies <= 10
    # Each recorded action must be legal at its step.
    for t in range(traj.plies):
        assert bool(traj.legal_masks[t, traj.actions[t]]), (
            f"action {traj.actions[t]} illegal at step {t}"
        )
    # Outcome is in legal range.
    o = traj.outcome_per_player
    assert o.shape == (2,)
    assert (o >= -1.0).all() and (o <= 1.0).all()


def test_arm_a_self_game_short():
    """Same with MctsArmA."""
    model, params = _random_net()
    arm_a = MctsArmA(model=model, params=params, num_simulations=4)

    def op(state):
        return arm_a.improve_at_state(state)

    traj = play_self_game(op, env=_ENV, max_plies=8,
                          temperature_threshold=4,
                          rng=np.random.default_rng(1))
    assert traj.plies <= 8
    for t in range(traj.plies):
        assert bool(traj.legal_masks[t, traj.actions[t]])


# ----- pi_improved must be valid distributions throughout -----

def test_pi_improved_normalized_each_step():
    model, params = _random_net()
    op = make_arm_b_op(model, params, K=4, k_plies=2, beta=0.0,
                        stratified=True, rng=np.random.default_rng(2), env=_ENV)
    traj = play_self_game(op, env=_ENV, max_plies=6,
                          temperature_threshold=3,
                          rng=np.random.default_rng(2))
    for t in range(traj.plies):
        s = float(traj.improved_policies[t].sum())
        assert s == pytest.approx(1.0, abs=1e-4) or traj.improved_policies[t].sum() == 0.0
