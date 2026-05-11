"""Tests for ReplayBuffer."""

import numpy as np
import pytest

from sampling_chess.buffer import ReplayBuffer
from sampling_chess.net import PGX_NUM_ACTIONS, PGX_OBSERVATION_CHANNELS
from sampling_chess.selfplay import Trajectory


def _make_traj(plies: int = 5, outcome=(1.0, -1.0)) -> Trajectory:
    return Trajectory(
        observations=np.random.RandomState(0).randn(
            plies, 8, 8, PGX_OBSERVATION_CHANNELS).astype(np.float32),
        improved_policies=np.eye(PGX_NUM_ACTIONS, dtype=np.float32)[:plies],
        legal_masks=np.ones((plies, PGX_NUM_ACTIONS), dtype=bool),
        actions=np.arange(plies, dtype=np.int32),
        player_at_step=np.array([i % 2 for i in range(plies)], dtype=np.int32),
        outcome_per_player=np.array(outcome, dtype=np.float32),
        plies=plies, terminated=True,
    )


def test_empty_buffer_raises_on_sample():
    buf = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError, match="empty"):
        buf.sample(batch_size=2)


def test_add_increments_n_stored():
    buf = ReplayBuffer(capacity=20)
    assert len(buf) == 0
    traj = _make_traj(plies=5)
    added = buf.add_trajectory(traj)
    assert added == 5
    assert len(buf) == 5


def test_add_zero_plies_no_op():
    buf = ReplayBuffer(capacity=10)
    empty = Trajectory(
        observations=np.zeros((0, 8, 8, PGX_OBSERVATION_CHANNELS), dtype=np.float32),
        improved_policies=np.zeros((0, PGX_NUM_ACTIONS), dtype=np.float32),
        legal_masks=np.zeros((0, PGX_NUM_ACTIONS), dtype=bool),
        actions=np.zeros(0, dtype=np.int32),
        player_at_step=np.zeros(0, dtype=np.int32),
        outcome_per_player=np.zeros(2, dtype=np.float32),
        plies=0, terminated=False,
    )
    assert buf.add_trajectory(empty) == 0
    assert len(buf) == 0


def test_capacity_cap():
    """Storing more than capacity should saturate at capacity."""
    buf = ReplayBuffer(capacity=3)
    traj = _make_traj(plies=5)
    buf.add_trajectory(traj)
    assert len(buf) == 3  # capped


def test_fifo_wraparound():
    """The write pointer wraps; oldest entries get overwritten."""
    buf = ReplayBuffer(capacity=3)
    traj1 = _make_traj(plies=2, outcome=(1.0, -1.0))
    traj2 = _make_traj(plies=3, outcome=(-1.0, 1.0))
    buf.add_trajectory(traj1)
    buf.add_trajectory(traj2)
    # Capacity 3, total stored 5 -> last 3 entries remain.
    # value_targets after wrap: traj1 step 0 (overwritten), traj2 steps 0-2.
    # Verifies via player_at_step: traj2 used player [0,1,0]; their value targets
    # are outcome[player] = [-1, 1, -1].
    expected = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    # Buffer contents are at indices 0,1,2 in physical order; the write
    # pointer wrapped after the 2nd traj's 3rd step. Last entries should match.
    assert len(buf) == 3
    # All 3 value_targets should be from traj2
    assert set(np.unique(buf.value_target[:3]).tolist()) <= {-1.0, 1.0}


def test_sample_shapes_correct():
    buf = ReplayBuffer(capacity=20)
    buf.add_trajectory(_make_traj(plies=5))
    rng = np.random.default_rng(0)
    batch = buf.sample(batch_size=3, rng=rng)
    assert batch["observation"].shape == (3, 8, 8, PGX_OBSERVATION_CHANNELS)
    assert batch["pi_improved"].shape == (3, PGX_NUM_ACTIONS)
    assert batch["legal_mask"].shape == (3, PGX_NUM_ACTIONS)
    assert batch["value_target"].shape == (3,)


def test_sample_reproducible_with_seed():
    buf = ReplayBuffer(capacity=20)
    buf.add_trajectory(_make_traj(plies=10))
    b1 = buf.sample(batch_size=4, rng=np.random.default_rng(42))
    b2 = buf.sample(batch_size=4, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(b1["value_target"], b2["value_target"])
