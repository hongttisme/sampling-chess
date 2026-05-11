"""Fixed-capacity FIFO replay buffer for Phase 2 self-play transitions.

Each transition stores the per-step state's
  observation    (8, 8, 119) float32
  pi_improved    (4672,)     float32  -- target policy from the improvement op
  legal_mask     (4672,)     bool
  value_target   ()          float32  -- z = outcome[player_at_step]
and the buffer is appended to from each completed Trajectory.

Sampling is uniform over all stored transitions (with replacement). For
training, batches return JAX arrays via numpy slices; the train_step
factory expects keys {obs, pi_improved, legal_mask, value_target}.

A 100k-transition buffer is the doc-aligned default (replay_buffer_size).
At typical 80-ply games that's ~1250 games worth of data.
"""

from typing import Optional

import numpy as np

from sampling_chess.net import PGX_NUM_ACTIONS, PGX_OBSERVATION_CHANNELS
from sampling_chess.selfplay import Trajectory


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.obs = np.zeros(
            (capacity, 8, 8, PGX_OBSERVATION_CHANNELS), dtype=np.float32)
        self.pi_improved = np.zeros((capacity, PGX_NUM_ACTIONS), dtype=np.float32)
        self.legal_mask = np.zeros((capacity, PGX_NUM_ACTIONS), dtype=bool)
        self.value_target = np.zeros(capacity, dtype=np.float32)
        self._write = 0
        self._n_stored = 0

    @property
    def n_stored(self) -> int:
        return self._n_stored

    def __len__(self) -> int:
        return self._n_stored

    def add_trajectory(self, traj: Trajectory) -> int:
        """Append one trajectory's per-step rows; returns rows added."""
        if traj.plies == 0:
            return 0
        z = traj.value_targets()  # (T,)
        for t in range(traj.plies):
            i = self._write
            self.obs[i] = traj.observations[t]
            self.pi_improved[i] = traj.improved_policies[t]
            self.legal_mask[i] = traj.legal_masks[t]
            self.value_target[i] = z[t]
            self._write = (i + 1) % self.capacity
        self._n_stored = min(self._n_stored + traj.plies, self.capacity)
        return traj.plies

    def sample(self, batch_size: int,
               rng: Optional[np.random.Generator] = None) -> dict:
        """Uniform-with-replacement sample of `batch_size` transitions."""
        if self._n_stored == 0:
            raise ValueError("cannot sample from empty buffer")
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, self._n_stored, size=batch_size)
        return {
            "observation": self.obs[idx],
            "pi_improved": self.pi_improved[idx],
            "legal_mask": self.legal_mask[idx],
            "value_target": self.value_target[idx],
        }
