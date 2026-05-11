"""Phase 2 self-play: generate (state, pi_improved, z) trajectories.

A self-play game starts at pgx env.init() (or a custom starting state),
applies an improvement operator at each ply to produce a target policy
pi_improved, samples or argmaxes an action from it (temperature schedule),
and steps the pgx env to the next state. Termination flips outcome
through pgx's rewards array; we record per-ply player so the value target
at each step is z = outcome[player_at_step], i.e., side-to-move POV.

Both arms expose `improve_at_state(pgx_state) -> result` with `result.pi_improved`
and `result.v_plus`. The same self-play loop runs Arm A or Arm B by swapping
the operator.

Temperature schedule (per doc 5.3 / AlphaZero convention):
  - First `temperature_threshold` plies: sample a ~ pi_improved (exploration).
  - After: argmax(pi_improved) (greedy exploitation).

Trajectory data is plain numpy; replay buffer + train loop in train.py + a
later phase-2 driver consume it.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

try:
    import pgx
    _PGX_AVAILABLE = True
except ImportError:
    pgx = None  # type: ignore
    _PGX_AVAILABLE = False

from sampling_chess.pgx_bridge import PGX_NUM_ACTIONS


# An improvement operator is any callable taking a pgx state and returning an
# object whose .pi_improved is a (4672,) numpy array summing to 1 over legal
# first moves. Both MctsArmA.improve_at_state and a thin wrapper around
# sample_improved_policy_pgx satisfy this protocol.
ImprovementOp = Callable[[object], object]


@dataclass
class Trajectory:
    observations: np.ndarray         # (T, 8, 8, 119) float32
    improved_policies: np.ndarray    # (T, 4672) float32
    legal_masks: np.ndarray          # (T, 4672) bool
    actions: np.ndarray              # (T,) int32 - action actually taken
    player_at_step: np.ndarray       # (T,) int32 - player to move at this step
    outcome_per_player: np.ndarray   # (2,) float32 - final pgx rewards
    plies: int
    terminated: bool

    def value_targets(self) -> np.ndarray:
        """z_t = outcome[player_at_step[t]] -> per-step target value (T,) float32."""
        return self.outcome_per_player[self.player_at_step].astype(np.float32)


def _empty_arrays():
    return (
        np.zeros((0, 8, 8, 119), dtype=np.float32),
        np.zeros((0, PGX_NUM_ACTIONS), dtype=np.float32),
        np.zeros((0, PGX_NUM_ACTIONS), dtype=bool),
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.int32),
    )


def play_self_game(
    op: ImprovementOp,
    *,
    env=None,
    max_plies: int = 400,
    temperature_threshold: int = 30,
    rng: Optional[np.random.Generator] = None,
    starting_state=None,
) -> Trajectory:
    """Play one self-play game, return its Trajectory.

    Args:
      op: improvement operator (callable: pgx_state -> result).
      env: pgx Chess env (created if None). Reuse across games for speed.
      max_plies: hard cap to bound wall-clock per game.
      temperature_threshold: sample first N plies, argmax after.
      rng: numpy Generator (defaults to default_rng()).
      starting_state: optional pgx State (e.g., for opening-book sampling);
                      defaults to env.init() with a fresh key.
    """
    if not _PGX_AVAILABLE:
        raise ImportError("pgx not installed")
    if env is None:
        env = pgx.make("chess")
    if rng is None:
        rng = np.random.default_rng()

    if starting_state is None:
        key = jax.random.key(int(rng.integers(2**31)))
        state = jax.jit(env.init)(key)
    else:
        state = starting_state

    obs_list, pi_list, mask_list, action_list, player_list = [], [], [], [], []

    for ply in range(max_plies):
        if bool(state.terminated):
            break

        result = op(state)
        pi_imp = np.asarray(result.pi_improved, dtype=np.float32)

        # Record current state's data BEFORE acting on it.
        obs_list.append(np.asarray(state.observation, dtype=np.float32))
        pi_list.append(pi_imp)
        mask_list.append(np.asarray(state.legal_action_mask, dtype=bool))
        player_list.append(int(state.current_player))

        # Action selection: sample for first N plies (exploration), then argmax.
        if pi_imp.sum() <= 0:
            # Degenerate: fall back to uniform over legal moves.
            mask = np.asarray(state.legal_action_mask, dtype=np.float32)
            probs = mask / max(mask.sum(), 1.0)
        else:
            probs = pi_imp / pi_imp.sum()

        if ply < temperature_threshold:
            action = int(rng.choice(PGX_NUM_ACTIONS, p=probs))
        else:
            action = int(np.argmax(probs))

        action_list.append(action)

        step_key = jax.random.key(int(rng.integers(2**31)))
        state = env.step(state, jnp.int32(action), step_key)

    if not action_list:
        e_obs, e_pi, e_mask, e_act, e_player = _empty_arrays()
        return Trajectory(
            observations=e_obs, improved_policies=e_pi, legal_masks=e_mask,
            actions=e_act, player_at_step=e_player,
            outcome_per_player=np.array(state.rewards, dtype=np.float32),
            plies=0, terminated=bool(state.terminated),
        )

    return Trajectory(
        observations=np.stack(obs_list),
        improved_policies=np.stack(pi_list),
        legal_masks=np.stack(mask_list),
        actions=np.array(action_list, dtype=np.int32),
        player_at_step=np.array(player_list, dtype=np.int32),
        outcome_per_player=np.array(state.rewards, dtype=np.float32),
        plies=len(action_list),
        terminated=bool(state.terminated),
    )


# ---------------------------------------------------------------------------
# Operator wrappers — adapt the two arms to the ImprovementOp protocol.
# ---------------------------------------------------------------------------

def make_arm_b_op(model, params, *, K: int = 64, k_plies: int = 8,
                  beta: float = 1.0, stratified: bool = False,
                  rng: Optional[np.random.Generator] = None,
                  env=None, use_jit: bool = True):
    """Wrap sample_improved_policy_pgx as an ImprovementOp (state -> result).

    Defaults to the jit-vectorized variant (~200x faster than the Python-loop
    fallback after compile warmup). Set use_jit=False if you need the
    stratified first-move mode, which is only implemented in the Python loop.
    """
    if rng is None:
        rng = np.random.default_rng()
    if env is None and _PGX_AVAILABLE:
        env = pgx.make("chess")

    if use_jit and stratified:
        import warnings
        warnings.warn(
            "make_arm_b_op: stratified=True is not implemented in the jit "
            "sampler; falling back to the Python-loop variant (~200x slower).",
            stacklevel=2,
        )

    if use_jit and not stratified:
        from sampling_chess.sampling_pgx import (
            make_jit_sampler,
            sample_improved_policy_pgx_jit,
        )
        sampler = make_jit_sampler(model, K=K, k_plies=k_plies, env=env)

        def op_jit(state):
            key = jax.random.key(int(rng.integers(2**31)))
            return sample_improved_policy_pgx_jit(
                root_state=state, sampler=sampler, params=params,
                K=K, beta=beta, rng_key=key,
            )

        return op_jit

    # Python-loop fallback (supports stratified).
    from sampling_chess.sampling_pgx import sample_improved_policy_pgx

    def apply_fn(states):
        obs = jnp.stack([s.observation for s in states])
        logits, values = model.apply({"params": params}, obs)
        return np.asarray(logits), np.asarray(values)

    def op(state):
        return sample_improved_policy_pgx(
            root_state=state, apply_fn=apply_fn,
            K=K, k_plies=k_plies, beta=beta,
            rng=rng, stratified=stratified, env=env,
        )

    return op
