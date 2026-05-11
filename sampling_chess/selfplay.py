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

def make_arm_b_op_builder(model, *, K: int = 64, k_plies: int = 8,
                          beta: float = 1.0, env=None,
                          rng: Optional[np.random.Generator] = None):
    """Returns op_builder(params) -> ImprovementOp (single-state).

    Builds the JIT'd sampler ONCE (via make_jit_sampler) and reuses it across
    iterations; only `params` change between iterations. Use this in the
    Phase 2 iteration driver so JIT cache doesn't reset every iter.
    """
    if rng is None:
        rng = np.random.default_rng()
    if env is None and _PGX_AVAILABLE:
        env = pgx.make("chess")
    from sampling_chess.sampling_pgx import (
        make_jit_sampler,
        sample_improved_policy_pgx_jit,
    )
    sampler = make_jit_sampler(model, K=K, k_plies=k_plies, env=env)

    def op_builder(params):
        def op(state):
            key = jax.random.key(int(rng.integers(2**31)))
            return sample_improved_policy_pgx_jit(
                root_state=state, sampler=sampler, params=params,
                K=K, beta=beta, rng_key=key,
            )
        return op

    return op_builder


# ---------------------------------------------------------------------------
# Batched (vmap-over-games) Arm B operator
# ---------------------------------------------------------------------------

def make_arm_b_batched_op_builder(model, *, K: int, k_plies: int,
                                  beta: float, n_games: int,
                                  env=None,
                                  rng: Optional[np.random.Generator] = None):
    """Returns op_builder(params) -> op_batched(states_batched, key)
                              -> (pi_improved (N, 4672), v_plus (N,)).

    The underlying jit sampler is vmapped over n_games so a single sample call
    processes all N games' rollouts in parallel on GPU. Massive throughput
    win on Blackwell vs the single-game sampler called N times sequentially.

    n_games must be fixed at build time (it's a vmap axis). To support a
    different n_games, build a fresh op_builder.
    """
    if rng is None:
        rng = np.random.default_rng()
    if env is None and _PGX_AVAILABLE:
        env = pgx.make("chess")
    if n_games < 1:
        raise ValueError(f"n_games must be >= 1, got {n_games}")

    from sampling_chess.sampling_pgx import (
        make_jit_sampler,
        PGX_NUM_ACTIONS,
    )

    sampler = make_jit_sampler(model, K=K, k_plies=k_plies, env=env)
    # vmap over root states: (params, batched_state, batched_key) -> batched outputs
    batched_sampler = jax.vmap(sampler, in_axes=(None, 0, 0))

    def op_builder(params):
        def op(states_batched, key):
            sub_keys = jax.random.split(key, n_games)
            (first_actions_j, leaf_v_stm_j, leaf_player_j,
             was_terminal_j, captured_rewards_j) = batched_sampler(
                params, states_batched, sub_keys
            )
            first_actions = np.asarray(first_actions_j)        # (N, K)
            leaf_v_stm = np.asarray(leaf_v_stm_j)              # (N, K)
            leaf_player = np.asarray(leaf_player_j)            # (N, K)
            was_terminal = np.asarray(was_terminal_j)          # (N, K)
            captured_rewards = np.asarray(captured_rewards_j)  # (N, K, 2)

            root_players = np.asarray(states_batched.current_player)  # (N,)

            pi_batch = np.zeros((n_games, PGX_NUM_ACTIONS), dtype=np.float32)
            v_batch = np.zeros(n_games, dtype=np.float32)
            for i in range(n_games):
                rp = int(root_players[i])
                v_nonterm = np.where(
                    leaf_player[i] == rp, leaf_v_stm[i], -leaf_v_stm[i]
                ).astype(np.float32)
                v_term = captured_rewards[i, :, rp].astype(np.float32)
                leaf_values = np.where(was_terminal[i], v_term, v_nonterm)
                z = beta * leaf_values
                z = z - z.max()
                w = np.exp(z)
                w = w / w.sum()
                pi = np.zeros(PGX_NUM_ACTIONS, dtype=np.float32)
                np.add.at(pi, first_actions[i].astype(np.int32),
                          w.astype(np.float32))
                pi_batch[i] = pi
                v_batch[i] = float((w * leaf_values).sum())

            return pi_batch, v_batch

        return op
    return op_builder


# ---------------------------------------------------------------------------
# Batched (vmap-over-games) self-play loop
# ---------------------------------------------------------------------------

def play_self_games_batched(
    op_batched: Callable,   # (states_batched, key) -> (pi (N, A), v_plus (N,))
    n_games: int,
    *,
    env=None,
    max_plies: int = 200,
    temperature_threshold: int = 30,
    rng: Optional[np.random.Generator] = None,
    starting_states=None,
) -> list:
    """Play `n_games` self-play games synchronized over plies via vmap.

    Each ply: 1 batched op call + 1 vmapped env.step. Per-game completion is
    tracked in Python (`done[i]` mask); after a game terminates we keep the
    pgx state stepping (it stays terminal in pgx) but don't append rows to
    the trajectory and freeze the captured rewards.

    Returns: list of `n_games` Trajectory objects.
    """
    if not _PGX_AVAILABLE:
        raise ImportError("pgx not installed")
    if env is None:
        env = pgx.make("chess")
    if rng is None:
        rng = np.random.default_rng()

    vmap_step = jax.jit(jax.vmap(env.step))
    vmap_init = jax.jit(jax.vmap(env.init))

    if starting_states is None:
        init_keys = jax.random.split(
            jax.random.key(int(rng.integers(2**31))), n_games
        )
        states = vmap_init(init_keys)
    else:
        states = starting_states

    # Per-game per-ply lists.
    obs_lists = [[] for _ in range(n_games)]
    pi_lists = [[] for _ in range(n_games)]
    mask_lists = [[] for _ in range(n_games)]
    action_lists = [[] for _ in range(n_games)]
    player_lists = [[] for _ in range(n_games)]

    done = np.array(states.terminated, copy=True)  # (N,) bool
    captured_rewards = np.array(states.rewards, copy=True)  # (N, 2)

    from sampling_chess.pgx_bridge import PGX_NUM_ACTIONS

    for ply in range(max_plies):
        if bool(done.all()):
            break

        op_key = jax.random.key(int(rng.integers(2**31)))
        pi_imp_batch, _ = op_batched(states, op_key)  # (N, A), (N,)
        pi_imp_batch = np.asarray(pi_imp_batch)

        legal_mask_batch = np.asarray(states.legal_action_mask)  # (N, A)
        observation_batch = np.asarray(states.observation)      # (N, 8, 8, 119)
        current_player_batch = np.asarray(states.current_player)  # (N,)

        actions = np.zeros(n_games, dtype=np.int32)
        for i in range(n_games):
            if done[i]:
                continue
            pi = pi_imp_batch[i]
            mass = float(pi.sum())
            if mass <= 0:
                mask_f = legal_mask_batch[i].astype(np.float32)
                probs = mask_f / max(float(mask_f.sum()), 1.0)
            else:
                probs = pi / mass
            if ply < temperature_threshold:
                actions[i] = int(rng.choice(PGX_NUM_ACTIONS, p=probs))
            else:
                actions[i] = int(np.argmax(probs))

        for i in range(n_games):
            if done[i]:
                continue
            obs_lists[i].append(observation_batch[i])
            pi_lists[i].append(pi_imp_batch[i])
            mask_lists[i].append(legal_mask_batch[i])
            action_lists[i].append(actions[i])
            player_lists[i].append(int(current_player_batch[i]))

        step_keys = jax.random.split(
            jax.random.key(int(rng.integers(2**31))), n_games
        )
        actions_jnp = jnp.asarray(actions, dtype=jnp.int32)
        new_states = vmap_step(states, actions_jnp, step_keys)

        new_term = np.asarray(new_states.terminated)
        newly_terminal = new_term & (~done)
        new_rewards = np.asarray(new_states.rewards)
        captured_rewards = np.where(
            newly_terminal[:, None], new_rewards, captured_rewards
        )
        done = done | new_term
        states = new_states

    trajs = []
    for i in range(n_games):
        T = len(obs_lists[i])
        if T == 0:
            trajs.append(Trajectory(
                observations=np.zeros((0, 8, 8, 119), dtype=np.float32),
                improved_policies=np.zeros((0, PGX_NUM_ACTIONS), dtype=np.float32),
                legal_masks=np.zeros((0, PGX_NUM_ACTIONS), dtype=bool),
                actions=np.zeros(0, dtype=np.int32),
                player_at_step=np.zeros(0, dtype=np.int32),
                outcome_per_player=captured_rewards[i].astype(np.float32),
                plies=0, terminated=bool(done[i]),
            ))
            continue
        trajs.append(Trajectory(
            observations=np.stack(obs_lists[i]).astype(np.float32),
            improved_policies=np.stack(pi_lists[i]).astype(np.float32),
            legal_masks=np.stack(mask_lists[i]),
            actions=np.array(action_lists[i], dtype=np.int32),
            player_at_step=np.array(player_lists[i], dtype=np.int32),
            outcome_per_player=captured_rewards[i].astype(np.float32),
            plies=T, terminated=bool(done[i]),
        ))
    return trajs


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
