"""Arm B (pgx-native): importance-weighted sampling on pgx Chess.

Same algorithm as sampling.py (doc 2.3), adapted to pgx state + 4672
action space so the operator is composable with the pgx + mctx stack.

  for i = 1..K:
      sample tau_i = (a_0^(i), ..., a_{k-1}^(i)) from pi_theta
      if game ends within k plies: V_i = terminal value, root POV
      else:                         V_i = V_theta(s_k^(i)), root POV

  pi_sample(a | s_0) = sum_{i: a_0^(i)=a} exp(beta V_i) / sum exp(beta V_i')
  V^+(s_0)           = sum_i w_i V_i,   w_i proportional to exp(beta V_i)

Two implementations:

  * sample_improved_policy_pgx (Python loop, model-agnostic apply_fn):
    iterates trajectories sequentially. Easier to debug; baseline correct.
    ~42 s/call on CPU for K=8/k=3 due to eager JAX env.step dispatch.

  * sample_improved_policy_pgx_jit (vmap + lax.scan + jax.jit):
    runs all K trajectories of k_plies steps inside a single jit'd graph.
    Requires (model, k_plies, K) to be static at trace time. Handles
    terminal trajectories via a captured-rewards carry, since pgx zeroes
    .rewards on subsequent steps after the first terminal transition.

apply_fn signature (Python version):
    list[pgx.State] -> (logits (N, 4672), values (N,))
    values are in side-to-move POV per pgx convention.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

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

# A network forward call: list of pgx states -> (policy_logits, values).
ApplyFn = Callable[[list], Tuple[np.ndarray, np.ndarray]]


@dataclass
class SamplingResultPgx:
    pi_improved: np.ndarray  # (4672,) float32, sums to 1 over legal first moves
    v_plus: float           # SNIS-weighted leaf value, root POV
    leaf_values: np.ndarray # (K,) float32, V_i for each trajectory (root POV)
    first_moves: np.ndarray # (K,) int32, pgx action index of a_0^(i)


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over a 1-D logits vector with a boolean legality mask."""
    masked = np.where(mask, logits, -np.inf)
    masked -= np.nanmax(masked)
    masked = np.where(mask, masked, -np.inf)
    e = np.exp(masked)
    s = e.sum()
    if s == 0:
        u = mask.astype(np.float32)
        return u / max(u.sum(), 1.0)
    return (e / s).astype(np.float32)


def _stratified_first_moves(legal_indices: List[int], K: int,
                            rng: np.random.Generator) -> np.ndarray:
    n_legal = len(legal_indices)
    base = K // n_legal
    rem = K - base * n_legal
    counts = np.full(n_legal, base, dtype=np.int32)
    if rem > 0:
        extra = rng.choice(n_legal, size=rem, replace=False)
        counts[extra] += 1
    out = np.empty(K, dtype=np.int32)
    pos = 0
    for idx, c in zip(legal_indices, counts):
        out[pos : pos + c] = idx
        pos += c
    rng.shuffle(out)
    return out


def _sign_flip_root_pov(value_leaf_stm: float, root_player: int,
                        leaf_player: int) -> float:
    """Convert leaf-POV V to root-POV (negate if leaf player != root player)."""
    return value_leaf_stm if leaf_player == root_player else -value_leaf_stm


def sample_improved_policy_pgx(
    root_state,
    apply_fn: ApplyFn,
    K: int,
    k_plies: int,
    beta: float,
    rng: Optional[np.random.Generator] = None,
    stratified: bool = False,
    env=None,
) -> SamplingResultPgx:
    """Run Algorithm 1 at `root_state` (a pgx Chess State).

    Args mirror sampling.sample_improved_policy. `env` should be
    `pgx.make("chess")` reused across calls (created lazily if None).
    """
    if not _PGX_AVAILABLE:
        raise ImportError("pgx not installed")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if k_plies < 1:
        raise ValueError(f"k_plies must be >= 1, got {k_plies}")
    if rng is None:
        rng = np.random.default_rng()
    if env is None:
        env = pgx.make("chess")

    root_player = int(root_state.current_player)
    A = PGX_NUM_ACTIONS

    # Terminal root: trivially return the terminal value, zero pi.
    legal_root = np.where(np.array(root_state.legal_action_mask))[0]
    if bool(root_state.terminated) or len(legal_root) == 0:
        rewards = np.array(root_state.rewards)
        v = float(rewards[root_player]) if root_state.terminated else 0.0
        return SamplingResultPgx(
            pi_improved=np.zeros(A, dtype=np.float32),
            v_plus=v,
            leaf_values=np.full(K, v, dtype=np.float32),
            first_moves=np.full(K, -1, dtype=np.int32),
        )

    # Step 0: pick first move per trajectory.
    first_moves = np.full(K, -1, dtype=np.int32)
    if stratified:
        first_moves[:] = _stratified_first_moves(legal_root.tolist(), K, rng)
    else:
        # Prior-weighted from pi_theta(.|root).
        logits, _ = apply_fn([root_state])
        mask_root = np.array(root_state.legal_action_mask)
        probs = _masked_softmax(np.asarray(logits[0]), mask_root)
        first_moves[:] = rng.choice(A, size=K, p=probs)

    # Initialize K trajectory states by stepping the root with each first move.
    states = []
    plies_pushed = np.zeros(K, dtype=np.int32)
    terminal_value = np.full(K, np.nan, dtype=np.float32)
    keys = jax.random.split(jax.random.key(int(rng.integers(2**31))), K)

    for i in range(K):
        s = env.step(root_state, jnp.int32(int(first_moves[i])), keys[i])
        states.append(s)
        plies_pushed[i] = 1
        if bool(s.terminated):
            r = np.array(s.rewards)
            terminal_value[i] = float(r[root_player])

    # Steps 1..k-1: sample subsequent moves from pi_theta.
    for ply in range(1, k_plies):
        active = [i for i in range(K) if np.isnan(terminal_value[i])]
        if not active:
            break
        active_states = [states[i] for i in active]
        logits, _ = apply_fn(active_states)
        for j, i in enumerate(active):
            mask_i = np.array(states[i].legal_action_mask)
            probs = _masked_softmax(np.asarray(logits[j]), mask_i)
            chosen = int(rng.choice(A, p=probs))
            sub_key = jax.random.key(int(rng.integers(2**31)))
            states[i] = env.step(states[i], jnp.int32(chosen), sub_key)
            plies_pushed[i] += 1
            if bool(states[i].terminated):
                r = np.array(states[i].rewards)
                terminal_value[i] = float(r[root_player])

    # Step k: V_theta at non-terminal leaves.
    leaf_values = np.zeros(K, dtype=np.float32)
    nonterm = [i for i in range(K) if np.isnan(terminal_value[i])]
    if nonterm:
        leaf_states = [states[i] for i in nonterm]
        _, vs = apply_fn(leaf_states)
        for j, i in enumerate(nonterm):
            v_leaf = float(vs[j])
            leaf_player = int(states[i].current_player)
            leaf_values[i] = _sign_flip_root_pov(v_leaf, root_player, leaf_player)
    for i in range(K):
        if not np.isnan(terminal_value[i]):
            leaf_values[i] = terminal_value[i]

    # SNIS aggregate.
    z = beta * leaf_values
    z -= z.max()
    w = np.exp(z)
    w /= w.sum()

    pi_improved = np.zeros(A, dtype=np.float32)
    np.add.at(pi_improved, first_moves, w.astype(np.float32))

    v_plus = float((w * leaf_values).sum())

    return SamplingResultPgx(
        pi_improved=pi_improved,
        v_plus=v_plus,
        leaf_values=leaf_values.astype(np.float32),
        first_moves=first_moves,
    )


# ---------------------------------------------------------------------------
# JIT-vectorized variant
# ---------------------------------------------------------------------------

def make_jit_sampler(model, K: int, k_plies: int, env=None):
    """Build a jit'd K-trajectory rollout function.

    K and k_plies become static at trace time. The returned `sampler` takes
    (params, root_state, key) and returns numpy arrays usable downstream.

    Returns a callable: sampler(params, root_state, key) ->
      (first_actions (K,) int32,
       leaf_v_stm  (K,) float32,    # V_theta at leaf, leaf-STM POV
       leaf_player (K,) int32,
       was_terminal (K,) bool,
       captured_rewards (K, 2) float32)  # rewards at first terminal transition
    Aggregation to (pi_improved, v_plus) happens outside jit.
    """
    if not _PGX_AVAILABLE:
        raise ImportError("pgx not installed")
    if env is None:
        env = pgx.make("chess")
    if K < 1 or k_plies < 1:
        raise ValueError(f"K and k_plies must be >= 1; got K={K}, k_plies={k_plies}")

    vmap_step = jax.vmap(env.step)

    @jax.jit
    def sampler(params, root_state, key):
        # ---- Step 0: K first-moves sampled prior-weighted from root ----
        logits_root, _ = model.apply(
            {"params": params}, root_state.observation[None]
        )
        masked_root = jnp.where(
            root_state.legal_action_mask, logits_root[0], -1e9
        )
        first_keys = jax.random.split(jax.random.fold_in(key, 0), K)
        first_actions = jax.vmap(
            lambda k: jax.random.categorical(k, masked_root)
        )(first_keys)

        # Replicate root state K times.
        root_batched = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None], (K,) + x.shape), root_state
        )
        step_keys_0 = jax.random.split(jax.random.fold_in(key, 1), K)
        states = vmap_step(root_batched, first_actions, step_keys_0)

        was_terminal = states.terminated
        captured_rewards = states.rewards  # (K, 2) — valid where was_terminal

        # ---- Scan steps 1..k_plies-1 ----
        def body(carry, scan_key):
            states, was_terminal, captured_rewards = carry
            apply_key, step_key = jax.random.split(scan_key)
            logits, _ = model.apply({"params": params}, states.observation)
            masked = jnp.where(states.legal_action_mask, logits, -1e9)
            apply_keys = jax.random.split(apply_key, K)
            actions = jax.vmap(jax.random.categorical)(apply_keys, masked)
            step_keys = jax.random.split(step_key, K)
            new_states = vmap_step(states, actions, step_keys)
            is_term_now = new_states.terminated
            newly_terminal = is_term_now & (~was_terminal)
            captured_rewards = jnp.where(
                newly_terminal[:, None], new_states.rewards, captured_rewards
            )
            was_terminal = was_terminal | is_term_now
            return (new_states, was_terminal, captured_rewards), None

        if k_plies > 1:
            scan_keys = jax.random.split(jax.random.fold_in(key, 2), k_plies - 1)
            (states, was_terminal, captured_rewards), _ = jax.lax.scan(
                body, (states, was_terminal, captured_rewards), scan_keys
            )

        # ---- Leaf V_theta ----
        _, leaf_v_stm = model.apply({"params": params}, states.observation)
        leaf_player = states.current_player

        return (
            first_actions.astype(jnp.int32),
            leaf_v_stm.astype(jnp.float32),
            leaf_player.astype(jnp.int32),
            was_terminal,
            captured_rewards.astype(jnp.float32),
        )

    return sampler


def sample_improved_policy_pgx_jit(
    root_state,
    sampler: Callable,  # output of make_jit_sampler
    params,
    K: int,
    beta: float,
    rng_key,
) -> SamplingResultPgx:
    """JIT-vectorized version. K must match the value used in make_jit_sampler."""
    first_actions_j, leaf_v_stm_j, leaf_player_j, was_terminal_j, captured_rewards_j = sampler(
        params, root_state, rng_key
    )
    first_actions = np.asarray(first_actions_j)
    leaf_v_stm = np.asarray(leaf_v_stm_j)
    leaf_player = np.asarray(leaf_player_j)
    was_terminal = np.asarray(was_terminal_j)
    captured_rewards = np.asarray(captured_rewards_j)

    root_player = int(root_state.current_player)

    # Compose leaf values in root POV.
    v_nonterm = np.where(
        leaf_player == root_player, leaf_v_stm, -leaf_v_stm
    ).astype(np.float32)
    v_term = captured_rewards[:, root_player].astype(np.float32)
    leaf_values = np.where(was_terminal, v_term, v_nonterm)

    # SNIS aggregate.
    z = beta * leaf_values
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()

    pi_improved = np.zeros(PGX_NUM_ACTIONS, dtype=np.float32)
    np.add.at(pi_improved, first_actions, w.astype(np.float32))

    v_plus = float((w * leaf_values).sum())

    return SamplingResultPgx(
        pi_improved=pi_improved,
        v_plus=v_plus,
        leaf_values=leaf_values.astype(np.float32),
        first_moves=first_actions.astype(np.int32),
    )
