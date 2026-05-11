"""Arm A: MCTS via DeepMind mctx + pgx Chess (Plan A).

Now that net.py + pgx_bridge are pgx-native, the recurrent_fn is JAX-pure
end-to-end:

  embedding = pgx State   (chess board, legal mask, observation, ...)
  step      = pgx env.step(state, action, rng_key)
  prior + V = net.apply({"params": ...}, state.observation[None])

Two-player handling:
  * `discount = -1` per step so each ply flips value sign back to the
    parent's POV (standard mctx 2-player convention).
  * Terminal nodes: discount = 0, reward carries the +/- 1 (or 0 draw)
    for the player who just moved.

Returned from MctsArmA.improve_at:
  * pi_improved : softmax-normalized visit-count distribution over the
                  4672 pgx action space.
  * v_plus      : scalar root value estimate from search.
  * visit_counts: per-action visit counts (int32).
"""

from dataclasses import dataclass
from typing import Optional

import chess
import jax
import jax.numpy as jnp
import numpy as np

from sampling_chess import pgx_bridge

try:
    import mctx
    import pgx as _pgx
    _MCTX_AVAILABLE = True
except ImportError:
    mctx = None  # type: ignore
    _pgx = None  # type: ignore
    _MCTX_AVAILABLE = False


@dataclass
class MctsResult:
    pi_improved: np.ndarray   # (PGX_NUM_ACTIONS,) float32, sums to 1 over legal moves
    v_plus: float             # value bootstrap from search
    visit_counts: np.ndarray  # (PGX_NUM_ACTIONS,) int32, visits per action


class MctsArmA:
    """MCTS policy improvement via mctx.muzero_policy on a pgx Chess env.

    Constructor takes a Flax model + params; net is expected to map a
    pgx observation (B, 8, 8, 119) to (policy_logits (B, 4672), value (B,))
    where value is in side-to-move POV.
    """

    def __init__(
        self,
        model,
        params,
        num_simulations: int = 100,
        c_puct_init: float = 1.25,
        c_puct_base: float = 19652.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
        rng_seed: int = 0,
    ):
        if not _MCTX_AVAILABLE:
            raise ImportError("mctx and pgx are required for MctsArmA")
        if not pgx_bridge.is_available():
            raise ImportError("pgx_bridge / pgx not available")

        self.model = model
        self.params = params
        self.num_simulations = num_simulations
        self.c_puct_init = c_puct_init
        self.c_puct_base = c_puct_base
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self._rng = jax.random.key(rng_seed)
        self._env = _pgx.make("chess")

        # Vectorize env.step over the batch dim (size 1 for now, but the API
        # is batched throughout).
        self._step_v = jax.vmap(self._env.step, in_axes=(0, 0, 0))

        # JIT-compiled root + recurrent fns.
        self._root_fn = jax.jit(self._build_root_fn())
        self._recurrent_fn = self._build_recurrent_fn()
        self._policy_fn = jax.jit(self._build_policy_fn())

    # ----- Root + recurrent functions -----

    def _build_root_fn(self):
        model = self.model
        def root_fn(params, state):
            obs = state.observation[None]  # (1, 8, 8, 119)
            logits, value = model.apply({"params": params}, obs)
            return mctx.RootFnOutput(
                prior_logits=logits,
                value=value,
                embedding=jax.tree_util.tree_map(lambda x: x[None], state),
            )
        return root_fn

    def _build_recurrent_fn(self):
        model = self.model
        step_v = self._step_v

        def recurrent_fn(params, rng_key, action, embedding):
            # action: (B,), embedding: pgx state batched over leading dim B
            keys = jax.random.split(rng_key, action.shape[0])
            new_state = step_v(embedding, action, keys)
            logits, value = model.apply({"params": params}, new_state.observation)

            # Reward from the POV of the player who JUST moved (= player whose
            # turn it WAS before this action). pgx assigns rewards[i] to
            # player i; we pick the player who acted by inverting the new
            # current_player on every step (single-player swap).
            # In pgx 2-player chess, current_player flips between {0, 1}.
            actor = 1 - new_state.current_player  # (B,)
            rewards = new_state.rewards            # (B, 2)
            reward = jnp.take_along_axis(rewards, actor[:, None], axis=1)[:, 0]

            # Discount: 0 at terminal (cuts off bootstrap), -1 otherwise
            # to invert child value back to parent's POV.
            discount = jnp.where(new_state.terminated, 0.0, -1.0)

            # Mask illegal logits at the new node (mctx prefers very negative
            # logits for invalid actions; we also pass invalid_actions).
            logits = jnp.where(new_state.legal_action_mask, logits, -1e9)

            output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            )
            return output, new_state

        return recurrent_fn

    def _build_policy_fn(self):
        recurrent_fn = self._recurrent_fn
        num_simulations = self.num_simulations
        c_puct_init = self.c_puct_init
        c_puct_base = self.c_puct_base
        dirichlet_alpha = self.dirichlet_alpha
        dirichlet_fraction = self.dirichlet_fraction

        def policy_fn(params, rng_key, root, invalid_actions):
            return mctx.muzero_policy(
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=invalid_actions,
                pb_c_init=c_puct_init,
                pb_c_base=c_puct_base,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_fraction=dirichlet_fraction,
            )

        return policy_fn

    # ----- Public API -----

    def improve_at_state(self, state) -> MctsResult:
        """Run num_simulations MCTS sims at a pgx State, return improved policy.

        Self-play loop should call this directly to avoid the slow
        chess_board_to_pgx_state conversion when state is already in pgx form.
        """
        root = self._root_fn(self.params, state)
        invalid_actions = ~state.legal_action_mask[None]

        self._rng, subkey = jax.random.split(self._rng)
        out = self._policy_fn(self.params, subkey, root, invalid_actions)

        weights = np.asarray(out.action_weights[0], dtype=np.float32)
        visits = (weights * self.num_simulations).astype(np.int32)
        try:
            v_plus = float(out.search_tree.summary().value[0])
        except Exception:
            v_plus = 0.0

        return MctsResult(
            pi_improved=weights,
            v_plus=v_plus,
            visit_counts=visits,
        )

    def improve_at(self, board: chess.Board) -> MctsResult:
        """chess.Board entry point; goes through the (slow) FEN bridge."""
        state = pgx_bridge.chess_board_to_pgx_state(board)
        return self.improve_at_state(state)


# ---------------------------------------------------------------------------
# Batched (vmap-over-games) MCTS for Phase 2 self-play
# ---------------------------------------------------------------------------

class MctsArmABatched:
    """MCTS arm processing N root states in parallel via mctx's native batch dim.

    Single MctsArmA instance + batch=1 leaves Blackwell at ~20% utilization in
    self-play because each per-game per-ply sampler call has Python+dispatch
    overhead that dwarfs the small GPU compute. This variant takes batched
    states (leading dim n_games) so a single mctx call covers all games' MCTS
    in one jit graph.

    n_games is fixed at construction time. Build a fresh instance for a
    different batch size.
    """

    def __init__(
        self,
        model,
        params,
        n_games: int,
        num_simulations: int = 100,
        c_puct_init: float = 1.25,
        c_puct_base: float = 19652.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
        rng_seed: int = 0,
        env=None,
    ):
        if not _MCTX_AVAILABLE:
            raise ImportError("mctx and pgx are required for MctsArmABatched")
        if not pgx_bridge.is_available():
            raise ImportError("pgx_bridge / pgx not available")
        if n_games < 1:
            raise ValueError(f"n_games must be >= 1, got {n_games}")

        self.model = model
        self.params = params
        self.n_games = n_games
        self.num_simulations = num_simulations
        self.c_puct_init = c_puct_init
        self.c_puct_base = c_puct_base
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self._rng = jax.random.key(rng_seed)
        self._env = env or _pgx.make("chess")
        self._step_v = jax.vmap(self._env.step, in_axes=(0, 0, 0))
        self._root_fn = jax.jit(self._build_root_fn())
        self._recurrent_fn = self._build_recurrent_fn()
        self._policy_fn = jax.jit(self._build_policy_fn())

    def _build_root_fn(self):
        model = self.model

        def root_fn(params, states):
            # states already batched with leading dim n_games.
            logits, value = model.apply({"params": params}, states.observation)
            return mctx.RootFnOutput(
                prior_logits=logits,
                value=value,
                embedding=states,
            )
        return root_fn

    def _build_recurrent_fn(self):
        # Identical to MctsArmA._build_recurrent_fn — already handles batched
        # action/embedding (mctx passes them through with the batch dim).
        model = self.model
        step_v = self._step_v

        def recurrent_fn(params, rng_key, action, embedding):
            keys = jax.random.split(rng_key, action.shape[0])
            new_state = step_v(embedding, action, keys)
            logits, value = model.apply({"params": params}, new_state.observation)
            actor = 1 - new_state.current_player
            rewards = new_state.rewards
            reward = jnp.take_along_axis(rewards, actor[:, None], axis=1)[:, 0]
            discount = jnp.where(new_state.terminated, 0.0, -1.0)
            logits = jnp.where(new_state.legal_action_mask, logits, -1e9)
            output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            )
            return output, new_state
        return recurrent_fn

    def _build_policy_fn(self):
        recurrent_fn = self._recurrent_fn
        num_simulations = self.num_simulations
        c_puct_init = self.c_puct_init
        c_puct_base = self.c_puct_base
        dirichlet_alpha = self.dirichlet_alpha
        dirichlet_fraction = self.dirichlet_fraction

        def policy_fn(params, rng_key, root, invalid_actions):
            return mctx.muzero_policy(
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=invalid_actions,
                pb_c_init=c_puct_init,
                pb_c_base=c_puct_base,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_fraction=dirichlet_fraction,
            )
        return policy_fn

    def improve_at_states(self, states):
        """Run mctx on n_games batched root states. Returns (pi (N, 4672) float32,
        v_plus (N,) float32)."""
        root = self._root_fn(self.params, states)
        invalid_actions = ~states.legal_action_mask  # (N, 4672), already batched

        self._rng, subkey = jax.random.split(self._rng)
        out = self._policy_fn(self.params, subkey, root, invalid_actions)
        weights = np.asarray(out.action_weights, dtype=np.float32)  # (N, 4672)
        try:
            v_plus = np.asarray(out.search_tree.summary().value, dtype=np.float32)
        except Exception:
            v_plus = np.zeros(self.n_games, dtype=np.float32)
        return weights, v_plus
