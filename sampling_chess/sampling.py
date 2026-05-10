"""Arm B: importance-weighted Monte Carlo sampling as the policy improvement
operator (doc Algorithm 1).

Given a learned (pi_theta, V_theta) and a root position s_0:

  for i = 1..K:
      sample tau_i = (a_0^(i), ..., a_{k-1}^(i))  from pi_theta
      if game ends within k plies:
          V_i = terminal value, from root POV
      else:
          V_i = V_theta(s_k^(i)), converted to root POV

  pi_sample(a | s_0) = sum_{i: a_0^(i) = a} exp(beta * V_i) / sum_i exp(beta * V_i)
  V^+(s_0)           = sum_i w_i * V_i,   w_i proportional to exp(beta * V_i)

Side-to-move parity:
  After p plies the leaf STM equals the root STM iff p is even. We convert
  V_theta(leaf) to root POV by negating when (plies pushed) is odd.
  Terminal values are computed directly from the outcome's winner.

The forward function `apply_fn` is provided by the caller and abstracts away
the Flax model: it takes a list of chess.Board and returns
(policy_logits: (N, NUM_ACTIONS), values: (N,)) as plain numpy arrays in
side-to-move POV. This keeps sampling.py independent of any specific model.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import chess
import numpy as np

from sampling_chess import board as B

# A network forward call: list of boards -> (policy_logits, values).
ApplyFn = Callable[[List[chess.Board]], Tuple[np.ndarray, np.ndarray]]


@dataclass
class SamplingResult:
    """Output of Algorithm 1 at a single root."""
    pi_sample: np.ndarray   # (NUM_ACTIONS,) float32, sums to 1 over legal moves
    v_plus: float           # SNIS-weighted leaf value, root POV
    leaf_values: np.ndarray # (K,) float32, V_i for each trajectory (root POV)
    first_moves: np.ndarray # (K,) int32, action index of a_0^(i)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over a 1-D logits with a boolean legality mask."""
    masked = np.where(mask, logits, -np.inf)
    masked -= np.nanmax(masked)
    masked = np.where(mask, masked, -np.inf)
    e = np.exp(masked)
    s = e.sum()
    if s == 0:
        # Fully-masked or numerical degenerate: uniform over legal moves.
        u = mask.astype(np.float32)
        return u / max(u.sum(), 1.0)
    return (e / s).astype(np.float32)


def _outcome_value_root_pov(board: chess.Board, root_stm: chess.Color) -> float:
    """Terminal value from the root player's POV, in [-1, +1]."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == root_stm else -1.0


def _stratified_first_moves(legal_indices: List[int], K: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Allocate K trajectories evenly across legal first moves; remainder
    distributed at random over legal indices."""
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


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

def sample_improved_policy(
    root: chess.Board,
    apply_fn: ApplyFn,
    K: int,
    k_plies: int,
    beta: float,
    rng: Optional[np.random.Generator] = None,
    stratified: bool = False,
) -> SamplingResult:
    """Run Algorithm 1 at `root`, return SamplingResult.

    Args:
      root        : root board state s_0 (not modified; we copy).
      apply_fn    : batched network forward; see module docstring for signature.
      K           : number of trajectories.
      k_plies     : truncation length (rollout depth).
      beta        : sharpening / inverse-temperature on leaf values.
      rng         : numpy Generator (defaults to default_rng()).
      stratified  : if True, allocate K trajectories evenly across legal first
                    moves (lower variance per move); if False, sample first
                    moves from pi_theta(.|s_0) (faithful IS proposal).
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if k_plies < 1:
        raise ValueError(f"k_plies must be >= 1, got {k_plies}")
    if rng is None:
        rng = np.random.default_rng()

    root_stm = root.turn
    A = B.NUM_ACTIONS

    # Initialize K trajectory boards as copies of the root.
    boards = [root.copy() for _ in range(K)]
    first_moves = np.full(K, -1, dtype=np.int32)
    plies_pushed = np.zeros(K, dtype=np.int32)
    terminal_value = np.full(K, np.nan, dtype=np.float32)

    # ---- Step 0: pick first move ----
    legal_root = list(root.legal_moves)
    if not legal_root:
        # Root is terminal: trivially return the terminal value, uniform pi.
        pi = np.zeros(A, dtype=np.float32)
        v = _outcome_value_root_pov(root, root_stm)
        return SamplingResult(
            pi_sample=pi, v_plus=v,
            leaf_values=np.full(K, v, dtype=np.float32),
            first_moves=np.full(K, -1, dtype=np.int32),
        )
    legal_indices_root = [B.move_to_index(m) for m in legal_root]

    if stratified:
        first_moves[:] = _stratified_first_moves(legal_indices_root, K, rng)
    else:
        # Prior-weighted sampling from pi_theta(.|s_0).
        logits, _ = apply_fn([root])
        mask = B.legal_action_mask(root)
        probs = _masked_softmax(logits[0], mask)
        first_moves[:] = rng.choice(A, size=K, p=probs)

    for i in range(K):
        boards[i].push(B.index_to_move(int(first_moves[i])))
        plies_pushed[i] = 1
        if boards[i].is_game_over(claim_draw=True):
            terminal_value[i] = _outcome_value_root_pov(boards[i], root_stm)

    # ---- Steps 1..k-1: roll out the rest of the trajectory ----
    for ply in range(1, k_plies):
        active_idx = [i for i in range(K) if np.isnan(terminal_value[i])]
        if not active_idx:
            break
        active_boards = [boards[i] for i in active_idx]
        logits, _ = apply_fn(active_boards)
        for j, i in enumerate(active_idx):
            mask = B.legal_action_mask(boards[i])
            probs = _masked_softmax(logits[j], mask)
            chosen = int(rng.choice(A, p=probs))
            boards[i].push(B.index_to_move(chosen))
            plies_pushed[i] += 1
            if boards[i].is_game_over(claim_draw=True):
                terminal_value[i] = _outcome_value_root_pov(boards[i], root_stm)

    # ---- Step k: V_theta at non-terminal leaves ----
    leaf_values = np.zeros(K, dtype=np.float32)
    nonterm = [i for i in range(K) if np.isnan(terminal_value[i])]
    if nonterm:
        leaf_boards = [boards[i] for i in nonterm]
        _, vs = apply_fn(leaf_boards)
        for j, i in enumerate(nonterm):
            v_leaf = float(vs[j])  # leaf STM POV
            # Convert to root POV: flip sign iff parity differs.
            # plies_pushed[i] is even -> leaf STM == root STM -> no flip.
            # plies_pushed[i] is odd  -> leaf STM != root STM -> flip.
            if plies_pushed[i] % 2 == 1:
                v_leaf = -v_leaf
            leaf_values[i] = v_leaf
    for i in range(K):
        if not np.isnan(terminal_value[i]):
            leaf_values[i] = terminal_value[i]

    # ---- Improvement target via SNIS over leaf values ----
    z = beta * leaf_values
    z -= z.max()  # numerical stability
    w = np.exp(z)
    w /= w.sum()

    pi_sample = np.zeros(A, dtype=np.float32)
    np.add.at(pi_sample, first_moves, w.astype(np.float32))

    v_plus = float((w * leaf_values).sum())

    return SamplingResult(
        pi_sample=pi_sample,
        v_plus=v_plus,
        leaf_values=leaf_values.astype(np.float32),
        first_moves=first_moves,
    )
