"""Inference-time MCTS over chess.Board, backed by a trained ChessTransformer.

Pure Python tree (no pgx) — keeps the deployment surface simple and matches the
SL training encoding (board.py's pieces + globals + legal_action_mask).

PUCT selection rule (AlphaZero):
    a* = argmax_a  Q(s, a) + c_puct * P(s, a) * sqrt(N_total) / (1 + N(s, a))

where Q is the mean child-value from the CURRENT PLAYER's POV.

Two-player handling: at each ply down the tree, the value perspective flips —
when we back up a leaf value v evaluated at the leaf's side-to-move, we
negate it at every other ply on the way up.

Public entry point:

    mcts_search(board, model, params, num_simulations=100, ...)
        -> MctsResult(best_move, action_probs, root_value, visit_counts)

`action_probs` is the visit-count distribution over the 4288-action space
(softmax-temp not applied — caller can re-temper or argmax as desired).
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import chess
import jax
import jax.numpy as jnp
import numpy as np

from sampling_chess import board as B
from sampling_chess.net import apply_legal_mask


@dataclass
class MctsResult:
    best_move: chess.Move
    action_probs: np.ndarray   # (NUM_ACTIONS,) float32, sums to 1 over legal moves
    root_value: float          # search-derived value at root, current-player POV
    visit_counts: np.ndarray   # (NUM_ACTIONS,) int32
    num_simulations: int
    wall_clock_sec: float


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    prior: float                         # P(s, a) -- prior under parent's policy
    to_play: int                         # 0 = white, 1 = black; whose turn AT this node
    visits: int = 0
    value_sum: float = 0.0               # cumulative value, from this-node's-side POV
    children: dict = field(default_factory=dict)  # action_idx -> _Node
    is_terminal: bool = False
    terminal_value: float = 0.0          # set if is_terminal — from this-node's-side POV
    expanded: bool = False
    legal_actions: list = field(default_factory=list)  # ints into NUM_ACTIONS
    legal_priors: dict = field(default_factory=dict)   # action_idx -> prior

    @property
    def Q(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


# ---------------------------------------------------------------------------
# Model wrapper: jit'd single-position forward
# ---------------------------------------------------------------------------

def make_eval_fn(model, params):
    """Returns eval_fn(board) -> (legal_logits np (NUM_ACTIONS,), value float).

    Forward call is JIT-compiled once; subsequent calls are cached. legal_logits
    is masked (-inf on illegal) but NOT softmaxed — caller softmaxes for priors.
    """
    @jax.jit
    def _fwd(p, pieces, globals_, mask):
        logits, value = model.apply(
            {"params": p}, pieces[None].astype(jnp.int32), globals_[None]
        )
        masked = apply_legal_mask(logits[0], mask)
        return masked, value[0]

    def eval_fn(board: chess.Board):
        pieces = jnp.asarray(B.board_to_planes(board))
        globals_ = jnp.asarray(B.board_to_global(board))
        mask = jnp.asarray(B.legal_action_mask(board))
        masked_logits, value = _fwd(params, pieces, globals_, mask)
        return np.asarray(masked_logits), float(value)

    return eval_fn


# ---------------------------------------------------------------------------
# Core MCTS
# ---------------------------------------------------------------------------

def _expand(node: _Node, board: chess.Board, eval_fn) -> float:
    """Run net at this node, fill children priors. Returns leaf value (this-side POV)."""
    if board.is_game_over(claim_draw=True):
        node.is_terminal = True
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            v = 0.0
        else:
            # Net's value is in side-to-move POV, but for terminal we get it from outcome.
            # If the side-to-move at this node is the winner -> +1 (impossible since
            # they have no legal moves), so winner is the OTHER side -> -1.
            v = -1.0 if outcome.winner != board.turn else 1.0
        node.terminal_value = v
        node.expanded = True
        return v

    masked_logits, value = eval_fn(board)
    # Softmax over masked logits to get priors over the legal action space.
    m = float(masked_logits.max())
    expv = np.exp(masked_logits - m)
    expv[~np.isfinite(masked_logits)] = 0.0
    priors = expv / max(expv.sum(), 1e-12)

    legal_idx = []
    for mv in board.legal_moves:
        a = B.move_to_index(mv)
        legal_idx.append(a)
        node.legal_priors[a] = float(priors[a])

    node.legal_actions = legal_idx
    node.expanded = True
    return value


def _select_child(node: _Node, c_puct: float) -> int:
    """PUCT child selection. Returns action_idx of chosen child."""
    n_total = max(node.visits, 1)
    sqrt_n = math.sqrt(n_total)
    best_score = -float("inf")
    best_action = node.legal_actions[0]
    for a in node.legal_actions:
        child = node.children.get(a)
        if child is None:
            visits = 0
            q = 0.0
        else:
            visits = child.visits
            # Child's Q is from CHILD's side POV; we want from node's side POV
            # → negate (2-player zero-sum).
            q = -child.Q
        prior = node.legal_priors[a]
        u = c_puct * prior * sqrt_n / (1 + visits)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


def mcts_search(
    board: chess.Board,
    model,
    params,
    *,
    num_simulations: int = 100,
    c_puct: float = 1.25,
    eval_fn=None,
) -> MctsResult:
    """Run `num_simulations` MCTS sims rooted at `board`. Returns MctsResult.

    `eval_fn` can be passed to share a JIT'd forward function across many calls
    (cheaper than rebuilding per call). If None, one is built internally.
    """
    if eval_fn is None:
        eval_fn = make_eval_fn(model, params)

    t0 = time.time()
    root = _Node(prior=1.0, to_play=0 if board.turn == chess.WHITE else 1)
    root_value = _expand(root, board, eval_fn)
    root.visits = 1
    root.value_sum = root_value

    for _ in range(num_simulations):
        node = root
        path = [node]
        sim_board = board.copy()

        # Selection: descend until we hit a leaf (un-expanded child or terminal).
        while node.expanded and not node.is_terminal:
            a = _select_child(node, c_puct)
            child = node.children.get(a)
            if child is None:
                # Edge not yet visited — create child node and expand.
                sim_board.push(B.index_to_move(a))
                child = _Node(
                    prior=node.legal_priors[a],
                    to_play=1 - node.to_play,
                )
                node.children[a] = child
                value_at_child = _expand(child, sim_board, eval_fn)
                child.visits = 1
                child.value_sum = value_at_child
                path.append(child)
                # Backup: propagate from child up to root, alternating sign.
                v_for_node = value_at_child
                for p_node in reversed(path[:-1]):
                    v_for_node = -v_for_node
                    p_node.visits += 1
                    p_node.value_sum += v_for_node
                break
            else:
                sim_board.push(B.index_to_move(a))
                node = child
                path.append(node)
        else:
            # Reached terminal node — backup its value.
            if node.is_terminal:
                v = node.terminal_value
                for p_node in reversed(path):
                    p_node.visits += 1
                    p_node.value_sum += v
                    v = -v

    # Build action probs from root visit counts.
    visits_arr = np.zeros(B.NUM_ACTIONS, dtype=np.int32)
    for a, child in root.children.items():
        visits_arr[a] = child.visits
    total = visits_arr.sum()
    if total > 0:
        probs = visits_arr.astype(np.float32) / total
    else:
        # Fallback: uniform over legal moves
        probs = np.zeros(B.NUM_ACTIONS, dtype=np.float32)
        for a in root.legal_actions:
            probs[a] = 1.0 / max(len(root.legal_actions), 1)

    best_action = int(visits_arr.argmax())
    best_move = B.index_to_move(best_action)
    root_q = root.Q

    return MctsResult(
        best_move=best_move,
        action_probs=probs,
        root_value=root_q,
        visit_counts=visits_arr,
        num_simulations=num_simulations,
        wall_clock_sec=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Per-sim benchmark — used by the UI to estimate "thinking time"
# ---------------------------------------------------------------------------

def benchmark_per_sim(model, params, *, n_sims: int = 50) -> dict:
    """Time a single mcts_search at startpos and return per-sim cost stats.

    First call also pays JIT compile cost; benchmark does a 1-sim warm-up,
    then times the actual run.
    """
    eval_fn = make_eval_fn(model, params)
    bd = chess.Board()

    # Warm JIT.
    _ = mcts_search(bd, model, params, num_simulations=1, eval_fn=eval_fn)

    out = mcts_search(bd, model, params, num_simulations=n_sims, eval_fn=eval_fn)
    per_sim_ms = 1000.0 * out.wall_clock_sec / max(n_sims, 1)
    return {
        "n_sims": n_sims,
        "wall_clock_sec": out.wall_clock_sec,
        "per_sim_ms": per_sim_ms,
        "best_move": out.best_move.uci(),
        "root_value": out.root_value,
    }
