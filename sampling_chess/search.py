"""Arm A: MCTS via DeepMind mctx + pgx Chess.

==========================================================================
SCAFFOLD STATE: Bridge utilities + architectural decision documented;
                actual mctx wiring deferred pending design decision.
==========================================================================

Doc 6.1 instruction: "use mctx; do not write your own MCTS." Three concrete
integration paths exist. They differ in (a) what's JAX-pure inside mctx's
JIT'd recurrent_fn, (b) how state encodes for the network, and (c) how
much existing code (board.py, net.py, labeled .npz, sampling.py) is reused.

----------------------------------------------------------------------
Plan A — pgx-native: adopt pgx's encoding throughout.
----------------------------------------------------------------------
  Net input  : pgx observation (8, 8, 119) instead of our (8x8 int8 +
               9-dim global).
  Action     : pgx 4672 instead of our 4288.
  Recurrent  : trivial (already JAX-pure since pgx is JAX-native).
  Cost       : net.py needs a new input layer + new action head; the 50k
               labeled positions still load via FEN, but their move
               indices live in our 4288 action space and need a one-time
               conversion to pgx's 4672 (or just discard policy targets
               and keep value targets only). Effectively redo Phase 1.
  Pro        : Cleanest mctx integration. mctx + pgx examples in the
               wild use exactly this stack.
  Con        : Throws away the work that wired up our action encoding,
               legal_action_mask, and labeled policy targets.

----------------------------------------------------------------------
Plan B — JAX-pure converters: keep our encoding, write JAX bridges.
----------------------------------------------------------------------
  Net input  : our (pieces (8,8) int8, globals (9,)) — unchanged.
  Action     : our 4288 — unchanged.
  Recurrent  : extracts our (pieces, globals) from pgx_state._x; maps
               action 4288 <-> 4672. Both must be JAX-pure to live
               inside the JIT'd recurrent_fn.
  Cost       : ~3-5h of careful JAX-pure code.
                * pgx stores _x.board in CURRENT-PLAYER POV (color=0
                  unmodified, color=1 negate values + flip ranks). A
                  JAX-pure converter must conditionally apply this flip.
                * Action space mapping: pgx's 4672 = 64 squares x 73
                  planes (AlphaZero); our 4288 = 4096 from-to + 192
                  promo channels. The (from, to, promo) -> pgx_idx
                  function and inverse are tedious but pure-numpy/JAX.
  Pro        : Reuses Phase 1 SL data + net + sampling.py without
               touching them.
  Con        : Most engineering work; brittle if pgx changes its
               internal POV convention.

----------------------------------------------------------------------
Plan D — host-callback: keep our encoding, run transitions in Python.
----------------------------------------------------------------------
  Net input  : our encoding.
  Action     : our 4288.
  Recurrent  : uses jax.experimental.io_callback inside recurrent_fn to
               call python-chess.Board.push() and our board.py encoders.
  Cost       : ~1-2h. No JAX-pure converters needed.
  Pro        : Fastest path to a working Arm A; reuses everything.
  Con        : Defeats mctx's vmap+JIT speedup. Per-rollout cost is
               Python-loop bound, so wall-clock is ~5-20x mctx's native
               speed. Phase 2 budget tightens accordingly. Also adds
               a tracing-vs-runtime gotcha (io_callback runs only at
               actual call time, not during JIT trace).

----------------------------------------------------------------------
Recommendation (to decide with fresh head, not at midnight):
  * For a credible MCTS baseline at the doc's promised throughput,
    Plan A is correct. Phase 1 redo cost is ~half day and we already
    have all the labeling/training infrastructure to repeat it.
  * For prototype-only "does the comparison even make sense", Plan D
    is fastest and reuses everything.
  * Plan B is the worst trade: most work, brittlest code, no clear
    long-term win over A.

Until that decision is made, this module exposes the bridge utilities
needed by all three plans (via pgx_bridge.py) and a typed stub class
for the eventual MCTS operator. Do not import from this module in
self-play / eval until a plan is chosen.
"""

from dataclasses import dataclass
from typing import Optional

import chess
import numpy as np

from sampling_chess import pgx_bridge


@dataclass
class MctsResult:
    """Output of one MCTS search, parallel to sampling.SamplingResult."""
    pi_improved: np.ndarray  # (NUM_ACTIONS,) float32, sums to 1 over legal moves
    v_plus: float            # value bootstrap from search tree
    visit_counts: np.ndarray  # (NUM_ACTIONS,) int32, visits per action


class MctsArmA:
    """Stubbed MCTS arm; wiring depends on plan A/B/D selection."""

    def __init__(self, num_simulations: int = 100, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        if not pgx_bridge.is_available():
            raise ImportError(
                "MctsArmA requires pgx; install with `pip install pgx`."
            )

    def improve_at(self, board: chess.Board, *, params: Optional[dict] = None) -> MctsResult:
        """Run num_simulations MCTS sims at `board`, return improved policy.

        Not implemented yet. See the module docstring for the architectural
        decision blocking this method.
        """
        raise NotImplementedError(
            "MctsArmA.improve_at: choose Plan A / B / D in the module "
            "docstring before implementing. Bridge utilities are ready in "
            "sampling_chess.pgx_bridge."
        )
