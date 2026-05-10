"""Bridge between python-chess and pgx Chess via FEN strings.

Key facts about pgx 2.6's chess module discovered while integrating:

  * `pgx.chess._from_fen(fen) -> State`: load any FEN as a pgx State.
    Round-trips with `_to_fen`. Solves data interop with our existing
    50k FEN-keyed labeled positions.

  * `pgx.chess._to_fen(state) -> str`: dump the absolute FEN regardless
    of pgx's internal player POV.

  * pgx stores `_x.board` in CURRENT-PLAYER POV: when current_player is
    black (color=1), board values are negated and rank-flipped relative
    to absolute white POV. `_to_fen` undoes this for output.

  * pgx action space: 4672 (AlphaZero standard, 64 squares x 73 planes).
    Different from our 4288. A pgx_action <-> our_action bidirectional
    mapping is a separate piece of work; not implemented here.

  * pgx observation: (8, 8, 119) with 8-step history of own/opponent
    pieces + misc. Different from our (pieces (8,8) int8, globals (9,)).
    Net adaptation needed if we want to feed pgx observations directly.

Functions in this module are PYTHON, not JAX-pure. They use FEN strings
as the wire format. Inside a JIT'd mctx recurrent_fn, FEN ops won't work;
see search.py for the architectural options to handle this.
"""

import chess
import jax

# Lazy import to avoid pulling pgx into modules that don't need it.
try:
    # pgx 2.6 deprecated the underscore-prefixed APIs in favor of these.
    from pgx.experimental.chess import from_fen as _pgx_from_fen
    from pgx.experimental.chess import to_fen as _pgx_to_fen
    _PGX_AVAILABLE = True
except ImportError:
    _PGX_AVAILABLE = False


def is_available() -> bool:
    return _PGX_AVAILABLE


def chess_board_to_pgx_state(board: chess.Board):
    """Round-trip a python-chess Board into a pgx Chess State via FEN.

    The returned object is a pgx State; downstream code can read
    `state.legal_action_mask`, `state.observation`, etc.
    """
    if not _PGX_AVAILABLE:
        raise ImportError("pgx not installed")
    return _pgx_from_fen(board.fen())


def pgx_state_to_chess_board(state) -> chess.Board:
    """Round-trip a pgx Chess State to a python-chess Board via FEN."""
    if not _PGX_AVAILABLE:
        raise ImportError("pgx not installed")
    return chess.Board(_pgx_to_fen(state))


def fen_round_trips_through_pgx(fen: str) -> bool:
    """Sanity helper: a FEN survives FEN -> pgx -> FEN unchanged."""
    if not _PGX_AVAILABLE:
        return False
    state = _pgx_from_fen(fen)
    return _pgx_to_fen(state) == fen
