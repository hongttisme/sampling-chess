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
import jax.numpy as jnp

# Lazy import to avoid pulling pgx into modules that don't need it.
try:
    # pgx 2.6 deprecated the underscore-prefixed APIs in favor of these.
    from pgx.experimental.chess import from_fen as _pgx_from_fen
    from pgx.experimental.chess import to_fen as _pgx_to_fen
    from pgx._src.games.chess import Action as _PgxAction
    _PGX_AVAILABLE = True
except ImportError:
    _PGX_AVAILABLE = False
    _PgxAction = None  # type: ignore


# pgx Chess action space size (AlphaZero standard).
PGX_NUM_ACTIONS = 4672


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


# ---------------------------------------------------------------------------
# Action mapping: chess.Move <-> pgx action_idx in [0, 4672)
# ---------------------------------------------------------------------------
#
# pgx encodes actions in CURRENT-PLAYER POV with vertical-flip:
#   - When black is to move, ranks are mirrored (rank 0 from white POV =
#     rank 7 in pgx coords for black).
#   - Files are NOT mirrored.
#   - pgx square index = file * 8 + rank (transposed from python-chess's
#     square = rank * 8 + file).
#
# Action label = from_sq * 73 + plane, where plane is:
#   - planes 0-8  : UNDERPROMOTION. plane = piece_idx * 3 + direction_idx
#                   piece_idx in {0=ROOK, 1=BISHOP, 2=KNIGHT}
#                   direction_idx in {0=forward, 1=right (file+1),
#                                     2=left (file-1)}, in current-player POV
#   - planes 9-72 : queen-like + knight moves (queen promotion uses these too)
#
# pgx.Action._to_label INTENTIONALLY ignores the underpromotion field — it
# always returns the queen-like plane via TO_PLANE[from, to]. So we
# construct underpromotion labels manually and only delegate to _to_label
# for non-underpromotion moves.

_PROMO_TO_PGX_PIECE_IDX = {
    chess.ROOK: 0,
    chess.BISHOP: 1,
    chess.KNIGHT: 2,
    # chess.QUEEN intentionally absent — queen promo uses queen-like planes.
}
_PGX_PIECE_IDX_TO_PROMO = {v: k for k, v in _PROMO_TO_PGX_PIECE_IDX.items()}


def _python_to_pgx_square(sq: int, turn: chess.Color) -> int:
    """python-chess square (rank*8+file, absolute) -> pgx square (file*8+rank,
    current-player POV with vertical flip for black)."""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    if turn == chess.BLACK:
        rank = 7 - rank
    return file * 8 + rank


def _pgx_to_python_square(pgx_sq: int, turn: chess.Color) -> int:
    file = pgx_sq // 8
    rank = pgx_sq % 8
    if turn == chess.BLACK:
        rank = 7 - rank
    return chess.square(file, rank)


def _file_change_to_direction_idx(from_file: int, to_file: int) -> int:
    """Pawn promo direction in pgx encoding: 0=forward, 1=right, 2=left."""
    df = to_file - from_file
    if df == 0:
        return 0
    if df == 1:
        return 1
    if df == -1:
        return 2
    raise ValueError(f"non-pawn-promo file change: {from_file} -> {to_file}")


def _direction_idx_to_file_change(direction: int) -> int:
    return {0: 0, 1: 1, 2: -1}[direction]


def chess_move_to_pgx_action(move: chess.Move, turn: chess.Color) -> int:
    """Convert a chess.Move (absolute coords) to a pgx action label.

    `turn` is whose move it is in the source position (chess.WHITE or
    chess.BLACK); pgx's POV-flip depends on it.
    """
    if not _PGX_AVAILABLE:
        raise ImportError("pgx not installed")
    pgx_from = _python_to_pgx_square(move.from_square, turn)
    pgx_to = _python_to_pgx_square(move.to_square, turn)

    # Underpromotion (R/B/N): manually compute label in plane 0-8 region.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = _PROMO_TO_PGX_PIECE_IDX.get(move.promotion)
        if piece_idx is None:
            raise ValueError(f"unsupported promotion piece: {move.promotion}")
        from_file = pgx_from // 8
        to_file = pgx_to // 8
        direction_idx = _file_change_to_direction_idx(from_file, to_file)
        plane = piece_idx * 3 + direction_idx
        return pgx_from * 73 + plane

    # Queen-promo or non-promo move: use Action._to_label (TO_PLANE lookup).
    action = _PgxAction(
        from_=jnp.int32(pgx_from),
        to=jnp.int32(pgx_to),
        underpromotion=jnp.int32(-1),
    )
    return int(action._to_label())


def pgx_action_to_chess_move(label: int, turn: chess.Color) -> chess.Move:
    """Convert a pgx action label to a chess.Move (absolute coords).

    `turn` is whose move it is in the source position.
    """
    if not _PGX_AVAILABLE:
        raise ImportError("pgx not installed")
    if not 0 <= label < PGX_NUM_ACTIONS:
        raise ValueError(f"label out of range: {label}")

    pgx_from = label // 73
    plane = label % 73

    if plane < 9:
        # Underpromotion. piece = plane // 3 (0=R, 1=B, 2=N),
        # direction = plane % 3 (0=fwd, 1=right, 2=left).
        piece_idx = plane // 3
        direction_idx = plane % 3
        promotion = _PGX_PIECE_IDX_TO_PROMO[piece_idx]
        # Pawn at rank 6 in pgx coords (= 7th rank of current player).
        from_file = pgx_from // 8
        from_rank = pgx_from % 8
        if from_rank != 6:
            raise ValueError(
                f"label {label} is underpromo but from_pgx rank != 6 "
                f"(from_pgx={pgx_from})"
            )
        to_file = from_file + _direction_idx_to_file_change(direction_idx)
        if not 0 <= to_file < 8:
            raise ValueError(f"label {label} decodes to off-board to_file={to_file}")
        to_rank = 7  # promotes to last rank
        pgx_to = to_file * 8 + to_rank
    else:
        # Use Action._from_label for queen-like + knight planes.
        action = _PgxAction._from_label(jnp.int32(label))
        pgx_from_check = int(action.from_)
        pgx_to = int(action.to)
        if pgx_to < 0:
            raise ValueError(
                f"label {label} decodes to invalid pgx_to={pgx_to} (illegal label)"
            )
        assert pgx_from_check == pgx_from
        promotion = None

    py_from = _pgx_to_python_square(pgx_from, turn)
    py_to = _pgx_to_python_square(pgx_to, turn)

    if promotion is None:
        # Could be either a non-promo move OR a queen-promo (queen-like plane
        # that lands on promo rank). Detect by pawn-on-promo-rank heuristic.
        py_from_rank = chess.square_rank(py_from)
        py_to_rank = chess.square_rank(py_to)
        is_promo_rank = (
            (turn == chess.WHITE and py_from_rank == 6 and py_to_rank == 7)
            or (turn == chess.BLACK and py_from_rank == 1 and py_to_rank == 0)
        )
        if is_promo_rank:
            promotion = chess.QUEEN

    return chess.Move(py_from, py_to, promotion=promotion)

