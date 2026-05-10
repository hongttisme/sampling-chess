"""Board encoding utilities: position <-> tensor, move <-> index.

The transformer in net.py consumes:
  - pieces: int8 array of shape (8, 8) with values in [0, 12]
      0 = empty, 1..6 = white P/N/B/R/Q/K, 7..12 = black P/N/B/R/Q/K.
  - global: float32 vector of shape (9,)
      [stm, w_oo, w_ooo, b_oo, b_ooo, ep_present, ep_file/7, hm/100, fm/200].

Move encoding is a flat index in [0, NUM_ACTIONS):
  indices 0..4095   : non-promo moves, idx = from_sq * 64 + to_sq
  indices 4096..4191: white pawn promotions (8 files x 3 dirs x 4 pieces)
  indices 4192..4287: black pawn promotions (same layout)
This is wider than the 1858 Leela move space; we rely on legal_action_mask
to zero out impossible moves before softmax, per doc section 4.1.
"""

import chess
import numpy as np

# ---------------------------------------------------------------------------
# Piece-plane encoding
# ---------------------------------------------------------------------------

NUM_PIECE_TYPES = 13  # incl. empty
BOARD_SHAPE = (8, 8)

_PIECE_TO_IDX = {
    (chess.PAWN, chess.WHITE): 1,
    (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,
    (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5,
    (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7,
    (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,
    (chess.ROOK, chess.BLACK): 10,
    (chess.QUEEN, chess.BLACK): 11,
    (chess.KING, chess.BLACK): 12,
}


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode piece placement as int8 array of shape (8, 8).

    planes[rank, file] indexes by python-chess square = rank * 8 + file,
    so planes[0] is white's 1st rank and planes[7] is black's 1st rank.
    """
    planes = np.zeros(BOARD_SHAPE, dtype=np.int8)
    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        planes[rank, file] = _PIECE_TO_IDX[(piece.piece_type, piece.color)]
    return planes


# ---------------------------------------------------------------------------
# Global features
# ---------------------------------------------------------------------------

NUM_GLOBAL_FEATURES = 9


def board_to_global(board: chess.Board) -> np.ndarray:
    g = np.zeros(NUM_GLOBAL_FEATURES, dtype=np.float32)
    g[0] = float(board.turn == chess.BLACK)  # 0 = white-to-move, 1 = black
    g[1] = float(board.has_kingside_castling_rights(chess.WHITE))
    g[2] = float(board.has_queenside_castling_rights(chess.WHITE))
    g[3] = float(board.has_kingside_castling_rights(chess.BLACK))
    g[4] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        g[5] = 1.0
        g[6] = chess.square_file(board.ep_square) / 7.0
    g[7] = min(board.halfmove_clock, 100) / 100.0
    g[8] = min(board.fullmove_number, 200) / 200.0
    return g


def encode_board(board: chess.Board) -> dict:
    """Convenience wrapper: {'pieces': (8,8) int8, 'global': (9,) float32}."""
    return {
        "pieces": board_to_planes(board),
        "global": board_to_global(board),
    }


# ---------------------------------------------------------------------------
# Move encoding
# ---------------------------------------------------------------------------

NUM_NONPROMO_MOVES = 64 * 64  # 4096
NUM_PROMO_PER_COLOR = 8 * 3 * 4  # files x directions x pieces = 96
NUM_ACTIONS = NUM_NONPROMO_MOVES + 2 * NUM_PROMO_PER_COLOR  # 4288

_PROMO_OFFSET_WHITE = NUM_NONPROMO_MOVES
_PROMO_OFFSET_BLACK = _PROMO_OFFSET_WHITE + NUM_PROMO_PER_COLOR

_PROMO_PIECE_TO_IDX = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 3,
}
_IDX_TO_PROMO_PIECE = {v: k for k, v in _PROMO_PIECE_TO_IDX.items()}


def move_to_index(move: chess.Move) -> int:
    """Encode chess.Move as integer index in [0, NUM_ACTIONS)."""
    if move.promotion is None:
        return move.from_square * 64 + move.to_square

    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    direction = to_file - from_file + 1  # -1,0,+1 -> 0,1,2
    if direction not in (0, 1, 2):
        raise ValueError(f"Invalid promo direction in move {move}")
    promo_piece_idx = _PROMO_PIECE_TO_IDX[move.promotion]

    if from_rank == 6:
        offset = _PROMO_OFFSET_WHITE
    elif from_rank == 1:
        offset = _PROMO_OFFSET_BLACK
    else:
        raise ValueError(f"Promotion from non-7th/2nd rank: {move}")

    return offset + from_file * 12 + direction * 4 + promo_piece_idx


def index_to_move(idx: int) -> chess.Move:
    """Decode an integer index back to a chess.Move (board-context-free)."""
    if not 0 <= idx < NUM_ACTIONS:
        raise ValueError(f"Index out of range: {idx}")

    if idx < NUM_NONPROMO_MOVES:
        from_sq, to_sq = divmod(idx, 64)
        return chess.Move(from_sq, to_sq)

    if idx < _PROMO_OFFSET_BLACK:
        rel = idx - _PROMO_OFFSET_WHITE
        from_rank, to_rank = 6, 7
    else:
        rel = idx - _PROMO_OFFSET_BLACK
        from_rank, to_rank = 1, 0

    from_file, dir_promo = divmod(rel, 12)
    direction, promo_piece_idx = divmod(dir_promo, 4)
    to_file = from_file + direction - 1
    if not 0 <= to_file < 8:
        raise ValueError(f"Decoded move has off-board to_file: idx={idx}")

    from_sq = from_rank * 8 + from_file
    to_sq = to_rank * 8 + to_file
    promo_piece = _IDX_TO_PROMO_PIECE[promo_piece_idx]
    return chess.Move(from_sq, to_sq, promotion=promo_piece)


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape (NUM_ACTIONS,): True iff index is legal at `board`."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask
