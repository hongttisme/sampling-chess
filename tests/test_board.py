"""Tests for board encoding (pieces, global features, move <-> index)."""

import chess
import numpy as np
import pytest

from sampling_chess import board as B


# ---------- piece planes ----------

def test_planes_startpos_layout():
    b = chess.Board()
    p = B.board_to_planes(b)
    assert p.shape == (8, 8)
    assert p.dtype == np.int8
    # White pawns on rank 1 (2nd rank), black pawns on rank 6 (7th rank)
    assert (p[1] == 1).all()
    assert (p[6] == 7).all()
    # Back ranks: R N B Q K B N R
    assert tuple(int(x) for x in p[0]) == (4, 2, 3, 5, 6, 3, 2, 4)
    assert tuple(int(x) for x in p[7]) == (10, 8, 9, 11, 12, 9, 8, 10)
    # Empty middle
    assert (p[2:6] == 0).all()


def test_planes_after_e4():
    b = chess.Board()
    b.push_san("e4")
    p = B.board_to_planes(b)
    # e2 (rank 1, file 4) is empty
    assert p[1, 4] == 0
    # e4 (rank 3, file 4) has white pawn
    assert p[3, 4] == 1


# ---------- global features ----------

def test_global_startpos():
    b = chess.Board()
    g = B.board_to_global(b)
    assert g.shape == (9,) and g.dtype == np.float32
    assert g[0] == 0.0  # white to move
    assert g[1:5].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert g[5] == 0.0  # no ep
    assert g[7] == 0.0  # halfmove clock
    assert g[8] == pytest.approx(1 / 200.0)  # fullmove


def test_global_ep_after_e4():
    b = chess.Board()
    b.push_san("e4")
    g = B.board_to_global(b)
    assert g[0] == 1.0  # black to move
    assert g[5] == 1.0  # ep present
    # ep target square is e3, file index 4
    assert g[6] == pytest.approx(4 / 7.0)


def test_global_castling_loss():
    b = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    b.push_san("Kd1")  # white king moves, loses both castling rights
    g = B.board_to_global(b)
    assert g[1] == 0.0 and g[2] == 0.0
    assert g[3] == 1.0 and g[4] == 1.0


# ---------- move <-> index round-trip ----------

def test_roundtrip_all_nonpromo():
    """All 4096 from-to pairs round-trip; same-square encoded but never legal."""
    for from_sq in range(64):
        for to_sq in range(64):
            mv = chess.Move(from_sq, to_sq)
            idx = B.move_to_index(mv)
            assert idx == from_sq * 64 + to_sq
            assert B.index_to_move(idx) == mv


def test_roundtrip_all_promotions():
    promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    # White: rank 6 -> rank 7
    for from_file in range(8):
        for direction in (-1, 0, 1):
            to_file = from_file + direction
            if not 0 <= to_file < 8:
                continue
            for promo in promo_pieces:
                mv = chess.Move(6 * 8 + from_file, 7 * 8 + to_file, promotion=promo)
                idx = B.move_to_index(mv)
                assert B._PROMO_OFFSET_WHITE <= idx < B._PROMO_OFFSET_BLACK
                assert B.index_to_move(idx) == mv
    # Black: rank 1 -> rank 0
    for from_file in range(8):
        for direction in (-1, 0, 1):
            to_file = from_file + direction
            if not 0 <= to_file < 8:
                continue
            for promo in promo_pieces:
                mv = chess.Move(1 * 8 + from_file, 0 * 8 + to_file, promotion=promo)
                idx = B.move_to_index(mv)
                assert B._PROMO_OFFSET_BLACK <= idx < B.NUM_ACTIONS
                assert B.index_to_move(idx) == mv


def test_index_range_for_legal_moves():
    """Every legal move at startpos has a valid index."""
    b = chess.Board()
    for mv in b.legal_moves:
        idx = B.move_to_index(mv)
        assert 0 <= idx < B.NUM_ACTIONS


def test_total_action_space():
    assert B.NUM_ACTIONS == 4288
    assert B.NUM_NONPROMO_MOVES == 4096
    assert B.NUM_PROMO_PER_COLOR == 96


# ---------- legal action mask ----------

def test_mask_startpos():
    b = chess.Board()
    mask = B.legal_action_mask(b)
    assert mask.shape == (B.NUM_ACTIONS,)
    assert mask.dtype == bool
    assert int(mask.sum()) == 20  # 16 pawn + 4 knight moves at startpos


def test_mask_promotion_position():
    """White pawn on a7 with adjacent enemy: 4 underpromotion moves available."""
    b = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    mask = B.legal_action_mask(b)
    promo_legal = [m for m in b.legal_moves if m.promotion is not None]
    assert len(promo_legal) == 4  # N, B, R, Q to a8
    for mv in promo_legal:
        assert mask[B.move_to_index(mv)]


def test_mask_no_double_count():
    """Legal-mask count equals number of legal moves (no aliasing)."""
    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/8/8/8/8/8/8/4k2K b - - 0 1",  # endgame
        "rnbq1bnr/pppPpppp/8/8/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 5",  # promo available
    ]
    for fen in fens:
        b = chess.Board(fen)
        n_legal = sum(1 for _ in b.legal_moves)
        mask = B.legal_action_mask(b)
        assert int(mask.sum()) == n_legal, f"mismatch at FEN: {fen}"
