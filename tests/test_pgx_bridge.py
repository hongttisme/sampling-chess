"""Tests for the python-chess <-> pgx FEN bridge."""

import chess
import pytest

pgx_bridge = pytest.importorskip("sampling_chess.pgx_bridge")
pytest.importorskip("pgx")

if not pgx_bridge.is_available():
    pytest.skip("pgx not available", allow_module_level=True)


def test_startpos_round_trips():
    fen = chess.STARTING_FEN
    state = pgx_bridge.chess_board_to_pgx_state(chess.Board(fen))
    bd = pgx_bridge.pgx_state_to_chess_board(state)
    assert bd.fen() == fen


def test_after_e4_round_trips():
    bd = chess.Board()
    bd.push_san("e4")
    state = pgx_bridge.chess_board_to_pgx_state(bd)
    bd2 = pgx_bridge.pgx_state_to_chess_board(state)
    assert bd2.fen() == bd.fen()


def test_castling_rights_preserved():
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    assert pgx_bridge.fen_round_trips_through_pgx(fen)


def test_en_passant_preserved_when_capturable():
    """pgx uses strict FEN: EP target only set if EP capture is actually
    available. Construct a position where black just played d5 and white
    really could play exd6 e.p., then check the EP target round-trips.
    """
    bd = chess.Board()
    for san in ["e4", "Nf6", "e5", "d5"]:
        bd.push_san(san)
    fen = bd.fen()
    # White can play exd6 e.p. -> python-chess marks d6 as EP target.
    assert "d6" in fen.split()[3]
    assert pgx_bridge.fen_round_trips_through_pgx(fen)


def test_position_round_trips_when_python_chess_strips_ep():
    """python-chess 1.11+ defaults to legal-EP FEN: after 1.e4 with no
    capturable EP, the EP square is dropped. Both libraries agree, so
    semantic round-trip via FEN is byte-identical here."""
    bd = chess.Board()
    bd.push_san("e4")
    fen = bd.fen()  # already strict, no EP target
    assert pgx_bridge.fen_round_trips_through_pgx(fen)


def test_black_to_move():
    """Black-to-move FEN survives: pgx does internal POV flip but _to_fen undoes it."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert pgx_bridge.fen_round_trips_through_pgx(fen)


def test_promotion_position():
    fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
    assert pgx_bridge.fen_round_trips_through_pgx(fen)


def test_pgx_state_legal_count_matches_python_chess():
    """A position's legal-move count via pgx must match python-chess."""
    bd = chess.Board()
    state = pgx_bridge.chess_board_to_pgx_state(bd)
    n_legal_pgx = int(state.legal_action_mask.sum())
    n_legal_pc = sum(1 for _ in bd.legal_moves)
    assert n_legal_pgx == n_legal_pc == 20


def test_pgx_state_legal_count_after_move():
    bd = chess.Board()
    bd.push_san("e4")
    bd.push_san("e5")
    state = pgx_bridge.chess_board_to_pgx_state(bd)
    n_legal_pgx = int(state.legal_action_mask.sum())
    n_legal_pc = sum(1 for _ in bd.legal_moves)
    assert n_legal_pgx == n_legal_pc
