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


# ----- chess.Move <-> pgx_action -----

import numpy as np  # noqa: E402


def _pgx_legal_set(state) -> set:
    return set(int(i) for i in np.where(np.array(state.legal_action_mask))[0])


def test_white_legal_moves_map_to_legal_pgx_actions():
    """All 20 startpos legal moves map to labels that are in pgx's legal mask."""
    bd = chess.Board()
    state = pgx_bridge.chess_board_to_pgx_state(bd)
    pgx_legal = _pgx_legal_set(state)
    for mv in bd.legal_moves:
        label = pgx_bridge.chess_move_to_pgx_action(mv, bd.turn)
        assert label in pgx_legal, f"move {mv.uci()} -> label {label} not legal in pgx"


def test_black_legal_moves_map_to_legal_pgx_actions():
    """Same for black-to-move position (verifies POV flip)."""
    bd = chess.Board()
    bd.push_san("e4")
    state = pgx_bridge.chess_board_to_pgx_state(bd)
    pgx_legal = _pgx_legal_set(state)
    for mv in bd.legal_moves:
        label = pgx_bridge.chess_move_to_pgx_action(mv, bd.turn)
        assert label in pgx_legal, f"black move {mv.uci()} -> {label} not legal in pgx"


def test_action_round_trip_white_startpos():
    """chess.Move -> label -> chess.Move round-trips at startpos."""
    bd = chess.Board()
    for mv in bd.legal_moves:
        label = pgx_bridge.chess_move_to_pgx_action(mv, bd.turn)
        back = pgx_bridge.pgx_action_to_chess_move(label, bd.turn)
        assert back == mv, f"mv={mv} -> label={label} -> back={back}"


def test_action_round_trip_black_after_e4():
    bd = chess.Board()
    bd.push_san("e4")
    for mv in bd.legal_moves:
        label = pgx_bridge.chess_move_to_pgx_action(mv, bd.turn)
        back = pgx_bridge.pgx_action_to_chess_move(label, bd.turn)
        assert back == mv, f"mv={mv} -> label={label} -> back={back}"


def test_promotion_round_trip_all_pieces():
    """Each promotion piece (Q, R, B, N) round-trips correctly via pgx labels."""
    bd = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")  # white pawn ready to promote
    promo_moves = [m for m in bd.legal_moves if m.promotion is not None]
    assert len(promo_moves) == 4
    for mv in promo_moves:
        label = pgx_bridge.chess_move_to_pgx_action(mv, bd.turn)
        back = pgx_bridge.pgx_action_to_chess_move(label, bd.turn)
        assert back == mv, f"promo mv={mv} -> {label} -> {back}"


def test_pgx_legal_actions_decode_to_python_legal_moves():
    """Inverse: every legal pgx action decodes to a legal python-chess move."""
    bd = chess.Board()
    state = pgx_bridge.chess_board_to_pgx_state(bd)
    py_legal = set(bd.legal_moves)
    for pgx_label in _pgx_legal_set(state):
        mv = pgx_bridge.pgx_action_to_chess_move(pgx_label, bd.turn)
        assert mv in py_legal, f"pgx label {pgx_label} decoded to {mv} not in python-chess legal"


def test_pgx_step_matches_python_chess_push():
    """Stepping a pgx state with the mapped action produces the same FEN as
    python-chess pushing the original move."""
    import jax
    import jax.numpy as jnp
    import pgx
    env = pgx.make("chess")
    bd = chess.Board()
    bd.push_san("e4")
    bd.push_san("e5")
    bd.push_san("Nf3")
    state = pgx_bridge.chess_board_to_pgx_state(bd)

    for mv in list(bd.legal_moves)[:5]:
        label = pgx_bridge.chess_move_to_pgx_action(mv, bd.turn)
        # Step pgx with that label
        s2 = env.step(state, jnp.int32(label), jax.random.key(0))
        pgx_after_fen = pgx_bridge._pgx_to_fen(s2)
        # Push python-chess
        bd2 = bd.copy()
        bd2.push(mv)
        py_after_fen = bd2.fen()
        assert pgx_after_fen == py_after_fen, (
            f"move {mv.uci()}: pgx={pgx_after_fen} != py={py_after_fen}"
        )
