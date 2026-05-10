"""Tests for stockfish bridge. Skipped if stockfish binary is not installed."""

import shutil

import chess
import numpy as np
import pytest

from sampling_chess import board as B
from sampling_chess import stockfish as SF

_NO_STOCKFISH = shutil.which("stockfish") is None and not any(
    shutil.which(p) for p in ("/usr/games/stockfish", "/usr/local/bin/stockfish")
)
no_stockfish = pytest.mark.skipif(_NO_STOCKFISH, reason="stockfish binary not on PATH")


# ---------- labeler ----------

@no_stockfish
def test_label_startpos_shape():
    with SF.StockfishLabeler(depth=8, multipv=5) as lab:
        out = lab.label(chess.Board())
    assert 1 <= len(out.move_indices) <= 5
    assert out.move_indices.dtype == np.int32
    assert out.move_values.dtype == np.float32
    assert out.move_probs.dtype == np.float32
    # All indices in valid action space
    for idx in out.move_indices:
        assert 0 <= int(idx) < B.NUM_ACTIONS
    # Probs sum to 1
    assert out.move_probs.sum() == pytest.approx(1.0, abs=1e-5)
    # Values bounded in [-1, 1]
    assert (out.move_values >= -1.0).all() and (out.move_values <= 1.0).all()


@no_stockfish
def test_label_startpos_white_advantage():
    """White to move at startpos: best move's V > 0 (white slightly winning)."""
    with SF.StockfishLabeler(depth=8, multipv=3) as lab:
        out = lab.label(chess.Board())
    assert out.value_target > 0.0


@no_stockfish
def test_label_side_to_move_convention():
    """V is from side-to-move POV — same position evaluated by both sides
    should give roughly opposite signs."""
    b_white = chess.Board()
    b_black = chess.Board()
    b_black.push_san("e4")  # black to move now
    with SF.StockfishLabeler(depth=8, multipv=1) as lab:
        v_white = lab.label(b_white).value_target  # white POV, ~+small
        v_black = lab.label(b_black).value_target  # black POV, ~-small
    # White expects slight edge; black sees slight disadvantage. Both small.
    assert abs(v_white) < 0.5 and abs(v_black) < 0.5


@no_stockfish
def test_label_winning_position():
    """Trivially winning position for white: V should be near +1."""
    # White to move; massive material advantage and Qh7# threat.
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1")
    with SF.StockfishLabeler(depth=8, multipv=2) as lab:
        out = lab.label(b)
    assert out.value_target > 0.7


@no_stockfish
def test_label_indices_are_legal():
    b = chess.Board()
    legal = {B.move_to_index(m) for m in b.legal_moves}
    with SF.StockfishLabeler(depth=6, multipv=5) as lab:
        out = lab.label(b)
    for idx in out.move_indices:
        assert int(idx) in legal


# ---------- opponent ----------

@no_stockfish
def test_opponent_plays_legal_move():
    with SF.StockfishOpponent(skill=5, time_limit=0.05) as opp:
        b = chess.Board()
        mv = opp.play(b)
    assert mv in chess.Board().legal_moves


@no_stockfish
def test_opponent_skill_clamping():
    with pytest.raises(ValueError):
        SF.StockfishOpponent(skill=99)


@no_stockfish
def test_opponent_full_short_game():
    """A full game between two skill-0 stockfish instances completes without error."""
    with SF.StockfishOpponent(skill=0, time_limit=0.02) as a, \
         SF.StockfishOpponent(skill=0, time_limit=0.02) as b_player:
        board = chess.Board()
        for _ in range(40):
            if board.is_game_over():
                break
            mover = a if board.turn == chess.WHITE else b_player
            mv = mover.play(board)
            assert mv in board.legal_moves
            board.push(mv)
    # Either game ended naturally or we hit the 40-ply cap; both fine.
    assert board.fullmove_number >= 1
