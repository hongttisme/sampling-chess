"""Tests for position sampling."""

import io

import chess
import pytest

from sampling_chess import data as D


# ---------- random self-play sampler ----------

def test_random_selfplay_yields_n():
    positions = list(D.iter_random_selfplay_positions(n=10, seed=0))
    assert len(positions) == 10


def test_random_selfplay_positions_are_legal():
    for board in D.iter_random_selfplay_positions(n=5, seed=1):
        assert isinstance(board, chess.Board)
        assert any(board.legal_moves)


def test_random_selfplay_seed_reproducibility():
    a = list(D.iter_random_selfplay_positions(n=5, seed=42))
    b = list(D.iter_random_selfplay_positions(n=5, seed=42))
    assert [x.fen() for x in a] == [x.fen() for x in b]


def test_random_selfplay_distinct_positions():
    """At least most positions in a 10-sample batch should be distinct."""
    positions = list(D.iter_random_selfplay_positions(n=10, seed=0))
    fens = {b.fen() for b in positions}
    assert len(fens) >= 8  # allow a couple of dupes from short collapsed games


# ---------- PGN sampler ----------

_TINY_PGN = """[Event "Test"]
[White "a"]
[Black "b"]
[WhiteElo "2200"]
[BlackElo "2150"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6
8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 1-0

[Event "Test"]
[White "c"]
[Black "d"]
[WhiteElo "1500"]
[BlackElo "1450"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 0-1

[Event "Test"]
[White "e"]
[Black "f"]
[WhiteElo "2400"]
[BlackElo "2350"]
[Result "1/2-1/2"]

1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 1/2-1/2
"""


def test_pgn_rating_filter(tmp_path):
    p = tmp_path / "tiny.pgn"
    p.write_text(_TINY_PGN)
    # min_rating=2000 should drop the second game (1500/1450)
    positions = list(D.iter_pgn_positions(str(p), min_rating=2000, seed=0))
    assert len(positions) == 2


def test_pgn_no_filter(tmp_path):
    p = tmp_path / "tiny.pgn"
    p.write_text(_TINY_PGN)
    positions = list(D.iter_pgn_positions(str(p), min_rating=0, seed=0))
    assert len(positions) == 3


def test_pgn_n_cap(tmp_path):
    p = tmp_path / "tiny.pgn"
    p.write_text(_TINY_PGN)
    positions = list(D.iter_pgn_positions(str(p), n=1, min_rating=0, seed=0))
    assert len(positions) == 1


def test_pgn_skip_short_games(tmp_path):
    p = tmp_path / "short.pgn"
    p.write_text(
        '[Event "x"]\n[WhiteElo "2200"]\n[BlackElo "2200"]\n[Result "1-0"]\n\n'
        "1. e4 e5 2. Nf3 1-0\n"
    )
    # 3 plies < skip_short_games default 10 -> dropped
    positions = list(D.iter_pgn_positions(str(p), min_rating=2000, seed=0))
    assert len(positions) == 0


def test_pgn_positions_have_legal_moves(tmp_path):
    p = tmp_path / "tiny.pgn"
    p.write_text(_TINY_PGN)
    for board in D.iter_pgn_positions(str(p), min_rating=0, seed=7):
        assert any(board.legal_moves)
