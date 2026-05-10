"""Tests for the eval harness."""

import math
import shutil

import chess
import pytest

from sampling_chess import eval as E


_NO_STOCKFISH = shutil.which("stockfish") is None and not any(
    shutil.which(p) for p in ("/usr/games/stockfish", "/usr/local/bin/stockfish")
)
no_stockfish = pytest.mark.skipif(_NO_STOCKFISH, reason="stockfish binary not on PATH")


# ---------- MatchResult math ----------

def test_score_basic():
    r = E.MatchResult(n_games=100, wins=50, draws=20, losses=30)
    assert r.score == pytest.approx(0.6)
    assert r.win_rate == pytest.approx(0.5)


def test_score_validation():
    with pytest.raises(ValueError):
        E.MatchResult(n_games=100, wins=50, draws=20, losses=20)  # 90 != 100


def test_wilson_ci_contains_score_and_has_width():
    r = E.MatchResult(n_games=100, wins=50, draws=0, losses=50)
    lo, hi = r.wilson_ci()
    assert lo < 0.5 < hi
    assert hi - lo > 0.05  # finite width


def test_wilson_ci_tightens_with_n():
    r_small = E.MatchResult(n_games=20, wins=10, draws=0, losses=10)
    r_large = E.MatchResult(n_games=2000, wins=1000, draws=0, losses=1000)
    w_small = r_small.wilson_ci()
    w_large = r_large.wilson_ci()
    assert (w_small[1] - w_small[0]) > (w_large[1] - w_large[0])


def test_elo_zero_at_50pct():
    r = E.MatchResult(n_games=100, wins=50, draws=0, losses=50)
    assert abs(r.elo()) < 1.0


def test_elo_positive_when_winning():
    # 70% score -> standard Elo formula gives ~+147
    r = E.MatchResult(n_games=1000, wins=700, draws=0, losses=300)
    elo = r.elo()
    assert 130 < elo < 160


def test_elo_symmetric():
    r_high = E.MatchResult(n_games=100, wins=70, draws=0, losses=30)
    r_low = E.MatchResult(n_games=100, wins=30, draws=0, losses=70)
    assert r_high.elo() == pytest.approx(-r_low.elo(), abs=0.1)


def test_summary_string_renders():
    r = E.MatchResult(n_games=100, wins=50, draws=10, losses=40)
    s = r.summary()
    assert "50W/10D/40L" in s
    assert "score=0.550" in s
    assert "elo=" in s


# ---------- play_one_game with random policies ----------

def test_random_vs_random_terminates():
    p1 = E.make_random_policy(seed=1)
    p2 = E.make_random_policy(seed=2)
    winner, board = E.play_one_game(p1, p2, max_plies=300)
    # Random vs random nearly always ends in draw or stalemate within 300 plies;
    # we just assert no crash and the function returned a sensible tuple.
    assert isinstance(board, chess.Board)
    assert winner in (chess.WHITE, chess.BLACK, None)


def test_play_one_game_illegal_move_raises():
    def bad_policy(board: chess.Board) -> chess.Move:
        return chess.Move(0, 56)  # a1 -> a8, illegal at startpos
    with pytest.raises(RuntimeError, match="illegal"):
        E.play_one_game(bad_policy, E.make_random_policy(0), max_plies=10)


def test_play_one_game_uses_starting_board():
    """A custom starting position is used; outcome reflects it."""
    # White to mate in 1: Qh7#
    start = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1")
    # Pick the policy that always plays Qh7 if available.
    def mate_policy(b: chess.Board) -> chess.Move:
        for mv in b.legal_moves:
            b.push(mv)
            if b.is_checkmate():
                b.pop()
                return mv
            b.pop()
        return next(iter(b.legal_moves))

    winner, _ = E.play_one_game(mate_policy, E.make_random_policy(0), start, max_plies=20)
    assert winner == chess.WHITE


# ---------- end-to-end with Stockfish ----------

@no_stockfish
def test_random_loses_to_skill_zero():
    """A uniform-random policy should lose almost everything to Stockfish skill 0."""
    p = E.make_random_policy(seed=42)
    result = E.play_match(
        p, opponent_skill=0, n_games=4,
        opponent_time=0.02, max_plies=200, seed=0,
    )
    assert result.n_games == 4
    # Random vs even skill-0 stockfish: stockfish wins almost every game.
    assert result.losses >= 2  # lower-bounded for a 4-game run
