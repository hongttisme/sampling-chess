"""Evaluation harness: play matches between a learned policy and Stockfish.

A `Policy` is any callable `chess.Board -> chess.Move`. The learned net's
greedy/sampled policy is wrapped with such a callable in net.py / search.py.

`play_match` plays `n_games` total: half as white, half as black. With an
optional opening book, both colors play from the same set of positions to
reduce variance, per doc section 3.4.

`MatchResult` reports score, win rate, 95% Wilson CI, and an Elo estimate
via the standard logistic transform.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import chess

from sampling_chess.stockfish import StockfishOpponent

PolicyFn = Callable[[chess.Board], chess.Move]

_Z_95 = 1.96  # normal quantile for 95% CI
_ELO_K = 400.0


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Aggregate result of a series of games. All counts from policy POV."""
    n_games: int
    wins: int
    draws: int
    losses: int

    def __post_init__(self):
        total = self.wins + self.draws + self.losses
        if total != self.n_games:
            raise ValueError(
                f"wins+draws+losses={total} != n_games={self.n_games}"
            )

    @property
    def score(self) -> float:
        """Score in [0, 1]: 1 per win, 0.5 per draw."""
        return (self.wins + 0.5 * self.draws) / max(self.n_games, 1)

    @property
    def win_rate(self) -> float:
        """Strict win rate (excludes draws)."""
        return self.wins / max(self.n_games, 1)

    def wilson_ci(self, z: float = _Z_95) -> tuple[float, float]:
        """Wilson interval on the score, treated as a Bernoulli proportion."""
        n = self.n_games
        if n == 0:
            return (0.0, 1.0)
        p = self.score
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    def elo(self) -> float:
        """Elo difference (positive = policy stronger than opponent)."""
        return _score_to_elo(self.score)

    def elo_ci(self) -> tuple[float, float]:
        lo, hi = self.wilson_ci()
        return (_score_to_elo(lo), _score_to_elo(hi))

    def summary(self) -> str:
        lo, hi = self.wilson_ci()
        elo = self.elo()
        elo_lo, elo_hi = self.elo_ci()
        return (
            f"{self.wins}W/{self.draws}D/{self.losses}L "
            f"score={self.score:.3f} (95% CI [{lo:.3f}, {hi:.3f}]) "
            f"elo={elo:+.1f} (95% CI [{elo_lo:+.1f}, {elo_hi:+.1f}])"
        )


def _score_to_elo(p: float) -> float:
    """Score in [0,1] -> Elo, clamped at +/- 800."""
    p = min(max(p, 1e-3), 1 - 1e-3)
    return -_ELO_K * math.log10(1.0 / p - 1.0)


# ---------------------------------------------------------------------------
# Game playing
# ---------------------------------------------------------------------------

def play_one_game(
    white_policy: PolicyFn,
    black_policy: PolicyFn,
    starting_board: Optional[chess.Board] = None,
    max_plies: int = 400,
) -> tuple[Optional[chess.Color], chess.Board]:
    """Play one game; return (winner, final_board).

    `winner` is chess.WHITE / chess.BLACK / None (draw or ply-cap reached).
    """
    board = starting_board.copy() if starting_board else chess.Board()
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        policy = white_policy if board.turn == chess.WHITE else black_policy
        mv = policy(board)
        if mv not in board.legal_moves:
            raise RuntimeError(
                f"Policy returned illegal move {mv} at FEN {board.fen()}"
            )
        board.push(mv)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None, board  # ply cap with no terminal state
    return outcome.winner, board  # may be None for stalemate / 50-move / etc.


def play_match(
    policy: PolicyFn,
    opponent_skill: int,
    n_games: int = 100,
    opening_book: Optional[Iterable[chess.Board]] = None,
    max_plies: int = 400,
    opponent_time: float = 0.05,
    seed: Optional[int] = None,
) -> MatchResult:
    """Play `n_games` against Stockfish at `opponent_skill`.

    Half played as white, half as black. With `opening_book`, both colors
    play from positions sampled from it.
    """
    rng = random.Random(seed)
    book = list(opening_book) if opening_book is not None else [chess.Board()]
    n_white = n_games // 2
    n_black = n_games - n_white
    wins = draws = losses = 0

    with StockfishOpponent(skill=opponent_skill, time_limit=opponent_time) as opp:
        opp_play = opp.play

        for _ in range(n_white):
            start = rng.choice(book)
            winner, _ = play_one_game(policy, opp_play, start, max_plies=max_plies)
            if winner == chess.WHITE:
                wins += 1
            elif winner == chess.BLACK:
                losses += 1
            else:
                draws += 1

        for _ in range(n_black):
            start = rng.choice(book)
            winner, _ = play_one_game(opp_play, policy, start, max_plies=max_plies)
            if winner == chess.BLACK:
                wins += 1
            elif winner == chess.WHITE:
                losses += 1
            else:
                draws += 1

    return MatchResult(n_games=n_games, wins=wins, draws=draws, losses=losses)


# ---------------------------------------------------------------------------
# Reference policies
# ---------------------------------------------------------------------------

def make_random_policy(seed: Optional[int] = None) -> PolicyFn:
    """Uniform-random move selection. Useful as a sanity baseline."""
    rng = random.Random(seed)

    def _policy(board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        return rng.choice(moves)

    return _policy
