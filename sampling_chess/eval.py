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
    """Uniform-random move selection. Useful as a Phase 0 baseline."""
    rng = random.Random(seed)

    def _policy(board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        return rng.choice(moves)

    return _policy


# ---------------------------------------------------------------------------
# Pgx-native eval (Plan A): policy operates on pgx state, opponent on chess.Board
# ---------------------------------------------------------------------------
#
# The existing play_match path requires PolicyFn: chess.Board -> chess.Move,
# which would force a chess_board_to_pgx_state (~1.6s) per move on the
# learned-policy side. Instead we keep the pgx state as the canonical game
# state and only emit a FEN for Stockfish (fast _to_fen). Stockfish's chess
# move is then mapped back to a pgx action label (chess_move_to_pgx_action,
# which is also fast — no _from_fen).

PgxPolicyFn = Callable[[object], int]  # pgx_state -> pgx action_idx


def make_pgx_greedy_policy(model, params) -> PgxPolicyFn:
    """Greedy: argmax of masked logits over pgx 4672 actions.

    The inner forward + argmax is JIT-compiled and reused across calls (params
    is closed over as a traced constant). Without JIT each call eager-dispatches
    every transformer layer; on a 16M-param model that is ~100 ms/call and
    eval dominates iteration wall-clock.
    """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def _policy_jit(observation, mask):
        logits, _ = model.apply({"params": params}, observation[None])
        masked = jnp.where(mask, logits[0], -1e9)
        return jnp.argmax(masked).astype(jnp.int32)

    def policy(state) -> int:
        return int(_policy_jit(state.observation, state.legal_action_mask))

    return policy


def play_pgx_eval_game(
    policy_fn: PgxPolicyFn,
    opponent,                # StockfishOpponent
    policy_plays_white: bool,
    *,
    env=None,
    starting_state=None,
    max_plies: int = 400,
    rng=None,
) -> Optional[chess.Color]:
    """Play one game; return winner color (chess.WHITE/BLACK) or None for draw/cap."""
    import jax
    import jax.numpy as jnp
    import pgx
    from pgx.experimental.chess import to_fen
    from sampling_chess.pgx_bridge import chess_move_to_pgx_action

    if env is None:
        env = pgx.make("chess")
    if rng is None:
        rng = random.Random()

    # JIT env.step + env.init for cache hit on every move (eager dispatch is
    # ~10-50 ms/call on the chess transition graph; jit'd is ~ms).
    step_jit = jax.jit(env.step)
    init_jit = jax.jit(env.init)

    if starting_state is None:
        key = jax.random.key(rng.randrange(2**31))
        state = init_jit(key)
    else:
        state = starting_state

    for _ in range(max_plies):
        if bool(state.terminated):
            break
        # pgx player 0 plays white in pgx's default mapping.
        is_policy_turn = (int(state.current_player) == 0) == policy_plays_white
        if is_policy_turn:
            action_idx = int(policy_fn(state))
        else:
            fen = to_fen(state)
            board = chess.Board(fen)
            move = opponent.play(board)
            action_idx = chess_move_to_pgx_action(move, board.turn)
        step_key = jax.random.key(rng.randrange(2**31))
        state = step_jit(state, jnp.int32(action_idx), step_key)

    if not bool(state.terminated):
        return None
    rewards = state.rewards
    r_white = float(rewards[0])
    if r_white > 0:
        return chess.WHITE
    if r_white < 0:
        return chess.BLACK
    return None


def play_pgx_match(
    policy_fn: PgxPolicyFn,
    opponent_skill: int,
    *,
    n_games: int = 100,
    opponent_time: float = 0.05,
    max_plies: int = 400,
    env=None,
    seed: Optional[int] = None,
) -> MatchResult:
    """Pgx-native match: policy plays half as white, half as black."""
    rng = random.Random(seed)
    from sampling_chess.stockfish import StockfishOpponent

    n_white = n_games // 2
    n_black = n_games - n_white
    wins = draws = losses = 0

    with StockfishOpponent(skill=opponent_skill, time_limit=opponent_time) as opp:
        for _ in range(n_white):
            w = play_pgx_eval_game(policy_fn, opp, True,
                                   env=env, max_plies=max_plies, rng=rng)
            if w == chess.WHITE:
                wins += 1
            elif w == chess.BLACK:
                losses += 1
            else:
                draws += 1
        for _ in range(n_black):
            w = play_pgx_eval_game(policy_fn, opp, False,
                                   env=env, max_plies=max_plies, rng=rng)
            if w == chess.BLACK:
                wins += 1
            elif w == chess.WHITE:
                losses += 1
            else:
                draws += 1

    return MatchResult(n_games=n_games, wins=wins, draws=draws, losses=losses)
