"""Tests for the pgx-native eval adapter."""

import shutil

import chess
import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("pgx")
import pgx  # noqa: E402

from sampling_chess.eval import (  # noqa: E402
    make_pgx_greedy_policy,
    play_pgx_eval_game,
    play_pgx_match,
)
from sampling_chess.net import ChessTransformerPgx  # noqa: E402


_NO_STOCKFISH = shutil.which("stockfish") is None and not any(
    shutil.which(p) for p in ("/usr/games/stockfish", "/usr/local/bin/stockfish")
)
no_stockfish = pytest.mark.skipif(_NO_STOCKFISH, reason="stockfish not on PATH")

_ENV = pgx.make("chess")


def _random_pgx_net():
    model = ChessTransformerPgx(
        n_layers=2, d_model=64, n_heads=4, ffn_dim=128
    )
    dummy = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    params = model.init(jax.random.key(0), dummy)["params"]
    return model, params


def test_pgx_greedy_policy_returns_legal_action_idx():
    model, params = _random_pgx_net()
    policy = make_pgx_greedy_policy(model, params)
    state = jax.jit(_ENV.init)(jax.random.key(0))
    a = policy(state)
    assert isinstance(a, int)
    assert 0 <= a < 4672
    assert bool(state.legal_action_mask[a])


@no_stockfish
def test_play_pgx_eval_game_random_net_loses_to_skill0():
    """A random pgx net should almost always lose to Stockfish even at skill 0."""
    model, params = _random_pgx_net()
    policy = make_pgx_greedy_policy(model, params)

    from sampling_chess.stockfish import StockfishOpponent
    with StockfishOpponent(skill=0, time_limit=0.02) as opp:
        winner = play_pgx_eval_game(
            policy, opp, policy_plays_white=True,
            env=_ENV, max_plies=200,
        )
    # Either Stockfish won or game was inconclusive — random policy should
    # not be the winner.
    assert winner != chess.WHITE


@no_stockfish
def test_play_pgx_match_returns_match_result():
    """A 4-game match returns a MatchResult; random net loses ≥ 2 games."""
    model, params = _random_pgx_net()
    policy = make_pgx_greedy_policy(model, params)
    result = play_pgx_match(
        policy, opponent_skill=0,
        n_games=4, opponent_time=0.02, max_plies=200,
        env=_ENV, seed=0,
    )
    assert result.n_games == 4
    # Random vs SF skill 0: expect mostly losses.
    assert result.losses >= 2
