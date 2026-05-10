"""Tests for the pgx-native net variant."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("pgx")

from sampling_chess.net import (  # noqa: E402
    ChessTransformerPgx,
    PGX_NUM_ACTIONS,
    PGX_OBSERVATION_CHANNELS,
    apply_legal_mask,
    count_params,
)
from sampling_chess.pgx_bridge import (  # noqa: E402
    chess_board_to_pgx_state,
)
import chess  # noqa: E402


def _real_pgx_obs_batch(n: int = 2):
    """A small batch of real pgx observations from random self-play seeds."""
    import random
    rng = random.Random(0)
    obs_list = []
    masks = []
    while len(obs_list) < n:
        bd = chess.Board()
        for _ in range(rng.randint(2, 20)):
            if bd.is_game_over():
                break
            bd.push(rng.choice(list(bd.legal_moves)))
        if any(bd.legal_moves):
            state = chess_board_to_pgx_state(bd)
            obs_list.append(np.array(state.observation, dtype=np.float32))
            masks.append(np.array(state.legal_action_mask, dtype=bool))
    return jnp.asarray(np.stack(obs_list)), jnp.asarray(np.stack(masks))


def test_pgx_obs_shape_correct():
    obs, mask = _real_pgx_obs_batch(n=3)
    assert obs.shape == (3, 8, 8, PGX_OBSERVATION_CHANNELS)
    assert mask.shape == (3, PGX_NUM_ACTIONS)


def test_forward_shapes_pgx():
    model = ChessTransformerPgx()
    obs, _ = _real_pgx_obs_batch(n=4)
    params = model.init(jax.random.key(0), obs)["params"]
    logits, value = model.apply({"params": params}, obs)
    assert logits.shape == (4, PGX_NUM_ACTIONS)
    assert value.shape == (4,)


def test_value_in_unit_range_pgx():
    model = ChessTransformerPgx()
    obs, _ = _real_pgx_obs_batch(n=4)
    params = model.init(jax.random.key(0), obs)["params"]
    _, value = model.apply({"params": params}, obs)
    assert jnp.all(value >= -1.0)
    assert jnp.all(value <= 1.0)


def test_param_count_pgx():
    """PGX variant has slightly different param count due to (119 -> d_model)
    input projection and 4672 vs 4288 action head."""
    model = ChessTransformerPgx()
    obs, _ = _real_pgx_obs_batch(n=1)
    params = model.init(jax.random.key(0), obs)["params"]
    n = count_params(params)
    assert 12_000_000 <= n <= 25_000_000, f"got {n:,}"


def test_legal_mask_softmax_pgx():
    obs, mask = _real_pgx_obs_batch(n=1)
    model = ChessTransformerPgx()
    params = model.init(jax.random.key(0), obs)["params"]
    logits, _ = model.apply({"params": params}, obs)
    masked = apply_legal_mask(logits[0], mask[0])
    probs = jax.nn.softmax(masked)
    illegal_mass = jnp.where(mask[0], 0.0, probs).sum()
    assert float(illegal_mass) < 1e-6
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_jit_compilable_pgx():
    model = ChessTransformerPgx()
    obs, _ = _real_pgx_obs_batch(n=2)
    params = model.init(jax.random.key(0), obs)["params"]

    @jax.jit
    def fwd(p, ob):
        return model.apply({"params": p}, ob)

    o1 = fwd(params, obs)
    o2 = fwd(params, obs)
    assert o1[0].shape == o2[0].shape
    assert o1[1].shape == o2[1].shape


def test_batch_size_agnostic_pgx():
    model = ChessTransformerPgx()
    obs1, _ = _real_pgx_obs_batch(n=1)
    params = model.init(jax.random.key(0), obs1)["params"]
    for n in (1, 3, 5):
        ob, _ = _real_pgx_obs_batch(n=n)
        l, v = model.apply({"params": params}, ob)
        assert l.shape == (n, PGX_NUM_ACTIONS)
        assert v.shape == (n,)
