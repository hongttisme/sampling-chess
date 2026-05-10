"""Tests for net.py: shape, mask, JIT, and parameter count.

These run on CPU JAX; full GPU benchmarks live in scripts/.
"""

import chess
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from sampling_chess import board as B  # noqa: E402
from sampling_chess.net import (  # noqa: E402
    ChessTransformer,
    apply_legal_mask,
    count_params,
)


def _make_inputs(batch: int = 2, seed: int = 0):
    """Encode `batch` random self-play positions into JAX arrays."""
    import random
    rng = random.Random(seed)
    boards = []
    while len(boards) < batch:
        bd = chess.Board()
        for _ in range(rng.randint(4, 30)):
            if bd.is_game_over():
                break
            bd.push(rng.choice(list(bd.legal_moves)))
        if any(bd.legal_moves):
            boards.append(bd)
    pieces = jnp.asarray(np.stack([B.board_to_planes(b) for b in boards]),
                         dtype=jnp.int32)
    globals_ = jnp.asarray(np.stack([B.board_to_global(b) for b in boards]),
                           dtype=jnp.float32)
    return pieces, globals_, boards


# ---------- forward pass ----------

def test_forward_shapes():
    model = ChessTransformer()
    pieces, globals_, _ = _make_inputs(batch=4)
    params = model.init(jax.random.key(0), pieces, globals_)["params"]
    logits, value = model.apply({"params": params}, pieces, globals_)
    assert logits.shape == (4, B.NUM_ACTIONS)
    assert value.shape == (4,)


def test_value_in_unit_range():
    model = ChessTransformer()
    pieces, globals_, _ = _make_inputs(batch=4)
    params = model.init(jax.random.key(0), pieces, globals_)["params"]
    _, value = model.apply({"params": params}, pieces, globals_)
    assert jnp.all(value >= -1.0)
    assert jnp.all(value <= 1.0)


def test_batch_size_agnostic():
    model = ChessTransformer()
    p1, g1, _ = _make_inputs(batch=1)
    params = model.init(jax.random.key(0), p1, g1)["params"]
    for B_ in (1, 3, 8):
        p, g, _ = _make_inputs(batch=B_)
        logits, value = model.apply({"params": params}, p, g)
        assert logits.shape == (B_, B.NUM_ACTIONS)
        assert value.shape == (B_,)


# ---------- parameter count ----------

def test_param_count_band():
    """Sanity-bound the param count.

    Doc 4.1 claims 7-8M params for this config but the arithmetic for
    d_model=384 / 8 layers / FFN 1536 / NUM_ACTIONS=4288 is ~16M (the doc
    likely under-counted the FFN contribution or the action-space head).
    We accept the actual count from these hyperparams and just sanity-bound
    it to catch accidental 10x blowups.
    """
    model = ChessTransformer()
    pieces, globals_, _ = _make_inputs(batch=1)
    params = model.init(jax.random.key(0), pieces, globals_)["params"]
    n_params = count_params(params)
    assert 10_000_000 <= n_params <= 25_000_000, f"got {n_params:,} params"


# ---------- legal mask ----------

def test_apply_legal_mask_zeros_illegal_softmax():
    """After masking + softmax, illegal moves carry near-zero probability."""
    pieces, globals_, boards = _make_inputs(batch=1)
    model = ChessTransformer()
    params = model.init(jax.random.key(0), pieces, globals_)["params"]
    logits, _ = model.apply({"params": params}, pieces, globals_)
    mask = jnp.asarray(B.legal_action_mask(boards[0]), dtype=bool)
    masked = apply_legal_mask(logits[0], mask)
    probs = jax.nn.softmax(masked)
    # Total prob mass on illegal moves ~ 0 (within float32 noise).
    illegal = jnp.where(mask, 0.0, probs).sum()
    assert float(illegal) < 1e-6
    # Probs sum to 1
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


# ---------- JIT compilation ----------

def test_forward_jit_compilable():
    """A jit'd forward pass runs without retracing on identical shapes."""
    model = ChessTransformer()
    pieces, globals_, _ = _make_inputs(batch=2)
    params = model.init(jax.random.key(0), pieces, globals_)["params"]

    @jax.jit
    def fwd(p, pi, gl):
        return model.apply({"params": p}, pi, gl)

    out1 = fwd(params, pieces, globals_)
    out2 = fwd(params, pieces, globals_)
    # Same shapes both calls; second uses the cached trace.
    assert out1[0].shape == out2[0].shape
    assert out1[1].shape == out2[1].shape


# ---------- mask helper ----------

def test_apply_legal_mask_preserves_legal_logits():
    logits = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask = jnp.array([True, False, True, False])
    out = apply_legal_mask(logits, mask)
    assert float(out[0]) == 1.0
    assert float(out[2]) == 3.0
    assert float(out[1]) < -1e8
    assert float(out[3]) < -1e8
