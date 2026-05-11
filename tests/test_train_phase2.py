"""Tests for Phase 2 dense policy loss + train_step_phase2."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from sampling_chess.net import (  # noqa: E402
    ChessTransformerPgx,
    PGX_NUM_ACTIONS,
    PGX_OBSERVATION_CHANNELS,
)
from sampling_chess.train import (  # noqa: E402
    dense_policy_loss_fn,
    init_train_state,
    make_optimizer,
    make_train_step_phase2,
)


# ----- dense loss math -----

def test_dense_policy_loss_zero_at_perfect_match():
    """Logits onehot on the only target action -> CE near 0."""
    A = PGX_NUM_ACTIONS
    logits = jnp.full((1, A), -1e3).at[0, 100].set(1e3)
    mask = jnp.ones((1, A), dtype=bool)
    pi_target = jnp.zeros((1, A), dtype=jnp.float32).at[0, 100].set(1.0)
    loss = float(dense_policy_loss_fn(logits, mask, pi_target))
    assert loss < 1e-3


def test_dense_policy_loss_zero_pi_target_excluded():
    """Indices with pi_target=0 should not contribute (no NaN from masked log_probs)."""
    A = PGX_NUM_ACTIONS
    logits = jnp.zeros((1, A))
    mask = jnp.zeros((1, A), dtype=bool).at[0, 0].set(True).at[0, 1].set(True)
    pi_target = jnp.zeros((1, A), dtype=jnp.float32).at[0, 0].set(0.6).at[0, 1].set(0.4)
    loss = float(dense_policy_loss_fn(logits, mask, pi_target))
    assert np.isfinite(loss)
    assert loss > 0


def test_dense_policy_loss_uniform_target_uniform_logits():
    """Uniform target + uniform logits over legal moves -> loss = -log(1/n_legal)."""
    A = PGX_NUM_ACTIONS
    legal_count = 20
    mask_row = np.zeros(A, dtype=bool)
    mask_row[:legal_count] = True
    mask = jnp.asarray(mask_row[None])
    logits = jnp.zeros((1, A))
    pi_target = jnp.asarray(
        (mask_row.astype(np.float32) / legal_count)[None]
    )
    expected = float(-np.log(1.0 / legal_count))
    loss = float(dense_policy_loss_fn(logits, mask, pi_target))
    assert loss == pytest.approx(expected, abs=1e-4)


# ----- Train-step end-to-end overfit on a tiny batch -----

def _make_dense_batch(batch: int = 4, seed: int = 0):
    rng = np.random.RandomState(seed)
    obs = rng.randn(batch, 8, 8, PGX_OBSERVATION_CHANNELS).astype(np.float32)
    mask = np.zeros((batch, PGX_NUM_ACTIONS), dtype=bool)
    pi = np.zeros((batch, PGX_NUM_ACTIONS), dtype=np.float32)
    # First 20 indices legal, target onehot on index i (per example).
    for b in range(batch):
        mask[b, :20] = True
        pi[b, b * 5 % 20] = 1.0
    value = np.array([0.3, -0.5, 0.1, 0.7], dtype=np.float32)[:batch]
    return {
        "observation": jnp.asarray(obs),
        "legal_mask": jnp.asarray(mask),
        "pi_improved": jnp.asarray(pi),
        "value_target": jnp.asarray(value),
    }


def test_train_step_phase2_overfits_tiny_batch():
    """One batch, 30 steps with a tiny model: loss must drop substantially."""
    model = ChessTransformerPgx(
        n_layers=2, d_model=64, n_heads=4, ffn_dim=128
    )
    dummy = jnp.zeros((1, 8, 8, PGX_OBSERVATION_CHANNELS), dtype=jnp.float32)
    params = model.init(jax.random.key(0), dummy)["params"]
    opt = make_optimizer(lr=1e-3, warmup_steps=5, total_steps=30)
    state = init_train_state(model, params, opt)
    train_step = make_train_step_phase2(model, lambda_v=1.0)
    batch = _make_dense_batch(batch=4, seed=0)

    losses = []
    for _ in range(30):
        state, metrics = train_step(state, batch)
        losses.append(float(metrics["loss"]))

    assert losses[-1] < 0.7 * losses[0], (
        f"loss should drop >=30%: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )
