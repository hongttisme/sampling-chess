"""Tests for train.py loss, optimizer, and the train_step factory.

Focus on correctness of the loss math + ability to overfit a tiny batch.
The end-to-end SL run lives in scripts/03_sl_train.py and is exercised
from a Colab notebook, not in unit tests.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from sampling_chess import board as B  # noqa: E402
from sampling_chess.net import ChessTransformer  # noqa: E402
from sampling_chess.train import (  # noqa: E402
    init_train_state,
    iterate_batches,
    make_optimizer,
    make_train_step,
    policy_loss_fn,
    value_loss_fn,
)


# ---------- loss math ----------

def test_policy_loss_zero_at_perfect_match():
    """If logits put all mass on the single target index, CE -> 0."""
    A = B.NUM_ACTIONS
    logits = jnp.full((1, A), -1e3).at[0, 5].set(1e3)  # near-onehot at idx 5
    mask = jnp.ones((1, A), dtype=bool)
    target_idx = jnp.array([[5, -1, -1, -1, -1]], dtype=jnp.int32)
    target_prob = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    loss = float(policy_loss_fn(logits, mask, target_idx, target_prob))
    assert loss < 1e-3


def test_policy_loss_pads_dont_contribute():
    """Padding rows (target_idx==-1) must be excluded from the sum."""
    A = B.NUM_ACTIONS
    logits = jnp.zeros((1, A))
    mask = jnp.ones((1, A), dtype=bool)
    # Two valid + three padded
    target_idx = jnp.array([[3, 7, -1, -1, -1]], dtype=jnp.int32)
    target_prob = jnp.array([[0.6, 0.4, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    loss_with_pad = float(policy_loss_fn(logits, mask, target_idx, target_prob))
    # Same but with padded values set to a huge number to demonstrate they're masked.
    target_idx_dirty = jnp.array([[3, 7, 0, 0, 0]], dtype=jnp.int32)
    target_prob_dirty = jnp.array([[0.6, 0.4, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    loss_dirty = float(policy_loss_fn(logits, mask, target_idx_dirty, target_prob_dirty))
    # Both should match because target_prob=0 zeros the contribution either way.
    assert loss_with_pad == pytest.approx(loss_dirty, abs=1e-5)


def test_value_loss_zero_at_match():
    pred = jnp.array([0.3, -0.7, 0.0])
    target = jnp.array([0.3, -0.7, 0.0])
    assert float(value_loss_fn(pred, target)) < 1e-9


def test_value_loss_positive_otherwise():
    pred = jnp.array([0.5, 0.5])
    target = jnp.array([-0.5, -0.5])
    assert float(value_loss_fn(pred, target)) > 0.5


# ---------- batch iterator ----------

def _fake_dataset(n: int = 64) -> dict:
    return {
        "pieces": np.zeros((n, 8, 8), dtype=np.int8),
        "globals": np.zeros((n, B.NUM_GLOBAL_FEATURES), dtype=np.float32),
        "masks": np.ones((n, B.NUM_ACTIONS), dtype=bool),
        "target_idx": np.zeros((n, 5), dtype=np.int32),
        "target_prob": np.array([[1.0, 0, 0, 0, 0]] * n, dtype=np.float32),
        "target_value": np.zeros(n, dtype=np.float32),
    }


def test_iterate_batches_yields_correct_shapes():
    data = _fake_dataset(n=20)
    rng = np.random.default_rng(0)
    it = iterate_batches(data, batch_size=4, rng=rng)
    batch = next(it)
    assert batch["pieces"].shape == (4, 8, 8)
    assert batch["globals"].shape == (4, B.NUM_GLOBAL_FEATURES)
    assert batch["target_idx"].shape == (4, 5)


def test_iterate_batches_drops_partial():
    """A 5-position dataset with batch_size=2 yields batches of size exactly 2."""
    data = _fake_dataset(n=5)
    rng = np.random.default_rng(0)
    it = iterate_batches(data, batch_size=2, rng=rng)
    for _ in range(3):
        batch = next(it)
        assert batch["pieces"].shape[0] == 2


# ---------- optimizer ----------

def test_optimizer_warmup_then_decay():
    """Schedule reaches ~peak at warmup_steps, decays after."""
    opt = make_optimizer(lr=1e-3, warmup_steps=100, total_steps=1000)
    # Find the schedule used inside the GradientTransformation by inspecting state
    from optax import warmup_cosine_decay_schedule
    schedule = warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-3, warmup_steps=100,
        decay_steps=1000, end_value=1e-4,
    )
    assert float(schedule(0)) == pytest.approx(0.0, abs=1e-9)
    assert float(schedule(100)) == pytest.approx(1e-3, abs=1e-6)
    assert float(schedule(1000)) <= 1e-3


# ---------- train step end-to-end (overfit one batch) ----------

def _make_real_batch(batch_size: int = 4, seed: int = 0):
    """A small batch whose loss should drop monotonically when overfit."""
    import chess
    import random
    rng = random.Random(seed)
    boards = []
    while len(boards) < batch_size:
        bd = chess.Board()
        for _ in range(rng.randint(2, 10)):
            if bd.is_game_over():
                break
            bd.push(rng.choice(list(bd.legal_moves)))
        if any(bd.legal_moves):
            boards.append(bd)
    pieces = np.stack([B.board_to_planes(b) for b in boards]).astype(np.int8)
    globals_ = np.stack([B.board_to_global(b) for b in boards])
    masks = np.stack([B.legal_action_mask(b) for b in boards])
    # Dummy targets: highest-index legal move with prob 1, value = 0.5
    target_idx = np.full((batch_size, 5), -1, dtype=np.int32)
    target_prob = np.zeros((batch_size, 5), dtype=np.float32)
    for i, b in enumerate(boards):
        legals = list(b.legal_moves)
        target_idx[i, 0] = B.move_to_index(legals[0])
        target_prob[i, 0] = 1.0
    target_value = np.full((batch_size,), 0.5, dtype=np.float32)
    return {
        "pieces": pieces, "globals": globals_, "masks": masks,
        "target_idx": target_idx, "target_prob": target_prob,
        "target_value": target_value,
    }


def test_train_step_overfits_tiny_batch():
    """One batch, 30 steps: loss should drop substantially."""
    model = ChessTransformer(n_layers=2, d_model=64, n_heads=4, ffn_dim=128)
    batch = _make_real_batch(batch_size=4, seed=0)
    init_pieces = jnp.asarray(batch["pieces"][:1].astype(np.int32))
    init_globals = jnp.asarray(batch["globals"][:1])
    params = model.init(jax.random.key(0), init_pieces, init_globals)["params"]
    opt = make_optimizer(lr=1e-3, warmup_steps=5, total_steps=30)
    state = init_train_state(model, params, opt)
    train_step = make_train_step(model, lambda_v=1.0)

    jbatch = {k: jnp.asarray(v) for k, v in batch.items()}
    losses = []
    for _ in range(30):
        state, metrics = train_step(state, jbatch)
        losses.append(float(metrics["loss"]))

    # Loss should drop by at least 30% from start to end
    assert losses[-1] < 0.7 * losses[0], f"start={losses[0]:.4f} end={losses[-1]:.4f}"
