"""SL training: load labeled .npz, train ChessTransformer with policy CE + value MSE.

Pure-library module: data loading, loss/grad computation, optimizer factory,
train-step factory. The CLI lives in scripts/03_sl_train.py.

Loss (per doc 4.2):
  policy = cross-entropy between softmax(masked_logits) and the soft target
           distribution given by (move_indices, move_probs) — sparse over the
           top-k moves. We gather log-probs at the target indices and weight
           by target probabilities, summing over k and averaging over batch.
  value  = mean-squared error between predicted V (tanh) and target V from
           Stockfish, also from side-to-move POV.
  total  = policy_loss + lambda_v * value_loss   (lambda_v = 1.0 default)
"""

from pathlib import Path
from typing import Iterator

import chess
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from sampling_chess import board as B


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> dict:
    """Load .npz produced by scripts/02_label_batch.py + pre-encode boards
    using OUR (pieces, globals, masks) encoding.

    Returns a dict with numpy arrays:
      pieces       (N, 8, 8) int8     piece-plane encoding
      globals      (N, 9) float32     global features
      masks        (N, NUM_ACTIONS) bool   legal-move mask (4288)
      target_idx   (N, k) int32       move indices in our 4288 space (padded -1)
      target_prob  (N, k) float32     soft policy targets (padded 0)
      target_value (N,) float32       scalar value target

    Use load_dataset_pgx for pgx-encoded data (Plan A pipeline).
    """
    raw = np.load(path)
    fens = raw["fens"]
    n = len(fens)

    pieces = np.zeros((n, 8, 8), dtype=np.int8)
    globals_ = np.zeros((n, B.NUM_GLOBAL_FEATURES), dtype=np.float32)
    masks = np.zeros((n, B.NUM_ACTIONS), dtype=bool)
    for i, fen in enumerate(fens):
        bd = chess.Board(str(fen))
        pieces[i] = B.board_to_planes(bd)
        globals_[i] = B.board_to_global(bd)
        masks[i] = B.legal_action_mask(bd)

    return {
        "pieces": pieces,
        "globals": globals_,
        "masks": masks,
        "target_idx": raw["move_indices"].astype(np.int32),
        "target_prob": raw["move_probs"].astype(np.float32),
        "target_value": raw["value_targets"].astype(np.float32),
    }


def load_dataset_pgx(path: Path) -> dict:
    """Plan A pipeline: load a .npz produced by scripts/04_relabel_pgx.py.

    The .npz is expected to already contain pre-encoded pgx observations
    and legal masks (relabel script computes these once per labeling job;
    pgx's _from_fen is ~30 ms/pos in JAX-traced overhead, fine once but
    a 17-hour cold-start penalty at 2M positions if done on every load).

    Returns a dict with numpy arrays:
      observation  (N, 8, 8, 119) float32   pgx state.observation
      masks        (N, 4672) bool           pgx legal_action_mask
      target_idx   (N, k) int32             move indices in pgx 4672 space
      target_prob  (N, k) float32           soft policy targets (padded 0)
      target_value (N,) float32             scalar value target

    Falls back to per-position pgx encoding if the .npz lacks observation
    + masks (older data files); slow, emits a warning.
    """
    raw = np.load(path)

    if "observation" in raw.files and "masks" in raw.files:
        return {
            "observation": raw["observation"].astype(np.float32),
            "masks": raw["masks"].astype(bool),
            "target_idx": raw["move_indices"].astype(np.int32),
            "target_prob": raw["move_probs"].astype(np.float32),
            "target_value": raw["value_targets"].astype(np.float32),
        }

    print("[warn] dataset lacks pre-encoded observation/masks; "
          "falling back to per-position pgx_from_fen (slow). "
          "Re-run scripts/04_relabel_pgx.py to refresh.")
    from sampling_chess.pgx_bridge import chess_board_to_pgx_state
    from sampling_chess.net import PGX_NUM_ACTIONS, PGX_OBSERVATION_CHANNELS

    fens = raw["fens"]
    n = len(fens)
    obs = np.zeros((n, 8, 8, PGX_OBSERVATION_CHANNELS), dtype=np.float32)
    masks = np.zeros((n, PGX_NUM_ACTIONS), dtype=bool)
    for i, fen in enumerate(fens):
        bd = chess.Board(str(fen))
        state = chess_board_to_pgx_state(bd)
        obs[i] = np.asarray(state.observation, dtype=np.float32)
        masks[i] = np.asarray(state.legal_action_mask, dtype=bool)

    return {
        "observation": obs,
        "masks": masks,
        "target_idx": raw["move_indices"].astype(np.int32),
        "target_prob": raw["move_probs"].astype(np.float32),
        "target_value": raw["value_targets"].astype(np.float32),
    }


def iterate_batches(data: dict, batch_size: int,
                    rng: np.random.Generator) -> Iterator[dict]:
    """Infinite iterator of shuffled batches; reshuffles each epoch.

    Drops the last partial batch (so all batches have equal shape -- JIT-friendly).
    """
    n = len(data["pieces"])
    while True:
        perm = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            yield {k: v[idx] for k, v in data.items()}


# ---------------------------------------------------------------------------
# Loss + train step
# ---------------------------------------------------------------------------

def policy_loss_fn(logits: jnp.ndarray, mask: jnp.ndarray,
                   target_idx: jnp.ndarray, target_prob: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy against a sparse soft target distribution over top-k moves.

    Args:
      logits      (B, A)
      mask        (B, A) bool
      target_idx  (B, k) int32, padded with -1 where invalid
      target_prob (B, k) float32, padded with 0 where invalid

    Returns: scalar loss.
    """
    masked = jnp.where(mask, logits, -1e9)
    log_probs = jax.nn.log_softmax(masked, axis=-1)  # (B, A)
    valid = target_idx >= 0  # (B, k)
    safe_idx = jnp.where(valid, target_idx, 0)
    sel_log_probs = jnp.take_along_axis(log_probs, safe_idx, axis=-1)  # (B, k)
    per_example = -(target_prob * sel_log_probs * valid).sum(axis=-1)
    return per_example.mean()


def value_loss_fn(pred_v: jnp.ndarray, target_v: jnp.ndarray) -> jnp.ndarray:
    return ((pred_v - target_v) ** 2).mean()


def make_train_step(model, lambda_v: float = 1.0):
    """Build a JIT'd train_step for the legacy (pieces, globals) net."""

    def loss_fn(params, batch):
        logits, value = model.apply(
            {"params": params},
            batch["pieces"].astype(jnp.int32),
            batch["globals"],
        )
        p_loss = policy_loss_fn(logits, batch["masks"],
                                batch["target_idx"], batch["target_prob"])
        v_loss = value_loss_fn(value, batch["target_value"])
        loss = p_loss + lambda_v * v_loss
        return loss, (p_loss, v_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(state, batch):
        (loss, (p_loss, v_loss)), grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)
        gn = optax.tree.norm(grads)
        return state, {
            "loss": loss,
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "grad_norm": gn,
        }

    return train_step


def make_train_step_pgx(model, lambda_v: float = 1.0):
    """Plan A SL: train_step factory for ChessTransformerPgx with sparse top-k targets."""

    def loss_fn(params, batch):
        logits, value = model.apply({"params": params}, batch["observation"])
        p_loss = policy_loss_fn(logits, batch["masks"],
                                batch["target_idx"], batch["target_prob"])
        v_loss = value_loss_fn(value, batch["target_value"])
        loss = p_loss + lambda_v * v_loss
        return loss, (p_loss, v_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(state, batch):
        (loss, (p_loss, v_loss)), grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)
        gn = optax.tree.norm(grads)
        return state, {
            "loss": loss,
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "grad_norm": gn,
        }

    return train_step


# ---------------------------------------------------------------------------
# Phase 2: dense policy targets (self-play improvement targets)
# ---------------------------------------------------------------------------

def dense_policy_loss_fn(logits: jnp.ndarray, legal_mask: jnp.ndarray,
                         pi_target: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy against the full pi_improved distribution.

    Args:
      logits      (B, A) raw policy logits
      legal_mask  (B, A) bool, True for legal actions
      pi_target   (B, A) float32, soft policy target (sums to 1 over legal)

    Returns: scalar loss.

    pi_target is 0 on illegal actions (by construction in the improvement
    operators), so the product pi_target * log_probs is well-defined even
    with masked logits.
    """
    masked = jnp.where(legal_mask, logits, -1e9)
    log_probs = jax.nn.log_softmax(masked, axis=-1)
    per_example = -(pi_target * log_probs).sum(axis=-1)
    return per_example.mean()


def make_train_step_phase2(model, lambda_v: float = 1.0):
    """Phase 2 train_step: dense pi_improved policy CE + value MSE."""

    def loss_fn(params, batch):
        logits, value = model.apply({"params": params}, batch["observation"])
        p_loss = dense_policy_loss_fn(
            logits, batch["legal_mask"], batch["pi_improved"]
        )
        v_loss = value_loss_fn(value, batch["value_target"])
        loss = p_loss + lambda_v * v_loss
        return loss, (p_loss, v_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(state, batch):
        (loss, (p_loss, v_loss)), grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)
        gn = optax.tree.norm(grads)
        return state, {
            "loss": loss,
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "grad_norm": gn,
        }

    return train_step


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def make_optimizer(lr: float = 3e-4, weight_decay: float = 0.01,
                   warmup_steps: int = 1000, total_steps: int = 50_000,
                   end_value_frac: float = 0.1) -> optax.GradientTransformation:
    """AdamW with warmup + cosine decay, per doc 4.2.

    Clamps warmup_steps to total_steps // 2 so smoke runs with --steps 30
    don't hit a negative cosine-decay length.
    """
    warmup_steps = min(warmup_steps, max(1, total_steps // 2))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * end_value_frac,
    )
    return optax.adamw(learning_rate=schedule, weight_decay=weight_decay)


def init_train_state(model, params, optimizer) -> train_state.TrainState:
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
