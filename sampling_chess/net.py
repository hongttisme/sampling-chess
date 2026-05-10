"""Encoder-only chess transformer with policy + value heads (Flax linen).

Two variants share TransformerBlock + heads:

  * ChessTransformer (legacy): doc 4.1 spec. Inputs are our (pieces (8,8)
    int8 + globals (9,) float32); output policy is over our 4288-action
    space. Used by Phase 1 SL training before we adopted pgx.

  * ChessTransformerPgx (Plan A): pgx-native. Inputs are pgx observation
    (8, 8, 119) float32; output policy is over pgx's 4672 actions
    (AlphaZero standard). Used by Phase 2 / mctx integration.

Both: 8 layers, d_model=384, 6 heads, FFN 1536, RMSNorm + GELU. Param
counts diverge slightly (pgx variant has a larger input projection but
a slightly different action-head size).

The forward pass is JIT-friendly; mask out illegal logits at inference
via apply_legal_mask before sampling/argmax.
"""

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from sampling_chess.board import (
    NUM_ACTIONS,
    NUM_GLOBAL_FEATURES,
    NUM_PIECE_TYPES,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Pre-norm self-attention residual block.
        h = nn.RMSNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
        )(h, h)
        x = x + h
        # Pre-norm FFN residual block.
        h = nn.RMSNorm()(x)
        h = nn.Dense(self.ffn_dim)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        return x + h


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ChessTransformer(nn.Module):
    n_layers: int = 8
    d_model: int = 384
    n_heads: int = 6
    ffn_dim: int = 1536
    n_actions: int = NUM_ACTIONS
    n_piece_types: int = NUM_PIECE_TYPES
    n_global_features: int = NUM_GLOBAL_FEATURES

    @nn.compact
    def __call__(self, pieces: jnp.ndarray, globals_: jnp.ndarray
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
          pieces:   (B, 8, 8) int32 in [0, 12]
          globals_: (B, NUM_GLOBAL_FEATURES) float32

        Returns:
          policy_logits: (B, n_actions) float32  (unmasked; mask externally)
          value:         (B,) float32 in [-1, 1]
        """
        B = pieces.shape[0]

        # Piece embedding: (B, 8, 8, d_model)
        piece_embed = nn.Embed(self.n_piece_types, self.d_model)(pieces)

        # Learned 2D positional embedding (separable rank + file).
        rank_embed = self.param(
            "rank_embed", nn.initializers.normal(0.02), (8, self.d_model))
        file_embed = self.param(
            "file_embed", nn.initializers.normal(0.02), (8, self.d_model))
        pos = rank_embed[:, None, :] + file_embed[None, :, :]  # (8, 8, d_model)
        x = piece_embed + pos[None]  # (B, 8, 8, d_model)
        x = x.reshape(B, 64, self.d_model)

        # Global features projected to a single token, prepended to the sequence.
        global_token = nn.Dense(self.d_model)(globals_)[:, None, :]  # (B, 1, d_model)
        x = jnp.concatenate([global_token, x], axis=1)  # (B, 65, d_model)

        # Transformer encoder.
        for _ in range(self.n_layers):
            x = TransformerBlock(self.d_model, self.n_heads, self.ffn_dim)(x)
        x = nn.RMSNorm()(x)

        # Mean-pool over all tokens (board + global).
        pooled = x.mean(axis=1)  # (B, d_model)

        # Policy head: per-action logits. Doc 4.1 calls for a learned action
        # embedding; a single Dense from pooled features is the limit of that
        # construction with no per-token cross-attention to actions.
        policy_logits = nn.Dense(self.n_actions, name="policy_head")(pooled)

        # Value head: 2-layer MLP -> tanh.
        h = nn.Dense(self.d_model, name="value_hidden")(pooled)
        h = nn.gelu(h)
        value = nn.Dense(1, name="value_out")(h)[..., 0]
        value = jnp.tanh(value)

        return policy_logits, value


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def apply_legal_mask(logits: jnp.ndarray, mask: jnp.ndarray,
                     neg_inf: float = -1e9) -> jnp.ndarray:
    """Zero out illegal logits (replace with -inf) before softmax/argmax."""
    return jnp.where(mask, logits, neg_inf)


def count_params(params) -> int:
    """Sum of leaf array sizes in a Flax param tree."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


# ---------------------------------------------------------------------------
# Pgx-native variant (Plan A)
# ---------------------------------------------------------------------------
#
# pgx Chess emits a (8, 8, 119) observation per state and uses a 4672-action
# space. The 119 channels encode 8 plies of board history (own + opponent
# pieces, repetition counts) plus side-to-move, castling rights, half-move
# clock, etc., in current-player POV. The variant below consumes this
# tensor directly and routes it through the same TransformerBlock stack.

PGX_NUM_ACTIONS = 4672
PGX_OBSERVATION_CHANNELS = 119


class ChessTransformerPgx(nn.Module):
    n_layers: int = 8
    d_model: int = 384
    n_heads: int = 6
    ffn_dim: int = 1536
    n_actions: int = PGX_NUM_ACTIONS
    obs_channels: int = PGX_OBSERVATION_CHANNELS

    @nn.compact
    def __call__(self, observation: jnp.ndarray
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
          observation: (B, 8, 8, 119) float32 — pgx's per-state observation

        Returns:
          policy_logits: (B, n_actions) float32   (mask externally)
          value:         (B,) float32 in [-1, 1]
        """
        B = observation.shape[0]

        # Per-square projection: 119 -> d_model
        x = observation.reshape(B, 64, self.obs_channels)
        x = nn.Dense(self.d_model, name="input_proj")(x)  # (B, 64, d_model)

        # Learned 2D positional embedding (separable rank + file).
        rank_embed = self.param(
            "rank_embed", nn.initializers.normal(0.02), (8, self.d_model))
        file_embed = self.param(
            "file_embed", nn.initializers.normal(0.02), (8, self.d_model))
        pos = (rank_embed[:, None, :] + file_embed[None, :, :]).reshape(
            64, self.d_model)
        x = x + pos[None]

        # Transformer encoder.
        for _ in range(self.n_layers):
            x = TransformerBlock(self.d_model, self.n_heads, self.ffn_dim)(x)
        x = nn.RMSNorm()(x)

        # Mean-pool over board tokens.
        pooled = x.mean(axis=1)

        policy_logits = nn.Dense(self.n_actions, name="policy_head")(pooled)

        h = nn.Dense(self.d_model, name="value_hidden")(pooled)
        h = nn.gelu(h)
        value = nn.Dense(1, name="value_out")(h)[..., 0]
        value = jnp.tanh(value)

        return policy_logits, value
