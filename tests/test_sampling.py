"""Tests for Arm B sampling (doc Algorithm 1).

Includes the doc section 7.4 invariant test:
  uniform pi_theta + V_theta == 0  =>  pi_sample uniform over legal first moves.
"""

import chess
import numpy as np
import pytest

from sampling_chess import board as B
from sampling_chess import sampling as S


# ---------------------------------------------------------------------------
# Mock apply_fns
# ---------------------------------------------------------------------------

def uniform_apply_fn(boards):
    """Returns zero logits (-> uniform after legal mask) and zero values."""
    n = len(boards)
    return (
        np.zeros((n, B.NUM_ACTIONS), dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    )


def biased_first_move_apply_fn(target_idx):
    """A fake net that strongly prefers a single action via huge logit."""
    def _fn(boards):
        n = len(boards)
        logits = np.zeros((n, B.NUM_ACTIONS), dtype=np.float32)
        logits[:, target_idx] = 1e3
        return logits, np.zeros(n, dtype=np.float32)
    return _fn


def constant_value_apply_fn(value: float):
    """Zero logits + constant value V at every leaf."""
    def _fn(boards):
        n = len(boards)
        return (
            np.zeros((n, B.NUM_ACTIONS), dtype=np.float32),
            np.full(n, value, dtype=np.float32),
        )
    return _fn


# ---------------------------------------------------------------------------
# Section 7.4 invariant: uniform pi + zero V -> uniform target
# ---------------------------------------------------------------------------

def test_uniform_pi_uniform_v_yields_uniform_target_stratified():
    """The doc §7.4 invariant under stratified first-move sampling.

    With pi_theta uniform and V_theta == 0, every weight w_i = 1/K and the
    improvement target's mass on each legal first move equals (count_a / K).
    Stratified sampling allocates exactly K/n trajectories per legal first
    move (when K is a multiple of n_legal), so the target is exactly uniform.
    """
    root = chess.Board()
    legal = list(root.legal_moves)
    n_legal = len(legal)
    K = 4 * n_legal  # 80

    rng = np.random.default_rng(0)
    out = S.sample_improved_policy(
        root=root, apply_fn=uniform_apply_fn,
        K=K, k_plies=4, beta=1.0, rng=rng, stratified=True,
    )

    legal_idx = [B.move_to_index(m) for m in legal]
    on_legal = out.pi_sample[legal_idx]
    # Uniform: each entry is 1/n_legal.
    expected = 1.0 / n_legal
    np.testing.assert_allclose(on_legal, expected, atol=1e-6)
    # No mass on illegal indices.
    illegal_mass = out.pi_sample.sum() - on_legal.sum()
    assert abs(illegal_mass) < 1e-6


def test_uniform_pi_uniform_v_yields_approx_uniform_prior_weighted():
    """Same invariant in expectation under prior-weighted sampling."""
    root = chess.Board()
    legal = list(root.legal_moves)
    n_legal = len(legal)
    K = 200 * n_legal  # large K so MC noise is small

    rng = np.random.default_rng(123)
    out = S.sample_improved_policy(
        root=root, apply_fn=uniform_apply_fn,
        K=K, k_plies=4, beta=1.0, rng=rng, stratified=False,
    )

    legal_idx = [B.move_to_index(m) for m in legal]
    on_legal = out.pi_sample[legal_idx]
    expected = 1.0 / n_legal
    # Tolerate ~30% relative error per move at K=200 per move.
    np.testing.assert_allclose(on_legal, expected, atol=0.3 * expected)


# ---------------------------------------------------------------------------
# Output shape and sum invariants
# ---------------------------------------------------------------------------

def test_pi_sample_sums_to_one_on_legal():
    rng = np.random.default_rng(0)
    out = S.sample_improved_policy(
        root=chess.Board(), apply_fn=uniform_apply_fn,
        K=20, k_plies=3, beta=0.0, rng=rng, stratified=True,
    )
    assert out.pi_sample.sum() == pytest.approx(1.0, abs=1e-5)


def test_pi_sample_assigns_zero_to_illegal_first_moves():
    rng = np.random.default_rng(0)
    out = S.sample_improved_policy(
        root=chess.Board(), apply_fn=uniform_apply_fn,
        K=20, k_plies=2, beta=1.0, rng=rng, stratified=True,
    )
    legal_indices = {B.move_to_index(m) for m in chess.Board().legal_moves}
    illegal_mass = sum(p for i, p in enumerate(out.pi_sample)
                       if i not in legal_indices)
    assert abs(illegal_mass) < 1e-6


def test_first_moves_are_legal():
    rng = np.random.default_rng(0)
    out = S.sample_improved_policy(
        root=chess.Board(), apply_fn=uniform_apply_fn,
        K=20, k_plies=2, beta=1.0, rng=rng, stratified=True,
    )
    legal = {B.move_to_index(m) for m in chess.Board().legal_moves}
    for fm in out.first_moves:
        assert int(fm) in legal


def test_v_plus_in_unit_range():
    rng = np.random.default_rng(0)
    out = S.sample_improved_policy(
        root=chess.Board(), apply_fn=constant_value_apply_fn(0.4),
        K=10, k_plies=2, beta=1.0, rng=rng, stratified=True,
    )
    # Even plies pushed (k=2) -> leaf STM == root STM -> no flip.
    # Wait: first move (1 ply) + 1 more = 2 plies pushed total. So even.
    # All trajectories survive 2 plies (random plays from startpos rarely end).
    # leaf_values should all be 0.4; v_plus = 0.4.
    assert out.v_plus == pytest.approx(0.4, abs=0.05)


# ---------------------------------------------------------------------------
# Side-to-move parity
# ---------------------------------------------------------------------------

def test_odd_ply_flips_leaf_value():
    """At k_plies=1, leaf is opponent-to-move. V_theta(leaf) = +0.5 in leaf
    POV means root POV is -0.5."""
    rng = np.random.default_rng(0)
    out = S.sample_improved_policy(
        root=chess.Board(), apply_fn=constant_value_apply_fn(0.5),
        K=10, k_plies=1, beta=0.0, rng=rng, stratified=True,
    )
    # beta=0 -> uniform weights -> v_plus = mean(leaf_values).
    # k_plies=1 means we pushed 1 ply, leaf POV != root POV, so V_root = -0.5.
    assert out.v_plus == pytest.approx(-0.5, abs=1e-5)


def test_even_ply_preserves_sign():
    rng = np.random.default_rng(0)
    out = S.sample_improved_policy(
        root=chess.Board(), apply_fn=constant_value_apply_fn(0.5),
        K=10, k_plies=2, beta=0.0, rng=rng, stratified=True,
    )
    # k_plies=2, all reach leaf at depth 2 (even) -> root POV = leaf POV = +0.5.
    assert out.v_plus == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# Sharpening (beta) behavior
# ---------------------------------------------------------------------------

def test_beta_zero_recovers_count_distribution():
    """beta=0 -> all weights equal -> pi_sample == empirical count over first moves."""
    rng = np.random.default_rng(0)
    K = 40
    out = S.sample_improved_policy(
        root=chess.Board(), apply_fn=constant_value_apply_fn(0.7),
        K=K, k_plies=2, beta=0.0, rng=rng, stratified=True,
    )
    # Stratified + uniform -> exactly K/n_legal trajectories per legal move.
    # At beta=0 the target is exactly uniform-on-legal.
    legal = list(chess.Board().legal_moves)
    legal_idx = [B.move_to_index(m) for m in legal]
    on_legal = out.pi_sample[legal_idx]
    expected = 1.0 / len(legal)
    np.testing.assert_allclose(on_legal, expected, atol=1e-6)


def test_high_beta_concentrates_vs_low_beta():
    """High beta yields a more concentrated improvement target than beta=0
    (entropy decreases). Test compares both modes with the same value
    function so any concentration must come from beta, not the value field
    or the rollout itself."""
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)

    # Per-call value cycler keyed off a private counter; identical sequence
    # for both runs because RNGs are seeded identically and the rollout path
    # is deterministic given the RNG.
    state_a = {"n": 0}
    state_b = {"n": 0}

    def make_fn(state):
        def _fn(boards):
            n = len(boards)
            vals = np.array([0.5 if (state["n"] + i) % 2 == 0 else -0.5
                             for i in range(n)], dtype=np.float32)
            state["n"] += n
            return (np.zeros((n, B.NUM_ACTIONS), dtype=np.float32), vals)
        return _fn

    K = 64
    out_low = S.sample_improved_policy(
        root=chess.Board(), apply_fn=make_fn(state_a),
        K=K, k_plies=2, beta=0.0, rng=rng_a, stratified=True,
    )
    out_high = S.sample_improved_policy(
        root=chess.Board(), apply_fn=make_fn(state_b),
        K=K, k_plies=2, beta=20.0, rng=rng_b, stratified=True,
    )
    # Entropy on legal moves: high beta should give lower entropy.
    legal = list(chess.Board().legal_moves)
    legal_idx = [B.move_to_index(m) for m in legal]

    def _entropy(pi):
        p = pi[legal_idx]
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    h_low = _entropy(out_low.pi_sample)
    h_high = _entropy(out_high.pi_sample)
    assert h_high < h_low, f"expected high-beta entropy {h_high:.3f} < low-beta {h_low:.3f}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_terminal_root_returns_immediately():
    """A terminal root should yield zero pi_sample and a terminal v_plus."""
    # Anastasia-style mate: black king on g8, white queen on g7 supported by
    # white king on g6. Black has no escape and can't capture queen.
    b_mate = chess.Board("6k1/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert b_mate.is_checkmate(), "FEN setup error: not actually mate"

    rng = np.random.default_rng(0)
    out = S.sample_improved_policy(
        root=b_mate, apply_fn=uniform_apply_fn,
        K=4, k_plies=2, beta=1.0, rng=rng, stratified=True,
    )
    # Black is mated -> from black's POV, value = -1.0
    assert out.v_plus == pytest.approx(-1.0, abs=1e-5)
    assert out.pi_sample.sum() == 0.0


def test_terminal_root_stalemate_returns_zero():
    """Stalemate at root: no legal moves, no checkmate -> v_plus = 0."""
    # White king on a1, black king on a3, black queen on b3 stalemates white.
    b_stale = chess.Board("8/8/8/8/8/kq6/8/K7 w - - 0 1")
    assert b_stale.is_stalemate()

    rng = np.random.default_rng(0)
    out = S.sample_improved_policy(
        root=b_stale, apply_fn=uniform_apply_fn,
        K=4, k_plies=2, beta=1.0, rng=rng, stratified=True,
    )
    assert out.v_plus == pytest.approx(0.0, abs=1e-5)
    assert out.pi_sample.sum() == 0.0


def test_invalid_K_or_k_plies_raises():
    with pytest.raises(ValueError):
        S.sample_improved_policy(
            root=chess.Board(), apply_fn=uniform_apply_fn,
            K=0, k_plies=2, beta=1.0,
        )
    with pytest.raises(ValueError):
        S.sample_improved_policy(
            root=chess.Board(), apply_fn=uniform_apply_fn,
            K=4, k_plies=0, beta=1.0,
        )
