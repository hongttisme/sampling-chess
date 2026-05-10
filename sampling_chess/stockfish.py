"""Stockfish bridge: subprocess labeler + skill-capped opponent.

Two roles, both wrapping python-chess.engine:

  1. StockfishLabeler — for Phase 1 SL bootstrap. Runs Stockfish at fixed
     depth with multipv=k, returns LabeledPosition with top-k moves, their
     centipawn evals, value V = tanh(cp/300) from side-to-move POV, and a
     soft policy = softmax(V / 0.1). Per doc section 4.2.

  2. StockfishOpponent — for evaluation. Configured with a Skill Level
     (0..20) and a fixed time-per-move; plays a single move on demand.
     Per doc section 3.4 we cap on skill, not depth.

For Phase 0 we use a single labeler / opponent at a time. A worker pool
for the 2M-position labeling job is added in Phase 1 if needed.
"""

import math
import shutil
from dataclasses import dataclass
from typing import Iterable, Optional

import chess
import chess.engine
import numpy as np

from sampling_chess import board as B

# Defaults from doc section 4.2.
DEFAULT_LABEL_DEPTH = 12
DEFAULT_MULTIPV = 5
SOFTMAX_TEMP = 0.1
CP_SCALE = 300.0
MATE_CP = 10_000

# Common locations for the stockfish binary.
_STOCKFISH_CANDIDATES = (
    "stockfish",
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
)


def find_stockfish() -> str:
    """Locate a stockfish binary, raising FileNotFoundError if not found."""
    for cand in _STOCKFISH_CANDIDATES:
        path = shutil.which(cand) or (cand if shutil.which("ls") and _exists(cand) else None)
        if path:
            return path
    raise FileNotFoundError(
        "stockfish binary not found. Install via `sudo apt install stockfish` "
        "(linux) or `brew install stockfish` (mac)."
    )


def _exists(path: str) -> bool:
    import os
    return os.path.isfile(path) and os.access(path, os.X_OK)


# ---------------------------------------------------------------------------
# Labeled-position container
# ---------------------------------------------------------------------------

@dataclass
class LabeledPosition:
    """One labeled training example for SL bootstrap."""
    fen: str
    move_indices: np.ndarray   # int32, shape (k,)
    move_values: np.ndarray    # float32, shape (k,) — V from side-to-move POV
    move_probs: np.ndarray     # float32, shape (k,) — softmax over values
    value_target: float        # scalar V, best line's value


def _cp_to_value(cp: int, white_to_move: bool) -> float:
    """Centipawns (white POV) -> V in [-1, 1] from side-to-move POV."""
    v = math.tanh(cp / CP_SCALE)
    return v if white_to_move else -v


# ---------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------

class StockfishLabeler:
    """Single Stockfish process for Phase 1 SL labeling.

    Use as a context manager, or call .close() explicitly.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        depth: int = DEFAULT_LABEL_DEPTH,
        multipv: int = DEFAULT_MULTIPV,
        threads: int = 1,
        hash_mb: int = 64,
    ):
        self.depth = depth
        self.multipv = multipv
        self._engine = chess.engine.SimpleEngine.popen_uci(path or find_stockfish())
        self._engine.configure({"Threads": threads, "Hash": hash_mb})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        try:
            self._engine.quit()
        except Exception:
            pass

    def label(self, board: chess.Board) -> LabeledPosition:
        info = self._engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=self.multipv,
        )
        white_to_move = board.turn == chess.WHITE
        moves: list[int] = []
        values: list[float] = []
        for line in info:
            mv = line["pv"][0]
            cp = line["score"].white().score(mate_score=MATE_CP)
            v = _cp_to_value(cp, white_to_move)
            moves.append(B.move_to_index(mv))
            values.append(v)

        moves_arr = np.asarray(moves, dtype=np.int32)
        values_arr = np.asarray(values, dtype=np.float32)
        # Soft policy over the top-k legal moves (doc section 4.2: temp 0.1).
        logits = values_arr / SOFTMAX_TEMP
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()

        return LabeledPosition(
            fen=board.fen(),
            move_indices=moves_arr,
            move_values=values_arr,
            move_probs=probs.astype(np.float32),
            value_target=float(values_arr[0]),
        )

    def label_many(self, boards: Iterable[chess.Board]) -> list[LabeledPosition]:
        return [self.label(b) for b in boards]


# ---------------------------------------------------------------------------
# Opponent
# ---------------------------------------------------------------------------

class StockfishOpponent:
    """Skill-capped Stockfish opponent used by the eval harness."""

    def __init__(
        self,
        path: Optional[str] = None,
        skill: int = 5,
        time_limit: float = 0.1,
        threads: int = 1,
        hash_mb: int = 16,
    ):
        if not 0 <= skill <= 20:
            raise ValueError(f"skill must be in [0, 20], got {skill}")
        self.skill = skill
        self.time_limit = time_limit
        self._engine = chess.engine.SimpleEngine.popen_uci(path or find_stockfish())
        self._engine.configure(
            {"Threads": threads, "Hash": hash_mb, "Skill Level": skill}
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        try:
            self._engine.quit()
        except Exception:
            pass

    def play(self, board: chess.Board) -> chess.Move:
        result = self._engine.play(board, chess.engine.Limit(time=self.time_limit))
        if result.move is None:
            raise RuntimeError(f"Stockfish returned no move at FEN: {board.fen()}")
        return result.move
