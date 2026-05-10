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
import multiprocessing as mp
import os
import shutil
import time
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


# ---------------------------------------------------------------------------
# Multiprocess pool labeler
# ---------------------------------------------------------------------------
#
# Each worker process holds a single long-lived StockfishLabeler. Boards are
# passed to workers as FEN strings (chess.Board pickles fine but FEN is the
# canonical wire format). Per the doc Phase 1 budget, this is what runs the
# 2M-position labeling job on free CPU before any Colab work begins.
# ---------------------------------------------------------------------------

_worker_labeler: Optional["StockfishLabeler"] = None


def _init_worker(path: Optional[str], depth: int, multipv: int,
                 threads: int, hash_mb: int) -> None:
    global _worker_labeler
    _worker_labeler = StockfishLabeler(
        path=path, depth=depth, multipv=multipv,
        threads=threads, hash_mb=hash_mb,
    )
    # SIGTERM handler: graceful close of stockfish before the worker exits.
    # atexit doesn't fire reliably when Pool.terminate() reaps spawn workers,
    # and chess.engine's asyncio subprocess can deadlock if we leave it open.
    import signal

    def _on_sigterm(signum, frame):
        try:
            if _worker_labeler is not None:
                _worker_labeler.close()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)


def _label_one_fen(fen: str) -> "LabeledPosition":
    assert _worker_labeler is not None, "worker labeler not initialized"
    return _worker_labeler.label(chess.Board(fen))


class StockfishPool:
    """Parallel Stockfish labeling via multiprocessing.

    Spawns `n_workers` processes; each holds one Stockfish engine for the
    pool's lifetime. Use as a context manager to guarantee clean shutdown.

    Shutdown uses Pool.terminate() rather than close()+join(): chess.engine
    runs an internal asyncio loop that can deadlock during graceful shutdown
    when invoked from a spawn worker. terminate() sends SIGTERM, which our
    worker handler turns into a Stockfish.close() + os._exit().
    """

    def __init__(
        self,
        n_workers: int,
        path: Optional[str] = None,
        depth: int = DEFAULT_LABEL_DEPTH,
        multipv: int = DEFAULT_MULTIPV,
        threads_per_worker: int = 1,
        hash_mb: int = 64,
    ):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        self.depth = depth
        self.multipv = multipv
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(path, depth, multipv, threads_per_worker, hash_mb),
        )
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._pool.terminate()
            self._pool.join()
        except Exception:
            pass

    def label_batch(self, boards: Iterable[chess.Board],
                    chunksize: int = 4) -> list[LabeledPosition]:
        fens = [b.fen() for b in boards]
        return self._pool.map(_label_one_fen, fens, chunksize=chunksize)

    def label_batch_iter(self, boards: Iterable[chess.Board],
                         chunksize: int = 4):
        """Streaming variant: yields LabeledPositions as workers finish them.

        Useful for long jobs where you want to checkpoint progress.
        """
        fens = [b.fen() for b in boards]
        return self._pool.imap_unordered(_label_one_fen, fens, chunksize=chunksize)


def benchmark_pool(n_positions: int = 100, n_workers: int = 4,
                   depth: int = 12, multipv: int = 5,
                   seed: int = 0) -> dict:
    """Time labeling N self-play random positions; return throughput stats."""
    import random
    rng = random.Random(seed)

    # Generate positions via random self-play (each game contributes one mid-game position).
    boards: list[chess.Board] = []
    while len(boards) < n_positions:
        b = chess.Board()
        steps = rng.randint(4, 60)
        for _ in range(steps):
            if b.is_game_over():
                break
            mv = rng.choice(list(b.legal_moves))
            b.push(mv)
        if not b.is_game_over() and any(b.legal_moves):
            boards.append(b)

    t0 = time.time()
    with StockfishPool(n_workers=n_workers, depth=depth, multipv=multipv) as pool:
        results = pool.label_batch(boards)
    dt = time.time() - t0

    return {
        "n_positions": len(results),
        "n_workers": n_workers,
        "depth": depth,
        "multipv": multipv,
        "wall_clock_sec": dt,
        "ms_per_position": 1000 * dt / len(results),
        "positions_per_sec": len(results) / dt,
        # Project to 2M labeling job under the same conditions.
        "projected_2M_hours": (2_000_000 * dt / len(results)) / 3600,
    }
