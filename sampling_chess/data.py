"""Position sampling for SL bootstrap.

Two sources, selected per phase stage:

  * `iter_random_selfplay_positions(n, seed)` — generates positions via random
    move self-play, taking one mid-game cut per game. Distributionally weird
    but cheap (no download). Used for Stage 1 prototyping (~50k positions).

  * `iter_pgn_positions(path, n, min_rating)` — streams positions from a PGN
    file, sampling one position per game uniformly with rating filter. Used
    for Stage 2 production labeling (~2M from Lichess monthly db). PGN files
    can be plain `.pgn` or zstd-compressed `.pgn.zst`.
"""

import io
import random
from typing import Iterator, Optional

import chess


# ---------------------------------------------------------------------------
# Random self-play seed positions (prototyping)
# ---------------------------------------------------------------------------

def iter_random_selfplay_positions(
    n: int,
    seed: int = 0,
    min_plies: int = 4,
    max_plies: int = 80,
) -> Iterator[chess.Board]:
    """Yield n positions, each from a different random-vs-random game.

    Each game is played for a uniformly random number of plies in
    [min_plies, max_plies]; the position at that ply (or the last legal one
    if the game ended early) is yielded. Positions where the side-to-move
    has no legal moves are skipped.
    """
    rng = random.Random(seed)
    yielded = 0
    while yielded < n:
        board = chess.Board()
        target_plies = rng.randint(min_plies, max_plies)
        for _ in range(target_plies):
            if board.is_game_over():
                break
            mv = rng.choice(list(board.legal_moves))
            board.push(mv)
        if not any(board.legal_moves):
            continue
        yield board
        yielded += 1


# ---------------------------------------------------------------------------
# PGN streaming (production)
# ---------------------------------------------------------------------------

def _open_pgn(path: str) -> io.TextIOBase:
    """Open a .pgn or .pgn.zst file as a text stream."""
    if path.endswith(".zst"):
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Reading .pgn.zst requires the `zstandard` package; "
                "install via `pip install zstandard`."
            ) from e
        f = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _game_passes_rating(headers, min_rating: int) -> bool:
    if min_rating <= 0:
        return True
    try:
        we = int(headers.get("WhiteElo", "0"))
        be = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False
    return we >= min_rating and be >= min_rating


def iter_pgn_positions(
    path: str,
    n: Optional[int] = None,
    min_rating: int = 2000,
    seed: int = 0,
    skip_short_games: int = 10,
) -> Iterator[chess.Board]:
    """Stream `n` positions from a PGN file, one per game, rating-filtered.

    Sampling: for each accepted game, a uniformly random ply in [0, len-1]
    is selected. This avoids intra-game correlation (per doc 4.2). Games
    shorter than `skip_short_games` plies are dropped — too noisy for SL.
    """
    import chess.pgn
    rng = random.Random(seed)
    yielded = 0
    f = _open_pgn(path)
    try:
        while True:
            if n is not None and yielded >= n:
                return
            game = chess.pgn.read_game(f)
            if game is None:
                return  # EOF
            if not _game_passes_rating(game.headers, min_rating):
                continue
            moves = list(game.mainline_moves())
            if len(moves) < skip_short_games:
                continue
            cut = rng.randint(0, len(moves) - 1)
            board = game.board()
            for i, mv in enumerate(moves):
                if i == cut:
                    break
                board.push(mv)
            if not any(board.legal_moves):
                continue
            yield board
            yielded += 1
    finally:
        f.close()
