"""Interactive play vs trained chess transformer.

Loads a checkpoint produced by scripts/03_sl_train.py and lets a human play
against the bot in the terminal via UCI moves (e.g., "e2e4", "e7e8q").

Usage:
    python scripts/11_play_vs_human.py --ckpt checkpoints/ckpt_0100000.pkl
    python scripts/11_play_vs_human.py --ckpt ... --human-color black
    python scripts/11_play_vs_human.py --ckpt ... --mode sample --temperature 0.5
    python scripts/11_play_vs_human.py --ckpt ... --start-fen "rnbqkbnr/..."

Modes:
    greedy : argmax of masked logits (deterministic; strongest single-move)
    sample : softmax(logits/temperature) sample (more variety; weaker if T high)

Type "quit" / "q" / "exit" / "resign" at the move prompt to end the game.
"""

import argparse
import pickle
import sys

import chess
import jax
import jax.numpy as jnp
import numpy as np

from sampling_chess import board as B
from sampling_chess.net import ChessTransformer, count_params


def _load_ckpt(ckpt_path: str):
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    if "params" not in ckpt:
        raise KeyError(f"checkpoint at {ckpt_path} missing 'params' key")
    return ckpt


def _build_model_from_config(config: dict | None) -> ChessTransformer:
    """Construct ChessTransformer with the same arch the ckpt was trained on.

    Falls back to doc-spec defaults if the config dict is absent / partial.
    """
    if not config:
        return ChessTransformer()
    return ChessTransformer(
        n_layers=int(config.get("n_layers", 8)),
        d_model=int(config.get("d_model", 384)),
        n_heads=int(config.get("n_heads", 6)),
        ffn_dim=int(config.get("ffn_dim", 1536)),
    )


def _make_action_fn(model: ChessTransformer):
    """Build a jit'd (params, pieces, globals_, mask) -> (masked_logits, value)."""

    @jax.jit
    def fwd(params, pieces, globals_, mask):
        logits, value = model.apply(
            {"params": params},
            pieces[None].astype(jnp.int32),
            globals_[None],
        )
        masked = jnp.where(mask, logits[0], jnp.float32(-1e9))
        return masked, value[0]

    return fwd


def _format_board(board: chess.Board) -> str:
    # python-chess unicode rendering with black-on-white pieces.
    return board.unicode(borders=True, empty_square=".")


def _get_human_move(board: chess.Board) -> chess.Move | None:
    """Read a UCI move from stdin; loop until valid or user quits."""
    while True:
        try:
            raw = input("\nYour move (UCI, e.g. e2e4): ").strip().lower()
        except EOFError:
            return None
        if raw in ("quit", "q", "exit", "resign"):
            return None
        try:
            move = chess.Move.from_uci(raw)
        except (ValueError, chess.InvalidMoveError):
            print("  bad UCI format. Examples: e2e4, e7e8q (promo to queen)")
            continue
        if move not in board.legal_moves:
            sample = sorted(m.uci() for m in board.legal_moves)
            shown = ", ".join(sample[:12]) + (" ..." if len(sample) > 12 else "")
            print(f"  illegal. Some legal moves: {shown}")
            continue
        return move


def _bot_action(action_fn, params, board: chess.Board,
                mode: str, temperature: float,
                rng: np.random.Generator) -> tuple[chess.Move, float]:
    pieces = jnp.asarray(B.board_to_planes(board))
    globals_ = jnp.asarray(B.board_to_global(board))
    mask = jnp.asarray(B.legal_action_mask(board))

    logits, value = action_fn(params, pieces, globals_, mask)
    logits_np = np.asarray(logits)
    value_v = float(value)

    if mode == "greedy":
        idx = int(np.argmax(logits_np))
    else:  # sample
        scaled = logits_np / max(temperature, 1e-3)
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        # Belt-and-braces: zero out illegal even though logits already -1e9.
        legal_mask = np.asarray(mask)
        probs = np.where(legal_mask, probs, 0.0)
        s = probs.sum()
        if s <= 0:
            probs = legal_mask.astype(np.float32)
            probs /= probs.sum()
        else:
            probs = probs / s
        idx = int(rng.choice(B.NUM_ACTIONS, p=probs))

    move = B.index_to_move(idx)
    return move, value_v


def play_one_game(model, params, *, mode: str, temperature: float,
                  human_color: chess.Color, start_fen: str | None,
                  max_plies: int) -> str:
    action_fn = _make_action_fn(model)
    rng = np.random.default_rng()
    board = chess.Board(start_fen) if start_fen else chess.Board()

    print("\n=== Game start ===")
    print(f"You are {'WHITE' if human_color == chess.WHITE else 'BLACK'}\n")

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        print(_format_board(board))
        print(f"Move {board.fullmove_number}, "
              f"{'WHITE' if board.turn else 'BLACK'} to move")

        if board.turn == human_color:
            mv = _get_human_move(board)
            if mv is None:
                print("\n[human resigned/quit]")
                return "human-quit"
        else:
            mv, value = _bot_action(action_fn, params, board, mode,
                                    temperature, rng)
            print(f"  Bot plays {mv.uci()}  (V={value:+.3f}, "
                  f"meaning {'+ for bot' if board.turn == (not human_color) else 'oops'})")

        board.push(mv)

    print("\n" + _format_board(board))
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        print(f"\n=== game ended at ply cap ({max_plies}) ===")
        return "ply-cap"

    winner_str = {
        chess.WHITE: "WHITE wins",
        chess.BLACK: "BLACK wins",
        None: "draw",
    }[outcome.winner]
    if outcome.winner is None:
        result = "draw"
    elif outcome.winner == human_color:
        result = "human wins"
    else:
        result = "bot wins"
    print(f"\n=== {winner_str}  ({outcome.termination.name}) — {result} ===")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ckpt", required=True, help="path to .pkl ckpt")
    parser.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="for sample mode (lower = more deterministic)")
    parser.add_argument("--human-color", choices=["white", "black"], default="white")
    parser.add_argument("--start-fen", default=None,
                        help="optional starting FEN; default = startpos")
    parser.add_argument("--max-plies", type=int, default=400)
    args = parser.parse_args()

    print(f"[load] {args.ckpt}")
    ckpt = _load_ckpt(args.ckpt)
    model = _build_model_from_config(ckpt.get("config"))
    params = ckpt["params"]
    print(f"[model] {count_params(params):,} params")
    print(f"[mode]  {args.mode}"
          + (f" T={args.temperature}" if args.mode == "sample" else ""))

    color = chess.WHITE if args.human_color == "white" else chess.BLACK
    play_one_game(
        model, params,
        mode=args.mode, temperature=args.temperature,
        human_color=color, start_fen=args.start_fen, max_plies=args.max_plies,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
