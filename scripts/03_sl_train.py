"""Phase 1 SL training driver: load .npz, train ChessTransformer.

Stage 1 prototype (50k random self-play seeds, 5k steps, batch 256):
    .venv/bin/python scripts/03_sl_train.py \\
        --data data/labels_50k_random.npz --steps 5000 --batch-size 256

Stage 2/3 (real Lichess data, ~50-100k steps on Colab A100):
    python scripts/03_sl_train.py --data data/labels_2M.npz \\
        --steps 100000 --batch-size 1024 --wandb
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sampling_chess import log
from sampling_chess.net import ChessTransformer, count_params
from sampling_chess.train import (
    init_train_state,
    iterate_batches,
    load_dataset,
    make_optimizer,
    make_train_step,
)


def _save_ckpt(ckpt_dir: Path, step: int, params, config: dict) -> Path:
    p = ckpt_dir / f"ckpt_{step:07d}.pkl"
    with open(p, "wb") as f:
        pickle.dump({"step": step, "params": params, "config": config}, f)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True, help="labeled .npz")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-v", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--name", default=None, help="wandb run name")
    parser.add_argument("--group", default="phase1-sl", help="wandb group")
    args = parser.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # wandb init
    wandb_active = False
    if args.wandb:
        wandb_active = log.init_run(
            project="sampling-chess",
            name=args.name or f"phase1-sl-seed{args.seed}",
            group=args.group,
            tags=["phase1", "sl"],
            config=vars(args),
        )
        if not wandb_active:
            print("[warn] --wandb requested but log.init_run returned False; "
                  "logging disabled (no API key?)")

    # Data
    print(f"[data] loading + encoding {args.data}")
    t0 = time.time()
    data = load_dataset(args.data)
    n_train = len(data["pieces"])
    print(f"[data] {n_train} positions encoded in {time.time()-t0:.1f}s | "
          f"batch={args.batch_size}")

    # Model + state
    model = ChessTransformer()
    init_pieces = jnp.asarray(data["pieces"][:1].astype(np.int32))
    init_globals = jnp.asarray(data["globals"][:1])
    params = model.init(jax.random.key(args.seed), init_pieces, init_globals)["params"]
    print(f"[model] {count_params(params):,} params")

    optimizer = make_optimizer(
        lr=args.lr, warmup_steps=args.warmup, total_steps=args.steps,
    )
    state = init_train_state(model, params, optimizer)
    train_step = make_train_step(model, lambda_v=args.lambda_v)

    # Training loop
    step = 0
    t_start = time.time()
    last_log_t = t_start
    metrics_acc: dict = {}

    for batch in iterate_batches(data, args.batch_size, rng):
        if step >= args.steps:
            break
        # Move to JAX arrays
        jbatch = {k: jnp.asarray(v) for k, v in batch.items()}
        state, metrics = train_step(state, jbatch)
        step += 1

        for k, v in metrics.items():
            metrics_acc[k] = metrics_acc.get(k, 0.0) + float(v)

        if step % args.log_every == 0:
            denom = args.log_every
            avg = {k: v / denom for k, v in metrics_acc.items()}
            now = time.time()
            ips = (args.batch_size * args.log_every) / (now - last_log_t)
            elapsed = now - t_start
            print(
                f"[train] step {step:>6} | loss {avg['loss']:.4f} | "
                f"p {avg['policy_loss']:.4f} | v {avg['value_loss']:.4f} | "
                f"gnorm {avg['grad_norm']:.3f} | {ips:>5.0f} samp/s | "
                f"{elapsed:.0f}s"
            )
            if wandb_active:
                log.log({**avg, "samples_per_sec": ips}, step=step)
            metrics_acc = {}
            last_log_t = now

        if step % args.ckpt_every == 0:
            ckpt_path = _save_ckpt(args.ckpt_dir, step, state.params, vars(args))
            print(f"[ckpt] {ckpt_path}")
            if wandb_active:
                log.log_artifact(str(ckpt_path), name=f"ckpt_{step}", artifact_type="model")

    final = _save_ckpt(args.ckpt_dir, step, state.params, vars(args))
    print(f"\n[done] {args.steps} steps in {time.time()-t_start:.0f}s | final {final}")
    if wandb_active:
        log.log_artifact(str(final), name=f"ckpt_final_{step}", artifact_type="model")
        log.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
