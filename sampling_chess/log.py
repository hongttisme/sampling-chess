"""Thin wandb wrapper used by training / self-play / eval scripts.

Design goals:
  - One module for all log calls; train / self-play / eval don't import wandb directly.
  - Degrades to no-op if wandb is missing OR no API key is configured. Lets
    unit tests and smoke scripts run without touching the wandb backend.
  - `init_run` is idempotent — re-calling it returns the active run.

Naming convention for runs (used across phases):
  - Phase 1 SL:        "phase1-sl-{seed}"
  - Phase 2 self-play: "phase2-{arm}-iter{n}"
  - Phase 3 ablation:  "phase3-{arm}-K{K}-k{k}-beta{beta}"
Use `group` to bind related runs (e.g., group="phase2-cross-arm").
"""

import os
from typing import Any, Optional

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


_active_run: Any = None  # wandb.sdk.wandb_run.Run when active, else None


def is_available() -> bool:
    """Wandb is importable."""
    return _WANDB_AVAILABLE


def is_enabled() -> bool:
    """Wandb is importable AND credentials look usable."""
    if not _WANDB_AVAILABLE:
        return False
    return bool(os.environ.get("WANDB_API_KEY")) or _wandb_netrc_present()


def _wandb_netrc_present() -> bool:
    """`wandb login` writes to ~/.netrc; treat that as 'logged in'."""
    netrc = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc):
        return False
    try:
        with open(netrc) as f:
            return "api.wandb.ai" in f.read()
    except OSError:
        return False


def init_run(
    project: str = "sampling-chess",
    name: Optional[str] = None,
    config: Optional[dict] = None,
    group: Optional[str] = None,
    tags: Optional[list[str]] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Start a wandb run; return True if active, False if no-op.

    `enabled=None` (default) auto-detects via is_enabled(). Pass enabled=False
    in tests / smoke runs to force no-op even when credentials are present.
    """
    global _active_run
    if enabled is None:
        enabled = is_enabled()
    if not enabled:
        return False
    if _active_run is not None:
        return True  # idempotent
    _active_run = wandb.init(
        project=project,
        name=name,
        config=config or {},
        group=group,
        tags=tags or [],
    )
    return True


def log(metrics: dict[str, Any], step: Optional[int] = None) -> None:
    """Forward metrics to the active run; no-op if none."""
    if _active_run is None:
        return
    _active_run.log(metrics, step=step)


def log_artifact(path: str, name: str, artifact_type: str = "model") -> None:
    """Upload a file (e.g., checkpoint) to the active run."""
    if _active_run is None:
        return
    art = wandb.Artifact(name=name, type=artifact_type)
    art.add_file(path)
    _active_run.log_artifact(art)


def finish() -> None:
    """Close the active run; safe to call multiple times."""
    global _active_run
    if _active_run is None:
        return
    _active_run.finish()
    _active_run = None
