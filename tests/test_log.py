"""Tests for log.py wandb wrapper. All run with enabled=False (no network)."""

from sampling_chess import log


def test_module_imports():
    assert callable(log.init_run)
    assert callable(log.log)
    assert callable(log.finish)
    assert callable(log.log_artifact)
    assert callable(log.is_enabled)
    assert callable(log.is_available)


def test_disabled_returns_false_and_noops():
    ok = log.init_run(project="test", enabled=False)
    assert ok is False
    # All subsequent calls must no-op cleanly (no exception, no state change).
    log.log({"loss": 1.0, "acc": 0.5}, step=10)
    log.log_artifact("/nonexistent/path", name="ckpt")
    log.finish()


def test_finish_idempotent():
    log.finish()
    log.finish()


def test_init_idempotent_when_disabled():
    """Two init_run(enabled=False) calls both return False, no state."""
    assert log.init_run(enabled=False) is False
    assert log.init_run(enabled=False) is False


def test_is_available_truthful():
    """Just check it returns a bool — value depends on environment."""
    assert isinstance(log.is_available(), bool)
    assert isinstance(log.is_enabled(), bool)
