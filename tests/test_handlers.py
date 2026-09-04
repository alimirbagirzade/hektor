"""Executor handler kaydı (handlers.py) testleri — offline, izole.

Güvenli salt-okuma handler kaydı + tehlikeli ajanların KAYDEDİLMEDİĞİ doğrulanır.
"""

from __future__ import annotations

import pytest

from app.agents.runtime import executor, task_queue
from app.agents.runtime.handlers import register_default_handlers
from app.memory.sqlite_store import SqliteStore


@pytest.fixture(autouse=True)
def _test_tracker(tmp_path):
    from app.agents.runtime.tracker import RunTracker, set_tracker

    set_tracker(RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs"))
    yield
    set_tracker(None)


@pytest.fixture(autouse=True)
def _clean_handlers():
    snap = dict(executor._HANDLERS)
    yield
    executor._HANDLERS.clear()
    executor._HANDLERS.update(snap)


@pytest.fixture
def st():
    return SqliteStore()


def test_registers_model_advisor() -> None:
    executor._HANDLERS.clear()
    newly = register_default_handlers()
    assert "model-advisor" in newly
    assert "model-advisor" in executor.registered_agents()


def test_idempotent() -> None:
    executor._HANDLERS.clear()
    register_default_handlers()
    assert register_default_handlers() == []  # ikinci kez: zaten kayıtlı → boş


def test_dangerous_agents_not_registered() -> None:
    executor._HANDLERS.clear()
    register_default_handlers()
    # Kural 8: tehlikeli/otonom ajanlar varsayılan handler DEĞİL
    regs = executor.registered_agents()
    assert "auto-lora-pipeline" not in regs
    assert "arxiv-fetcher" not in regs
    assert "rag-learning-loop" not in regs


def test_model_advisor_task_runs_end_to_end(st) -> None:
    """Kayıtlı model-advisor görevi executor üzerinden çalışır (salt-okuma → tamamlanır)."""
    register_default_handlers()
    t = task_queue.create_task("model-advisor", "model öner", store=st)
    out = executor.run_task(t, store=st)
    assert out["ok"] is True
    assert task_queue.get_task(t.task_id, store=st).status.value == "completed"
