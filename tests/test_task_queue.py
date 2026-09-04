"""Task queue (Phase 2) testleri — offline, izole DB."""

from __future__ import annotations

import pytest

from app.agents.runtime import task_queue
from app.memory.sqlite_store import SqliteStore


@pytest.fixture(autouse=True)
def _test_tracker(tmp_path):
    """Sistem olaylarını gerçek reports/agent_runs'a sızdırmamak için izole tracker."""
    from app.agents.runtime.tracker import RunTracker, set_tracker

    set_tracker(RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs"))
    yield
    set_tracker(None)


@pytest.fixture
def st():
    return SqliteStore()


def test_create_list_get(st) -> None:
    t = task_queue.create_task("arxiv-fetcher", "momentum çek", store=st)
    assert t.task_id.startswith("atask_")
    assert t.status.value == "pending"
    got = task_queue.get_task(t.task_id, store=st)
    assert got is not None and got.task_id == t.task_id
    assert any(x.task_id == t.task_id for x in task_queue.list_tasks(store=st))


def test_claim_then_complete(st) -> None:
    t = task_queue.create_task("agentA", "iş", store=st)
    c = task_queue.claim_task(t.task_id, store=st)
    assert c is not None and c.status.value == "claimed" and c.claimed_at
    done = task_queue.complete_task(t.task_id, summary="bitti", store=st)
    assert done is not None and done.status.value == "completed" and done.completed_at


def test_fail(st) -> None:
    t = task_queue.create_task("agentA", "iş", store=st)
    f = task_queue.fail_task(t.task_id, "patladı", store=st)
    assert f is not None and f.status.value == "failed" and f.error == "patladı"


def test_cancel(st) -> None:
    t = task_queue.create_task("agentA", "iş", store=st)
    c = task_queue.cancel_task(t.task_id, reason="gerek yok", store=st)
    assert c is not None and c.status.value == "cancelled"


def test_claim_only_pending(st) -> None:
    t = task_queue.create_task("agentA", "iş", store=st)
    task_queue.claim_task(t.task_id, store=st)
    again = task_queue.claim_task(t.task_id, store=st)  # zaten claimed → değişmez
    assert again is not None and again.status.value == "claimed"


def test_terminal_guard(st) -> None:
    t = task_queue.create_task("agentA", "iş", store=st)
    task_queue.complete_task(t.task_id, store=st)
    # completed terminal → cancel no-op
    c = task_queue.cancel_task(t.task_id, store=st)
    assert c is not None and c.status.value == "completed"


def test_list_filter_by_agent_and_status(st) -> None:
    task_queue.create_task("agentX", "x", store=st)
    only_x = task_queue.list_tasks(agent_id="agentX", store=st)
    assert only_x and all(t.agent_id == "agentX" for t in only_x)
    pend = task_queue.list_tasks(status="pending", store=st)
    assert all(t.status.value == "pending" for t in pend)


def test_get_missing_returns_none(st) -> None:
    assert task_queue.get_task("atask_yok", store=st) is None


def test_try_claim_atomic_only_once(st) -> None:
    """Atomik CAS: aynı pending görevde yalnız İLK try_claim kazanır; ikinci None döner."""
    t = task_queue.create_task("agentA", "iş", store=st)
    first = task_queue.try_claim_task(t.task_id, store=st)
    second = task_queue.try_claim_task(t.task_id, store=st)
    assert first is not None and first.status.value == "claimed"
    assert second is None  # ikinci çağrı yarışı kaybetti (çift-çalıştırma önlendi)


def test_try_claim_missing_returns_none(st) -> None:
    assert task_queue.try_claim_task("atask_yok", store=st) is None
