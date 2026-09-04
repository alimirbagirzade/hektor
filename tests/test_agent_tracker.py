"""Agent run tracker (Phase 1) — koşu + olay kaydı testleri (offline).

Tracker, izole test DB'sine (conftest `_isolate_storage`) ve geçici bir JSONL
dizinine yazar. Dekoratör testleri ``set_tracker`` ile enjekte edilir.
"""

from __future__ import annotations

import pytest

from app.agents.runtime import AgentRunStatus, RunTracker, set_tracker, tracked
from app.memory.sqlite_store import SqliteStore


@pytest.fixture
def tracker(tmp_path):
    return RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs")


def test_start_run_creates_running_record(tracker) -> None:
    run_id = tracker.start_run("test-agent", trigger_type="manual")
    assert run_id.startswith("arun_")
    run = tracker.store.get_agent_run(run_id)
    assert run is not None
    assert run["agent_id"] == "test-agent"
    assert run["status"] == "running"
    assert run["trigger_type"] == "manual"


def test_start_writes_start_event(tracker) -> None:
    run_id = tracker.start_run("test-agent")
    kinds = [e["kind"] for e in tracker.store.list_agent_events(run_id)]
    assert "start" in kinds


def test_finish_completed(tracker) -> None:
    run_id = tracker.start_run("test-agent")
    tracker.finish_run(run_id, status=AgentRunStatus.completed, summary={"x": 1})
    run = tracker.store.get_agent_run(run_id)
    assert run["status"] == "completed"
    assert run["finished_at"]
    assert run["summary"] == {"x": 1}
    kinds = [e["kind"] for e in tracker.store.list_agent_events(run_id)]
    assert "finish" in kinds


def test_finish_failed_records_error(tracker) -> None:
    run_id = tracker.start_run("test-agent")
    tracker.finish_run(run_id, status=AgentRunStatus.failed, error="boom")
    run = tracker.store.get_agent_run(run_id)
    assert run["status"] == "failed"
    assert run["error"] == "boom"


def test_jsonl_log_exists(tracker, tmp_path) -> None:
    run_id = tracker.start_run("test-agent")
    tracker.finish_run(run_id)
    path = tmp_path / "agent_runs" / f"{run_id}.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2  # run başlığı + start (+ finish + run_final)


def test_list_agent_runs_and_filter(tracker) -> None:
    r1 = tracker.start_run("agent-a")
    tracker.finish_run(r1)
    r2 = tracker.start_run("agent-b")
    tracker.finish_run(r2)
    assert len(tracker.store.list_agent_runs(limit=50)) >= 2
    only_a = tracker.store.list_agent_runs(agent_id="agent-a")
    assert only_a and all(r["agent_id"] == "agent-a" for r in only_a)


def test_list_agent_events(tracker) -> None:
    run_id = tracker.start_run("agent-evt")
    tracker.log_event(run_id, "step", "ara adım")
    tracker.finish_run(run_id)
    events = tracker.store.list_agent_events(run_id)
    messages = [e["message"] for e in events]
    assert any(m == "ara adım" for m in messages)


def test_decorator_records_completed_and_failed(tracker) -> None:
    set_tracker(tracker)
    try:

        @tracked("dec-agent")
        def ok_fn(x):
            return {"ok": True, "value": x}

        @tracked("dec-agent")
        def boom_fn():
            raise ValueError("patladı")

        assert ok_fn(5) == {"ok": True, "value": 5}
        with pytest.raises(ValueError):
            boom_fn()

        statuses = {r["status"] for r in tracker.store.list_agent_runs(agent_id="dec-agent")}
        assert "completed" in statuses
        assert "failed" in statuses
    finally:
        set_tracker(None)


def test_decorator_ok_false_is_failed(tracker) -> None:
    set_tracker(tracker)
    try:

        @tracked("okfalse-agent")
        def soft_fail():
            return {"ok": False, "reason": "yetersiz"}

        soft_fail()
        runs = tracker.store.list_agent_runs(agent_id="okfalse-agent")
        assert runs and runs[0]["status"] == "failed"
        assert "yetersiz" in (runs[0]["error"] or "")
    finally:
        set_tracker(None)


def test_async_decorator_runs(tmp_path) -> None:
    """Async dekoratör yolu da koşu kaydeder (asyncio.run ile, offline — marker gerekmez)."""
    import asyncio

    tr = RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs")
    set_tracker(tr)
    try:

        @tracked("async-agent")
        async def afn():
            return {"ok": True}

        asyncio.run(afn())
        runs = tr.store.list_agent_runs(agent_id="async-agent")
        assert runs and runs[0]["status"] == "completed"
    finally:
        set_tracker(None)


def test_async_decorator_cancelled_not_left_running(tmp_path) -> None:
    """İptal edilen async koşu 'running' kalmaz; 'cancelled' kaydedilir, iptal fırlatılır."""
    import asyncio

    tr = RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs")
    set_tracker(tr)
    try:

        @tracked("cancel-agent")
        async def cancel_fn():
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(cancel_fn())

        runs = tr.store.list_agent_runs(agent_id="cancel-agent")
        assert runs and runs[0]["status"] == "cancelled"
        assert all(r["status"] != "running" for r in runs)
    finally:
        set_tracker(None)
