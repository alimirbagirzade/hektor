"""Supervisor (Phase 2) testleri — STOP_ALL + onay kapısı (offline, izole)."""

from __future__ import annotations

import pytest

from app.agents.runtime import approvals, supervisor
from app.memory.sqlite_store import SqliteStore


@pytest.fixture(autouse=True)
def _test_tracker(tmp_path):
    from app.agents.runtime.tracker import RunTracker, set_tracker

    set_tracker(RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs"))
    yield
    set_tracker(None)


@pytest.fixture
def st():
    return SqliteStore()


def test_stop_all_create_clear(tmp_path) -> None:
    assert supervisor.is_stop_all_active(root=tmp_path) is False
    supervisor.create_stop_all("bakım", root=tmp_path)
    assert supervisor.is_stop_all_active(root=tmp_path) is True
    supervisor.clear_stop_all(root=tmp_path)
    assert supervisor.is_stop_all_active(root=tmp_path) is False


def test_stop_all_blocks_dangerous(st, tmp_path) -> None:
    supervisor.create_stop_all("dur", root=tmp_path)
    dec = supervisor.can_run_agent(
        "auto-lora-pipeline", action="auto_lora_start_training", store=st, root=tmp_path
    )
    assert dec.allowed is False
    assert dec.blocked_by == "stop_all"


def test_readonly_agent_allowed_under_stop_all(st, tmp_path) -> None:
    supervisor.create_stop_all("dur", root=tmp_path)
    # model-advisor tehlikeli değil → STOP_ALL altında bile çalıştırılabilir (okunur)
    dec = supervisor.can_run_agent("model-advisor", store=st, root=tmp_path)
    assert dec.allowed is True


def test_autonomous_pipeline_allowed_without_fresh_approval(st, tmp_path) -> None:
    dec = supervisor.can_run_agent(
        "auto-lora-pipeline", action="auto_lora_start_training", store=st, root=tmp_path
    )
    assert dec.allowed is True
    assert dec.requires_approval is False


def test_fresh_approval_does_not_change_autonomous_manifest_policy(st, tmp_path) -> None:
    d = approvals.require_fresh_approval(
        "auto-lora-pipeline", "auto_lora_start_training", "critical", "s", store=st
    )
    approvals.approve(d.approval_id, store=st)
    dec = supervisor.can_run_agent(
        "auto-lora-pipeline", action="auto_lora_start_training", store=st, root=tmp_path
    )
    assert dec.allowed is True
    assert dec.requires_approval is False


def test_unknown_agent_blocked(st, tmp_path) -> None:
    dec = supervisor.can_run_agent("yok-boyle-ajan", store=st, root=tmp_path)
    assert dec.allowed is False
    assert dec.blocked_by == "unknown_agent"


def test_run_with_supervision_blocked_returns_dict(st, tmp_path) -> None:
    supervisor.create_stop_all("dur", root=tmp_path)
    out = supervisor.run_with_supervision(
        "auto-lora-pipeline",
        lambda: {"ran": True},
        "auto_lora_start_training",
        store=st,
        root=tmp_path,
    )
    assert isinstance(out, dict) and out.get("blocked") is True
    assert out.get("blocked_by") == "stop_all"


def test_run_with_supervision_runs_readonly(st, tmp_path) -> None:
    out = supervisor.run_with_supervision(
        "model-advisor", lambda: {"ran": True}, "recommend", store=st, root=tmp_path
    )
    assert out == {"ran": True}


def test_startup_sweep_cancels_stale_running(st) -> None:
    """Startup sweep: bayat (eski) 'running' koşuları 'cancelled' yapar; tazeyi korur."""
    import datetime as dt

    from app.agents.runtime.tracker import cancel_stale_running_agent_runs

    old_ts = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=48)).isoformat()
    fresh_ts = dt.datetime.now(dt.UTC).isoformat()
    st.create_agent_run(run_id="arun_old", agent_id="x", status="running", started_at=old_ts)
    st.create_agent_run(run_id="arun_fresh", agent_id="x", status="running", started_at=fresh_ts)

    swept = cancel_stale_running_agent_runs(store=st, stale_after_hours=6)
    assert "arun_old" in swept
    assert "arun_fresh" not in swept  # taze koşu korunur (eşzamanlı koşu güvenliği)

    old_run = st.get_agent_run("arun_old")
    fresh_run = st.get_agent_run("arun_fresh")
    assert old_run is not None and old_run["status"] == "cancelled"
    assert fresh_run is not None and fresh_run["status"] == "running"
