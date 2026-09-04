"""Runtime ön-uçuş (preflight) testleri — offline, izole DB."""

from __future__ import annotations

import pytest

from app.agents.runtime.preflight import runtime_preflight
from app.memory.sqlite_store import SqliteStore


@pytest.fixture
def st():
    return SqliteStore()


def test_preflight_ok_on_initialized_db(st) -> None:
    r = runtime_preflight(st)
    assert r["ok"] is True
    assert r["errors"] == []
    # manifest gerçek automation_manifest.yaml'dan okunur
    assert r["agents"] > 0
    assert r["dangerous"] >= 1  # auto-lora-pipeline tehlikeli
    assert r["approval_required"] >= 1
    # 4 Phase-2 tablosu da sorgulanabilir olmalı
    assert r["tables"]["agent_runs"] is True
    assert r["tables"]["automation_tasks"] is True
    assert r["tables"]["approval_requests"] is True
    assert isinstance(r["stop_all"], bool)


def test_preflight_result_shape(st) -> None:
    r = runtime_preflight(st)
    for key in ("ok", "agents", "dangerous", "approval_required", "tables", "stop_all", "errors"):
        assert key in r
