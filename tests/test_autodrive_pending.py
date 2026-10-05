"""Sürüş başlangıcındaki görünürlük ve çift başlatma yarışı."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.orchestration import engine_procs, engines
from app.orchestration.driver import AutoDriver
from app.web import orchestration_routes as routes


def test_pending_drive_visible_and_duplicate_rejected(monkeypatch):
    orch = SimpleNamespace(store=SimpleNamespace(get_run=lambda _: {}))
    monkeypatch.setattr(routes, "_orchestrator", lambda: orch)
    monkeypatch.setattr(engines, "run_blocked_reason", lambda _: "")
    monkeypatch.setattr(engines, "drive_supported", lambda _: True)
    monkeypatch.setattr(engines, "get_engine", lambda _: SimpleNamespace(drive_hardened=True))
    monkeypatch.setattr(engine_procs, "is_run_live", lambda _: False)
    monkeypatch.setattr(routes, "_drives", set())
    req = routes.OrchestrationAutodriveRequest(execute=True, engine="codex")
    bg = BackgroundTasks()
    assert routes.orchestration_autodrive("r", req, bg)["ok"]
    assert routes._driver_running("r")
    with pytest.raises(HTTPException) as exc:
        routes.orchestration_autodrive("r", req, bg)
    assert exc.value.status_code == 409
    assert len(bg.tasks) == 1
    monkeypatch.setattr(AutoDriver, "drive", lambda *a, **kw: {"ok": True})
    routes._run_autodrive_bg("r", "codex", "drive")
    assert not routes._driver_running("r")


def test_failure_releases_pending_and_records_error(monkeypatch):
    monkeypatch.setattr(routes, "_drives", {"r"})
    record = Mock()
    monkeypatch.setattr(
        routes, "_orchestrator", lambda: SimpleNamespace(store=SimpleNamespace(add_event=record))
    )
    monkeypatch.setattr(AutoDriver, "drive", Mock(side_effect=RuntimeError("failure")))
    routes._run_autodrive_bg("r", "codex", "drive")
    assert "r" not in routes._drives
    record.assert_called_once()
