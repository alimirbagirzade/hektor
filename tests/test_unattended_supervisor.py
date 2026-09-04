from __future__ import annotations

import asyncio

from app.orchestration.unattended_supervisor import UnattendedSupervisor


def test_reconcile_keeps_existing_motor_without_spawn(monkeypatch, tmp_path) -> None:
    controller = UnattendedSupervisor(tmp_path / "state.json")
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda: False)
    monkeypatch.setattr("app.orchestration.engine_procs.live_count", lambda: 1)
    monkeypatch.setattr(
        controller,
        "_ensure_run",
        lambda: (_ for _ in ()).throw(AssertionError("new run created")),
    )

    result = asyncio.run(controller.reconcile_once())

    assert result["action"] == "motor_alive"
    assert controller.state.status == "running"


def test_reconcile_starts_one_codex_motor(monkeypatch, tmp_path) -> None:
    controller = UnattendedSupervisor(tmp_path / "state.json")
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda: False)
    monkeypatch.setattr("app.orchestration.engine_procs.live_count", lambda: 0)
    monkeypatch.setattr(controller, "_ensure_run", lambda: "orc_auto")
    calls: list[tuple[str, bool, str, str]] = []

    def _drive(self, run_id, *, execute, engine, mode):
        calls.append((run_id, execute, engine, mode))
        return {"ok": True}

    monkeypatch.setattr("app.orchestration.driver.AutoDriver.drive", _drive)

    async def _run() -> dict:
        result = await controller.reconcile_once()
        assert controller._driver_task is not None
        await controller._driver_task
        return result

    result = asyncio.run(_run())

    assert result == {"ok": True, "action": "motor_started", "run_id": "orc_auto"}
    assert calls == [("orc_auto", True, "codex", "drive")]


def test_stop_all_prevents_motor_start(monkeypatch, tmp_path) -> None:
    controller = UnattendedSupervisor(tmp_path / "state.json")
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda: True)
    result = asyncio.run(controller.reconcile_once())
    assert result["action"] == "stop_all"
    assert controller.state.status == "blocked_stop_all"
