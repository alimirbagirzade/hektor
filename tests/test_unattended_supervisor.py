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


def test_backoff_triggers_when_drive_verdict_fails(tmp_path) -> None:
    """`ok=True` ama `drive_passed=False` → geri çekilme YAZILMALI.

    Regresyon: geri çekilme yalnız `ok`'a bakıyordu; sür verdict'i düşen koşu da
    `ok=True` döndüğü için `failures` her turda sıfırlanıyor, motor dakikada bir
    yeniden doğup abonelik kotasını yakıyordu.
    """
    controller = UnattendedSupervisor(tmp_path / "state.json")

    controller._record_driver_result({"ok": True, "drove": True, "drive_passed": False})

    assert controller.state.failures == 1
    assert controller.state.retry_after
    assert controller._in_backoff()


def test_backoff_clears_after_productive_run(tmp_path) -> None:
    """Verdict geçen koşu sayacı ve `retry_after`'ı sıfırlar."""
    controller = UnattendedSupervisor(tmp_path / "state.json")
    controller._record_driver_result({"ok": True, "drive_passed": False})

    controller._record_driver_result({"ok": True, "drive_passed": True})

    assert controller.state.failures == 0
    assert controller.state.retry_after == ""
    assert not controller._in_backoff()


def test_stop_all_result_is_not_counted_as_failure(tmp_path) -> None:
    """⛔ DURDUR insan kararıdır — motorun başarısızlığı sayılmaz."""
    controller = UnattendedSupervisor(tmp_path / "state.json")

    controller._record_driver_result({"ok": True, "stopped": True, "drive_passed": False})

    assert controller.state.failures == 0
    assert controller.state.retry_after == ""


def test_driver_error_still_backs_off(tmp_path) -> None:
    """`ok=False` (spawn/timeout hatası) davranışı değişmedi."""
    controller = UnattendedSupervisor(tmp_path / "state.json")

    controller._record_driver_result({"ok": False, "reason": "CLI PATH'te yok"})

    assert controller.state.failures == 1
    assert controller._in_backoff()
