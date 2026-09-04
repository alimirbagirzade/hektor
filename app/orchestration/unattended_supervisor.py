"""Durable reconciler that keeps the Hektor motor attached without a human operator."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _state_file() -> Path:
    """Durum dosyası — CWD'ye değil `settings.state_dir`'e bağlı."""
    from app.config import get_settings

    return get_settings().state_dir / "unattended_supervisor_state.json"


_BASE_BACKOFF_S = 300
_MAX_BACKOFF_S = 21600


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _is_productive(result: dict[str, Any]) -> bool:
    """Motor koşusu ilerleme sağladı mı? (geri çekilme kararının tek ölçütü).

    ``ok`` YALNIZ "sürücü çağrısı hata vermeden döndü" demektir — motorun verdict'i
    PASS demek DEĞİLDİR. Eskiden geri çekilme yalnız ``ok``'a bakıyordu: verdict'i
    FAIL/unknown olan sür koşusu da ``ok=True`` döndüğü için ``failures`` her turda
    sıfırlanıyor, ``retry_after`` hiç yazılmıyordu → motor dakikada bir yeniden
    doğuyor, hep aynı yerde düşüyor ve abonelik kotasını yakıyordu. Verdict bayrakları
    artık kapıya dâhildir.

    ⛔ DURDUR (STOP_ALL) ile kesilen koşu başarısızlık SAYILMAZ: bu insan kararıdır,
    motorun hatası değil — ayrıca stop_all guard'ı zaten yeni doğuşu engeller.
    """
    if not result.get("ok"):
        return False
    if result.get("stopped"):
        return True
    return all(bool(result[flag]) for flag in ("drive_passed", "hunt_passed") if flag in result)


@dataclass
class UnattendedState:
    enabled: bool = True
    status: str = "idle"
    engine: str = "codex"
    run_id: str = ""
    last_reconcile_at: str = ""
    last_motor_start_at: str = ""
    last_result: dict[str, Any] = field(default_factory=dict)
    failures: int = 0
    retry_after: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class UnattendedSupervisor:
    """Reconcile desired state: server up, one motor attached, pipelines progressing."""

    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = state_path or _state_file()
        self.state = self._load()
        self._driver_task: asyncio.Task[dict[str, Any]] | None = None
        self._lock = asyncio.Lock()

    def _load(self) -> UnattendedState:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            allowed = {f.name for f in dataclasses.fields(UnattendedState)}
            state = UnattendedState(**{k: v for k, v in raw.items() if k in allowed})
            state.status = "idle"  # process-local tasks never survive a server restart
            return state
        except Exception:
            return UnattendedState()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def status(self) -> dict[str, Any]:
        return self.state.to_dict()

    def _in_backoff(self) -> bool:
        try:
            return _now() < dt.datetime.fromisoformat(self.state.retry_after)
        except ValueError:
            return False

    def _record_driver_result(self, result: dict[str, Any]) -> None:
        self.state.last_result = result
        if _is_productive(result):
            self.state.failures = 0
            self.state.retry_after = ""
        else:
            self.state.failures += 1
            delay = min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * 2 ** (self.state.failures - 1))
            self.state.retry_after = (_now() + dt.timedelta(seconds=delay)).isoformat()
            log.warning(
                "Gözetimsiz motor ilerleme sağlamadı (%d. kez) → %d sn geri çekilme. Sebep: %s",
                self.state.failures,
                delay,
                result.get("reason") or (result.get("verdict") or {}).get("summary") or "-",
            )
        self.state.status = "idle"
        self._save()

    def _ensure_run(self) -> str:
        from app.config import get_settings
        from app.orchestration.orchestrator import TrainingOrchestrator

        orch = TrainingOrchestrator()
        runs = orch.list_runs(limit=1)
        if runs and runs[0].get("status") not in {"completed", "cancelled", "failed"}:
            return str(runs[0]["run_id"])
        settings = get_settings()
        return orch.start(
            model=settings.peft_base_model,
            profile="discipline_safe_local",
            adapter_name="hektor_lora",
            params={"iters": 300, "hunt_ack": False, "unattended": True},
        )

    async def reconcile_once(self) -> dict[str, Any]:
        async with self._lock:
            self.state.last_reconcile_at = _now().isoformat()
            if not self.state.enabled:
                self.state.status = "disabled"
                self._save()
                return {"ok": True, "action": "disabled"}

            from app.agents.runtime import supervisor
            from app.orchestration import engine_procs

            if supervisor.is_stop_all_active():
                self.state.status = "blocked_stop_all"
                self._save()
                return {"ok": True, "action": "stop_all"}

            if self._driver_task is not None and self._driver_task.done():
                try:
                    self._record_driver_result(self._driver_task.result())
                except Exception as exc:
                    self._record_driver_result({"ok": False, "reason": str(exc)})
                self._driver_task = None

            if engine_procs.live_count() > 0 or (
                self._driver_task is not None and not self._driver_task.done()
            ):
                self.state.status = "running"
                self._save()
                return {"ok": True, "action": "motor_alive", "run_id": self.state.run_id}

            if self._in_backoff():
                self.state.status = "backoff"
                self._save()
                return {"ok": True, "action": "backoff", "retry_after": self.state.retry_after}

            run_id = await asyncio.to_thread(self._ensure_run)
            self.state.run_id = run_id
            self.state.status = "starting"
            self.state.last_motor_start_at = _now().isoformat()
            self._save()

            from app.orchestration.driver import AutoDriver

            self._driver_task = asyncio.create_task(
                asyncio.to_thread(
                    AutoDriver().drive,
                    run_id,
                    execute=True,
                    engine=self.state.engine,
                    mode="drive",
                )
            )
            return {"ok": True, "action": "motor_started", "run_id": run_id}

    async def background_loop(self, interval_s: float = 60.0) -> None:
        while True:
            try:
                await self.reconcile_once()
            except Exception:
                log.exception("Unattended supervisor reconcile failed")
            await asyncio.sleep(max(5.0, interval_s))


_supervisor: UnattendedSupervisor | None = None


def get_unattended_supervisor() -> UnattendedSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = UnattendedSupervisor()
    return _supervisor
