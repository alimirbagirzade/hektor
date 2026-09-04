"""Bounded self-healing loop: observe, diagnose, repair, verify, back off."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.monitoring.sentinel import Sentinel, SentinelReport

log = logging.getLogger(__name__)


def _state_file() -> Path:
    """Durum dosyası — CWD'ye değil `settings.state_dir`'e bağlı."""
    from app.config import get_settings

    return get_settings().state_dir / "self_heal_state.json"


_FAILURE_THRESHOLD = 3
_BASE_COOLDOWN_S = 300
_MAX_COOLDOWN_S = 21600
Repair = Callable[[], Awaitable[dict[str, Any]]]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass
class SelfHealState:
    enabled: bool = True
    running: bool = False
    last_check_at: str = ""
    last_action_at: str = ""
    consecutive: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    cooldown_until: dict[str, str] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class SelfHealingController:
    """Only executes registered idempotent runbooks; never shell, git, or free-form code."""

    def __init__(
        self,
        *,
        sentinel: Sentinel | None = None,
        repairs: dict[str, Repair] | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.sentinel = sentinel or Sentinel()
        self.state_path = state_path or _state_file()
        self.state = self._load()
        self.repairs = repairs or {
            "orchestration": self._repair_orchestration,
            "rag_loop": self._repair_rag_loop,
        }
        self._lock = asyncio.Lock()

    def _load(self) -> SelfHealState:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            allowed = {f.name for f in dataclasses.fields(SelfHealState)}
            return SelfHealState(**{k: v for k, v in raw.items() if k in allowed})
        except Exception:
            return SelfHealState()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def status(self) -> dict[str, Any]:
        return self.state.to_dict()

    def _cooling_down(self, name: str) -> bool:
        try:
            return _now() < dt.datetime.fromisoformat(self.state.cooldown_until.get(name, ""))
        except ValueError:
            return False

    async def run_once(self, report: SentinelReport | None = None) -> dict[str, Any]:
        async with self._lock:
            if self.state.running:
                return {"ok": False, "reason": "self-heal zaten çalışıyor"}
            self.state.running = True
            self._save()

        actions: list[dict[str, Any]] = []
        try:
            if report is None:
                active_report = await asyncio.to_thread(self.sentinel.run, persist=True)
            else:
                active_report = report
            self.state.last_check_at = _now().isoformat()
            by_name = {p.name: p for p in active_report.probes}
            for name, repair in self.repairs.items():
                probe = by_name.get(name)
                unhealthy = probe is not None and probe.status in {"warn", "fail"}
                self.state.consecutive[name] = (
                    self.state.consecutive.get(name, 0) + 1 if unhealthy else 0
                )
                if (
                    not unhealthy
                    or self.state.consecutive[name] < _FAILURE_THRESHOLD
                    or self._cooling_down(name)
                ):
                    continue

                attempt = self.state.attempts.get(name, 0) + 1
                self.state.attempts[name] = attempt
                result = await repair()
                verified = bool(result.get("verified"))
                action: dict[str, Any] = {
                    "at": _now().isoformat(),
                    "probe": name,
                    "attempt": attempt,
                    "verified": verified,
                    "result": result,
                }
                actions.append(action)
                self.state.history.append(action)
                self.state.last_action_at = action["at"]
                if verified:
                    self.state.consecutive[name] = 0
                    self.state.attempts[name] = 0
                    self.state.cooldown_until.pop(name, None)
                else:
                    delay = min(_MAX_COOLDOWN_S, _BASE_COOLDOWN_S * (2 ** (attempt - 1)))
                    self.state.cooldown_until[name] = (
                        _now() + dt.timedelta(seconds=delay)
                    ).isoformat()

            self.state.history = self.state.history[-100:]
            return {"ok": True, "overall": active_report.overall, "actions": actions}
        finally:
            self.state.running = False
            self._save()

    async def _repair_orchestration(self) -> dict[str, Any]:
        from app.orchestration.orchestrator import TrainingOrchestrator

        recovered = await asyncio.to_thread(TrainingOrchestrator().recover_stale, timeout_min=30.0)
        verify = await asyncio.to_thread(self.sentinel.run, persist=True)
        probe = next((p for p in verify.probes if p.name == "orchestration"), None)
        return {
            "runbook": "recover_stale",
            "recovered": recovered,
            "verified": bool(probe and probe.status == "ok"),
        }

    async def _repair_rag_loop(self) -> dict[str, Any]:
        from app.research.rag_learning_loop import get_rag_loop

        loop = get_rag_loop()
        result = await loop.run_one_cycle()
        verified = loop.get_status().get("stage") != "error" and bool(result.get("ok"))
        return {"runbook": "rag_cycle_retry", "cycle": result, "verified": verified}

    async def background_loop(self, interval_s: float = 60.0) -> None:
        while True:
            if self.state.enabled:
                try:
                    await self.run_once()
                except Exception:
                    log.exception("Self-heal turu başarısız")
            await asyncio.sleep(max(5.0, interval_s))


_controller: SelfHealingController | None = None


def get_self_healer() -> SelfHealingController:
    global _controller
    if _controller is None:
        _controller = SelfHealingController()
    return _controller
