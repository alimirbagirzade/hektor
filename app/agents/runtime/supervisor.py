"""Supervisor (Phase 2) — tehlikeli ajan çalıştırma için TEK kapı.

Kontroller (sırayla): ajan registry'de mi → görev iptal mi → STOP_ALL aktif mi
(yalnız TEHLİKELİ aksiyonları bloklar; salt-okuma ajanlar STOP_ALL altında bile
çalışır) → ajan zaten çalışıyor mu (yalnız tehlikeli) → taze onay gerekiyor mu.

STOP_ALL küresel acil-durdurma anahtarıdır: ``storage/STOP_ALL`` dosyası varken
hiçbir tehlikeli aksiyon çalışmaz. Bu modül davranışı yalnız ENGELLER; hiçbir
ağır iş başlatmaz (eğitim/terfi çağıranın sorumluluğunda, onay tüketildikten sonra).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.agents.runtime.schemas import AgentAutonomy, AgentSpec, SupervisorDecision
from app.config import get_settings

if TYPE_CHECKING:
    from app.memory.sqlite_store import SqliteStore

log = logging.getLogger(__name__)

_APPROVAL_AUTONOMY = {AgentAutonomy.requires_approval, AgentAutonomy.dangerous_without_approval}


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _root(root: str | Path | None) -> Path:
    return Path(root) if root is not None else get_settings().root


def _stop_all_path(root: str | Path | None = None) -> Path:
    return _root(root) / "storage" / "STOP_ALL"


def _store(store: SqliteStore | None) -> SqliteStore:
    if store is not None:
        return store
    from app.memory.sqlite_store import SqliteStore as _S

    return _S(check_same_thread=False)


def _spec(agent_id: str) -> AgentSpec | None:
    try:
        from app.agents.runtime.registry import get_agent

        return get_agent(agent_id)
    except Exception:
        return None


def _needs_approval(spec: AgentSpec) -> bool:
    return spec.approval_required or spec.autonomy in _APPROVAL_AUTONOMY


def _risk_for(spec: AgentSpec) -> str:
    return "high" if spec.dangerous else "medium"


# --- STOP_ALL kill switch -------------------------------------------------
def is_stop_all_active(root: str | Path | None = None) -> bool:
    return _stop_all_path(root).exists()


def create_stop_all(reason: str | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Küresel acil-durdurmayı ETKİNLEŞTİR (storage/STOP_ALL yaz)."""
    p = _stop_all_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reason": reason or "", "at": _utcnow()}
    import json

    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _system_event(f"STOP_ALL ETKİN: {reason or '—'}", "supervisor", "stop_all_create", "warning")
    log.warning("STOP_ALL etkinleştirildi: %s", reason or "")
    return {"ok": True, "active": True, "reason": reason or ""}


def clear_stop_all(root: str | Path | None = None) -> dict[str, Any]:
    """Küresel acil-durdurmayı KALDIR."""
    p = _stop_all_path(root)
    existed = p.exists()
    with contextlib.suppress(Exception):
        if existed:
            p.unlink()
    _system_event("STOP_ALL kaldırıldı", "supervisor", "stop_all_clear", "info")
    log.info("STOP_ALL kaldırıldı (vardı=%s)", existed)
    return {"ok": True, "active": False, "was_active": existed}


def _system_event(message: str, agent_id: str, action: str, level: str) -> None:
    try:
        from app.agents.runtime.tracker import log_system_event

        log_system_event(message, agent_id=agent_id, level=level, payload={"action": action})
    except Exception:
        log.debug("supervisor system event yazılamadı", exc_info=True)


def _agent_running(agent_id: str, store: SqliteStore) -> bool:
    try:
        return bool(store.list_agent_runs(limit=1, agent_id=agent_id, status="running"))
    except Exception:
        return False


# --- karar + denetimli çalıştırma ----------------------------------------
def can_run_agent(
    agent_id: str,
    action: str | None = None,
    task_id: str | None = None,
    store: SqliteStore | None = None,
    root: str | Path | None = None,
) -> SupervisorDecision:
    """Bu ajan ŞİMDİ çalıştırılabilir mi? (Salt KONTROL — onay tüketmez.)"""
    spec = _spec(agent_id)
    if spec is None:
        return SupervisorDecision(
            allowed=False, reason=f"Registry'de yok: {agent_id}", blocked_by="unknown_agent"
        )

    st = _store(store)

    # görev iptal edildiyse çalıştırma
    if task_id:
        task = st.get_automation_task(task_id)
        if task and task["status"] == "cancelled":
            return SupervisorDecision(
                allowed=False, reason="Görev iptal edilmiş", blocked_by="cancelled"
            )

    # STOP_ALL yalnız TEHLİKELİ aksiyonları bloklar (salt-okuma ajanlar serbest)
    if spec.dangerous and is_stop_all_active(root):
        return SupervisorDecision(
            allowed=False,
            reason="STOP_ALL aktif — tehlikeli aksiyon bloklandı",
            blocked_by="stop_all",
        )

    # aynı tehlikeli ajanın ikinci eşzamanlı koşusunu engelle
    if spec.dangerous and _agent_running(agent_id, st):
        return SupervisorDecision(
            allowed=False, reason="Ajan zaten çalışıyor", blocked_by="already_running"
        )

    # taze onay gerekiyor mu?
    if _needs_approval(spec):
        from app.agents.runtime.approvals import has_fresh_approval

        if not has_fresh_approval(agent_id, action or "run", store=st):
            return SupervisorDecision(
                allowed=False,
                reason="Taze onay gerekli (standing yetki yok)",
                requires_approval=True,
                blocked_by="approval",
            )
        return SupervisorDecision(allowed=True, requires_approval=True, reason="Taze onay mevcut")

    return SupervisorDecision(allowed=True, reason="izinli")


def run_with_supervision(
    agent_id: str,
    callable_fn: Callable[[], Any],
    action: str,
    task_id: str | None = None,
    params: dict[str, Any] | None = None,
    store: SqliteStore | None = None,
    root: str | Path | None = None,
) -> dict[str, Any] | Any:
    """Bir aksiyonu denetim altında çalıştır.

    Engellenirse {"ok": False, "blocked": True, ...} döner ve (varsa) görevi
    bloklandı işaretler. İzinliyse: tehlikeli/onaylı ajanlar için TAZE ONAY TÜKETİR,
    sonra callable'ı tracker koşusu içinde çalıştırıp sonucunu döndürür.
    """
    dec = can_run_agent(agent_id, action=action, task_id=task_id, store=store, root=root)
    if not dec.allowed:
        if task_id and dec.blocked_by in ("stop_all", "approval"):
            from app.agents.runtime import task_queue

            task_queue.mark_blocked(task_id, dec.blocked_by, store=store)
        return {
            "ok": False,
            "blocked": True,
            "blocked_by": dec.blocked_by,
            "reason": dec.reason,
            "approval_id": dec.approval_id,
        }

    spec = _spec(agent_id)
    if spec is not None and _needs_approval(spec):
        from app.agents.runtime.approvals import require_fresh_approval

        d = require_fresh_approval(
            agent_id, action, _risk_for(spec), f"supervised: {action}", task_id=task_id, store=store
        )
        if not d.authorized:
            return {
                "ok": False,
                "blocked": True,
                "blocked_by": "approval",
                "approval_id": d.approval_id,
                "reason": "Taze onay gerekli (standing yetki yok)",
            }

    from app.agents.runtime.tracker import track_agent_run

    with track_agent_run(
        agent_id, task_id=task_id, trigger_type="supervisor", trigger_payload=params
    ) as run:
        run.log_event("step", f"denetimli çalıştırma: {action}")
        return callable_fn()
