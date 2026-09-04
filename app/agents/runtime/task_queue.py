"""Task queue (Phase 2) — basit, izlenebilir otomasyon görev kuyruğu.

Windows Task Scheduler DIŞ cron olarak KALIR (Phase 2'de app içine taşınmaz —
kullanıcı kararı): bu kuyruk yalnız görevleri KAYIT + DURUM olarak izler. Görev
yürütücü (claim→running→complete) ileride supervisor üstünden bağlanır; şimdilik
durum geçişleri + iptal sağlanır.

Durumlar: pending, claimed, running, completed, failed, cancelled,
blocked_approval, blocked_stop_all.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import TYPE_CHECKING

from app.agents.runtime.schemas import AutomationTask, TaskStatus

if TYPE_CHECKING:
    from app.memory.sqlite_store import SqliteStore

log = logging.getLogger(__name__)

_TERMINAL = {TaskStatus.completed.value, TaskStatus.failed.value, TaskStatus.cancelled.value}


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _new_id() -> str:
    return f"atask_{uuid.uuid4().hex[:12]}"


def _store(store: SqliteStore | None) -> SqliteStore:
    if store is not None:
        return store
    from app.memory.sqlite_store import SqliteStore as _S

    return _S(check_same_thread=False)


def _event(message: str, agent_id: str, task_id: str, level: str = "info") -> None:
    try:
        from app.agents.runtime.tracker import log_system_event

        log_system_event(message, agent_id=agent_id, level=level, payload={"task_id": task_id})
    except Exception:
        log.debug("task event yazılamadı", exc_info=True)


def create_task(
    agent_id: str,
    title: str,
    description: str | None = None,
    params: dict | None = None,
    requires_approval: bool = False,
    schedule: str | None = None,
    store: SqliteStore | None = None,
) -> AutomationTask:
    """Yeni bir PENDING görev oluştur."""
    st = _store(store)
    task_id = _new_id()
    st.create_automation_task(
        task_id=task_id,
        agent_id=agent_id,
        title=title,
        description=description,
        params=params,
        schedule=schedule,
        status=TaskStatus.pending.value,
        requires_approval=requires_approval,
    )
    _event(f"Görev oluşturuldu: {title}", agent_id, task_id)
    row = st.get_automation_task(task_id)
    assert row is not None
    return AutomationTask(**row)


def list_tasks(
    limit: int = 50,
    status: str | None = None,
    agent_id: str | None = None,
    store: SqliteStore | None = None,
) -> list[AutomationTask]:
    st = _store(store)
    return [
        AutomationTask(**r)
        for r in st.list_automation_tasks(limit=limit, status=status, agent_id=agent_id)
    ]


def get_task(task_id: str, store: SqliteStore | None = None) -> AutomationTask | None:
    st = _store(store)
    row = st.get_automation_task(task_id)
    return AutomationTask(**row) if row else None


def claim_task(task_id: str, store: SqliteStore | None = None) -> AutomationTask | None:
    """pending → claimed. Yalnız pending claim edilebilir (aksi halde değişmeden döner).

    NOT: Bu fonksiyon get+update (iki transaction) yapar — geriye-dönük uyumlu ama
    eşzamanlı yarışta atomik DEĞİLDİR. Yürütücü (executor) çift-çalıştırmayı önlemek
    için ``try_claim_task`` (atomik CAS) kullanır.
    """
    st = _store(store)
    cur = st.get_automation_task(task_id)
    if cur is None:
        return None
    if cur["status"] != TaskStatus.pending.value:
        return AutomationTask(**cur)
    row = st.update_automation_task(task_id, status=TaskStatus.claimed.value, claimed_at=_utcnow())
    return AutomationTask(**row) if row else None


def try_claim_task(task_id: str, store: SqliteStore | None = None) -> AutomationTask | None:
    """ATOMİK claim (CAS): yalnız ``pending`` ise ``claimed`` yapar — TEK transaction.

    Görevi YALNIZ bu çağrı claim ettiyse (yarışı kazandıysa) döndürür; aksi halde
    (yok / pending değil / eşzamanlı başka çağrı aldı) ``None`` döner. Böylece iki
    eşzamanlı yürütücü aynı görevi iki kez ÇALIŞTIRAMAZ.
    """
    st = _store(store)
    row = st.claim_automation_task_atomic(task_id)
    return AutomationTask(**row) if row else None


def complete_task(
    task_id: str, summary: str | None = None, store: SqliteStore | None = None
) -> AutomationTask | None:
    """Görevi tamamlandı işaretle (terminal durumdaysa değişmez)."""
    st = _store(store)
    cur = st.get_automation_task(task_id)
    if cur is None:
        return None
    if cur["status"] in _TERMINAL:
        return AutomationTask(**cur)
    row = st.update_automation_task(
        task_id, status=TaskStatus.completed.value, completed_at=_utcnow(), error=None
    )
    if summary:
        _event(f"Görev tamamlandı: {summary}", cur["agent_id"], task_id)
    return AutomationTask(**row) if row else None


def fail_task(task_id: str, error: str, store: SqliteStore | None = None) -> AutomationTask | None:
    st = _store(store)
    cur = st.get_automation_task(task_id)
    if cur is None:
        return None
    if cur["status"] in _TERMINAL:
        return AutomationTask(**cur)
    row = st.update_automation_task(
        task_id, status=TaskStatus.failed.value, completed_at=_utcnow(), error=error
    )
    _event(f"Görev başarısız: {error}", cur["agent_id"], task_id, level="error")
    return AutomationTask(**row) if row else None


def cancel_task(
    task_id: str, reason: str | None = None, store: SqliteStore | None = None
) -> AutomationTask | None:
    """Görevi iptal et (zaten terminal ise değişmez)."""
    st = _store(store)
    cur = st.get_automation_task(task_id)
    if cur is None:
        return None
    if cur["status"] in _TERMINAL:
        return AutomationTask(**cur)
    row = st.update_automation_task(
        task_id,
        status=TaskStatus.cancelled.value,
        completed_at=_utcnow(),
        error=reason or "cancelled",
    )
    _event(f"Görev iptal edildi: {reason or '—'}", cur["agent_id"], task_id, level="warning")
    return AutomationTask(**row) if row else None


def mark_blocked(
    task_id: str, blocked_by: str, store: SqliteStore | None = None
) -> AutomationTask | None:
    """Görevi STOP_ALL veya onay nedeniyle bloklandı işaretle (supervisor kullanır)."""
    st = _store(store)
    cur = st.get_automation_task(task_id)
    if cur is None or cur["status"] in _TERMINAL:
        return AutomationTask(**cur) if cur else None
    status = (
        TaskStatus.blocked_stop_all.value
        if blocked_by == "stop_all"
        else TaskStatus.blocked_approval.value
    )
    row = st.update_automation_task(task_id, status=status)
    _event(f"Görev bloklandı: {blocked_by}", cur["agent_id"], task_id, level="warning")
    return AutomationTask(**row) if row else None


_BLOCKED = {TaskStatus.blocked_approval.value, TaskStatus.blocked_stop_all.value}


def requeue_task(task_id: str, store: SqliteStore | None = None) -> AutomationTask | None:
    """Bloklanmış (``blocked_approval`` / ``blocked_stop_all``) bir görevi yeniden PENDING yap.

    Yalnız ``blocked_*`` durumundakiler yeniden kuyruğa alınır (claim/error temizlenir);
    diğer durumlar değişmeden döner. Onay verildikten/STOP_ALL kalktıktan sonra
    executor'ın görevi tekrar denemesini sağlar.
    """
    st = _store(store)
    cur = st.get_automation_task(task_id)
    if cur is None:
        return None
    if cur["status"] not in _BLOCKED:
        return AutomationTask(**cur)
    row = st.update_automation_task(
        task_id, status=TaskStatus.pending.value, claimed_at=None, error=None
    )
    _event("Görev yeniden kuyruğa alındı", cur["agent_id"], task_id)
    return AutomationTask(**row) if row else None
