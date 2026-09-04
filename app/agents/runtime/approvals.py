"""Approval system (Phase 2) — tehlikeli aksiyonlar için TEK KULLANIMLIK taze onay.

İlke: standing / kalıcı yetki YOK. Her tehlikeli aksiyon (gerçek LoRA eğitimi,
adapter terfisi, kural uygulama) çalışmadan önce TAZE bir onay tüketir. Onay verme
kararı her zaman insandadır (CLI/web). Onaylanmış bir istek bir kez tüketilince
(``consumed_at``) bir daha kullanılamaz → her koşu yeni onay ister.

Bu modül davranışı zorlamaz; çağıran (train komutu, auto_pipeline, supervisor)
``require_fresh_approval`` sonucuna göre aksiyonu başlatır veya engeller.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import TYPE_CHECKING

from app.agents.runtime.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    RiskLevel,
)

if TYPE_CHECKING:
    from app.memory.sqlite_store import SqliteStore

log = logging.getLogger(__name__)


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _new_id() -> str:
    return f"apr_{uuid.uuid4().hex[:12]}"


def _store(store: SqliteStore | None) -> SqliteStore:
    if store is not None:
        return store
    from app.memory.sqlite_store import SqliteStore as _S

    return _S(check_same_thread=False)


def _risk_str(risk: RiskLevel | str) -> str:
    return risk.value if isinstance(risk, RiskLevel) else str(risk)


def _system_event(message: str, agent_id: str, action: str, approval_id: str, level: str) -> None:
    """Onay yaşam döngüsünü genel olay akışına (run_id='system') yaz. Asla fırlatmaz."""
    try:
        from app.agents.runtime.tracker import log_system_event

        log_system_event(
            message,
            agent_id=agent_id,
            level=level,
            payload={"action": action, "approval_id": approval_id},
        )
    except Exception:
        log.debug("approval system event yazılamadı", exc_info=True)


def request_approval(
    agent_id: str,
    action: str,
    summary: str,
    risk: RiskLevel | str,
    run_id: str | None = None,
    task_id: str | None = None,
    store: SqliteStore | None = None,
) -> ApprovalRequest:
    """Yeni bir PENDING onay isteği oluştur."""
    st = _store(store)
    approval_id = _new_id()
    st.create_approval_request(
        approval_id=approval_id,
        agent_id=agent_id,
        action=action,
        summary=summary,
        risk=_risk_str(risk),
        status=ApprovalStatus.pending.value,
        run_id=run_id,
        task_id=task_id,
    )
    _system_event(
        f"Onay istendi: {action} ({_risk_str(risk)})", agent_id, action, approval_id, "warning"
    )
    row = st.get_approval_request(approval_id)
    assert row is not None
    return ApprovalRequest(**row)


def list_approvals(
    status: str | None = None, limit: int = 50, store: SqliteStore | None = None
) -> list[ApprovalRequest]:
    st = _store(store)
    return [ApprovalRequest(**r) for r in st.list_approval_requests(status=status, limit=limit)]


def get_approval(approval_id: str, store: SqliteStore | None = None) -> ApprovalRequest | None:
    st = _store(store)
    row = st.get_approval_request(approval_id)
    return ApprovalRequest(**row) if row else None


def approve(
    approval_id: str,
    decided_by: str = "user",
    note: str | None = None,
    store: SqliteStore | None = None,
) -> ApprovalRequest | None:
    """Bekleyen bir onayı ONAYLA. Yalnız 'pending' onaylanabilir."""
    st = _store(store)
    cur = st.get_approval_request(approval_id)
    if cur is None:
        return None
    if cur["status"] != ApprovalStatus.pending.value:
        return ApprovalRequest(**cur)  # idempotent: zaten karar verilmiş
    row = st.update_approval_request(
        approval_id,
        status=ApprovalStatus.approved.value,
        decided_at=_utcnow(),
        decided_by=decided_by,
        decision_note=note,
    )
    _system_event(
        f"Onay ONAYLANDI: {cur['action']}", cur["agent_id"], cur["action"], approval_id, "info"
    )
    return ApprovalRequest(**row) if row else None


def reject(
    approval_id: str,
    decided_by: str = "user",
    note: str | None = None,
    store: SqliteStore | None = None,
) -> ApprovalRequest | None:
    """Bekleyen bir onayı REDDET."""
    st = _store(store)
    cur = st.get_approval_request(approval_id)
    if cur is None:
        return None
    if cur["status"] != ApprovalStatus.pending.value:
        return ApprovalRequest(**cur)
    row = st.update_approval_request(
        approval_id,
        status=ApprovalStatus.rejected.value,
        decided_at=_utcnow(),
        decided_by=decided_by,
        decision_note=note,
    )
    _system_event(
        f"Onay REDDEDİLDİ: {cur['action']}", cur["agent_id"], cur["action"], approval_id, "warning"
    )
    return ApprovalRequest(**row) if row else None


def require_fresh_approval(
    agent_id: str,
    action: str,
    risk: RiskLevel | str,
    summary: str,
    run_id: str | None = None,
    task_id: str | None = None,
    store: SqliteStore | None = None,
) -> ApprovalDecision:
    """Taze onay zorunluluğu (tek kapı).

    Onaylanmış + tüketilmemiş bir onay varsa → TÜKET ve yetki ver (authorized=True).
    Yoksa → yeni PENDING onay oluştur ve yetki VERME (authorized=False) → çağıran
    aksiyonu başlatmamalı, kullanıcıya approval_id'yi göstermeli.
    """
    st = _store(store)
    # Atomik BUL+TÜKET (tek transaction; TOCTOU çift-tüketim yok — bkz. consume_fresh_approval).
    fresh = st.consume_fresh_approval(agent_id, action)
    if fresh:
        _system_event(
            f"Taze onay TÜKETİLDİ: {action}", agent_id, action, fresh["approval_id"], "info"
        )
        return ApprovalDecision(
            authorized=True,
            approval_id=fresh["approval_id"],
            status=ApprovalStatus.approved,
            created=False,
        )
    req = request_approval(
        agent_id, action, summary, risk, run_id=run_id, task_id=task_id, store=st
    )
    return ApprovalDecision(
        authorized=False,
        approval_id=req.approval_id,
        status=ApprovalStatus.pending,
        created=True,
    )


def has_fresh_approval(agent_id: str, action: str, store: SqliteStore | None = None) -> bool:
    """Tüketmeden, taze (approved + unconsumed) onay var mı diye bak (supervisor için)."""
    return _store(store).find_fresh_approval(agent_id, action) is not None
