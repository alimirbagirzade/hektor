"""Phase 4D-1 — /api/training/run taze-onay kapısı (offline; GERÇEK eğitim YOK).

Audit bulgusu (4D-0): web `/api/training/run` → `launch()` doğrudan çağırıyordu →
spawn edilen `train --run` `ACHILLES_TRAIN_SUPERVISED=1` ile fresh-approval gate'i
bypass ediyordu. Bu testler endpoint'in artık CLI ile AYNI kapıdan geçtiğini
doğrular: STOP_ALL bloklar, onay yoksa eğitim BAŞLAMAZ (needs_approval), onay
varsa launch çağrılır. `launch` her zaman MOCK'lanır — hiçbir testte gerçek eğitim
başlamaz, ağ/Ollama yok.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.agents.runtime.schemas import ApprovalDecision, ApprovalStatus  # noqa: E402
from app.web.server import app  # noqa: E402

client = TestClient(app)

_PAYLOAD = {"adapter_name": "test_adapter", "iterations": 3}

_SUP = "app.agents.runtime.supervisor.is_stop_all_active"
_REQ = "app.agents.runtime.approvals.require_fresh_approval"
_LAUNCH = "app.training.detached_launch.launch"


def test_run_blocked_when_stop_all_active() -> None:
    """STOP_ALL aktifse endpoint training başlatmaz, launch çağrılmaz."""
    with (
        patch(_SUP, return_value=True),
        patch(_LAUNCH) as m_launch,
    ):
        r = client.post("/api/training/run", json=_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "blocked"
    assert body["ok"] is False
    m_launch.assert_not_called()


def test_run_requires_fresh_approval() -> None:
    """Onay yoksa: training başlamaz, needs_approval + approval_id + komut döner."""
    decision = ApprovalDecision(
        authorized=False,
        approval_id="apr_test123",
        status=ApprovalStatus.pending,
        created=True,
    )
    with (
        patch(_SUP, return_value=False),
        patch(_REQ, return_value=decision),
        patch(_LAUNCH) as m_launch,
    ):
        r = client.post("/api/training/run", json=_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "needs_approval"
    assert body["ok"] is False
    assert body["approval_id"] == "apr_test123"
    assert "approval-approve apr_test123" in body["approve_command"]
    m_launch.assert_not_called()  # GERÇEK eğitim başlamadı


def test_run_starts_when_authorized() -> None:
    """Onay tüketilebiliyorsa launch çağrılır (MOCK) ve started döner."""
    decision = ApprovalDecision(
        authorized=True,
        approval_id="apr_ok999",
        status=ApprovalStatus.approved,
        created=False,
    )
    fake = {"ok": True, "message": "Detached eğitim başlatıldı.", "adapter": "test_adapter"}
    with (
        patch(_SUP, return_value=False),
        patch(_REQ, return_value=decision),
        patch(_LAUNCH, return_value=fake) as m_launch,
    ):
        r = client.post("/api/training/run", json=_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "started"
    m_launch.assert_called_once()


def test_run_real_approval_flow_consumes_single_use() -> None:
    """Gerçek approval DB (temp) ile uçtan uca: taze onay TÜKETİLİR (tek kullanımlık).

    launch MOCK — gerçek eğitim yok. Diğer testlerden kalan taze onaylar drenajla
    temizlenir → deterministik.
    """
    from app.agents.runtime import approvals
    from app.memory.sqlite_store import SqliteStore

    st = SqliteStore()

    # Kalan taze onayları tüket (deterministik başlangıç).
    while approvals.has_fresh_approval("lora-trainer", "train_run", store=st):
        approvals.require_fresh_approval("lora-trainer", "train_run", "critical", "drain", store=st)

    # Onay yoksa endpoint needs_approval döndürür (launch çağrılmaz).
    fake = {"ok": True, "message": "ok", "adapter": "a"}
    with patch(_SUP, return_value=False), patch(_LAUNCH, return_value=fake) as m_launch:
        r0 = client.post("/api/training/run", json=_PAYLOAD)
        assert r0.json()["status"] == "needs_approval"
        m_launch.assert_not_called()

        # Bekleyen onayı ONAYLA → sonra istek eğitim başlatabilir.
        pending = approvals.list_approvals(status="pending", store=st)
        mine = next(a for a in pending if a.agent_id == "lora-trainer" and a.action == "train_run")
        approvals.approve(mine.approval_id, store=st)
        assert approvals.has_fresh_approval("lora-trainer", "train_run", store=st) is True

        r1 = client.post("/api/training/run", json=_PAYLOAD)
        assert r1.json()["ok"] is True
        assert r1.json()["status"] == "started"
        m_launch.assert_called_once()

    # Onay TÜKETİLDİ → tek kullanımlık (standing yetki yok).
    consumed = approvals.get_approval(mine.approval_id, store=st)
    assert consumed is not None and consumed.consumed_at is not None
    assert approvals.has_fresh_approval("lora-trainer", "train_run", store=st) is False
