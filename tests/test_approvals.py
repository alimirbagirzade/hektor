"""Approval system (Phase 2) testleri — tek kullanımlık taze onay (offline)."""

from __future__ import annotations

import pytest

from app.agents.runtime import approvals
from app.memory.sqlite_store import SqliteStore


@pytest.fixture(autouse=True)
def _test_tracker(tmp_path):
    from app.agents.runtime.tracker import RunTracker, set_tracker

    set_tracker(RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs"))
    yield
    set_tracker(None)


@pytest.fixture
def st():
    return SqliteStore()


def test_request_list_get(st) -> None:
    r = approvals.request_approval("auto-lora-pipeline", "train_run", "yap", "critical", store=st)
    assert r.status.value == "pending"
    got = approvals.get_approval(r.approval_id, store=st)
    assert got is not None and got.approval_id == r.approval_id
    assert any(a.approval_id == r.approval_id for a in approvals.list_approvals(store=st))


def test_approve_and_reject(st) -> None:
    r = approvals.request_approval("ag", "act1", "s", "low", store=st)
    a = approvals.approve(r.approval_id, note="ok", store=st)
    assert a is not None and a.status.value == "approved" and a.decided_by == "user"
    r2 = approvals.request_approval("ag", "act2", "s", "low", store=st)
    rj = approvals.reject(r2.approval_id, store=st)
    assert rj is not None and rj.status.value == "rejected"


def test_require_fresh_creates_pending_when_none(st) -> None:
    d = approvals.require_fresh_approval("ag", "train_run", "critical", "s", store=st)
    assert d.authorized is False
    assert d.created is True
    assert d.status.value == "pending"


def test_require_fresh_consumes_approved_single_use(st) -> None:
    # 1) ilk istek → pending, onayla
    d1 = approvals.require_fresh_approval("ag", "train_run", "critical", "s", store=st)
    approvals.approve(d1.approval_id, store=st)
    # 2) ikinci istek → onaylı olanı TÜKETİR ve yetki verir
    d2 = approvals.require_fresh_approval("ag", "train_run", "critical", "s", store=st)
    assert d2.authorized is True
    assert d2.approval_id == d1.approval_id  # aynı onay tüketildi
    # 3) tek kullanımlık: üçüncü istek taze onay bulamaz → yeni pending
    d3 = approvals.require_fresh_approval("ag", "train_run", "critical", "s", store=st)
    assert d3.authorized is False
    assert d3.approval_id != d1.approval_id


def test_approved_request_only_one_action(st) -> None:
    """Onaylı bir istek YALNIZ bir aksiyon için geçerli (tüketilince biter)."""
    d = approvals.require_fresh_approval("ag2", "promote", "high", "s", store=st)
    approvals.approve(d.approval_id, store=st)
    assert approvals.has_fresh_approval("ag2", "promote", store=st) is True
    # ilk tüketim yetki verir
    used = approvals.require_fresh_approval("ag2", "promote", "high", "s", store=st)
    assert used.authorized is True
    # tüketildikten sonra taze onay yok
    assert approvals.has_fresh_approval("ag2", "promote", store=st) is False


def test_has_fresh_only_after_approve(st) -> None:
    d = approvals.require_fresh_approval("ag3", "actZ", "medium", "s", store=st)
    assert approvals.has_fresh_approval("ag3", "actZ", store=st) is False  # pending
    approvals.approve(d.approval_id, store=st)
    assert approvals.has_fresh_approval("ag3", "actZ", store=st) is True


def test_approve_missing_returns_none(st) -> None:
    assert approvals.approve("apr_yok", store=st) is None
    assert approvals.reject("apr_yok", store=st) is None


def test_consume_fresh_approval_atomic_single_use(st) -> None:
    """consume_fresh_approval (atomik CAS) bir onayı YALNIZ bir kez tüketir.

    Eski find+update iki ayrı transaction'dı → eşzamanlı çift-tüketim (TOCTOU) mümkündü.
    Yeni koşullu UPDATE + rowcount: ilk çağrı tüketir, ikinci çağrı None alır.
    """
    d = approvals.require_fresh_approval("ag-cas", "train_run", "critical", "s", store=st)
    approvals.approve(d.approval_id, store=st)
    assert st.find_fresh_approval("ag-cas", "train_run") is not None  # tüketilmeden taze
    # 1) ilk tüketim → onay döner ve consumed_at damgalanır
    first = st.consume_fresh_approval("ag-cas", "train_run")
    assert first is not None and first["approval_id"] == d.approval_id
    assert first["consumed_at"] is not None
    # 2) ikinci tüketim → artık taze onay yok (tek kullanımlık)
    assert st.consume_fresh_approval("ag-cas", "train_run") is None
    assert st.find_fresh_approval("ag-cas", "train_run") is None
