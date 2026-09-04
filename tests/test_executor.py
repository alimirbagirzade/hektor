"""Executor (Phase 2.5) testleri — kuyruk→supervisor köprüsü (offline, izole).

Allow-list (yalnız kayıtlı handler), STOP_ALL/taze-onay kapısı ve hata izolasyonu
kanıtlanır. Paylaşılan oturum DB'sine karşı dayanıklı: kendi task_id'lerimizin
durumuna bakarız, tüm kuyruğa değil.
"""

from __future__ import annotations

import pytest

from app.agents.runtime import approvals, executor, supervisor, task_queue
from app.memory.sqlite_store import SqliteStore


@pytest.fixture(autouse=True)
def _test_tracker(tmp_path):
    from app.agents.runtime.tracker import RunTracker, set_tracker

    set_tracker(RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs"))
    yield
    set_tracker(None)


@pytest.fixture(autouse=True)
def _clean_handlers():
    """Global handler kaydını her testten sonra eski haline döndür (test sızıntısı yok)."""
    snapshot = dict(executor._HANDLERS)
    yield
    executor._HANDLERS.clear()
    executor._HANDLERS.update(snapshot)


@pytest.fixture
def st():
    return SqliteStore()


# --- allow-list -----------------------------------------------------------
def test_unknown_agent_fails_never_runs(st) -> None:
    """Handler'ı olmayan agent_id sessizce çalışmaz → görev başarısız."""
    t = task_queue.create_task("model-advisor", "kayıtsız", store=st)
    out = executor.run_task(t, store=st)
    assert out["ok"] is False
    assert out["reason"] == "handler yok"
    assert task_queue.get_task(t.task_id, store=st).status.value == "failed"


def test_register_guard_and_listing() -> None:
    executor.register_handler("model-advisor", lambda task: {"ok": True})
    assert "model-advisor" in executor.registered_agents()
    with pytest.raises(ValueError):
        executor.register_handler("model-advisor", lambda task: None)  # replace=False
    executor.register_handler("model-advisor", lambda task: {"ok": True}, replace=True)  # OK


# --- mutlu yol + hata yolları --------------------------------------------
def test_handler_completes(st) -> None:
    seen = {}
    executor.register_handler(
        "model-advisor", lambda task: seen.update(id=task.task_id) or {"ok": True}
    )
    t = task_queue.create_task("model-advisor", "öneri", store=st)
    out = executor.run_task(t, store=st)
    assert out["ok"] is True
    assert seen["id"] == t.task_id  # handler gerçekten çağrıldı
    assert task_queue.get_task(t.task_id, store=st).status.value == "completed"


def test_handler_exception_fails_task(st) -> None:
    def boom(task):
        raise RuntimeError("patladı")

    executor.register_handler("model-advisor", boom)
    t = task_queue.create_task("model-advisor", "hata", store=st)
    out = executor.run_task(t, store=st)
    assert out["ok"] is False
    done = task_queue.get_task(t.task_id, store=st)
    assert done.status.value == "failed"
    assert "patladı" in (done.error or "")


def test_handler_ok_false_marks_failed(st) -> None:
    executor.register_handler("model-advisor", lambda task: {"ok": False, "reason": "boş"})
    t = task_queue.create_task("model-advisor", "ok-false", store=st)
    out = executor.run_task(t, store=st)
    assert out["ok"] is False and out["reason"] == "boş"
    assert task_queue.get_task(t.task_id, store=st).status.value == "failed"


def test_run_pending_processes_own_tasks(st) -> None:
    executor.register_handler("model-advisor", lambda task: {"ok": True})
    a = task_queue.create_task("model-advisor", "a", store=st)
    b = task_queue.create_task("model-advisor", "b", store=st)
    executor.run_pending(limit=50, store=st)
    assert task_queue.get_task(a.task_id, store=st).status.value == "completed"
    assert task_queue.get_task(b.task_id, store=st).status.value == "completed"


# --- onay kapısı (Kural 8) ------------------------------------------------
def test_approval_gate_blocks_then_runs(st, tmp_path) -> None:
    """rules-updater onay ister: ilk koşu bloklanır; onay+requeue sonrası çalışır."""
    executor.register_handler("rules-updater", lambda task: {"ok": True})
    t = task_queue.create_task("rules-updater", "kural önerisi", store=st)

    # 1) taze onay yok → blocked_approval
    out = executor.run_task(t, store=st)
    assert out.get("blocked") is True and out["blocked_by"] == "approval"
    assert task_queue.get_task(t.task_id, store=st).status.value == "blocked_approval"

    # 2) insan onayı ver (action 'run' executor'ın türettiğiyle eşleşmeli)
    d = approvals.require_fresh_approval("rules-updater", "run", "medium", "test", store=st)
    approvals.approve(d.approval_id, store=st)

    # 3) yeniden kuyruğa al + çalıştır → tüketir ve tamamlar
    task_queue.requeue_task(t.task_id, store=st)
    out2 = executor.run_task(t, store=st)
    assert out2["ok"] is True
    assert task_queue.get_task(t.task_id, store=st).status.value == "completed"


def test_run_pending_retry_blocked_requeues(st) -> None:
    """retry_blocked=True: onay verilmiş bloklu görevi yeniden deneyip tamamlar."""
    executor.register_handler("rules-updater", lambda task: {"ok": True})
    t = task_queue.create_task("rules-updater", "iş", store=st)
    executor.run_task(t, store=st)  # → blocked_approval
    d = approvals.require_fresh_approval("rules-updater", "run", "medium", "s", store=st)
    approvals.approve(d.approval_id, store=st)

    executor.run_pending(limit=50, retry_blocked=True, store=st)
    assert task_queue.get_task(t.task_id, store=st).status.value == "completed"


def test_stop_all_blocks_dangerous_via_executor(st, tmp_path) -> None:
    """STOP_ALL aktifken tehlikeli ajan executor üzerinden de bloklanır.

    root=tmp_path ile izole: gerçek proje köküne STOP_ALL yazılmaz.
    """
    executor.register_handler("auto-lora-pipeline", lambda task: {"ok": True})
    supervisor.create_stop_all("test-dur", root=tmp_path)
    t = task_queue.create_task("auto-lora-pipeline", "eğitim", store=st)
    out = executor.run_task(t, store=st, root=tmp_path)
    assert out.get("blocked") is True
    assert out["blocked_by"] == "stop_all"
    assert task_queue.get_task(t.task_id, store=st).status.value == "blocked_stop_all"


def test_handler_non_dict_return_completes(st) -> None:
    """dict-olmayan dönüş (None/str) başarı sayılır → tamamlanır (yalnız {'ok': False} hata)."""
    executor.register_handler("model-advisor", lambda task: None)
    t = task_queue.create_task("model-advisor", "non-dict", store=st)
    out = executor.run_task(t, store=st)
    assert out["ok"] is True
    assert task_queue.get_task(t.task_id, store=st).status.value == "completed"


def test_retry_blocked_recovers_stop_all_task(st, tmp_path) -> None:
    """retry_blocked, blocked_stop_all görevi de requeue eder (yalnız blocked_approval değil)."""
    executor.register_handler("auto-lora-pipeline", lambda task: {"ok": True})
    supervisor.create_stop_all("dur", root=tmp_path)
    t = task_queue.create_task("auto-lora-pipeline", "iş", store=st)
    executor.run_task(t, store=st, root=tmp_path)  # → blocked_stop_all
    assert task_queue.get_task(t.task_id, store=st).status.value == "blocked_stop_all"

    # STOP_ALL kalk + taze onay → retry tamamlamalı (onay tüketilir, sızıntı yok)
    supervisor.clear_stop_all(root=tmp_path)
    d = approvals.require_fresh_approval("auto-lora-pipeline", "run", "high", "s", store=st)
    approvals.approve(d.approval_id, store=st)
    executor.run_pending(limit=50, retry_blocked=True, store=st, root=tmp_path)
    assert task_queue.get_task(t.task_id, store=st).status.value == "completed"
