"""Orkestrasyon çekirdeği — çevrimdışı testler (sahte delege + tmp SQLite).

Gerçek Ollama/torch/eğitim YOK. Durum makinesi, checkpoint/resume, panic-recovery
ve Kural-8 onay sınırı doğrulanır.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from app.orchestration.orchestrator import RunContext, StageResult, TrainingOrchestrator
from app.orchestration.pipeline import PIPELINE, RunStatus, StageStatus
from app.orchestration.store import OrchestrationStore


@pytest.fixture
def store(tmp_path: Path) -> OrchestrationStore:
    return OrchestrationStore(db_path=tmp_path / "orc_test.db")


def _all_complete_delegates(calls: list[str]) -> dict:
    def make(name: str):
        def d(ctx: RunContext) -> StageResult:
            calls.append(name)
            return StageResult(StageStatus.completed, f"{name} ok", {"stage": name})

        return d

    return {s.name: make(s.name) for s in PIPELINE}


# ── store ─────────────────────────────────────────────────────────────────────


def test_create_run_creates_all_stages(store: OrchestrationStore) -> None:
    run_id = store.create_run(
        model="qwen2.5:1.5b", profile="discipline_safe_local", adapter_name="a1"
    )
    run = store.get_run(run_id)
    assert run is not None
    assert run["status"] == RunStatus.pending.value
    stages = store.get_stages(run_id)
    assert [s["name"] for s in stages] == [s.name for s in PIPELINE]
    assert all(s["status"] == StageStatus.pending.value for s in stages)
    # sıralı
    assert [s["order"] for s in stages] == sorted(s["order"] for s in stages)


def test_events_and_params_roundtrip(store: OrchestrationStore) -> None:
    run_id = store.create_run(
        model="m", profile="p", adapter_name="a", params={"iters": 300, "hunt_ack": True}
    )
    assert store.get_run(run_id)["params"] == {"iters": 300, "hunt_ack": True}
    store.add_event(run_id, "preflight", "info", "merhaba")
    events = store.get_events(run_id)
    # create_run de bir olay yazar → en az 2
    assert any(e["message"] == "merhaba" for e in events)


# ── happy path ──────────────────────────────────────────────────────────────


def test_happy_path_completes_all_stages(store: OrchestrationStore) -> None:
    calls: list[str] = []
    orch = TrainingOrchestrator(store=store, delegates=_all_complete_delegates(calls))
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    snap = orch.run_until_blocked(run_id)
    assert snap["run"]["status"] == RunStatus.completed.value
    assert calls == [s.name for s in PIPELINE]  # her aşama tam bir kez
    assert all(s["status"] == StageStatus.completed.value for s in snap["stages"])


def test_step_advances_exactly_one_stage(store: OrchestrationStore) -> None:
    calls: list[str] = []
    orch = TrainingOrchestrator(store=store, delegates=_all_complete_delegates(calls))
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    orch.step(run_id)
    assert calls == [PIPELINE[0].name]
    stages = store.get_stages(run_id)
    assert stages[0]["status"] == StageStatus.completed.value
    assert stages[1]["status"] == StageStatus.pending.value


# ── blocked + resume ──────────────────────────────────────────────────────────


def test_blocked_stage_halts_then_resumes(store: OrchestrationStore) -> None:
    calls: list[str] = []
    delegates = _all_complete_delegates(calls)

    def block_hunt(ctx: RunContext) -> StageResult:
        calls.append("deep-hunt")
        return StageResult(StageStatus.blocked, "hunt_ack yok", {})

    delegates["deep-hunt"] = block_hunt
    orch = TrainingOrchestrator(store=store, delegates=delegates)
    run_id = orch.start(model="m", profile="p", adapter_name="a")

    snap = orch.run_until_blocked(run_id)
    assert snap["run"]["status"] == RunStatus.blocked.value
    by_name = {s["name"]: s for s in snap["stages"]}
    assert by_name["preflight"]["status"] == StageStatus.completed.value
    assert by_name["deep-hunt"]["status"] == StageStatus.blocked.value
    assert by_name["data-gate"]["status"] == StageStatus.pending.value

    # onay geldi → deep-hunt artık tamamlanıyor; resume tamamlananı TEKRAR ÇALIŞTIRMAZ
    preflight_calls_before = calls.count("preflight")

    def pass_hunt(ctx: RunContext) -> StageResult:
        calls.append("deep-hunt")
        return StageResult(StageStatus.completed, "onaylandı", {})

    orch._delegates["deep-hunt"] = pass_hunt
    snap2 = orch.run_until_blocked(run_id)
    assert snap2["run"]["status"] == RunStatus.completed.value
    assert calls.count("preflight") == preflight_calls_before  # checkpoint: yeniden koşmadı


def test_failed_delegate_marks_run_failed(store: OrchestrationStore) -> None:
    calls: list[str] = []
    delegates = _all_complete_delegates(calls)

    def fail_data(ctx: RunContext) -> StageResult:
        return StageResult(StageStatus.failed, "veri yok", {})

    delegates["data-gate"] = fail_data
    orch = TrainingOrchestrator(store=store, delegates=delegates)
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    snap = orch.run_until_blocked(run_id)
    assert snap["run"]["status"] == RunStatus.failed.value
    assert "veri yok" in (snap["run"]["error"] or "")


def test_delegate_exception_is_caught_not_crash(store: OrchestrationStore) -> None:
    def boom(ctx: RunContext) -> StageResult:
        raise RuntimeError("patladı")

    orch = TrainingOrchestrator(store=store, delegates={"preflight": boom})
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    snap = orch.step(run_id)  # exception yutulmalı
    assert snap["run"]["status"] == RunStatus.failed.value
    by_name = {s["name"]: s for s in snap["stages"]}
    assert by_name["preflight"]["status"] == StageStatus.failed.value
    assert "patladı" in by_name["preflight"]["message"]


# ── panic recovery ────────────────────────────────────────────────────────────


def test_panic_recovery_marks_stale_running_failed(store: OrchestrationStore) -> None:
    orch = TrainingOrchestrator(store=store, delegates={})
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    old = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=120)).isoformat()
    store.update_stage(run_id, "preflight", status=StageStatus.running.value, heartbeat_at=old)

    recovered = orch.recover_stale(timeout_min=30.0)
    assert {"run_id": run_id, "stage": "preflight"} in recovered
    assert store.get_run(run_id)["status"] == RunStatus.failed.value
    assert store.get_stage(run_id, "preflight")["status"] == StageStatus.failed.value


def test_fresh_running_stage_not_recovered(store: OrchestrationStore) -> None:
    orch = TrainingOrchestrator(store=store, delegates={})
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    fresh = dt.datetime.now(dt.UTC).isoformat()
    store.update_stage(run_id, "preflight", status=StageStatus.running.value, heartbeat_at=fresh)
    assert orch.recover_stale(timeout_min=30.0) == []


def test_cancel_marks_run_cancelled_and_blocks_steps(store: OrchestrationStore) -> None:
    calls: list[str] = []
    orch = TrainingOrchestrator(store=store, delegates=_all_complete_delegates(calls))
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    orch.cancel(run_id, reason="elle iptal")
    snap = orch.step(run_id)  # terminal → ilerlemez
    assert snap["run"]["status"] == RunStatus.cancelled.value
    assert calls == []


# ── gerçek varsayılan delege bağlama (offline) ────────────────────────────────


def test_non_serializable_output_does_not_crash(store: OrchestrationStore) -> None:
    """Delege JSON-dışı çıktı (datetime) dönse bile step() çökmez (default=str)."""

    def weird(ctx: RunContext) -> StageResult:
        return StageResult(StageStatus.completed, "ok", {"ts": dt.datetime(2026, 1, 1)})

    orch = TrainingOrchestrator(store=store, delegates={"preflight": weird})
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    snap = orch.step(run_id)
    by_name = {s["name"]: s for s in snap["stages"]}
    assert by_name["preflight"]["status"] == StageStatus.completed.value
    assert "ts" in by_name["preflight"]["output"]  # serileşti (str olarak)


def test_max_steps_truncation_leaves_resumable_run(store: OrchestrationStore) -> None:
    """max_steps tükenirse koşu 'running' kalır ama resume edilebilir (note ile işaretli)."""
    calls: list[str] = []
    orch = TrainingOrchestrator(store=store, delegates=_all_complete_delegates(calls))
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    snap = orch.run_until_blocked(run_id, max_steps=2)  # 10 aşama, 2 adım → tükenir
    assert snap["run"]["status"] == RunStatus.running.value
    assert snap["note"] == "max_steps reached"
    # resume tamamlar
    final = orch.run_until_blocked(run_id, max_steps=50)
    assert final["run"]["status"] == RunStatus.completed.value


# ── eşzamanlılık sertleştirme (Kademe-2 av) ──────────────────────────────────


def test_claim_stage_running_is_atomic_cas(store: OrchestrationStore) -> None:
    """claim_stage_running yalnız pending/blocked/failed aşamayı 'running'e alır (CAS)."""
    run_id = store.create_run(model="m", profile="p", adapter_name="a")
    assert store.claim_stage_running(run_id, "preflight") is True
    assert store.get_stage(run_id, "preflight")["status"] == StageStatus.running.value
    # zaten 'running' → ikinci claim başarısız (başka eşzamanlı step kapamaz)
    assert store.claim_stage_running(run_id, "preflight") is False


def test_step_noop_when_stage_already_claimed(store: OrchestrationStore) -> None:
    """Aşama başka bir 'thread'ce kapılmışsa step() delegeyi ÇALIŞTIRMAZ (çift-delege yok)."""
    calls: list[str] = []
    orch = TrainingOrchestrator(store=store, delegates=_all_complete_delegates(calls))
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    store.claim_stage_running(run_id, "preflight")  # eşzamanlı step simülasyonu
    snap = orch.step(run_id)
    assert snap["note"] == "stage already claimed"
    assert calls == []


def test_step_does_not_clobber_concurrent_cancel(store: OrchestrationStore) -> None:
    """Delege sürerken araya giren cancel, step()'in sonuç yazımıyla clobber edilmez."""

    def cancel_mid(ctx: RunContext) -> StageResult:
        ctx.store.update_run(ctx.run_id, status=RunStatus.cancelled.value, error="araya iptal")
        return StageResult(StageStatus.completed, "ok", {})

    orch = TrainingOrchestrator(store=store, delegates={"preflight": cancel_mid})
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    snap = orch.step(run_id)
    assert snap["run"]["status"] == RunStatus.cancelled.value  # 'completed' ile ezilmedi
    assert snap["note"] == "run finalized during stage"
    # kapılan aşama zombi 'running' kalmaz → skipped olarak temizlenir
    assert store.get_stage(run_id, "preflight")["status"] == StageStatus.skipped.value


def test_recover_stale_does_not_clobber_cancelled_run(store: OrchestrationStore) -> None:
    """cancelled koşunun asılı aşaması recover_stale tarafından 'failed'a clobber EDİLMEZ."""
    orch = TrainingOrchestrator(store=store, delegates={})
    run_id = orch.start(model="m", profile="p", adapter_name="a")
    old = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=120)).isoformat()
    store.update_stage(run_id, "preflight", status=StageStatus.running.value, heartbeat_at=old)
    orch.cancel(run_id, reason="elle iptal")
    # cancel çalışan aşamayı terminalize etti → recover hiçbir şey bulmamalı
    recovered = orch.recover_stale(timeout_min=30.0)
    assert all(r["run_id"] != run_id for r in recovered)
    assert store.get_run(run_id)["status"] == RunStatus.cancelled.value  # failed DEĞİL


def test_approval_delegate_uses_unattended_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gözetimsiz eğitim AÇIKKEN, gate zinciri geçen koşu tek politikadan yetki alır.

    Ayar varsayılan KAPALIDIR (her eğitim taze insan onayı ister); bu test onu
    bilinçli açar → testin sonucu varsayılana bağlı kalmaz.
    """
    import app.agents.runtime.supervisor as supervisor_mod
    from app.config import settings as settings_mod
    from app.orchestration import delegates

    monkeypatch.setattr(supervisor_mod, "is_stop_all_active", lambda *a, **k: False)
    monkeypatch.setenv("ACHILLES_UNATTENDED_TRAINING_ENABLED", "true")
    settings_mod.get_settings.cache_clear()

    ctx = RunContext(run_id="r", stage="approval", run={"adapter_name": "a"}, params={}, store=None)  # type: ignore[arg-type]
    res = delegates.approval(ctx)
    assert res.status == StageStatus.completed
    assert res.output["authorization_mode"] == "unattended_policy"


def test_approval_delegate_requires_human_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Varsayılan (gözetimsiz KAPALI): taze onay yoksa approval aşaması BLOKLANIR."""
    import app.agents.runtime.supervisor as supervisor_mod
    from app.orchestration import delegates

    monkeypatch.setattr(supervisor_mod, "is_stop_all_active", lambda *a, **k: False)

    ctx = RunContext(run_id="r", stage="approval", run={"adapter_name": "a"}, params={}, store=None)  # type: ignore[arg-type]
    res = delegates.approval(ctx)
    assert res.status == StageStatus.blocked


def test_default_delegates_halt_at_human_gate_offline(
    store: OrchestrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gerçek varsayılan delegelerle çevrimdışı: import temiz + insan-kapısında durur.

    preflight gerçek çalışır; smoke runtime erişilemez → skip (hat durmaz); deep-hunt
    (hunt_ack yok) blocked olur → gerçek eğitim ASLA gözetimsiz başlamaz (Kural 8).
    Runtime'ı erişilemez gösteririz ki test çevrimdışı + deterministik kalsın (smoke ağ vurmaz).
    """
    import app.brain.local_llm as llm_mod

    monkeypatch.setattr(llm_mod.LocalLLM, "active_backend", lambda self: "none")
    monkeypatch.setattr(llm_mod.LocalLLM, "available", lambda self: False)

    orch = TrainingOrchestrator(store=store)  # default_delegates()
    run_id = orch.start(model="qwen2.5:1.5b", profile="discipline_safe_local", adapter_name="a")
    snap = orch.run_until_blocked(run_id)
    assert snap["run"]["status"] in {RunStatus.blocked.value, RunStatus.failed.value}
    by_name = {s["name"]: s for s in snap["stages"]}
    # smoke runtime yokken atlanır (kusur değil, "şimdi test edilemez")
    assert by_name["smoke"]["status"] == StageStatus.skipped.value
    # train aşaması ASLA completed olmamalı (gözetimsiz eğitim yok)
    assert by_name["train"]["status"] != StageStatus.completed.value
    assert by_name["evaluate"]["status"] != StageStatus.completed.value
