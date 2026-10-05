"""Sohbet kaynak koruması (sunucu tarafı) + eğitim başlatmayla yarış.

Kabul şartları: RAM ve VRAM ayrı değerlendirilir; Ollama'daki toplam model boyutu doğrudan
ek RAM ihtiyacı sayılmaz; zaten yüklü model ek yükleme gerektirmez; ölçüm yoksa eğitim
sırasında cevaplama kapalıdır ama eğitim dışında ölçüm yolu vardır (kalıcı kilit yok); koruma
sunucuda uygulanır; kontrol ile başlatma arasındaki yarış iki taraflı kilitle kapatılır.
"""

from __future__ import annotations

import pytest
from tests.chat_learning_helpers import FakeLLM, StubRetriever, iso  # noqa: F401

from app.feedback.chat_store import ChatStore
from app.feedback.resource_guard import evaluate

GB = 1024**3
TRAINING = {"active": True, "starting": False, "source": "detached"}
IDLE = {"active": False, "starting": False, "source": ""}


def _ev(**kw):
    base = {
        "activity": TRAINING,
        "ps_entries": [],
        "ram_available_gb": 40.0,
        "gpu_mem": None,
        "footprint": None,
        "margin_gb": 4.0,
    }
    base.update(kw)
    return evaluate("m", **base)


def test_no_training_allows_without_measurement() -> None:
    assert _ev(activity=IDLE).allowed


def test_training_without_measurement_blocks_with_reason() -> None:
    d = _ev()
    assert not d.allowed and "ölçümü yok" in d.reason and "Kaynak ölç" in d.reason


def test_already_loaded_model_is_allowed() -> None:
    d = _ev(ps_entries=[{"name": "m", "size": 20 * GB, "size_vram": 0}], ram_available_gb=1.0)
    assert d.allowed and "zaten bellekte" in d.reason


def test_total_size_is_not_ram_need() -> None:
    fp = {"size_gb": 18.0, "ram_gb": 10.0, "vram_gb": 8.0}
    # Toplam 18 GB > boş RAM 15 GB ama RAM kısmı 10 + 4 pay = 14 ≤ 15 ve VRAM sığıyor → izin.
    d = _ev(footprint=fp, ram_available_gb=15.0, gpu_mem=(2.0, 12.0))
    assert d.allowed
    assert not _ev(footprint=fp, ram_available_gb=13.0, gpu_mem=(2.0, 12.0)).allowed  # RAM
    assert not _ev(footprint=fp, ram_available_gb=40.0, gpu_mem=(6.0, 12.0)).allowed  # VRAM


def test_vram_unknown_blocks_unless_unified() -> None:
    fp = {"size_gb": 18.0, "ram_gb": 10.0, "vram_gb": 8.0}
    assert not _ev(footprint=fp, gpu_mem=None).allowed
    assert _ev(footprint=fp, gpu_mem=None, unified_memory=True, ram_available_gb=30.0).allowed
    assert not _ev(footprint=fp, gpu_mem=None, unified_memory=True, ram_available_gb=20.0).allowed


def test_starting_and_ollama_unreachable_block() -> None:
    assert not _ev(activity={"active": False, "starting": True}).allowed
    assert not _ev(ps_entries=None).allowed


def test_server_side_guard_blocks_send(iso, monkeypatch) -> None:  # noqa: F811
    from app.feedback import resource_guard
    from app.feedback.chat_service import send

    monkeypatch.setattr(resource_guard, "training_activity", lambda root=None: dict(TRAINING))
    llm = FakeLLM()
    store = ChatStore()
    conv = store.create_conversation()["conversation_id"]
    turn, _ = send(conv, "Soru?", "r1", retriever=StubRetriever(), llm=llm, store=store)
    assert turn["status"] == "blocked" and "Eğitim sürüyor" in turn["status_detail"]
    assert llm.calls == []  # model HİÇ çağrılmadı
    # Geçmiş görüntüleme ve inceleme açık kalır.
    assert store.list_turns(conv)[0]["turn_id"] == turn["turn_id"]


def test_measure_refused_during_training(iso, monkeypatch) -> None:  # noqa: F811
    from app.feedback import resource_guard
    from app.feedback.chat_service import measure

    monkeypatch.setattr(resource_guard, "training_activity", lambda root=None: dict(TRAINING))
    with pytest.raises(RuntimeError, match="eğitim dışında"):
        measure(llm=FakeLLM())


def test_lease_race_both_sides(iso) -> None:  # noqa: F811
    from app.config import get_settings
    from app.feedback.resource_guard import (
        acquire_chat_lease,
        active_chat_leases,
        chat_lease_blocker,
        release_chat_lease,
    )
    from app.training import detached_launch as dl

    root = get_settings().root
    # Eğitim başlatma kilidi alınmışsa sohbet kirası geri çekilir.
    assert dl._acquire_launch_lock(root)
    lease, why = acquire_chat_lease("t1")
    assert lease is None and "Eğitim" in why
    assert active_chat_leases() == []
    dl._release_launch_lock(root)
    # Sohbet kirası varken eğitim başlatma ön-kontrolü (onay tüketilmeden önce) durur.
    lease, _ = acquire_chat_lease("t2")
    assert lease is not None
    assert chat_lease_blocker() is not None
    pre = dl.preflight_launch("hektor_lora_test", run_load_doctor=False)
    assert not pre["ok"] and "Sohbet cevabı üretiliyor" in pre["message"]
    release_chat_lease(lease)
    assert chat_lease_blocker() is None


def test_stale_lease_is_ignored(iso, monkeypatch) -> None:  # noqa: F811
    from app.feedback.resource_guard import acquire_chat_lease, active_chat_leases

    lease, _ = acquire_chat_lease("old")
    assert lease is not None
    assert active_chat_leases(ttl_s=-1) == []  # bayat kira kalıcı engel olmaz
