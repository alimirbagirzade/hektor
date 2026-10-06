"""Faz 2A — model seçimi ve etkinleştirme.

Kabul şartları: ana ve deneme modeli ayrı; 'Kabul' kararı olmayan aday ana model olamaz;
'Yetersiz kanıt' yalnız etiketli deneme sohbetinde; kritik ret hiçbir yuvaya; mevcut aktif
model başlangıç kaydıdır (geriye dönük 'değerlendirildi' sayılmaz, okuma diske yazmaz); geri
dönüşte önceki model + digest doğrulanır; etkinleştirme kaydı ile adapter production kaydı
tutarlı (yarım işlem tamamlanır, iki aktif model bilgisi kalmaz); devam eden cevap başladığı
modelle tamamlanır, değişim sonraki isteklere uygulanır.
"""

from __future__ import annotations

import json

import pytest
from tests.chat_learning_helpers import FakeLLM, StubRetriever, iso  # noqa: F401

from app.evals.candidate_decisions import record_decision
from app.feedback import model_activation as ma
from app.feedback.chat_store import ChatStore

BASE = "base-model"
CAND = "cand-v1"
DIG_BASE = "b" * 64
DIG_CAND = "c" * 64


@pytest.fixture
def ollama(monkeypatch, iso):  # noqa: F811
    """Ollama /api/tags sahtesi — testler etiket→digest eşlemini değiştirebilir."""
    from app.config import settings as settings_mod

    monkeypatch.setenv("HEKTOR_CHAT_MODEL", BASE)
    settings_mod.get_settings.cache_clear()
    models = {BASE: DIG_BASE, CAND: DIG_CAND}
    monkeypatch.setattr(
        ma,
        "tags_provider",
        lambda: [{"name": f"{k}:latest", "digest": f"sha256:{v}"} for k, v in models.items()],
    )
    return models


def _decide(decision: str, *, tag: str = CAND, digest: str = DIG_CAND, adapter_id: str = ""):
    return record_decision(
        {
            "candidate_tag": tag,
            "candidate_digest": digest,
            "decision": decision,
            "comparison_id": "cmp_test",
            "active_tag": BASE,  # karar GÜNCEL ana modele karşı (Kademe 2 F4-1)
            "adapter_id": adapter_id,
        }
    )


def _adapter(status: str = "eval_passed") -> str:
    from app.lora.adapter_registry import AdapterRecord, AdapterRegistry, AdapterStatus

    return AdapterRegistry(ma.registry_path()).register(
        AdapterRecord(adapter_name="hektor_lora_t", status=AdapterStatus(status))
    )


def test_baseline_is_virtual_unevaluated_and_not_written(ollama) -> None:
    st = ma.load_state()
    assert st["virtual"] and st["main"]["tag"] == BASE
    assert st["main"]["evaluation"] == ma.BASELINE_EVAL
    assert "değerlendirildi' sayılmaz" in st["main"]["note"]
    assert not ma.state_path().exists()  # okuma diske yazmaz
    assert ma.resolve_chat_tag("main") == BASE and ma.resolve_chat_tag("trial") == ""


def test_no_decision_blocks_everything(ollama) -> None:
    with pytest.raises(ma.ActivationError, match="karşılaştırma kararı yok"):
        ma.activate_main(CAND, "yeterince uzun gerekçe")
    with pytest.raises(ma.ActivationError):
        ma.activate_trial(CAND)


def test_insufficient_evidence_only_trial(ollama) -> None:
    _decide("yetersiz_kanit")
    with pytest.raises(ma.ActivationError, match="yalnız 'Kabul'"):
        ma.activate_main(CAND, "yeterince uzun gerekçe")
    st = ma.activate_trial(CAND)
    assert st["trial"]["tag"] == CAND and st["main"]["tag"] == BASE
    assert ma.resolve_chat_tag("trial") == CAND and ma.resolve_chat_tag("main") == BASE


@pytest.mark.parametrize("decision", ["ret", "kritik_ret"])
def test_rejected_candidates_never_activate(ollama, decision) -> None:
    _decide(decision)
    with pytest.raises(ma.ActivationError) as main_exc:
        ma.activate_main(CAND, "yeterince uzun gerekçe")
    if decision == "kritik_ret":
        assert "KRİTİK RET" in str(main_exc.value)
    with pytest.raises(ma.ActivationError, match=r"deneme sohbetinde de|KRİTİK RET"):
        ma.activate_trial(CAND)


def test_latest_decision_wins_and_critical_after_accept_blocks(ollama) -> None:
    _decide("kabul")
    _decide("kritik_ret")
    with pytest.raises(ma.ActivationError, match="KRİTİK RET"):
        ma.activate_main(CAND, "yeterince uzun gerekçe")


def test_k2_f4_2_critical_reject_is_sticky(ollama) -> None:
    """Kademe 2 F4-2: kritik retten SONRA yazılan kabul onu kaldırmaz (ana + deneme)."""
    _decide("kritik_ret")
    _decide("kabul")
    with pytest.raises(ma.ActivationError, match="KRİTİK RET"):
        ma.activate_main(CAND, "yeterince uzun gerekçe")
    with pytest.raises(ma.ActivationError, match="KRİTİK RET"):
        ma.activate_trial(CAND)


def test_k2_f4_1_accept_must_be_against_current_main(ollama) -> None:
    """Kademe 2 F4-1: zayıf ya da eski bir modele karşı alınmış kabul ana modelin yerine geçemez."""
    record_decision(
        {
            "candidate_tag": CAND,
            "candidate_digest": DIG_CAND,
            "decision": "kabul",
            "comparison_id": "cmp_x",
            "adapter_id": "",
            "active_tag": "kucuk-zayif-model",
        }
    )
    with pytest.raises(ma.ActivationError, match="şu anki ana model"):
        ma.activate_main(CAND, "yeterince uzun gerekçe")


def test_digest_mismatch_blocks(ollama) -> None:
    _decide("kabul", digest="d" * 64)  # karşılaştırılan model başka ağırlıklar
    with pytest.raises(ma.ActivationError, match="başka bir digest"):
        ma.activate_main(CAND, "yeterince uzun gerekçe")
    ollama.pop(CAND)
    with pytest.raises(ma.ActivationError, match="bulunamadı"):
        ma.activate_trial(CAND)


def test_accept_activates_main_and_syncs_registry(ollama) -> None:
    aid = _adapter()
    _decide("kabul", adapter_id=aid)
    with pytest.raises(ma.ActivationError, match="10 karakter"):
        ma.activate_main(CAND, "kısa")
    st = ma.activate_main(CAND, "Karşılaştırma kabul, inceledim.")
    assert st["main"]["tag"] == CAND and st["main"]["digest"] == DIG_CAND
    prev = st["main"]["previous"]
    assert prev["tag"] == BASE and prev["digest"] == DIG_BASE
    assert prev["evaluation"] == ma.BASELINE_EVAL  # başlangıç kaydı hâlâ değerlendirilmemiş
    assert ma.registry_production_id() == aid
    assert ma.consistency()["consistent"]
    assert not ma.journal_path().exists()
    events = [json.loads(x) for x in ma.log_path().read_text("utf-8").splitlines()]
    assert events[-1]["op"] == "activate_main"


def test_registry_status_checked_before_any_write(ollama) -> None:
    aid = _adapter(status="candidate")
    _decide("kabul", adapter_id=aid)
    with pytest.raises(ma.ActivationError, match="production'a alınamaz"):
        ma.activate_main(CAND, "yeterince uzun gerekçe")
    assert not ma.state_path().exists() and not ma.journal_path().exists()


def test_rollback_checks_presence_and_digest(ollama) -> None:
    aid = _adapter()
    _decide("kabul", adapter_id=aid)
    ma.activate_main(CAND, "Karşılaştırma kabul, inceledim.")
    ollama[BASE] = "e" * 64  # önceki etiket başka ağırlıklarla yeniden oluşturulmuş
    with pytest.raises(ma.ActivationError, match="başka ağırlıklarla"):
        ma.rollback_main("Aday sohbette kötü davrandı.")
    ollama.pop(BASE)
    with pytest.raises(ma.ActivationError, match="bulunamadı"):
        ma.rollback_main("Aday sohbette kötü davrandı.")
    ollama[BASE] = DIG_BASE
    st = ma.rollback_main("Aday sohbette kötü davrandı.")
    assert st["main"]["tag"] == BASE and st["main"]["previous"]["tag"] == CAND
    assert ma.registry_production_id() == ""  # adaptersiz modele dönüldü → production arşivde
    assert ma.consistency()["consistent"]


def test_interrupted_commit_is_completed_on_next_read(ollama, monkeypatch) -> None:
    aid = _adapter()
    _decide("kabul", adapter_id=aid)

    def crash(journal, root=None):
        raise RuntimeError("güç kesildi")

    monkeypatch.setattr(ma, "_apply_journal", crash)
    with pytest.raises(RuntimeError):
        ma.activate_main(CAND, "Karşılaştırma kabul, inceledim.")
    monkeypatch.undo()
    monkeypatch.setattr(
        ma,
        "tags_provider",
        lambda: [{"name": BASE, "digest": DIG_BASE}, {"name": CAND, "digest": DIG_CAND}],
    )
    assert ma.journal_path().exists()
    st = ma.load_state()  # okuma yarım işlemi tamamlar
    assert st["main"]["tag"] == CAND and not ma.journal_path().exists()
    assert ma.registry_production_id() == aid and ma.consistency()["consistent"]


def test_recovery_failure_never_breaks_chat(ollama, monkeypatch) -> None:
    ma._write_atomic(
        ma.journal_path(),
        {
            "op_id": "x",
            "to_state": {"version": 1, "main": {"tag": CAND, "adapter_id": "adapter_yok"}},
            "event": {"op": "t"},
        },
    )
    st = ma.load_state()
    assert "tamamlanamadı" in st["recovery_error"]
    assert ma.resolve_chat_tag("main") == BASE  # eski (tutarlı) durum kullanılır
    _decide("kabul")
    with pytest.raises(ma.ActivationError, match="tamamlanamadı"):
        ma.activate_main(CAND, "yeterince uzun gerekçe")


# ── sohbet entegrasyonu ──────────────────────────────────────────────────────


class SwitchingLLM(FakeLLM):
    """Cevap üretilirken ana modeli değiştirir (eşzamanlı etkinleştirme benzetimi)."""

    def generate(self, prompt, **kw):
        ma.activate_main(CAND, "Cevap sürerken etkinleştirildi.")
        return super().generate(prompt, **kw)


def test_inflight_answer_keeps_start_model(ollama) -> None:
    from app.feedback.chat_service import send

    _decide("kabul")
    store = ChatStore()
    conv = store.create_conversation()["conversation_id"]
    t1, _ = send(
        conv, "Birinci soru?", "a1", retriever=StubRetriever(), llm=SwitchingLLM(), store=store
    )
    assert t1["status"] == "answered" and t1["model_tag"] == BASE  # başladığı modelle bitti
    assert ma.resolve_chat_tag("main") == CAND
    t2, _ = send(conv, "İkinci soru?", "a2", retriever=StubRetriever(), llm=FakeLLM(), store=store)
    assert t2["model_tag"] == CAND  # değişim sonraki isteğe uygulandı
    assert t2["model_info"]["setting_source"] == "etkinleştirme kaydı"


def test_trial_conversation_is_separate_and_labeled(ollama) -> None:
    from app.feedback.chat_service import send
    from app.feedback.learning import LearningError, LearningService

    store = ChatStore()
    trial = store.create_conversation(slot="trial")["conversation_id"]
    blocked, _ = send(
        trial, "Deneme sorusu?", "t1", retriever=StubRetriever(), llm=FakeLLM(), store=store
    )
    assert (
        blocked["status"] == "blocked"
        and "Deneme yuvasında etkin model yok" in (blocked["status_detail"])
    )
    _decide("yetersiz_kanit")
    ma.activate_trial(CAND)
    t, _ = send(
        trial, "Deneme sorusu iki?", "t2", retriever=StubRetriever(), llm=FakeLLM(), store=store
    )
    assert t["model_tag"] == CAND and t["model_info"]["slot"] == "trial"
    assert t["model_info"]["slot_decision"] == "yetersiz_kanit"
    main = store.create_conversation()["conversation_id"]
    m, _ = send(main, "Ana soru?", "m1", retriever=StubRetriever(), llm=FakeLLM(), store=store)
    assert m["model_tag"] == BASE  # deneme modeli ana sohbete cevap vermez
    svc = LearningService(store)
    with pytest.raises(LearningError, match="Deneme sohbetindeki"):
        svc.learn(t["turn_id"])
    cand, _ = svc.correct(t["turn_id"], "İnsanın yazdığı doğru metin burada duruyor.")
    assert cand["kind"] == "correct"


def test_old_database_gets_new_columns(iso) -> None:  # noqa: F811
    import sqlite3

    from app.config import get_settings

    db = get_settings().sqlite_file
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE chat_conversations (conversation_id VARCHAR(40) PRIMARY KEY, "
        "title TEXT, created_at VARCHAR(40), updated_at VARCHAR(40))"
    )
    con.execute("INSERT INTO chat_conversations VALUES ('conv_eski', 'eski', 'x', 'x')")
    con.commit()
    con.close()
    store = ChatStore()
    assert store.get_conversation("conv_eski")["slot"] == "main"


def test_web_activation_endpoints(ollama) -> None:
    from fastapi.testclient import TestClient

    from app.web.security import require_human
    from app.web.server import app

    client = TestClient(app)
    ov = client.get("/api/models/activation").json()
    assert ov["main"]["evaluation"] == ma.BASELINE_EVAL and ov["trial"] is None
    assert not ma.state_path().exists()  # GET yazmaz
    _decide("yetersiz_kanit")
    ov = client.get("/api/models/activation").json()
    c = ov["candidates"][0]
    assert c["trial_allowed"] and not c["main_allowed"]
    bad = client.post("/api/models/activate-main", json={"tag": CAND, "reason": "uzun gerekçe x"})
    assert bad.status_code == 422
    ok = client.post("/api/models/activate-trial", json={"tag": CAND})
    assert ok.status_code == 200 and ok.json()["trial"]["tag"] == CAND
    conv = client.post("/api/chat/conversations", json={"slot": "trial"}).json()
    assert conv["slot"] == "trial"
    m = client.get("/api/chat/model?slot=trial").json()
    assert m["tag"] == CAND and "DENEME" in m["slot_info"]["banner"]
    human = {
        "/api/models/activate-main",
        "/api/models/activate-trial",
        "/api/models/clear-trial",
        "/api/models/rollback",
        "/api/models/repair-registry",
    }
    for rt in client.app.routes:
        if getattr(rt, "path", "") in human:
            assert require_human in {d.call for d in rt.dependant.dependencies}


# ── Kademe 2 (2026-10-06) F4-8: geri dönüşte karar deposu yeniden denetlenir ──────────


def test_f4_8_rollback_rechecks_decision_of_previous_model(ollama) -> None:
    cand2, dig2 = "cand-v2", "d" * 64
    ollama[cand2] = dig2
    _decide("kabul", adapter_id=_adapter())
    ma.activate_main(CAND, "Karşılaştırma kabul, inceledim.")
    record_decision(
        {
            "candidate_tag": cand2,
            "candidate_digest": dig2,
            "decision": "kabul",
            "comparison_id": "cmp_test2",
            "active_tag": CAND,
            "adapter_id": "",
        }
    )
    ma.activate_main(cand2, "İkinci aday da kabul aldı.")
    _decide("ret")  # CAND sonradan yeniden karşılaştırıldı → artık 'Ret'
    with pytest.raises(ma.ActivationError, match="güncel karar 'Ret'"):
        ma.rollback_main("Aday sohbette kötü davrandı.")
    assert ma.load_state()["main"]["tag"] == cand2  # hiçbir şey yazılmadı
    _decide("kritik_ret")
    with pytest.raises(ma.ActivationError, match="KRİTİK RET"):
        ma.rollback_main("Aday sohbette kötü davrandı.")


def test_f4_8_rollback_to_baseline_blocked_only_by_critical_reject(ollama) -> None:
    _decide("kabul", adapter_id=_adapter())
    ma.activate_main(CAND, "Karşılaştırma kabul, inceledim.")
    _decide("ret", tag=BASE, digest=DIG_BASE)  # baseline için sıradan ret engel değil
    _decide("kritik_ret", tag=BASE, digest=DIG_BASE)
    with pytest.raises(ma.ActivationError, match="KRİTİK RET"):
        ma.rollback_main("Aday sohbette kötü davrandı.")
