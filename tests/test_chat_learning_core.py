"""Sohbetten öğrenme Faz 1 — sohbet geçmişi, model kimliği, geri bildirim, aday doğrulama.

Kabul şartları (kullanıcı, 2026-10-05): geçmiş saklanır ve aktarılan bölüm görünür; önceki
cevaplar kanıt değildir; aynı istek çift tur/aday üretmez; her cevap gerçekten kullandığı
model kimliğiyle kaydedilir; Faydalı/Hatalı aday üretmez; tek kontrol tüm cevabı doğrulamaz;
kontrol edilecek iddia bulunamaması başarı değildir; atıf kimliği geçerliliği destekle
karışmaz; insan onayı çürütülmüş ifadeyi geçerli kılamaz.
"""

from __future__ import annotations

import httpx
import pytest
from tests.chat_learning_helpers import (
    SUPPORTED_SENTENCE,
    FakeLLM,
    StubRetriever,
    chunk,
    iso,  # noqa: F401  (fixture)
    send,
)

from app.brain.rag_answerer import HISTORY_HEADER, build_rag_prompt
from app.feedback.chat_store import ChatStore


@pytest.fixture
def store(iso):  # noqa: F811
    return ChatStore()


@pytest.fixture
def conv(store):
    return store.create_conversation()["conversation_id"]


def _svc(store):
    from app.feedback.learning import LearningService

    return LearningService(store)


# ── istem + geçmiş ───────────────────────────────────────────────────────────


def test_build_rag_prompt_history_none_is_byte_identical(iso) -> None:  # noqa: F811
    chunks = [chunk()]
    base_sys, base_user = build_rag_prompt("Soru nedir?", chunks)
    for hist in (None, []):
        sys2, user2 = build_rag_prompt("Soru nedir?", chunks, history=hist)
        assert (sys2, user2) == (base_sys, base_user)
    assert base_user.startswith("SOURCES / KAYNAKLAR:\n")
    _s, with_hist = build_rag_prompt("Soru?", chunks, history=[(1, "Önceki soru", "Önceki cevap")])
    assert with_hist.startswith(HISTORY_HEADER)
    assert with_hist.index("Önceki cevap") < with_hist.index("SOURCES / KAYNAKLAR:")
    assert with_hist.endswith(base_user.replace("Soru nedir?", "Soru?"))


def test_history_really_transferred_and_visible(store, conv) -> None:
    llm = FakeLLM(["Birinci cevap metni burada.", "İkinci cevap metni burada."])
    ret = StubRetriever()
    t1 = send(store, conv, "Volatilite kümelenmesi nedir?", llm=llm, retriever=ret)
    t2 = send(store, conv, "Bunu biraz açar mısın?", llm=llm, retriever=ret)
    second_prompt = llm.calls[1]["prompt"]
    assert HISTORY_HEADER in second_prompt
    assert "Volatilite kümelenmesi nedir?" in second_prompt
    assert "Birinci cevap metni burada." in second_prompt
    assert t2["history_turn_ids"] == [t1["turn_id"]]
    hist = next(c for c in t2["checks"] if c["kind"] == "gecmis")
    assert hist["turn_indexes"] == [1]
    # Takip sorusunun retrieval sorgusu önceki kullanıcı sorusunu da içerir.
    assert "Volatilite kümelenmesi nedir?" in ret.queries[1]
    assert t2["user_prompt"] == second_prompt  # gönderilen istemin TAMAMI saklanır


def test_history_limited_but_full_history_kept(store, conv, monkeypatch) -> None:
    from app.config import settings as settings_mod

    monkeypatch.setenv("HEKTOR_CHAT_HISTORY_TURNS", "2")
    settings_mod.get_settings.cache_clear()
    llm = FakeLLM()
    for i in range(1, 5):
        send(store, conv, f"Soru numarası {i} hakkında ne biliyoruz?", llm=llm)
    last = llm.calls[-1]["prompt"]
    assert "Soru numarası 1 hakkında" not in last
    assert "Soru numarası 2 hakkında" in last and "Soru numarası 3 hakkında" in last
    assert [t["turn_index"] for t in store.list_turns(conv)] == [1, 2, 3, 4]


def test_previous_answer_is_not_evidence(store, conv) -> None:
    """Önceki cevapta geçen ama turun kaynaklarında olmayan iddia desteklenmiş SAYILMAZ."""
    claim = "Kuantum tavlama algoritmaları portföy optimizasyonunda kesinlikle üstün sonuç verir."
    llm = FakeLLM([claim, f"{SUPPORTED_SENTENCE} [p1:c1]"])
    send(store, conv, "Kuantum tavlama ne işe yarar?", llm=llm)
    t2 = send(store, conv, "Peki volatilite kümelenmesi?", llm=llm)
    assert claim in t2["user_prompt"]  # geçmişte var
    cand, _ = _svc(store).correct(t2["turn_id"], claim)
    kaynak = next(c for c in cand["verification"]["checks"] if c["kind"] == "kaynak")
    assert kaynak["status"] != "gecti"
    assert cand["status"] != "eligible"


# ── idempotentlik + model kimliği ────────────────────────────────────────────


def test_same_request_does_not_create_second_turn(store, conv) -> None:
    from app.feedback.chat_service import send as raw_send

    llm = FakeLLM()
    a, replay_a = raw_send(
        conv, "Aynı soru?", "req-1", retriever=StubRetriever(), llm=llm, store=store
    )
    b, replay_b = raw_send(
        conv, "Aynı soru?", "req-1", retriever=StubRetriever(), llm=llm, store=store
    )
    assert (replay_a, replay_b) == (False, True)
    assert a["turn_id"] == b["turn_id"]
    assert len(llm.calls) == 1
    assert len(store.list_turns(conv)) == 1


def test_pending_duplicate_is_busy(store, conv) -> None:
    from app.feedback.chat_service import ChatBusy
    from app.feedback.chat_service import send as raw_send

    store.reserve_turn(conv, "req-x", "Soru?")
    with pytest.raises(ChatBusy):
        raw_send(conv, "Soru?", "req-x", retriever=StubRetriever(), llm=FakeLLM(), store=store)


def test_model_identity_recorded_per_turn(store, conv, monkeypatch) -> None:
    from app.config import settings as settings_mod

    def handler(req: httpx.Request) -> httpx.Response:
        tag = settings_mod.get_settings().effective_chat_model
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": tag, "digest": "sha-before"}]})
        if req.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": tag, "digest": "sha-after", "size": 2**31, "size_vram": 2**30}
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    from app.feedback.chat_service import send as raw_send

    monkeypatch.setenv("HEKTOR_CHAT_MODEL", "model-a")
    settings_mod.get_settings.cache_clear()
    t1, _ = raw_send(
        conv,
        "Birinci soru?",
        "q1",
        retriever=StubRetriever(),
        llm=FakeLLM(),
        transport=transport,
        store=store,
    )
    monkeypatch.setenv("HEKTOR_CHAT_MODEL", "model-b")
    settings_mod.get_settings.cache_clear()
    t2, _ = raw_send(
        conv,
        "İkinci soru?",
        "q2",
        retriever=StubRetriever(),
        llm=FakeLLM(),
        transport=transport,
        store=store,
    )
    assert (t1["model_tag"], t2["model_tag"]) == ("model-a", "model-b")
    assert t1["model_digest"] == "sha-after"
    assert "değişti" in t1["model_info"]["digest_note"]
    # Eğitim yokken cevap sonrası bellek ayak izi ÖLÇÜLÜR (RAM = toplam − VRAM).
    fp = t1["model_info"]["footprint"]
    assert fp["vram_gb"] == 1.0 and fp["ram_gb"] == 1.0


def test_chat_model_defaults_to_llm_model(iso) -> None:  # noqa: F811
    from app.config import get_settings

    s = get_settings()
    assert s.chat_model == "" and s.effective_chat_model == s.llm_model


# ── düğmeler ─────────────────────────────────────────────────────────────────


def test_useful_and_wrong_do_not_create_candidates(store, conv) -> None:
    t = send(store, conv, "Volatilite kümelenmesi nedir?")
    svc = _svc(store)
    svc.set_feedback(t["turn_id"], "useful")
    svc.set_feedback(t["turn_id"], "wrong", "eksik", [{"start": 0, "end": 5}])
    assert store.list_candidates() == []
    q = svc.error_queue()
    assert len(q) == 1 and q[0]["has_correction"] is False
    assert q[0]["flagged_spans"] == [{"start": 0, "end": 5, "note": ""}]


def test_learn_single_click_and_idempotent(store, conv) -> None:
    t = send(store, conv, "Volatilite kümelenmesi nedir?")
    svc = _svc(store)
    c1, created1 = svc.learn(t["turn_id"])
    c2, created2 = svc.learn(t["turn_id"])
    assert (created1, created2) == (True, False)
    assert c1["candidate_id"] == c2["candidate_id"]
    assert c1["kind"] == "learn" and c1["target_text"].startswith(SUPPORTED_SENTENCE)
    assert len(store.list_candidates()) == 1


def test_correct_updates_single_candidate(store, conv) -> None:
    t = send(store, conv, "Volatilite kümelenmesi nedir?")
    svc = _svc(store)
    a, created = svc.correct(t["turn_id"], "İlk düzeltme metni, yeterince uzun bir cümle.")
    b, created2 = svc.correct(t["turn_id"], "İkinci düzeltme metni, yine yeterince uzun bir cümle.")
    assert created and not created2
    assert a["candidate_id"] == b["candidate_id"] and b["revision"] == 2
    again, _ = svc.correct(t["turn_id"], "İkinci düzeltme metni, yine yeterince uzun bir cümle.")
    assert again["revision"] == 2  # aynı içerik → değişiklik yok


def test_wrong_then_correct_pairs_error_queue(store, conv) -> None:
    t = send(store, conv, "Volatilite kümelenmesi nedir?")
    svc = _svc(store)
    svc.set_feedback(t["turn_id"], "wrong")
    svc.correct(t["turn_id"], SUPPORTED_SENTENCE)
    assert svc.error_queue()[0]["has_correction"] is True


def test_no_prompt_turn_cannot_become_candidate(store, conv) -> None:
    from app.feedback.learning import LearningError

    t = send(store, conv, "Hiç kaynağı olmayan soru?", retriever=StubRetriever([]))
    assert t["status"] == "no_llm"
    with pytest.raises(LearningError):
        _svc(store).learn(t["turn_id"])


# ── doğrulama kapsamı ────────────────────────────────────────────────────────


def _checks(c):
    return {k["kind"]: k for k in c["verification"]["checks"]}


def test_fully_supported_is_auto_eligible(store, conv) -> None:
    t = send(store, conv, "Volatilite kümelenmesi nedir?")
    c, _ = _svc(store).correct(t["turn_id"], SUPPORTED_SENTENCE)
    assert c["status"] == "eligible"
    assert c["verification"]["class"] == "auto"
    assert _checks(c)["kaynak"]["status"] == "gecti"


def test_correct_math_does_not_verify_whole_answer(store, conv) -> None:
    t = send(store, conv, "Kelly oranı nasıl hesaplanır?")
    text = (
        "Önce 0.6 - 0.4 / 1 = 0.2 bulunur. Bu yüzden her piyasa koşulunda pozisyonu "
        "sermayenin beşte biri yapmak en doğru yaklaşımdır."
    )
    c, _ = _svc(store).correct(t["turn_id"], text)
    ch = _checks(c)
    assert ch["hesap"]["status"] == "gecti" and ch["hesap"]["scope"] == "1/1 ifade"
    assert ch["kaynak"]["status"] != "gecti"
    assert c["status"] == "review"
    assert c["verification"]["coverage"]["uncovered"]


def test_wrong_math_rejected_and_human_cannot_override(store, conv) -> None:
    from app.feedback.learning import LearningError

    t = send(store, conv, "İki artı iki kaçtır?")
    c, _ = _svc(store).correct(t["turn_id"], "2 + 2 = 5")
    assert c["status"] == "rejected" and "hesap" in c["reason_codes"]
    with pytest.raises(LearningError):
        _svc(store).approve(c["candidate_id"], "Bence bu doğru, onaylıyorum.")
    fixed = _svc(store).edit(c["candidate_id"], "2 + 2 = 4")
    assert fixed["status"] == "eligible"  # düzeltilince geçer (saf hesap birimi kapsandı)


def test_nothing_checkable_is_not_success(store, conv) -> None:
    t = send(store, conv, "Volatilite kümelenmesi nedir?")
    c, _ = _svc(store).correct(t["turn_id"], "Evet, öyle.")
    assert c["status"] == "review"
    assert "kapsam_yok" in c["reason_codes"]
    assert "doğrulama başarısı değildir" in c["status_reason"]


def test_citation_id_validity_is_not_support(store, conv) -> None:
    t = send(store, conv, "Volatilite kümelenmesi nedir?")
    svc = _svc(store)
    valid_id, _ = svc.correct(
        t["turn_id"], "Merkez bankaları faiz kararlarını yalnız enflasyona bakarak verir [p1:c1]."
    )
    ch = _checks(valid_id)
    assert ch["atif_kimligi"]["status"] == "gecti"
    assert ch["kaynak"]["status"] != "gecti"
    assert valid_id["status"] == "review"
    fake_id = svc.edit(valid_id["candidate_id"], f"{SUPPORTED_SENTENCE} [p9:c9]")
    assert _checks(fake_id)["atif_kimligi"]["status"] == "kaldi"
    assert fake_id["status"] == "rejected"


def test_human_approval_is_separate_and_dropped_on_edit(store, conv) -> None:
    t = send(store, conv, "Kavramsal soru: risk nedir?")
    svc = _svc(store)
    c, _ = svc.correct(t["turn_id"], "Risk, beklenen sonuçtan sapma olasılığının ölçüsüdür.")
    assert c["status"] == "review"
    from app.feedback.learning import LearningError

    with pytest.raises(LearningError):
        svc.approve(c["candidate_id"], "kısa")
    ok = svc.approve(c["candidate_id"], "Ders kitabı tanımıyla birebir uyumlu.")
    assert ok["status"] == "eligible" and ok["verification"]["class"] == "human"
    assert ok["human_approval"]["reason"].startswith("Ders kitabı")
    edited = svc.edit(c["candidate_id"], "Risk, kayıp olasılığıdır; tek ölçüsü yoktur.")
    assert edited["human_approval"] == {} and edited["status"] == "review"


def test_trading_performance_claim_needs_backtest(store, conv) -> None:
    from app.feedback.learning import LearningError

    t = send(store, conv, "RSI stratejisi işe yarar mı?")
    c, _ = _svc(store).correct(t["turn_id"], "Bu RSI stratejisi geçmiş veride %35 getiri sağladı.")
    assert c["domain"] == "trading"
    assert _checks(c)["backtest"]["status"] == "yapilamadi"
    assert c["status"] == "review"
    with pytest.raises(LearningError):
        _svc(store).approve(c["candidate_id"], "Grafikte gördüm, doğru görünüyor.")


def test_guarantee_language_rejected(store, conv) -> None:
    t = send(store, conv, "Strateji nasıl?")
    c, _ = _svc(store).correct(t["turn_id"], "Bu strateji garantili kâr sağlar, asla kaybetmezsin.")
    assert c["status"] == "rejected" and "guvenlik" in c["reason_codes"]


def test_exclude_and_undo(store, conv) -> None:
    t = send(store, conv, "Volatilite kümelenmesi nedir?")
    svc = _svc(store)
    c, _ = svc.correct(t["turn_id"], SUPPORTED_SENTENCE)
    assert c["status"] == "eligible"
    svc.set_excluded(t["turn_id"], True, "kişisel not")
    assert store.get_candidate(c["candidate_id"])["status"] == "excluded"
    svc.set_excluded(t["turn_id"], False)
    assert store.get_candidate(c["candidate_id"])["status"] == "eligible"


def test_eval_leak_is_excluded(store, conv, monkeypatch) -> None:
    from app.evals.profile.schema import EvalItem
    from app.feedback import learning

    q = "Volatilite kümelenmesi GARCH modelinde nasıl ölçülür ve neden önemlidir?"
    monkeypatch.setattr(
        learning,
        "leak_items_provider",
        lambda: [EvalItem(id="gizli-1", domain="statistics", question=q)],
    )
    t = send(store, conv, q)
    c, _ = _svc(store).correct(t["turn_id"], SUPPORTED_SENTENCE)
    assert c["status"] == "leak" and "gizli-1" in c["status_reason"]


def test_eval_question_in_history_leaks_later_turn(store, conv, monkeypatch) -> None:
    """Önceki turdaki gizli soru geçmiş olarak isteme girer → o tur da sızıntı sayılır."""
    from app.evals.profile.schema import EvalItem
    from app.feedback import learning

    q = "Gizli değerlendirme sorusu: EMA katsayısı pencere uzunluğuyla nasıl ilişkilidir?"
    monkeypatch.setattr(
        learning, "leak_items_provider", lambda: [EvalItem(id="gizli-2", domain="math", question=q)]
    )
    send(store, conv, q)
    t2 = send(store, conv, "Volatilite kümelenmesi nedir?")
    c, _ = _svc(store).correct(t2["turn_id"], SUPPORTED_SENTENCE)
    assert c["status"] == "leak"


# ── aileler ──────────────────────────────────────────────────────────────────


def test_bridge_between_train_and_eval_families_is_conflict(store, monkeypatch) -> None:
    from app.config import settings as settings_mod

    monkeypatch.setenv("HEKTOR_LEARNING_FAMILY_JACCARD", "0.2")
    settings_mod.get_settings.cache_clear()
    svc = _svc(store)
    conv = store.create_conversation()["conversation_id"]
    ta = send(store, conv, "alpha beta gamma delta epsilon")
    tb = send(store, conv, "zeta eta theta iota kappa")
    ca, _ = svc.correct(ta["turn_id"], SUPPORTED_SENTENCE)
    cb, _ = svc.correct(tb["turn_id"], SUPPORTED_SENTENCE + " Ek açıklama cümlesi burada.")
    store.upsert_family(ca["family_id"], split="train")
    store.upsert_family(cb["family_id"], split="eval")
    tc = send(store, conv, "alpha beta gamma delta zeta eta theta iota")
    cc, _ = svc.correct(tc["turn_id"], SUPPORTED_SENTENCE)
    assert cc["status"] == "conflict" and "Sızıntı çatışması" in cc["status_reason"]
    assert cc["family_id"] == ""


def test_family_duplicate_and_numeric_conflict(store) -> None:
    svc = _svc(store)
    conv = store.create_conversation()["conversation_id"]
    q = "On bölü dört kaç eder, hesaplar mısın?"
    t1 = send(store, conv, q)
    t2 = send(store, conv, q + " ")
    c1, _ = svc.correct(t1["turn_id"], "10 / 4 = 2.5")
    c2, _ = svc.correct(t2["turn_id"], "10 / 4 = 2.5")
    assert c1["family_id"] == c2["family_id"]
    assert store.get_candidate(c2["candidate_id"])["status"] == "duplicate"
    t3 = send(store, conv, q)
    c3, _ = svc.correct(t3["turn_id"], "10 / 2 = 5")
    statuses = {store.get_candidate(c["candidate_id"])["status"] for c in (c1, c3)}
    assert statuses == {"conflict"}


def test_citation_only_line_is_not_a_claim(store, conv) -> None:
    t = send(store, conv, "Volatilite kümelenmesi nedir?")
    target = SUPPORTED_SENTENCE + chr(10) + "[p1:c1]"
    c, _ = _svc(store).correct(t["turn_id"], target)
    assert c["verification"]["coverage"]["units"] == 1
    assert c["status"] == "eligible"
