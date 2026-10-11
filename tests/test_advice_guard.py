"""Kural 1 — al/sat KARARI sorusu + yönlendirici cevap (bulgu: cmp_759c2d9b609d · bi-04).

"BTC alayım mı?" → "Hayır." sohbette uyarısız gösteriliyor ve eğitim adayı olabiliyordu.
Çevrimdışı: sahte LLM/retriever.
"""

from __future__ import annotations

import pytest
from tests.chat_learning_helpers import FakeLLM, iso, send  # noqa: F401

from app.feedback import advice_guard as ag


@pytest.mark.parametrize(
    "q",
    [
        "BTC alayım mı?",
        "ETH'yi şimdi SATAYIM MI",
        "Altın almalı mıyım?",
        "THYAO'ya girsem mi?",
        "Pozisyonu kapatayım mı?",
        "Bugün long mu short mu?",
        "Should I buy Bitcoin now?",
        "buy or sell AAPL?",
    ],
)
def test_advice_requests_detected(q: str) -> None:
    assert ag.is_advice_request(q)


@pytest.mark.parametrize(
    "q",
    [
        "Volatilite kümelenmesi nedir?",
        "BTC yükselir mi?",  # tahmin sorusu — karar değil
        "Alım satım maliyeti nasıl hesaplanır?",
        "Sat fiyatı ile alış fiyatı arasındaki makas nedir?",
        "Bu strateji neden short tarafta zayıf?",
        "Altının fiyatı neden düştü?",
    ],
)
def test_non_advice_questions_not_flagged(q: str) -> None:
    assert not ag.is_advice_request(q)


@pytest.mark.parametrize(
    "a",
    [
        "Hayır.",
        "Evet, şimdi iyi bir zaman.",
        "No.",
        "Bence alın.",
        "Satmalısınız.",
        "Almanızı öneririm.",
        "You should buy now.",
    ],
)
def test_directive_answers_detected(a: str) -> None:
    assert ag.directive_answer(a)


def test_hypothesis_answer_not_directive() -> None:
    a = (
        "Bu bir al/sat kararı sorusu; tavsiye veremem. Test edilebilir hipotez: 20/50 EMA "
        "kesişimi maliyetler dahil örneklem dışında pozitif mi? Test noktası: 2020-2024 verisi."
    )
    assert not ag.directive_answer(a)
    assert ag.candidate_reason("BTC alayım mı?", a) is None


def test_chat_shows_guard_but_keeps_raw_answer(iso) -> None:  # noqa: F811
    from app.feedback.chat_store import ChatStore

    store = ChatStore()
    conv = store.create_conversation("k1")["conversation_id"]
    turn = send(store, conv, "BTC alayım mı?", llm=FakeLLM(["Hayır. [p1:c1]"]))
    assert turn["status"] == "answered"
    assert "Trading uyarıları:" in turn["answer"] and ag.NOTE_LINE in turn["answer"]
    assert "Trading uyarıları:" not in turn["raw_answer"]
    guard = [c for c in turn["checks"] if c["kind"] == "guvenlik"]
    assert guard and guard[0]["status"] == "ihlal"


def test_chat_concept_question_untouched(iso) -> None:  # noqa: F811
    from app.feedback.chat_store import ChatStore

    store = ChatStore()
    conv = store.create_conversation("k1")["conversation_id"]
    turn = send(store, conv, "Volatilite kümelenmesi nedir?")
    assert not [c for c in turn["checks"] if c["kind"] == "guvenlik"]
    assert ag.NOTE_LINE not in turn["answer"]


def test_learn_candidate_from_bare_no_is_rejected(iso) -> None:  # noqa: F811
    from app.feedback.chat_store import ChatStore
    from app.feedback.learning import LearningService

    store = ChatStore()
    conv = store.create_conversation("k1")["conversation_id"]
    turn = send(store, conv, "BTC alayım mı?", llm=FakeLLM(["Hayır. [p1:c1]"]))
    cand, _ = LearningService(store).learn(turn["turn_id"])
    assert cand["status"] == "rejected" and "guvenlik" in cand["reason_codes"]
    assert "Kural 1" in cand["status_reason"]
