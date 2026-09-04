"""Kalite filtresi testleri."""

from __future__ import annotations

from app.lora.quality_filter import QualityFilter, check_quality


def _card(question: str, answer: str) -> dict:
    return {"question": question, "answer": answer}


def test_short_answer_fails() -> None:
    """50 karakterden kısa cevap reddedilmeli."""
    result = check_quality(_card("RSI nedir?", "kısa"))
    assert result.passed is False
    assert "kısa" in result.reason


def test_long_distinct_answer_passes() -> None:
    """Yeterli uzunlukta ve özgün cevap geçmeli."""
    answer = (
        "RSI, fiyatın aşırı alım ve aşırı satım bölgelerini ölçen bir "
        "momentum osilatörüdür ve 0 ile 100 arasında değer alır."
    )
    result = check_quality(_card("RSI nedir?", answer))
    assert result.passed is True


def test_answer_repeats_question_fails() -> None:
    """Cevap soruyu birebir tekrar ediyorsa reddedilmeli."""
    text = "aşırı alım aşırı satım momentum osilatör değer bölge fiyat ölçer gösterge"
    result = check_quality(_card(text, text))
    assert result.passed is False
    assert "tekrar" in result.reason


def test_score_in_unit_interval() -> None:
    """Skor 0-1 aralığında olmalı."""
    answer = "Bu yeterince uzun ve özgün bir açıklama metnidir, gerçekten."
    result = check_quality(_card("soru?", answer))
    assert 0.0 <= result.score <= 1.0


def test_filter_batch_separates_passed_and_rejected() -> None:
    """filter_batch geçen ve reddedilenleri ayırmalı."""
    long_answer = "Bu, elli karakterden uzun, anlamlı ve özgün bir cevaptır gerçekten."
    cards = [
        _card("Soru bir?", long_answer),
        _card("Soru iki?", "çok kısa"),
    ]
    passed, rejected = QualityFilter().filter_batch(cards)
    assert len(passed) == 1
    assert len(rejected) == 1


def test_filter_batch_removes_duplicates() -> None:
    """Aynı içerik ikinci kez gelince duplicate olarak reddedilmeli."""
    answer = "Bu uzun ve benzersiz açıklama metni elli karakteri rahatça aşar bence."
    cards = [_card("Aynı soru?", answer), _card("Aynı soru?", answer)]
    passed, rejected = QualityFilter().filter_batch(cards)
    assert len(passed) == 1
    assert len(rejected) == 1
    assert rejected[0]["_quality_reason"] == "duplicate içerik"


def test_filter_batch_keeps_distinct_content() -> None:
    """Farklı içerikler duplicate sayılmamalı."""
    cards = [
        _card("Soru A?", "Birinci özgün ve yeterince uzun açıklama metnidir kesinlikle gerçek."),
        _card("Soru B?", "İkinci tamamen farklı ve uzun bir açıklama metnidir bambaşka konu."),
    ]
    passed, _ = QualityFilter().filter_batch(cards)
    assert len(passed) == 2


def test_rejected_card_carries_reason() -> None:
    """Reddedilen kart '_quality_reason' notu taşımalı."""
    _, rejected = QualityFilter().filter_batch([_card("s?", "kısa")])
    assert "_quality_reason" in rejected[0]
