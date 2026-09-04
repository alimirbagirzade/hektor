"""Metin temizliği skoru testleri (salt-regex, deterministik)."""

from __future__ import annotations

from app.ingestion.clean_text_scorer import score_clean_text


def test_clean_text_high() -> None:
    text = (
        "Bu makale volatilite rejimlerini inceler. Momentum stratejileri yüksek "
        "volatilitede zayıflar. Sonuçlar backtest ile doğrulanmıştır. " * 5
    )
    assert score_clean_text(text) >= 9.0


def test_empty_is_zero() -> None:
    assert score_clean_text("") == 0.0
    assert score_clean_text("   ") == 0.0


def test_control_chars_penalized() -> None:
    dirty = "metin" + ("\x00\x07\x1f" * 50) + "devam" * 50
    clean = "metin devam " * 50
    assert score_clean_text(dirty) < score_clean_text(clean)


def test_replacement_chars_penalized() -> None:
    text = "kelime " * 100
    dirty = text + ("�" * 30)
    assert score_clean_text(dirty) < score_clean_text(text)


def test_deterministic() -> None:
    t = "deterministik metin örneği " * 20
    assert score_clean_text(t) == score_clean_text(t)
