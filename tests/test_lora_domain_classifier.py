"""Domain sınıflandırma (anahtar kelime) testleri."""

from __future__ import annotations

from app.lora.domain_classifier import Domain, classify_domains


def test_mathematics_detected_english() -> None:
    """İngilizce matematik anahtar kelimesi MATHEMATICS döndürmeli."""
    assert Domain.MATHEMATICS in classify_domains("The derivative of the formula")


def test_uppercase_dotless_i_keyword_matched() -> None:
    """BÜYÜK harf 'ı'lı anahtar ('HIZ') de eşleşmeli ('HIZ'.lower()=='hiz' bypass'ı kapalı)."""
    assert classify_domains("hız") == classify_domains("HIZ")
    assert classify_domains("HIZ")  # boş değil — 'hız' anahtarı yakalandı


def test_mathematics_detected_turkish() -> None:
    """Türkçe matematik anahtar kelimesi MATHEMATICS döndürmeli."""
    assert Domain.MATHEMATICS in classify_domains("Türev ve integral hesabı")


def test_trading_and_risk_detected_together() -> None:
    """Bir metin birden çok domaine ait olabilmeli."""
    domains = classify_domains("backtest stratejisi ve drawdown riski")
    assert Domain.TRADING in domains
    assert Domain.RISK_MANAGEMENT in domains


def test_ai_system_design_detected() -> None:
    """RAG/LoRA terimleri AI_SYSTEM_DESIGN döndürmeli."""
    assert Domain.AI_SYSTEM_DESIGN in classify_domains("RAG ve LoRA fine-tuning")


def test_statistics_detected() -> None:
    """İstatistik anahtar kelimesi STATISTICS döndürmeli."""
    assert Domain.STATISTICS in classify_domains("regresyon ve korelasyon analizi")


def test_machine_learning_detected() -> None:
    """Kademe-2 kör noktası: 'machine/deep learning', 'neural network' AI_SYSTEM_DESIGN.

    Önceden bu kartlar (olasılıksal-ML) hiçbir domain'e eşlenmiyordu → Gate 0/3 FAIL.
    """
    assert Domain.AI_SYSTEM_DESIGN in classify_domains("machine learning for forecasting")
    assert Domain.AI_SYSTEM_DESIGN in classify_domains("a deep neural network model")
    assert Domain.AI_SYSTEM_DESIGN in classify_domains("LSTM tabanlı tahmin")


def test_markov_stochastic_detected_mathematics() -> None:
    """Kademe-2 kör noktası: 'markov'/'stochastic' MATHEMATICS döndürmeli."""
    assert Domain.MATHEMATICS in classify_domains("Markov chain analysis")
    assert Domain.MATHEMATICS in classify_domains("stokastik süreç")


def test_empty_text_returns_empty_list() -> None:
    """Boş metin için boş liste dönmeli."""
    assert classify_domains("") == []


def test_no_match_returns_empty_list() -> None:
    """Eşleşme yoksa boş liste dönmeli (zorla atama yok)."""
    assert classify_domains("xyzzy plugh foobar") == []


def test_result_follows_enum_order() -> None:
    """Sonuç Domain enum sırasını korumalı."""
    domains = classify_domains("python kodu ile backtest")
    assert domains == sorted(domains, key=list(Domain).index)
