"""Matematik / istatistik doğrulayıcı testleri."""

from __future__ import annotations

from app.lora.math_verifier import verify_math_content


def test_clean_text_passes_without_review() -> None:
    """Temiz metin geçmeli ve inceleme gerektirmemeli."""
    result = verify_math_content("RSI 0 ile 100 arasında değer alan bir osilatördür.")
    assert result.passed is True
    assert result.requires_review is False
    assert result.issues == []


def test_lookahead_bias_flagged_for_review() -> None:
    """Look-ahead bias terimi inceleme işaretlemeli."""
    result = verify_math_content("Bu strateji look-ahead bias içeriyor olabilir.")
    assert result.requires_review is True
    assert any("look-ahead" in issue for issue in result.issues)


def test_overconfident_phrase_fails() -> None:
    """Aşırı emin yatırım ifadesi blocker olmalı (passed=False)."""
    result = verify_math_content("Bu strateji kesinlikle garanti kazandırır.")
    assert result.passed is False
    assert any("garanti" in issue for issue in result.issues)


def test_overconfident_uppercase_fails() -> None:
    """BÜYÜK HARF aşırı emin ifade de blocker olmalı (Türkçe İ bypass'ı kapalı)."""
    result = verify_math_content("BU STRATEJİ KESİNLİKLE GARANTİ KAZANDIRIR.")
    assert result.passed is False
    assert result.issues


def test_suspicious_high_return_flagged() -> None:
    """Aşırı yüksek getiri iddiası inceleme işaretlemeli."""
    result = verify_math_content("Yıllık getiri %5000 olur.")
    assert result.requires_review is True
    assert any("getiri" in issue for issue in result.issues)


def test_risk_percentage_over_100_flagged() -> None:
    """%100'ü aşan risk yüzdesi tutarsızlık olarak işaretlenmeli."""
    result = verify_math_content("Pozisyon başına risk %150 olmalı.")
    assert result.requires_review is True
    assert any("risk" in issue.lower() for issue in result.issues)


def test_empty_text_passes() -> None:
    """Boş metin sorunsuz geçmeli."""
    result = verify_math_content("")
    assert result.passed is True
    assert result.requires_review is False


# --------------------------------------------------------------------------- #
# Doğrulanmamış performans iddiaları (Gate 5 disiplin tespiti) — adversarial
# --------------------------------------------------------------------------- #
# Gerçek-pozitif: çıplak (bağlamsız) sayısal performans iddiaları işaretlenmeli
# ama BLOKLANMAMALI (requires_review, passed=True). Yanlış-pozitif: kanıt
# bağlamı (backtest/OOS/dönem) bitişikse işaretlenMEMELİ.


def test_bare_accuracy_claim_flagged_for_review() -> None:
    """Çıplak '92% accuracy' iddiası inceleme işaretlemeli ama bloklamamalı."""
    result = verify_math_content("This model predicts price movements with 92% accuracy.")
    assert result.requires_review is True
    assert result.passed is True  # blok değil — yalnız inceleme
    assert any("performans iddiası" in issue for issue in result.issues)


def test_outperform_by_percent_flagged_for_review() -> None:
    """'outperforming ... by 15%' çıplak üstünlük iddiası inceleme işaretlemeli."""
    result = verify_math_content("A framework outperforming traditional time-series models by 15%.")
    assert result.requires_review is True
    assert result.passed is True


def test_sharpe_at_least_claim_flagged_for_review() -> None:
    """'Sharpe ratio of at least 1.5' çıplak iddiası inceleme işaretlemeli."""
    result = verify_math_content("We propose a strategy with a Sharpe ratio of at least 1.5.")
    assert result.requires_review is True
    assert result.passed is True


def test_accuracy_with_backtest_evidence_not_flagged() -> None:
    """Bitişik backtest/OOS/dönem kanıtı olan doğruluk ölçümü işaretlenMEMELİ (FP)."""
    result = verify_math_content("Out-of-sample backtest (2010-2020) reported 72% accuracy.")
    assert result.requires_review is False
    assert result.passed is True
    assert not any("performans iddiası" in issue for issue in result.issues)


def test_turkish_contextual_hit_rate_not_flagged() -> None:
    """Meşru bağlamlı '%60 isabet (backtest, OOS)' yanlış-pozitif olmamalı."""
    result = verify_math_content(
        "Backtest sonucunda %60 isabet elde edildi (2010-2020, OOS dahil)."
    )
    assert result.requires_review is False
    assert result.passed is True


def test_sharpe_at_least_with_evidence_not_flagged() -> None:
    """Dönem kanıtı bitişik 'Sharpe en az 1.2' meşru ölçüm — işaretlenMEMELİ (FP)."""
    result = verify_math_content("Out-of-sample testte Sharpe oranı en az 1.2 çıktı (2015-2022).")
    assert result.requires_review is False
    assert result.passed is True
