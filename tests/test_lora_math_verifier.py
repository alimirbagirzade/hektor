"""Matematik / istatistik doğrulayıcı testleri."""

from __future__ import annotations

import pytest

from app.lora.math_verifier import verify_math_content


@pytest.mark.parametrize(
    "text",
    [
        # Canlı vaka (2026-09-27 lora-audit, card_d489175087d1 — Cholesky/Bayesian Filtering).
        "Higher-order filters increase computational complexity without guaranteed "
        "performance gains.",
        "Convergence is not guaranteed for non-convex objectives.",
        "Profits are not necessarily guaranteed.",
        "There are no guaranteed returns in live markets.",
        "Getiri garanti değildir.",
        "Kâr garantisi yoktur.",
    ],
)
def test_negated_overconfident_phrase_is_not_blocker(text: str) -> None:
    """Olumsuzlanmış aşırı-emin ifade ihtiyatlı dildir → blok DEĞİL (Gate 5 yanlış reddi)."""
    result = verify_math_content(text)
    assert result.passed is True, result.issues
    assert not any("aşırı emin" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "text",
    [
        "This strategy delivers guaranteed returns.",
        "You will never lose, guaranteed.",  # olumsuzlayıcı ifadeye değil başka söze bağlı
        "Not only profitable but guaranteed to win.",  # olumsuzlayıcı iki+ kelime uzakta
        "Convergence is not guaranteed, but profit is guaranteed.",  # ikinci geçiş çıplak
        "Bu yöntem garantili getiri sağlar.",
    ],
)
def test_unnegated_overconfident_phrase_still_fails(text: str) -> None:
    """Olumsuzlama muafiyeti gerçek aşırı-emin dili KAÇIRMAMALI (yanlış negatif koruması)."""
    assert verify_math_content(text).passed is False


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


def test_cumulative_return_over_100_not_flagged_as_risk() -> None:
    """Regresyon (2026-09-27 lora-audit, card_89d30aac91b1): "risk" metnin BAŞKA bir yerinde
    geçince %100 üstü kümülatif getiri risk yüzdesi sanılıyordu."""
    text = (
        "The strategy manages downside risk with volatility targeting. It achieves cumulative "
        "returns of 340%, 185%, 371%, and 360% respectively over 2010-2018."
    )
    result = verify_math_content(text)
    assert not any("risk yüzdesi" in issue for issue in result.issues), result.issues


def test_risk_percentage_far_from_risk_word_not_flagged() -> None:
    """Risk kelimesi yüzdeden uzaktaysa (ayrı bağlam) kural tetiklenmemeli."""
    text = "Risk is discussed in chapter two. " + "Filler text. " * 10 + "Volume grew 250%."
    assert not any("risk yüzdesi" in i for i in verify_math_content(text).issues)


def test_risk_percentage_over_100_english_flagged() -> None:
    """Yerel risk bağlamı İngilizce de yakalanmalı (yanlış negatif koruması)."""
    result = verify_math_content("Set the risk per trade to 150% of equity.")
    assert any("risk yüzdesi" in issue for issue in result.issues)


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
