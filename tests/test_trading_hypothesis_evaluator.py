"""Trading hipotez değerlendiricisi testleri (CLAUDE.md Kural 1-4 denetimi, offline)."""

from __future__ import annotations

from app.evals.trading_hypothesis_evaluator import evaluate_hypothesis, evaluate_many

_GOOD = {
    "hypothesis_text": (
        "Yüksek volatilite rejiminde basit momentum sinyalleri zayıflayabilir; bu hipotez "
        "likidite filtresiyle backtest edilerek test edilmeli. Örneklem-dışı (out-of-sample) "
        "doğrulama yapılmalı ve komisyon + slippage maliyetleri dahil edilmeli. Pozisyon "
        "riski stop-loss ile sınırlanmalı."
    )
}


def test_good_hypothesis_is_candidate() -> None:
    res = evaluate_hypothesis(_GOOD)
    assert res.verdict == "candidate"
    assert res.score == 1.0
    assert all(res.checklist.values())


def test_advice_language_rejected() -> None:
    res = evaluate_hypothesis("Bu strateji %100 kazandırır, garanti kârlı, hemen al.")
    assert res.verdict == "rejected"
    assert res.checklist["no_advice"] is False
    assert any("Kural 1" in w for w in res.warnings)


def test_guaranteed_english_rejected() -> None:
    res = evaluate_hypothesis(
        "This is a risk-free, guaranteed, surefire strategy that always wins."
    )
    assert res.verdict == "rejected"
    assert res.checklist["no_advice"] is False


def test_missing_costs_needs_revision() -> None:
    text = (
        "Momentum zayıflayabilir; backtest ile test edilmeli, örneklem-dışı doğrula, "
        "riski stop-loss ile sınırla."
    )
    res = evaluate_hypothesis(text)
    assert res.verdict == "needs_revision"
    assert res.checklist["costs"] is False
    assert res.checklist["testable"] is True


def test_missing_oos_needs_revision() -> None:
    text = (
        "Momentum zayıflayabilir; backtest ile test edilmeli, komisyon ve slippage dahil, "
        "risk stop-loss ile yönetilmeli."
    )
    res = evaluate_hypothesis(text)
    assert res.verdict == "needs_revision"
    assert res.checklist["out_of_sample"] is False


def test_non_testable_directive_rejected() -> None:
    res = evaluate_hypothesis("Altını şimdi al, fiyat yükselecek.")
    assert res.verdict == "rejected"
    assert res.checklist["testable"] is False


def test_dict_with_separate_fields() -> None:
    res = evaluate_hypothesis(
        {
            "title": "Volatilite-momentum etkileşimi",
            "hypothesis_text": "Eğer volatilite yüksekse momentum zayıflar ise test edilmeli.",
            "risk_notes": "drawdown ve stop-loss dikkate alınmalı",
            "assumptions": ["komisyon ve slippage dahil", "örneklem-dışı doğrulama"],
        }
    )
    assert res.checklist["costs"] is True
    assert res.checklist["out_of_sample"] is True
    assert res.checklist["risk_noted"] is True


def test_evaluate_many_assigns_ids() -> None:
    res = evaluate_many([_GOOD, "garanti %100 kâr"])
    assert len(res) == 2
    assert res[0].hypothesis_id == "hyp_0"
    assert res[1].verdict == "rejected"


def test_advice_regex_variants_rejected() -> None:
    # tireli/fiil/boşluklu varyantlar da yakalanmalı (gate kör noktası kapatıldı)
    for text in (
        "This is a sure-fire strategy.",
        "This will guarantee profits next quarter.",
        "You simply can t lose with this setup.",
    ):
        res = evaluate_hypothesis(text)
        assert res.checklist["no_advice"] is False, text
        assert res.verdict == "rejected", text


def test_negated_certainty_not_flagged_tr() -> None:
    """OLUMSUZLANMIŞ kesinlik dili ('risksiz değildir') reddi TETİKLEMEMELİ (Kural 1 alçakgönüllü).

    Regresyon: naif kelime-eşleşmesi mükemmel, risk-bilinçli hipotezleri HARD-reddediyordu
    (örn. score 0.8 → 'rejected'). Olumsuzlama-bilinçli tarama bunu düzeltir.
    """
    text = (
        "Hipotez: RSI<30 sonrası bir sonraki barda getiri artar mı? Backtest ve örneklem-dışı "
        "ile test et, komisyon ve slippage dahil. Hiçbir strateji risksiz değildir; "
        "stop-loss kullan."
    )
    res = evaluate_hypothesis(text)
    assert res.checklist["no_advice"] is True
    assert res.verdict == "candidate"


def test_negated_certainty_not_flagged_en() -> None:
    """'No strategy is risk-free' (olumsuz, alçakgönüllü) reddi TETİKLEMEMELİ."""
    text = (
        "If momentum rises then returns increase? Test with backtest and out-of-sample, "
        "including commission and slippage. No strategy is risk-free; note drawdown risk."
    )
    res = evaluate_hypothesis(text)
    assert res.checklist["no_advice"] is True


def test_genuine_advice_still_rejected_despite_nearby_negation() -> None:
    """Metinde başka yerde olumsuzlama olsa da GERÇEK kesinlik iddiası hâlâ reddedilmeli.

    'risksiz değildir' atlanır AMA 'garanti kazandırır' / 'kesin kâr' gerçek iddiadır → reddet.
    Fix'in cerrahi olduğunu (advice'ı kör etmediğini) kanıtlar.
    """
    text = "Risksiz değildir ama bu sistem garanti kazandırır, kesin kâr sağlar."
    res = evaluate_hypothesis(text)
    assert res.checklist["no_advice"] is False
    assert res.verdict == "rejected"


def test_advice_not_suppressed_by_cross_sentence_negation() -> None:
    """Ayrı CÜMLEDEKİ olumsuzlama gerçek tavsiyeyi bastırmamalı (Kademe-2 anchor'suz bypass).

    'Guaranteed profit. No doubt about it.' → 'No' ayrı cümlede; eski ±pencere bunu
    'Guaranteed'i alçakgönüllü sayıp tavsiyeyi kapıdan geçiriyordu (Kural 1 sahte-PASS).
    """
    res = evaluate_hypothesis("Guaranteed profit. No doubt about it.")
    assert res.checklist["no_advice"] is False
    assert res.verdict == "rejected"


def test_advice_not_suppressed_by_cross_clause_negation_tr() -> None:
    """Ayrı CÜMLECİKTEKİ ('asla satma') olumsuzlama 'Kesinlikle al' komutunu bastırmamalı."""
    res = evaluate_hypothesis("Kesinlikle al, asla satma.")
    assert res.checklist["no_advice"] is False
    assert res.verdict == "rejected"


def test_advice_not_suppressed_by_english_postfix_never() -> None:
    """İngilizce sonrası 'never fails' üstünlük iddiasıdır, olumsuzlama değil → bastırmamalı."""
    res = evaluate_hypothesis("This is risk-free and never fails to deliver.")
    assert res.checklist["no_advice"] is False
    assert res.verdict == "rejected"
