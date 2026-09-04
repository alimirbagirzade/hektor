"""Offline RAGAS-tarzı metrik testleri — deterministik, LLM'siz."""

from __future__ import annotations

import pytest

from app.evals.rag_ragas_offline import (
    context_precision,
    context_recall,
    evaluate_rag_answer,
    faithfulness,
)


def test_faithfulness_partial_support() -> None:
    """Bağlamca desteklenen cümle sayılır; dayanaksız cümle skoru düşürür."""
    answer = "RSI momentum sinyali üretir. Kediler havada uçar dans eder."
    contexts = ["RSI momentum sinyali ve volatilite rejimleri hakkında analiz."]
    # 1. cümle bağlamca desteklenir (yüksek örtüşme), 2. cümle desteksiz → 1/2.
    assert faithfulness(answer, contexts) == pytest.approx(0.5)


def test_faithfulness_empty_answer() -> None:
    assert faithfulness("", ["herhangi bağlam metni"]) == 0.0


def test_faithfulness_full_support() -> None:
    answer = "ATR volatilite ölçer."
    contexts = ["ATR volatilite ölçer ve stop mesafesi belirler."]
    assert faithfulness(answer, contexts) == pytest.approx(1.0)


def test_context_precision_filters_noise() -> None:
    """Cevaba katkısız (gürültü) bağlam parçası precision'ı düşürür."""
    answer = "RSI momentum sinyali güçlüdür"
    contexts = [
        "RSI momentum hesaplama detayları burada",  # alakalı
        "tamamen alakasız yemek tarifi malzeme listesi",  # gürültü
    ]
    assert context_precision(answer, contexts) == pytest.approx(0.5)


def test_context_precision_empty_contexts() -> None:
    assert context_precision("herhangi cevap", []) == 0.0


def test_context_recall_partial_coverage() -> None:
    """Referans cevabın bağlamca kapsanan token oranı."""
    reference = "RSI ATR birlikte kullanılır"  # içerik token: rsi, atr, birlikte, kullanılır
    contexts = ["RSI göstergesi tanımı", "ATR volatilite ölçümü"]
    # ref ∩ bağlam = {rsi, atr} → 2/4
    assert context_recall(reference, contexts) == pytest.approx(0.5)


def test_context_recall_empty_reference() -> None:
    assert context_recall("", ["bağlam"]) == 0.0


def test_evaluate_rag_answer_with_and_without_reference() -> None:
    answer = "RSI momentum sinyali üretir."
    contexts = ["RSI momentum sinyali analizi ve volatilite."]

    without = evaluate_rag_answer(answer, contexts)
    assert without.context_recall is None
    assert without.n_contexts == 1
    assert without.n_answer_sentences == 1
    assert 0.0 <= without.faithfulness <= 1.0
    assert 0.0 <= without.context_precision <= 1.0

    with_ref = evaluate_rag_answer(answer, contexts, reference="RSI momentum sinyali")
    assert with_ref.context_recall is not None
    assert 0.0 <= with_ref.context_recall <= 1.0


def test_deterministic_repeat() -> None:
    """Aynı girdi → aynı skor (determinizm, Kural 6)."""
    answer = "ATR volatilite ölçer ve stop belirler."
    contexts = ["ATR volatilite ölçer.", "alakasız metin"]
    a = evaluate_rag_answer(answer, contexts, reference="ATR stop")
    b = evaluate_rag_answer(answer, contexts, reference="ATR stop")
    assert a == b
