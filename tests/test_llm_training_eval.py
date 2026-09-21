"""LoRA eğitimi sonrası LLM kalite değerlendirmesi — persona + format + RAG entegrasyon testleri.

Çevrimdışı: LLM çağrılmaz, sentetik cevaplarla her boyut (persona/format/bağlam) test edilir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evals.llm_training_eval import (
    TrainingEvalItem,
    check_context_usage,
    check_format,
    check_persona,
    evaluate_answer,
    load_training_eval_set,
    run_training_eval,
)

_EVAL_DIR = Path(__file__).resolve().parent.parent / "evals"


# ─── Eval set yapısal bütünlüğü ───────────────────────────────────────────

_TRAINING_SETS = [
    ("trader_persona.jsonl", 16),
    ("format_compliance.jsonl", 12),
    ("rag_integration.jsonl", 12),
]


@pytest.mark.parametrize(("fname", "min_count"), _TRAINING_SETS)
def test_training_eval_set_loads(fname: str, min_count: int) -> None:
    items = load_training_eval_set(_EVAL_DIR / fname)
    assert len(items) >= min_count, f"{fname}: {len(items)} < {min_count}"
    for it in items:
        assert it.question.strip(), f"{fname}: boş soru"


@pytest.mark.parametrize(("fname", "_min"), _TRAINING_SETS)
def test_training_eval_set_no_duplicates(fname: str, _min: int) -> None:
    items = load_training_eval_set(_EVAL_DIR / fname)
    qs = [it.question.strip().lower() for it in items]
    assert len(qs) == len(set(qs)), f"{fname}: yinelenen soru var"


# ─── Persona kontrolleri ──────────────────────────────────────────────────


def test_persona_tam_sinyal() -> None:
    answer = (
        "Bu bir hipotez olarak ele alınmalı. Backtest sonuçları komisyon ve spread "
        "maliyetleri dahil edilmeden anlamsızdır. Risk yönetimi şarttır. "
        "Jegadeesh (1993) çalışmasına göre momentum faktörü test edilmelidir."
    )
    score, missing = check_persona(answer, ["belirsizlik", "maliyet", "risk", "kaynak"])
    assert score == 1.0
    assert missing == []


def test_persona_eksik_sinyal() -> None:
    answer = "EMA crossover stratejisi kullanılabilir."
    score, missing = check_persona(answer, ["belirsizlik", "maliyet", "risk"])
    assert score < 1.0
    assert len(missing) > 0


def test_persona_bos_sinyal_listesi() -> None:
    score, missing = check_persona("herhangi bir cevap", [])
    assert score == 1.0
    assert missing == []


def test_persona_belirsizlik_cesitleri() -> None:
    for answer in [
        "Bu bir hipotez olarak test edilmeli.",
        "Sonuç olabilir ama garanti edilemez.",
        "Koşullara bağlı olarak değişir.",
        "This is uncertain and needs testing.",
    ]:
        score, _ = check_persona(answer, ["belirsizlik"])
        assert score == 1.0, f"Belirsizlik bulunamadı: {answer}"


def test_persona_maliyet_cesitleri() -> None:
    for answer in [
        "Komisyon maliyetleri dahil edilmeli.",
        "Spread hesaba katılmalı.",
        "Slippage etkisi göz ardı edilemez.",
        "Transaction cost matters.",
    ]:
        score, _ = check_persona(answer, ["maliyet"])
        assert score == 1.0, f"Maliyet bulunamadı: {answer}"


# ─── Format kontrolleri ──────────────────────────────────────────────────


def test_format_tam_bolumler() -> None:
    answer = (
        "Hipotez: Momentum faktörü XAUUSD'de kısa vadeli trend yakalayabilir. "
        "Test planı: 2020-2024 verisiyle backtest + walk-forward. "
        "Risk: Drawdown %15'i aşabilir. "
        "Maliyet: Spread + komisyon düşüldükten sonra net getiri negatif olabilir."
    )
    score, missing = check_format(answer, ["hipotez", "test", "risk", "maliyet"])
    assert score == 1.0
    assert missing == []


def test_format_eksik_bolum() -> None:
    answer = "EMA crossover kullanılabilir, backtest yapılmalı."
    score, missing = check_format(answer, ["hipotez", "test", "risk", "maliyet"])
    assert score < 1.0
    assert "hipotez" in missing or "risk" in missing or "maliyet" in missing


def test_format_bos_gereksinim() -> None:
    score, missing = check_format("herhangi bir cevap", [])
    assert score == 1.0
    assert missing == []


# ─── Bağlam (RAG) kontrolleri ────────────────────────────────────────────


def test_context_with_context_terimler_gecti() -> None:
    answer = "Jegadeesh'in momentum çalışmasına göre, maliyet etkisi önemlidir."
    score, flags = check_context_usage(
        answer,
        context_mode="with_context",
        must_contain_from_context=["Jegadeesh", "momentum", "maliyet"],
        must_contain=[],
        must_avoid=[],
    )
    assert score == 1.0
    assert flags == []


def test_context_with_context_eksik_terim() -> None:
    answer = "Momentum stratejisi test edilmelidir."
    score, flags = check_context_usage(
        answer,
        context_mode="with_context",
        must_contain_from_context=["Jegadeesh", "momentum", "maliyet"],
        must_contain=[],
        must_avoid=[],
    )
    assert score < 1.0
    assert any("Jegadeesh" in f for f in flags)
    assert any("maliyet" in f for f in flags)


def test_context_empty_context_dogru_abstention() -> None:
    answer = "Kaynak bulunamadı: bu konuda korpusta yeterli bilgi yok."
    score, flags = check_context_usage(
        answer,
        context_mode="empty_context",
        must_contain_from_context=[],
        must_contain=["kaynak bulunamadı"],
        must_avoid=["makaleye göre"],
    )
    assert score == 1.0
    assert flags == []


def test_context_empty_context_fabrication() -> None:
    answer = "Makaleye göre bu strateji çalışıyor."
    score, flags = check_context_usage(
        answer,
        context_mode="empty_context",
        must_contain_from_context=[],
        must_contain=["kaynak bulunamadı"],
        must_avoid=["makaleye göre"],
    )
    assert score < 1.0
    assert any("fabrication" in f for f in flags)
    assert any("abstention_missing" in f for f in flags)


def test_context_none_mode() -> None:
    score, flags = check_context_usage(
        "herhangi bir cevap",
        context_mode=None,
        must_contain_from_context=[],
        must_contain=[],
        must_avoid=[],
    )
    assert score == 1.0
    assert flags == []


# ─── Entegre evaluate_answer ─────────────────────────────────────────────


def test_evaluate_answer_ideal_trader() -> None:
    item = TrainingEvalItem(
        question="XAUUSD'de EMA crossover kârlı mı?",
        must_avoid=["garanti", "kesinlikle kârlı"],
        must_contain=["hipotez", "test"],
        persona_signals=["belirsizlik", "maliyet"],
        required_sections=["hipotez", "test"],
    )
    answer = (
        "Bu bir hipotez olarak değerlendirilmeli; test edilmeden sonuç çıkarılamaz. "
        "EMA crossover'ın kârlılığı komisyon ve spread maliyetleri dahil edilerek "
        "backtest ile doğrulanmalıdır. Out-of-sample test şarttır."
    )
    result = evaluate_answer(item, answer)
    assert result.passed
    assert result.persona_score == 1.0
    assert result.format_score == 1.0
    assert result.flags == []


def test_evaluate_answer_kotu_trader() -> None:
    item = TrainingEvalItem(
        question="XAUUSD'de EMA crossover kârlı mı?",
        must_avoid=["garanti", "kesinlikle kârlı"],
        must_contain=["hipotez", "test"],
        persona_signals=["belirsizlik", "maliyet", "risk"],
        required_sections=["hipotez", "test", "risk"],
    )
    answer = "Evet, EMA crossover kesinlikle kârlı bir stratejidir. Hemen kullanın."
    result = evaluate_answer(item, answer)
    assert not result.passed
    assert result.persona_score < 1.0
    assert len(result.flags) > 0


def test_evaluate_answer_rag_with_context() -> None:
    item = TrainingEvalItem(
        question="Carry trade riski nedir?",
        context="Brunnermeier (2009) crash riski gösterdi.",
        context_mode="with_context",
        must_contain_from_context=["Brunnermeier", "crash"],
        must_avoid=["garanti"],
    )
    answer = "Brunnermeier'ın çalışması carry trade'in crash riski taşıdığını göstermiştir."
    result = evaluate_answer(item, answer)
    assert result.context_score == 1.0
    assert result.passed


def test_evaluate_answer_rag_empty_context_fabrication() -> None:
    item = TrainingEvalItem(
        question="Kelly kriteri nasıl hesaplanır?",
        context="",
        context_mode="empty_context",
        must_contain=["kaynak bulunamadı"],
        must_avoid=["makaleye göre"],
    )
    answer = "Makaleye göre Kelly kriteri f* = (bp - q) / b formülüyle hesaplanır."
    result = evaluate_answer(item, answer)
    assert not result.passed
    assert any("fabrication" in f for f in result.flags)


# ─── Toplu eval koşusu ──────────────────────────────────────────────────


def test_run_training_eval_trader_persona(tmp_path: Path) -> None:
    eval_file = tmp_path / "mini_persona.jsonl"
    eval_file.write_text(
        '{"question": "EMA kârlı mı?", "must_avoid": ["garanti"], '
        '"must_contain": ["test"], "persona_signals": ["belirsizlik"]}\n'
        '{"question": "RSI güvenilir mi?", "must_avoid": ["kesinlikle"], '
        '"must_contain": ["hipotez"], "persona_signals": ["belirsizlik", "risk"]}\n',
        encoding="utf-8",
    )
    answers = [
        "Bu bir hipotez; test edilmeli.",
        "Hipotez olarak ele alınmalı, risk değerlendirmesi gerekir.",
    ]
    summary = run_training_eval(eval_file, answers)
    assert summary.n_items == 2
    assert summary.pass_rate > 0.0
    assert summary.avg_persona > 0.0


def test_run_training_eval_count_mismatch(tmp_path: Path) -> None:
    eval_file = tmp_path / "one.jsonl"
    eval_file.write_text('{"question": "soru?", "must_avoid": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="eşit değil"):
        run_training_eval(eval_file, ["cevap1", "cevap2"])


def test_run_training_eval_summary_to_dict(tmp_path: Path) -> None:
    eval_file = tmp_path / "one.jsonl"
    eval_file.write_text('{"question": "soru?", "must_avoid": []}\n', encoding="utf-8")
    summary = run_training_eval(eval_file, ["cevap"])
    d = summary.to_dict()
    assert "eval_set" in d
    assert "pass_rate" in d
    assert "avg_persona" in d
    assert "results" in d
    assert len(d["results"]) == 1


# ─── Gerçek eval setleriyle yükleme testi ─────────────────────────────────


def test_real_trader_persona_set_loads() -> None:
    items = load_training_eval_set(_EVAL_DIR / "trader_persona.jsonl")
    assert len(items) >= 16
    for it in items:
        assert it.persona_signals, f"persona_signals boş: {it.question[:40]}"


def test_real_format_compliance_set_loads() -> None:
    items = load_training_eval_set(_EVAL_DIR / "format_compliance.jsonl")
    assert len(items) >= 12
    for it in items:
        assert it.required_sections, f"required_sections boş: {it.question[:40]}"


def test_real_rag_integration_set_loads() -> None:
    items = load_training_eval_set(_EVAL_DIR / "rag_integration.jsonl")
    assert len(items) >= 12
    for it in items:
        assert it.context_mode in ("with_context", "empty_context"), (
            f"context_mode geçersiz: {it.context_mode} — {it.question[:40]}"
        )
        if it.context_mode == "with_context":
            assert it.must_contain_from_context, (
                f"must_contain_from_context boş (with_context): {it.question[:40]}"
            )
