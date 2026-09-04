"""UnderstandingScore testleri — objektif geçme-oranı; skipped/no_data paydaya girmez."""

from __future__ import annotations

from dataclasses import dataclass

from app.verification.exams.l3_application import ExamResult
from app.verification.exams.l5_composition import CompositionResult, GateResult
from app.verification.exams.understanding_score import (
    aggregate,
    composition_to_result,
    rag_answers_to_results,
)


@dataclass
class _Ans:
    question_type: str = "main_claim"
    requires_abstention: bool = False
    answer_text: str = "geçerli bir cevap"
    citation_score: float = 1.0
    grounding_score: float = 1.0
    abstention_correct: bool = False
    hallucination_detected: bool = False


def _r(level: str, status: str) -> ExamResult:
    return ExamResult(level=level, name="x", passed=(status == "passed"), status=status, seed=0)


def test_pass_rate_skipped_haric() -> None:
    results = [
        _r("L3", "passed"),
        _r("L3", "passed"),
        _r("L3", "failed"),
        _r("L4", "skipped"),  # paydaya girmez
        _r("L4", "no_data"),  # paydaya girmez
    ]
    score = aggregate(results)
    assert score.total == 5
    assert score.passed == 2
    assert score.failed == 1
    assert score.skipped == 1
    assert score.no_data == 1
    assert score.graded == 3
    assert abs(score.pass_rate - 2 / 3) < 1e-9
    assert score.status == "scored"


def test_hic_notlanan_yoksa_insufficient() -> None:
    score = aggregate([_r("L3", "skipped"), _r("L4", "skipped")])
    assert score.pass_rate is None
    assert score.status == "insufficient_data"
    assert score.graded == 0


def test_min_graded_guard_kucuk_n_sismeyi_onler() -> None:
    """graded < eşik(3) → 'scored %100' DEĞİL, 'insufficient_data'.

    Çevrimdışı tam-merdivende L3/L4 'skipped' olup yalnız deterministik L5 örneği
    notlanırsa (graded=1) tek geçen sınav pass_rate=%100 + 'scored' verip auto_pipeline
    base-vs-adapter TERFİ kapısını şişirirdi (v5-gerileme sınıfı küçük-n sahte güveni).
    """
    one = aggregate([_r("L5", "passed")])  # tek notlanan sınav geçti
    assert one.graded == 1
    assert one.pass_rate is None
    assert one.status == "insufficient_data"

    two = aggregate([_r("L5", "passed"), _r("L3", "failed")])
    assert two.graded == 2
    assert two.status == "insufficient_data"

    # eşik (3) sağlanınca normal notlama (mevcut 'graded=3 → scored' sözleşmesi korunur)
    three = aggregate([_r("L3", "passed"), _r("L3", "passed"), _r("L3", "failed")])
    assert three.graded == 3
    assert three.status == "scored"
    assert abs(three.pass_rate - 2 / 3) < 1e-9


def test_score_indicator_exams_llm_yoksa_insufficient(monkeypatch) -> None:
    # LLM erişilemezse tüm sınavlar 'skipped' → insufficient_data, çökme yok.
    from app.brain.local_llm import LocalLLM
    from app.verification.exams.understanding_score import score_indicator_exams

    monkeypatch.setattr(LocalLLM, "available", lambda self: False)
    score = score_indicator_exams(seed=0)
    assert score.status == "insufficient_data"
    assert score.graded == 0
    assert score.skipped == score.total > 0


def test_score_indicator_exams_llm_timeout_500_yok(monkeypatch) -> None:
    # REGRESYON: Ollama erişilebilir görünüp generate() zaman aşımına uğrarsa
    # (yavaş CPU) önceden ham httpx.ReadTimeout sızıp /api/understanding-score'u
    # 500 yapıyordu. Artık 'skipped' → insufficient_data olmalı, çökme YOK.
    import httpx

    from app.brain.local_llm import LocalLLM
    from app.verification.exams.understanding_score import score_indicator_exams

    monkeypatch.setattr(LocalLLM, "available", lambda self: True)

    def _timeout(self, *a, **k):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(LocalLLM, "generate", _timeout)
    score = score_indicator_exams(seed=0)  # çökmemeli
    assert score.status == "insufficient_data"
    assert score.graded == 0
    assert score.skipped > 0


def test_by_level_kirilimi() -> None:
    score = aggregate([_r("L3", "passed"), _r("L4", "failed"), _r("L4", "passed")])
    assert score.by_level["L3"]["passed"] == 1
    assert score.by_level["L4"]["failed"] == 1
    assert score.by_level["L4"]["passed"] == 1


def test_composition_aday_passed() -> None:
    comp = CompositionResult(
        name="rte",
        candidate=True,
        verdict="candidate",
        gates=[
            GateResult("math", True),
            GateResult("novelty", True),
            GateResult("backtest", True),
        ],
    )
    assert composition_to_result(comp).status == "passed"


def test_composition_veri_yok_skipped() -> None:
    comp = CompositionResult(
        name="rte",
        candidate=False,
        verdict="rejected",
        gates=[
            GateResult("math", True),
            GateResult("novelty", True),
            GateResult("backtest", False, ["veri yok — backtest sertifikalanamadı (aday değil)"]),
        ],
    )
    # math+novelty geçti ama test edilemedi → skipped (paydaya girmez)
    assert composition_to_result(comp).status == "skipped"


def test_composition_substantif_red_failed() -> None:
    comp = CompositionResult(
        name="kotu",
        candidate=False,
        verdict="rejected",
        gates=[
            GateResult("math", False, ["RSI kuralı aralık dışı"]),
            GateResult("novelty", True),
            GateResult("backtest", False, ["matematik kapısı geçilmedi — backtest atlandı"]),
        ],
    )
    assert composition_to_result(comp).status == "failed"


# ---------------------------------------------------------------- RAG L1/L2 adaptörü
def test_rag_iyi_cevap_l1_l2_passed() -> None:
    levels = {r.level: r for r in rag_answers_to_results([_Ans()])}
    assert levels["L1"].status == "passed"
    assert levels["L2"].status == "passed"


def test_rag_halusinasyon_l2_failed() -> None:
    res = rag_answers_to_results([_Ans(hallucination_detected=True)])
    l2 = next(r for r in res if r.level == "L2")
    assert l2.status == "failed"


def test_rag_dusuk_citation_l1_failed() -> None:
    res = rag_answers_to_results([_Ans(citation_score=0.1)])
    l1 = next(r for r in res if r.level == "L1")
    assert l1.status == "failed"


def test_rag_abstention_dogru_taban_passed() -> None:
    res = rag_answers_to_results(
        [_Ans(requires_abstention=True, abstention_correct=True, answer_text="")]
    )
    assert len(res) == 1
    assert res[0].level == "Taban"
    assert res[0].status == "passed"


def test_rag_cevap_yok_no_data() -> None:
    res = rag_answers_to_results([_Ans(answer_text="")])
    assert all(r.status == "no_data" for r in res)
    assert aggregate(res).graded == 0  # no_data paydaya girmez


def test_rag_l3_birlesik_skor() -> None:
    combined = [
        *rag_answers_to_results([_Ans(), _Ans(hallucination_detected=True)]),
        _r("L3", "passed"),
    ]
    score = aggregate(combined)
    assert {"L1", "L2", "L3"}.issubset(score.by_level)
    assert score.graded == score.passed + score.failed
