"""MasteryScorer birim testleri."""

from __future__ import annotations

from app.learning.mastery_scorer import MasteryScore, MasteryScorer
from app.learning.paper_inspector import InspectionResult
from app.learning.rag_exam_runner import ExamAnswer


def _make_inspection(
    parse: float = 10.0,
    meta: float = 5.0,
    chunk: float = 15.0,
    index: float = 10.0,
) -> InspectionResult:
    r = InspectionResult(paper_id="p1")
    r.parse_score = parse
    r.metadata_score = meta
    r.chunk_quality = chunk
    r.index_score = index
    return r


def _make_answer(
    requires_abstention: bool = False,
    context_sufficient: bool = True,
    citation_score: float = 1.0,
    grounding_score: float = 1.0,
    abstention_correct: bool = True,
    hallucination_detected: bool = False,
    passed: bool = True,
) -> ExamAnswer:
    return ExamAnswer(
        answer_id="a1",
        question_id="q1",
        test_id="t1",
        paper_id="p1",
        question_text="Soru?",
        question_type="structural",
        requires_abstention=requires_abstention,
        answer_text="Cevap.",
        cited_paper_ids=["p1"],
        citation_score=citation_score,
        grounding_score=grounding_score,
        context_sufficient=context_sufficient,
        abstention_correct=abstention_correct,
        hallucination_detected=hallucination_detected,
        passed=passed,
    )


def test_perfect_score() -> None:
    inspection = _make_inspection()
    answers = [_make_answer() for _ in range(10)]
    scorer = MasteryScorer()
    score = scorer.compute(inspection, answers, test_id="t1")
    assert score.total_score >= 90.0
    assert score.final_status == "learned"


def test_zero_answers_partial_score() -> None:
    inspection = _make_inspection(parse=5.0, meta=3.0, chunk=8.0, index=5.0)
    scorer = MasteryScorer()
    score = scorer.compute(inspection, [], test_id="t1")
    assert score.total_score >= 0.0
    assert score.total_score <= 60.0


def test_fail_answers_lowers_score() -> None:
    inspection = _make_inspection()
    bad = [_make_answer(passed=False, citation_score=0.0, grounding_score=0.0) for _ in range(10)]
    scorer = MasteryScorer()
    score = scorer.compute(inspection, bad, test_id="t1")
    assert score.total_score < 75.0


def test_status_boundaries() -> None:
    def _score_with_total(t: float) -> str:
        s = MasteryScore.__new__(MasteryScore)
        s.paper_id = "p"
        s.test_id = "t"
        s.parse_score = t * 0.1
        s.metadata_score = t * 0.05
        s.chunk_quality_score = t * 0.15
        s.index_score = t * 0.10
        s.retrieval_score = t * 0.15
        s.citation_score = t * 0.15
        s.grounding_score = t * 0.15
        s.abstention_score = t * 0.10
        s.formula_argument_score = t * 0.05
        return s.final_status

    # Just test that MasteryScore.final_status works with varied totals
    s90 = MasteryScore(
        paper_id="p",
        test_id="t",
        parse_score=10,
        metadata_score=5,
        chunk_quality_score=15,
        index_score=10,
        retrieval_score=15,
        citation_score=15,
        grounding_score=15,
        abstention_score=10,
        formula_argument_score=5,
    )
    assert s90.final_status == "learned"

    s75 = MasteryScore(
        paper_id="p",
        test_id="t",
        parse_score=8,
        metadata_score=4,
        chunk_quality_score=12,
        index_score=8,
        retrieval_score=11,
        citation_score=11,
        grounding_score=11,
        abstention_score=7,
        formula_argument_score=3,
    )
    assert s75.final_status in ("usable_needs_review", "learned")


def test_hallucination_penalty() -> None:
    inspection = _make_inspection()
    hallucinated = [_make_answer(hallucination_detected=True) for _ in range(5)]
    clean = [_make_answer() for _ in range(5)]
    scorer = MasteryScorer()
    score_with = scorer.compute(inspection, hallucinated + clean, test_id="t1")
    score_without = scorer.compute(inspection, clean * 2, test_id="t2")
    assert score_with.total_score <= score_without.total_score


def test_abstention_scoring() -> None:
    inspection = _make_inspection()
    good_abstention = _make_answer(requires_abstention=True, abstention_correct=True)
    bad_abstention = _make_answer(requires_abstention=True, abstention_correct=False)
    scorer = MasteryScorer()
    score_good = scorer.compute(inspection, [good_abstention] * 5, test_id="t1")
    score_bad = scorer.compute(inspection, [bad_abstention] * 5, test_id="t2")
    assert score_good.abstention_score >= score_bad.abstention_score
