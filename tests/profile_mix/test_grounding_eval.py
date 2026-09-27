"""Grounding / citation (chunk_id eşleşmesi) + halüsinasyon + judge sözleşmesi."""

from __future__ import annotations

import pytest

from app.evals.profile.abstention_eval import evaluate_abstention
from app.evals.profile.grounding_eval import evaluate_grounding, parse_citations
from app.evals.profile.judge import JudgeVerdict, parse_judge_output
from app.evals.profile.schema import EvalItem

RAG_ITEM = EvalItem(
    id="g", domain="reasoning", question="q", requires_rag=True, expected_chunk_ids=["c1"]
)


def test_parse_citations_with_page_suffix() -> None:
    assert parse_citations("bkz [p1:c1, s.3] ve [p2:c7]") == [("p1", "c1"), ("p2", "c7")]


def test_grounded_and_correct_citation() -> None:
    r = evaluate_grounding(RAG_ITEM, "X doğrudur [p1:c1].", ["c1", "c2"])
    assert r["grounding"] == 1.0 and r["citation_accuracy"] == 1.0 and r["fabricated"] == 0


def test_retrieved_but_wrong_chunk_cited() -> None:
    r = evaluate_grounding(RAG_ITEM, "X [p2:c2].", ["c1", "c2"])
    assert r["grounding"] == 1.0 and r["citation_accuracy"] == 0.0


def test_fabricated_citation_counts_as_hallucination() -> None:
    r = evaluate_grounding(RAG_ITEM, "X [p9:c99] ve [p1:c1].", ["c1"])
    assert r["grounding"] == 0.5 and r["fabricated"] == 1
    ab = evaluate_abstention(RAG_ITEM, "X [p9:c99]", fabricated_citations=r["fabricated"])
    assert ab["hallucinated"] is True


def test_rag_required_but_no_citation_is_ungrounded() -> None:
    r = evaluate_grounding(RAG_ITEM, "X doğrudur.", ["c1"])
    assert r["grounding"] == 0.0 and r["citation_accuracy"] == 0.0


def test_non_rag_system_has_no_grounding_score() -> None:
    assert evaluate_grounding(RAG_ITEM, "X [p1:c1]", None)["grounding"] is None


def test_abstention_scoring() -> None:
    must = EvalItem(id="a", domain="trading", question="q", requires_abstention=True)
    assert evaluate_abstention(must, "Kaynaklarda bu bilgi yok.")["abstention_correct"]
    uyd = evaluate_abstention(must, "Yarın 42 olacak.")
    assert uyd["hallucinated"] and not uyd["abstention_correct"]
    normal = EvalItem(id="n", domain="trading", question="q")
    assert not evaluate_abstention(normal, "Bilmiyorum.")[
        "abstention_correct"
    ]  # gereksiz çekimserlik


def test_judge_stores_structured_fields_not_only_score() -> None:
    raw = (
        'bla {"score": 0.7, "decision": "PARTIAL", "supported_claims": ["a"], '
        '"unsupported_claims": ["b"], "contradicted_claims": [], "evidence_chunk_ids": ["c1"], '
        '"short_justification": "' + "x" * 1000 + '", "can_promote": true}'
    )
    v = parse_judge_output(raw, aspect="reasoning_quality", judge_model="qwen")
    assert v.decision == "PARTIAL" and v.unsupported_claims == ["b"]
    assert v.evidence_chunk_ids == ["c1"] and len(v.short_justification) == 300
    assert v.can_promote is False  # judge tek başına terfi veremez


def test_judge_not_allowed_for_deterministic_domains() -> None:
    with pytest.raises(ValueError):
        JudgeVerdict(aspect="math_accuracy", score=1.0, decision="PASS")
