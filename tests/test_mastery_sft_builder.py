"""MasterySFTBuilder birim testleri."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from app.training.mastery_sft_builder import MasterySFTBuilder


def _make_paper(paper_id: str = "p1") -> MagicMock:
    p = MagicMock()
    p.paper_id = paper_id
    return p


def _make_score(total: float, test_id: str = "t1") -> dict:
    return {"total_score": total, "test_id": test_id, "final_status": "learned"}


def _make_question(qid: str, qtype: str = "structural") -> dict:
    return {
        "question_id": qid,
        "question_text": f"Soru {qid}?",
        "question_type": qtype,
    }


def _make_answer(qid: str, passed: bool = True, citation: float = 0.8) -> dict:
    return {
        "question_id": qid,
        "answer_text": f"Cevap {qid}.",
        "passed": passed,
        "citation_score": citation,
    }


def _builder(sqlite_mock: MagicMock, mastery_mock: MagicMock) -> MasterySFTBuilder:
    return MasterySFTBuilder(sqlite_store=sqlite_mock, mastery_store=mastery_mock)


def test_collect_returns_examples_above_threshold() -> None:
    sq = MagicMock()
    sq.list_papers.return_value = [_make_paper("p1")]
    ms = MagicMock()
    ms.get_latest_score.return_value = _make_score(85.0, "t1")
    ms.list_questions.return_value = [_make_question("q1")]
    ms.list_answers.return_value = [_make_answer("q1")]
    examples = _builder(sq, ms).collect(min_mastery_score=75.0)
    assert len(examples) == 1
    assert examples[0].source == "mastery:p1"
    assert examples[0].quality_score == 0.8


def test_skips_papers_below_min_score() -> None:
    sq = MagicMock()
    sq.list_papers.return_value = [_make_paper("p1")]
    ms = MagicMock()
    ms.get_latest_score.return_value = _make_score(60.0, "t1")
    examples = _builder(sq, ms).collect(min_mastery_score=75.0)
    assert examples == []


def test_skips_failed_answers() -> None:
    sq = MagicMock()
    sq.list_papers.return_value = [_make_paper("p1")]
    ms = MagicMock()
    ms.get_latest_score.return_value = _make_score(90.0, "t1")
    ms.list_questions.return_value = [_make_question("q1")]
    ms.list_answers.return_value = [_make_answer("q1", passed=False)]
    examples = _builder(sq, ms).collect()
    assert examples == []


def test_skips_low_citation() -> None:
    sq = MagicMock()
    sq.list_papers.return_value = [_make_paper("p1")]
    ms = MagicMock()
    ms.get_latest_score.return_value = _make_score(90.0, "t1")
    ms.list_questions.return_value = [_make_question("q1")]
    ms.list_answers.return_value = [_make_answer("q1", citation=0.2)]
    examples = _builder(sq, ms).collect(citation_threshold=0.5)
    assert examples == []


def test_build_jsonl_creates_file(tmp_path: Path) -> None:
    sq = MagicMock()
    sq.list_papers.return_value = [_make_paper("p1")]
    ms = MagicMock()
    ms.get_latest_score.return_value = _make_score(90.0, "t1")
    ms.list_questions.return_value = [_make_question("q1")]
    ms.list_answers.return_value = [_make_answer("q1")]
    out = tmp_path / "out.jsonl"
    path, n = _builder(sq, ms).build_jsonl(output_path=out)
    assert path.exists()
    assert n == 1
    import json

    data = json.loads(path.read_text())
    assert data["source"] == "mastery:p1"


def test_instruction_varies_by_type() -> None:
    sq = MagicMock()
    sq.list_papers.return_value = [_make_paper("p1")]
    ms = MagicMock()
    ms.get_latest_score.return_value = _make_score(90.0, "t1")
    ms.list_questions.return_value = [_make_question("q1", "trading_hypothesis")]
    ms.list_answers.return_value = [_make_answer("q1")]
    examples = _builder(sq, ms).collect()
    assert (
        "trading" in examples[0].instruction.lower() or "hipotez" in examples[0].instruction.lower()
    )
