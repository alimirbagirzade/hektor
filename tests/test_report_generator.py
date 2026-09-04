"""ReportGenerator birim testleri."""

from __future__ import annotations

import json
from pathlib import Path

from app.learning.mastery_scorer import MasteryScore
from app.learning.report_generator import ReportGenerator
from app.memory.mastery_store import MasteryStore


def _gen(tmp_path: Path) -> ReportGenerator:
    store = MasteryStore(db_path=tmp_path / "mastery.db")
    return ReportGenerator(store=store, report_dir=tmp_path / "reports")


def _score(paper_id: str = "p1", test_id: str = "t1") -> MasteryScore:
    return MasteryScore(
        paper_id=paper_id,
        test_id=test_id,
        parse_score=8.0,
        metadata_score=4.0,
        chunk_quality_score=12.0,
        index_score=8.0,
        retrieval_score=10.0,
        citation_score=10.0,
        grounding_score=10.0,
        abstention_score=7.0,
        formula_argument_score=4.0,
    )


def test_generates_json_and_md(tmp_path: Path) -> None:
    json_path, md_path = _gen(tmp_path).generate("p1", "t1", _score())
    assert json_path.exists()
    assert md_path.exists()


def test_json_valid_structure(tmp_path: Path) -> None:
    json_path, _ = _gen(tmp_path).generate("p1", "t1", _score())
    data = json.loads(json_path.read_text())
    assert data["paper_id"] == "p1"
    assert "score" in data
    assert "total_score" in data["score"]
    assert "final_status" in data["score"]


def test_md_contains_paper_id_and_score(tmp_path: Path) -> None:
    _, md_path = _gen(tmp_path).generate("p1", "t1", _score())
    content = md_path.read_text()
    assert "p1" in content
    assert "73" in content or "Toplam Skor" in content


def test_idempotent_overwrite(tmp_path: Path) -> None:
    gen = _gen(tmp_path)
    j1, _ = gen.generate("p1", "t1", _score())
    j2, _ = gen.generate("p1", "t1", _score())
    assert j1 == j2


def test_different_papers_different_files(tmp_path: Path) -> None:
    gen = _gen(tmp_path)
    j1, _ = gen.generate("p1", "t1", _score("p1", "t1"))
    j2, _ = gen.generate("p2", "t2", _score("p2", "t2"))
    assert j1 != j2
    assert j1.exists() and j2.exists()
