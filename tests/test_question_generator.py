"""QuestionGenerator birim testleri."""

from __future__ import annotations

from pathlib import Path

from app.learning.question_generator import MasteryQuestion, QuestionGenerator
from app.memory.sqlite_store import SqliteStore


def _store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(db_path=tmp_path / "test.db")


def _seed_paper(store: SqliteStore, paper_id: str) -> None:
    store.upsert_paper(
        paper_id=paper_id,
        file_hash=f"h_{paper_id}",
        source_path=f"/tmp/{paper_id}.pdf",
        title="Momentum Strategies in High Volatility",
        year="2023",
    )


def test_generate_returns_questions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_paper(store, "p1")
    gen = QuestionGenerator(store=store)
    qs = gen.generate("p1", test_id="t1", count=10)
    assert len(qs) >= 1
    assert all(isinstance(q, MasteryQuestion) for q in qs)


def test_respects_count_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_paper(store, "p2")
    gen = QuestionGenerator(store=store)
    qs = gen.generate("p2", test_id="t2", count=3)
    assert len(qs) <= 3


def test_abstention_questions_generated(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_paper(store, "p3")
    gen = QuestionGenerator(store=store)
    qs = gen.generate("p3", test_id="t3", count=20)
    abstention_qs = [q for q in qs if q.requires_abstention]
    assert len(abstention_qs) >= 1


def test_no_duplicate_questions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_paper(store, "p4")
    gen = QuestionGenerator(store=store)
    qs = gen.generate("p4", test_id="t4", count=20)
    texts = [q.question_text for q in qs]
    assert len(texts) == len(set(texts))


def test_card_hypotheses_included(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_paper(store, "p5")
    store.save_knowledge_card(
        card_id="card_p5",
        paper_id="p5",
        model="test",
        card={
            "paper_id": "p5",
            "title": "T",
            "main_claim": "Momentum çalışır",
            "possible_strategy_hypotheses": ["RSI momentum düşük vol"],
        },
    )
    gen = QuestionGenerator(store=store)
    qs = gen.generate("p5", test_id="t5", count=20)
    trading_qs = [q for q in qs if q.question_type == "trading_hypothesis"]
    assert len(trading_qs) >= 1


def test_empty_for_unknown_paper(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gen = QuestionGenerator(store=store)
    qs = gen.generate("ghost_paper", test_id="t0", count=10)
    assert qs == []
