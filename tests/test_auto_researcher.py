"""auto_researcher pipeline birim testleri — çevrimdışı, mock'lu."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.memory.sqlite_store import SqliteStore
from app.pipeline.auto_researcher import PipelineRun, _extract_questions, run_pipeline
from app.training.tool_use_trainer import ToolUseSession


def _store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(db_path=tmp_path / "test.db")


def _seed_approved_card(store: SqliteStore, paper_id: str, hypotheses: list[str]) -> None:
    store.upsert_paper(
        paper_id=paper_id,
        file_hash=f"h_{paper_id}",
        source_path=f"/tmp/{paper_id}.pdf",
        title="Test",
    )
    store.save_knowledge_card(
        card_id=f"card_{paper_id}",
        paper_id=paper_id,
        model="test",
        card={
            "paper_id": paper_id,
            "title": "T",
            "main_claim": "x",
            "possible_strategy_hypotheses": hypotheses,
        },
    )
    store.approve_card(f"card_{paper_id}")


# ---- _extract_questions ----


def test_extract_questions_from_approved_cards(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_approved_card(store, "p1", ["RSI momentum düşük vol", "EMA çaprazlama"])
    qs = _extract_questions(store, max_questions=5, only_approved=True)
    assert len(qs) == 2
    assert all("backtest" in q for q in qs)


def test_extract_questions_max_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_approved_card(store, "p2", [f"hyp_{i}" for i in range(10)])
    qs = _extract_questions(store, max_questions=3, only_approved=True)
    assert len(qs) == 3


def test_extract_questions_deduplicates(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_approved_card(store, "pa", ["aynı hipotez", "aynı hipotez"])
    qs = _extract_questions(store, max_questions=10, only_approved=True)
    assert len(qs) == 1


def test_extract_questions_empty_when_no_cards(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert _extract_questions(store, max_questions=5, only_approved=True) == []


def test_extract_questions_from_unapproved_cards(tmp_path: Path) -> None:
    # only_approved=False artık list_cards (card_json'lu) kullanır → onaysız karttan da
    # soru çıkar (eski list_training_examples card_json içermeyip sessizce 0 soru veriyordu).
    store = _store(tmp_path)
    store.upsert_paper(paper_id="pu", file_hash="h_pu", source_path="/tmp/pu.pdf", title="T")
    store.save_knowledge_card(
        card_id="card_pu",
        paper_id="pu",
        model="test",
        card={"paper_id": "pu", "title": "T", "possible_strategy_hypotheses": ["onaysız hipotez"]},
    )  # approve_card YOK → pending
    assert len(_extract_questions(store, max_questions=5, only_approved=False)) == 1
    assert _extract_questions(store, max_questions=5, only_approved=True) == []


# ---- PipelineRun.summary ----


def test_pipeline_run_summary_fields() -> None:
    run = PipelineRun(n_cards_scanned=3, n_questions=2, n_sessions=2, n_scored=2)
    s = run.summary()
    assert "Kart: 3" in s
    assert "Seans: 2" in s


# ---- run_pipeline dry_run ----


def test_run_pipeline_dry_run_no_sessions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_approved_card(store, "pb", ["momentum filtresi"])
    run = run_pipeline(store=store, max_questions=5, dry_run=True)
    assert run.n_questions == 1
    assert run.n_sessions == 0


def test_run_pipeline_no_cards_returns_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = run_pipeline(store=store, max_questions=5)
    assert run.n_questions == 0
    assert run.n_sessions == 0


# ---- run_pipeline with mocked trainer ----


@patch("app.pipeline.auto_researcher.ToolUseTrainer")
@patch("app.pipeline.auto_researcher.score_and_save_sessions")
def test_run_pipeline_calls_trainer(mock_score, mock_trainer_cls, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_approved_card(store, "pc", ["EMA momentum"])

    mock_session = MagicMock(spec=ToolUseSession)
    mock_session.final_verdict = "pass"
    mock_trainer = MagicMock()
    mock_trainer.run_session.return_value = mock_session
    mock_trainer_cls.return_value = mock_trainer
    mock_score.return_value = [{"session_id": "s1", "label": "chosen"}]

    run = run_pipeline(store=store, max_questions=2)

    assert run.n_sessions == 1
    assert run.n_scored == 1
    mock_trainer.run_session.assert_called_once()


@patch("app.pipeline.auto_researcher.ToolUseTrainer")
@patch("app.pipeline.auto_researcher.score_and_save_sessions")
def test_run_pipeline_handles_session_error(mock_score, mock_trainer_cls, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_approved_card(store, "pd", ["hatalı hipotez"])

    mock_trainer = MagicMock()
    mock_trainer.run_session.side_effect = RuntimeError("backtest patladı")
    mock_trainer_cls.return_value = mock_trainer
    mock_score.return_value = []

    run = run_pipeline(store=store, max_questions=1)

    assert run.n_sessions == 0
    assert len(run.errors) == 1
    assert "backtest patladı" in run.errors[0]
