"""mastery_store.py — Paper Mastery verilerini tutan ORM modelleri ve erişim katmanı.

Aynı SQLite dosyasını SqliteStore ile paylaşır; kendi engine'ini oluşturur.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Float, Integer, String, Text, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


def _sqlite_pragmas(dbapi_conn: object, _record: object) -> None:
    """Her bağlantıda WAL + busy_timeout. Aynı SQLite dosyası RAG öğrenme döngüsü
    (SqliteStore), RLM koşuları (RlmStore) ve diğer store'larla paylaşıldığından
    eşzamanlı yazımda 'database is locked' riski var; WAL eşzamanlı okur+tek-yazıcıya
    izin verir, busy_timeout kilit beklemesini 30sn'ye çıkarır (yavaş CPU'da kart
    yazımı sürerken mastery testi düşmesin). WAL kalıcı/per-veritabanı; busy_timeout
    per-bağlantı olduğundan TÜM yazıcılar açmalı — bu yüzden mastery_store da açar
    (rlm_store._sqlite_pragmas ile birebir aynı desen)."""
    cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
    finally:
        cur.close()


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:16]


class MasteryBase(DeclarativeBase):
    pass


class PaperLearningQueue(MasteryBase):
    __tablename__ = "paper_learning_queue"

    queue_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    priority: Mapped[int] = mapped_column(Integer, default=5)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    last_error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)
    updated_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class PaperMasteryTest(MasteryBase):
    __tablename__ = "paper_mastery_tests"

    test_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(64), index=True)
    started_at: Mapped[str] = mapped_column(String(40), default=_utcnow)
    finished_at: Mapped[str | None] = mapped_column(String(40), default=None)
    status: Mapped[str] = mapped_column(String(32), default="running")
    question_count: Mapped[int] = mapped_column(Integer, default=0)
    passed_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    report_path: Mapped[str | None] = mapped_column(Text, default=None)


class PaperMasteryQuestion(MasteryBase):
    __tablename__ = "paper_mastery_questions"

    question_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    test_id: Mapped[str] = mapped_column(String(40), index=True)
    paper_id: Mapped[str] = mapped_column(String(64), index=True)
    question_text: Mapped[str] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String(40))
    expected_chunk_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    requires_abstention: Mapped[int] = mapped_column(Integer, default=0)
    difficulty: Mapped[str] = mapped_column(String(16), default="medium")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class PaperMasteryAnswer(MasteryBase):
    __tablename__ = "paper_mastery_answers"

    answer_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    question_id: Mapped[str] = mapped_column(String(40), index=True)
    test_id: Mapped[str] = mapped_column(String(40), index=True)
    paper_id: Mapped[str] = mapped_column(String(64))
    answer_text: Mapped[str] = mapped_column(Text, default="")
    cited_paper_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    citation_score: Mapped[float] = mapped_column(Float, default=0.0)
    grounding_score: Mapped[float] = mapped_column(Float, default=0.0)
    context_sufficient: Mapped[int] = mapped_column(Integer, default=0)
    abstention_correct: Mapped[int] = mapped_column(Integer, default=0)
    hallucination_detected: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class PaperMasteryScore(MasteryBase):
    __tablename__ = "paper_mastery_scores"

    score_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(64), index=True)
    test_id: Mapped[str] = mapped_column(String(40))
    total_score: Mapped[float] = mapped_column(Float, default=0.0)
    parse_score: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_score: Mapped[float] = mapped_column(Float, default=0.0)
    chunk_quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    index_score: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_score: Mapped[float] = mapped_column(Float, default=0.0)
    citation_score: Mapped[float] = mapped_column(Float, default=0.0)
    grounding_score: Mapped[float] = mapped_column(Float, default=0.0)
    abstention_score: Mapped[float] = mapped_column(Float, default=0.0)
    formula_argument_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_status: Mapped[str] = mapped_column(String(40), default="untested")
    report_path: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class PaperStatusHistory(MasteryBase):
    __tablename__ = "paper_status_history"

    history_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(64), index=True)
    old_status: Mapped[str | None] = mapped_column(String(40), default=None)
    new_status: Mapped[str] = mapped_column(String(40))
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class MasteryStore:
    """Paper Mastery verilerine erişim sağlayan depo sınıfı."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = db_path or get_settings().sqlite_file
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, connect_args={"timeout": 30.0}
        )
        event.listen(self._engine, "connect", _sqlite_pragmas)
        MasteryBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._Session() as s:
            yield s
            s.commit()

    # ── Learning Queue ──────────────────────────────────────────────────────

    def enqueue(self, paper_id: str, priority: int = 5) -> str:
        queue_id = _new_id("q_")
        with self.session() as s:
            existing = s.scalar(
                select(PaperLearningQueue).where(PaperLearningQueue.paper_id == paper_id)
            )
            if existing:
                return existing.queue_id
            s.add(PaperLearningQueue(queue_id=queue_id, paper_id=paper_id, priority=priority))
        return queue_id

    def list_queue(self, status: str | None = None) -> list[dict[str, Any]]:
        with self.session() as s:
            q = select(PaperLearningQueue).order_by(
                PaperLearningQueue.priority.desc(), PaperLearningQueue.created_at
            )
            if status:
                q = q.where(PaperLearningQueue.status == status)
            rows = s.scalars(q).all()
            return [_queue_to_dict(r) for r in rows]

    def get_next_queued(self) -> dict[str, Any] | None:
        with self.session() as s:
            # 'failed' de dahil: başarısız ama attempts<max_attempts olan kayıtlar
            # yeniden denenmeli (eskiden yalnız 'pending' seçildiğinden retry ölü koddu —
            # geçici hata = kalıcı atlama). max_attempts kapağı sonsuz döngüyü önler.
            row = s.scalar(
                select(PaperLearningQueue)
                .where(PaperLearningQueue.status.in_(["pending", "failed"]))
                .where(PaperLearningQueue.attempts < PaperLearningQueue.max_attempts)
                .order_by(PaperLearningQueue.priority.desc(), PaperLearningQueue.created_at)
                .limit(1)
            )
            return _queue_to_dict(row) if row else None

    def update_queue_status(self, queue_id: str, status: str, error: str | None = None) -> None:
        with self.session() as s:
            row = s.get(PaperLearningQueue, queue_id)
            if row:
                row.status = status
                row.updated_at = _utcnow()
                if status == "running":
                    row.attempts += 1
                if error:
                    row.last_error = error

    # ── Mastery Tests ────────────────────────────────────────────────────────

    def create_test(self, paper_id: str) -> str:
        test_id = _new_id("t_")
        with self.session() as s:
            s.add(PaperMasteryTest(test_id=test_id, paper_id=paper_id))
        return test_id

    def finish_test(self, test_id: str, passed: int, failed: int) -> None:
        with self.session() as s:
            row = s.get(PaperMasteryTest, test_id)
            if row:
                row.finished_at = _utcnow()
                row.status = "done"
                row.passed_count = passed
                row.failed_count = failed
                row.question_count = passed + failed

    def set_test_report(self, test_id: str, report_path: str) -> None:
        with self.session() as s:
            row = s.get(PaperMasteryTest, test_id)
            if row:
                row.report_path = report_path

    def list_tests(self, paper_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(PaperMasteryTest)
                .where(PaperMasteryTest.paper_id == paper_id)
                .order_by(PaperMasteryTest.started_at.desc())
            ).all()
            return [_test_to_dict(r) for r in rows]

    # ── Questions ────────────────────────────────────────────────────────────

    def save_questions(self, questions: list[dict[str, Any]]) -> None:
        with self.session() as s:
            for q in questions:
                s.merge(
                    PaperMasteryQuestion(
                        question_id=q["question_id"],
                        test_id=q["test_id"],
                        paper_id=q["paper_id"],
                        question_text=q["question_text"],
                        question_type=q["question_type"],
                        expected_chunk_ids_json=json.dumps(q.get("expected_chunk_ids", [])),
                        requires_abstention=int(q.get("requires_abstention", False)),
                        difficulty=q.get("difficulty", "medium"),
                    )
                )

    def list_questions(self, test_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(PaperMasteryQuestion).where(PaperMasteryQuestion.test_id == test_id)
            ).all()
            return [_question_to_dict(r) for r in rows]

    # ── Answers ──────────────────────────────────────────────────────────────

    def save_answer(self, ans: dict[str, Any]) -> None:
        with self.session() as s:
            s.merge(
                PaperMasteryAnswer(
                    answer_id=ans["answer_id"],
                    question_id=ans["question_id"],
                    test_id=ans["test_id"],
                    paper_id=ans["paper_id"],
                    answer_text=ans.get("answer_text", ""),
                    cited_paper_ids_json=json.dumps(ans.get("cited_paper_ids", [])),
                    citation_score=ans.get("citation_score", 0.0),
                    grounding_score=ans.get("grounding_score", 0.0),
                    context_sufficient=int(ans.get("context_sufficient", False)),
                    abstention_correct=int(ans.get("abstention_correct", False)),
                    hallucination_detected=int(ans.get("hallucination_detected", False)),
                    passed=int(ans.get("passed", False)),
                )
            )

    def list_answers(self, test_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(PaperMasteryAnswer).where(PaperMasteryAnswer.test_id == test_id)
            ).all()
            return [_answer_to_dict(r) for r in rows]

    # ── Scores ───────────────────────────────────────────────────────────────

    def save_score(self, score: dict[str, Any]) -> None:
        with self.session() as s:
            s.merge(
                PaperMasteryScore(
                    score_id=score.get("score_id") or _new_id("sc_"),
                    paper_id=score["paper_id"],
                    test_id=score["test_id"],
                    total_score=score.get("total_score", 0.0),
                    parse_score=score.get("parse_score", 0.0),
                    metadata_score=score.get("metadata_score", 0.0),
                    chunk_quality_score=score.get("chunk_quality_score", 0.0),
                    index_score=score.get("index_score", 0.0),
                    retrieval_score=score.get("retrieval_score", 0.0),
                    citation_score=score.get("citation_score", 0.0),
                    grounding_score=score.get("grounding_score", 0.0),
                    abstention_score=score.get("abstention_score", 0.0),
                    formula_argument_score=score.get("formula_argument_score", 0.0),
                    final_status=score.get("final_status", "untested"),
                    report_path=score.get("report_path"),
                )
            )

    def get_latest_score(self, paper_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.scalar(
                select(PaperMasteryScore)
                .where(PaperMasteryScore.paper_id == paper_id)
                .order_by(PaperMasteryScore.created_at.desc())
                .limit(1)
            )
            return _score_to_dict(row) if row else None

    # ── Status History ───────────────────────────────────────────────────────

    def record_status_change(
        self,
        paper_id: str,
        new_status: str,
        old_status: str | None = None,
        reason: str | None = None,
    ) -> None:
        with self.session() as s:
            s.add(
                PaperStatusHistory(
                    history_id=_new_id("h_"),
                    paper_id=paper_id,
                    old_status=old_status,
                    new_status=new_status,
                    reason=reason,
                )
            )

    def get_current_status(self, paper_id: str) -> str:
        with self.session() as s:
            row = s.scalar(
                select(PaperStatusHistory)
                .where(PaperStatusHistory.paper_id == paper_id)
                .order_by(PaperStatusHistory.created_at.desc())
                .limit(1)
            )
            return row.new_status if row else "uploaded"

    def get_status_history(self, paper_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(PaperStatusHistory)
                .where(PaperStatusHistory.paper_id == paper_id)
                .order_by(PaperStatusHistory.created_at.desc())
            ).all()
            return [
                {
                    "history_id": r.history_id,
                    "paper_id": r.paper_id,
                    "old_status": r.old_status,
                    "new_status": r.new_status,
                    "reason": r.reason,
                    "created_at": r.created_at,
                }
                for r in rows
            ]


# ── Private serializers ────────────────────────────────────────────────────


def _queue_to_dict(r: PaperLearningQueue) -> dict[str, Any]:
    return {
        "queue_id": r.queue_id,
        "paper_id": r.paper_id,
        "status": r.status,
        "priority": r.priority,
        "attempts": r.attempts,
        "max_attempts": r.max_attempts,
        "last_error": r.last_error,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
    }


def _test_to_dict(r: PaperMasteryTest) -> dict[str, Any]:
    return {
        "test_id": r.test_id,
        "paper_id": r.paper_id,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
        "status": r.status,
        "question_count": r.question_count,
        "passed_count": r.passed_count,
        "failed_count": r.failed_count,
        "report_path": r.report_path,
    }


def _question_to_dict(r: PaperMasteryQuestion) -> dict[str, Any]:
    return {
        "question_id": r.question_id,
        "test_id": r.test_id,
        "paper_id": r.paper_id,
        "question_text": r.question_text,
        "question_type": r.question_type,
        "expected_chunk_ids": json.loads(r.expected_chunk_ids_json or "[]"),
        "requires_abstention": bool(r.requires_abstention),
        "difficulty": r.difficulty,
        "created_at": r.created_at,
    }


def _answer_to_dict(r: PaperMasteryAnswer) -> dict[str, Any]:
    return {
        "answer_id": r.answer_id,
        "question_id": r.question_id,
        "test_id": r.test_id,
        "paper_id": r.paper_id,
        "answer_text": r.answer_text,
        "cited_paper_ids": json.loads(r.cited_paper_ids_json or "[]"),
        "citation_score": r.citation_score,
        "grounding_score": r.grounding_score,
        "context_sufficient": bool(r.context_sufficient),
        "abstention_correct": bool(r.abstention_correct),
        "hallucination_detected": bool(r.hallucination_detected),
        "passed": bool(r.passed),
        "created_at": r.created_at,
    }


def _score_to_dict(r: PaperMasteryScore) -> dict[str, Any]:
    return {
        "score_id": r.score_id,
        "paper_id": r.paper_id,
        "test_id": r.test_id,
        "total_score": r.total_score,
        "parse_score": r.parse_score,
        "metadata_score": r.metadata_score,
        "chunk_quality_score": r.chunk_quality_score,
        "index_score": r.index_score,
        "retrieval_score": r.retrieval_score,
        "citation_score": r.citation_score,
        "grounding_score": r.grounding_score,
        "abstention_score": r.abstention_score,
        "formula_argument_score": r.formula_argument_score,
        "final_status": r.final_status,
        "report_path": r.report_path,
        "created_at": r.created_at,
    }
