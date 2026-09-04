"""rlm_store.py — RLM koşu (run) loglarını tutan ORM modelleri ve erişim katmanı.

Talimat §6'daki rlm_runs / rlm_steps / rlm_evidence / rlm_verifications tablolarını
karşılar. Aynı SQLite dosyasını SqliteStore ile paylaşır; kendi engine'ini oluşturur
(mastery_store.py ile aynı desen).
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Float, Integer, String, Text, create_engine, event, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


def _sqlite_pragmas(dbapi_conn: object, _record: object) -> None:
    """Her bağlantıda WAL + busy_timeout. Aynı SQLite dosyası RAG öğrenme döngüsü
    (SqliteStore) ile paylaşıldığından eşzamanlı yazımda 'database is locked' riski var;
    WAL eşzamanlı okur+tek-yazıcıya izin verir, busy_timeout kilit beklemesini 5sn'den
    30sn'ye çıkarır (yavaş CPU'da kart yazımı sürerken RLM run'u düşmesin)."""
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


class RlmBase(DeclarativeBase):
    pass


class RlmRun(RlmBase):
    __tablename__ = "rlm_runs"

    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_query: Mapped[str] = mapped_column(Text)
    task_type: Mapped[str] = mapped_column(String(40), default="general_paper_question")
    model_name: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(32), default="running")
    # running / answered / answered_with_limitation / abstained / no_llm / failed
    final_answer: Mapped[str] = mapped_column(Text, default="")
    final_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    report_path: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RlmStep(RlmBase):
    __tablename__ = "rlm_steps"

    step_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    step_order: Mapped[int] = mapped_column(Integer, default=0)
    step_type: Mapped[str] = mapped_column(String(40))
    # classify / plan / retrieval / evidence / draft / verify / contradiction / synthesize
    input_text: Mapped[str] = mapped_column(Text, default="")
    output_text: Mapped[str] = mapped_column(Text, default="")
    tool_used: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RlmEvidence(RlmBase):
    __tablename__ = "rlm_evidence"

    evidence_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    paper_id: Mapped[str] = mapped_column(String(64), default="")
    chunk_id: Mapped[str] = mapped_column(String(80), default="")
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    used_in_final_answer: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RlmVerification(RlmBase):
    __tablename__ = "rlm_verifications"

    verification_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    supported_claims_json: Mapped[str] = mapped_column(Text, default="[]")
    unsupported_claims_json: Mapped[str] = mapped_column(Text, default="[]")
    contradiction_json: Mapped[str] = mapped_column(Text, default="[]")
    citation_score: Mapped[float] = mapped_column(Float, default=0.0)
    grounding_score: Mapped[float] = mapped_column(Float, default=0.0)
    context_sufficiency_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_decision: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RlmStore:
    """RLM koşu loglarına erişim sağlayan depo sınıfı."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = db_path or get_settings().sqlite_file
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, connect_args={"timeout": 30.0}
        )
        event.listen(self._engine, "connect", _sqlite_pragmas)
        RlmBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._Session() as s:
            yield s
            s.commit()

    # ── Runs ────────────────────────────────────────────────────────────────

    def create_run(self, user_query: str, task_type: str, model_name: str) -> str:
        run_id = _new_id("rlm_")
        with self.session() as s:
            s.add(
                RlmRun(
                    run_id=run_id,
                    user_query=user_query,
                    task_type=task_type,
                    model_name=model_name,
                )
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        final_answer: str,
        final_confidence: float,
        evidence_score: float,
        report_path: str | None = None,
    ) -> None:
        with self.session() as s:
            row = s.get(RlmRun, run_id)
            if row:
                row.status = status
                row.final_answer = final_answer
                row.final_confidence = final_confidence
                row.evidence_score = evidence_score
                row.report_path = report_path

    def mark_stale_running_failed(self, max_age_minutes: int = 60) -> int:
        """`max_age_minutes`'tan uzun süredir 'running' kalmış run'ları 'failed' yap (reaper).

        Controller'daki try/except yalnız YAKALANABİLİR hataları 'failed' işaretler; dış
        kill / timeout / crash ile yarıda kalan run sonsuza dek 'running' asılı kalırdı
        (gerçek DB'de gözlendi). Bu reaper o artıkları temizler → rlm-runs/audit gerçeği
        yansıtsın. created_at ISO8601-UTC (hepsi _utcnow) → leksikografik karşılaştırma
        doğru. Yaş eşiği aktif/eşzamanlı run'ları korur (genç 'running' dokunulmaz)."""
        cutoff = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=max_age_minutes)).isoformat()
        with self.session() as s:
            rows = s.scalars(
                select(RlmRun).where(RlmRun.status == "running", RlmRun.created_at < cutoff)
            ).all()
            for r in rows:
                r.status = "failed"
                if not r.final_answer:
                    r.final_answer = (
                        "[yarıda kaldı — süre aşımı/kesinti; reaper 'failed' işaretledi]"
                    )
            return len(rows)

    # ── Steps ─────────────────────────────────────────────────────────────────

    def add_step(
        self,
        run_id: str,
        step_order: int,
        step_type: str,
        *,
        input_text: str = "",
        output_text: str = "",
        tool_used: str = "",
    ) -> None:
        with self.session() as s:
            s.add(
                RlmStep(
                    step_id=_new_id("s_"),
                    run_id=run_id,
                    step_order=step_order,
                    step_type=step_type,
                    input_text=input_text[:4000],
                    output_text=output_text[:4000],
                    tool_used=tool_used,
                )
            )

    # ── Evidence ──────────────────────────────────────────────────────────────

    def add_evidence(
        self,
        run_id: str,
        paper_id: str,
        chunk_id: str,
        relevance_score: float,
        used_in_final_answer: bool,
    ) -> None:
        with self.session() as s:
            s.add(
                RlmEvidence(
                    evidence_id=_new_id("e_"),
                    run_id=run_id,
                    paper_id=paper_id,
                    chunk_id=chunk_id,
                    relevance_score=relevance_score,
                    used_in_final_answer=1 if used_in_final_answer else 0,
                )
            )

    # ── Verification ────────────────────────────────────────────────────────────

    def set_verification(
        self,
        run_id: str,
        *,
        supported_claims: list[str],
        unsupported_claims: list[str],
        contradictions: list[str],
        citation_score: float,
        grounding_score: float,
        context_sufficiency_score: float,
        final_decision: str,
    ) -> None:
        with self.session() as s:
            s.add(
                RlmVerification(
                    verification_id=_new_id("v_"),
                    run_id=run_id,
                    supported_claims_json=json.dumps(supported_claims, ensure_ascii=False),
                    unsupported_claims_json=json.dumps(unsupported_claims, ensure_ascii=False),
                    contradiction_json=json.dumps(contradictions, ensure_ascii=False),
                    citation_score=citation_score,
                    grounding_score=grounding_score,
                    context_sufficiency_score=context_sufficiency_score,
                    final_decision=final_decision,
                )
            )

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(RlmRun, run_id)
            if not row:
                return None
            return _run_to_dict(row)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(select(RlmRun).order_by(RlmRun.created_at.desc()).limit(limit)).all()
            return [_run_to_dict(r) for r in rows]

    def count_runs(self) -> int:
        """Toplam RLM koşu sayısı (aday seçiminde sessiz-kesme tespiti için)."""
        with self.session() as s:
            return int(s.scalar(select(func.count()).select_from(RlmRun)) or 0)

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(RlmStep).where(RlmStep.run_id == run_id).order_by(RlmStep.step_order)
            ).all()
            return [
                {
                    "step_order": r.step_order,
                    "step_type": r.step_type,
                    "input_text": r.input_text,
                    "output_text": r.output_text,
                    "tool_used": r.tool_used,
                }
                for r in rows
            ]

    def get_evidence(self, run_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(select(RlmEvidence).where(RlmEvidence.run_id == run_id)).all()
            return [
                {
                    "paper_id": r.paper_id,
                    "chunk_id": r.chunk_id,
                    "relevance_score": r.relevance_score,
                    "used_in_final_answer": bool(r.used_in_final_answer),
                }
                for r in rows
            ]

    def get_verification(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.scalar(select(RlmVerification).where(RlmVerification.run_id == run_id))
            if not row:
                return None
            return {
                "supported_claims": json.loads(row.supported_claims_json),
                "unsupported_claims": json.loads(row.unsupported_claims_json),
                "contradictions": json.loads(row.contradiction_json),
                "citation_score": row.citation_score,
                "grounding_score": row.grounding_score,
                "context_sufficiency_score": row.context_sufficiency_score,
                "final_decision": row.final_decision,
            }


def _run_to_dict(r: RlmRun) -> dict[str, Any]:
    return {
        "run_id": r.run_id,
        "user_query": r.user_query,
        "task_type": r.task_type,
        "model_name": r.model_name,
        "status": r.status,
        "final_answer": r.final_answer,
        "final_confidence": r.final_confidence,
        "evidence_score": r.evidence_score,
        "report_path": r.report_path,
        "created_at": r.created_at,
    }
