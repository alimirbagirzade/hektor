"""store.py — kullanıcı düzeltme (feedback) kayıtlarının kalıcı (SQLite) deposu.

`rlm_store.py` / `app/orchestration/store.py` desenini izler: aynı SQLite dosyasını
paylaşır, kendi engine + DeclarativeBase'ini kurar, her bağlantıda WAL + busy_timeout uygular.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import String, Text, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


def _sqlite_pragmas(dbapi_conn: object, _record: object) -> None:
    """Her bağlantıda WAL + busy_timeout (paylaşılan SQLite'ta eşzamanlı yazım kilidi)."""
    cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
    finally:
        cur.close()


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _new_id(prefix: str = "fb_") -> str:
    return prefix + uuid.uuid4().hex[:16]


class FeedbackBase(DeclarativeBase):
    pass


class FeedbackCorrection(FeedbackBase):
    __tablename__ = "feedback_corrections"

    correction_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), default="manual")
    # rlm / eval / backtest / manual / <run_id veya bağlam>
    question: Mapped[str] = mapped_column(Text, default="")
    bad_answer: Mapped[str] = mapped_column(Text, default="")
    correction: Mapped[str] = mapped_column(Text, default="")
    correction_type: Mapped[str] = mapped_column(String(40), default="other")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    # pending / approved / rejected / exported
    reject_reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)
    updated_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class FeedbackStore:
    """Feedback düzeltme kayıtlarına erişim deposu."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = db_path or get_settings().sqlite_file
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, connect_args={"timeout": 30.0}
        )
        event.listen(self._engine, "connect", _sqlite_pragmas)
        FeedbackBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._Session() as s:
            yield s
            s.commit()

    def add(
        self,
        *,
        source: str,
        question: str,
        bad_answer: str,
        correction: str,
        correction_type: str = "other",
        status: str = "pending",
        reject_reason: str = "",
    ) -> str:
        correction_id = _new_id()
        now = utcnow()
        with self.session() as s:
            s.add(
                FeedbackCorrection(
                    correction_id=correction_id,
                    source=source,
                    question=question,
                    bad_answer=bad_answer,
                    correction=correction,
                    correction_type=correction_type,
                    status=status,
                    reject_reason=reject_reason,
                    created_at=now,
                    updated_at=now,
                )
            )
        return correction_id

    def get(self, correction_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(FeedbackCorrection, correction_id)
            return _to_dict(row) if row else None

    def list(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.session() as s:
            stmt = select(FeedbackCorrection)
            if status:
                stmt = stmt.where(FeedbackCorrection.status == status)
            stmt = stmt.order_by(FeedbackCorrection.created_at.desc()).limit(max(1, limit))
            rows = s.execute(stmt).scalars().all()
            return [_to_dict(r) for r in rows]

    def set_status(self, correction_id: str, status: str, reason: str = "") -> bool:
        with self.session() as s:
            row = s.get(FeedbackCorrection, correction_id)
            if row is None:
                return False
            row.status = status
            if reason:
                row.reject_reason = reason
            row.updated_at = utcnow()
            return True

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self.session() as s:
            rows = s.execute(select(FeedbackCorrection.status)).scalars().all()
            for st in rows:
                out[st] = out.get(st, 0) + 1
        return out


def _to_dict(r: FeedbackCorrection) -> dict[str, Any]:
    return {
        "correction_id": r.correction_id,
        "source": r.source,
        "question": r.question,
        "bad_answer": r.bad_answer,
        "correction": r.correction,
        "correction_type": r.correction_type,
        "status": r.status,
        "reject_reason": r.reject_reason,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
    }
