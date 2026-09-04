"""store.py — Sentinel sağlık-kontrol geçmişinin kalıcı (SQLite) deposu.

`rlm_store.py` / `orchestration/store.py` desenini izler: aynı SQLite dosyasını paylaşır,
kendi engine + DeclarativeBase'ini kurar, her bağlantıda WAL + busy_timeout uygular
(paylaşılan dosyada TÜM yazıcılar açmalı — bkz memory sqlite-shared-file-wal-pragmas).

Tablo: sentinel_checks — her nöbetçi koşusunun bütünsel verdicti + probe detayları.
Budama: UI periyodik yenilediği için sınırsız büyümesin → her eklemede en yeni
`keep_last` kayıt korunur (agent_events'in budama gerekçesiyle aynı).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import String, Text, create_engine, delete, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings

log = logging.getLogger(__name__)

_KEEP_LAST_DEFAULT = 1000


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


def _new_id() -> str:
    return "sen_" + uuid.uuid4().hex[:16]


class MonitoringBase(DeclarativeBase):
    pass


class SentinelCheck(MonitoringBase):
    __tablename__ = "sentinel_checks"

    check_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    overall: Mapped[str] = mapped_column(String(16), default="skip", index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    probes_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow, index=True)


class MonitoringStore:
    """Sentinel geçmişine erişim deposu (kayıt + liste + budama)."""

    def __init__(
        self, db_path: str | Path | None = None, *, keep_last: int = _KEEP_LAST_DEFAULT
    ) -> None:
        path = db_path or get_settings().sqlite_file
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(1, keep_last)
        self._engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, connect_args={"timeout": 30.0}
        )
        event.listen(self._engine, "connect", _sqlite_pragmas)
        MonitoringBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._Session() as s:
            yield s
            s.commit()

    def record(self, *, overall: str, summary: str, probes: list[dict[str, Any]]) -> str:
        """Bir nöbetçi koşusunu kaydet; en yeni `keep_last` dışındakileri buda."""
        check_id = _new_id()
        with self.session() as s:
            s.add(
                SentinelCheck(
                    check_id=check_id,
                    overall=overall,
                    summary=summary,
                    probes_json=json.dumps(probes, ensure_ascii=False, default=str),
                    created_at=utcnow(),
                )
            )
            # Budama: en yeni keep_last kaydı koru (created_at ISO8601 → leksikografik sıralı).
            # cutoff = keep_last'ıncı en yeni kaydın damgası (offset keep_last-1); altındakiler
            # KESİN küçüktür (<) ile silinir → farklı damgalarda tam keep_last kalır.
            cutoff_row = (
                s.execute(
                    select(SentinelCheck.created_at)
                    .order_by(SentinelCheck.created_at.desc())
                    .offset(self.keep_last - 1)
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if cutoff_row is not None:
                # KESİN küçüktür (<): aynı-timestamp bağında (mikrosaniye çakışması) sınırdaki
                # kayıtlar — yeni eklenen dahil — SİLİNMEZ; keep_last'tan biraz fazla tutmak
                # güvenli yön, az tutmak veri kaybı (review bulgusu: <= yeni kaydı silebiliyordu).
                s.execute(delete(SentinelCheck).where(SentinelCheck.created_at < cutoff_row))
        return check_id

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = (
                s.execute(
                    select(SentinelCheck)
                    .order_by(SentinelCheck.created_at.desc())
                    .limit(max(1, limit))
                )
                .scalars()
                .all()
            )
            return [_to_dict(r) for r in rows]


def _to_dict(r: SentinelCheck) -> dict[str, Any]:
    try:
        probes = json.loads(r.probes_json or "[]")
    except json.JSONDecodeError:
        # Bozuk kayıt "hiç probe yok"tan ayırt edilebilsin diye logla (gözlemlenebilirlik).
        log.warning(
            "MonitoringStore: bozuk probes_json (check_id=%s) — boş listeye düşüldü", r.check_id
        )
        probes = []
    return {
        "check_id": r.check_id,
        "overall": r.overall,
        "summary": r.summary,
        "probes": probes,
        "created_at": r.created_at,
    }
