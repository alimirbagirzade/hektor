"""strategy_store.py — sohbet stratejileri, test koşuları ve dönem erişim kayıtları (SQLite).

Yalnız YENİ tablolar kurar (mevcut ``strategies``/``backtests`` tablolarına dokunmaz):
- ``chat_strategies``     : değişmez strateji tanımları (içerik özetinden kimlik; düzenleme
                            yeni kimlik + aynı aile + ``parent_id``).
- ``strategy_runs``       : her test koşusu (dönem, veri özeti, motor/metrik sürümü, parmak izi).
- ``period_protocols``    : temiz veri özeti başına SABİT dönem sınırları (geliştirme /
                            doğrulama / final). İlk kullanımda belirlenir, sonra değişmez.
- ``period_access``       : doğrulama ve final dönemine her bakış (aile, strateji, koşu, zaman).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, String, Text, create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings
from app.feedback.chat_store import _sqlite_pragmas, dumps, loads, new_id, utcnow


class StratBase(DeclarativeBase):
    pass


class ChatStrategy(StratBase):
    __tablename__ = "chat_strategies"

    strategy_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    family_id: Mapped[str] = mapped_column(String(40), index=True)
    parent_id: Mapped[str] = mapped_column(String(40), default="")
    name: Mapped[str] = mapped_column(String(200), default="")
    spec_json: Mapped[str] = mapped_column(Text, default="{}")
    source_json: Mapped[str] = mapped_column(Text, default="{}")
    origin: Mapped[str] = mapped_column(String(20), default="draft")  # draft|edit|simplified
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class StrategyRun(StratBase):
    __tablename__ = "strategy_runs"

    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(40), index=True)
    family_id: Mapped[str] = mapped_column(String(40), index=True)
    stage: Mapped[str] = mapped_column(String(16))  # gelistirme | dogrulama | final
    clean_sha256: Mapped[str] = mapped_column(String(64), index=True)
    data_report: Mapped[str] = mapped_column(Text, default="{}")
    period_start: Mapped[str] = mapped_column(String(40), default="")
    period_end: Mapped[str] = mapped_column(String(40), default="")
    oos_status: Mapped[str] = mapped_column(String(40), default="")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    engine_version: Mapped[str] = mapped_column(String(40), default="")
    metrics_version: Mapped[str] = mapped_column(String(10), default="")
    fingerprint: Mapped[str] = mapped_column(String(64), default="")
    report_path: Mapped[str] = mapped_column(Text, default="")
    run_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class PeriodProtocol(StratBase):
    __tablename__ = "period_protocols"

    clean_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    dev_end: Mapped[str] = mapped_column(String(40))  # doğrulama bu damgadan başlar
    val_end: Mapped[str] = mapped_column(String(40))  # final bu damgadan sonra başlar
    n_bars: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class PeriodAccess(StratBase):
    __tablename__ = "period_access"

    access_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    family_id: Mapped[str] = mapped_column(String(40), index=True)
    clean_sha256: Mapped[str] = mapped_column(String(64), index=True)
    stage: Mapped[str] = mapped_column(String(16))
    strategy_id: Mapped[str] = mapped_column(String(40))
    run_id: Mapped[str] = mapped_column(String(40), default="")
    at: Mapped[str] = mapped_column(String(40), default=utcnow)


STRAT_TABLES = tuple(StratBase.metadata.sorted_tables)


def _row(r: Any) -> dict[str, Any]:
    return {c.name: getattr(r, c.name) for c in r.__table__.columns}


class StrategyStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path or get_settings().sqlite_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 30.0})
        event.listen(self._engine, "connect", _sqlite_pragmas)
        StratBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._Session() as s:
            yield s
            s.commit()

    # ── stratejiler ──────────────────────────────────────────────────────────

    def save_strategy(
        self,
        strategy_id: str,
        *,
        family_id: str,
        parent_id: str,
        name: str,
        spec: dict[str, Any],
        source: dict[str, Any],
        origin: str,
    ) -> tuple[dict[str, Any], bool]:
        """Aynı içerik (kimlik) zaten varsa mevcut kayıt döner (değişmez)."""
        existing = self.get_strategy(strategy_id)
        if existing is not None:
            return existing, False
        with self._Session() as s:
            s.add(
                ChatStrategy(
                    strategy_id=strategy_id,
                    family_id=family_id,
                    parent_id=parent_id,
                    name=name,
                    spec_json=dumps(spec),
                    source_json=dumps(source),
                    origin=origin,
                    created_at=utcnow(),
                )
            )
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                got = self.get_strategy(strategy_id)
                assert got is not None
                return got, False
        got = self.get_strategy(strategy_id)
        assert got is not None
        return got, True

    def get_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            r = s.get(ChatStrategy, strategy_id)
            if r is None:
                return None
            d = _row(r)
            d["spec"] = loads(r.spec_json, {})
            d["source"] = loads(r.source_json, {})
            return d

    def family_strategies(self, family_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(
                select(ChatStrategy)
                .where(ChatStrategy.family_id == family_id)
                .order_by(ChatStrategy.created_at)
            ).scalars()
            return [_row(r) for r in rows]

    def strategies_for_turn(self, turn_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(select(ChatStrategy).order_by(ChatStrategy.created_at)).scalars()
            out = []
            for r in rows:
                src = loads(r.source_json, {})
                if src.get("turn_id") == turn_id:
                    d = _row(r)
                    d["spec"] = loads(r.spec_json, {})
                    d["source"] = src
                    out.append(d)
            return out

    # ── dönem protokolü ──────────────────────────────────────────────────────

    def get_protocol(self, clean_sha: str) -> dict[str, Any] | None:
        with self.session() as s:
            r = s.get(PeriodProtocol, clean_sha)
            return _row(r) if r else None

    def ensure_protocol(self, clean_sha: str, dev_end: str, val_end: str, n_bars: int) -> dict:
        """İlk kayıt kalıcıdır; sonraki çağrılar mevcut sınırları döndürür."""
        got = self.get_protocol(clean_sha)
        if got is not None:
            return got
        with self._Session() as s:
            s.add(
                PeriodProtocol(
                    clean_sha256=clean_sha, dev_end=dev_end, val_end=val_end, n_bars=n_bars
                )
            )
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
        got = self.get_protocol(clean_sha)
        assert got is not None
        return got

    def accesses(self, family_id: str, clean_sha: str, stage: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(
                select(PeriodAccess)
                .where(
                    PeriodAccess.family_id == family_id,
                    PeriodAccess.clean_sha256 == clean_sha,
                    PeriodAccess.stage == stage,
                )
                .order_by(PeriodAccess.at)
            ).scalars()
            return [_row(r) for r in rows]

    def add_access(
        self, family_id: str, clean_sha: str, stage: str, strategy_id: str, run_id: str
    ) -> None:
        with self.session() as s:
            s.add(
                PeriodAccess(
                    access_id=new_id("acc_"),
                    family_id=family_id,
                    clean_sha256=clean_sha,
                    stage=stage,
                    strategy_id=strategy_id,
                    run_id=run_id,
                    at=utcnow(),
                )
            )

    # ── koşular ──────────────────────────────────────────────────────────────

    def save_run(self, **fields: Any) -> dict[str, Any]:
        json_fields = {"data_report", "result_json"}
        with self.session() as s:
            s.add(
                StrategyRun(**{k: (dumps(v) if k in json_fields else v) for k, v in fields.items()})
            )
        got = self.get_run(fields["run_id"])
        assert got is not None
        return got

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            r = s.get(StrategyRun, run_id)
            if r is None:
                return None
            d = _row(r)
            d["data_report"] = loads(r.data_report, {})
            d["result"] = loads(r.result_json, {})
            d.pop("result_json", None)
            return d

    def list_runs(self, strategy_id: str = "", family_id: str = "") -> list[dict[str, Any]]:
        with self.session() as s:
            stmt = select(StrategyRun).order_by(StrategyRun.run_at)
            if strategy_id:
                stmt = stmt.where(StrategyRun.strategy_id == strategy_id)
            if family_id:
                stmt = stmt.where(StrategyRun.family_id == family_id)
            out = []
            for r in s.execute(stmt).scalars():
                d = _row(r)
                res = loads(r.result_json, {})
                d["metrics"] = res.get("metrics", {})
                d.pop("result_json", None)
                d.pop("data_report", None)
                out.append(d)
            return out


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
