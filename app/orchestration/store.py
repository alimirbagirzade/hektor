"""store.py — orkestrasyon koşularının kalıcı (SQLite) deposu.

`rlm_store.py` / `mastery_store.py` desenini izler: aynı SQLite dosyasını paylaşır,
kendi engine + DeclarativeBase'ini kurar, her bağlantıda WAL + busy_timeout uygular.

Tablolar:
  orchestration_runs    — bir uçtan-uca eğitim koşusu (model, profil, durum, aşama).
  orchestration_stages  — koşu içindeki her aşamanın durumu + checkpoint çıktısı.
  orchestration_events  — zaman çizelgesi (dashboard + denetim için).
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Integer, String, Text, create_engine, event, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings
from app.orchestration.pipeline import PIPELINE, RunStatus, StageStatus, stage_order


def _sqlite_pragmas(dbapi_conn: object, _record: object) -> None:
    """Her bağlantıda WAL + busy_timeout (rlm_store ile aynı gerekçe: paylaşılan
    SQLite'ta eşzamanlı yazımda 'database is locked' riskini azaltır)."""
    cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
    finally:
        cur.close()


def utcnow() -> str:
    """ISO8601 UTC damgası (leksikografik karşılaştırma sıralamaya uygun)."""
    return dt.datetime.now(dt.UTC).isoformat()


def iso_minutes_ago(minutes: float) -> str:
    """Şimdiden `minutes` dakika öncesinin ISO8601 UTC damgası (stale kesimi için)."""
    return (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=minutes)).isoformat()


def _new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:16]


class OrchestrationBase(DeclarativeBase):
    pass


class OrchestrationRun(OrchestrationBase):
    __tablename__ = "orchestration_runs"

    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    model: Mapped[str] = mapped_column(String(64), default="")
    profile: Mapped[str] = mapped_column(String(64), default="")
    adapter_name: Mapped[str] = mapped_column(String(80), default="")
    status: Mapped[str] = mapped_column(String(32), default=RunStatus.pending.value)
    current_stage: Mapped[str] = mapped_column(String(40), default="")
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)
    updated_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class OrchestrationStage(OrchestrationBase):
    __tablename__ = "orchestration_stages"

    stage_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    name: Mapped[str] = mapped_column(String(40))
    stage_order: Mapped[int] = mapped_column(Integer, default=0)
    kind: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(32), default=StageStatus.pending.value, index=True)
    message: Mapped[str] = mapped_column(Text, default="")
    output_json: Mapped[str] = mapped_column(Text, default="{}")
    started_at: Mapped[str] = mapped_column(String(40), default="")
    finished_at: Mapped[str] = mapped_column(String(40), default="")
    heartbeat_at: Mapped[str] = mapped_column(String(40), default="")


class OrchestrationEvent(OrchestrationBase):
    __tablename__ = "orchestration_events"

    event_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    stage: Mapped[str] = mapped_column(String(40), default="")
    level: Mapped[str] = mapped_column(String(16), default="info")
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class OrchestrationStore:
    """Orkestrasyon koşu/aşama/olay kayıtlarına erişim deposu."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = db_path or get_settings().sqlite_file
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, connect_args={"timeout": 30.0}
        )
        event.listen(self._engine, "connect", _sqlite_pragmas)
        OrchestrationBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._Session() as s:
            yield s
            s.commit()

    # ── Runs ─────────────────────────────────────────────────────────────────

    def create_run(
        self,
        *,
        model: str,
        profile: str,
        adapter_name: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Yeni koşu + tüm PIPELINE aşamalarını (pending) oluşturur; run_id döner."""
        run_id = _new_id("orc_")
        now = utcnow()
        with self.session() as s:
            s.add(
                OrchestrationRun(
                    run_id=run_id,
                    model=model,
                    profile=profile,
                    adapter_name=adapter_name,
                    status=RunStatus.pending.value,
                    current_stage=PIPELINE[0].name if PIPELINE else "",
                    params_json=json.dumps(params or {}, ensure_ascii=False, default=str),
                    created_at=now,
                    updated_at=now,
                )
            )
            for st in PIPELINE:
                s.add(
                    OrchestrationStage(
                        stage_id=_new_id("orst_"),
                        run_id=run_id,
                        name=st.name,
                        stage_order=stage_order(st.name),
                        kind=st.kind.value,
                        status=StageStatus.pending.value,
                    )
                )
            s.add(
                OrchestrationEvent(
                    event_id=_new_id("orev_"),
                    run_id=run_id,
                    stage="",
                    level="info",
                    message=f"Koşu oluşturuldu: model={model} profil={profile}",
                    created_at=now,
                )
            )
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(OrchestrationRun, run_id)
            return _run_to_dict(row) if row else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = (
                s.execute(
                    select(OrchestrationRun)
                    .order_by(OrchestrationRun.created_at.desc())
                    .limit(max(1, limit))
                )
                .scalars()
                .all()
            )
            return [_run_to_dict(r) for r in rows]

    def update_run(self, run_id: str, **fields: Any) -> None:
        with self.session() as s:
            row = s.get(OrchestrationRun, run_id)
            if row is None:
                return
            for k, v in fields.items():
                if hasattr(row, k):
                    setattr(row, k, v)
            row.updated_at = utcnow()

    # ── Stages ───────────────────────────────────────────────────────────────

    def get_stages(self, run_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = (
                s.execute(
                    select(OrchestrationStage)
                    .where(OrchestrationStage.run_id == run_id)
                    .order_by(OrchestrationStage.stage_order.asc())
                )
                .scalars()
                .all()
            )
            return [_stage_to_dict(r) for r in rows]

    def get_stage(self, run_id: str, name: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.execute(
                select(OrchestrationStage).where(
                    OrchestrationStage.run_id == run_id,
                    OrchestrationStage.name == name,
                )
            ).scalar_one_or_none()
            return _stage_to_dict(row) if row else None

    def update_stage(self, run_id: str, name: str, **fields: Any) -> None:
        with self.session() as s:
            row = s.execute(
                select(OrchestrationStage).where(
                    OrchestrationStage.run_id == run_id,
                    OrchestrationStage.name == name,
                )
            ).scalar_one_or_none()
            if row is None:
                return
            for k, v in fields.items():
                if k == "output":  # dict → JSON kolaylığı
                    # default=str: delege JSON-dışı bir değer (ör. datetime) koysa bile
                    # step() çökmesin (output_json best-effort serileştirilir).
                    row.output_json = json.dumps(v, ensure_ascii=False, default=str)
                elif hasattr(row, k):
                    setattr(row, k, v)

    def touch_heartbeat(self, run_id: str, name: str) -> None:
        """Çalışan bir aşamanın kalp atışını günceller (panic-recovery için).

        NOT: Varsayılan delegeler senkron + hızlı (salt-okuma) ve tehlikeli train/eval/
        registry aşamaları handoff olduğundan bugün HİÇBİR çağıran yok. İleride inline,
        uzun-süren (timeout_min'i aşan) bir "yürüten delege" bağlanırsa, o delege bunu
        periyodik çağırmalı; aksi halde recover_stale onu yanlışlıkla stale sayabilir.
        """
        self.update_stage(run_id, name, heartbeat_at=utcnow())

    def claim_stage_running(self, run_id: str, name: str) -> bool:
        """Aşamayı atomik (CAS) olarak 'running'e al; yalnız status pending/blocked/failed
        ise günceller. rowcount==1 → bu çağrı sahiplendi (True); 0 → başka bir eşzamanlı
        step() (FastAPI sync uçları threadpool'da koşar) zaten kapmış → çift-delege önlenir.
        consume_fresh_approval / claim_automation_task ile aynı koşullu-UPDATE deseni."""
        now = utcnow()
        with self.session() as s:
            res = s.execute(
                update(OrchestrationStage)
                .where(
                    OrchestrationStage.run_id == run_id,
                    OrchestrationStage.name == name,
                    OrchestrationStage.status.in_(
                        [
                            StageStatus.pending.value,
                            StageStatus.blocked.value,
                            StageStatus.failed.value,
                        ]
                    ),
                )
                .values(
                    status=StageStatus.running.value,
                    started_at=now,
                    heartbeat_at=now,
                    message="",
                    finished_at="",
                )
                .execution_options(synchronize_session=False)
            )
            return cast("Any", res).rowcount == 1

    def find_stale_running_stages(self, cutoff_iso: str) -> list[tuple[str, str]]:
        """heartbeat_at < cutoff olan 'running' aşamaları (run_id, name) döner.

        Koşusu zaten terminal (completed/cancelled) olan aşamalar HARİÇ tutulur: yoksa
        kullanıcı bilinçle iptal ettiği (cancelled) bir koşunun asılı kalmış 'running'
        aşaması, sonraki recover çağrısında koşuyu sessizce 'cancelled'→'failed' clobber
        ederdi (eşzamanlı cancel + recover yarışı)."""
        final = [RunStatus.completed.value, RunStatus.cancelled.value]
        with self.session() as s:
            rows = (
                s.execute(
                    select(OrchestrationStage)
                    .join(
                        OrchestrationRun,
                        OrchestrationRun.run_id == OrchestrationStage.run_id,
                    )
                    .where(
                        OrchestrationStage.status == StageStatus.running.value,
                        OrchestrationStage.heartbeat_at != "",
                        OrchestrationStage.heartbeat_at < cutoff_iso,
                        OrchestrationRun.status.notin_(final),
                    )
                )
                .scalars()
                .all()
            )
            return [(r.run_id, r.name) for r in rows]

    # ── Events ───────────────────────────────────────────────────────────────

    def add_event(self, run_id: str, stage: str, level: str, message: str) -> str:
        event_id = _new_id("orev_")
        with self.session() as s:
            s.add(
                OrchestrationEvent(
                    event_id=event_id,
                    run_id=run_id,
                    stage=stage,
                    level=level,
                    message=message,
                    created_at=utcnow(),
                )
            )
        return event_id

    def get_events(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = (
                s.execute(
                    select(OrchestrationEvent)
                    .where(OrchestrationEvent.run_id == run_id)
                    .order_by(OrchestrationEvent.created_at.asc())
                    .limit(max(1, limit))
                )
                .scalars()
                .all()
            )
            return [_event_to_dict(r) for r in rows]


# ── Satır → dict yardımcıları ────────────────────────────────────────────────


def _run_to_dict(r: OrchestrationRun) -> dict[str, Any]:
    try:
        params = json.loads(r.params_json or "{}")
    except json.JSONDecodeError:
        params = {}
    return {
        "run_id": r.run_id,
        "model": r.model,
        "profile": r.profile,
        "adapter_name": r.adapter_name,
        "status": r.status,
        "current_stage": r.current_stage,
        "params": params,
        "error": r.error,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
    }


def _stage_to_dict(r: OrchestrationStage) -> dict[str, Any]:
    try:
        output = json.loads(r.output_json or "{}")
    except json.JSONDecodeError:
        output = {}
    return {
        "name": r.name,
        "order": r.stage_order,
        "kind": r.kind,
        "status": r.status,
        "message": r.message,
        "output": output,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
        "heartbeat_at": r.heartbeat_at,
    }


def _event_to_dict(r: OrchestrationEvent) -> dict[str, Any]:
    return {
        "event_id": r.event_id,
        "stage": r.stage,
        "level": r.level,
        "message": r.message,
        "created_at": r.created_at,
    }
