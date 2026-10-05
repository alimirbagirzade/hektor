"""chat_store.py — sohbet geçmişi + öğrenme adayları + sohbet veri sürümleri (SQLite).

Mevcut SQLite dosyasını paylaşır (`FeedbackStore` deseni) ama YALNIZ YENİ tablolar kurar;
mevcut tablolara (Echo `feedback_corrections` dahil) dokunmaz → geriye uyumlu. Geri alma:
`drop_chat_tables()` yalnız bu modülün tablolarını siler (bkz. scripts/chat_learning_rollback.py).

Tablolar:
- ``chat_conversations`` / ``chat_turns``: konuşmanın TAMAMI; her tur modele gerçekten giden
  istemi, aktarılan geçmiş turlarını, cevabı veren model kimliğini ve kaynakları saklar.
- ``learning_candidates``: yalnız "Öğrensin" / "Düzelt" ile oluşan eğitim adayları.
- ``learning_families``: yakın-varyant soru aileleri + kalıcı train/eval ataması.
- ``chat_dataset_*``: değişmez veri sürümleri, üyeleri, kanonik eğitim dosyasına bağlanma
  kayıtları ve gözlenen eğitim koşuları.

İdempotentlik: aynı (konuşma, istemci istek kimliği) ikinci kez tur üretmez; aynı (tur, tür)
ikinci kez aday üretmez; aynı içerikli veri sürümü ikinci kez oluşmaz (benzersiz kısıtlar).
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


def _sqlite_pragmas(dbapi_conn: object, _record: object) -> None:
    cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
    finally:
        cur.close()


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


class ChatBase(DeclarativeBase):
    pass


class ChatConversation(ChatBase):
    __tablename__ = "chat_conversations"

    conversation_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    title: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)
    updated_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class ChatTurn(ChatBase):
    __tablename__ = "chat_turns"
    __table_args__ = (
        UniqueConstraint("conversation_id", "client_request_id", name="uq_turn_request"),
        UniqueConstraint("conversation_id", "turn_index", name="uq_turn_index"),
    )

    turn_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(40), index=True)
    turn_index: Mapped[int] = mapped_column(Integer)
    client_request_id: Mapped[str] = mapped_column(String(80))
    question: Mapped[str] = mapped_column(Text, default="")
    retrieval_query: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    user_prompt: Mapped[str] = mapped_column(Text, default="")
    prompt_sha256: Mapped[str] = mapped_column(String(64), default="")
    history_turn_ids: Mapped[str] = mapped_column(Text, default="[]")
    model_tag: Mapped[str] = mapped_column(String(200), default="")
    model_digest: Mapped[str] = mapped_column(String(100), default="")
    model_info: Mapped[str] = mapped_column(Text, default="{}")
    sources: Mapped[str] = mapped_column(Text, default="[]")
    checks: Mapped[str] = mapped_column(Text, default="[]")
    answer: Mapped[str] = mapped_column(Text, default="")
    raw_answer: Mapped[str] = mapped_column(Text, default="")
    llm_used: Mapped[bool] = mapped_column(Boolean, default=False)
    # pending → answered | no_llm | blocked | error
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    status_detail: Mapped[str] = mapped_column(Text, default="")
    feedback: Mapped[str] = mapped_column(String(16), default="")  # useful | wrong | ""
    feedback_note: Mapped[str] = mapped_column(Text, default="")
    flagged_spans: Mapped[str] = mapped_column(Text, default="[]")
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    exclude_reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)
    updated_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class LearningCandidate(ChatBase):
    __tablename__ = "learning_candidates"
    __table_args__ = (UniqueConstraint("turn_id", "kind", name="uq_candidate_turn_kind"),)

    candidate_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    turn_id: Mapped[str] = mapped_column(String(40), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # learn | correct
    target_text: Mapped[str] = mapped_column(Text, default="")
    target_sha: Mapped[str] = mapped_column(String(64), default="")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    flagged_spans: Mapped[str] = mapped_column(Text, default="[]")
    domain: Mapped[str] = mapped_column(String(20), default="general")
    domain_source: Mapped[str] = mapped_column(String(10), default="auto")  # auto | user
    # review | eligible | rejected | excluded | conflict | leak | duplicate
    status: Mapped[str] = mapped_column(String(16), default="review", index=True)
    status_reason: Mapped[str] = mapped_column(Text, default="")
    reason_codes: Mapped[str] = mapped_column(Text, default="[]")
    verification: Mapped[str] = mapped_column(Text, default="{}")
    human_approval: Mapped[str] = mapped_column(Text, default="{}")
    family_id: Mapped[str] = mapped_column(String(40), default="", index=True)
    as_of: Mapped[str] = mapped_column(String(40), default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)
    updated_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class LearningFamily(ChatBase):
    __tablename__ = "learning_families"

    family_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    # train | eval (kalıcı) | time (trading: her sürümde zaman sırasıyla bölünür)
    split: Mapped[str] = mapped_column(String(8), default="train")
    domain: Mapped[str] = mapped_column(String(20), default="general")
    anchor_question: Mapped[str] = mapped_column(Text, default="")
    merged_into: Mapped[str] = mapped_column(String(40), default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class ChatDatasetVersion(ChatBase):
    __tablename__ = "chat_dataset_versions"

    version_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, unique=True)
    content_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    dir_path: Mapped[str] = mapped_column(Text, default="")
    train_sha256: Mapped[str] = mapped_column(String(64), default="")
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_eval: Mapped[int] = mapped_column(Integer, default=0)
    n_train_families: Mapped[int] = mapped_column(Integer, default=0)
    n_eval_families: Mapped[int] = mapped_column(Integer, default=0)
    params: Mapped[str] = mapped_column(Text, default="{}")
    registry_id: Mapped[str] = mapped_column(String(40), default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class ChatDatasetMember(ChatBase):
    __tablename__ = "chat_dataset_members"

    version_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    split: Mapped[str] = mapped_column(String(8), default="train")
    family_id: Mapped[str] = mapped_column(String(40), default="")
    target_sha: Mapped[str] = mapped_column(String(64), default="")
    verification_class: Mapped[str] = mapped_column(String(10), default="auto")


class ChatDatasetBinding(ChatBase):
    """Bir sohbet sürümünün kanonik `lora_sft.jsonl`'e girdiği an (= "planlandı")."""

    __tablename__ = "chat_dataset_bindings"
    __table_args__ = (UniqueConstraint("version_id", "lora_sft_sha256", name="uq_binding"),)

    binding_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    version_id: Mapped[str] = mapped_column(String(40), index=True)
    lora_sft_sha256: Mapped[str] = mapped_column(String(64), index=True)
    stats: Mapped[str] = mapped_column(Text, default="{}")
    bound_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class ChatDatasetRun(ChatBase):
    """Bağlanmış veriyle gözlenen eğitim koşusu (başlatıldı | sürüyor | tamamlandı | başarısız)."""

    __tablename__ = "chat_dataset_runs"

    run_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    version_id: Mapped[str] = mapped_column(String(40), index=True)
    lora_sft_sha256: Mapped[str] = mapped_column(String(64), default="")
    adapter: Mapped[str] = mapped_column(String(120), default="")
    started_at: Mapped[str] = mapped_column(String(40), default="")
    state: Mapped[str] = mapped_column(String(16), default="started")
    observed_at: Mapped[str] = mapped_column(String(40), default=utcnow)


CHAT_TABLES = tuple(ChatBase.metadata.sorted_tables)


class ChatStore:
    """Sohbet + öğrenme tablolarına ince erişim katmanı."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = db_path or get_settings().sqlite_file
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, connect_args={"timeout": 30.0}
        )
        event.listen(self._engine, "connect", _sqlite_pragmas)
        # Yalnız eksik tabloları kurar; var olan hiçbir tabloya/kolona dokunmaz.
        ChatBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._Session() as s:
            yield s
            s.commit()

    # ── konuşmalar ───────────────────────────────────────────────────────────

    def create_conversation(self, title: str = "") -> dict[str, Any]:
        cid = new_id("conv_")
        now = utcnow()
        with self.session() as s:
            s.add(
                ChatConversation(conversation_id=cid, title=title, created_at=now, updated_at=now)
            )
        conv = self.get_conversation(cid)
        assert conv is not None
        return conv

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(ChatConversation, conversation_id)
            return _row(row) if row else None

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(
                select(ChatConversation)
                .order_by(ChatConversation.updated_at.desc())
                .limit(max(1, limit))
            ).scalars()
            out = []
            for r in rows:
                d = _row(r)
                d["n_turns"] = s.scalar(
                    select(func.count())
                    .select_from(ChatTurn)
                    .where(ChatTurn.conversation_id == r.conversation_id)
                )
                out.append(d)
            return out

    def touch_conversation(self, conversation_id: str, title_if_empty: str = "") -> None:
        with self.session() as s:
            row = s.get(ChatConversation, conversation_id)
            if row is None:
                return
            row.updated_at = utcnow()
            if title_if_empty and not row.title:
                row.title = title_if_empty[:120]

    # ── turlar ───────────────────────────────────────────────────────────────

    def reserve_turn(
        self, conversation_id: str, client_request_id: str, question: str
    ) -> tuple[dict[str, Any], bool]:
        """Turu 'pending' olarak AYIR (cevaptan önce). (tur, yeni_mi) döndürür.

        Aynı (konuşma, istek kimliği) zaten varsa mevcut tur döner (çift tur yok). Sıra
        numarası çakışırsa (eşzamanlı gönderim) yeniden denenir.
        """
        for _ in range(5):
            existing = self.find_turn_by_request(conversation_id, client_request_id)
            if existing is not None:
                return existing, False
            with self._Session() as s:
                last = s.scalar(
                    select(func.max(ChatTurn.turn_index)).where(
                        ChatTurn.conversation_id == conversation_id
                    )
                )
                turn = ChatTurn(
                    turn_id=new_id("turn_"),
                    conversation_id=conversation_id,
                    turn_index=int(last or 0) + 1,
                    client_request_id=client_request_id,
                    question=question,
                    status="pending",
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
                s.add(turn)
                try:
                    s.commit()
                except IntegrityError:
                    s.rollback()
                    continue
                return _turn(turn), True
        raise RuntimeError("Tur ayrılamadı (eşzamanlı istek çakışması).")

    def find_turn_by_request(
        self, conversation_id: str, client_request_id: str
    ) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.scalar(
                select(ChatTurn).where(
                    ChatTurn.conversation_id == conversation_id,
                    ChatTurn.client_request_id == client_request_id,
                )
            )
            return _turn(row) if row else None

    def update_turn(self, turn_id: str, **fields: Any) -> dict[str, Any] | None:
        json_fields = {"history_turn_ids", "model_info", "sources", "checks", "flagged_spans"}
        with self.session() as s:
            row = s.get(ChatTurn, turn_id)
            if row is None:
                return None
            for k, v in fields.items():
                setattr(row, k, dumps(v) if k in json_fields else v)
            row.updated_at = utcnow()
        return self.get_turn(turn_id)

    def get_turn(self, turn_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(ChatTurn, turn_id)
            return _turn(row) if row else None

    def list_turns(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(
                select(ChatTurn)
                .where(ChatTurn.conversation_id == conversation_id)
                .order_by(ChatTurn.turn_index)
            ).scalars()
            return [_turn(r) for r in rows]

    def list_turns_by_feedback(self, feedback: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(
                select(ChatTurn)
                .where(ChatTurn.feedback == feedback)
                .order_by(ChatTurn.created_at.desc())
                .limit(max(1, limit))
            ).scalars()
            return [_turn(r) for r in rows]

    # ── adaylar ──────────────────────────────────────────────────────────────

    def get_candidate_for_turn(self, turn_id: str, kind: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.scalar(
                select(LearningCandidate).where(
                    LearningCandidate.turn_id == turn_id, LearningCandidate.kind == kind
                )
            )
            return _cand(row) if row else None

    def insert_candidate(self, **fields: Any) -> tuple[dict[str, Any], bool]:
        """Aday ekle; aynı (tur, tür) varsa mevcut olanı döndür (çift aday yok)."""
        cid = new_id("cand_")
        now = utcnow()
        with self._Session() as s:
            s.add(
                LearningCandidate(
                    candidate_id=cid,
                    created_at=now,
                    updated_at=now,
                    **_encode_cand(fields),
                )
            )
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                existing = self.get_candidate_for_turn(fields["turn_id"], fields["kind"])
                if existing is None:  # pragma: no cover - kısıt başka nedenle düştü
                    raise
                return existing, False
        got = self.get_candidate(cid)
        assert got is not None
        return got, True

    def update_candidate(self, candidate_id: str, **fields: Any) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(LearningCandidate, candidate_id)
            if row is None:
                return None
            for k, v in _encode_cand(fields).items():
                setattr(row, k, v)
            row.updated_at = utcnow()
        return self.get_candidate(candidate_id)

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(LearningCandidate, candidate_id)
            return _cand(row) if row else None

    def list_candidates(
        self, status: str | None = None, limit: int = 10_000
    ) -> list[dict[str, Any]]:
        with self.session() as s:
            stmt = select(LearningCandidate)
            if status:
                stmt = stmt.where(LearningCandidate.status == status)
            stmt = stmt.order_by(
                LearningCandidate.created_at, LearningCandidate.candidate_id
            ).limit(max(1, limit))
            return [_cand(r) for r in s.execute(stmt).scalars()]

    def candidates_for_turn(self, turn_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(
                select(LearningCandidate).where(LearningCandidate.turn_id == turn_id)
            ).scalars()
            return [_cand(r) for r in rows]

    # ── aileler ──────────────────────────────────────────────────────────────

    def get_family(self, family_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(LearningFamily, family_id)
            return _row(row) if row else None

    def resolve_family(self, family_id: str) -> str:
        """Birleşmiş ailelerin zincirini izle → canlı aile kimliği."""
        seen: set[str] = set()
        fid = family_id
        while fid and fid not in seen:
            seen.add(fid)
            fam = self.get_family(fid)
            if fam is None or not fam["merged_into"]:
                return fid
            fid = fam["merged_into"]
        return fid

    def upsert_family(self, family_id: str, **fields: Any) -> dict[str, Any]:
        with self.session() as s:
            row = s.get(LearningFamily, family_id)
            if row is None:
                row = LearningFamily(family_id=family_id, created_at=utcnow(), **fields)
                s.add(row)
            else:
                for k, v in fields.items():
                    setattr(row, k, v)
        fam = self.get_family(family_id)
        assert fam is not None
        return fam

    # ── veri sürümleri ───────────────────────────────────────────────────────

    def find_version_by_sha(self, content_sha256: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.scalar(
                select(ChatDatasetVersion).where(
                    ChatDatasetVersion.content_sha256 == content_sha256
                )
            )
            return _version(row) if row else None

    def next_version_seq(self) -> int:
        with self.session() as s:
            last = s.scalar(select(func.max(ChatDatasetVersion.seq)))
            return int(last or 0) + 1

    def insert_version(
        self, version: dict[str, Any], members: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], bool]:
        with self._Session() as s:
            s.add(ChatDatasetVersion(**{**version, "params": dumps(version.get("params", {}))}))
            for m in members:
                s.add(ChatDatasetMember(version_id=version["version_id"], **m))
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                existing = self.find_version_by_sha(version["content_sha256"])
                if existing is None:
                    raise
                return existing, False
        got = self.get_version(version["version_id"])
        assert got is not None
        return got, True

    def set_version_registry(self, version_id: str, registry_id: str) -> None:
        with self.session() as s:
            row = s.get(ChatDatasetVersion, version_id)
            if row is not None:
                row.registry_id = registry_id

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(ChatDatasetVersion, version_id)
            return _version(row) if row else None

    def list_versions(self) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(
                select(ChatDatasetVersion).order_by(ChatDatasetVersion.seq.desc())
            ).scalars()
            return [_version(r) for r in rows]

    def version_members(self, version_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(
                select(ChatDatasetMember).where(ChatDatasetMember.version_id == version_id)
            ).scalars()
            return [_row(r) for r in rows]

    def all_members(self) -> list[dict[str, Any]]:
        with self.session() as s:
            return [_row(r) for r in s.execute(select(ChatDatasetMember)).scalars()]

    def add_binding(self, version_id: str, lora_sft_sha256: str, stats: dict[str, Any]) -> None:
        with self._Session() as s:
            s.add(
                ChatDatasetBinding(
                    binding_id=new_id("bind_"),
                    version_id=version_id,
                    lora_sft_sha256=lora_sft_sha256,
                    stats=dumps(stats),
                    bound_at=utcnow(),
                )
            )
            try:
                s.commit()
            except IntegrityError:
                s.rollback()  # aynı (sürüm, dosya) zaten bağlı — idempotent

    def list_bindings(self) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(
                select(ChatDatasetBinding).order_by(ChatDatasetBinding.bound_at)
            ).scalars()
            out = []
            for r in rows:
                d = _row(r)
                d["stats"] = loads(r.stats, {})
                out.append(d)
            return out

    def upsert_run(self, run_key: str, **fields: Any) -> None:
        with self.session() as s:
            row = s.get(ChatDatasetRun, run_key)
            if row is None:
                s.add(ChatDatasetRun(run_key=run_key, observed_at=utcnow(), **fields))
            else:
                for k, v in fields.items():
                    setattr(row, k, v)
                row.observed_at = utcnow()

    def list_runs(self) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.execute(select(ChatDatasetRun).order_by(ChatDatasetRun.started_at)).scalars()
            return [_row(r) for r in rows]


def drop_chat_tables(db_path: str | Path) -> list[str]:
    """YALNIZ bu modülün tablolarını sil (geri alma). Diğer tablolara dokunmaz."""
    engine = create_engine(f"sqlite:///{db_path}")
    names = [t.name for t in CHAT_TABLES]
    ChatBase.metadata.drop_all(engine)
    engine.dispose()
    return names


# ── satır → sözlük ───────────────────────────────────────────────────────────


def _row(r: Any) -> dict[str, Any]:
    return {c.name: getattr(r, c.name) for c in r.__table__.columns}


def _turn(r: ChatTurn) -> dict[str, Any]:
    d = _row(r)
    d["history_turn_ids"] = loads(r.history_turn_ids, [])
    d["model_info"] = loads(r.model_info, {})
    d["sources"] = loads(r.sources, [])
    d["checks"] = loads(r.checks, [])
    d["flagged_spans"] = loads(r.flagged_spans, [])
    return d


_CAND_JSON = ("flagged_spans", "reason_codes", "verification", "human_approval")


def _encode_cand(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: (dumps(v) if k in _CAND_JSON else v) for k, v in fields.items()}


def _cand(r: LearningCandidate) -> dict[str, Any]:
    d = _row(r)
    d["flagged_spans"] = loads(r.flagged_spans, [])
    d["reason_codes"] = loads(r.reason_codes, [])
    d["verification"] = loads(r.verification, {})
    d["human_approval"] = loads(r.human_approval, {})
    return d


def _version(r: ChatDatasetVersion) -> dict[str, Any]:
    d = _row(r)
    d["params"] = loads(r.params, {})
    return d
