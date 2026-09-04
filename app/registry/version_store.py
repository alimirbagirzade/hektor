"""Model/veri kayıt defteri — dataset / RAG-index / embedding / RLM-reward sürümleri.

Bu katman, sistemde ZATEN olmayan tek parçayı tamamlar: eğitim-dışı varlıkların
(veri seti, vektör indeks, embedding modeli, ödül seti) sürümlenmesi ve her terfi
kararının denetlenebilir kaydı. Adapter yaşam döngüsü ayrı kalır
(``app/lora/adapter_registry.py`` + ``app/training/adapter_registry.py``).

Tasarım ilkeleri:
- Tek SQLite store üzerine kurulu (yeni veritabanı yok).
- ``content_hash`` verilirse dataset kaydı idempotenttir (aynı veri → aynı sürüm).
- RAG-index / embedding anlık görüntüleri SQLite sayımlarından üretilir → ChromaDB
  ya da ağ gerektirmez (çevrimdışı testlerle uyumlu).
- Hiçbir şey ``app/rlm/`` paketini içe aktarmaz/düzenlemez.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, cast

from sqlalchemy import func, select, update

from app.config import get_settings
from app.memory.sqlite_store import (
    Chunk,
    DatasetVersion,
    EmbeddingModelVersion,
    Paper,
    PromotionDecision,
    RagIndexVersion,
    RlmRewardVersion,
    SqliteStore,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class RegistryStore:
    """SQLite tabanlı sürüm kayıt defteri (ince ORM sarmalayıcı)."""

    def __init__(self, store: SqliteStore | None = None) -> None:
        self.store = store or SqliteStore()

    # --- dataset sürümleri ------------------------------------------------
    def register_dataset(
        self,
        *,
        name: str,
        path: str | Path | None = None,
        source_type: str = "sft",
        content_hash: str | None = None,
        n_records: int = 0,
        domain_distribution: dict[str, int] | None = None,
        quality_score: float | None = None,
    ) -> dict[str, Any]:
        """Yeni dataset sürümü kaydet. ``content_hash`` verilip eşleşirse idempotent."""
        if content_hash:
            existing = self.find_dataset_by_hash(content_hash)
            if existing is not None:
                return existing
        vid = _new_id("ds")
        with self.store.session() as s:
            s.add(
                DatasetVersion(
                    dataset_version_id=vid,
                    name=name,
                    path=str(path) if path is not None else None,
                    source_type=source_type,
                    content_hash=content_hash,
                    n_records=int(n_records),
                    domain_distribution_json=json.dumps(
                        domain_distribution or {}, ensure_ascii=False
                    ),
                    quality_score=quality_score,
                )
            )
        result = self.get_dataset(vid)
        assert result is not None  # az önce eklendi
        return result

    def register_dataset_from_file(
        self, path: str | Path, *, name: str | None = None, source_type: str = "sft"
    ) -> dict[str, Any]:
        """Bir dataset dosyasından (JSONL) sürüm kaydet: içerik-hash + kayıt sayısı otomatik.

        ``content_hash`` = SHA-256(dosya baytları) → aynı dosya tekrar kaydedilirse idempotent.
        ``n_records`` = boş-olmayan satır sayısı. Eğitim öncesi "bu veriyi sürümle"nin operasyonel
        ucu (sonra ``registry-promote-dataset`` ile İNSAN ONAYI; Kural 8).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset dosyası yok: {p}")
        content_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        n_records = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        return self.register_dataset(
            name=name or p.stem,
            path=str(p),
            source_type=source_type,
            content_hash=content_hash,
            n_records=n_records,
        )

    def find_dataset_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self.store.session() as s:
            row = s.scalar(
                select(DatasetVersion)
                .where(DatasetVersion.content_hash == content_hash)
                .order_by(DatasetVersion.created_at.desc())
                .limit(1)
            )
            return _dataset_to_dict(row) if row else None

    def get_dataset(self, dataset_version_id: str) -> dict[str, Any] | None:
        with self.store.session() as s:
            row = s.get(DatasetVersion, dataset_version_id)
            return _dataset_to_dict(row) if row else None

    def list_datasets(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.store.session() as s:
            rows = s.scalars(
                select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(limit)
            )
            return [_dataset_to_dict(r) for r in rows]

    def set_dataset_status(self, dataset_version_id: str, status: str) -> str | None:
        """Onay durumunu güncelle; önceki durumu döndür (yoksa None)."""
        with self.store.session() as s:
            row = s.get(DatasetVersion, dataset_version_id)
            if row is None:
                return None
            prev = row.approval_status
            row.approval_status = status
            return prev

    def cas_dataset_status(
        self, dataset_version_id: str, *, expected: str, new_status: str
    ) -> bool:
        """Atomik durum geçişi: yalnız ``approval_status == expected`` iken ``new_status`` yaz.

        Koşullu UPDATE + ``rowcount`` (consume_fresh_approval ile aynı CAS deseni) →
        eşzamanlı iki çağrıdan yalnız BİRİ rowcount=1 alır (TOCTOU çift-karar önlenir).
        ``expected`` durum makinesini de zorlar: yalnız ``pending``'den terminal'e geçilir
        (approved/rejected terminaldir → çapraz geçiş engellenir). ``True`` = bu çağrı geçişi
        yaptı; ``False`` = yok / ``expected`` değil / yarışı kaybetti.
        """
        with self.store.session() as s:
            res = s.execute(
                update(DatasetVersion)
                .where(
                    DatasetVersion.dataset_version_id == dataset_version_id,
                    DatasetVersion.approval_status == expected,
                )
                .values(approval_status=new_status)
                .execution_options(synchronize_session=False)
            )
            return cast("Any", res).rowcount == 1

    # --- RAG indeks sürümleri --------------------------------------------
    def register_rag_index(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        n_chunks: int,
        n_papers: int,
        vector_db: str = "chroma",
        chunking_strategy: str | None = None,
    ) -> dict[str, Any]:
        vid = _new_id("rag")
        with self.store.session() as s:
            s.add(
                RagIndexVersion(
                    rag_index_version_id=vid,
                    vector_db=vector_db,
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    chunking_strategy=chunking_strategy,
                    n_chunks=int(n_chunks),
                    n_papers=int(n_papers),
                )
            )
        result = self.get_rag_index(vid)
        assert result is not None
        return result

    def snapshot_rag_index(self) -> dict[str, Any]:
        """Mevcut SQLite durumundan bir RAG-indeks anlık görüntüsü üret (çevrimdışı).

        Chunk/Paper sayımları SQLite'tan okunur → ChromaDB/ağ gerektirmez. Embedding
        modeli ve collection adı ayarlardan alınır.
        """
        settings = get_settings()
        with self.store.session() as s:
            n_chunks = int(s.scalar(select(func.count()).select_from(Chunk)) or 0)
            n_papers = int(s.scalar(select(func.count()).select_from(Paper)) or 0)
        provider_fake = settings.allow_fake_embeddings
        embed = f"fake:{settings.embed_model}" if provider_fake else settings.embed_model
        return self.register_rag_index(
            collection_name="papers",
            embedding_model=embed,
            n_chunks=n_chunks,
            n_papers=n_papers,
            chunking_strategy=f"size={settings.chunk_size},overlap={settings.chunk_overlap}",
        )

    def get_rag_index(self, rag_index_version_id: str) -> dict[str, Any] | None:
        with self.store.session() as s:
            row = s.get(RagIndexVersion, rag_index_version_id)
            return _rag_index_to_dict(row) if row else None

    def list_rag_indices(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.store.session() as s:
            rows = s.scalars(
                select(RagIndexVersion).order_by(RagIndexVersion.created_at.desc()).limit(limit)
            )
            return [_rag_index_to_dict(r) for r in rows]

    # --- embedding modeli sürümleri --------------------------------------
    def register_embedding(
        self,
        *,
        model_name: str,
        dimension: int | None = None,
        provider: str | None = None,
        local_path: str | None = None,
    ) -> dict[str, Any]:
        vid = _new_id("emb")
        with self.store.session() as s:
            s.add(
                EmbeddingModelVersion(
                    embedding_model_id=vid,
                    model_name=model_name,
                    dimension=dimension,
                    provider=provider,
                    local_path=local_path,
                )
            )
        with self.store.session() as s:
            row = s.get(EmbeddingModelVersion, vid)
            assert row is not None
            return _embedding_to_dict(row)

    def snapshot_embedding(self) -> dict[str, Any]:
        settings = get_settings()
        provider = "fake" if settings.allow_fake_embeddings else "ollama"
        return self.register_embedding(model_name=settings.embed_model, provider=provider)

    def list_embeddings(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.store.session() as s:
            rows = s.scalars(
                select(EmbeddingModelVersion)
                .order_by(EmbeddingModelVersion.created_at.desc())
                .limit(limit)
            )
            return [_embedding_to_dict(r) for r in rows]

    # --- RLM ödül seti sürümleri (yalnız sürümleme; rlm paketi içe aktarılmaz) --
    def register_reward(
        self,
        *,
        name: str,
        dataset_path: str | None = None,
        method: str | None = None,
        n_examples: int = 0,
    ) -> dict[str, Any]:
        vid = _new_id("rew")
        with self.store.session() as s:
            s.add(
                RlmRewardVersion(
                    reward_version_id=vid,
                    name=name,
                    dataset_path=dataset_path,
                    method=method,
                    n_examples=int(n_examples),
                )
            )
        with self.store.session() as s:
            row = s.get(RlmRewardVersion, vid)
            assert row is not None
            return _reward_to_dict(row)

    def set_reward_scan_flags(
        self, reward_version_id: str, *, secret_scanned: int, pii_scanned: int
    ) -> bool:
        with self.store.session() as s:
            row = s.get(RlmRewardVersion, reward_version_id)
            if row is None:
                return False
            row.secret_scanned = int(secret_scanned)
            row.pii_scanned = int(pii_scanned)
            return True

    def list_rewards(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.store.session() as s:
            rows = s.scalars(
                select(RlmRewardVersion).order_by(RlmRewardVersion.created_at.desc()).limit(limit)
            )
            return [_reward_to_dict(r) for r in rows]

    # --- terfi kararları (denetim izi) -----------------------------------
    def log_decision(
        self,
        *,
        target_type: str,
        target_id: str,
        to_status: str,
        decision: str,
        from_status: str | None = None,
        reason: str | None = None,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        did = _new_id("dec")
        with self.store.session() as s:
            s.add(
                PromotionDecision(
                    decision_id=did,
                    target_type=target_type,
                    target_id=target_id,
                    from_status=from_status,
                    to_status=to_status,
                    decision=decision,
                    reason=reason,
                    approved_by=approved_by,
                )
            )
        with self.store.session() as s:
            row = s.get(PromotionDecision, did)
            assert row is not None
            return _decision_to_dict(row)

    def list_decisions(
        self, target_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self.store.session() as s:
            stmt = select(PromotionDecision).order_by(PromotionDecision.created_at.desc())
            if target_id:
                stmt = stmt.where(PromotionDecision.target_id == target_id)
            stmt = stmt.limit(limit)
            return [_decision_to_dict(r) for r in s.scalars(stmt)]


# --- dict dönüştürücüler ---------------------------------------------------
def _dataset_to_dict(r: DatasetVersion) -> dict[str, Any]:
    return {
        "dataset_version_id": r.dataset_version_id,
        "name": r.name,
        "path": r.path,
        "source_type": r.source_type,
        "content_hash": r.content_hash,
        "n_records": r.n_records,
        "domain_distribution": json.loads(r.domain_distribution_json or "{}"),
        "quality_score": r.quality_score,
        "approval_status": r.approval_status,
        "created_at": r.created_at,
    }


def _rag_index_to_dict(r: RagIndexVersion) -> dict[str, Any]:
    return {
        "rag_index_version_id": r.rag_index_version_id,
        "vector_db": r.vector_db,
        "collection_name": r.collection_name,
        "embedding_model": r.embedding_model,
        "chunking_strategy": r.chunking_strategy,
        "n_chunks": r.n_chunks,
        "n_papers": r.n_papers,
        "created_at": r.created_at,
    }


def _embedding_to_dict(r: EmbeddingModelVersion) -> dict[str, Any]:
    return {
        "embedding_model_id": r.embedding_model_id,
        "model_name": r.model_name,
        "dimension": r.dimension,
        "provider": r.provider,
        "local_path": r.local_path,
        "created_at": r.created_at,
    }


def _reward_to_dict(r: RlmRewardVersion) -> dict[str, Any]:
    return {
        "reward_version_id": r.reward_version_id,
        "name": r.name,
        "dataset_path": r.dataset_path,
        "method": r.method,
        "n_examples": r.n_examples,
        "secret_scanned": r.secret_scanned,
        "pii_scanned": r.pii_scanned,
        "created_at": r.created_at,
    }


def _decision_to_dict(r: PromotionDecision) -> dict[str, Any]:
    return {
        "decision_id": r.decision_id,
        "target_type": r.target_type,
        "target_id": r.target_id,
        "from_status": r.from_status,
        "to_status": r.to_status,
        "decision": r.decision,
        "reason": r.reason,
        "approved_by": r.approved_by,
        "created_at": r.created_at,
    }
