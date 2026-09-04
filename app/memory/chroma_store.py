"""ChromaDB persistent vector memory.

Stores paper chunks (and later: summaries, strategy notes, failure notes).
IDs are the same stable chunk_ids used in SQLite so the two stores stay linked.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "paper_chunks"
_CHROMA_LOCK = threading.RLock()
_SHARED: dict[tuple[str, str], tuple[Any, Any]] = {}


class ChromaStore:
    def __init__(self, path: str | Path | None = None, collection: str = _DEFAULT_COLLECTION):
        settings = get_settings()
        self.path = Path(path) if path else settings.chroma_dir
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection
        self._client: Any = None
        self._collection: Any = None

    def _ensure(self) -> Any:
        with _CHROMA_LOCK:
            if self._collection is not None:
                return self._collection
            key = (str(self.path.resolve()), self.collection_name)
            shared = _SHARED.get(key)
            if shared is None:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                client = chromadb.PersistentClient(
                    path=str(self.path),
                    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
                )
                collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                shared = (client, collection)
                _SHARED[key] = shared
            self._client, self._collection = shared
            return self._collection

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        with _CHROMA_LOCK:
            col = self._ensure()
            col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info("ChromaDB'ye %d chunk yazıldı.", len(ids))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 6,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        with _CHROMA_LOCK:
            col = self._ensure()
            res = col.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        out: list[dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, cid in enumerate(ids):
            out.append(
                {
                    "chunk_id": cid,
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out

    def delete_by_paper(self, paper_id: str) -> None:
        """Bir makaleye ait tüm chunk'ları koleksiyondan sil (force re-index temizliği)."""
        with _CHROMA_LOCK:
            self._ensure().delete(where={"paper_id": paper_id})

    def count(self) -> int:
        with _CHROMA_LOCK:
            return int(self._ensure().count())

    def get_all(self, page: int = 5000) -> list[dict[str, Any]]:
        """Tüm chunk'ları döndür (BM25 korpus indeksi kurmak için).

        Her öğe: {chunk_id, document, metadata}. Boş koleksiyonda boş liste.

        SAYFALANIR (limit/offset): tek `get()` çağrısı tüm id'leri tek SQL sorgusuna
        koyar → büyük korpusta (>~30k chunk) SQLite "too many SQL variables" hatası
        verir ve get_all KOMPLE BAŞARISIZ olurdu (BM25 hibrit sessizce ölür, her sorgu
        ~20s boşa gider). Sayfalama bu limiti aşar; davranış küçük korpusta aynıdır.
        """
        with _CHROMA_LOCK:
            col = self._ensure()
            out: list[dict[str, Any]] = []
            offset = 0
            while True:
                res = col.get(include=["documents", "metadatas"], limit=page, offset=offset)
                ids = res.get("ids", []) or []
                if not ids:
                    break
                docs = res.get("documents", []) or []
                metas = res.get("metadatas", []) or []
                for i, cid in enumerate(ids):
                    out.append(
                        {
                            "chunk_id": cid,
                            "document": docs[i] if i < len(docs) else "",
                            "metadata": metas[i] if i < len(metas) else {},
                        }
                    )
                if len(ids) < page:
                    break
                offset += page
        return out

    def reset(self) -> None:
        with _CHROMA_LOCK:
            self._ensure()
            self._client.reset()
            _SHARED.pop((str(self.path.resolve()), self.collection_name), None)
            self._collection = None
