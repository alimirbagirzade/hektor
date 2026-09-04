"""FlashRank reranker — CPU'da HIZLI cross-encoder reranking (ONNX-int8).

`bge-reranker-base` (sentence-transformers/PyTorch) bu GPU'suz CPU'da ~24 adayı
**>15s** rerank ediyordu (kullanılamaz, bkz. reports/rag_retrieval_ab_findings.md).
FlashRank ONNX-int8 cross-encoder'ları (torch GEREKMEZ) aynı işi **~30-100ms**'de
yapar (web-araştırma: bağımsız CPU bench ~31ms/16 aday). Bu yüzden "reranking CPU'da
kullanılamaz" kararı FlashRank için GEÇERSİZ — yeniden ölçülmeli (reports/rag_deep_
research_roadmap.md, Zincir 3).

Arayüz `Reranker`/`CrossEncoderReranker` ile aynı: `rerank(query, chunks)->chunks`.
flashrank kurulu değilse veya model yüklenemezse sessizce heuristik `Reranker`'a düşer
(sistem her zaman çalışır). Model modül düzeyinde cache'lenir (her sorguda yeniden yok).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from app.config import get_settings
from app.memory.reranker import Reranker
from app.memory.retrieval_service import RetrievedChunk

logger = logging.getLogger(__name__)

# Modül düzeyi model cache: {model_name: Ranker | None}. None = yüklenemedi (bir daha deneme).
_ranker_cache: dict[str, Any] = {}


def _load_ranker(model_name: str) -> Any:
    """FlashRank Ranker'ı (cache'li) yükle. Başarısızsa None (graceful)."""
    if model_name in _ranker_cache:
        return _ranker_cache[model_name]
    ranker: Any = None
    try:
        from flashrank import Ranker

        ranker = Ranker(model_name=model_name)
        logger.info("FlashRank reranker yüklendi: %s", model_name)
    except ImportError:
        logger.warning(
            "flashrank kurulu değil — heuristik reranker'a düşülüyor. Kur: uv pip install flashrank"
        )
    except Exception as exc:  # model indirme/yükleme hatası
        logger.warning("FlashRank yüklenemedi (%s): %s — heuristiğe düşülüyor.", model_name, exc)
    _ranker_cache[model_name] = ranker
    return ranker


class FlashRankReranker:
    """FlashRank (ONNX-int8 cross-encoder) ile reranking; yoksa heuristiğe düşer.

    Args:
        model_name: FlashRank model adı. Verilmezse settings (`rag_flashrank_model`).
        fallback: Model yokken sıralayıcı (varsayılan `Reranker`).
        ranker: Test/enjeksiyon için hazır ranker (`.rerank(RerankRequest)->[{id,score}]`).
    """

    def __init__(
        self,
        model_name: str | None = None,
        fallback: Reranker | None = None,
        ranker: Any | None = None,
    ) -> None:
        self.settings = get_settings()
        self.model_name = model_name or self.settings.rag_flashrank_model
        self.fallback = fallback or Reranker()
        self._injected_ranker = ranker

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        ranker = (
            self._injected_ranker
            if self._injected_ranker is not None
            else _load_ranker(self.model_name)
        )
        if ranker is None:
            return self.fallback.rerank(query, chunks)

        try:
            passages = [{"id": i, "text": c.text or ""} for i, c in enumerate(chunks)]
            # RerankRequest yalnız flashrank ile gelir. Enjekte edilen ranker (test/offline)
            # için flashrank KURULU OLMAYABİLİR → aynı arayüzü (query/passages) taşıyan hafif
            # bir shim kullan; böylece enjekte ranker flashrank'sız da çalışır. Gerçek
            # flashrank kuruluysa onun tipi kullanılır (davranış birebir aynı).
            try:
                from flashrank import RerankRequest

                req: Any = RerankRequest(query=query, passages=passages)
            except ImportError:
                req = SimpleNamespace(query=query, passages=passages)
            ranked = ranker.rerank(req)
            # ranked: skora göre azalan {id,text,score}. id → orijinal index.
            order = [int(r["id"]) for r in ranked if 0 <= int(r["id"]) < len(chunks)]
        except Exception as exc:  # çıkarım/şema hatası → heuristik
            logger.warning("FlashRank rerank hatası: %s — heuristiğe düşülüyor.", exc)
            return self.fallback.rerank(query, chunks)

        seen = set(order)
        # eksik kalan (güvenlik) id'leri sona ekle → hiç chunk düşmesin
        order += [i for i in range(len(chunks)) if i not in seen]
        return [chunks[i] for i in order]
