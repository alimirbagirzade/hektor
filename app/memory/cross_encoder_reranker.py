"""Cross-encoder reranker (Faz A8) — graceful fallback'li.

Cross-encoder, (soru, chunk) çiftini birlikte puanlar; bi-encoder/heuristik
sıralayıcıdan daha doğru ama daha ağırdır (model indirme + CPU latency). Bu yüzden
OPT-IN'dir (settings.rag_cross_encoder). `sentence-transformers` kurulu değilse ya
da model yüklenemezse, sessizce heuristik `Reranker`'a düşer — sistem her zaman çalışır.

Arayüz `Reranker.rerank(query, chunks)` ile aynı → `RerankingRetriever` ikisini de
kullanabilir. Model modül düzeyinde cache'lenir (her sorguda yeniden yüklenmez).

Açmak için:  ACHILLES_RAG_CROSS_ENCODER=true  +  uv pip install sentence-transformers
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.memory.reranker import Reranker
from app.memory.retrieval_service import RetrievedChunk

logger = logging.getLogger(__name__)

# Modül düzeyi model cache: {model_name: model | None}. None = yüklenemedi (bir daha deneme).
_model_cache: dict[str, Any] = {}


def _load_model(model_name: str) -> Any:
    """Cross-encoder modelini (cache'li) yükle. Başarısızsa None döndür (graceful)."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    model: Any = None
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        logger.info("Cross-encoder yüklendi: %s", model_name)
    except ImportError:
        logger.warning(
            "sentence-transformers kurulu değil — cross-encoder atlanıyor "
            "(heuristik reranker'a düşülüyor). Açmak için: uv pip install sentence-transformers"
        )
    except Exception as exc:  # model indirme/yükleme hatası
        logger.warning(
            "Cross-encoder yüklenemedi (%s): %s — heuristiğe düşülüyor.", model_name, exc
        )
    _model_cache[model_name] = model
    return model


class CrossEncoderReranker:
    """Cross-encoder ile yeniden sıralama; model yoksa heuristik `Reranker`'a düşer.

    Args:
        model_name: HF cross-encoder model adı. Verilmezse settings'ten.
        fallback: Model yokken kullanılacak sıralayıcı (varsayılan `Reranker`).
        model: Test/enjeksiyon için hazır model (`.predict(pairs)->skorlar`). Verilirse
            yükleme atlanır.
    """

    def __init__(
        self,
        model_name: str | None = None,
        fallback: Reranker | None = None,
        model: Any | None = None,
    ) -> None:
        self.settings = get_settings()
        self.model_name = model_name or self.settings.rag_cross_encoder_model
        self.fallback = fallback or Reranker()
        self._injected_model = model

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = (
            self._injected_model
            if self._injected_model is not None
            else _load_model(self.model_name)
        )
        if model is None:
            return self.fallback.rerank(query, chunks)

        pairs = [(query, c.text or "") for c in chunks]
        try:
            scores = model.predict(pairs)
        except Exception as exc:  # çıkarım hatası → heuristik
            logger.warning("Cross-encoder predict hatası: %s — heuristiğe düşülüyor.", exc)
            return self.fallback.rerank(query, chunks)

        order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
        return [chunks[i] for i in order]
