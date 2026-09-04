"""RerankingRetriever testleri — over-fetch + heuristik rerank + truncate.

Tamamen çevrimdışı: dense retrieval bir stub ile taklit edilir; gerçek
`Reranker` (LLM-free) kullanılır. Ollama/Chroma gerektirmez.
"""

from __future__ import annotations

from app.memory.reranker import Reranker
from app.memory.reranking_retriever import RerankingRetriever
from app.memory.retrieval_service import RetrievedChunk


def _chunk(cid: str, text: str, distance: float, section: str = "results") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        paper_id="paper1",
        text=text,
        page_number=1,
        section_name=section,
        title="Test Paper",
        distance=distance,
    )


class _StubBase:
    """Dense retrieval taklidi. Çağrılan `top_k`'yı kaydeder, ilk `top_k` chunk'ı döner."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_top_k: int | None = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.last_top_k = top_k
        return self._chunks[: (top_k or len(self._chunks))]


def test_overfetch_requests_wider_candidate_pool() -> None:
    """Rerank açıkken base'ten top_k * overfetch aday istenmeli."""
    base = _StubBase([_chunk(f"c{i}", "metin", 0.1 * i) for i in range(20)])
    rr = RerankingRetriever(base=base, reranker=Reranker(), overfetch=4, enabled=True)

    rr.retrieve("sorgu", top_k=3)

    assert base.last_top_k == 12  # 3 * 4


def test_truncates_to_top_k() -> None:
    """Geniş havuz çekilse de yalnızca top_k sonuç dönmeli."""
    base = _StubBase([_chunk(f"c{i}", "metin", 0.1 * i) for i in range(20)])
    rr = RerankingRetriever(base=base, reranker=Reranker(), overfetch=4, enabled=True)

    out = rr.retrieve("sorgu", top_k=3)

    assert len(out) == 3


def test_rerank_changes_order_formula_first() -> None:
    """Formül içeren chunk, daha kötü dense skoruna rağmen öne çıkmalı."""
    plain = _chunk("c_plain", "Momentum is a known factor.", distance=0.1)
    formula = _chunk(
        "c_formula",
        r"ATR is $ATR_t = \frac{1}{n}\sum TR_i$ true range.",
        distance=0.5,  # daha kötü dense skoru
    )
    base = _StubBase([plain, formula])
    rr = RerankingRetriever(base=base, reranker=Reranker(), overfetch=4, enabled=True)

    out = rr.retrieve("ATR formula", top_k=2)

    assert out[0].chunk_id == "c_formula"


def test_disabled_falls_back_to_plain_dense() -> None:
    """Rerank kapalıyken base.retrieve(top_k) aynen kullanılmalı (over-fetch yok)."""
    base = _StubBase([_chunk(f"c{i}", "metin", 0.1 * i) for i in range(20)])
    rr = RerankingRetriever(base=base, reranker=Reranker(), overfetch=4, enabled=False)

    out = rr.retrieve("sorgu", top_k=5)

    assert base.last_top_k == 5  # over-fetch yapılmadı
    assert len(out) == 5


def test_empty_base_returns_empty() -> None:
    """Base boş dönerse boş liste dönmeli (refusal davranışı RagAnswerer'da)."""
    rr = RerankingRetriever(base=_StubBase([]), reranker=Reranker(), enabled=True)
    assert rr.retrieve("sorgu", top_k=5) == []
