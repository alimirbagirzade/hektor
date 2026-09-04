"""FlashRankReranker testleri — çevrimdışı (enjekte edilen sahte ranker; model indirmez).

FlashRank ONNX cross-encoder'ı CPU'da bge-reranker'dan hızlıdır ama GERÇEK latency bu
donanımda ölçülmeli (bkz. reports/rag_deep_research_roadmap.md Zincir 3). Bu testler
yalnız ENTEGRASYON sözleşmesini doğrular: sıra ranker'dan gelir, tüm chunk korunur,
hata/eksik durumda heuristiğe düşülür (sistem her zaman çalışır).
"""

from __future__ import annotations

from app.memory.flashrank_reranker import FlashRankReranker
from app.memory.retrieval_service import RetrievedChunk


def _chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            paper_id="p",
            text=f"belge metni {i}",
            page_number=1,
            section_name=None,
            title=None,
            distance=0.5,
        )
        for i in range(n)
    ]


class _FakeRanker:
    """flashrank.Ranker taklidi: passages'ı id'ye göre AZALAN sırada döndürür."""

    def rerank(self, req):
        return [
            {"id": p["id"], "score": float(-p["id"])}
            for p in sorted(req.passages, key=lambda x: -x["id"])
        ]


class _BrokenRanker:
    def rerank(self, req):
        raise RuntimeError("onnx patladı")


def test_flashrank_reorders_by_ranker() -> None:
    """Sıra ranker'dan gelir (id azalan → c3,c2,c1,c0); tüm chunk korunur."""
    chunks = _chunks(4)
    out = FlashRankReranker(ranker=_FakeRanker()).rerank("soru", chunks)
    assert [c.chunk_id for c in out] == ["c3", "c2", "c1", "c0"]
    assert len(out) == 4


def test_flashrank_empty() -> None:
    assert FlashRankReranker(ranker=_FakeRanker()).rerank("soru", []) == []


def test_flashrank_falls_back_on_error() -> None:
    """Ranker patlarsa heuristik Reranker'a düşer — chunk DÜŞMEZ (sistem çalışır)."""
    chunks = _chunks(3)
    out = FlashRankReranker(ranker=_BrokenRanker()).rerank("soru", chunks)
    assert {c.chunk_id for c in out} == {"c0", "c1", "c2"}


def test_flashrank_partial_ids_preserved() -> None:
    """Ranker eksik id döndürse bile tüm chunk'lar sonuçta kalır (sona eklenir)."""

    class _PartialRanker:
        def rerank(self, req):
            # yalnız ilk iki id'yi döndür
            return [{"id": p["id"], "score": 1.0} for p in req.passages[:2]]

    chunks = _chunks(5)
    out = FlashRankReranker(ranker=_PartialRanker()).rerank("soru", chunks)
    assert {c.chunk_id for c in out} == {f"c{i}" for i in range(5)}
    assert len(out) == 5
