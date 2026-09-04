"""Sorgu yönlendirici + konveks füzyon testleri (deterministik, Chroma'sız).

classify_query + convex_fuse saf fonksiyon → tam birim test. RerankingRetriever'ın
SEMANTİK yolu (saf dense) stub base ile test edilir; LEXICAL yol (BM25 füzyon) canlı
Chroma gerektirir → A/B ölçümünde doğrulanır.
"""

from __future__ import annotations

from app.memory.query_router import classify_query, convex_fuse
from app.memory.retrieval_service import RetrievedChunk


# ---- classify_query ----
def test_empty_is_semantic() -> None:
    assert classify_query("") == "semantic"
    assert classify_query("   ") == "semantic"


def test_single_term_is_lexical() -> None:
    assert classify_query("GARCH") == "lexical"
    assert classify_query("Sharpe ratio") == "lexical"  # 2 kelime


def test_short_with_acronym_or_number_is_lexical() -> None:
    assert classify_query("what is the GARCH model") == "lexical"  # kısa + kısaltma
    assert classify_query("CVaR at 95% level") == "lexical"  # kısa + rakam
    assert classify_query('explain "regime switching"') == "lexical"  # tırnak


def test_long_natural_language_is_semantic() -> None:
    q = "how does volatility clustering affect the persistence of momentum signals over time"
    assert classify_query(q) == "semantic"


def test_long_even_with_acronym_is_semantic() -> None:
    # uzun (>6 kelime) → kısaltma olsa da semantik (dense kazanır)
    q = "discuss how the GARCH framework models conditional heteroskedasticity in returns"
    assert classify_query(q) == "semantic"


# ---- convex_fuse ----
def test_fuse_dense_only_when_bm25_empty() -> None:
    dense = {"a": 0.9, "b": 0.5, "c": 0.3}
    out = convex_fuse(dense, {}, alpha=0.7)
    assert out == ["a", "b", "c"]  # dense sırası korunur


def test_fuse_alpha_one_is_pure_dense() -> None:
    out = convex_fuse({"a": 0.2, "b": 0.9}, {"a": 0.9, "b": 0.1}, alpha=1.0)
    assert out[0] == "b"  # yalnız dense → b önde


def test_fuse_alpha_zero_is_pure_bm25() -> None:
    out = convex_fuse({"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}, alpha=0.0)
    assert out[0] == "b"  # yalnız bm25 → b önde


def test_fuse_combines_both_signals() -> None:
    # her iki listede de güçlü olan id en öne çıkmalı
    dense = {"x": 1.0, "y": 0.5, "z": 0.0}
    bm25 = {"x": 1.0, "y": 0.6, "z": 0.0}
    out = convex_fuse(dense, bm25, alpha=0.7)
    assert out[0] == "x"
    assert set(out) == {"x", "y", "z"}


def test_fuse_stable_tiebreak() -> None:
    # eşit skor → id'ye göre kararlı (determinizm, Kural 6)
    out = convex_fuse({"b": 0.5, "a": 0.5}, {}, alpha=0.7)
    assert out == ["a", "b"]


# ---- RerankingRetriever router (semantik yol — saf dense) ----
class _StubBase:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._c = chunks
        self.calls: list[tuple[str, int | None]] = []
        self.chroma = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self._c


def test_router_semantic_uses_pure_dense() -> None:
    from app.memory.reranking_retriever import RerankingRetriever

    chunks = [
        RetrievedChunk("c1", "p1", "t", 1, "Methods", "T", 0.2),
        RetrievedChunk("c2", "p1", "t", 1, "Methods", "T", 0.3),
    ]
    base = _StubBase(chunks)
    r = RerankingRetriever(base=base, router=True, enabled=True)  # type: ignore[arg-type]
    out = r.retrieve(
        "how does volatility clustering influence momentum persistence over long horizons",
        top_k=2,
    )
    assert out == chunks  # semantik → saf dense (rerank/füzyon yok)
    assert base.calls and base.calls[-1][1] == 2  # base.retrieve top_k=2 ile çağrıldı
