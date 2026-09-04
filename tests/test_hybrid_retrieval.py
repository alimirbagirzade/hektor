"""Hibrit retrieval (Faz A3) testleri — BM25 korpus + RerankingRetriever hybrid.

Çevrimdışı: sahte Chroma + stub base. Chroma/Ollama gerektirmez.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.memory.bm25_corpus import get_corpus_bm25, reset_cache
from app.memory.bm25_index import BM25Index
from app.memory.reranker import Reranker
from app.memory.reranking_retriever import RerankingRetriever
from app.memory.retrieval_service import RetrievedChunk


def _chunk(cid: str, text: str, distance: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        paper_id="p",
        text=text,
        page_number=1,
        section_name="results",
        title="T",
        distance=distance,
    )


def _dbchunk(cid: str, text: str, pid: str = "p", embedded: int = 1):
    """SQLite Chunk-benzeri (BM25 korpusu artık SQLite'tan kurulur)."""
    return SimpleNamespace(
        chunk_id=cid,
        paper_id=pid,
        chunk_index=0,
        section_name="results",
        page_number=1,
        text=text,
        embedded=embedded,
    )


class _FakeStore:
    """get_corpus_bm25'in SQLite kaynağını taklit eden stub (list_all_chunks + list_papers)."""

    def __init__(self, chunks: list) -> None:
        self._chunks = chunks

    def list_all_chunks(self) -> list:
        return self._chunks

    def list_papers(self) -> list:
        pids = {c.paper_id for c in self._chunks}
        return [SimpleNamespace(paper_id=p, title="T") for p in pids]


class _StubBase:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return list(self._chunks)


# --------------------------------------------------------------- bm25_corpus
def test_get_corpus_bm25_builds_and_searches() -> None:
    reset_cache()
    store = _FakeStore(
        [
            _dbchunk("p_c0", "ATR average true range volatility"),
            _dbchunk("p_c1", "Sharpe ratio risk adjusted return"),
        ]
    )
    bm25, chunks = get_corpus_bm25(store=store)
    assert bm25 is not None
    assert len(chunks) == 2
    hits = bm25.search("Sharpe ratio", top_k=2)
    assert hits and hits[0][0] == "p_c1"
    assert chunks["p_c1"].text.startswith("Sharpe")
    assert chunks["p_c1"].title == "T"  # paper başlığı SQLite'tan join'lendi
    reset_cache()


def test_get_corpus_bm25_rebuilds_on_equal_count_content_change() -> None:
    # REGRESYON (Kademe-2 bulgusu): cache yalnız chunk SAYISINA göre geçersizleşseydi,
    # eşit-sayıda içerik değişimi (chunk yerinde yeniden indekslenir / aynı sayıda chunk
    # üreten force re-index) SESSİZCE bayat indeks bırakır, eski sonuç dönerdi (Kural 7).
    # İçerik imzası (count + toplam karakter) bunu reset_cache OLMADAN da yakalamalı.
    reset_cache()
    store_v1 = _FakeStore(
        [
            _dbchunk("p_c0", "ATR average true range volatility"),
            _dbchunk("p_c1", "Sharpe ratio risk adjusted return"),
        ]
    )
    bm25_v1, _ = get_corpus_bm25(store=store_v1)
    assert bm25_v1 is not None
    hits_v1 = bm25_v1.search("Sharpe", top_k=2)
    assert hits_v1 and hits_v1[0][0] == "p_c1"  # ilk indeks: Sharpe eşleşir

    # AYNI chunk SAYISI (2), FARKLI içerik (farklı toplam karakter) → reset_cache YOK.
    # İmza tek başına bayatlığı yakalayıp indeksi yeniden kurmalı.
    store_v2 = _FakeStore(
        [
            _dbchunk("p_c0", "Kalman filter recursive state estimation for noisy series"),
            _dbchunk("p_c1", "Markov regime switching detection model with hidden states"),
        ]
    )
    bm25_v2, chunks_v2 = get_corpus_bm25(store=store_v2)
    assert chunks_v2["p_c1"].text.startswith("Markov")  # taze metin chunk haritasında
    hits_kalman = bm25_v2.search("Kalman", top_k=2)
    assert hits_kalman and hits_kalman[0][0] == "p_c0"  # yeni içerik indekslendi
    assert bm25_v2.search("Sharpe", top_k=2) == []  # eski (bayat) içerik artık eşleşmez
    reset_cache()


def test_get_corpus_bm25_excludes_unembedded_chunks() -> None:
    # LOW-2 (Kademe-2 av): embedded=0 chunk (Chroma'ya yazılamamış — yarım/başarısız ingest)
    # BM25 korpusuna ALINMAMALI. Aksi halde dense yolun görmediği metni hibrit/RRF/router'da
    # keyword adayı olarak alıntılardık → BUG-M6 koruması asimetrik kalır (Kural 7).
    reset_cache()
    store = _FakeStore(
        [
            _dbchunk("p_c0", "Sharpe ratio risk adjusted return", embedded=1),
            _dbchunk("p_c1", "Kalman filter unembedded ghost chunk", embedded=0),
        ]
    )
    bm25, chunks = get_corpus_bm25(store=store)
    assert bm25 is not None
    assert set(chunks) == {"p_c0"}  # yalnız gömülü chunk korpusta
    assert bm25.search("Sharpe", top_k=2)  # gömülü içerik aranabilir
    assert bm25.search("ghost", top_k=2) == []  # gömülmemiş hayalet chunk aranamaz
    reset_cache()


def test_reset_cache_acquires_build_lock() -> None:
    # MEDIUM-1 (Kademe-2 av): reset_cache() _build_lock ALTINDA çalışmalı — yoksa sürmekte olan
    # bir build'in (warm-up ~170s) son _cache.update'i reset'i EZER (lost-update) → ingest
    # sonrası eski korpus sessizce sunulur (Kural 7). Bu test reset'in kilidi gerçekten aldığını
    # doğrular: build'i taklit etmek için kilit dışarıda tutulurken reset ilerleyemez.
    import threading

    from app.memory import bm25_corpus

    reset_cache()
    assert bm25_corpus._build_lock.acquire(timeout=2)  # "in-flight build" taklidi
    done = threading.Event()
    t = threading.Thread(target=lambda: (reset_cache(), done.set()))
    t.start()
    try:
        assert not done.wait(0.3)  # kilit tutulurken reset BLOKLANIR (kilidi alıyor)
    finally:
        bm25_corpus._build_lock.release()
    assert done.wait(2)  # kilit bırakılınca reset tamamlanır
    t.join(2)
    reset_cache()


# NOT: `HybridRetriever` sınıfı kaldırıldı (hiçbir üretim yolu çağırmıyordu). Canlı
# hibrit yol `reranking_retriever._add_bm25_candidates` + `query_router.convex_fuse`;
# ikisinin de kendi testleri aşağıda/`test_query_router.py`'de.


def test_get_corpus_bm25_empty_returns_none() -> None:
    reset_cache()
    bm25, chunks = get_corpus_bm25(store=_FakeStore([]))
    assert bm25 is None
    assert chunks == {}


# --------------------------------------------------- RerankingRetriever hybrid
def test_hybrid_adds_bm25_keyword_candidate(monkeypatch) -> None:
    # Dense yalnız 'momentum' chunk'ı döner; BM25 korpusunda 'ATR' chunk'ı var ama
    # dense kaçırmış → hibrit onu eklemeli.
    dense = [_chunk("p_c0", "momentum persistence in returns", 0.2)]
    bm25 = BM25Index()
    bm25.add_document("p_c9", "ATR average true range filter for volatility")
    chunk_map = {"p_c9": _chunk("p_c9", "ATR average true range filter for volatility", None)}
    monkeypatch.setattr(
        "app.memory.bm25_corpus.get_corpus_bm25", lambda chroma=None: (bm25, chunk_map)
    )

    rr = RerankingRetriever(base=_StubBase(dense), reranker=Reranker(), enabled=True, hybrid=True)
    out = rr.retrieve("ATR filter", top_k=5)

    assert "p_c9" in {c.chunk_id for c in out}  # keyword adayı dahil edildi


def test_hybrid_graceful_when_corpus_empty(monkeypatch) -> None:
    dense = [_chunk("p_c0", "x momentum y", 0.2)]
    monkeypatch.setattr("app.memory.bm25_corpus.get_corpus_bm25", lambda chroma=None: (None, {}))
    rr = RerankingRetriever(base=_StubBase(dense), reranker=Reranker(), enabled=True, hybrid=True)
    out = rr.retrieve("ATR", top_k=5)
    assert [c.chunk_id for c in out] == ["p_c0"]  # dense-only'e düşer


def test_hybrid_off_skips_bm25(monkeypatch) -> None:
    called = {"n": 0}

    def _spy(chroma=None):
        called["n"] += 1
        return None, {}

    monkeypatch.setattr("app.memory.bm25_corpus.get_corpus_bm25", _spy)
    rr = RerankingRetriever(
        base=_StubBase([_chunk("p_c0", "momentum", 0.2)]),
        reranker=Reranker(),
        enabled=True,
        hybrid=False,
    )
    rr.retrieve("ATR", top_k=5)
    assert called["n"] == 0  # hybrid kapalı → BM25'e hiç gidilmez


# ----------------------------------------------------- RRF füzyon modu (opt-in)
def test_rrf_mode_fuses_dense_and_bm25(monkeypatch) -> None:
    # Dense sırası: c0, c1. BM25 sırası: c1, c9. c1 her iki listede de var → RRF onu
    # en üste taşımalı (uzlaşma); c9 BM25-only ama chunk_map'te metni var → eklenir.
    dense = [_chunk("p_c0", "momentum returns", 0.1), _chunk("p_c1", "volatility regime", 0.2)]
    bm25 = BM25Index()
    bm25.add_document("p_c1", "volatility regime atr")
    bm25.add_document("p_c9", "atr average true range")
    chunk_map = {
        "p_c1": _chunk("p_c1", "volatility regime atr", None),
        "p_c9": _chunk("p_c9", "atr average true range", None),
    }
    monkeypatch.setattr(
        "app.memory.bm25_corpus.get_corpus_bm25", lambda chroma=None: (bm25, chunk_map)
    )
    rr = RerankingRetriever(
        base=_StubBase(dense), reranker=Reranker(), enabled=True, hybrid=True, rrf=True
    )
    out = rr.retrieve("atr volatility", top_k=5)
    ids = [c.chunk_id for c in out]
    assert ids[0] == "p_c1"  # iki listede de üst → RRF tepeye taşır
    assert "p_c9" in ids  # BM25-only aday (metni var) dahil edildi


def test_rrf_mode_dense_only_when_bm25_empty(monkeypatch) -> None:
    dense = [_chunk("p_c0", "momentum", 0.2), _chunk("p_c1", "trend", 0.3)]
    monkeypatch.setattr("app.memory.bm25_corpus.get_corpus_bm25", lambda chroma=None: (None, {}))
    rr = RerankingRetriever(
        base=_StubBase(dense), reranker=Reranker(), enabled=True, hybrid=True, rrf=True
    )
    out = rr.retrieve("x", top_k=5)
    assert [c.chunk_id for c in out] == ["p_c0", "p_c1"]  # dense sırası korunur


# ----------------------------------------------------- Graf modu (SPRIG-lite, opt-in)
def test_graph_mode_pulls_multihop_chunk(monkeypatch) -> None:
    # Dense yalnız c0 döner; korpus grafında c1, c0 ile 'rsi' terimini paylaşır →
    # PPR (c0'dan tohumlu) c1'i yüzeye çıkarmalı, dense kaçırmış olsa bile.
    from app.memory.graph_retriever import build_graph

    dense = [_chunk("p_c0", "rsi momentum strategy", 0.2)]
    texts = {"p_c0": "rsi momentum strategy", "p_c1": "rsi volatility filter"}
    graph = build_graph(texts, max_df_ratio=1.0)
    corpus_chunks = {
        "p_c0": _chunk("p_c0", texts["p_c0"], None),
        "p_c1": _chunk("p_c1", texts["p_c1"], None),
    }
    monkeypatch.setattr(
        "app.memory.graph_corpus.get_corpus_graph", lambda chroma=None: (graph, corpus_chunks)
    )
    rr = RerankingRetriever(base=_StubBase(dense), enabled=True, graph=True)
    out = rr.retrieve("rsi", top_k=5)
    ids = {c.chunk_id for c in out}
    assert "p_c0" in ids and "p_c1" in ids  # graf çok-hop chunk'ı ekledi


def test_graph_mode_dense_only_when_graph_empty(monkeypatch) -> None:
    dense = [_chunk("p_c0", "momentum", 0.2), _chunk("p_c1", "trend", 0.3)]
    monkeypatch.setattr("app.memory.graph_corpus.get_corpus_graph", lambda chroma=None: (None, {}))
    rr = RerankingRetriever(base=_StubBase(dense), enabled=True, graph=True)
    out = rr.retrieve("x", top_k=5)
    assert [c.chunk_id for c in out] == ["p_c0", "p_c1"]  # graf yok → dense-only
