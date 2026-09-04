"""BM25 indeksi testleri."""

from __future__ import annotations

from app.memory.bm25_index import BM25Index


def test_basic_search() -> None:
    """3 belge ekle; ilgili belge ilk sırada dönsün."""
    idx = BM25Index()
    idx.add_document("doc1", "ATR momentum volatility filter trading strategy")
    idx.add_document("doc2", "Sharpe ratio risk adjusted return portfolio")
    idx.add_document("doc3", "Machine learning neural network deep learning")

    results = idx.search("volatility momentum trading", top_k=3)
    assert len(results) > 0
    top_id, top_score = results[0]
    assert top_id == "doc1", f"İlk sonuç doc1 olmalı, alınan: {top_id}"
    assert top_score > 0


def test_empty_index() -> None:
    """Boş indekste arama boş liste döndürmeli."""
    idx = BM25Index()
    results = idx.search("any query", top_k=5)
    assert results == []


def test_top_k_limit() -> None:
    """top_k parametresi sonuç sayısını sınırlamalı."""
    idx = BM25Index()
    for i in range(10):
        idx.add_document(f"doc{i}", f"keyword{i} common term finance momentum")

    results = idx.search("keyword common term", top_k=3)
    assert len(results) <= 3


def test_unknown_query_returns_empty() -> None:
    """Hiçbir belgede olmayan sorgu boş liste döndürmeli."""
    idx = BM25Index()
    idx.add_document("doc1", "trading strategy momentum")
    idx.add_document("doc2", "risk management portfolio")

    results = idx.search("zzz_nonexistent_term_xyz", top_k=5)
    assert results == []


def test_scores_are_positive() -> None:
    """Tüm BM25 skorları pozitif olmalı."""
    idx = BM25Index()
    idx.add_document("doc1", "volatility regime filter ATR")
    idx.add_document("doc2", "momentum signal entry exit")

    results = idx.search("volatility ATR", top_k=5)
    for _, score in results:
        assert score > 0, f"Negatif skor: {score}"


def test_len() -> None:
    """İndeks uzunluğu belge sayısını döndürmeli."""
    idx = BM25Index()
    assert len(idx) == 0
    idx.add_document("d1", "text one")
    idx.add_document("d2", "text two")
    assert len(idx) == 2


def test_tie_break_deterministic() -> None:
    """Eşit skorlu belgeler doc_id'ye göre deterministik sıralanmalı (Kural 6).

    Aynı metinli belgeler özdeş skor alır; tie-break olmadan sıra set-iterasyonuna
    (PYTHONHASHSEED) bağlı kalır. doc_id artan tie-break ile sıra her zaman aynı olmalı.
    """
    idx = BM25Index()
    # Bilinçli olarak id-sırası dışında eklenmiş, özdeş metinli belgeler.
    for doc_id in ("d_c", "d_a", "d_e", "d_b", "d_d"):
        idx.add_document(doc_id, "momentum volatility regime filter")

    results = idx.search("momentum volatility", top_k=5)
    ids = [doc_id for doc_id, _ in results]
    scores = [round(s, 6) for _, s in results]
    # Hepsi özdeş skor → tamamı berabere.
    assert len(set(scores)) == 1, f"Skorlar berabere olmalı: {scores}"
    # Berabere → doc_id artan sırada deterministik.
    assert ids == sorted(ids), f"Berabere sıralama doc_id artan olmalı, alınan: {ids}"
