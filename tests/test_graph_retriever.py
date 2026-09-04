"""Graf-tabanlı retrieval (SPRIG-lite) testleri — term–chunk graf + PPR, deterministik offline."""

from __future__ import annotations

from app.memory.graph_retriever import (
    build_graph,
    extract_terms,
    graph_rank,
    personalized_pagerank,
    seed_weights_from_ids,
)


def test_extract_terms_keeps_short_acronyms_drops_stopwords() -> None:
    terms = extract_terms("RSI momentum and the ATR signal for volatility")
    assert "rsi" in terms and "atr" in terms  # 3-karakter finans kısaltmaları korunur
    assert "and" not in terms and "the" not in terms and "for" not in terms  # durak-kelime


def test_build_graph_postings() -> None:
    g = build_graph(
        {"c1": "rsi momentum strategy", "c2": "rsi volatility regime"},
        max_df_ratio=1.0,  # pruning yok → paylaşılan terim korunur
    )
    assert g.term_chunks["rsi"] == {"c1", "c2"}
    assert g.term_chunks["momentum"] == {"c1"}
    assert "rsi" in g.chunk_terms["c1"] and "rsi" in g.chunk_terms["c2"]


def test_hub_pruning_drops_overcommon_terms() -> None:
    # "common" 4/4 chunk'ta → hub (ratio 0.5, max_df=2) → atılır; benzersizler kalır.
    chunks = {
        "c1": "common alpha",
        "c2": "common beta",
        "c3": "common gamma",
        "c4": "common delta",
    }
    g = build_graph(chunks, max_df_ratio=0.5)
    assert "common" not in g.term_chunks
    assert "alpha" in g.term_chunks


def test_ppr_seeded_propagation_reaches_connected_not_unrelated() -> None:
    """Tohumlanmamış ama paylaşılan terimle bağlı chunk skor alır; alakasız almaz."""
    chunks = {
        "c1": "rsi momentum strategy",
        "c2": "rsi volatility filter",  # c1 ile 'rsi' paylaşır
        "c3": "cooking pasta recipe",  # hiç paylaşmaz
    }
    g = build_graph(chunks, max_df_ratio=1.0)
    scores = personalized_pagerank(g, {"c1": 1.0})
    assert scores["c1"] > 0
    assert scores.get("c2", 0.0) > 0.0  # çok-hop: 'rsi' üzerinden erişildi
    assert scores.get("c3", 0.0) == 0.0  # bağlantısız → erişilemez
    assert scores["c1"] > scores["c2"]  # tohum en yüksek


def test_ppr_empty_seeds_returns_empty() -> None:
    g = build_graph({"c1": "rsi momentum"}, max_df_ratio=1.0)
    assert personalized_pagerank(g, {}) == {}


def test_ppr_deterministic() -> None:
    g = build_graph(
        {"c1": "rsi momentum", "c2": "rsi atr", "c3": "atr volatility"}, max_df_ratio=1.0
    )
    a = personalized_pagerank(g, {"c1": 1.0})
    b = personalized_pagerank(g, {"c1": 1.0})
    assert a == b


def test_graph_rank_orders_and_excludes_unreachable() -> None:
    chunks = {
        "c1": "rsi momentum strategy",
        "c2": "rsi volatility filter",
        "c3": "cooking pasta recipe",
    }
    ranked = graph_rank(chunks, {"c1": 1.0}, top_k=5, max_df_ratio=1.0)
    assert ranked[0] == "c1"
    assert "c2" in ranked
    assert "c3" not in ranked  # skor 0 → listeye girmez


def test_seed_weights_from_ids_descending() -> None:
    assert seed_weights_from_ids(["a", "b", "c"]) == {"a": 3.0, "b": 2.0, "c": 1.0}
