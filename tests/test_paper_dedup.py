"""Paper-düzeyi dedup testleri — aynı makalenin RAG'a 2 kez girmesini engelle.

`find_paper_by_title` aynı başlıklı makaleyi (farklı bytes/hash olsa bile) yakalar;
`_normalize_title` noktalama/boşluk/case farklarına dayanıklıdır.
"""

from __future__ import annotations

from app.memory.sqlite_store import SqliteStore, _normalize_title


def test_normalize_title_strips_punctuation_and_case() -> None:
    assert _normalize_title("Deep Learning, for Trading!") == "deep learning for trading"
    assert _normalize_title("  ATR &  Momentum--Filter  ") == "atr momentum filter"


def test_find_paper_by_title_matches_despite_formatting() -> None:
    store = SqliteStore()
    store.upsert_paper(
        paper_id="pdedup_a",
        file_hash="hash_a",
        source_path="x/a.pdf",
        title="Quantum Momentum Strategies in Equity Markets",
    )
    # Farklı case + noktalama + tire → yine de aynı makale olarak eşleşmeli.
    dup = store.find_paper_by_title("quantum-momentum STRATEGIES, in equity markets.")
    assert dup is not None
    assert dup.paper_id == "pdedup_a"


def test_find_paper_by_title_no_false_match() -> None:
    store = SqliteStore()
    store.upsert_paper(
        paper_id="pdedup_b",
        file_hash="hash_b",
        source_path="x/b.pdf",
        title="Volatility Clustering Review",
    )
    assert store.find_paper_by_title("Zzz Tamamen Farkli Benzersiz Baslik 9f3a Xyz") is None


def test_find_paper_by_title_ignores_short_title() -> None:
    # Çok kısa/güvenilmez başlık (<12 karakter normalize) → dedup uygulanmaz.
    store = SqliteStore()
    store.upsert_paper(paper_id="pdedup_c", file_hash="hash_c", source_path="x/c.pdf", title="RSI")
    assert store.find_paper_by_title("RSI") is None
