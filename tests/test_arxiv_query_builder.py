"""arXiv düz sorgu → alan önekli AND + finans kategorisi (2026-10-17 alaka düzeltmesi)."""

from __future__ import annotations

from app.ingestion.arxiv_fetcher import FINANCE_CATEGORIES, build_search_query


def test_plain_terms_are_anded_and_category_limited():
    q = build_search_query("backtest overfitting")
    assert q.startswith("(all:backtest AND all:overfitting) AND (")
    assert all(f"cat:{c}" in q for c in FINANCE_CATEGORIES)


def test_finance_filter_can_be_disabled():
    assert build_search_query("backtest overfitting", finance_only=False) == (
        "all:backtest AND all:overfitting"
    )


def test_advanced_queries_pass_through_unchanged():
    for q in ("ti:momentum AND cat:q-fin.TR", "all:a OR all:b", "au:Lopez"):
        assert build_search_query(q) == q


def test_hyphenated_terms_kept_and_empty_safe():
    assert build_search_query("mean-reversion", finance_only=False) == "all:mean-reversion"
    assert build_search_query("") == ""
