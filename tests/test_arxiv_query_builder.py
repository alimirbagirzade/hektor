"""arXiv sorgu kurucusu + çağıran kapsamı (2026-10-17 alaka düzeltmesi ve Kademe 2 ARX-1..5)."""

from __future__ import annotations

import urllib.parse

import pytest

from app.ingestion import arxiv_fetcher as af
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


def test_lowercase_words_and_colons_are_plain_queries():
    # ARX-4: küçük harfli 'and' ya da rastgele 'kelime:' gelişmiş sorgu SAYILMAZ.
    assert build_search_query("momentum and mean", finance_only=False) == (
        "all:momentum AND all:mean"
    )
    assert build_search_query("Note: momentum", finance_only=False) == ("all:Note AND all:momentum")


def test_symbols_do_not_become_required_fragments():
    # ARX-3: tek karakter / yalnız rakam / baştaki '-' / tekrar terim olmaz.
    assert build_search_query("S&P 500 momentum", finance_only=False) == "all:momentum"
    assert build_search_query("GARCH(1,1) volatility", finance_only=False) == (
        "all:GARCH AND all:volatility"
    )
    assert build_search_query("momentum -crypto", finance_only=False) == (
        "all:momentum AND all:crypto"
    )
    assert build_search_query("mean-reversion", finance_only=False) == "all:mean-reversion"
    assert build_search_query("") == ""


class _Resp:
    text = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    status_code = 200

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def captured(monkeypatch):
    seen: list[str] = []

    def fake_get(client, url, *a, **k):
        seen.append(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["search_query"][0])
        return _Resp()

    monkeypatch.setattr(af, "polite_get", fake_get)
    return seen


def test_default_callers_send_raw_query(captured):
    # ARX-1/ARX-2/ARX-5: yöntem tarayıcıları, araştırma işçisi ve kaynak-tamamlayıcı
    # varsayılanı kullanır → sorgu AYNEN gider (finans kategorisi yok).
    af.search_arxiv("LoRA low-rank adaptation parameter efficient fine-tuning")
    assert captured == ["LoRA low-rank adaptation parameter efficient fine-tuning"]


def test_trading_callers_get_finance_filter(captured):
    af.search_arxiv("backtest overfitting", trading_query=True)
    assert captured[0].startswith("(all:backtest AND all:overfitting) AND (cat:q-fin.TR")


def test_error_feed_entry_is_not_a_paper(monkeypatch):
    class _Err(_Resp):
        text = (
            '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            "<id>http://arxiv.org/api/errors#incorrect_id_format</id><title>Error</title>"
            "</entry></feed>"
        )

    monkeypatch.setattr(af, "polite_get", lambda *a, **k: _Err())
    assert af.search_arxiv("x") == []
