"""arXiv API kısma + 429 yeniden deneme (çevrimdışı; ağ ve uyku sahte)."""

from __future__ import annotations

import httpx
import pytest

from app.ingestion import arxiv_fetcher as af

_FEED = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


@pytest.fixture
def fake_net(monkeypatch):
    clock = {"t": 1000.0}
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        clock["t"] += s

    monkeypatch.setattr(af, "_sleep", sleep)
    monkeypatch.setattr(af, "_clock", lambda: clock["t"])
    monkeypatch.setattr(af, "_last_request", 0.0)
    statuses: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        code = statuses.pop(0) if statuses else 200
        return httpx.Response(code, text=_FEED, request=request)

    real = httpx.Client

    def client(*a, **kw):
        return real(*a, transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(af.httpx, "Client", client)
    return sleeps, statuses


def test_consecutive_queries_are_spaced(fake_net):
    sleeps, _ = fake_net
    af.search_arxiv("a")
    af.search_arxiv("b")
    af.search_arxiv("c")
    assert sleeps == [pytest.approx(af._MIN_INTERVAL_S)] * 2


def test_429_is_retried_then_succeeds(fake_net):
    sleeps, statuses = fake_net
    statuses.extend([429, 429])
    assert af.search_arxiv("q") == []
    assert sum(sleeps) >= 5.0 + 10.0


def test_persistent_429_raises_after_bounded_retries(fake_net):
    sleeps, statuses = fake_net
    statuses.extend([429] * 5)
    with pytest.raises(httpx.HTTPStatusError):
        af.search_arxiv("q")
    assert len(statuses) == 5 - (af._MAX_RETRIES + 1)
    assert max(sleeps) <= af._MAX_RETRY_WAIT_S


def test_pdf_downloads_share_throttle_and_retry(fake_net, tmp_path):
    """Kademe 2 E-5: PDF indirmeleri de kısılır ve 429'da yeniden denenir."""
    sleeps, statuses = fake_net
    statuses.extend([429])
    with af.httpx.Client() as client:
        assert af.polite_get(client, "https://arxiv.org/pdf/1.pdf").status_code == 200
        af.polite_get(client, "https://arxiv.org/pdf/2.pdf")
    assert sleeps[0] >= 5.0 and sleeps[-1] == pytest.approx(af._MIN_INTERVAL_S)
