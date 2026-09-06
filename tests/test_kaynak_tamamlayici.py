"""Kaynak Tamamlayıcı testleri — çevrimdışı (arXiv / disk / LLM yok).

Alaka kapısı, bugün ölçülen gerçek çöp örnekleriyle sınanır: "path integral" araması
parçacık fiziği, "risk management" araması şehir sel yönetimi getirmişti.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.research.kaynak_tamamlayici import (
    Makale,
    alaka_skoru,
    kaynak_terimleri,
    run_tamamlayici,
    secilecek_makaleler,
    sorgu_uret,
    tokenize,
)

KART = {
    "domain": "Quantum Finance",
    "methods": ["path integral", "Black-Scholes option pricing"],
    "main_claim": "Path integrals price exotic options under stochastic volatility.",
}


def test_tokenize_stop_ve_cogul() -> None:
    t = tokenize("A Novel Approach to Option Pricing with Path Integrals and Models")
    assert "novel" not in t and "approach" not in t and "model" not in t
    assert t == ["option", "pricing", "integral"]  # 'integrals' → 'integral', sıra korunur


def test_sorgu_uret_baslik_ve_kart() -> None:
    q = sorgu_uret("A Path Integral Approach to Option Pricing with Stochastic Volatility", KART)
    assert q.split()[:4] == ["integral", "option", "pricing", "stochastic"]
    assert len(q.split()) <= 8


def test_sorgu_uret_sablon_basligi_atlar_kartla_kurtarir() -> None:
    q = sorgu_uret("Published as a conference paper at ICLR 2023", KART)
    assert "published" not in q and "conference" not in q
    assert "quantum" in q and "finance" in q


def test_sorgu_uret_yetersizse_bos() -> None:
    assert sorgu_uret("Preprint", None) == ""
    assert sorgu_uret("Markov", None) == ""  # tek terim → uydurma sorgu yok


def test_alaka_kapisi_cop_reddeder_ilgili_gecirir() -> None:
    kaynak = kaynak_terimleri("Path Integral Approach to Option Pricing", KART)
    cop = alaka_skoru(
        kaynak,
        "Complex Monopoles in the Path Integral",
        "We study monopole configurations in gauge theory path integrals.",
    )
    ilgili = alaka_skoru(
        kaynak,
        "Path integral approach to Asian options in the Black-Scholes model",
        "We price Asian options via path integrals under stochastic volatility.",
    )
    assert cop < 3 <= ilgili


def test_secilecek_makaleler_esik_defter_sinir() -> None:
    ms = [
        Makale("p_iyi", "İyi", KART, 82.0),
        Makale("p_dusuk", "Düşük", KART, 31.0),
        Makale("p_kartsiz", "Kartsız", None, None),
        Makale("p_dolu", "Defter dolu", KART, 10.0),
        Makale("p_cok_dusuk", "Çok düşük", KART, 5.0),
    ]
    sec = secilecek_makaleler(ms, esik=50, defter={"p_dolu": 2}, max_deneme=2, max_paper=2)
    assert [m.paper_id for m in sec] == ["p_cok_dusuk", "p_dusuk"]  # en düşük önce, sınır 2
    sec = secilecek_makaleler(ms, esik=50, defter={}, max_deneme=2, max_paper=10)
    assert [m.paper_id for m in sec][-1] == "p_kartsiz"  # kartsız sona


class _Store:
    def __init__(self, makaleler: list[Makale]) -> None:
        self._m = makaleler

    def list_papers(self):
        return [SimpleNamespace(paper_id=m.paper_id, title=m.title) for m in self._m]

    def list_cards(self):
        return [
            {"paper_id": m.paper_id, "card_json": m.card_json, "review_status": "approved"}
            for m in self._m
            if m.card_json is not None
        ]

    def get_comprehension_score(self, pid: str):
        m = next(x for x in self._m if x.paper_id == pid)
        return None if m.skor is None else SimpleNamespace(total_score=m.skor)


def _entry(aid: str, title: str, abstract: str):
    return SimpleNamespace(arxiv_id=aid, title=title, abstract=abstract, pdf_url="")


def _kur(tmp_path: Path):
    store = _Store([Makale("p1", "Path Integral Approach to Option Pricing", KART, 20.0)])
    raw = tmp_path / "raw_pdf"
    raw.mkdir()
    (raw / "arxiv_0906.4456.pdf").write_bytes(b"%PDF-zaten")  # zaten korpusta

    def searcher(q: str, max_results: int = 6):
        return [
            _entry("hep-th/9809072", "Complex Monopoles in the Path Integral", "gauge theory"),
            _entry(
                "0906.4456", "Path integral approach to Asian options", "options path integrals"
            ),
            _entry(
                "1510.04370",
                "Extending Black-Scholes option pricing with path integrals",
                "We price options under stochastic volatility via path integral methods.",
            ),
            _entry(
                "2502.12174",
                "Robust blue-green urban flood risk management",
                "cities flooding infrastructure",
            ),
        ]

    indirme: list[str] = []

    def fetcher(q: str, entries=None, dest_dir=None):
        out = []
        for e in entries or []:
            indirme.append(e.arxiv_id)
            out.append(
                SimpleNamespace(arxiv_id=e.arxiv_id, title=e.title, skipped=False, error=None)
            )
        return out

    ingest_calls: list[int] = []

    def ingest():
        ingest_calls.append(1)
        return [SimpleNamespace(skipped=False)]

    return store, raw, searcher, fetcher, indirme, ingest, ingest_calls


def test_run_alaka_kapisi_yalniz_ilgiliyi_indirir(tmp_path: Path) -> None:
    store, raw, searcher, fetcher, indirme, ingest, ingest_calls = _kur(tmp_path)
    r = run_tamamlayici(
        esik=50,
        searcher=searcher,
        fetcher=fetcher,
        ingest=ingest,
        store=store,
        raw_dir=raw,
        state_path=tmp_path / "defter.json",
    )
    assert r["ok"] and r["secilen"] == 1
    assert indirme == ["1510.04370"]  # monopol + sel → alaka yok; 0906.4456 → zaten korpusta
    m = r["makaleler"][0]
    sebepler = {x.get("id"): x["sebep"] for x in m["reddedilen"]}
    assert sebepler["hep-th/9809072"].startswith("alaka yok")
    assert sebepler["2502.12174"].startswith("alaka yok")
    assert sebepler["0906.4456"] == "zaten korpusta"
    assert r["indirilen"] == 1 and r["indekslenen"] == 1 and ingest_calls == [1]
    defter = json.loads((tmp_path / "defter.json").read_text(encoding="utf-8"))
    assert defter["deneme"]["p1"] == 1


def test_run_dry_run_indirmez_defter_yazmaz(tmp_path: Path) -> None:
    store, raw, searcher, fetcher, indirme, ingest, ingest_calls = _kur(tmp_path)
    r = run_tamamlayici(
        esik=50,
        dry_run=True,
        searcher=searcher,
        fetcher=fetcher,
        ingest=ingest,
        store=store,
        raw_dir=raw,
        state_path=tmp_path / "defter.json",
    )
    assert r["dry_run"] and indirme == [] and ingest_calls == []
    assert r["makaleler"][0]["alinan"][0]["id"] == "1510.04370"
    assert not (tmp_path / "defter.json").exists()


def test_run_defter_dolunca_bir_daha_aramaz(tmp_path: Path) -> None:
    store, raw, searcher, fetcher, indirme, ingest, _ = _kur(tmp_path)
    p = tmp_path / "defter.json"
    p.write_text(json.dumps({"deneme": {"p1": 2}}), encoding="utf-8")
    r = run_tamamlayici(
        searcher=searcher, fetcher=fetcher, ingest=ingest, store=store, raw_dir=raw, state_path=p
    )
    assert r["secilen"] == 0 and indirme == []


def test_run_ag_yoksa_cokmez(tmp_path: Path) -> None:
    store, raw, _s, fetcher, _indirme, ingest, _ = _kur(tmp_path)

    def bozuk(q, max_results=6):
        raise ConnectionError("arxiv yok")

    r = run_tamamlayici(
        searcher=bozuk,
        fetcher=fetcher,
        ingest=ingest,
        store=store,
        raw_dir=raw,
        state_path=tmp_path / "d.json",
    )
    assert r["ok"] and r["indirilen"] == 0 and r["makaleler"][0]["aday"] == 0


def test_agent_id_manifestte_kayitli() -> None:
    from app.agents.runtime import list_agents
    from app.research.kaynak_tamamlayici import AGENT_ID

    assert AGENT_ID in {a.agent_id for a in list_agents()}
