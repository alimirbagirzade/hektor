"""Başlık-dedup'ı şablon/banner satırlarında devre dışı bırakma testleri.

Çevrimdışı: sahte store/chroma/embedder + parse/chunk monkeypatch.

Bulgu (155 makalelik korpusta ölçüldü): `extract_metadata` ilk sayfanın ilk uzun satırını
başlık sanıyor. ICLR makalelerinde bu satır "Published as a conference paper at ICLR 2023"
banner'ı. `ingest_one`'daki başlık-dedup'ı bu yüzden ReAct (`2210.03629`) makalesini
Self-Consistency (`2203.11171`) makalesinin kopyası sanıp **sessizce** düşürdü — log'a
yalnız INFO düştü, hata verilmedi, 155 PDF → 153 makale.

Fix: başlık şablon/banner desenine uyuyorsa dosya adından türetilir; dedup böylece
gerçek başlıklar üzerinde çalışmaya devam eder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ingestion.metadata_extractor import PaperMetadata
from app.ingestion.paper_loader import DiscoveredPaper
from app.ingestion.pdf_parser import ParsedPdf
from app.memory.paper_indexer import is_boilerplate_title

BANNER = "Published as a conference paper at ICLR 2023"
GERCEK_BASLIK = "Hidden Markov Models Applied To Intraday Momentum Trading"


@pytest.mark.parametrize(
    "metin",
    [
        BANNER,
        "Under review as a conference paper at ICLR 2024",
        "Preprint, Under Review",
        "To appear in Journal of Finance",
        "Microsoft Word - working_paper_ML.docx",
        "Overleaf Example",
        "DjVu Document",
        "Springer Nature 2021 LATEX template",
        "Springer Texts in Statistics",
        "Copyright Cambridge University Press",
        "This is page i",
        "AU3772_C000.fm",
        "March 25, 2025",
        "25 March 2025",
        "SIXTH EDITION",
        "",
        "   ",
    ],
)
def test_sablon_satirlari_baslik_sayilmaz(metin: str) -> None:
    assert is_boilerplate_title(metin)


@pytest.mark.parametrize(
    "metin",
    [
        "LoRA Learns Less and Forgets Less",
        GERCEK_BASLIK,
        "Risk-Constrained Kelly Gambling",
        "What drives bitcoin? An approach from continuous local transfer entropy",
        "Entropy Analysis of Financial Time Series",
        "A Stochastic Processes Toolkit for Risk Management",
        # "Preprint" kelimesi BAŞTA değilse gerçek başlıktır — desen ankrajlı olmalı
        "Forecasting with Preprint Data: A Survey",
    ],
)
def test_gercek_basliklar_sablon_sayilmaz(metin: str) -> None:
    assert not is_boilerplate_title(metin)


class _Store:
    """`find_paper_by_title` YALNIZ verilen başlık için eşleşme döner."""

    def __init__(self, mevcut_baslik: str) -> None:
        self._mevcut = mevcut_baslik
        self.eklenen: list[list] = []

    def get_paper_by_hash(self, file_hash: str):
        return None

    def has_embedded_chunks(self, paper_id: str) -> bool:
        return False

    def find_paper_by_title(self, title: str):
        if title == self._mevcut:
            return SimpleNamespace(paper_id="paper_onceki", title=title)
        return None

    def upsert_paper(self, **kwargs: object) -> None:
        pass

    def delete_chunks_for_paper(self, paper_id: str) -> None:
        pass

    def add_chunks(self, rows: list) -> int:
        self.eklenen.append(rows)
        return len(rows)

    def mark_chunks_embedded(self, ids: list) -> None:
        pass


class _Chroma:
    def delete_by_paper(self, paper_id: str) -> None:
        pass

    def add(self, **kwargs: object) -> None:
        pass


class _Embedder:
    mode = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]


def _indexer(monkeypatch, tmp_path, store, cikan_baslik: str):
    from app.memory import bm25_corpus, graph_corpus, paper_indexer

    monkeypatch.setattr(bm25_corpus, "reset_cache", lambda: None)
    monkeypatch.setattr(graph_corpus, "reset_cache", lambda: None)
    monkeypatch.setattr(
        paper_indexer, "parse_pdf", lambda path: ParsedPdf(path=path, pages=["metin"])
    )
    monkeypatch.setattr(
        paper_indexer,
        "extract_metadata",
        lambda text: PaperMetadata(title=cikan_baslik, authors=[], year="2023"),
    )
    monkeypatch.setattr(
        paper_indexer,
        "chunk_parsed_pdf",
        lambda pid, parsed: [
            SimpleNamespace(
                chunk_id=f"{pid}_c0",
                paper_id=pid,
                chunk_index=0,
                section_name=None,
                page_number=1,
                text="metin",
                char_count=5,
                token_estimate=2,
            )
        ],
    )
    monkeypatch.setattr("app.config.settings.PROJECT_ROOT", tmp_path)
    idx = paper_indexer.PaperIndexer(store=store, chroma=_Chroma(), embedder=_Embedder())
    monkeypatch.setattr(idx, "enrich_corpus", lambda: [])
    return idx


def test_ayni_banner_iki_makaleyi_birlestirmez(monkeypatch, tmp_path) -> None:
    # Korpusta zaten banner başlıklı bir makale var; AYNI banner'ı taşıyan FARKLI makale
    # gelince atlanmamalı — asıl bug buydu (ReAct sessizce düştü).
    store = _Store(mevcut_baslik=BANNER)
    idx = _indexer(monkeypatch, tmp_path, store, cikan_baslik=BANNER)
    disc = DiscoveredPaper(path=tmp_path / "arxiv_2210.03629.pdf", file_hash="a" * 20)

    sonuc = idx.ingest_one(disc, enrich=False)

    assert not sonuc.skipped, "banner çakışması makaleyi düşürmemeli"
    assert sonuc.notes and "duplicate_title" not in sonuc.notes
    assert sonuc.title == "arxiv 2210.03629"  # dosya adından türetildi
    assert store.eklenen, "chunk'lar gerçekten yazılmalı"


def test_gercek_baslik_tekrari_hala_atlanir(monkeypatch, tmp_path) -> None:
    # Amaçlanan davranış KORUNMALI: aynı GERÇEK başlık (yeniden indirilmiş / farklı
    # export / farklı arXiv sürümü) hâlâ dedup edilir.
    store = _Store(mevcut_baslik=GERCEK_BASLIK)
    idx = _indexer(monkeypatch, tmp_path, store, cikan_baslik=GERCEK_BASLIK)
    disc = DiscoveredPaper(path=tmp_path / "baska_dosya_adi.pdf", file_hash="b" * 20)

    sonuc = idx.ingest_one(disc, enrich=False)

    assert sonuc.skipped
    assert sonuc.notes == ["duplicate_title"]
    assert sonuc.paper_id == "paper_onceki"
