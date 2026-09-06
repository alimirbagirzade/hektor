"""Toplu ingest'te korpus-geneli zenginleştirme PARTİ BAŞINA bir kez koşmalı.

Çevrimdışı: sahte store/chroma/embedder + parse/chunk/discover monkeypatch.

Bulgu (155 makalelik korpusta ölçüldü): `ingest_one` her makaleden sonra kavram grafiğini
(`ConceptGraph.build_from_papers`) ve çapraz sentezi (`CrossPaperSynthesizer.synthesize_all`)
çağırıyordu. İkisi de TÜM korpusu tarar → maliyet makale sayısıyla kareye yakın büyür; GPU'suz
makinede ingest saatlerce tek makalede kilitlendi.

Fix: `ingest_directory` makaleleri `enrich=False` ile işler, zenginleştirmeyi dizin SONUNDA
bir kez koşar. Tek-makale yolu (web yüklemesi) varsayılan `enrich=True` ile korunur.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.ingestion.metadata_extractor import PaperMetadata
from app.ingestion.paper_loader import DiscoveredPaper
from app.ingestion.pdf_parser import ParsedPdf


class _NewStore:
    """Hiçbir makale kayıtlı değil → her PDF tam ingest yolundan geçer."""

    def get_paper_by_hash(self, file_hash: str):
        return None

    def has_embedded_chunks(self, paper_id: str) -> bool:
        return False

    def find_paper_by_title(self, title: str):
        return None

    def upsert_paper(self, **kwargs: object) -> None:
        pass

    def delete_chunks_for_paper(self, paper_id: str) -> None:
        pass

    def add_chunks(self, rows: list) -> int:
        return len(rows)

    def mark_chunks_embedded(self, ids: list) -> None:
        pass


class _FakeChromaStore:
    def delete_by_paper(self, paper_id: str) -> None:
        pass

    def add(self, **kwargs: object) -> None:
        pass


class _FakeEmbedder:
    mode = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]


def _build_indexer(monkeypatch, tmp_path, sayac: list[str], formul_sayac: list[str] | None = None):
    """Ağır boru hattı stub'lanmış PaperIndexer; enrich_corpus + formül çağrılarını sayar."""
    formul_sayac = [] if formul_sayac is None else formul_sayac
    from app.memory import bm25_corpus, graph_corpus, paper_indexer

    monkeypatch.setattr(bm25_corpus, "reset_cache", lambda: None)
    monkeypatch.setattr(graph_corpus, "reset_cache", lambda: None)

    parsed = ParsedPdf(path=tmp_path / "x.pdf", pages=["rsi momentum strategy text"])
    monkeypatch.setattr(paper_indexer, "parse_pdf", lambda path: parsed)
    monkeypatch.setattr(
        paper_indexer,
        "extract_metadata",
        lambda text: PaperMetadata(title="Sahte Makale", authors=[], year="2024"),
    )

    def _chunks(pid: str, _parsed: object) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                chunk_id=f"{pid}_c0",
                paper_id=pid,
                chunk_index=0,
                section_name="results",
                page_number=1,
                text="rsi momentum",
                char_count=12,
                token_estimate=3,
            )
        ]

    monkeypatch.setattr(paper_indexer, "chunk_parsed_pdf", _chunks)

    def _formul(self, pid: str) -> list:
        formul_sayac.append(pid)
        return []

    monkeypatch.setattr(
        "app.research.formula_extractor.FormulaExtractor",
        type("_SayanFormula", (), {"extract_from_paper": _formul}),
    )
    monkeypatch.setattr("app.config.settings.PROJECT_ROOT", tmp_path)

    idx = paper_indexer.PaperIndexer(
        store=_NewStore(), chroma=_FakeChromaStore(), embedder=_FakeEmbedder()
    )
    monkeypatch.setattr(idx, "enrich_corpus", lambda: (sayac.append("x"), [])[1], raising=True)
    return idx, paper_indexer


def test_toplu_ingest_zenginlestirmeyi_bir_kez_kosar(monkeypatch, tmp_path) -> None:
    # 3 makale → zenginleştirme 3 kez DEĞİL, TEK kez koşmalı (kareye yakın maliyet fix'i).
    sayac: list[str] = []
    idx, paper_indexer = _build_indexer(monkeypatch, tmp_path, sayac)
    bulunanlar = [
        DiscoveredPaper(path=tmp_path / f"{i}.pdf", file_hash=f"{i}23456789abcdef01234")
        for i in range(3)
    ]
    monkeypatch.setattr(paper_indexer, "discover_pdfs", lambda directory: bulunanlar)

    sonuclar = idx.ingest_directory(tmp_path)

    assert len(sonuclar) == 3
    assert all(not r.skipped for r in sonuclar)
    assert len(sayac) == 1  # ← parti başına bir kez


def test_toplu_ingest_formul_cikarmaz(monkeypatch, tmp_path) -> None:
    # Formül çıkarma chunk BAŞINA LLM çağrısıdır → toplu ingest'te KOŞMAMALI.
    # Formüller ayrı adımdan gelir: `hektor extract-formulas`.
    sayac: list[str] = []
    formul: list[str] = []
    idx, paper_indexer = _build_indexer(monkeypatch, tmp_path, sayac, formul)
    bulunanlar = [
        DiscoveredPaper(path=tmp_path / f"{i}.pdf", file_hash=f"{i}23456789abcdef01234")
        for i in range(3)
    ]
    monkeypatch.setattr(paper_indexer, "discover_pdfs", lambda directory: bulunanlar)

    idx.ingest_directory(tmp_path)

    assert formul == []  # ← tek bir LLM formül çağrısı bile yapılmadı


def test_tek_makale_yolu_hala_zenginlestirir(monkeypatch, tmp_path) -> None:
    # Web'in tek-PDF yükleme yolu (`ingest_one`) varsayılan davranışı KORUMALI:
    # hem formül çıkarır hem korpus zenginleştirmesi yapar.
    sayac: list[str] = []
    formul: list[str] = []
    idx, _ = _build_indexer(monkeypatch, tmp_path, sayac, formul)
    disc = DiscoveredPaper(path=tmp_path / "x.pdf", file_hash="abc123def4567890abcd")

    sonuc = idx.ingest_one(disc)

    assert not sonuc.skipped
    assert len(sayac) == 1
    assert len(formul) == 1


def test_hepsi_atlanirsa_zenginlestirme_kosmaz(monkeypatch, tmp_path) -> None:
    # Değişiklik yoksa (hepsi zaten indeksli) boşuna LLM yakma.
    sayac: list[str] = []
    idx, paper_indexer = _build_indexer(monkeypatch, tmp_path, sayac)
    monkeypatch.setattr(idx, "ingest_one", lambda disc, **kw: _atlandi(disc))
    monkeypatch.setattr(
        paper_indexer,
        "discover_pdfs",
        lambda directory: [DiscoveredPaper(path=tmp_path / "x.pdf", file_hash="a" * 20)],
    )

    idx.ingest_directory(tmp_path)

    assert sayac == []


def _atlandi(disc: DiscoveredPaper):
    from app.memory.paper_indexer import IngestResult

    return IngestResult(
        paper_id="paper_x", title=None, n_chunks=0, skipped=True, notes=["already ingested"]
    )
