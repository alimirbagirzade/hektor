"""Ingest idempotency / yarım-ingest onarımı testleri (Kademe-2 av MEDIUM-2).

Çevrimdışı: sahte store/chroma/embedder + parse/chunk monkeypatch. Chroma/Ollama gerektirmez.

Bulgu: upsert_paper kendi transaction'ında önce commit edilir; embed/chroma.add adımı çökerse
paper satırı kalır ama chunk'lar embedded=0'da takılır. Eski kapı yalnız "paper satırı var mı"
baktığından force'suz re-run makaleyi KALICI atlardı (file_hash sabit) → RAG'a hiç girmez (Kural 7).
Fix: kapı "gömülü chunk var mı" kontrol eder; yoksa force gibi yeniden işler.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.ingestion.metadata_extractor import PaperMetadata
from app.ingestion.paper_loader import DiscoveredPaper
from app.ingestion.pdf_parser import ParsedPdf


class _ExistingStore:
    """Paper satırı VAR; has_embedded_chunks dönüşü parametrik (tam vs yarım ingest)."""

    def __init__(self, *, embedded: bool) -> None:
        self._embedded = embedded
        self.deleted: list[str] = []
        self.added: list[list] = []

    def get_paper_by_hash(self, file_hash: str):
        return SimpleNamespace(paper_id=f"paper_{file_hash[:12]}", title="Eski Başlık")

    def has_embedded_chunks(self, paper_id: str) -> bool:
        return self._embedded

    def find_paper_by_title(self, title: str):
        return None

    def upsert_paper(self, **kwargs: object) -> None:
        pass

    def delete_chunks_for_paper(self, paper_id: str) -> None:
        self.deleted.append(paper_id)

    def add_chunks(self, rows: list) -> int:
        self.added.append(rows)
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


def _build_indexer(monkeypatch, tmp_path, store):
    """ingest_one'ın ağır boru-hattını stub'layıp izole bir PaperIndexer kur."""
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
    fake_chunk = SimpleNamespace(
        chunk_id="paper_abc123def456_c0",
        paper_id="paper_abc123def456",
        chunk_index=0,
        section_name="results",
        page_number=1,
        text="rsi momentum",
        char_count=12,
        token_estimate=3,
    )
    monkeypatch.setattr(paper_indexer, "chunk_parsed_pdf", lambda pid, parsed: [fake_chunk])
    monkeypatch.setattr(
        "app.research.formula_extractor.FormulaExtractor",
        type("_NoFormula", (), {"extract_from_paper": lambda self, pid: []}),
    )
    monkeypatch.setattr(
        "app.research.concept_graph.ConceptGraph",
        type("_NoConcept", (), {"build_from_papers": lambda self: 0}),
    )
    monkeypatch.setattr(
        "app.research.cross_paper_synthesizer.CrossPaperSynthesizer",
        type("_NoSynth", (), {"synthesize_all": lambda self: 0}),
    )
    monkeypatch.setattr("app.config.settings.PROJECT_ROOT", tmp_path)
    return paper_indexer.PaperIndexer(
        store=store, chroma=_FakeChromaStore(), embedder=_FakeEmbedder()
    )


def test_skip_when_fully_embedded(monkeypatch, tmp_path) -> None:
    # Tam ingest (gömülü chunk var) → force'suz çağrı ATLAR (mevcut, doğru davranış korunur).
    store = _ExistingStore(embedded=True)
    idx = _build_indexer(monkeypatch, tmp_path, store)
    disc = DiscoveredPaper(path=tmp_path / "x.pdf", file_hash="abc123def4567890abcd")
    result = idx.ingest_one(disc)
    assert result.skipped
    assert result.notes == ["already ingested"]
    assert store.added == []  # yeniden işlenmedi
    assert store.deleted == []


def test_repair_when_no_embedded_chunks(monkeypatch, tmp_path) -> None:
    # Yarım ingest (paper satırı var ama gömülü chunk YOK) → force'suz çağrı ONARMALI:
    # atlamamalı, chunk'ları gerçekten yeniden yazmalı (Kural 7 + idempotency).
    store = _ExistingStore(embedded=False)
    idx = _build_indexer(monkeypatch, tmp_path, store)
    disc = DiscoveredPaper(path=tmp_path / "x.pdf", file_hash="abc123def4567890abcd")
    result = idx.ingest_one(disc)
    assert not result.skipped  # ATLAMADI — onardı
    assert result.n_chunks == 1  # chunk'lar gerçekten yazıldı
    assert store.deleted == ["paper_abc123def456"]  # eski (yarım) chunk'lar temizlendi
    assert len(store.added) == 1 and len(store.added[0]) == 1  # taze chunk eklendi
