"""graph_corpus korpus-graf cache testleri — geçersizleştirme + ingestion wiring.

Çevrimdışı: sahte Chroma + stub store/embedder. Chroma/Ollama gerektirmez.

graph_corpus cache'i YALNIZ chunk-SAYISI anahtarlıdır (bm25_corpus'tan bilinçli fark:
pahalı get_all() yeniden-kurma dalına ertelendiğinden imzaya toplam-karakter eklemek her
çağrıda get_all'ı zorlardı → perf regresyonu). Bu yüzden aynı-sayıda içerik değişimini imza
yakalayamaz; `reset_cache()` OTORİTATİF geçersizleştirmedir ve ingestion mutasyon yolu
(PaperIndexer.ingest_one) onu çağırmalı — yoksa graf sessizce bayatlar (Kural 7).
"""

from __future__ import annotations

from types import SimpleNamespace

from app.ingestion.metadata_extractor import PaperMetadata
from app.ingestion.paper_loader import DiscoveredPaper
from app.ingestion.pdf_parser import ParsedPdf
from app.memory import graph_corpus
from app.memory.graph_corpus import get_corpus_graph, reset_cache


def _row(cid: str, doc: str, pid: str = "p") -> dict:
    """Chroma get_all() satırı taklidi."""
    return {"chunk_id": cid, "document": doc, "metadata": {"paper_id": pid, "title": "T"}}


class _FakeChroma:
    """get_corpus_graph'in Chroma kaynağını taklit eden stub (count + get_all)."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def count(self) -> int:
        return len(self._rows)

    def get_all(self) -> list[dict]:
        return list(self._rows)


# --------------------------------------------------------------- reset_cache
def test_reset_cache_clears() -> None:
    reset_cache()
    chroma = _FakeChroma([_row("p_c0", "rsi momentum strategy"), _row("p_c1", "atr volatility")])
    graph, chunks = get_corpus_graph(chroma=chroma)
    assert graph is not None
    assert len(chunks) == 2
    assert graph_corpus._cache["count"] == 2  # cache dolu

    reset_cache()
    assert graph_corpus._cache["count"] == -1  # sıfırlandı
    assert graph_corpus._cache["graph"] is None
    assert graph_corpus._cache["chunks"] == {}


def test_empty_corpus_returns_none() -> None:
    reset_cache()
    graph, chunks = get_corpus_graph(chroma=_FakeChroma([]))
    assert graph is None
    assert chunks == {}
    reset_cache()


def test_reset_cache_acquires_build_lock() -> None:
    # MEDIUM-1 (Kademe-2 av): reset_cache() _build_lock ALTINDA çalışmalı — yoksa sürmekte olan
    # bir build'in (get_all + build_graph, saniyeler) son _cache.update'i reset'i EZER
    # (lost-update) → ingest sonrası bayat graf sessizce sunulur (Kural 7). Build'i taklit etmek
    # için kilit dışarıda tutulurken reset ilerleyememeli.
    import threading

    reset_cache()
    assert graph_corpus._build_lock.acquire(timeout=2)  # "in-flight build" taklidi
    done = threading.Event()
    t = threading.Thread(target=lambda: (reset_cache(), done.set()))
    t.start()
    try:
        assert not done.wait(0.3)  # kilit tutulurken reset BLOKLANIR (kilidi alıyor)
    finally:
        graph_corpus._build_lock.release()
    assert done.wait(2)  # kilit bırakılınca reset tamamlanır
    t.join(2)
    reset_cache()


def test_same_count_content_change_requires_reset() -> None:
    # graph_corpus imzası YALNIZ chunk SAYISIdır (deferred-get_all tasarımı). Aynı-sayıda
    # içerik değişimini (force re-index / iyileşen parser / delete+add) imza yakalayamaz →
    # reset_cache OLMADAN bayat graf sunulur. reset_cache() otoritatif geçersizleştirmedir.
    reset_cache()
    v1 = _FakeChroma([_row("p_c0", "rsi momentum strategy"), _row("p_c1", "sharpe ratio return")])
    _, chunks_v1 = get_corpus_graph(chroma=v1)
    assert chunks_v1["p_c1"].text == "sharpe ratio return"

    # AYNI sayı (2), FARKLI içerik, reset YOK → count değişmediğinden bayat cache döner.
    v2 = _FakeChroma([_row("p_c0", "kalman filter state"), _row("p_c1", "markov regime switch")])
    _, chunks_stale = get_corpus_graph(chroma=v2)
    assert chunks_stale["p_c1"].text == "sharpe ratio return"  # BAYAT (tasarım gereği)

    # Otoritatif geçersizleştirme → taze içerik kurulur.
    reset_cache()
    _, chunks_fresh = get_corpus_graph(chroma=v2)
    assert chunks_fresh["p_c1"].text == "markov regime switch"  # TAZE
    reset_cache()


def test_rebuilds_on_count_change() -> None:
    reset_cache()
    v1 = _FakeChroma([_row("p_c0", "rsi momentum")])
    _, chunks_v1 = get_corpus_graph(chroma=v1)
    assert set(chunks_v1) == {"p_c0"}

    v2 = _FakeChroma([_row("p_c0", "rsi momentum"), _row("p_c1", "atr volatility filter")])
    _, chunks_v2 = get_corpus_graph(chroma=v2)
    assert set(chunks_v2) == {"p_c0", "p_c1"}  # sayı değişti → yeniden kuruldu
    reset_cache()


# ------------------------------------------------- PaperIndexer ingestion wiring
class _FakeStore:
    """ingest_one'ın dokunduğu SqliteStore yüzeyi (hepsi no-op / yeni-makale yolu)."""

    def get_paper_by_hash(self, file_hash: str) -> None:
        return None

    def find_paper_by_title(self, title: str) -> None:
        return None

    def upsert_paper(self, **kwargs: object) -> None:
        pass

    def delete_chunks_for_paper(self, paper_id: str) -> None:
        pass

    def add_chunks(self, rows: list) -> None:
        pass

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


def test_ingestion_resets_graph_cache(monkeypatch, tmp_path) -> None:
    # PaperIndexer.ingest_one HEM bm25_corpus HEM graph_corpus cache'ini sıfırlamalı —
    # yoksa graf sessizce bayatlar (Kural 7). Tüm I/O stub'lı; ağır alt-sistemler nötr.
    from app.memory import bm25_corpus, paper_indexer

    calls = {"bm25": 0, "graph": 0}
    monkeypatch.setattr(
        bm25_corpus, "reset_cache", lambda: calls.__setitem__("bm25", calls["bm25"] + 1)
    )
    monkeypatch.setattr(
        graph_corpus, "reset_cache", lambda: calls.__setitem__("graph", calls["graph"] + 1)
    )

    # Ağır boru-hattı adımlarını stub'la (parse/metadata/chunk).
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

    # reset SONRASI çalışan alt-sistemleri nötrle (gerçek DB/LLM'e gitmesin).
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

    # Disk yazımlarını tmp_path'e yönlendir (gerçek proje data dizinlerini kirletme):
    # extracted_text_dir / metadata_dir modül-düzeyi PROJECT_ROOT'tan türer (setter yok).
    # İndexer'ı patch SONRASI kur → ensure_dirs() tmp altına oluştursun.
    monkeypatch.setattr("app.config.settings.PROJECT_ROOT", tmp_path)
    idx = paper_indexer.PaperIndexer(
        store=_FakeStore(), chroma=_FakeChromaStore(), embedder=_FakeEmbedder()
    )

    disc = DiscoveredPaper(path=tmp_path / "x.pdf", file_hash="abc123def4567890abcd")
    result = idx.ingest_one(disc)

    assert not result.skipped
    assert calls["graph"] == 1  # graf cache sıfırlandı (asıl iddia)
    assert calls["bm25"] == 1  # bm25 cache de sıfırlandı (regresyon koruması)
