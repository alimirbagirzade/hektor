"""İçe-alım kalite skoru (100 puan) — compute-on-demand, çevrimdışı, deterministik.

Mevcut SQLite verisinden (Paper/Chunk/Formula/ChunkQualityFlag) hesaplar; PaperIndexer'ın
SICAK yolunu DEĞİŞTİRMEZ (eş zamanlı içe-alım çalışmasıyla çakışmamak için). Skor NULL
kalan eski makaleler retrieval'ı ETKİLEMEZ.

Rubrik (100): parse 15 · metadata 10 · section 15 · formula 15 · table 15 · figure 10 ·
ocr 10 · cleantext 10. Eşik: ≥90 ready_for_rag · 70-89 usable · 50-69 slow_but_usable ·
40-49 unstable · <40 failed.

NOT: Bu bir SEZGİSEL kalite tahminidir (yer-doğrusu yok). formula/table/figure YOKLUĞU,
parse başarılıysa "nötr" puanlanır (eksik çıkarımı haksızca cezalandırmamak için); parse
başarısızsa (chunk yok) çıkarım bileşenleri 0'lanır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.ingestion.clean_text_scorer import score_clean_text

if TYPE_CHECKING:
    from app.memory.sqlite_store import SqliteStore

_FIGURE_RE = re.compile(r"(?i)\b(?:figure|şekil|fig\.)\s*\d+")
_MAX_CLEAN_SAMPLE = 40_000  # temizlik örneklemi (büyük korpusta hız)


@dataclass
class IngestionInputs:
    """Skorlama için ham girdiler (DB'den toplanır ya da test için elle verilir)."""

    n_pages: int
    n_chars: int
    has_title: bool
    has_authors: bool
    has_year: bool
    n_chunks: int
    n_sections: int
    n_formulas: int
    n_tables: int
    n_figure_captions: int
    clean_text_score: float  # 0-10


@dataclass
class IngestionQualityResult:
    """100-puanlık içe-alım kalite sonucu + durum + bileşen kırılımı."""

    total: float
    status: str
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "status": self.status,
            "components": self.components,
            "notes": self.notes,
        }


def _status_for(total: float) -> str:
    if total >= 90:
        return "ready_for_rag"
    if total >= 70:
        return "usable"
    if total >= 50:
        return "slow_but_usable"
    if total >= 40:
        return "unstable"
    return "failed"


def score_ingestion(inp: IngestionInputs) -> IngestionQualityResult:
    """Saf rubrik skorlayıcı (DB'siz, deterministik) — IngestionInputs → sonuç."""
    notes: list[str] = []
    chars_per_page = inp.n_chars / max(1, inp.n_pages)

    # metadata her zaman sayılır (parse başarısız olsa da bilinir)
    metadata = (
        (4.0 if inp.has_title else 0.0)
        + (3.0 if inp.has_authors else 0.0)
        + (3.0 if inp.has_year else 0.0)
    )

    if inp.n_chunks == 0:
        # parse başarısız → çıkarım bileşenleri 0; yalnız metadata kalır
        notes.append("Chunk yok — parse başarısız sayıldı (çıkarım bileşenleri 0).")
        comp = {
            "parse": 0.0,
            "metadata": round(metadata, 2),
            "section": 0.0,
            "formula": 0.0,
            "table": 0.0,
            "figure": 0.0,
            "ocr": 0.0,
            "cleantext": 0.0,
        }
        total = round(sum(comp.values()), 2)
        return IngestionQualityResult(
            total=total, status=_status_for(total), components=comp, notes=notes
        )

    parse = 15.0 * min(1.0, chars_per_page / 1200.0)
    section = 15.0 * min(1.0, inp.n_sections / 4.0)
    formula = 15.0 if inp.n_formulas > 0 else 7.5
    table = 15.0 if inp.n_tables > 0 else 7.5
    figure = 10.0 if inp.n_figure_captions > 0 else 5.0
    ocr = 10.0 * min(1.0, chars_per_page / 800.0)
    cleantext = max(0.0, min(10.0, inp.clean_text_score))

    if inp.n_formulas == 0:
        notes.append("Formül bulunamadı — nötr puanlandı (eksik olabilir).")
    if inp.n_tables == 0:
        notes.append("Tablo bulunamadı — nötr puanlandı.")
    if chars_per_page < 400:
        notes.append("Düşük karakter yoğunluğu — taranmış/OCR zayıf olabilir.")

    comp = {
        "parse": round(parse, 2),
        "metadata": round(metadata, 2),
        "section": round(section, 2),
        "formula": round(formula, 2),
        "table": round(table, 2),
        "figure": round(figure, 2),
        "ocr": round(ocr, 2),
        "cleantext": round(cleantext, 2),
    }
    total = round(sum(comp.values()), 2)
    return IngestionQualityResult(
        total=total, status=_status_for(total), components=comp, notes=notes
    )


def gather_inputs(store: SqliteStore, paper_id: str) -> IngestionInputs:
    """SQLite'tan bir makalenin skorlama girdilerini topla (ChromaDB/ağ gerekmez)."""
    from app.memory.sqlite_store import ChunkQualityFlag, Paper

    with store.session() as s:
        paper = s.get(Paper, paper_id)
        if paper is None:
            raise ValueError(f"Bilinmeyen makale: {paper_id}")
        has_title = bool((paper.title or "").strip())
        has_authors = bool((paper.authors or "").strip() and paper.authors not in ("[]", "null"))
        has_year = bool((paper.year or "").strip())
        n_pages = int(paper.n_pages or 0)
        n_chars = int(paper.n_chars or 0)

    chunks = store.list_chunks(paper_id)
    n_chunks = len(chunks)
    sections = {
        (c.section_name or "").strip().lower() for c in chunks if (c.section_name or "").strip()
    }
    n_sections = len(sections)
    text_sample = "  ".join(c.text or "" for c in chunks)[:_MAX_CLEAN_SAMPLE]
    n_figure_captions = len(_FIGURE_RE.findall(text_sample))
    clean = score_clean_text(text_sample) if text_sample else 0.0

    n_formulas = len(store.list_formulas(paper_id))

    chunk_ids = [c.chunk_id for c in chunks]
    n_tables = 0
    if chunk_ids:
        from sqlalchemy import select

        with store.session() as s:
            rows = s.scalars(
                select(ChunkQualityFlag).where(ChunkQualityFlag.chunk_id.in_(chunk_ids))
            )
            n_tables = sum(1 for f in rows if f.has_table)

    # n_chars boşsa chunk'lardan tahmin et (eski kayıtlarda n_chars NULL olabilir)
    if n_chars == 0 and chunks:
        n_chars = sum(len(c.text or "") for c in chunks)
    if n_pages == 0:
        n_pages = max(1, n_chunks // 2)  # kaba sayfa tahmini (yoğunluk için)

    return IngestionInputs(
        n_pages=n_pages,
        n_chars=n_chars,
        has_title=has_title,
        has_authors=has_authors,
        has_year=has_year,
        n_chunks=n_chunks,
        n_sections=n_sections,
        n_formulas=n_formulas,
        n_tables=n_tables,
        n_figure_captions=n_figure_captions,
        clean_text_score=clean,
    )


def score_paper(store: SqliteStore, paper_id: str) -> IngestionQualityResult:
    """Bir makalenin içe-alım kalitesini DB'den hesapla (compute-on-demand)."""
    return score_ingestion(gather_inputs(store, paper_id))


@dataclass
class CorpusQualitySummary:
    """Tüm korpusun içe-alım kalite taraması özeti (dağılım + en zayıflar)."""

    total: int  # toplam makale
    scored: int  # skorlanabilen
    by_status: dict[str, int]  # durum → sayı (ready_for_rag/usable/.../failed)
    avg_score: float
    worst: list[dict[str, Any]] = field(default_factory=list)  # en düşük skorlular

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "scored": self.scored,
            "by_status": self.by_status,
            "avg_score": self.avg_score,
            "worst": self.worst,
        }


def score_all_papers(
    store: SqliteStore, *, record: bool = False, worst_n: int = 10
) -> CorpusQualitySummary:
    """Tüm makaleleri içe-alım kalitesi için tara (deterministik; salt-skor / opt-in record).

    Operasyonel kullanım: "korpusumda hangi makaleler RAG'a hazır değil?" Her makale tek tek
    skorlanır; ``record=True`` ise ``paper_ingestion_runs`` + ``papers.quality_score`` yazılır.
    """
    papers = store.list_papers()
    by_status: dict[str, int] = {}
    scored: list[tuple[str, str, IngestionQualityResult]] = []
    for p in papers:
        try:
            inp = gather_inputs(store, p.paper_id)
        except ValueError:
            continue
        res = score_ingestion(inp)
        by_status[res.status] = by_status.get(res.status, 0) + 1
        scored.append((p.paper_id, p.title or "", res))
        if record:
            store.add_ingestion_run(
                paper_id=p.paper_id,
                status=res.status,
                quality_score=res.total,
                component_scores=res.components,
                n_chunks=inp.n_chunks,
                n_formulas=inp.n_formulas,
                notes=res.notes,
            )
    avg = round(sum(r.total for _, _, r in scored) / len(scored), 2) if scored else 0.0
    worst = sorted(scored, key=lambda t: t[2].total)[: max(0, worst_n)]
    return CorpusQualitySummary(
        total=len(papers),
        scored=len(scored),
        by_status=by_status,
        avg_score=avg,
        worst=[
            {"paper_id": pid, "title": title[:48], "total": r.total, "status": r.status}
            for pid, title, r in worst
        ],
    )
