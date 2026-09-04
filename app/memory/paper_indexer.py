"""End-to-end ingestion orchestration.

Pipeline:  discover PDF -> parse -> metadata -> chunk -> SQLite -> embed -> Chroma

Idempotent: a PDF already present (matched by file_hash) is skipped unless
``force=True``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunker import chunk_parsed_pdf
from app.ingestion.metadata_extractor import extract_metadata
from app.ingestion.paper_loader import DiscoveredPaper, discover_pdfs
from app.ingestion.pdf_parser import parse_pdf
from app.memory.chroma_store import ChromaStore
from app.memory.embedding_service import EmbeddingService
from app.memory.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    paper_id: str
    title: str | None
    n_chunks: int
    skipped: bool = False
    notes: list[str] = field(default_factory=list)


def build_embed_text(text: str, title: str | None, section: str | None, contextual: bool) -> str:
    """Embed edilecek metni kur (Contextual Retrieval, Faz P2).

    `contextual=True` ise chunk'a "başlık / bölüm:" ön-eki eklenir (retrieval
    doğruluğunu artırır). Ön-ek YALNIZ embedding içindir; Chroma `document` olarak
    orijinal `text` saklanır (Anthropic yaklaşımı; sorguya ön-ek EKLENMEZ).
    """
    if not contextual:
        return text
    prefix = " / ".join(p.strip() for p in (title or "", section or "") if p and p.strip())
    return f"{prefix}: {text}" if prefix else text


class PaperIndexer:
    def __init__(
        self,
        store: SqliteStore | None = None,
        chroma: ChromaStore | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.settings.ensure_dirs()
        self.store = store or SqliteStore()
        self.chroma = chroma or ChromaStore()
        self.embedder = embedder or EmbeddingService()

    def ingest_one(self, disc: DiscoveredPaper, *, force: bool = False) -> IngestResult:
        existing = self.store.get_paper_by_hash(disc.file_hash)
        if existing and not force:
            # Yarım kalmış ingest onarımı (Kural 7 + ingestion idempotent sözleşmesi):
            # upsert_paper kendi transaction'ında önce commit edilir; embed/chroma.add adımı
            # çökerse (Ollama timeout, chromadb hatası, process/oturum kill) paper satırı kalır
            # ama chunk'lar embedded=0'da takılır (Chroma'da YOK). Yalnız "paper satırı var" diye
            # atlarsak makale RAG'a KALICI olarak hiç girmez: file_hash sabit olduğundan her
            # force'suz re-run aynı noktada atlar, retrieval'da sessizce eksik kalır. Bu yüzden
            # "gömülü chunk var mı" kontrol et; yoksa force gibi yeniden işle (aşağıdaki
            # delete_chunks_for_paper + chroma.delete_by_paper idempotent temizler → çift ingest
            # üretmez; aynı paper_id=hash → başlık-dedup da bloklamaz).
            if self.store.has_embedded_chunks(existing.paper_id):
                logger.info("Zaten var, atlanıyor: %s", disc.path.name)
                return IngestResult(
                    paper_id=existing.paper_id,
                    title=existing.title,
                    n_chunks=0,
                    skipped=True,
                    notes=["already ingested"],
                )
            logger.warning(
                "Yarım kalmış ingest (gömülü chunk yok) — yeniden işleniyor: %s", disc.path.name
            )

        parsed = parse_pdf(disc.path)
        meta = extract_metadata(parsed.text)

        # Baslik bulunamazsa dosya adindan uret (alt cizgi/tire -> bosluk)
        if not meta.title:
            meta.title = disc.path.stem.replace("_", " ").replace("-", " ").strip()

        # Paper-düzeyi dedup: aynı başlıklı makale zaten varsa (farklı bytes/hash olsa
        # bile — yeniden indirilmiş, farklı PDF export, farklı arxiv sürümü) RAG'a 2.
        # kez girmesin. file_hash kontrolü yalnız birebir aynı dosyayı yakalar.
        if not force:
            dup = self.store.find_paper_by_title(meta.title)
            if dup is not None and dup.paper_id != disc.paper_id:
                logger.info(
                    "Ayni baslikli makale zaten var (%s) — atlaniyor: %s",
                    dup.paper_id,
                    meta.title,
                )
                return IngestResult(
                    paper_id=dup.paper_id,
                    title=dup.title,
                    n_chunks=0,
                    skipped=True,
                    notes=["duplicate_title"],
                )

        # save extracted text + metadata to disk
        (self.settings.extracted_text_dir / f"{disc.paper_id}.txt").write_text(
            parsed.text, encoding="utf-8"
        )
        (self.settings.metadata_dir / f"{disc.paper_id}.json").write_text(
            json.dumps(
                {
                    "paper_id": disc.paper_id,
                    "file_hash": disc.file_hash,
                    "title": meta.title,
                    "authors": meta.authors,
                    "year": meta.year,
                    "source": meta.source,
                    "n_pages": parsed.n_pages,
                    "n_chars": parsed.n_chars,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.store.upsert_paper(
            paper_id=disc.paper_id,
            file_hash=disc.file_hash,
            source_path=str(disc.path),
            title=meta.title,
            authors=json.dumps(meta.authors, ensure_ascii=False),
            year=meta.year,
            source=meta.source,
            n_pages=parsed.n_pages,
            n_chars=parsed.n_chars,
        )

        # Yeniden indeksleme (force) öncesi eski chunk'ları TEMİZLE: yeni chunk'lama
        # daha az parça üretirse öksüz/bayat chunk kalmasın (idempotent re-index).
        # İlk ingest'te no-op. SQLite + Chroma birlikte temizlenir.
        self.store.delete_chunks_for_paper(disc.paper_id)
        self.chroma.delete_by_paper(disc.paper_id)

        chunks = chunk_parsed_pdf(disc.paper_id, parsed)
        # BUG-M6 fix: embedded=0 olarak yaz; Chroma başarılı olunca embedded=1'e güncelle.
        # Eski: embedded=1 önceden yazılıyordu → Chroma hatasında SQLite'ta "gömülü"
        # görünür ama retrieval'da kayıp olurdu (sessiz bozulma).
        self.store.add_chunks(
            [
                {
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "chunk_index": c.chunk_index,
                    "section_name": c.section_name,
                    "page_number": c.page_number,
                    "text": c.text,
                    "char_count": c.char_count,
                    "token_estimate": c.token_estimate,
                    "embedded": 0,
                }
                for c in chunks
            ]
        )

        # embed + write to chroma. Contextual (P2): embed metni ön-ekli, document orijinal.
        contextual = self.settings.rag_contextual_embed
        embeddings = self.embedder.embed(
            [build_embed_text(c.text, meta.title, c.section_name, contextual) for c in chunks]
        )
        self.chroma.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "paper_id": c.paper_id,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number if c.page_number is not None else -1,
                    "section_name": c.section_name or "",
                    "title": meta.title or "",
                }
                for c in chunks
            ],
        )
        # Chroma başarılıysa embedded=1 olarak işaretle
        self.store.mark_chunks_embedded([c.chunk_id for c in chunks])

        # BM25 korpus cache'i (içerik-imzası anahtarlı) içerik değişince bayatlar —
        # ingestion sonrası sıfırla ki hibrit retrieval taze metni görsün.
        try:
            from app.memory import bm25_corpus

            bm25_corpus.reset_cache()
        except Exception:  # pragma: no cover — cache reset hatası ingestion'ı bloklamamalı
            logger.debug("BM25 cache sıfırlanamadı")

        # Graf korpus cache'i yalnız chunk-SAYISI anahtarlı; aynı-sayıda içerik değişimini
        # (force re-index / iyileşen parser / delete+add) imza yakalayamaz → otoritatif reset
        # budur. Ayrı try/except ki graf reset hatası ingestion'ı bloklamasın (Kural 7).
        try:
            from app.memory import graph_corpus

            graph_corpus.reset_cache()
        except Exception:  # pragma: no cover — cache reset hatası ingestion'ı bloklamamalı
            logger.debug("Graf korpus cache sıfırlanamadı")

        notes: list[str] = [f"embedding_mode={self.embedder.mode}"]

        # 1. Formül çıkarma — LLM yoksa kural tabanlı yedek, hata ingestion'ı bloklamaz
        try:
            from app.research.formula_extractor import FormulaExtractor

            n_formulas = len(FormulaExtractor().extract_from_paper(disc.paper_id))
            if n_formulas:
                notes.append(f"formulas={n_formulas}")
                logger.info("Formül çıkarıldı: %s → %d formül", disc.paper_id, n_formulas)
        except Exception as exc:
            logger.debug("Formül çıkarma atlandı (%s): %s", disc.paper_id, exc)

        # 2. Kavram grafiği yeniden oluştur — tüm makaleler arası ilişkileri güncelle
        try:
            from app.research.concept_graph import ConceptGraph

            n_links = ConceptGraph().build_from_papers()
            if n_links:
                notes.append(f"concept_links={n_links}")
                logger.info("Kavram grafiği güncellendi: %d bağlantı", n_links)
        except Exception as exc:
            logger.debug("Kavram grafiği güncellenemedi: %s", exc)

        # 3. Çapraz makale sentezi — yeni kategori çiftleri için eğitim verisi üret
        try:
            from app.research.cross_paper_synthesizer import CrossPaperSynthesizer

            n_synth = CrossPaperSynthesizer().synthesize_all()
            if n_synth:
                notes.append(f"synthesis_examples={n_synth}")
                logger.info("Çapraz sentez: %d yeni eğitim örneği", n_synth)
        except Exception as exc:
            logger.debug("Çapraz sentez atlandı: %s", exc)

        logger.info("Indexlendi: %s (%d chunk)", disc.paper_id, len(chunks))
        return IngestResult(
            paper_id=disc.paper_id,
            title=meta.title,
            n_chunks=len(chunks),
            notes=notes,
        )

    def ingest_directory(self, directory: str | Path | None = None, *, force: bool = False):
        results = []
        for disc in discover_pdfs(directory):
            results.append(self.ingest_one(disc, force=force))
        return results
