"""PDF -> text extraction.

Primary backend is PyMuPDF (fitz); falls back to pypdf if unavailable.
Returns both the full concatenated text and per-page text so the chunker
can attach page numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedPdf:
    path: Path
    pages: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def n_chars(self) -> int:
        return len(self.text)


def _parse_with_pymupdf(path: Path) -> list[str] | None:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages


def _parse_with_pypdf(path: Path) -> list[str] | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def parse_pdf(path: str | Path) -> ParsedPdf:
    """Extract text from a PDF file, page by page."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    pages = _parse_with_pymupdf(path)
    if pages is None:
        logger.warning("PyMuPDF yok, pypdf'e geçiliyor: %s", path.name)
        pages = _parse_with_pypdf(path)
    if pages is None:
        raise RuntimeError(
            "PDF okuyabilmek için 'pymupdf' veya 'pypdf' kurulu olmalı. Çözüm: uv sync"
        )

    pages = [p.strip() for p in pages]
    logger.info("PDF okundu: %s (%d sayfa)", path.name, len(pages))
    return ParsedPdf(path=path, pages=pages)
