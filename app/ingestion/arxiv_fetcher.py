"""arXiv makale arama ve otomatik PDF indirme.

Harici bağımlılık: yalnız zaten mevcut `httpx`.
PDF'ler data/papers/raw_pdf/ dizinine kaydedilir; ardından standart
ingestion pipeline ile indekslenir.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.agents.runtime import log_step, tracked
from app.config import get_settings

_SEARCH_URL = "https://export.arxiv.org/api/query"
_PDF_BASE = "https://arxiv.org/pdf/{arxiv_id}.pdf"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_HEADERS = {"User-Agent": "Achilles/1.0 (academic-research; contact: noreply@achilles)"}


@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    pdf_url: str = ""
    published: str = ""


@dataclass
class FetchResult:
    arxiv_id: str
    title: str
    pdf_path: Path
    skipped: bool  # True → dosya zaten vardı veya hata
    error: str | None = None  # hata mesajı (skipped=True ise dolu)


def search_arxiv(query: str, max_results: int = 10) -> list[ArxivEntry]:
    """arXiv'de arama yap; PDF indirmez, yalnız metadata döndürür."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "max_results": min(max_results, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"{_SEARCH_URL}?{params}"
    with httpx.Client(headers=_HEADERS, timeout=30) as client:
        resp = client.get(url)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entries: list[ArxivEntry] = []
    for node in root.findall("atom:entry", _NS):
        raw_id = node.findtext("atom:id", default="", namespaces=_NS) or ""
        # "http://arxiv.org/abs/2301.12345v2" → "2301.12345"
        # Eski stil "http://arxiv.org/abs/hep-th/9901001v2" → "hep-th/9901001" (kategori KORUNUR;
        # eski rsplit('/')+split('v') kategoriyi düşürüp 404 PDF URL + dosya çakışması yaratıyordu).
        stripped = raw_id.split("/abs/")[-1].split("/pdf/")[-1] if raw_id else ""
        arxiv_id = re.sub(r"v\d+$", "", stripped)
        if not arxiv_id:
            continue
        title = (node.findtext("atom:title", default="", namespaces=_NS) or "").strip()
        abstract = (node.findtext("atom:summary", default="", namespaces=_NS) or "").strip()
        published = (node.findtext("atom:published", default="", namespaces=_NS) or "")[:10]
        authors = [
            (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()
            for a in node.findall("atom:author", _NS)
        ]
        entries.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                pdf_url=_PDF_BASE.format(arxiv_id=arxiv_id),
                published=published,
            )
        )
    return entries


@tracked("arxiv-fetcher", trigger_type="manual")
def fetch_arxiv_papers(
    query: str,
    max_results: int = 5,
    dest_dir: Path | None = None,
) -> list[FetchResult]:
    """arXiv'de ara → eşleşen PDF'leri indir → raw_pdf/ dizinine kaydet.

    Aynı ID ile dosya varsa yeniden indirmez (idempotent).
    """
    out_dir = dest_dir or get_settings().raw_pdf_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = search_arxiv(query, max_results=max_results)
    results: list[FetchResult] = []
    log_step(f"{len(entries)} arXiv sonucu indirilecek")

    with httpx.Client(headers=_HEADERS, timeout=90, follow_redirects=True) as client:
        for entry in entries:
            safe_id = entry.arxiv_id.replace("/", "_")
            filename = f"arxiv_{safe_id}.pdf"
            dest = out_dir / filename

            if dest.exists():
                results.append(
                    FetchResult(
                        arxiv_id=entry.arxiv_id,
                        title=entry.title,
                        pdf_path=dest,
                        skipped=True,
                    )
                )
                continue

            try:
                resp = client.get(entry.pdf_url)
                resp.raise_for_status()
                pdf_bytes = resp.content
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "PDF indirilemedi: %s — %s", entry.arxiv_id, exc
                )
                results.append(
                    FetchResult(
                        arxiv_id=entry.arxiv_id,
                        title=entry.title,
                        pdf_path=dest,
                        skipped=True,
                        error=str(exc),
                    )
                )
                continue

            if not pdf_bytes.startswith(b"%PDF"):
                results.append(
                    FetchResult(
                        arxiv_id=entry.arxiv_id,
                        title=entry.title,
                        pdf_path=dest,
                        skipped=True,
                        error="PDF içeriği geçersiz (magic bytes hatalı)",
                    )
                )
                continue

            dest.write_bytes(pdf_bytes)
            results.append(
                FetchResult(
                    arxiv_id=entry.arxiv_id,
                    title=entry.title,
                    pdf_path=dest,
                    skipped=False,
                )
            )

    return results
