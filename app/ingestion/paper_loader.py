"""Discover PDFs and assign stable identifiers.

``paper_id`` is derived from the file content hash so re-ingesting the same
file yields the same id (idempotent ingestion).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings


@dataclass
class DiscoveredPaper:
    path: Path
    file_hash: str

    @property
    def paper_id(self) -> str:
        # short, stable, human-pasteable
        return f"paper_{self.file_hash[:12]}"


def compute_file_hash(path: str | Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def discover_pdfs(directory: str | Path | None = None) -> list[DiscoveredPaper]:
    """Find all PDFs in the raw_pdf directory (recursively)."""
    settings = get_settings()
    base = Path(directory) if directory else settings.raw_pdf_dir
    base.mkdir(parents=True, exist_ok=True)
    found: list[DiscoveredPaper] = []
    for p in sorted(base.rglob("*.pdf")):
        found.append(DiscoveredPaper(path=p, file_hash=compute_file_hash(p)))
    return found


def iter_pdfs(paths: Iterable[str | Path]) -> list[DiscoveredPaper]:
    out: list[DiscoveredPaper] = []
    for p in paths:
        p = Path(p)
        out.append(DiscoveredPaper(path=p, file_hash=compute_file_hash(p)))
    return out
