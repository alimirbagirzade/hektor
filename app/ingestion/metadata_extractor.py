"""Lightweight, dependency-free metadata heuristics.

These are intentionally simple. A later stage can replace/augment them with
GROBID, an LLM extraction pass, or arXiv/SSRN API lookups. The goal here is a
"good enough" title/year/authors guess from the first page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

YEAR_RE = re.compile(r"\b(19[7-9]\d|20[0-4]\d)\b")
# crude author line: "First Last, First Last and First Last"
AUTHOR_HINT_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+")


@dataclass
class PaperMetadata:
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: str | None = None
    source: str = "manual"


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def guess_title(first_page: str) -> str | None:
    lines = [_clean(line) for line in first_page.splitlines() if _clean(line)]
    # Title is usually one of the first non-trivial lines, reasonably long,
    # not all-caps boilerplate, not an email/url.
    for line in lines[:8]:
        low = line.lower()
        if any(tok in low for tok in ("http", "@", "arxiv", "doi:")):
            continue
        if 15 <= len(line) <= 200 and not line.isupper():
            return line
    return lines[0] if lines else None


def guess_year(first_page: str) -> str | None:
    m = YEAR_RE.findall(first_page)
    return m[0] if m else None


def guess_authors(first_page: str) -> list[str]:
    lines = [_clean(line) for line in first_page.splitlines() if _clean(line)]
    authors: list[str] = []
    for line in lines[1:12]:
        if "@" in line or "http" in line.lower():
            continue
        if AUTHOR_HINT_RE.match(line) and len(line) <= 120:
            parts = re.split(r",| and ", line)
            authors = [p.strip() for p in parts if len(p.strip()) > 3]
            if authors:
                break
    return authors[:10]


def extract_metadata(full_text: str, *, source: str = "manual") -> PaperMetadata:
    first_page = full_text[:4000]
    return PaperMetadata(
        title=guess_title(first_page),
        authors=guess_authors(first_page),
        year=guess_year(first_page),
        source=source,
    )
