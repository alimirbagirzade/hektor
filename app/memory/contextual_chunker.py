"""Contextual chunk quality annotator.

Detects quality flags (formula, table, definition, theorem, incomplete argument)
for each chunk and establishes prev/next neighbor links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.chunker import TextChunk

# ---------------------------------------------------------------------------
# LaTeX formül desenleri
# ---------------------------------------------------------------------------
_LATEX_INLINE = re.compile(r"\$[^$\n]+\$")
_LATEX_DISPLAY = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_LATEX_ENV_OPEN = re.compile(r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}")
_LATEX_ENV_CLOSE = re.compile(r"\\end\{(equation|align|gather|multline|eqnarray)\*?\}")
_LATEX_PAREN_OPEN = re.compile(r"\\\(")
_LATEX_PAREN_CLOSE = re.compile(r"\\\)")
_LATEX_BRACKET_OPEN = re.compile(r"\\\[")
_LATEX_BRACKET_CLOSE = re.compile(r"\\\]")
_LATEX_MACRO = re.compile(r"\\(frac|sum|int|prod|sqrt|lim|infty|alpha|beta|gamma)\b")

# Tablo deseni (markdown pipe tablosu)
_TABLE_ROW = re.compile(r"^\s*\|.+\|", re.MULTILINE)

# Tanım / teorem desenleri
_DEFINITION_RE = re.compile(
    r"\b(definition|tanım|define|defined as|is defined|denoted by)\b",
    re.IGNORECASE,
)
_THEOREM_RE = re.compile(
    r"\b(theorem|teorem|lemma|corollary|proposition|proof|kanıt)\b",
    re.IGNORECASE,
)

# Argüman bitimsizliği sinyalleri
_INCOMPLETE_ARG_RE = re.compile(
    r"\b(however|but|although|despite|on the other hand|ise|ancak|fakat|rağmen)\s*$",
    re.IGNORECASE,
)

# Bölüm başlığı (## veya # ile başlayan satır)
_SECTION_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_SUBSECTION_HEADING_RE = re.compile(r"^#{3,4}\s+(.+)$", re.MULTILINE)


def _count_unmatched(text: str, open_pat: re.Pattern[str], close_pat: re.Pattern[str]) -> int:
    """Açılmamış/kapanmamış delimiter sayısını döndür."""
    opens = len(open_pat.findall(text))
    closes = len(close_pat.findall(text))
    return abs(opens - closes)


def _detect_has_formula(text: str) -> bool:
    """Metinde LaTeX formülü var mı?"""
    return bool(
        _LATEX_INLINE.search(text)
        or _LATEX_DISPLAY.search(text)
        or _LATEX_ENV_OPEN.search(text)
        or _LATEX_PAREN_OPEN.search(text)
        or _LATEX_BRACKET_OPEN.search(text)
        or _LATEX_MACRO.search(text)
    )


def _detect_incomplete_formula(text: str) -> bool:
    """Chunk eksik/yarım formül içeriyor mu?"""
    # Eşleşmemiş delimiter
    if _count_unmatched(text, _LATEX_PAREN_OPEN, _LATEX_PAREN_CLOSE) > 0:
        return True
    if _count_unmatched(text, _LATEX_BRACKET_OPEN, _LATEX_BRACKET_CLOSE) > 0:
        return True
    # Açık ortam kapanmamış
    open_envs = len(_LATEX_ENV_OPEN.findall(text))
    close_envs = len(_LATEX_ENV_CLOSE.findall(text))
    if open_envs != close_envs:
        return True
    # Tek dolar: tek sayıda $ → muhtemelen yarım
    stripped = _LATEX_DISPLAY.sub("", text)
    inline_count = stripped.count("$")
    return inline_count % 2 != 0


def _detect_has_table(text: str) -> bool:
    """Markdown tablo sözdizimi var mı?"""
    return bool(_TABLE_ROW.search(text))


def _detect_section_name(text: str) -> str | None:
    m = _SECTION_HEADING_RE.search(text)
    return m.group(1).strip() if m else None


def _detect_subsection_name(text: str) -> str | None:
    m = _SUBSECTION_HEADING_RE.search(text)
    return m.group(1).strip() if m else None


@dataclass
class ChunkQualityFlags:
    """Bir chunk için hesaplanan kalite ve bağlam bayrakları."""

    chunk_id: str
    has_formula: bool = False
    has_incomplete_formula: bool = False
    has_table: bool = False
    has_definition: bool = False
    has_theorem: bool = False
    has_incomplete_argument: bool = False
    needs_adjacent_context: bool = False
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    paper_title: str = ""
    section_name: str | None = None
    subsection_name: str | None = None
    argument_type: str | None = None  # "deductive" | "inductive" | "analogy" | None
    # Argüman tipi tespit edilirse doldurulur (gelecek genişleme için)
    extra: dict = field(default_factory=dict)


class ContextualChunker:
    """Chunk listesine kalite bayrakları ve bağlam bağlantıları ekler.

    Mevcut chunker'a bağımlı değildir; TextChunk listesi alır ve
    ChunkQualityFlags listesi döndürür.
    """

    def annotate(
        self,
        chunks: list[TextChunk],
        paper_title: str = "",
    ) -> list[ChunkQualityFlags]:
        """Her chunk için kalite bayraklarını tespit et ve komşu bağlantıları kur.

        Args:
            chunks: TextChunk listesi (sıralı olmalı).
            paper_title: Makale başlığı (isteğe bağlı).

        Returns:
            Chunk sayısıyla birebir örtüşen ChunkQualityFlags listesi.
        """
        if not chunks:
            return []

        flags: list[ChunkQualityFlags] = []

        for i, chunk in enumerate(chunks):
            text = chunk.text
            has_f = _detect_has_formula(text)
            has_if = _detect_incomplete_formula(text) if has_f else False
            has_t = _detect_has_table(text)
            has_d = bool(_DEFINITION_RE.search(text))
            has_th = bool(_THEOREM_RE.search(text))
            has_ia = bool(_INCOMPLETE_ARG_RE.search(text.strip()))

            # Komşu chunk bağlantıları
            prev_id = chunks[i - 1].chunk_id if i > 0 else None
            next_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None

            needs_adj = has_if or has_ia

            sec = _detect_section_name(text) or chunk.section_name
            subsec = _detect_subsection_name(text)

            flags.append(
                ChunkQualityFlags(
                    chunk_id=chunk.chunk_id,
                    has_formula=has_f,
                    has_incomplete_formula=has_if,
                    has_table=has_t,
                    has_definition=has_d,
                    has_theorem=has_th,
                    has_incomplete_argument=has_ia,
                    needs_adjacent_context=needs_adj,
                    previous_chunk_id=prev_id,
                    next_chunk_id=next_id,
                    paper_title=paper_title,
                    section_name=sec,
                    subsection_name=subsec,
                    argument_type=None,
                )
            )

        return flags
