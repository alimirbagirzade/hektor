"""Bilimsel chunk bütünlüğü testleri — LaTeX, tablo, önceki/sonraki bağlantılar."""

from __future__ import annotations

from app.ingestion.chunker import TextChunk
from app.memory.contextual_chunker import ContextualChunker


def _make_chunk(paper_id: str, index: int, text: str) -> TextChunk:
    return TextChunk(
        paper_id=paper_id,
        chunk_index=index,
        text=text,
        page_number=1,
        section_name=None,
    )


def test_latex_detection() -> None:
    """LaTeX inline formülü içeren chunk has_formula=True olmalı."""
    chunker = ContextualChunker()
    chunks = [_make_chunk("paper1", 0, r"The energy is defined as $E=mc^2$ where m is mass.")]
    flags = chunker.annotate(chunks)
    assert len(flags) == 1
    assert flags[0].has_formula is True, "Inline $ formülü tespit edilmeli"


def test_display_latex_detection() -> None:
    r"""Display formülü \[...\] içeren chunk has_formula=True olmalı."""
    chunker = ContextualChunker()
    chunks = [
        _make_chunk(
            "paper1",
            0,
            r"The formula is \[ \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2} \]",
        )
    ]
    flags = chunker.annotate(chunks)
    assert flags[0].has_formula is True


def test_incomplete_formula_detection() -> None:
    r"""Eşleşmemiş \( ile biten chunk has_incomplete_formula=True olmalı."""
    chunker = ContextualChunker()
    chunks = [
        _make_chunk(
            "paper1",
            0,
            r"The following formula continues: \( ATR = \frac{1}{n}",
        )
    ]
    flags = chunker.annotate(chunks)
    assert flags[0].has_formula is True
    assert flags[0].has_incomplete_formula is True, "Eksik formül tespit edilmeli"


def test_prev_next_links() -> None:
    """3 chunk'lu zincirde önceki/sonraki bağlantılar doğru kurulmalı."""
    chunker = ContextualChunker()
    chunks = [
        _make_chunk("paper1", 0, "First chunk content."),
        _make_chunk("paper1", 1, "Second chunk content."),
        _make_chunk("paper1", 2, "Third chunk content."),
    ]
    flags = chunker.annotate(chunks)

    assert len(flags) == 3

    # İlk chunk: önceki yok, sonraki var
    assert flags[0].previous_chunk_id is None
    assert flags[0].next_chunk_id == "paper1_c0001"

    # Orta chunk: önceki ve sonraki var
    assert flags[1].previous_chunk_id == "paper1_c0000"
    assert flags[1].next_chunk_id == "paper1_c0002"

    # Son chunk: önceki var, sonraki yok
    assert flags[2].previous_chunk_id == "paper1_c0001"
    assert flags[2].next_chunk_id is None


def test_table_detection() -> None:
    """Markdown tablo içeren chunk has_table=True olmalı."""
    chunker = ContextualChunker()
    table_text = """Results:

| Strategy | Sharpe | Return |
|----------|--------|--------|
| Momentum | 1.2    | 15%    |
| BuyHold  | 0.8    | 10%    |
"""
    chunks = [_make_chunk("paper1", 0, table_text)]
    flags = chunker.annotate(chunks)
    assert flags[0].has_table is True


def test_definition_detection() -> None:
    """'definition' kelimesi içeren chunk has_definition=True olmalı."""
    chunker = ContextualChunker()
    chunks = [_make_chunk("paper1", 0, "By definition, the Sharpe ratio is risk-adjusted return.")]
    flags = chunker.annotate(chunks)
    assert flags[0].has_definition is True


def test_no_formula_plain_text() -> None:
    """Düz metin chunk has_formula=False olmalı."""
    chunker = ContextualChunker()
    chunks = [_make_chunk("paper1", 0, "This is a simple sentence without any math.")]
    flags = chunker.annotate(chunks)
    assert flags[0].has_formula is False
    assert flags[0].has_incomplete_formula is False


def test_empty_chunk_list() -> None:
    """Boş liste boş liste döndürmeli."""
    chunker = ContextualChunker()
    flags = chunker.annotate([])
    assert flags == []
