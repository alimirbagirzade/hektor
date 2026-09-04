"""PDF parser tests.

Generates a tiny real PDF with PyMuPDF (installed) so the extraction path is
exercised offline and deterministically — no fixture binary checked into git.
"""

from __future__ import annotations

import pytest

from app.ingestion.pdf_parser import ParsedPdf, parse_pdf


def _make_pdf(path, pages_text: list[str]) -> None:
    import fitz  # PyMuPDF

    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def test_parse_single_page(tmp_path):
    pdf = tmp_path / "one.pdf"
    _make_pdf(pdf, ["Volatilite kumelenmesi momentum kaliciligini etkiler."])

    parsed = parse_pdf(pdf)

    assert isinstance(parsed, ParsedPdf)
    assert parsed.n_pages == 1
    assert "Volatilite" in parsed.text
    assert parsed.n_chars > 0


def test_parse_multi_page_concatenates(tmp_path):
    pdf = tmp_path / "multi.pdf"
    _make_pdf(pdf, ["BIRINCI SAYFA metni", "IKINCI SAYFA metni"])

    parsed = parse_pdf(pdf)

    assert parsed.n_pages == 2
    assert "BIRINCI" in parsed.text
    assert "IKINCI" in parsed.text


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "nope.pdf")
