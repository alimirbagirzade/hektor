"""Atıf doğrulayıcı testleri."""

from __future__ import annotations

from app.memory.retrieval_service import RetrievedChunk
from app.verification.citation_verifier import CitationVerifier


def _make_chunk(
    chunk_id: str, paper_id: str, text: str = "ATR measures volatility."
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        page_number=1,
        section_name="introduction",
        title="Test Paper",
        distance=0.3,
    )


def test_valid_citation() -> None:
    """Geçerli atıf exists=True döndürmeli."""
    verifier = CitationVerifier()
    chunks = [
        _make_chunk(
            chunk_id="paper1_c0001",
            paper_id="paper1",
            text="The ATR indicator measures volatility in financial markets.",
        )
    ]
    answer = "ATR volatility measure [paper1:paper1_c0001] is commonly used."
    results = verifier.verify(answer, chunks)

    assert len(results) == 1
    assert results[0].exists is True
    assert results[0].paper_id == "paper1"
    assert results[0].chunk_id == "paper1_c0001"


def test_citation_with_page_suffix() -> None:
    """Sayfalı atıf ([paper:chunk, s.N]) — sistemin kendi ürettiği gerçek format."""
    verifier = CitationVerifier()
    chunk = _make_chunk("paper1_c0007", "paper1")
    assert chunk.citation == "[paper1:paper1_c0007, s.1]"  # RetrievedChunk gerçek çıktısı
    answer = f"İddia {chunk.citation} doğru."
    results = verifier.verify(answer, [chunk])
    assert len(results) == 1
    assert results[0].chunk_id == "paper1_c0007"  # ", s.1" ayıklandı
    assert results[0].exists is True


def test_invalid_citation() -> None:
    """Sahte atıf exists=False döndürmeli."""
    verifier = CitationVerifier()
    chunks = [
        _make_chunk(
            chunk_id="paper1_c0001",
            paper_id="paper1",
        )
    ]
    answer = "Some claim [fake_paper:c9999] about momentum."
    results = verifier.verify(answer, chunks)

    assert len(results) == 1
    assert results[0].exists is False


def test_no_citations_empty_result() -> None:
    """Atıf olmayan cevap boş liste döndürmeli."""
    verifier = CitationVerifier()
    chunks = [_make_chunk("paper1_c0001", "paper1")]
    answer = "This sentence has no citations at all."
    results = verifier.verify(answer, chunks)
    assert results == []


def test_multiple_citations() -> None:
    """Birden fazla atıf doğru şekilde kontrol edilmeli."""
    verifier = CitationVerifier()
    chunks = [
        _make_chunk("paper1_c0001", "paper1"),
        _make_chunk("paper2_c0005", "paper2"),
    ]
    answer = (
        "First point [paper1:paper1_c0001]. "
        "Second point [paper2:paper2_c0005]. "
        "Third [ghost:ghost_c001]."
    )
    results = verifier.verify(answer, chunks)

    assert len(results) == 3
    existing = [r for r in results if r.exists]
    missing = [r for r in results if not r.exists]
    assert len(existing) == 2
    assert len(missing) == 1
