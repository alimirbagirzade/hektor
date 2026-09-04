"""Bağlam yeterliliği sınıflandırıcı testleri."""

from __future__ import annotations

from app.memory.retrieval_service import RetrievedChunk
from app.verification.context_sufficiency import (
    ContextSufficiencyClassifier,
    SufficiencyLevel,
)


def _make_chunk(chunk_id: str = "paper1_c0001") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id="paper1",
        text="Momentum strategies have been shown to outperform buy-and-hold.",
        page_number=1,
        section_name="introduction",
        title="Test Paper",
        distance=0.3,
    )


def test_no_chunks_insufficient() -> None:
    """Boş chunk listesi INSUFFICIENT ve can_answer=False olmalı."""
    classifier = ContextSufficiencyClassifier()
    result = classifier.classify("volatility query", [])

    assert result.level == SufficiencyLevel.INSUFFICIENT
    assert result.can_answer is False
    assert len(result.missing_items) > 0


def test_good_chunks_sufficient() -> None:
    """Üç veya daha fazla normal chunk SUFFICIENT ve can_answer=True olmalı."""
    classifier = ContextSufficiencyClassifier()
    chunks = [
        _make_chunk("paper1_c0001"),
        _make_chunk("paper1_c0002"),
        _make_chunk("paper1_c0003"),
    ]
    result = classifier.classify("momentum strategy", chunks)

    assert result.level == SufficiencyLevel.SUFFICIENT
    assert result.can_answer is True


def test_single_chunk_partially_sufficient() -> None:
    """Tek chunk PARTIALLY_SUFFICIENT ve can_answer=True olmalı."""
    classifier = ContextSufficiencyClassifier()
    chunks = [_make_chunk("paper1_c0001")]
    result = classifier.classify("simple query", chunks)

    assert result.level == SufficiencyLevel.PARTIALLY_SUFFICIENT
    assert result.can_answer is True


def test_quality_flags_impact() -> None:
    """Kalite bayrakları sınıflandırmayı etkilemeli."""
    from app.memory.contextual_chunker import ChunkQualityFlags

    classifier = ContextSufficiencyClassifier()
    chunks = [_make_chunk("paper1_c0001")]

    # Eksik formül bayrağı + komşu yok
    flags = [
        ChunkQualityFlags(
            chunk_id="paper1_c0001",
            has_formula=True,
            has_incomplete_formula=True,
            needs_adjacent_context=True,
            next_chunk_id=None,  # Komşu yok
        )
    ]
    result = classifier.classify("formula query", chunks, quality_flags=flags)

    # Formül eksik ve komşu yoksa yeterlilik düşmeli
    assert result.level in (
        SufficiencyLevel.MISSING_FORMULA_CONTINUATION,
        SufficiencyLevel.PARTIALLY_SUFFICIENT,
    )


def test_mismatched_quality_flags_do_not_force_missing_formula() -> None:
    """chunks ile HİZASIZ quality_flags (başka chunk_id'ler) yanlış MISSING_FORMULA_CONTINUATION
    üretmemeli. all_have_incomplete kıyası yalnız getirilen chunk'ların bayraklarına dayanmalı —
    aksi halde len(incomplete_ids)==len(chunks) iki AYRI koleksiyonu kıyaslar ve can_answer'ı
    haksız yere False yapardı."""
    from app.memory.contextual_chunker import ChunkQualityFlags

    classifier = ContextSufficiencyClassifier()
    chunks = [_make_chunk("paper1_c0001")]  # sağlam chunk (eksik formül YOK)
    # Bayrak BAŞKA bir chunk için — getirilen sette yok → dikkate alınmamalı.
    flags = [
        ChunkQualityFlags(
            chunk_id="other_paper_c9999",
            has_formula=True,
            has_incomplete_formula=True,
            needs_adjacent_context=True,
            next_chunk_id=None,
        )
    ]
    result = classifier.classify("query", chunks, quality_flags=flags)
    assert result.level != SufficiencyLevel.MISSING_FORMULA_CONTINUATION
    assert result.can_answer is True
