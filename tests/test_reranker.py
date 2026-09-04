"""Reranker testleri — formül ve anahtar kelime bazlı sıralama."""

from __future__ import annotations

from app.memory.reranker import Reranker
from app.memory.retrieval_service import RetrievedChunk


def _make_chunk(
    chunk_id: str,
    text: str,
    section_name: str = "introduction",
    distance: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id="paper1",
        text=text,
        page_number=1,
        section_name=section_name,
        title="Test Paper",
        distance=distance,
    )


def test_formula_chunk_ranks_higher() -> None:
    """LaTeX formülü olan chunk, düz metin chunk'ından daha yüksek sıralanmalı."""
    reranker = Reranker()

    plain_chunk = _make_chunk(
        "paper1_c0001",
        text="Momentum is a well-known factor in asset pricing.",
        distance=0.4,
    )
    formula_chunk = _make_chunk(
        "paper1_c0002",
        text=r"The ATR is $ATR_t = \frac{1}{n}\sum_{i=1}^{n} TR_i$ where TR is true range.",
        distance=0.5,  # Daha kötü semantik skor
    )

    results = reranker.rerank("ATR formula", [plain_chunk, formula_chunk])
    assert results[0].chunk_id == "paper1_c0002", "Formül içeren chunk ilk sırada olmalı"


def test_keyword_overlap() -> None:
    """Sorgu anahtar kelimelerini içeren chunk daha yüksek sıralanmalı."""
    reranker = Reranker()

    off_topic = _make_chunk(
        "paper1_c0001",
        text="General discussion about portfolio management and diversification.",
        distance=0.3,
    )
    on_topic = _make_chunk(
        "paper1_c0002",
        text="Volatility momentum strategy with ATR filter shows strong performance.",
        distance=0.5,  # Daha kötü semantik skor
    )

    results = reranker.rerank("volatility momentum ATR strategy", [off_topic, on_topic])
    assert results[0].chunk_id == "paper1_c0002", (
        "Anahtar kelime örtüşmesi yüksek chunk ilk sırada olmalı"
    )


def test_abstract_section_priority() -> None:
    """Abstract bölümündeki chunk, references bölümünden daha yüksek sıralanmalı."""
    reranker = Reranker()

    ref_chunk = _make_chunk(
        "paper1_c0010",
        text="Smith et al. (2019) shows momentum effect.",
        section_name="references",
        distance=0.3,
    )
    abstract_chunk = _make_chunk(
        "paper1_c0001",
        text="We study the momentum effect in financial markets.",
        section_name="abstract",
        distance=0.4,
    )

    results = reranker.rerank("momentum effect", [ref_chunk, abstract_chunk])
    assert results[0].chunk_id == "paper1_c0001", (
        "Abstract bölümü references bölümünden önce gelmeli"
    )


def test_reranker_empty_list() -> None:
    """Boş liste ile boş liste döndürmeli."""
    reranker = Reranker()
    results = reranker.rerank("any query", [])
    assert results == []


def test_reranker_single_chunk() -> None:
    """Tek chunk ile liste sıra değişmeden döndürülmeli."""
    reranker = Reranker()
    chunk = _make_chunk("paper1_c0001", "Some content.")
    results = reranker.rerank("query", [chunk])
    assert len(results) == 1
    assert results[0].chunk_id == "paper1_c0001"
