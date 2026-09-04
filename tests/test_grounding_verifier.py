"""Grounding doğrulayıcı testleri — hedge'li halüsinasyon regresyonu."""

from __future__ import annotations

from app.memory.retrieval_service import RetrievedChunk
from app.verification.grounding_verifier import GroundingLevel, GroundingVerifier


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c0",
        paper_id="p",
        text=text,
        page_number=1,
        section_name=None,
        title=None,
        distance=0.1,
    )


def test_speculative_but_unsupported_is_unsupported() -> None:
    # Hedge'li ('might/possibly') AMA chunk'larda dayanağı OLMAYAN iddia: SPECULATIVE(0.3)
    # değil UNSUPPORTED(0.0) sayılmalı (örtülü halüsinasyon skoru şişmesin).
    gv = GroundingVerifier()
    chunks = [_chunk("Completely unrelated content about cooking recipes and kitchen utensils.")]
    res = gv.verify(
        "This trading approach might possibly yield enormous guaranteed returns soon.", chunks
    )
    assert res
    assert all(r.level == GroundingLevel.UNSUPPORTED for r in res)
