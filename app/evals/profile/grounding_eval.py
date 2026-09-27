"""Grounding + atıf doğruluğu — document_id/chunk_id eşleşmesiyle (LLM judge YOK).

Atıf biçimi Hektor RAG'ının ``RetrievedChunk.citation`` biçimidir: ``[paper_id:chunk_id]``
(opsiyonel ``, s.N`` sayfa eki).

- grounding          = atıf yapılan chunk'lardan GERÇEKTEN getirilmiş olanların oranı.
                       requires_rag iken hiç atıf yoksa 0. RAG'sız sistemde None.
- citation_accuracy  = atıfların beklenen chunk'larla (varsa; yoksa getirilenlerle) eşleşme
                       oranı.
- fabricated         = getirilmemiş bir chunk'a atıf sayısı (uydurma kaynak → halüsinasyon).
"""

from __future__ import annotations

import re
from typing import Any

from app.evals.profile.schema import EvalItem

_CITATION = re.compile(r"\[([\w.\-]+):([\w.\-]+)(?:,[^\]]*)?\]")


def parse_citations(answer: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _CITATION.finditer(answer)]


def evaluate_grounding(
    item: EvalItem, answer: str, retrieved_chunk_ids: list[str] | None
) -> dict[str, Any]:
    if retrieved_chunk_ids is None:  # RAG'sız sistem
        return {"grounding": None, "citation_accuracy": None, "fabricated": 0, "citations": []}
    cites = parse_citations(answer)
    cited_chunks = [c for _, c in cites]
    retrieved = set(retrieved_chunk_ids)
    if not cited_chunks:
        grounding = 0.0 if item.requires_rag and not item.requires_abstention else None
        return {
            "grounding": grounding,
            "citation_accuracy": 0.0 if item.requires_rag and grounding is not None else None,
            "fabricated": 0,
            "citations": [],
        }
    in_retrieved = [c for c in cited_chunks if c in retrieved]
    fabricated = len(cited_chunks) - len(in_retrieved)
    reference = set(item.expected_chunk_ids) or retrieved
    correct = [c for c in cited_chunks if c in reference]
    return {
        "grounding": len(in_retrieved) / len(cited_chunks),
        "citation_accuracy": len(correct) / len(cited_chunks),
        "fabricated": fabricated,
        "citations": [f"{p}:{c}" for p, c in cites],
    }
