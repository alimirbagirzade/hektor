"""Retrieval değerlendirici — ``expected_chunk_ids`` / ``expected_document_ids`` ile deterministik.

Mevcut ``app.evals.retrieval_eval`` (GoldenQuestion tabanlı) değişmez; bu modül profil eval
şemasına (EvalItem) göre ve RetrievalTrace üzerinden çalışır.
"""

from __future__ import annotations

from typing import Any

from app.evals.profile.schema import EvalItem


def evaluate_retrieval(
    item: EvalItem, retrieved_chunk_ids: list[str], retrieved_document_ids: list[str]
) -> dict[str, Any]:
    exp_chunks = set(item.expected_chunk_ids)
    exp_docs = set(item.expected_document_ids)
    out: dict[str, Any] = {
        "recall_at_k": None,
        "precision": None,
        "context_relevance": None,
        "k": len(retrieved_chunk_ids),
    }
    if exp_chunks:
        hit = [c for c in retrieved_chunk_ids if c in exp_chunks]
        out["recall_at_k"] = len(set(hit)) / len(exp_chunks)
        out["precision"] = len(hit) / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0.0
    if exp_docs:
        rel = [d for d in retrieved_document_ids if d in exp_docs]
        out["context_relevance"] = (
            len(rel) / len(retrieved_document_ids) if retrieved_document_ids else 0.0
        )
        if out["recall_at_k"] is None:  # chunk beklentisi yoksa doküman düzeyinde recall
            out["recall_at_k"] = len(set(rel)) / len(exp_docs)
    return out
