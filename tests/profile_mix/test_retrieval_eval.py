"""Retrieval eval: expected_chunk_ids / expected_document_ids ile deterministik."""

from __future__ import annotations

from app.evals.profile.retrieval_eval import evaluate_retrieval
from app.evals.profile.schema import EvalItem
from app.memory.rag_version import RetrievalTrace

from .mix_helpers import FakeChunk
from .test_eval_runner import snapshot


def _item(**kw: object) -> EvalItem:
    return EvalItem(id="x", domain="trading", question="q", **kw)  # type: ignore[arg-type]


def test_recall_and_precision_by_chunk() -> None:
    r = evaluate_retrieval(_item(expected_chunk_ids=["c1", "c2"]), ["c1", "c9", "c8"], [])
    assert r["recall_at_k"] == 0.5
    assert abs(r["precision"] - 1 / 3) < 1e-12 and r["k"] == 3


def test_document_level_fallback_and_context_relevance() -> None:
    r = evaluate_retrieval(_item(expected_document_ids=["p1"]), ["c1", "c2"], ["p1", "p2"])
    assert r["recall_at_k"] == 1.0 and r["context_relevance"] == 0.5


def test_no_expectation_means_unmeasured_not_zero() -> None:
    r = evaluate_retrieval(_item(), ["c1"], ["p1"])
    assert r["recall_at_k"] is None and r["precision"] is None


def test_empty_retrieval_scores_zero() -> None:
    r = evaluate_retrieval(_item(expected_chunk_ids=["c1"]), [], [])
    assert r["recall_at_k"] == 0.0 and r["precision"] == 0.0


def test_retrieval_trace_records_versions_and_scores() -> None:
    snap = snapshot()
    t = RetrievalTrace.from_chunks(snap, [FakeChunk("c1", "p1", "x", distance=0.2)])
    d = t.to_dict()
    assert d["rag_version"] == snap.rag_version
    assert d["embedding_model"] == "nomic-embed-text" and d["reranker"] == "heuristic"
    assert d["retrieved_document_ids"] == ["p1"] and d["retrieved_chunk_ids"] == ["c1"]
    assert d["retrieval_scores"] == [0.2] and d["top_k"] == 2
