"""Retrieval evaluator — computes retrieval metrics against golden questions."""

from __future__ import annotations

from dataclasses import dataclass

from app.evals.golden_dataset import GoldenQuestion
from app.evals.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from app.memory.retrieval_service import Retriever


@dataclass
class RetrievalEvalResult:
    """Tek bir soru için retrieval değerlendirme sonucu."""

    question_id: str
    recall_5: float
    recall_10: float
    precision_5: float
    mrr: float
    ndcg: float


class RetrievalEvaluator:
    """Altın veri setiyle retrieval sistemini değerlendiren sınıf."""

    def __init__(self, retriever: Retriever) -> None:
        # `Retriever` arayüzü: düz `RetrievalService` veya robust `RerankingRetriever`
        # (over-fetch + rerank) geçilebilir → eval, canlı RAG yolunu ölçebilir.
        self._retriever = retriever

    def evaluate(self, questions: list[GoldenQuestion]) -> list[RetrievalEvalResult]:
        """Her soru için retrieval metriklerini hesapla.

        Args:
            questions: Değerlendirilecek altın sorular.

        Returns:
            Her soru için RetrievalEvalResult listesi.
        """
        results: list[RetrievalEvalResult] = []

        for q in questions:
            try:
                chunks_10 = self._retriever.retrieve(q.question_text, top_k=10)
            except Exception:
                chunks_10 = []

            retrieved_ids = [c.chunk_id for c in chunks_10]
            relevant_ids = q.expected_chunk_ids

            results.append(
                RetrievalEvalResult(
                    question_id=q.question_id,
                    recall_5=recall_at_k(retrieved_ids, relevant_ids, 5),
                    recall_10=recall_at_k(retrieved_ids, relevant_ids, 10),
                    precision_5=precision_at_k(retrieved_ids, relevant_ids, 5),
                    mrr=mrr(retrieved_ids, relevant_ids),
                    ndcg=ndcg_at_k(retrieved_ids, relevant_ids, 10),
                )
            )

        return results
