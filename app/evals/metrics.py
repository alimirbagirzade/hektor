"""Retrieval metrics — pure Python, no external dependencies.

Computes Recall@k, Precision@k, MRR, and NDCG@k.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@k — getirilen ilk k chunk'ta ilgili chunk oranı.

    Args:
        retrieved_ids: Getirilen chunk ID'leri (sıralı).
        relevant_ids: Gerçek ilgili chunk ID'leri.
        k: Kesim noktası.

    Returns:
        0.0–1.0 arasında Recall@k değeri.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Precision@k — getirilen ilk k chunk'ta ilgili olanların oranı.

    Args:
        retrieved_ids: Getirilen chunk ID'leri (sıralı).
        relevant_ids: Gerçek ilgili chunk ID'leri.
        k: Kesim noktası.

    Returns:
        0.0–1.0 arasında Precision@k değeri.
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant)
    return hits / k


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank — ilk ilgili sonucun ters sıra değeri.

    Args:
        retrieved_ids: Getirilen chunk ID'leri (sıralı).
        relevant_ids: Gerçek ilgili chunk ID'leri.

    Returns:
        0.0–1.0 arasında MRR değeri.
    """
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """NDCG@k — normalised discounted cumulative gain.

    Args:
        retrieved_ids: Getirilen chunk ID'leri (sıralı).
        relevant_ids: Gerçek ilgili chunk ID'leri.
        k: Kesim noktası.

    Returns:
        0.0–1.0 arasında NDCG@k değeri.
    """
    if not relevant_ids or k == 0:
        return 0.0

    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, rid in enumerate(top_k, start=1):
        if rid in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: relevant olanları sırayla koy
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
