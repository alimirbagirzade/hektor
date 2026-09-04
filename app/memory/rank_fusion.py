"""Reciprocal Rank Fusion (RRF) — sıra-tabanlı liste birleştirme.

Birden fazla retrieval listesini (dense, BM25, çoklu-sorgu varyantları) skorları
normalize etmeye gerek kalmadan birleştirir. Bir öğenin RRF katkısı her liste için
``w / (k + rank)`` (rank 1-tabanlı); ``k`` sabiti (yaygın varsayılan 60) tek bir
listenin sıralamayı domine etmesini engeller. Skor kalibrasyonu gerektirmediği için
karşılaştırılamaz skorlu kaynaklarda (semantik kosinüs mesafesi vs. BM25 frekans skoru)
alpha-harmanından daha sağlamdır.

Kaynak: Cormack, Clarke & Büttcher (2009) "Reciprocal Rank Fusion outperforms Condorcet
and individual rank learning methods"; RAG-Fusion (Raudaschl, 2024). Hibrit aramada
RRF, BM25+vektör birleşiminde ad-hoc skor toplamaya kıyasla tutarlı NDCG/MRR kazancı
raporlanan parametre-az bir temel kabul edilir.

Tasarım: saf Python (bağımlılık yok), **deterministik** (eşit skorda id'ye göre kararlı
sıralama — CLAUDE.md Kural 6), LLM-free (çevrimdışı testlerle uyumlu).
"""

from __future__ import annotations

from collections.abc import Sequence

#: RRF sabiti için yaygın varsayılan (Cormack ve ark. 2009).
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Sıralı id listelerini RRF ile birleştirip ``{id: birleşik_skor}`` döndür.

    Args:
        ranked_lists: Her biri en-iyiden-en-kötüye sıralı id dizisi (örn. dense
            sonuçları, BM25 sonuçları). Boş listeler yok sayılır.
        k: RRF sabiti; büyük ``k`` baş sıraların avantajını yumuşatır (varsayılan 60).
            ``k <= 0`` geçersizdir (sıfıra bölme riski).
        weights: Liste-başı ağırlık (örn. dense'e BM25'ten fazla güven). Verilmezse
            tüm listeler 1.0 ağırlıklı. Uzunluğu ``ranked_lists`` ile eşleşmeli.

    Returns:
        id → birleşik RRF skoru sözlüğü (büyük değer = daha alakalı).

    Raises:
        ValueError: ``k <= 0`` ise veya ``weights`` uzunluğu uyuşmazsa.
    """
    if k <= 0:
        raise ValueError("RRF k > 0 olmalı")
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights uzunluğu ranked_lists ile eşleşmeli")

    fused: dict[str, float] = {}
    for li, lst in enumerate(ranked_lists):
        w = 1.0 if weights is None else float(weights[li])
        for rank, item in enumerate(lst):  # rank 0-tabanlı → formülde +1 ile 1-tabanlı
            fused[item] = fused.get(item, 0.0) + w * (1.0 / (k + rank + 1))
    return fused


def fuse_ranked(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """RRF skoruna göre azalan, **deterministik** birleşik id listesi döndür.

    Eşit RRF skorunda id alfabetik sıraya göre kararlılaştırılır (deterministik çıktı).

    Args:
        ranked_lists: Sıralı id dizileri (bkz. :func:`reciprocal_rank_fusion`).
        k: RRF sabiti.
        weights: Liste-başı ağırlık.

    Returns:
        Birleşik, tekilleştirilmiş, en-alakalıdan en-aza sıralı id listesi.
    """
    scores = reciprocal_rank_fusion(ranked_lists, k=k, weights=weights)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
