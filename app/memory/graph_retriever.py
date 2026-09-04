"""Graf-tabanlı retrieval — term–chunk bipartite graf + Personalized PageRank (SPRIG-lite).

CPU-only, LLM'siz, deterministik bir GraphRAG dilimi. SPRIG ("Democratizing GraphRAG:
Linear, CPU-Only Graph Retrieval", arXiv:2602.23372) reçetesini Achilles verisiyle uygular:
pahalı LLM graf-inşası yerine **hafif term co-occurrence** ile entity–doküman (term–chunk)
bipartite grafı kurar; sorgu/dense-hit'lerden **tohumlanmış Personalized PageRank** ile
çok-hop ilgili chunk'ları yüzeye çıkarır. Dense'in kaçırdığı (ama paylaşılan terimlerle
bağlı) chunk'ları getirebildiği için RRF ile füzyonu retrieval recall'ını artırabilir.

Tasarım: saf Python (bağımlılık yok), **deterministik** (sabit iterasyon + eşitlikte id ile
kararlı sıralama — CLAUDE.md Kural 6), LLM-free / offline. Anlamsal değil yapısal bir sinyaldir
(proxy); mutlak değil, dense+RRF ile birlikte KIYAS/füzyon için kullanılır.

Hub gürültüsünü sınırlamak için çok sık geçen terimler (`max_df_ratio`) atılır (SPRIG hub
pruning). PageRank bipartite iki-adım güç-iterasyonudur; her iterasyon toplam posting sayısında
lineerdir.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_TERM_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]{3,}")
# Çok yaygın, ayırt edici olmayan terimler (graf kenarı şişmesini engeller).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "was",
        "has",
        "had",
        "can",
        "but",
        "not",
        "you",
        "all",
        "any",
        "one",
        "our",
        "out",
        "its",
        "who",
        "how",
        "may",
        "per",
        "via",
        "use",
        "new",
        "this",
        "that",
        "with",
        "from",
        "have",
        "they",
        "will",
        "would",
        "there",
        "their",
        "what",
        "which",
        "when",
        "into",
        "than",
        "then",
        "them",
        "these",
        "such",
        "also",
        "been",
        "more",
        "most",
        "between",
        "where",
        "while",
        "using",
        "used",
        "based",
        "olan",
        "olarak",
        "daha",
        "için",
        "gibi",
        "veya",
        "ile",
        "ise",
        "göre",
        "kadar",
        "bu",
        "bir",
        "ben",
        "sen",
        "çok",
        "ama",
    }
)


@dataclass
class TermChunkGraph:
    """Term–chunk bipartite graf (postingler + dereceler)."""

    chunk_terms: dict[str, set[str]]  # chunk_id → terim kümesi
    term_chunks: dict[str, set[str]]  # terim → chunk_id kümesi


def extract_terms(text: str, min_len: int = 3) -> set[str]:
    """Metinden ayırt edici terim kümesi (küçük harf, ≥min_len, durak-kelimesiz)."""
    return {t for t in _TERM_RE.findall(text.lower()) if len(t) >= min_len and t not in _STOPWORDS}


def build_graph(
    chunks: Mapping[str, str],
    *,
    min_term_len: int = 3,
    max_df_ratio: float = 0.5,
) -> TermChunkGraph:
    """Chunk metinlerinden term–chunk bipartite grafı kur (deterministik, lineer).

    Args:
        chunks: chunk_id → metin.
        min_term_len: Minimum terim uzunluğu.
        max_df_ratio: Bir terim chunk'ların bu orandan fazlasında geçiyorsa **hub** sayılıp
            atılır (gürültü/maliyet azaltma; SPRIG hub pruning). 1.0 → pruning yok.

    Returns:
        `TermChunkGraph`. Boş girdi → boş graf.
    """
    n = len(chunks)
    chunk_terms: dict[str, set[str]] = {}
    term_chunks: dict[str, set[str]] = {}
    for cid, text in chunks.items():
        terms = extract_terms(text, min_len=min_term_len)
        chunk_terms[cid] = terms
        for t in terms:
            term_chunks.setdefault(t, set()).add(cid)

    if n and max_df_ratio < 1.0:
        max_df = max(1, int(n * max_df_ratio))
        hubs = {t for t, cs in term_chunks.items() if len(cs) > max_df}
        for t in hubs:
            for cid in term_chunks[t]:
                chunk_terms[cid].discard(t)
            del term_chunks[t]

    return TermChunkGraph(chunk_terms=chunk_terms, term_chunks=term_chunks)


def personalized_pagerank(
    graph: TermChunkGraph,
    seeds: Mapping[str, float],
    *,
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """Tohumlanmış (personalized) PageRank — bipartite term–chunk graf üzerinde.

    İki-adımlı güç iterasyonu: chunk → terim → chunk. Yeniden başlatma (restart, olasılık
    ``1-damping``) tohum dağılımına döner → skor tohumların grafsal komşuluğuna yayılır.

    Args:
        graph: Term–chunk grafı.
        seeds: chunk_id → ağırlık (örn. dense-hit skorları). Normalize edilir; boşsa {} döner.
        damping: Sönümleme (yayılma) katsayısı; ``1-damping`` restart olasılığı.
        iterations: Sabit iterasyon sayısı (determinizm).

    Returns:
        chunk_id → PPR skoru (yalnız tohum komşuluğundaki chunk'lar > 0).
    """
    chunk_terms = graph.chunk_terms
    term_chunks = graph.term_chunks
    if not seeds or not chunk_terms:
        return {}

    total = sum(seeds.values())
    if total <= 0:
        # Ağırlıklar geçersiz → tohumlara eşit kütle ver.
        restart = {c: 1.0 / len(seeds) for c in seeds if c in chunk_terms}
    else:
        restart = {c: w / total for c, w in seeds.items() if c in chunk_terms and w > 0}
    if not restart:
        return {}

    rank: dict[str, float] = dict(restart)  # chunk skorları (başlangıç = tohum dağılımı)
    for _ in range(iterations):
        # 1) chunk → terim: terim skoru, içeren chunk'ların derece-normalize toplamı.
        term_rank: dict[str, float] = {}
        for cid, score in rank.items():
            terms = chunk_terms.get(cid)
            if not terms:
                continue
            share = score / len(terms)
            for t in terms:
                term_rank[t] = term_rank.get(t, 0.0) + share
        # 2) terim → chunk: yeni chunk skoru + restart.
        new_rank: dict[str, float] = {c: (1.0 - damping) * w for c, w in restart.items()}
        for t, tscore in term_rank.items():
            chunk_ids = term_chunks.get(t)
            if not chunk_ids:
                continue
            share = damping * tscore / len(chunk_ids)
            for cid in chunk_ids:
                new_rank[cid] = new_rank.get(cid, 0.0) + share
        rank = new_rank

    return rank


def graph_rank(
    chunks: Mapping[str, str],
    seeds: Mapping[str, float],
    *,
    top_k: int | None = None,
    damping: float = 0.85,
    iterations: int = 20,
    max_df_ratio: float = 0.5,
) -> list[str]:
    """Tohumdan PPR ile **deterministik** sıralı chunk_id listesi döndür (kolaylık sarmalayıcı).

    Eşit skorda chunk_id alfabetik (kararlı). Tohumun grafsal olarak erişemediği chunk'lar
    listeye girmez (skor 0).
    """
    graph = build_graph(chunks, max_df_ratio=max_df_ratio)
    scores = personalized_pagerank(graph, seeds, damping=damping, iterations=iterations)
    ranked = sorted(
        (c for c, s in scores.items() if s > 0.0),
        key=lambda c: (-scores[c], c),
    )
    return ranked[:top_k] if top_k is not None else ranked


def seed_weights_from_ids(seed_ids: Sequence[str]) -> dict[str, float]:
    """Sıralı id listesini azalan tohum ağırlıklarına çevir (1. sıra en yüksek)."""
    n = len(seed_ids)
    return {cid: float(n - i) for i, cid in enumerate(seed_ids)}
