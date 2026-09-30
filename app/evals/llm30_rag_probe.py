"""RAG proje-dokümanı yanlılığı ölçümü (protokol Aşama 4) — SALT OKUMA.

Soru: LLM-30 teknik sorularına retrieval neden projenin kendi kılavuzlarını getiriyor?
Her soru için aşama aşama (dense top-24 → BM25 top-24 → sezgisel rerank top-6) iç-doküman
payını ve ablasyonları ölçer: anahtar-kelime terimi olmadan rerank, nomic ``search_query:``
öneki, aynı sorunun İngilizce karşılığı (dil hipotezi). Ayar/indeks DEĞİŞTİRMEZ; sonuç
``reports/evals/llm30/rag_guide_bias_probe.json``. Bir RAG düzeltmesinden önce/sonra aynı
komutla koşulur (retrieval değişimi ayrı deney olarak işaretlenir).

    uv run python -m app.evals.llm30_rag_probe
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from app.evals.llm30 import load_llm30
from app.memory import reranker as rr
from app.memory.bm25_corpus import get_corpus_bm25
from app.memory.reranking_retriever import RerankingRetriever
from app.memory.retrieval_service import RetrievedChunk

#: Korpustaki proje iç dokümanları (Türkçe kılavuzlar; papers.source hepsinde 'manual' →
#: bugün kaynak alanıyla ayırt edilemiyor).
INTERNAL_DOCS: frozenset[str] = frozenset({"paper_cd76dd9a893f", "paper_3afecc5fb4da"})
K, CANDIDATES = 6, 24

#: Dil hipotezi: aynı içerik, İngilizce. Elle çevrildi (2026-09-30); sabit tutulur.
ENGLISH_VARIANTS: dict[str, str] = {
    "llm30-s12": "Write the standard EMA formula where alpha is the weight of the new "
    "observation. Compute the result for initial EMA=100, new price=110, alpha=0.1 and 0.5. "
    "How do lag and smoothing change as alpha decreases? Is fixed-alpha EMA linear? IIR view.",
    "llm30-s14": "Write the one-dimensional local level Kalman filter state and observation "
    "equations, explain Q and R. Prior estimate 100, prior variance 4, observation 106, R=2: "
    "compute the Kalman gain and the updated state. Filtering versus smoothing.",
    "llm30-s18": "Spread x - beta*y for mean reversion: must individual prices be stationary? "
    "Correlation versus cointegration. Estimate beta, test spread stationarity with ADF, "
    "structural breaks. ADF null hypothesis and p=0.20 interpretation.",
    "llm30-s19": "Compute the Shannon entropy in bits of p=[0.5,0.5] and p=[1,0] with fixed "
    "bins. Distinguish histogram density output from probability mass. Is entropy evidence of "
    "trend or mean reversion? Regime filter test plan.",
    "llm30-s22": "State sequence [0,1,0,2,1,0]: compute transition counts and the "
    "row-normalized Markov transition matrix. Avoid index alignment errors. Confidence or "
    "posterior intervals for rare transitions with small samples.",
    "llm30-s24": "Hidden Markov model regimes: hidden state, transition and emission "
    "distributions. Choosing the number of regimes with AIC/BIC, chronological validation, "
    "label switching. Filtered p(z_t|x_1:t) versus smoothed or Viterbi labels.",
}


def internal_share(chunks: list[RetrievedChunk]) -> list[int]:
    return [sum(c.paper_id in INTERNAL_DOCS for c in chunks), len(chunks)]


def rerank_without_keyword(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Sezgisel rerank'in anahtar-kelime terimi (0.30) çıkarılmış hâli (ablasyon)."""
    scored = []
    for c in chunks:
        text = c.text or ""
        raw = (
            0.40 * rr._semantic_score(c.distance)
            + 0.20 * rr._section_priority_score(c.section_name)
            + 0.10 * (1.0 if rr._has_formula(text) else 0.0)
        ) / 0.70
        scored.append((c, math.tanh(raw * 2.0) / math.tanh(2.0)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def probe_question(
    retriever: RerankingRetriever,
    bm25: Any,
    chunk_map: dict[str, RetrievedChunk],
    qid: str,
    question: str,
) -> dict[str, Any]:
    dense = retriever.base.retrieve(question, top_k=CANDIDATES)
    lexical = [chunk_map[cid] for cid, _ in bm25.search(question, CANDIDATES) if cid in chunk_map]
    pool = list(dense)
    seen = {c.chunk_id for c in pool}
    for c in lexical:
        if c.chunk_id not in seen:
            pool.append(c)
            seen.add(c.chunk_id)
    final = rr.Reranker().rerank(question, pool)[:K]
    internal = [c for c in pool if c.paper_id in INTERNAL_DOCS]
    external = [c for c in pool if c.paper_id not in INTERNAL_DOCS]
    row: dict[str, Any] = {
        "id": qid,
        "dense24": internal_share(dense),
        "bm25_24": internal_share(lexical),
        "final6": internal_share(final),
        "dense6": internal_share(dense[:K]),
        "rerank_no_keyword6": internal_share(rerank_without_keyword(pool)[:K]),
        "nomic_prefix6": internal_share(
            retriever.base.retrieve("search_query: " + question, top_k=K)
        ),
        "bm25_without_distance": sum(c.distance is None for c in lexical),
        "keyword_overlap_internal": _mean(
            [rr._keyword_overlap_score(question, c.text or "") for c in internal]
        ),
        "keyword_overlap_external": _mean(
            [rr._keyword_overlap_score(question, c.text or "") for c in external]
        ),
        "final_paper_ids": [c.paper_id for c in final],
    }
    if qid in ENGLISH_VARIANTS:
        en = ENGLISH_VARIANTS[qid]
        en_final = retriever.retrieve(en, top_k=K)
        row["english_dense6"] = internal_share(retriever.base.retrieve(en, top_k=K))
        row["english_final6"] = internal_share(en_final)
        row["english_final_titles"] = [c.title for c in en_final]
    return row


def main() -> None:
    retriever = RerankingRetriever()
    bm25, chunk_map = get_corpus_bm25()
    if bm25 is None:
        raise SystemExit("BM25 korpusu yok — ölçüm yapılamadı")
    rows = []
    totals: Counter[str] = Counter()
    for it in load_llm30(purpose="audit"):
        row = probe_question(retriever, bm25, chunk_map, it.id, it.question)
        rows.append(row)
        for key in (
            "dense24",
            "bm25_24",
            "final6",
            "dense6",
            "rerank_no_keyword6",
            "nomic_prefix6",
        ):
            totals[f"{key}_internal"] += row[key][0]
            totals[f"{key}_n"] += row[key][1]
        totals["questions_with_internal_in_final6"] += row["final6"][0] > 0
        totals["questions_all_internal_final6"] += row["final6"][0] == row["final6"][1]
        print(
            f"{it.id} final6={row['final6']} dense24={row['dense24']} bm25_24={row['bm25_24']}",
            flush=True,
        )
    out = Path("reports/evals/llm30/rag_guide_bias_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"rows": rows, "totals": dict(totals)}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dict(totals), ensure_ascii=False))
    print(f"Yazıldı: {out}")


if __name__ == "__main__":
    main()
