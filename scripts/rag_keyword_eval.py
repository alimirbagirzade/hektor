"""Keyword golden-eval — router/hibridin KEYWORD sorgularda faydasını ölçer (Zincir 1).

Mevcut self-retrieval metriği UZUN/semantik (kart-türevi) sorgular kullanıyordu → dense'i
kayırıyor, hibridi adil ölçemiyordu. Bu araç tamamlayıcı: korpustan **ayırt edici NADİR
terimler** (düşük paper-frequency: ticker/kısaltma/teknik terim) çıkarıp KISA keyword
sorguları üretir; her sorgunun golden cevabı o terimin geldiği makaledir. BM25 tam burada
yardımcı olmalı (exact-match). dense-only vs router (lexical→konveks-hibrit) karşılaştırır.

Deterministik, LLM'siz, sızıntı-az (sorgu = ham nadir terim, kart-özeti değil). Temiz
ortamda + TEK BAŞINA çalıştır (Chroma çekişmesi get_all'ı düşürür → boş korpus). Embed
sahte-fallback'i önlemek için BM25 ÖNCE kurulur (CPU doysa bile embed o sırada yapılmaz),
sonra ısınma embed (ollama saptanır), sonra eval.

Kullanım: RAG_KW_LIMIT=50 uv run --no-sync python scripts/rag_keyword_eval.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}")  # ≥4 karakter terim
_STOP = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "which",
    "where",
    "these",
    "those",
    "their",
    "between",
    "using",
    "based",
    "model",
    "models",
    "results",
    "method",
    "methods",
    "figure",
    "table",
    "section",
    "paper",
    "equation",
    "value",
    "values",
    "function",
    "given",
    "shown",
    "different",
}


def main() -> None:
    from app.memory.bm25_corpus import get_corpus_bm25
    from app.memory.reranking_retriever import RerankingRetriever

    limit = int(os.environ.get("RAG_KW_LIMIT", "50"))

    # 1) BM25'i ÖNCE kur (chunk_map = id→RetrievedChunk, paper_id'li). CPU-ağır ama embed yok.
    t = time.perf_counter()
    _bm25, chunk_map = get_corpus_bm25()
    print(f"# bm25 build: {time.perf_counter() - t:.1f}s, chunks={len(chunk_map)}", flush=True)
    if not chunk_map:
        print("# HATA: korpus boş (Chroma çekişmesi? TEK başına çalıştır)", flush=True)
        return

    # 2) Terim → makale-frekansı (kaç farklı makalede geçiyor) + makale → terimler.
    paper_of: dict[str, set[str]] = {}  # term -> {paper_id}
    paper_terms: dict[str, set[str]] = {}  # paper_id -> {term}
    for ch in chunk_map.values():
        pid = ch.paper_id
        toks = {w.lower() for w in _TOKEN_RE.findall(ch.text or "") if w.lower() not in _STOP}
        paper_terms.setdefault(pid, set()).update(toks)
        for w in toks:
            paper_of.setdefault(w, set()).add(pid)

    # 3) Her makale için EN AYIRT EDİCİ terim (en düşük paper-frequency; nadir).
    #    Golden = o makale. Sorgu = nadir terim (gerekirse 2. nadir terimle 2-kelime).
    queries: list[tuple[str, str]] = []  # (query, golden_paper_id)
    for pid, terms in sorted(paper_terms.items()):
        ranked = sorted(terms, key=lambda w: (len(paper_of[w]), -len(w), w))
        rare = [w for w in ranked if len(paper_of[w]) <= 2][:2]  # ≤2 makalede geçen
        if not rare:
            continue
        queries.append((" ".join(rare), pid))
    queries.sort(key=lambda x: x[1])
    if limit:
        queries = queries[:limit]
    print(f"# keyword queries: {len(queries)}", flush=True)

    # 4) Isınma embed (BM25 zaten kurulu → CPU serbest → ollama saptanır, sahte-fallback yok).
    warm = RerankingRetriever(enabled=False)
    warm.retrieve("warmup volatility", top_k=3)

    def measure(name: str, retr: object) -> dict:
        rec = {1: 0, 5: 0, 10: 0}
        rr = 0.0
        lat: list[float] = []
        n = 0
        for q, gold in queries:
            t0 = time.perf_counter()
            try:
                hits = retr.retrieve(q, top_k=10)  # type: ignore[attr-defined]
            except Exception:
                continue
            lat.append((time.perf_counter() - t0) * 1000)
            rank = next((i for i, h in enumerate(hits, 1) if h.paper_id == gold), 0)
            n += 1
            if rank:
                rr += 1.0 / rank
                for kk in rec:
                    if rank <= kk:
                        rec[kk] += 1

        def pct(x: int) -> float:
            return round(100.0 * x / n, 1) if n else 0.0

        return {
            "config": name,
            "n": n,
            "recall@1": pct(rec[1]),
            "recall@5": pct(rec[5]),
            "recall@10": pct(rec[10]),
            "mrr": round(rr / n, 4) if n else 0.0,
            "lat_p50": round(statistics.median(lat), 1) if lat else 0.0,
        }

    configs = [
        ("dense_only", RerankingRetriever(enabled=False)),
        ("router(lexical->convex-hybrid)", RerankingRetriever(enabled=True, router=True)),
    ]
    rows = []
    for name, r in configs:
        row = measure(name, r)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print("\n# config                            recall@1 recall@5 mrr    lat_p50", flush=True)
    for r in rows:
        print(
            f"# {r['config']:<33} {r['recall@1']:>7} {r['recall@5']:>8} "
            f"{r['mrr']:>6} {r['lat_p50']:>8}",
            flush=True,
        )


if __name__ == "__main__":
    main()
