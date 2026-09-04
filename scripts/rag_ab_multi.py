"""RAG retrieval A/B — TEK süreçte birden çok config (BM25 bir kez kurulur).

`rag_retrieval_ab.py` tek config ölçer ve her koşu BM25'i yeniden kurar (~125s).
Bu araç embedder/Chroma/BM25'i BİR kez ısıtır, sonra `RerankingRetriever`'ı farklı
bayraklarla (dense-only / hibrit+rerank / RRF) kurarak aynı sorgu setinde karşılaştırır.
Böylece config'ler ADİL (aynı koşul) ve hızlı ölçülür.

Metrik: makale-düzeyi self-retrieval recall@1/3/5/10 + MRR + gecikme (ms).
Çıktı: her config için tek satır JSON + sonda karşılaştırma tablosu.

Kullanım:
    RAG_AB_LIMIT=60 uv run --no-sync python scripts/rag_ab_multi.py
"""

from __future__ import annotations

import json
import os
import statistics
import time


def _card_query(cj: dict) -> str:
    parts = [str(cj.get("title") or ""), str(cj.get("main_claim") or "")]
    dom = cj.get("domain")
    if dom:
        parts.append(str(dom))
    return ". ".join(p for p in parts if p).strip()[:400]


def _measure(name: str, retriever, items: list[tuple[str, dict]]) -> dict:
    rec = {1: 0, 3: 0, 5: 0, 10: 0}
    rr = 0.0
    lat: list[float] = []
    n = 0
    for pid, cj in items:
        q = _card_query(cj)
        if not q:
            continue
        t0 = time.perf_counter()
        try:
            chunks = retriever.retrieve(q, top_k=10)
        except Exception:
            continue
        lat.append((time.perf_counter() - t0) * 1000.0)
        rank = 0
        for i, ch in enumerate(chunks, start=1):
            if ch.paper_id == pid:
                rank = i
                break
        n += 1
        if rank:
            rr += 1.0 / rank
            for k in rec:
                if rank <= k:
                    rec[k] += 1

    def pct(x: int) -> float:
        return round(100.0 * x / n, 1) if n else 0.0

    return {
        "config": name,
        "n": n,
        "recall@1": pct(rec[1]),
        "recall@3": pct(rec[3]),
        "recall@5": pct(rec[5]),
        "recall@10": pct(rec[10]),
        "mrr": round(rr / n, 4) if n else 0.0,
        "lat_ms_mean": round(statistics.mean(lat), 1) if lat else 0.0,
        "lat_ms_p50": round(statistics.median(lat), 1) if lat else 0.0,
    }


def main() -> None:
    from app.memory.bm25_corpus import get_corpus_bm25
    from app.memory.reranking_retriever import RerankingRetriever
    from app.memory.sqlite_store import SqliteStore
    from app.research.rag_learning_loop import is_substantive_card

    limit = int(os.environ.get("RAG_AB_LIMIT", "0"))
    store = SqliteStore()
    seen: dict[str, dict] = {}
    for c in store.list_approved_cards():
        cj = c.get("card_json") or {}
        pid = str(c.get("paper_id") or "")
        if not pid or not is_substantive_card(cj):
            continue
        seen[pid] = cj
    items = sorted(seen.items())
    if limit:
        items = items[:limit]

    # BM25'i bir kez ısıt (build ~125s; sonraki retrieve'ler cache'li).
    t = time.perf_counter()
    _bm25, cm = get_corpus_bm25()
    print(
        f"# bm25 warm: {time.perf_counter() - t:.1f}s, chunks={len(cm)}, queries={len(items)}",
        flush=True,
    )

    configs = [
        ("dense_only", RerankingRetriever(enabled=False)),
        ("hybrid+rerank", RerankingRetriever(enabled=True, hybrid=True, rrf=False, graph=False)),
        ("rrf", RerankingRetriever(enabled=True, hybrid=True, rrf=True, graph=False)),
    ]
    results = []
    for name, retr in configs:
        r = _measure(name, retr, items)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    # karşılaştırma tablosu
    print("\n# config            recall@1 recall@3 recall@5 mrr    lat_p50ms", flush=True)
    for r in results:
        print(
            f"# {r['config']:<17} {r['recall@1']:>7} {r['recall@3']:>8} "
            f"{r['recall@5']:>8} {r['mrr']:>6} {r['lat_ms_p50']:>9}",
            flush=True,
        )


if __name__ == "__main__":
    main()
