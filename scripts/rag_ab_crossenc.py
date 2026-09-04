"""Cross-encoder A/B — dense-only vs dense+cross-encoder (gerçek reranker).

Sezgisel reranker self-retrieval recall'ı DÜŞÜRMÜŞTÜ (bkz. rag_ab_multi). Bu araç
GERÇEK cross-encoder (bge-reranker-base) dense adayları yeniden sıralayınca recall@1
artıyor mu ölçer. Hibrit/BM25 KULLANILMAZ (o yol yavaş+faydasız çıktı) → cross-encoder
yalnız dense over-fetch havuzunu yeniden sıralar. Maliyet: model + CPU latency.

Kullanım: RAG_AB_LIMIT=40 uv run --no-sync python scripts/rag_ab_crossenc.py
"""

from __future__ import annotations

import json
import os
import statistics
import time


def _card_query(cj: dict) -> str:
    parts = [str(cj.get("title") or ""), str(cj.get("main_claim") or "")]
    if cj.get("domain"):
        parts.append(str(cj.get("domain")))
    return ". ".join(p for p in parts if p).strip()[:400]


def _measure(name: str, retriever, items) -> dict:
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
        rank = next((i for i, ch in enumerate(chunks, 1) if ch.paper_id == pid), 0)
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
        "mrr": round(rr / n, 4) if n else 0.0,
        "lat_ms_p50": round(statistics.median(lat), 1) if lat else 0.0,
    }


def main() -> None:
    from app.memory.cross_encoder_reranker import CrossEncoderReranker
    from app.memory.reranking_retriever import RerankingRetriever
    from app.memory.sqlite_store import SqliteStore
    from app.research.rag_learning_loop import is_substantive_card

    limit = int(os.environ.get("RAG_AB_LIMIT", "0"))
    store = SqliteStore()
    seen: dict[str, dict] = {}
    for c in store.list_approved_cards():
        cj = c.get("card_json") or {}
        pid = str(c.get("paper_id") or "")
        if pid and is_substantive_card(cj):
            seen[pid] = cj
    items = sorted(seen.items())
    if limit:
        items = items[:limit]
    print(f"# queries={len(items)}", flush=True)

    # cross-encoder modelini bir kez yükle (ilk kullanımda indirir)
    t = time.perf_counter()
    ce = CrossEncoderReranker()
    # ısıt: küçük bir rerank ile modeli yükle
    print(f"# cross-encoder init: {time.perf_counter() - t:.1f}s", flush=True)

    configs = [
        ("dense_only", RerankingRetriever(enabled=False)),
        (
            "dense+cross_encoder",
            RerankingRetriever(enabled=True, hybrid=False, rrf=False, graph=False, reranker=ce),
        ),
    ]
    results = []
    for name, retr in configs:
        r = _measure(name, retr, items)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    print("\n# config               recall@1 recall@3 recall@5 mrr    lat_p50ms", flush=True)
    for r in results:
        print(
            f"# {r['config']:<20} {r['recall@1']:>7} {r['recall@3']:>8} "
            f"{r['recall@5']:>8} {r['mrr']:>6} {r['lat_ms_p50']:>9}",
            flush=True,
        )


if __name__ == "__main__":
    main()
