"""RAG retrieval A/B ölçüm aracı — korpusa-bağlı, deterministik, LLM-judge'sız.

Her bilgi kartından (paper_id bilinir) yapısal bir sorgu üretir (başlık+ana iddia),
CANLI retrieval yolunu (`RerankingRetriever`, ayarlardan okur) çalıştırır ve
**makale-düzeyi self-retrieval** sinyalini ölçer: kartın sorgusu o kartın makalesini
geri getiriyor mu, hangi sırada?

Ölçülenler: recall@1/3/5/10 (makale-düzeyi), MRR ve sorgu başına gecikme (ms).
recall@10 doysa bile recall@1/MRR reranker farkını ayırt eder; gecikme "hız" sinyalidir.

Config tamamen ortam değişkenleriyle kontrol edilir (ACHILLES_RAG_RRF,
ACHILLES_RAG_CROSS_ENCODER, ACHILLES_RAG_CONTEXTUAL_EMBED ...) → her A/B koşusu AYRI
süreçte çalıştırılır (get_settings cache'i bayatlamaz). Çıktı tek satır JSON.

Kullanım:
    uv run --no-sync python scripts/rag_retrieval_ab.py            # tüm kartlar
    RAG_AB_LIMIT=60 uv run --no-sync python scripts/rag_retrieval_ab.py
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


def main() -> None:
    from app.config import get_settings
    from app.memory.reranking_retriever import RerankingRetriever
    from app.memory.sqlite_store import SqliteStore
    from app.research.rag_learning_loop import is_substantive_card

    limit = int(os.environ.get("RAG_AB_LIMIT", "0"))  # 0 = tümü
    store = SqliteStore()
    seen: dict[str, dict] = {}
    for c in store.list_approved_cards():
        cj = c.get("card_json") or {}
        pid = str(c.get("paper_id") or "")
        if not pid or not is_substantive_card(cj):
            continue
        seen[pid] = cj  # makale başına en yeni kart
    items = sorted(seen.items())  # determinizm
    if limit:
        items = items[:limit]

    s = get_settings()
    retr = RerankingRetriever()
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
            chunks = retr.retrieve(q, top_k=10)
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

    out = {
        "config": {
            "rrf": s.rag_rrf,
            "cross_encoder": s.rag_cross_encoder,
            "graph": s.rag_graph,
            "contextual": s.rag_contextual_embed,
            "hybrid": s.rag_hybrid,
            "rerank": s.rag_rerank,
            "top_k": s.rag_top_k,
            "overfetch": s.rag_overfetch,
        },
        "n_queries": n,
        "recall@1": pct(rec[1]),
        "recall@3": pct(rec[3]),
        "recall@5": pct(rec[5]),
        "recall@10": pct(rec[10]),
        "mrr": round(rr / n, 4) if n else 0.0,
        "latency_ms_mean": round(statistics.mean(lat), 1) if lat else 0.0,
        "latency_ms_p50": round(statistics.median(lat), 1) if lat else 0.0,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
