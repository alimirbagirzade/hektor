"""RAG deneyi: sorgu çevirisi × amaç filtresi (protokol Aşama 4) — LLM-30 soruları.

Değişen unsur YALNIZ retrieval'dır (korpus/indeks/embedding/reranker sabit; ``index_hash``
manifestte). Varyantlar:

    V0 mevcut         çeviri off       filtre yok
    V1 filtre         çeviri off       proje_dokumani dışlanır
    V2 çeviri         çeviri en        filtre yok
    V3 çeviri+filtre  çeviri en        proje_dokumani dışlanır
    V4 iki dil+filtre bilingual        proje_dokumani dışlanır

İlgililik etiketli set olmadığından TREC tarzı HAVUZLAMA: varyantların top-k birleşimi
``pool.jsonl``'e yazılır; etiketler (0 alakasız / 1 kısmen / 2 doğrudan) ayrı dosyada
(``evals/rag_relevance/``) tutulur ve ``--labels`` ile puanlanır: P@k, nDCG@k, havuz-recall@k
(havuzdaki ilgililere göre — gerçek recall'un ÜST sınırı), iç-doküman payı.

    uv run python -m app.evals.llm30_rag_experiment run
    uv run python -m app.evals.llm30_rag_experiment score --run <dir> --labels <jsonl>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evals.llm30 import load_llm30
from app.memory.doc_purpose import excluded_paper_ids
from app.memory.rag_version import current_rag_snapshot

K = 6
VARIANTS: dict[str, dict[str, Any]] = {
    "V0": {"translate": "off", "exclude": frozenset()},
    "V1": {"translate": "off", "exclude": frozenset({"proje_dokumani"})},
    "V2": {"translate": "en", "exclude": frozenset()},
    "V3": {"translate": "en", "exclude": frozenset({"proje_dokumani"})},
    "V4": {"translate": "bilingual", "exclude": frozenset({"proje_dokumani"})},
}
INTERNAL = excluded_paper_ids(frozenset({"proje_dokumani"}))


def run(out_root: Path, top_k: int = K) -> Path:
    from app.config import get_settings
    from app.memory.query_translation import translate_query
    from app.memory.reranking_retriever import RerankingRetriever
    from app.memory.retrieval_service import RetrievalService

    settings = get_settings()
    items = load_llm30(purpose="audit")
    run_dir = out_root / ("rag_exp_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    # Çeviriler önce, tek model yüklemesiyle (önbelleğe yazılır; varyantlar oradan okur).
    translations = {
        it.id: translate_query(it.question, model=settings.rag_translate_model) for it in items
    }
    base = RetrievalService()
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, cfg in VARIANTS.items():
        rr = RerankingRetriever(
            base=base,
            exclude_purposes=cfg["exclude"],
            translate=cfg["translate"],
            translate_model=settings.rag_translate_model,
        )
        results[name] = {}
        for it in items:
            chunks = rr.retrieve(it.question, top_k=top_k)
            results[name][it.id] = [
                {
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "title": c.title,
                    "distance": c.distance,
                    "internal": c.paper_id in INTERNAL,
                }
                for c in chunks
            ]
        print(
            f"[{name}] {sum(r['internal'] for v in results[name].values() for r in v)} "
            f"iç-doküman / {len(items) * top_k}",
            flush=True,
        )

    pool: dict[tuple[str, str], dict[str, Any]] = {}
    texts = _chunk_texts({r["chunk_id"] for v in results.values() for q in v.values() for r in q})
    for name, per_q in results.items():
        for qid, rows in per_q.items():
            for r in rows:
                entry = pool.setdefault(
                    (qid, r["chunk_id"]),
                    {
                        "question_id": qid,
                        "chunk_id": r["chunk_id"],
                        "paper_id": r["paper_id"],
                        "title": r["title"],
                        "text": texts.get(r["chunk_id"], ""),
                        "variants": [],
                    },
                )
                entry["variants"].append(name)
    with (run_dir / "pool.jsonl").open("w", encoding="utf-8") as fh:
        for entry in sorted(pool.values(), key=lambda e: (e["question_id"], e["chunk_id"])):
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    snapshot = current_rag_snapshot(top_k=top_k)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": "docs/PROTOKOL_LORA_RAG_IYILESTIRME.md Aşama 4 (değişen: retrieval)",
        "variants": {
            k: {"translate": v["translate"], "exclude": sorted(v["exclude"])}
            for k, v in VARIANTS.items()
        },
        "translate_model": settings.rag_translate_model,
        "base_rag": snapshot.to_dict(),
        "top_k": top_k,
        "translations": translations,
        "pool_size": len(pool),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (run_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"Havuz: {len(pool)} (soru, parça) → {run_dir / 'pool.jsonl'}")
    return run_dir


def _chunk_texts(chunk_ids: set[str]) -> dict[str, str]:
    import sqlite3

    from app.config import get_settings

    db = get_settings().root / "storage" / "sqlite" / "hektor_trader_ai.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, str] = {}
    ids = sorted(chunk_ids)
    for i in range(0, len(ids), 500):
        part = ids[i : i + 500]
        marks = ",".join("?" * len(part))
        for cid, text in con.execute(
            f"select chunk_id, text from chunks where chunk_id in ({marks})", part
        ):
            out[cid] = text or ""
    return out


def _dcg(rels: list[int]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def score(run_dir: Path, labels_path: Path, top_k: int = K) -> dict[str, Any]:
    """Havuz etiketleriyle varyant başına P@k, nDCG@k, havuz-recall@k, iç-doküman payı."""
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    labels: dict[tuple[str, str], int] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[(row["question_id"], row["chunk_id"])] = int(row["relevance"])
    relevant_by_q: dict[str, int] = {}
    ideal_by_q: dict[str, list[int]] = {}
    for (qid, _cid), rel in labels.items():
        relevant_by_q[qid] = relevant_by_q.get(qid, 0) + (rel >= 1)
        ideal_by_q.setdefault(qid, []).append(rel)
    report: dict[str, Any] = {"labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest()}
    for name, per_q in results.items():
        p, nd, rc, internal, missing, zero = [], [], [], 0, 0, 0
        for qid, rows in per_q.items():
            rels = []
            for r in rows[:top_k]:
                key = (qid, r["chunk_id"])
                if key not in labels:
                    missing += 1
                rels.append(labels.get(key, 0))
                internal += bool(r["internal"])
            p.append(sum(x >= 1 for x in rels) / top_k)
            ideal = sorted(ideal_by_q.get(qid, []), reverse=True)[:top_k]
            nd.append(_dcg(rels) / _dcg(ideal) if ideal and _dcg(ideal) > 0 else 0.0)
            total_rel = relevant_by_q.get(qid, 0)
            rc.append(sum(x >= 1 for x in rels) / total_rel if total_rel else 0.0)
            zero += sum(x >= 1 for x in rels) == 0
        n = len(per_q)
        report[name] = {
            "P@k": round(sum(p) / n, 3),
            "nDCG@k": round(sum(nd) / n, 3),
            "havuz_recall@k": round(sum(rc) / n, 3),
            "hic_ilgili_yok_soru": zero,
            "ic_dokuman": f"{internal}/{n * top_k}",
            "etiketsiz": missing,
        }
    (run_dir / "score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-30 RAG deneyi (çeviri × amaç filtresi)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, default=Path("reports/evals/llm30"))
    s = sub.add_parser("score")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--labels", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.out)
    else:
        print(json.dumps(score(args.run, args.labels), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
