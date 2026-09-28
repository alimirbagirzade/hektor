"""Gerçek RAG hattında (RerankingRetriever + rag_answer prompt'u) belirli Ollama modelleriyle
soru cevaplat ve sonucu JSON'a yaz. Adapter'lı modeli (ör. hektor-v10) base ile kıyaslamak ve
cevabın HANGİ modelden geldiğini kanıtlamak için: her cevapla birlikte Ollama'nın döndürdüğü
model adı + digest kaydedilir.

EĞİTİM BAŞLATMAZ. Yatırım tavsiyesi değildir; cevaplar hipotez + test noktasıdır (Kural 1).
Determinizm: temperature=0.2, seed=42 (RagAnswerer varsayılanı, Kural 6).

Kullanım:
    uv run python scripts/rag_model_run.py --models hektor-v10 qwen3:4b-instruct-2507-q4_K_M \
        --questions sorular.txt --out reports/evals/rag_run.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.brain.local_llm import LocalLLM
from app.brain.rag_answerer import RagAnswerer
from app.config import get_settings


def _model_info(host: str, name: str) -> dict[str, Any]:
    """Ollama /api/show + /api/tags → aile, parametre, kuantizasyon, digest, ADAPTER satırı."""
    info: dict[str, Any] = {"name": name}
    try:
        show = httpx.post(f"{host}/api/show", json={"model": name}, timeout=30).json()
        info["details"] = show.get("details", {})
        info["adapter_lines"] = [
            ln for ln in str(show.get("modelfile", "")).splitlines() if ln.startswith("ADAPTER")
        ]
        info["from_lines"] = [
            ln for ln in str(show.get("modelfile", "")).splitlines() if ln.startswith("FROM")
        ]
        tags = httpx.get(f"{host}/api/tags", timeout=30).json().get("models", [])
        for m in tags:
            if name in (m.get("name"), m.get("model")) or f"{name}:latest" in (
                m.get("name"),
                m.get("model"),
            ):
                info["digest"] = m.get("digest", "")
                info["size"] = m.get("size", 0)
    except httpx.HTTPError as exc:
        info["error"] = str(exc)
    return info


def _loaded_models(host: str) -> list[str]:
    try:
        ps = httpx.get(f"{host}/api/ps", timeout=10).json().get("models", [])
        return [str(m.get("name", "")) for m in ps]
    except httpx.HTTPError:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Gerçek RAG hattında model(ler)le soru cevaplat")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--questions", type=Path, required=True, help="Satır başına bir soru")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    settings = get_settings()
    host = settings.ollama_host.rstrip("/")
    questions = [
        q.strip() for q in args.questions.read_text(encoding="utf-8").splitlines() if q.strip()
    ]
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "llm_default_max_tokens": settings.llm_default_max_tokens,
        "models": {m: _model_info(host, m) for m in args.models},
        "runs": [],
    }
    for model in args.models:
        answerer = RagAnswerer(llm=LocalLLM(model=model))
        for i, q in enumerate(questions, 1):
            t0 = time.perf_counter()
            ans = answerer.answer(q, top_k=args.top_k)
            dt = time.perf_counter() - t0
            loaded = _loaded_models(host)
            result["runs"].append(
                {
                    "model": model,
                    "i": i,
                    "question": q,
                    "answer": ans.answer,
                    "llm_used": ans.llm_used,
                    "seconds": round(dt, 1),
                    "ollama_loaded_after": loaded,
                    "sources": [
                        {"citation": c.citation, "title": c.title or ""} for c in ans.sources
                    ],
                }
            )
            print(f"[{model}] {i}/{len(questions)} {dt:.0f}s llm_used={ans.llm_used}", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Yazıldı: {args.out}")


if __name__ == "__main__":
    main()
