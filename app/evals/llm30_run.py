"""LLM-30 kontrollü 2×2 koşusu (protokol Aşama 2) — base/LoRA × RAG kapalı/açık.

Koşullar (PROTOKOL harfleri; ``hektor mix eval`` harfleriyle karıştırma):
    A = base,        RAG kapalı      B = base + LoRA, RAG kapalı
    C = base,        RAG açık        D = base + LoRA, RAG açık

Eşitlik: dört koşulda aynı sistem istemi (``app.evals.llm30.SYSTEM_PROMPT``), aynı Ollama
çalışma zamanı, aynı chat şablonu, aynı nicemleme tarifi (base de ``adapter_to_ollama.ps1
-BaseRepo`` ile LoRA ile AYNI tarifle GGUF'a çevrilmeli), aynı num_ctx/num_predict/decoding.
Retrieval soru başına BİR KEZ yapılır ve ``contexts.jsonl``'e dondurulur; C ve D byte-aynı
bağlamı aynı sırayla alır. Manifest iki modelin şablon hash'ini karşılaştırır — farklıysa
koşu BAŞLAMAZ.

Çıktı ``reports/evals/llm30/<run_id>/``: ``manifest.json`` · ``contexts.jsonl`` ·
``raw.jsonl`` (ham cevap; temizlenmez) · ``summary.json``. Puan alanları (``score_by_dimension``,
``critical_errors``) BOŞ yazılır: rubrikle insan/hakem puanlar, anahtar kelimeyle değil.
Kesintide aynı ``--run-id`` ile sürdürülür (tamamlanan kayıtlar atlanır).

EĞİTİM BAŞLATMAZ; yalnız yerel Ollama. Yatırım tavsiyesi değildir (Kural 1).

    uv run python -m app.evals.llm30_run --base hektor-base-30b-q4a8 --lora hektor-v13-30b \
        --adapter hektor_lora_v13_30b --seeds 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.brain.answer_quality import reorder_lost_in_middle
from app.brain.rag_answerer import _format_context
from app.config import get_settings
from app.evals.llm30 import SYSTEM_PROMPT, answer_flags, llm30_root, load_llm30, user_prompt
from app.evals.profile.dataset_loader import dataset_version
from app.evals.profile.schema import Split
from app.memory.rag_version import current_rag_snapshot

CONDITIONS: dict[str, tuple[str, bool]] = {  # harf → (model rolü, rag)
    "A": ("base", False),
    "B": ("lora", False),
    "C": ("base", True),
    "D": ("lora", True),
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_info(host: str, name: str) -> dict[str, Any]:
    show = httpx.post(f"{host}/api/show", json={"model": name}, timeout=60).json()
    if "error" in show:
        raise SystemExit(f"Ollama modeli yok: {name} ({show['error']})")
    tags = httpx.get(f"{host}/api/tags", timeout=30).json().get("models", [])
    digest = next(
        (
            m.get("digest", "")
            for m in tags
            if name in (m.get("name"), m.get("model"))
            or f"{name}:latest" in (m.get("name"), m.get("model"))
        ),
        "",
    )
    modelfile = str(show.get("modelfile", ""))
    return {
        "name": name,
        "digest": digest,
        "details": show.get("details", {}),
        "template_sha256": sha(str(show.get("template", ""))),
        "from_line": next((ln for ln in modelfile.splitlines() if ln.startswith("FROM")), ""),
        "header": modelfile.splitlines()[0] if modelfile else "",
        "parameters": show.get("parameters", ""),
    }


def chat_stream(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Streaming /api/chat: üretim ortasında iptal (ör. Ollama "token repeat limit") ham
    kısmi cevabıyla SONUÇ olarak döner (``done_reason = "iptal:..."``); koşu durmaz.
    Üretim başlamadan gelen hata (model yok vb.) ölümcüldür."""
    parts: list[str] = []
    out: dict[str, Any] = {"prompt_eval_count": 0, "eval_count": 0, "done_reason": ""}
    with client.stream("POST", url, json={**payload, "stream": True}) as resp:
        for line in resp.iter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if "error" in event:
                err = str(event["error"])
                if not parts:
                    raise SystemExit(f"Ollama hatası ({payload['model']}): {err}")
                kind = "tekrar_limiti" if "repeat" in err.lower() else err[:60]
                out["done_reason"] = f"iptal:{kind}"
                out["eval_count"] = len(parts)  # akış parçası ≈ token (sayaç gelmedi)
                break
            parts.append(str(event.get("message", {}).get("content", "")))
            if event.get("done"):
                out["prompt_eval_count"] = int(event.get("prompt_eval_count") or 0)
                out["eval_count"] = int(event.get("eval_count") or 0)
                out["done_reason"] = str(event.get("done_reason", ""))
    out["content"] = "".join(parts)
    return out


def freeze_contexts(items: list[Any], path: Path, top_k: int | None) -> dict[str, dict[str, Any]]:
    """Soru başına tek retrieval → dondurulmuş bağlam (varsa dosyadan okunur, yeniden YAPILMAZ)."""
    if path.exists():
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln]
        return {r["question_id"]: r for r in rows}
    from app.memory.reranking_retriever import RerankingRetriever

    settings = get_settings()
    retriever = RerankingRetriever()
    out: dict[str, dict[str, Any]] = {}
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            chunks = retriever.retrieve(it.question, top_k=top_k)
            ordered = reorder_lost_in_middle(chunks) if settings.rag_reorder_context else chunks
            context = _format_context(ordered) if ordered else ""
            row = {
                "question_id": it.id,
                "context": context,
                "context_sha256": sha(context),
                "source_ids": [c.citation for c in ordered],
                "paper_ids": [c.paper_id for c in ordered],
                "distances": [c.distance for c in ordered],
                "n_chunks": len(ordered),
            }
            out[it.id] = row
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[retrieval] {it.id} {len(ordered)} chunk", flush=True)
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-30 kontrollü 2×2 (base/LoRA × RAG)")
    ap.add_argument("--base", required=True, help="Ollama base modeli (LoRA ile aynı tarif)")
    ap.add_argument("--lora", required=True, help="Ollama birleşik LoRA modeli")
    ap.add_argument("--adapter", required=True, help="models/adapters altındaki adapter adı")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--questions", nargs="*", default=None, help="Alt küme (ör. llm30-s06)")
    ap.add_argument("--num-predict", type=int, default=4096)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None, help="retrieval top_k (varsayılan ayar)")
    ap.add_argument("--run-id", default=None, help="Sürdürmek için mevcut koşu kimliği")
    args = ap.parse_args()

    settings = get_settings()
    host = settings.ollama_host.rstrip("/")
    items = load_llm30(purpose="selection")
    if args.questions:
        items = [it for it in items if it.id in set(args.questions)]
    run_id = args.run_id or "llm30_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_dir = Path("reports/evals/llm30") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {"base": model_info(host, args.base), "lora": model_info(host, args.lora)}
    if models["base"]["template_sha256"] != models["lora"]["template_sha256"]:
        raise SystemExit("Base ve LoRA chat şablonları FARKLI — 2×2 eşit değil, koşu başlamadı.")
    b_det, l_det = models["base"]["details"], models["lora"]["details"]
    quant_equal = b_det.get("quantization_level") == l_det.get("quantization_level")

    adapter_dir = Path("models/adapters") / args.adapter
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    decoding = {
        "temperature": args.temperature,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
        "num_predict": args.num_predict,
        "num_ctx": args.num_ctx,
    }
    snapshot = current_rag_snapshot(top_k=args.top_k)
    contexts = freeze_contexts(items, out_dir / "contexts.jsonl", args.top_k)

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "protocol": "docs/PROTOKOL_LORA_RAG_IYILESTIRME.md Aşama 2",
            "conditions": {
                k: {"model_role": r, "rag_enabled": g} for k, (r, g) in CONDITIONS.items()
            },
            "split": "validation (gelistirme; final test DEGIL)",
            "dataset_version": dataset_version(Split.VALIDATION, llm30_root()),
            "models": models,
            "quantization_equal": quant_equal,
            "runtime": "ollama",
            "adapter": {
                "id": args.adapter,
                "adapter_sha256": file_sha(adapter_dir / "adapter_model.safetensors"),
                "base_model": adapter_cfg.get("base_model_name_or_path"),
                "r": adapter_cfg.get("r"),
                "lora_alpha": adapter_cfg.get("lora_alpha"),
                "target_modules": adapter_cfg.get("target_modules"),
            },
            "system_prompt_sha256": sha(SYSTEM_PROMPT),
            "system_prompt": SYSTEM_PROMPT,
            "decoding": decoding,
            "seeds": args.seeds,
            "rag": snapshot.to_dict(),
            "contexts_sha256": file_sha(out_dir / "contexts.jsonl"),
            "git_commit": git_commit(),
            "notes": [
                "score_by_dimension / critical_errors bos: rubrikle insan/hakem puanlar.",
                "Ayni seed farkli runtime'da birebir determinizm garanti etmez.",
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        # Sürdürme (Kademe 2 F4-1): manifest eskiden HİÇ karşılaştırılmıyordu → aynı adla
        # yeniden kurulmuş model / farklı adapter / farklı çözme ayarı / farklı RAG eski
        # satırlarla sessizce karışırdı. Kimlik alanları birebir eşleşmeli.
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        now = {
            "base_digest": models["base"]["digest"],
            "lora_digest": models["lora"]["digest"],
            "adapter_sha256": file_sha(adapter_dir / "adapter_model.safetensors"),
            "decoding": decoding,
            "seeds": args.seeds,
            "rag_version": snapshot.rag_version,
            "system_prompt_sha256": sha(SYSTEM_PROMPT),
        }
        was = {
            "base_digest": old["models"]["base"]["digest"],
            "lora_digest": old["models"]["lora"]["digest"],
            "adapter_sha256": old["adapter"]["adapter_sha256"],
            "decoding": old["decoding"],
            "seeds": old["seeds"],
            "rag_version": old["rag"].get("rag_version"),
            "system_prompt_sha256": old["system_prompt_sha256"],
        }
        diff = [k for k in now if now[k] != was[k]]
        if diff:
            raise SystemExit(
                f"{run_id} sürdürülemez: manifest ile şimdiki koşu farklı ({', '.join(diff)}). "
                "Yeni --run-id ile başlat."
            )

    raw_path = out_dir / "raw.jsonl"
    done: set[tuple[str, str, int]] = set()
    if raw_path.exists():
        for ln in raw_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                done.add((r["condition"], r["question_id"], r["seed"]))

    # Model takasını azaltmak için model rolüne göre grupla: A,C (base) sonra B,D (lora).
    order = sorted(args.conditions, key=lambda c: (CONDITIONS[c][0] != "base", c))
    total = len(order) * len(items) * len(args.seeds)
    n = len(done)
    with raw_path.open("a", encoding="utf-8") as fh, httpx.Client(timeout=3600) as client:
        for cond in order:
            role, rag = CONDITIONS[cond]
            model = models[role]
            for it in items:
                ctx = contexts[it.id] if rag else None
                prompt = user_prompt(it.question, ctx["context"] if ctx else None)
                for seed in args.seeds:
                    if (cond, it.id, seed) in done:
                        continue
                    t0 = time.perf_counter()
                    resp = chat_stream(
                        client,
                        f"{host}/api/chat",
                        {
                            "model": model["name"],
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            "options": {**decoding, "seed": seed},
                        },
                    )
                    latency = time.perf_counter() - t0
                    answer = resp["content"]
                    p_tok = resp["prompt_eval_count"]
                    o_tok = resp["eval_count"]
                    reason = resp["done_reason"]
                    rec = {
                        "run_id": run_id,
                        "condition": cond,
                        "question_id": it.id,
                        "split": "validation",
                        "base_id": models["base"]["name"],
                        "base_hash": models["base"]["digest"],
                        "model": model["name"],
                        "model_digest": model["digest"],
                        "adapter_id": args.adapter if role == "lora" else None,
                        "template_sha256": model["template_sha256"],
                        "runtime": "ollama",
                        "quantization": model["details"].get("quantization_level"),
                        "rag_enabled": rag,
                        "prompt_sha256": sha(SYSTEM_PROMPT + "\x00" + prompt),
                        "context_sha256": ctx["context_sha256"] if ctx else None,
                        "source_ids": ctx["source_ids"] if ctx else [],
                        "decoding": decoding,
                        "seed": seed,
                        "input_tokens": p_tok,
                        "output_tokens": o_tok,
                        "latency_s": round(latency, 2),
                        "finish_reason": reason,
                        "flags": answer_flags(
                            answer,
                            done_reason=reason,
                            prompt_tokens=p_tok,
                            output_tokens=o_tok,
                            num_ctx=args.num_ctx,
                        ),
                        "raw_answer": answer,
                        "score_by_dimension": None,
                        "critical_errors": None,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    n += 1
                    print(
                        f"[{n}/{total}] {cond} {it.id} seed={seed} {latency:.0f}s "
                        f"out={o_tok} {reason} {rec['flags'] or ''}",
                        flush=True,
                    )

    rows = [json.loads(ln) for ln in raw_path.read_text(encoding="utf-8").splitlines() if ln]
    summary: dict[str, Any] = {}
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        # Bayraklar HAM cevaptan güncel kuralla yeniden hesaplanır (raw.jsonl'deki ``flags``
        # üretim anındaki kural sürümüdür; ham cevap değişmez).
        flag_counts: dict[str, int] = {}
        for r in rs:
            current = answer_flags(
                r["raw_answer"],
                done_reason=r["finish_reason"],
                prompt_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                num_ctx=r["decoding"]["num_ctx"],
            )
            for f in current:
                flag_counts[f] = flag_counts.get(f, 0) + 1
        summary[cond] = {
            "n": len(rs),
            "model": rs[0]["model"],
            "rag_enabled": rs[0]["rag_enabled"],
            "ort_cikti_token": round(sum(r["output_tokens"] for r in rs) / len(rs)),
            "ort_sure_s": round(sum(r["latency_s"] for r in rs) / len(rs), 1),
            "bayraklar": flag_counts,
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Bitti: {out_dir}")


if __name__ == "__main__":
    main()
