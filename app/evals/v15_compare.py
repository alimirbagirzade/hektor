"""Ortak Ollama koşusunda ilk denemeleri kaydet; altyapı retry ve rescue'yu ayır."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.evals.llm30 import SYSTEM_PROMPT, answer_flags  # noqa: E402
from app.evals.v15_protocol import DECODING, sha256, verify_lock  # noqa: E402


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--set", choices=["critical", "development", "llm30", "final"], required=True
    )
    parser.add_argument("--checkpoint-lock", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports/v15/common")
    args = parser.parse_args()
    errors = verify_lock(ROOT / "reports/v15/protocol_v2/locks/lock.json")
    if errors:
        raise ValueError(errors)
    if args.set == "final":
        if not args.checkpoint_lock or not args.checkpoint_lock.is_file():
            raise ValueError("Final öncesi checkpoint kilidi gerekli")
        checkpoint = json.loads(args.checkpoint_lock.read_text(encoding="utf-8"))
        if checkpoint.get("selection_complete") is not True:
            raise ValueError("Model/config/checkpoint seçimi tamamlanmamış")
    source = (
        ROOT / "evals/llm30/validation.jsonl"
        if args.set in {"critical", "llm30"}
        else ROOT / f"evals/v15/protocol_v2/{args.set}.jsonl"
    )
    questions = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if args.set == "critical":
        ids = {f"llm30-s{n:02}" for n in (7, 9, 14, 20, 24, 28, 29)}
        questions = [q for q in questions if q["id"] in ids]
    out = args.output_root / args.set
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url="http://127.0.0.1:11434", timeout=1200) as client:
        metadata = {}
        for model in args.models:
            response = client.post("/api/show", json={"model": model})
            response.raise_for_status()
            metadata[model] = response.json()
        templates = {hash_text(v["template"]) for v in metadata.values()}
        if len(templates) != 1:
            raise ValueError("Chat template farklı; ortak koşu yapılmadı")
        manifest = {
            "started_at": datetime.now(UTC).isoformat(),
            "models": metadata,
            "ollama_version": client.get("/api/version").json(),
            "decoding": DECODING,
            "system_prompt": SYSTEM_PROMPT,
            "system_hash": hash_text(SYSTEM_PROMPT),
            "dataset_hash": sha256(source),
            "rag": False,
            "question_ids": [q["id"] for q in questions],
            "scoring": "pending_blind_review",
        }
        manifest_file = out / "manifest.json"
        if manifest_file.exists():
            raise ValueError("Koşu mevcut; üzerine yazılmaz")
        manifest_file.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (out / "raw.jsonl").open("x", encoding="utf-8") as raw:
            for model in args.models:
                for question in questions:
                    prompt = question["question"]
                    request = {
                        "model": model,
                        "stream": False,
                        "options": DECODING,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                    }
                    start = time.monotonic()
                    row = {
                        "version": model,
                        "question_id": question["id"],
                        "attempt": 1,
                        "started_at": datetime.now(UTC).isoformat(),
                        "settings": DECODING,
                        "prompt_hash": hash_text(
                            json.dumps(request["messages"], ensure_ascii=False)
                        ),
                        "rag": False,
                        "retry": False,
                        "rescue": False,
                        "score": None,
                    }
                    try:
                        result = client.post("/api/chat", json=request)
                        result.raise_for_status()
                        result = result.json()
                        answer = result["message"]["content"]
                        row.update(
                            raw_answer=answer,
                            answer_hash=hash_text(answer),
                            input_tokens=result.get("prompt_eval_count", 0),
                            output_tokens=result.get("eval_count", 0),
                            finish_reason=result.get("done_reason", "unknown"),
                            ollama_response=result,
                        )
                        row["flags"] = answer_flags(
                            answer,
                            done_reason=row["finish_reason"],
                            prompt_tokens=row["input_tokens"],
                            output_tokens=row["output_tokens"],
                            num_ctx=16384,
                        )
                    except (httpx.HTTPError, KeyError, ValueError) as exc:
                        row.update(
                            error=str(exc),
                            finish_reason="infrastructure_error",
                            flags=["infrastructure_error"],
                            raw_answer="",
                        )
                    row["time_s"] = round(time.monotonic() - start, 3)
                    raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw.flush()
                    print(
                        f"{model} {question['id']} {row['finish_reason']} {row['time_s']}s",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
