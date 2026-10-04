"""v15 teslim raporu: yalnız diskte bulunan çalıştırma kanıtlarından durum üret."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evals.v15_protocol import decision, sha256  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    out = ROOT / "reports/v15"
    adapter = ROOT / "models/adapters/hektor_lora_v15_localrev_pilot"
    training = read(out / "pilot_v4/training_result.json")
    complete = read(adapter / "run_complete.json")
    merge = read(ROOT / "models/merged/hektor_lora_v15_localrev_pilot/merge_info.json")
    merge_attempt = read(out / "evidence/merge_attempt.json")
    evidence = {
        "training_complete": training.get("ok") is True and complete.get("global_step") == 3,
        "transfer_verified": bool(merge) and merge.get("kl_gate") == 0.01,
        "transfer_gate_failed": merge_attempt.get("passed") is False,
        "final_locked": (out / "protocol_v2/locks/lock.json").is_file(),
        "checkpoint_locked": (out / "checkpoint_lock.json").is_file(),
        "final_complete": False,
        "blind_review_complete": False,
        "paired_v14_complete": False,
        "critical_model_fixtures_pass": False,
        "repetition_improved": False,
        "first_attempt_only": True,
        "merge_max_kl": merge.get("kl_peft_vs_merged", merge_attempt.get("max_kl")),
        "merge_scope": merge.get("measurement_scope", merge_attempt.get("measurement_scope")),
        "merge_kl_direction": merge.get("kl_direction", merge_attempt.get("kl_direction")),
    }
    verdict, reasons = decision(evidence)
    (out / "decision.json").write_text(
        json.dumps(
            {"decision": verdict, "reasons": reasons, "evidence": evidence},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    files = {}
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer.json",
        "run_complete.json",
    ):
        p = adapter / name
        if p.is_file():
            files[name] = {"path": str(p), "sha256": sha256(p)}
    (out / "model_card.json").write_text(
        json.dumps(
            {
                "model": "hektor_lora_v15_localrev_pilot",
                "status": verdict,
                "experiment": "local_revision_separate_experiment",
                "base": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "base_commit": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
                "v14_same_base_revision_proven": False,
                "training_result": training,
                "run_complete": complete,
                "artifacts": files,
                "transfer_test": merge_attempt,
                "limitations": [
                    "19 açıklama/4 validation ile üç optimizer adımlık pipeline pilotu",
                    "Konular geliştirmede görülmüştür; trading başarısı kanıtı değildir",
                    "Sabit son checkpoint; davranış ölçütleriyle tam aday seçimi tamamlanmadı",
                    "Final holdout ve kör alt madde puanları olmadan kabul verilmez",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    fields = [
        "version",
        "set",
        "question_id",
        "subitem",
        "raw_path",
        "raw_hash",
        "score",
        "critical_error",
        "contradiction",
        "code_failure",
        "repeat",
        "unmeasured_claim",
        "verification_evidence",
        "input_tokens",
        "output_tokens",
        "time_s",
        "finish_reason",
        "settings",
    ]
    counts = {}
    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for p in sorted((out / "common").glob("*/raw.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                key = f"{p.parent.name}/{row['version']}"
                counts[key] = counts.get(key, 0) + 1
                for subitem in ("core", "assumptions_evidence"):
                    writer.writerow(
                        {
                            "version": row["version"],
                            "set": p.parent.name,
                            "question_id": row["question_id"],
                            "subitem": subitem,
                            "raw_path": str(p),
                            "raw_hash": sha256(p),
                            "score": "",
                            "repeat": any("tekrar" in f for f in row["flags"]),
                            "verification_evidence": "pending_blind_scoring_no_code_execution",
                            "input_tokens": row.get("input_tokens"),
                            "output_tokens": row.get("output_tokens"),
                            "time_s": row["time_s"],
                            "finish_reason": row["finish_reason"],
                            "settings": json.dumps(row["settings"]),
                        }
                    )
    with (out / "ablation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "model", "rag", "status", "reason"])
        for condition, model, rag in (
            ("A", "base", False),
            ("B", "v15", False),
            ("C", "base", True),
            ("D", "v15", True),
        ):
            writer.writerow(
                [
                    condition,
                    model,
                    rag,
                    "not_completed",
                    "v15_transfer_gate_failed_ablation_not_started",
                ]
            )
    (out / "run_coverage.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": verdict,
                "training_complete": evidence["training_complete"],
                "common_counts": counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
