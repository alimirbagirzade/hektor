"""Kaydedilmiş bf16 birleşik modeli yeniden yükleyip bütün referans konumları ölç."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.merge_adapter import distribution_metrics, gate_failures  # noqa: E402

from app.evals.v15_protocol import sha256  # noqa: E402


def main():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    data = load_file(str(args.reference))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    metrics = []
    with torch.no_grad():
        for i in range(3):
            after = model(input_ids=data[f"ids_{i}"]).logits[0].float()
            metrics.append(distribution_metrics(data[f"before_{i}"], after, data[f"base_{i}"]))
    failures = gate_failures(metrics)
    result = {
        "passed": not failures,
        "gate_failures": failures,
        "probes": metrics,
        "reference_sha256": sha256(args.reference),
        "model": str(args.model),
        "scope": "all_prompt_positions",
        "kl_gate": 0.01,
        "comparison": "trained_mergeaware_reference_vs_standard_reloaded_bf16_model",
    }
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
