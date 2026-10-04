# ruff: noqa: E501
"""Kilitli doğrulanmış veriyle ayrı merge-aware v15 eğitimi; varsayılan dry-run."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evals.v15_protocol import sha256, split_errors, verify_lock, write_lock  # noqa: E402
from app.training.peft_lora_train import PeftTrainConfig, dry_run, train  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    report = ROOT / "reports/v15/continuation_r4"
    report.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data/training/v15_repair_r3"
    output = ROOT / "models/adapters/hektor_lora_v15_mergeaware_r4"
    for lock in (
        ROOT / "reports/v15/protocol_v2/locks/lock.json",
        ROOT / "reports/v15/repair_r3/locks/lock.json",
    ):
        if errors := verify_lock(lock):
            raise ValueError(errors)
    splits = {
        name: [
            json.loads(s) for s in (data / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for name in ("train", "valid")
    }
    for name in ("development",):
        splits[name] = [
            json.loads(s)
            for s in (ROOT / f"evals/v15/protocol_v2/{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
    if errors := split_errors(splits):
        raise ValueError(errors)
    evidence_path = ROOT / "reports/v15/repair_r3/code_oracle_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence["passed"] != 54 or len(splits["train"]) != 73 or len(splits["valid"]) != 4:
        raise ValueError("Beklenen veri/çalıştırılmış oracle kapsamı eksik")
    base = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
    )
    cfg = PeftTrainConfig(
        base_model=str(base),
        train_jsonl=data / "train.jsonl",
        valid_jsonl=data / "valid.jsonl",
        adapter_output_path=output,
        iterations=240,  # GA8 ile 30 optimizer bütçesi; gerçek örnek sayısı 219.
        batch_size=1,
        learning_rate=0.00005,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0,
        merge_aware_bf16=True,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        max_seq_length=6144,
        assistant_only_loss=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        neftune_noise_alpha=0,
        warmup_ratio=0.05,
        eval_every_examples=32,
        eval_max_examples=4,
        load_best_model_at_end=False,
        loss_weighting="example",
        seed=42,
    )
    plan = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
    plan.update(
        experiment="separate_local_revision_mergeaware_continuation_r4",
        intended_full_epochs=3,
        intended_optimizer_steps=30,
        intended_processed_examples=219,
        iteration_field_semantics="240 is conversion budget for 30 optimizer steps, NOT 240 processed examples; HF flushes 73-row epoch remainder",
        previous_candidate="hektor_lora_v15_mergeaware_r2 rejected on critical development answers",
        initialization="fresh adapter on same immutable base; not optimizer resume or previous adapter continuation",
        scheduler_change="30-step cosine schedule and warmup derived from new total; last checkpoint fixed",
        checkpoint_selection="last_checkpoint_fixed_before_training",
        started_at=datetime.now(UTC).isoformat(),
        authorization="user_repair_and_retrain_local_revision",
        rationale="Same locked 73 examples, three full epochs, not new facts; first candidate had only one epoch. Longer exposure is a hypothesis, not improved answer quality proof; final content never read by this runner",
        limitations="CPU bounded candidate, not sufficient training or quality acceptance claim; v14 exact base revision unknown; eval shares learned concepts but numeric code templates source groups remain train-only",
    )
    plan_path = report / "training_config.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = report / "locks/lock.json"
    if not lock.exists():
        write_lock(
            report / "locks",
            [
                data / "train.jsonl",
                data / "valid.jsonl",
                plan_path,
                evidence_path,
                Path(__file__),
                ROOT / "app/training/merge_aware_lora.py",
                ROOT / "app/training/peft_lora_train.py",
            ],
        )
    if errors := verify_lock(lock):
        raise ValueError(errors)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HEKTOR_TRAIN_DTYPE"] = "bf16"
    if args.run:
        if output.exists():
            raise FileExistsError("Ayrı yeni adapter klasörü gereklidir")
        import numpy as np
        import torch

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
    result = train(cfg) if args.run else dry_run(cfg)
    if args.run and result.get("ok"):
        state = json.loads(
            (output / "checkpoint-30/trainer_state.json").read_text(encoding="utf-8")
        )
        if state["global_step"] != 30 or abs(state["epoch"] - 3.0) > 1e-9:
            result["ok"] = False
            result["verification_error"] = "Üç tam epoch / 30 adım gerçekleşmedi; kabul edilmez"
        (report / "adapter_hash.txt").write_text(
            sha256(output / "adapter_model.safetensors"), encoding="utf-8"
        )
    target = report / ("training_result.json" if args.run else "dry_run.json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(result, default=str), flush=True)
    return 0 if not args.run or result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
