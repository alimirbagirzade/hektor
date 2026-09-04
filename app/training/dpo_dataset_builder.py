"""dpo_dataset_builder.py — Reward sinyallerinden DPO eğitim verisi üretir.

Akış:
  1. tool_use_examples → session başına metin zinciri
  2. reward_signals'tan chosen / rejected etiketlerini çek
  3. chosen–rejected çiftlerini eşleştir (min_gap filtresiyle)
  4. {prompt, chosen, rejected} üçlüsü olarak JSONL'a yaz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.memory.sqlite_store import SqliteStore
from app.training.reward_signal import RewardCriteria, build_preference_pairs, compute_reward


def _session_text(steps: list[dict]) -> str:
    parts = []
    for s in sorted(steps, key=lambda x: x["step_index"]):
        tag = s["step_type"].upper()
        parts.append(f"[{tag}] {s['content']}")
    return "\n".join(parts)


def _to_rc(sig: dict) -> RewardCriteria:
    return RewardCriteria(
        execution_ok=sig["execution_ok"],
        trade_count_ok=sig["trade_count_ok"],
        sharpe_ok=sig["sharpe_ok"],
        drawdown_ok=sig["drawdown_ok"],
        return_ok=sig["return_ok"],
        win_rate_ok=sig["win_rate_ok"],
    )


def score_and_save_sessions(store: SqliteStore | None = None) -> list[dict[str, Any]]:
    """Skorsuz tool-use seanslarını değerlendir ve reward_signals'a kaydet."""
    s = store or SqliteStore()
    rows = s.list_tool_use_examples(limit=5000)

    sessions: dict[str, list[dict]] = {}
    for r in rows:
        sessions.setdefault(r["session_id"], []).append(r)

    results = []
    for session_id, steps in sessions.items():
        if s.get_reward_signal(session_id):
            continue
        call_steps = [st for st in steps if st["step_type"] == "call"]
        if not call_steps:
            continue
        last_call = call_steps[-1]
        metrics = last_call.get("tool_output", {}).get("metrics", {})
        verdict = last_call.get("verdict") or "fail"
        had_error = bool(last_call.get("tool_output", {}).get("error"))
        rc = compute_reward(metrics, verdict=verdict, had_error=had_error)
        s.save_reward_signal(session_id, rc, raw_metrics=metrics)
        results.append({"session_id": session_id, "composite": rc.composite, "label": rc.label})

    return results


def build_dpo_dataset(
    store: SqliteStore | None = None,
    output_path: Path | None = None,
    min_gap: float = 0.25,
) -> list[dict[str, Any]]:
    """`reward_signals` tablosundan DPO {prompt, chosen, rejected} çiftleri üret."""
    s = store or SqliteStore()
    chosen_sigs = s.list_reward_signals(label="chosen")
    rejected_sigs = s.list_reward_signals(label="rejected")
    if not chosen_sigs or not rejected_sigs:
        return []

    all_rows = s.list_tool_use_examples(limit=5000)
    session_steps: dict[str, list[dict]] = {}
    for r in all_rows:
        session_steps.setdefault(r["session_id"], []).append(r)

    scored = [(sig["session_id"], _to_rc(sig)) for sig in chosen_sigs] + [
        (sig["session_id"], _to_rc(sig)) for sig in rejected_sigs
    ]
    pairs = build_preference_pairs(scored, min_gap=min_gap)

    examples = []
    for pair in pairs:
        c_steps = session_steps.get(pair["chosen_id"], [])
        r_steps = session_steps.get(pair["rejected_id"], [])
        if not c_steps or not r_steps:
            continue
        prompt = c_steps[0]["question"] if c_steps else ""
        rejected_question = r_steps[0]["question"] if r_steps else ""
        # DPO sözleşmesi: chosen ve rejected AYNI prompt'a iki yanıt olmalı. Farklı soruya
        # ait rejected, prompt'la ilgisiz → gürültülü/yanlış tercih gradyanı; çifti atla.
        if prompt != rejected_question:
            continue
        examples.append(
            {
                "prompt": prompt,
                "chosen": _session_text(c_steps),
                "rejected": _session_text(r_steps),
                "metadata": {"chosen_id": pair["chosen_id"], "rejected_id": pair["rejected_id"]},
            }
        )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return examples


def get_dpo_stats(store: SqliteStore | None = None) -> dict[str, Any]:
    s = store or SqliteStore()
    all_sigs = s.list_reward_signals()
    dist: dict[str, int] = {}
    for sig in all_sigs:
        dist[sig["label"]] = dist.get(sig["label"], 0) + 1
    return {
        "n_signals": len(all_sigs),
        "label_distribution": dist,
        "dpo_eligible_pairs": dist.get("chosen", 0) * dist.get("rejected", 0),
    }
