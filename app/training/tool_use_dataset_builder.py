"""tool_use_dataset_builder.py — tool_use_examples → SFT eğitim verisi.

DB'deki tool-use seanslarını okur, SFT formatına dönüştürür
(`instruction` / `input` / `output`) ve JSON-Lines dosyası olarak yazar.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.memory.sqlite_store import SqliteStore


def _group_by_session(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["session_id"]].append(r)
    for steps in groups.values():
        steps.sort(key=lambda x: x["step_index"])
    return dict(groups)


def _session_to_sft(session_id: str, steps: list[dict]) -> dict[str, Any] | None:
    if not steps:
        return None
    question = steps[0]["question"]
    call_steps = [s for s in steps if s["step_type"] == "call"]
    conclude_steps = [s for s in steps if s["step_type"] == "conclude"]
    if not call_steps or not conclude_steps:
        return None

    call_summary = "\n".join(
        f"  [{i + 1}] {s['tool_name'] or 'tool'}("
        f"{json.dumps(s['tool_input'])[:100]}) → "
        f"verdict={s['tool_output'].get('verdict', '?')}, "
        f"n_trades={s['tool_output'].get('metrics', {}).get('n_trades', '?')}"
        for i, s in enumerate(call_steps)
    )
    final_verdict = conclude_steps[-1].get("verdict") or "inconclusive"
    return {
        "instruction": (
            "Sen bir ticaret stratejisi araştırmacısısın. "
            "Verilen soruyu analiz et, backtest aracını kullan ve sonucu değerlendir."
        ),
        "input": f"Araştırma Sorusu: {question}\n\nAraç Çağrıları:\n{call_summary}",
        "output": conclude_steps[-1]["content"],
        "metadata": {
            "session_id": session_id,
            "final_verdict": final_verdict,
            "n_tool_calls": len(call_steps),
        },
    }


def build_tool_use_dataset(
    store: SqliteStore | None = None,
    output_path: Path | None = None,
    min_steps: int = 4,
    only_verdict: str | None = None,
) -> list[dict[str, Any]]:
    """tool_use_examples tablosundan SFT örnekleri üret.

    Args:
        store: SqliteStore bağlantısı (None → varsayılan).
        output_path: JSONL çıktı dosyası. None ise yazmaz.
        min_steps: Seans başına minimum adım sayısı (kalite filtresi).
        only_verdict: "pass" / "fail" — sadece bu verdict filtrelenir.

    Returns:
        SFT örnek listesi.
    """
    s = store or SqliteStore()
    rows = s.list_tool_use_examples(limit=5000)
    groups = _group_by_session(rows)

    examples: list[dict[str, Any]] = []
    for session_id, steps in groups.items():
        if len(steps) < min_steps:
            continue
        if only_verdict:
            conclude = [st for st in steps if st["step_type"] == "conclude"]
            if not conclude or conclude[-1].get("verdict") != only_verdict:
                continue
        ex = _session_to_sft(session_id, steps)
        if ex:
            examples.append(ex)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return examples


def get_tool_use_stats(store: SqliteStore | None = None) -> dict[str, Any]:
    """Özet istatistik: seans sayısı, verdict dağılımı, toplam adım."""
    s = store or SqliteStore()
    rows = s.list_tool_use_examples(limit=5000)
    groups = _group_by_session(rows)

    verdict_counts: dict[str, int] = defaultdict(int)
    for steps in groups.values():
        conclude = [st for st in steps if st["step_type"] == "conclude"]
        v = conclude[-1].get("verdict", "unknown") if conclude else "unknown"
        verdict_counts[v] += 1

    return {
        "n_sessions": len(groups),
        "n_steps": len(rows),
        "verdict_distribution": dict(verdict_counts),
        "sft_eligible": sum(1 for steps in groups.values() if len(steps) >= 4),
    }
