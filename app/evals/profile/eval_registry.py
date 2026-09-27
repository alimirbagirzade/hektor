"""Eval koşu kayıt defteri — manifest + kalem sonuçları + metrikler, koşu başına dizin.

Düzen::

    reports/eval/runs/<run_id>/manifest.json
    reports/eval/runs/<run_id>/items.jsonl     (sistem × kalem skorları)
    reports/eval/runs/<run_id>/metrics.json    (sistem → metrikler)
    reports/eval/runs/index.jsonl              (özet satır)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.evals.profile.schema import ItemScore
from app.lora.mix_common import append_jsonl, read_jsonl, write_jsonl


def default_runs_dir() -> Path:
    return get_settings().root / "reports" / "eval" / "runs"


@dataclass
class StoredRun:
    run_id: str
    manifest: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    items: list[ItemScore]


class EvalRegistry:
    def __init__(self, runs_dir: Path | None = None) -> None:
        self.runs_dir = runs_dir or default_runs_dir()

    def save(
        self,
        manifest: dict[str, Any],
        metrics: dict[str, dict[str, Any]],
        items: list[ItemScore],
        summary: dict[str, Any] | None = None,
    ) -> Path:
        run_id = manifest["run_id"]
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (d / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_jsonl(d / "items.jsonl", (s.model_dump() for s in items))
        append_jsonl(
            self.runs_dir / "index.jsonl",
            {
                "run_id": run_id,
                "timestamp": manifest.get("timestamp"),
                "profile": manifest.get("profile"),
                "eval_dataset": manifest.get("eval_dataset"),
                "replay_key": manifest.get("replay_key"),
                "systems": manifest.get("systems"),
                **(summary or {}),
            },
        )
        return d

    def load(self, run_id: str) -> StoredRun:
        d = self.runs_dir / run_id
        if not d.exists():
            raise FileNotFoundError(f"eval koşusu yok: {run_id}")
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        items = [ItemScore.model_validate(r) for r in read_jsonl(d / "items.jsonl")]
        return StoredRun(run_id, manifest, metrics, items)

    def index(self) -> list[dict[str, Any]]:
        return read_jsonl(self.runs_dir / "index.jsonl")

    def latest_for_profile(self, profile: str, eval_dataset_prefix: str = "") -> StoredRun | None:
        for row in reversed(self.index()):
            if row.get("profile") == profile and str(row.get("eval_dataset", "")).startswith(
                eval_dataset_prefix
            ):
                return self.load(row["run_id"])
        return None
