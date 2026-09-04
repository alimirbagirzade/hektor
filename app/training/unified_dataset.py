"""unified_dataset.py — Tüm SFT kaynaklarını tek JSONL'de birleştirir.

Kaynaklar:
  1. Bilgi kartları (DatasetBuilder.collect)
  2. Mastery sınavı (MasterySFTBuilder.collect)
  3. Tool-use seansları (build_tool_use_dataset)

Çıktı: data/training/unified_sft.jsonl  —  {prompt, completion} (MLX formatı)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.memory.mastery_store import MasteryStore
from app.memory.sqlite_store import SqliteStore
from app.training.dataset_builder import DatasetBuilder
from app.training.mastery_sft_builder import MasterySFTBuilder
from app.training.tool_use_dataset_builder import build_tool_use_dataset

_OUTPUT_PATH = Path("data/training/unified_sft.jsonl")


@dataclass
class UnifiedStats:
    card_count: int
    mastery_count: int
    tool_use_count: int
    total: int
    output_path: Path

    def summary(self) -> str:
        return (
            f"Bilgi kartı: {self.card_count}  "
            f"Mastery: {self.mastery_count}  "
            f"Tool-use: {self.tool_use_count}  "
            f"→ Toplam: {self.total}"
        )


def _to_mlx(instruction: str, input_text: str, output_text: str) -> dict[str, Any]:
    prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return {"prompt": prompt, "completion": output_text}


class UnifiedDatasetBuilder:
    """Tüm SFT kaynaklarını birleştirip tek JSONL olarak yazar."""

    def __init__(
        self,
        sqlite_store: SqliteStore | None = None,
        mastery_store: MasteryStore | None = None,
        seed: int = 42,
    ) -> None:
        self._store = sqlite_store or SqliteStore()
        self._ms = mastery_store or MasteryStore()
        self._seed = seed

    def build(
        self,
        output_path: Path | None = None,
        min_mastery_score: float = 75.0,
        citation_threshold: float = 0.5,
        shuffle: bool = True,
    ) -> UnifiedStats:
        """Tüm kaynakları topla, birleştir, yaz."""
        records: list[dict[str, Any]] = []

        # 1. Bilgi kartları
        card_examples = DatasetBuilder(store=self._store).collect(lora_eligible_only=False)
        card_count = len(card_examples)
        records.extend(card_examples)

        # 2. Mastery SFT
        mastery_examples = MasterySFTBuilder(
            sqlite_store=self._store, mastery_store=self._ms
        ).collect(
            min_mastery_score=min_mastery_score,
            citation_threshold=citation_threshold,
        )
        mastery_count = len(mastery_examples)
        records.extend(_to_mlx(e.instruction, e.input, e.output) for e in mastery_examples)

        # 3. Tool-use seansları
        tool_examples = build_tool_use_dataset(store=self._store)
        tool_count = len(tool_examples)
        records.extend(
            _to_mlx(e["instruction"], e.get("input", ""), e["output"]) for e in tool_examples
        )

        if shuffle:
            random.Random(self._seed).shuffle(records)

        out = output_path or _OUTPUT_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
            encoding="utf-8",
        )

        return UnifiedStats(
            card_count=card_count,
            mastery_count=mastery_count,
            tool_use_count=tool_count,
            total=len(records),
            output_path=out,
        )
