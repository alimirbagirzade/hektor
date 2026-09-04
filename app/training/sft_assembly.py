"""Birleşik SFT seti kurma — sentetik QA + onaylı kart + adversarial disiplin karışımı.

`lora-cloud-prep` (eğitim paketi) ve `pretrain-gate` (offline kalite kapısı) ORTAK yolu;
ikisi aynı birleştirme mantığını kullansın diye buraya çıkarıldı (drift önlenir).

Sıra: sentetik QA dosyası + onaylı kart örnekleri → hash + near-duplicate dedup (A7) →
disiplin örneklerini DEDUP'TAN SONRA ~%25 karıştır (#4 Fix B; şablon örnekleri near-dup
filtresine toplu takılmasın). EĞİTİM BAŞLATMAZ (kural 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.brain.synthetic_qa_builder import dedup_jsonl_lines
from app.lora.dataset_builder import build_dataset
from app.training.discipline_dataset import discipline_jsonl_lines, mix_discipline


@dataclass
class AssemblyResult:
    """Birleşik SFT satırları + nereden geldiği şeffaflığı."""

    lines: list[str]
    synth_n: int
    card_n: int
    deduped: int  # synth + kart, dedup sonrası
    discipline: dict[str, Any] | None = field(default=None)

    @property
    def total(self) -> int:
        return len(self.lines)


def assemble_sft_lines(
    settings: Any,
    *,
    discipline: bool = True,
    discipline_ratio: float = 0.25,
    seed: int = 0,
) -> AssemblyResult:
    """Birleşik SFT JSONL satırlarını kur (eğitim BAŞLATMAZ).

    Args:
        settings: `get_settings()` çıktısı (`.root` kullanılır).
        discipline: Adversarial disiplin örneklerini karıştır (#4 Fix B).
        discipline_ratio: Disiplin payı (disiplin/(taban+disiplin)); v5 dersi ~0.25.
        seed: Determinizm tabanı (karıştırma — kural 6).
    """
    lora_dir = settings.root / "data" / "lora_sft"
    synth_path = lora_dir / "synthetic_qa.jsonl"

    lines: list[str] = []
    synth_n = 0
    if synth_path.exists():
        synth_lines = [
            ln for ln in synth_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        synth_n = len(synth_lines)
        lines += synth_lines

    card_n = 0
    try:
        from app.memory.sqlite_store import SqliteStore

        card_lines = [
            ex.to_jsonl_line() for ex in build_dataset(SqliteStore().list_approved_cards())
        ]
        card_n = len(card_lines)
        lines += card_lines
    except Exception:
        pass

    merged = dedup_jsonl_lines(lines)
    deduped = len(merged)

    disc_stats: dict[str, Any] | None = None
    if discipline and discipline_ratio > 0:
        disc_lines = discipline_jsonl_lines(seed=seed)
        merged, disc_stats = mix_discipline(merged, disc_lines, ratio=discipline_ratio, seed=seed)

    return AssemblyResult(
        lines=merged,
        synth_n=synth_n,
        card_n=card_n,
        deduped=deduped,
        discipline=disc_stats,
    )
