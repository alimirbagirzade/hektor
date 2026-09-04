"""SFT birleştirme (synth + kart → dedup → disiplin mix) testleri — DB/LLM'siz (stub'lı)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.training import sft_assembly
from app.training.discipline_dataset import discipline_jsonl_lines


class _EmptyStore:
    def list_approved_cards(self) -> list:
        return []


def _write_synth(tmp_path, n: int) -> None:
    lora_dir = tmp_path / "data" / "lora_sft"
    lora_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"soru {i}"},
                    {
                        "role": "assistant",
                        "content": f"Bu {i}. sentetik cevap yeterince ayrıksıdır {i}.",
                    },
                ]
            },
            ensure_ascii=False,
        )
        for i in range(n)
    ]
    (lora_dir / "synthetic_qa.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_assemble_mixes_discipline(tmp_path, monkeypatch) -> None:
    _write_synth(tmp_path, 50)
    monkeypatch.setattr(sft_assembly, "build_dataset", lambda cards: [])
    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", _EmptyStore)

    settings = SimpleNamespace(root=tmp_path)
    asm = sft_assembly.assemble_sft_lines(settings, discipline=True, discipline_ratio=0.25, seed=0)

    assert asm.synth_n == 50
    assert asm.card_n == 0
    assert asm.discipline is not None
    assert asm.discipline["discipline_used"] > 0
    assert asm.total == asm.discipline["total"]
    # disiplin satırları gerçekten karışımda (havuz ∩ set = used).
    disc = set(discipline_jsonl_lines(seed=0))
    assert len(set(asm.lines) & disc) == asm.discipline["discipline_used"]


def test_assemble_no_discipline(tmp_path, monkeypatch) -> None:
    _write_synth(tmp_path, 30)
    monkeypatch.setattr(sft_assembly, "build_dataset", lambda cards: [])
    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", _EmptyStore)

    settings = SimpleNamespace(root=tmp_path)
    asm = sft_assembly.assemble_sft_lines(settings, discipline=False)

    assert asm.discipline is None
    assert asm.total == asm.deduped == 30


def test_assemble_deterministic(tmp_path, monkeypatch) -> None:
    _write_synth(tmp_path, 40)
    monkeypatch.setattr(sft_assembly, "build_dataset", lambda cards: [])
    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", _EmptyStore)

    settings = SimpleNamespace(root=tmp_path)
    a = sft_assembly.assemble_sft_lines(settings, seed=5)
    b = sft_assembly.assemble_sft_lines(settings, seed=5)
    assert a.lines == b.lines
