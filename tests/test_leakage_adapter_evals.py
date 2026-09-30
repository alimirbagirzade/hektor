"""Sızıntı kapısı adapter eval setlerini de kapsar (Kademe-2 C6, 2026-09-30) — çevrimdışı."""

from __future__ import annotations

import json
from pathlib import Path

from app.evals.profile.leakage import check_leakage
from app.lora.mix_cli import adapter_eval_items


def test_adapter_eval_items_loaded_from_repo() -> None:
    items = adapter_eval_items(Path(__file__).resolve().parents[1] / "evals")
    assert len(items) >= 80
    assert all(it.id.startswith("adapter_eval:") for it in items)


def test_adapter_eval_question_in_train_is_leak(tmp_path: Path) -> None:
    (tmp_path / "discipline_core.jsonl").write_text(
        json.dumps({"question": "Sadece hayatta kalan coinlerle backtest yeterli mi?"}) + "\n",
        encoding="utf-8",
    )
    items = adapter_eval_items(tmp_path)
    leaked = [
        {
            "messages": [
                {"role": "user", "content": "Sadece hayatta kalan coinlerle backtest yeterli mi?"},
                {"role": "assistant", "content": "Hayır."},
            ]
        }
    ]
    assert check_leakage(items, leaked).hits
    clean = [{"messages": [{"role": "user", "content": "RSI nedir?"}]}]
    assert not check_leakage(items, clean).hits
