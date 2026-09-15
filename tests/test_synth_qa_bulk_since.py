"""synth-qa-bulk `--since` filtresi + hedef-aşılmış davranışı — çevrimdışı (LLM/DB sahte).

Kademe-2 av (2026-09-15): mevcut çıktı `--target`'ı zaten aşınca komut ilk batch'ten önce
durup "bitti / Stage 2 karşılandı" basıyordu → yeni makaleler hiç işlenmeden başarı bildirimi.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from app.main import app


class _LLM:
    def available(self) -> bool:
        return True

    def active_backend(self) -> str:
        return "sahte"


class _Store:
    def list_papers(self):
        return [
            SimpleNamespace(paper_id="yeni", created_at="2026-09-14T10:00:00+00:00"),
            SimpleNamespace(paper_id="eski", created_at="2026-06-01T10:00:00+00:00"),
        ]


def _install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seen: list[list[str]]) -> None:
    from app.config import settings as settings_mod

    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    settings_mod.get_settings.cache_clear()
    monkeypatch.setattr("app.brain.local_llm.LocalLLM", _LLM)
    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", _Store)

    def _gen(store, *, paper_ids, **kwargs):
        seen.append(list(paper_ids))
        return [], {}

    monkeypatch.setattr("app.brain.synthetic_qa_builder.generate_synthetic_dataset", _gen)


def test_since_processes_only_new_papers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    _install(monkeypatch, tmp_path, seen)
    res = CliRunner().invoke(
        app, ["synth-qa-bulk", "--since", "2026-09-13", "--output", str(tmp_path / "out")]
    )
    assert res.exit_code == 0, res.output
    assert seen == [["yeni"]]


def test_target_already_met_reports_unprocessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []
    _install(monkeypatch, tmp_path, seen)
    out = tmp_path / "out"
    out.mkdir()
    rows = [
        json.dumps({"messages": [{"role": "user", "content": f"q{i}"}], "metadata": {}})
        for i in range(3)
    ]
    (out / "synthetic_qa.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    res = CliRunner().invoke(app, ["synth-qa-bulk", "--target", "2", "--output", str(out)])
    assert res.exit_code == 0, res.output
    assert seen == []  # hiçbir batch koşmadı
    assert "İŞLENMEDİ" in res.output
    assert "Stage 2" not in res.output
