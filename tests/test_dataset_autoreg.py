"""Eğitim hattı → registry otomatik dataset sürümleme testleri (Modül 8, offline)."""

from __future__ import annotations

from pathlib import Path

from app.config.settings import Settings
from app.memory.sqlite_store import SqliteStore
from app.registry import RegistryStore
from app.training.detached_launch import _auto_register_dataset


def _settings(tmp: Path) -> Settings:
    return Settings(sqlite_path=tmp / "auto.db", allow_fake_embeddings=True)


def test_auto_register_versions_canonical_dataset(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    src = tmp_path / "lora_sft.jsonl"
    src.write_text('{"messages":[]}\n{"messages":[]}\n\n{"messages":[]}\n', encoding="utf-8")

    _auto_register_dataset(s, src)

    rows = RegistryStore(SqliteStore(s.sqlite_file)).list_datasets()
    assert len(rows) == 1
    assert rows[0]["name"] == "lora_sft"
    assert rows[0]["n_records"] == 3
    assert rows[0]["approval_status"] == "pending"  # terfi İNSAN ONAYI (Kural 8)


def test_auto_register_idempotent(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    src = tmp_path / "lora_sft.jsonl"
    src.write_text('{"messages":[]}\n', encoding="utf-8")
    _auto_register_dataset(s, src)
    _auto_register_dataset(s, src)  # aynı içerik → aynı hash → tek sürüm
    assert len(RegistryStore(SqliteStore(s.sqlite_file)).list_datasets()) == 1


def test_auto_register_skips_empty(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    src = tmp_path / "empty.jsonl"
    src.write_text("", encoding="utf-8")
    _auto_register_dataset(s, src)  # hata yok, kayıt yok
    assert not RegistryStore(SqliteStore(s.sqlite_file)).list_datasets()


def test_auto_register_missing_file_is_noop(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    _auto_register_dataset(s, tmp_path / "yok.jsonl")  # best-effort → sessiz
    assert not RegistryStore(SqliteStore(s.sqlite_file)).list_datasets()
