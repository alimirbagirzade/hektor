"""Kayıt defteri sürümleme testleri (çevrimdışı; sentetik satırlar)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.memory.sqlite_store import SqliteStore
from app.registry import RegistryStore


@pytest.fixture
def reg(tmp_path: Path) -> RegistryStore:
    return RegistryStore(SqliteStore(db_path=tmp_path / "reg.db"))


def test_register_and_list_dataset(reg: RegistryStore) -> None:
    out = reg.register_dataset(name="lora_sft", source_type="sft", n_records=669)
    assert out["dataset_version_id"].startswith("ds_")
    assert out["approval_status"] == "pending"
    rows = reg.list_datasets()
    assert len(rows) == 1
    assert rows[0]["n_records"] == 669


def test_dataset_idempotent_by_hash(reg: RegistryStore) -> None:
    a = reg.register_dataset(name="d", content_hash="abc123", n_records=10)
    b = reg.register_dataset(name="d", content_hash="abc123", n_records=10)
    assert a["dataset_version_id"] == b["dataset_version_id"]
    assert len(reg.list_datasets()) == 1


def test_dataset_different_hash_distinct(reg: RegistryStore) -> None:
    reg.register_dataset(name="d", content_hash="h1", n_records=1)
    reg.register_dataset(name="d", content_hash="h2", n_records=2)
    assert len(reg.list_datasets()) == 2


def test_snapshot_rag_index_counts(reg: RegistryStore) -> None:
    store = reg.store
    store.upsert_paper(paper_id="p1", file_hash="fh1", source_path="x.pdf", title="T")
    store.add_chunks(
        [
            {"chunk_id": "p1_0", "paper_id": "p1", "chunk_index": 0, "text": "abc"},
            {"chunk_id": "p1_1", "paper_id": "p1", "chunk_index": 1, "text": "def"},
        ]
    )
    snap = reg.snapshot_rag_index()
    assert snap["n_papers"] == 1
    assert snap["n_chunks"] == 2
    assert snap["rag_index_version_id"].startswith("rag_")


def test_register_embedding_and_snapshot(reg: RegistryStore) -> None:
    out = reg.register_embedding(model_name="nomic-embed-text", dimension=768, provider="ollama")
    assert out["dimension"] == 768
    snap = reg.snapshot_embedding()  # ayarlardan (fake embeddings test ortamında)
    assert snap["model_name"]
    assert len(reg.list_embeddings()) == 2


def test_register_reward_and_flags(reg: RegistryStore) -> None:
    out = reg.register_reward(name="dpo_v1", method="dpo", n_examples=42)
    assert out["secret_scanned"] == 0
    assert reg.set_reward_scan_flags(out["reward_version_id"], secret_scanned=1, pii_scanned=1)
    rows = reg.list_rewards()
    assert rows[0]["secret_scanned"] == 1


def test_decision_log_roundtrip(reg: RegistryStore) -> None:
    reg.log_decision(
        target_type="dataset",
        target_id="ds_x",
        to_status="approved",
        decision="approved",
        reason="test",
        approved_by="ali",
    )
    rows = reg.list_decisions(target_id="ds_x")
    assert len(rows) == 1
    assert rows[0]["approved_by"] == "ali"


# --- dosyadan dataset kaydı ------------------------------------------------
def test_register_dataset_from_file(reg: RegistryStore, tmp_path: Path) -> None:
    f = tmp_path / "lora_sft.jsonl"
    f.write_text('{"a":1}\n{"b":2}\n\n{"c":3}\n', encoding="utf-8")  # 3 dolu satır
    out = reg.register_dataset_from_file(f, source_type="sft")
    assert out["n_records"] == 3
    assert out["content_hash"] and len(out["content_hash"]) == 64  # sha-256 hex
    assert out["name"] == "lora_sft"


def test_register_dataset_from_file_idempotent(reg: RegistryStore, tmp_path: Path) -> None:
    f = tmp_path / "d.jsonl"
    f.write_text('{"x":1}\n', encoding="utf-8")
    a = reg.register_dataset_from_file(f)
    b = reg.register_dataset_from_file(f)  # aynı içerik → aynı hash → idempotent
    assert a["dataset_version_id"] == b["dataset_version_id"]
    assert len(reg.list_datasets()) == 1


def test_register_dataset_from_missing_file_raises(reg: RegistryStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        reg.register_dataset_from_file(tmp_path / "yok.jsonl")
