"""DatasetBuilder birim testleri — çevrimdışı (in-memory SQLite)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.memory.sqlite_store import SqliteStore, TrainingExample
from app.training.dataset_builder import DatasetBuilder


def _make_store_with_examples(n: int) -> SqliteStore:
    tmp = tempfile.mkstemp(suffix=".db")[1]
    store = SqliteStore(db_path=tmp)
    with store.session() as s:
        for i in range(n):
            s.add(
                TrainingExample(
                    example_id=f"ex_{i:04d}",
                    source_paper_id=f"paper_{i % 3}",
                    example_type="summarize",
                    instruction=f"Instruction {i}",
                    input_text=f"Input {i}",
                    output_text=f"Output {i} (unique content to avoid dedup)",
                )
            )
    return store


def _builder_with_store(n: int, tmp_path: Path) -> DatasetBuilder:
    store = _make_store_with_examples(n)
    b = DatasetBuilder(store=store)
    b.settings = type("S", (), {"jsonl_dir": tmp_path})()  # type: ignore[assignment]
    return b


def test_empty_store_returns_zero_without_writing(tmp_path: Path) -> None:
    # Boş veri → 0/0 döner ve dosya yazmaz (clobber koruması).
    b = _builder_with_store(0, tmp_path)
    r = b.build(lora_eligible_only=False)
    assert r.n_train == 0
    assert r.n_valid == 0


def test_empty_build_does_not_clobber_existing(tmp_path: Path) -> None:
    # Önceden hazır train.jsonl varsa, boş build onu SIFIRLAMAMALI (veri kaybı yok).
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    valid.write_text('{"prompt": "a", "completion": "b"}\n', encoding="utf-8")
    b = _builder_with_store(0, tmp_path)
    b.build(lora_eligible_only=False)
    assert train.read_text(encoding="utf-8").strip() != ""  # korundu
    assert valid.read_text(encoding="utf-8").strip() != ""


def test_few_examples_valid_disjoint_from_train(tmp_path: Path) -> None:
    # 5 örnek < 8 → valid azınlıkta, train çoğunlukta, AYRIK (kopyalama yok).
    b = _builder_with_store(5, tmp_path)
    r = b.build(lora_eligible_only=False)
    assert r.n_train + r.n_valid == 5
    assert r.n_train > r.n_valid
    with open(r.train_path, encoding="utf-8") as f:
        train_recs = {line.strip() for line in f if line.strip()}
    with open(r.valid_path, encoding="utf-8") as f:
        valid_recs = {line.strip() for line in f if line.strip()}
    assert train_recs.isdisjoint(valid_recs)  # OOS: hiç örtüşme yok


def test_enough_examples_proper_split(tmp_path: Path) -> None:
    b = _builder_with_store(20, tmp_path)
    r = b.build(valid_ratio=0.15, lora_eligible_only=False)
    assert r.n_valid >= 4
    assert r.n_train > 0
    assert r.n_train + r.n_valid == 20


def test_valid_set_min_4_enforced(tmp_path: Path) -> None:
    # 20 örnek, ratio=0.05 → int(20*0.05)=1 < 4 → min 4 olmalı
    b = _builder_with_store(20, tmp_path)
    r = b.build(valid_ratio=0.05, lora_eligible_only=False)
    assert r.n_valid >= 4


def test_files_are_valid_jsonl(tmp_path: Path) -> None:
    b = _builder_with_store(15, tmp_path)
    r = b.build(lora_eligible_only=False)
    with open(r.train_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert all("prompt" in rec and "completion" in rec for rec in lines)


def test_deduplication(tmp_path: Path) -> None:
    store = _make_store_with_examples(0)
    with store.session() as s:
        for i in range(3):
            s.add(
                TrainingExample(
                    example_id=f"dup_{i}",
                    source_paper_id="p1",
                    example_type="t",
                    instruction="Same",
                    input_text="Same",
                    output_text="Same",
                )
            )
    b = DatasetBuilder(store=store)
    b.settings = type("S", (), {"jsonl_dir": tmp_path})()  # type: ignore[assignment]
    records = b.collect(lora_eligible_only=False)
    assert len(records) == 1  # 3 kopya → 1 unique


def test_collect_includes_synthesis_examples(tmp_path: Path) -> None:
    # cross_paper_synthesis (source_paper_id=None) lora_eligible_only=True'da bile dahil edilmeli.
    store = _make_store_with_examples(0)
    with store.session() as s:
        s.add(
            TrainingExample(
                example_id="synth_0",
                source_paper_id=None,
                example_type="cross_paper_synthesis",
                instruction="İki makaleyi birleştir",
                input_text="",
                output_text="Sentez sonucu (benzersiz)",
            )
        )
    b = DatasetBuilder(store=store)
    b.settings = type("S", (), {"jsonl_dir": tmp_path})()  # type: ignore[assignment]
    records = b.collect(lora_eligible_only=True)
    assert len(records) == 1  # sentez örneği filtreyle elenmedi


def test_curriculum_phase_no_synthesis_duplication(tmp_path: Path) -> None:
    # Faz yolunda current/prev/nxt sentez örneğini 3x getirebilir; global dedup tek bırakmalı.
    store = _make_store_with_examples(0)
    with store.session() as s:
        s.add(
            TrainingExample(
                example_id="synth_x",
                source_paper_id=None,
                example_type="cross_paper_synthesis",
                instruction="birleştir",
                input_text="",
                output_text="sentez (benzersiz)",
            )
        )
    b = DatasetBuilder(store=store)
    b.settings = type("S", (), {"jsonl_dir": tmp_path})()  # type: ignore[assignment]
    r = b.build(phase=2, lora_eligible_only=False)
    assert r.n_train + r.n_valid == 1  # 3 kopya değil


def test_deterministic_with_seed(tmp_path: Path) -> None:
    b = _builder_with_store(20, tmp_path)
    r1 = b.build(seed=42, lora_eligible_only=False)
    r2 = b.build(seed=42, lora_eligible_only=False)
    assert r1.content_hash == r2.content_hash


def test_content_hash_changes_with_different_data(tmp_path: Path) -> None:
    b1 = _builder_with_store(10, tmp_path / "a")
    b2 = _builder_with_store(15, tmp_path / "b")
    r1 = b1.build(lora_eligible_only=False)
    r2 = b2.build(lora_eligible_only=False)
    assert r1.content_hash != r2.content_hash
