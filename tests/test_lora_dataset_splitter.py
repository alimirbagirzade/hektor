"""Dataset bölücü testleri."""

from __future__ import annotations

from app.lora.dataset_splitter import (
    DatasetSplit,
    check_leakage,
    split_dataset,
)


def _example(source_id: str, idx: int) -> dict:
    return {"metadata": {"source_id": source_id}, "idx": idx}


def _many_sources(n: int) -> list[dict]:
    return [_example(f"src_{i}", i) for i in range(n)]


def test_split_is_deterministic_with_seed() -> None:
    """Aynı seed aynı bölmeyi üretmeli."""
    data = _many_sources(20)
    a = split_dataset(data, seed=42)
    b = split_dataset(data, seed=42)
    assert [e["idx"] for e in a.train] == [e["idx"] for e in b.train]


def test_split_covers_all_examples() -> None:
    """Her örnek tam olarak bir bölmede yer almalı."""
    data = _many_sources(20)
    split = split_dataset(data, seed=7)
    total = len(split.train) + len(split.valid) + len(split.test)
    assert total == len(data)


def test_no_source_leakage_when_grouped() -> None:
    """Aynı source_id'nin tüm örnekleri tek bölmede olmalı (sızıntı yok)."""
    data = [_example("shared", i) for i in range(4)] + _many_sources(10)
    split = split_dataset(data, seed=3)
    assert check_leakage(split) == []


def test_check_leakage_detects_overlap() -> None:
    """Elle oluşturulan örtüşme sızıntı olarak yakalanmalı."""
    split = DatasetSplit(
        train=[_example("dup", 0)],
        valid=[],
        test=[_example("dup", 1)],
    )
    issues = check_leakage(split)
    assert len(issues) == 1
    assert "dup" in issues[0]


def test_empty_input_returns_empty_split() -> None:
    """Boş girdi boş bölme döndürmeli."""
    split = split_dataset([])
    assert split.train == []
    assert split.valid == []
    assert split.test == []
