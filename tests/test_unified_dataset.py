"""UnifiedDatasetBuilder birim testleri."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def _sq() -> MagicMock:
    m = MagicMock()
    m.list_papers.return_value = []
    m.list_tool_use_examples.return_value = []
    return m


def _ms() -> MagicMock:
    return MagicMock()


@patch("app.training.unified_dataset.build_tool_use_dataset", return_value=[])
@patch("app.training.unified_dataset.MasterySFTBuilder")
@patch("app.training.unified_dataset.DatasetBuilder")
def test_empty_sources_writes_empty_file(MockDB, MockMastery, MockTU, tmp_path: Path) -> None:
    MockDB.return_value.collect.return_value = []
    MockMastery.return_value.collect.return_value = []
    from app.training.unified_dataset import UnifiedDatasetBuilder

    stats = UnifiedDatasetBuilder(sqlite_store=_sq(), mastery_store=_ms()).build(
        output_path=tmp_path / "u.jsonl"
    )
    assert stats.total == 0
    assert (tmp_path / "u.jsonl").exists()


@patch("app.training.unified_dataset.build_tool_use_dataset")
@patch("app.training.unified_dataset.MasterySFTBuilder")
@patch("app.training.unified_dataset.DatasetBuilder")
def test_merges_all_sources(MockDB, MockMastery, MockTU, tmp_path: Path) -> None:
    MockDB.return_value.collect.return_value = [
        {"prompt": "p1", "completion": "c1"},
    ]
    mastery_ex = MagicMock()
    mastery_ex.instruction = "inst"
    mastery_ex.input = "inp"
    mastery_ex.output = "out"
    MockMastery.return_value.collect.return_value = [mastery_ex]
    MockTU.return_value = [{"instruction": "i", "input": "q", "output": "a"}]
    from app.training.unified_dataset import UnifiedDatasetBuilder

    stats = UnifiedDatasetBuilder(sqlite_store=_sq(), mastery_store=_ms()).build(
        output_path=tmp_path / "u.jsonl"
    )
    assert stats.card_count == 1
    assert stats.mastery_count == 1
    assert stats.tool_use_count == 1
    assert stats.total == 3


@patch("app.training.unified_dataset.build_tool_use_dataset", return_value=[])
@patch("app.training.unified_dataset.MasterySFTBuilder")
@patch("app.training.unified_dataset.DatasetBuilder")
def test_summary_string(MockDB, MockMastery, MockTU, tmp_path: Path) -> None:
    MockDB.return_value.collect.return_value = []
    MockMastery.return_value.collect.return_value = []
    from app.training.unified_dataset import UnifiedDatasetBuilder

    stats = UnifiedDatasetBuilder(sqlite_store=_sq(), mastery_store=_ms()).build(
        output_path=tmp_path / "u.jsonl"
    )
    assert "Toplam" in stats.summary()
