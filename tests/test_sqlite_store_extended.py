"""SqliteStore yeni metotları için ek testler — çevrimdışı."""

from __future__ import annotations

import tempfile

import pytest

from app.memory.sqlite_store import SqliteStore, TrainingExample


@pytest.fixture
def store() -> SqliteStore:
    tmp = tempfile.mkstemp(suffix=".db")[1]
    return SqliteStore(db_path=tmp)


def _add_example(s: SqliteStore, eid: str, text: str = "x") -> None:
    with s.session() as sess:
        sess.add(
            TrainingExample(
                example_id=eid,
                source_paper_id="p1",
                example_type="test",
                instruction=text,
                input_text="",
                output_text=text,
            )
        )


# ---------- list_training_examples ----------
def test_list_training_examples_empty(store: SqliteStore) -> None:
    assert store.list_training_examples() == []


def test_list_training_examples_returns_all(store: SqliteStore) -> None:
    _add_example(store, "e1", "a")
    _add_example(store, "e2", "b")
    rows = store.list_training_examples()
    assert len(rows) == 2
    assert {r["example_id"] for r in rows} == {"e1", "e2"}


def test_list_training_examples_limit(store: SqliteStore) -> None:
    for i in range(10):
        _add_example(store, f"e{i:02d}", f"text_{i}")
    rows = store.list_training_examples(limit=3)
    assert len(rows) == 3


def test_list_training_examples_fields(store: SqliteStore) -> None:
    _add_example(store, "ex1", "hello")
    row = store.list_training_examples()[0]
    for field in (
        "example_id",
        "source_paper_id",
        "example_type",
        "instruction",
        "output_text",
        "created_at",
    ):
        assert field in row


# ---------- delete_training_example ----------
def test_delete_existing_example(store: SqliteStore) -> None:
    _add_example(store, "del1")
    result = store.delete_training_example("del1")
    assert result is True
    assert store.list_training_examples() == []


def test_delete_missing_returns_false(store: SqliteStore) -> None:
    result = store.delete_training_example("nonexistent")
    assert result is False


# ---------- list_backtests ----------
def test_list_backtests_empty(store: SqliteStore) -> None:
    assert store.list_backtests() == []


def test_list_backtests_after_save(store: SqliteStore) -> None:
    store.save_strategy(
        strategy_id="s1",
        name="my_strat",
        market="XAUUSD",
        timeframe="15m",
        ir_json="{}",
        origin="manual",
    )
    store.save_backtest(
        backtest_id="bt1",
        strategy_id="s1",
        data_file="synthetic",
        n_trades=50,
        total_return_pct=12.5,
        sharpe=1.1,
        sortino=1.2,
        max_drawdown_pct=-15.0,
        profit_factor=1.4,
        win_rate_pct=55.0,
        metrics_json="{}",
        verdict="fail",
    )
    rows = store.list_backtests()
    assert len(rows) == 1
    assert rows[0]["strategy_name"] == "my_strat"
    assert rows[0]["verdict"] == "fail"
    assert rows[0]["n_trades"] == 50


def test_list_backtests_limit(store: SqliteStore) -> None:
    store.save_strategy(
        strategy_id="s2", name="s", market="X", timeframe="1h", ir_json="{}", origin="manual"
    )
    for i in range(5):
        store.save_backtest(
            backtest_id=f"bt{i}",
            strategy_id="s2",
            data_file="x",
            n_trades=10,
            total_return_pct=1.0,
            sharpe=0.5,
            sortino=0.5,
            max_drawdown_pct=-5.0,
            profit_factor=1.1,
            win_rate_pct=50.0,
            metrics_json="{}",
        )
    assert len(store.list_backtests(limit=3)) == 3
