"""Reward signal ve DPO dataset builder birim testleri — çevrimdışı."""

from __future__ import annotations

from pathlib import Path

from app.memory.sqlite_store import SqliteStore
from app.training.dpo_dataset_builder import (
    build_dpo_dataset,
    get_dpo_stats,
    score_and_save_sessions,
)
from app.training.reward_signal import (
    REWARD_PASS,
    REWARD_REJECT,
    RewardCriteria,
    build_preference_pairs,
    compute_reward,
)


def _store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(db_path=tmp_path / "test.db")


# ---- compute_reward ----


def test_error_gives_zero_composite() -> None:
    rc = compute_reward({}, verdict="fail", had_error=True)
    assert rc.composite == 0.0
    assert rc.label == "rejected"
    assert rc.execution_ok == 0.0


def test_perfect_metrics_chosen() -> None:
    metrics = {
        "n_trades": 50,
        "sharpe_ratio": 2.5,
        "max_drawdown_pct": 10.0,
        "total_return_pct": 120.0,
        "win_rate": 0.60,
    }
    rc = compute_reward(metrics, verdict="pass")
    assert rc.execution_ok == 1.0
    assert rc.trade_count_ok == 1.0
    assert rc.return_ok == 1.0
    assert rc.composite >= REWARD_PASS
    assert rc.label == "chosen"


def test_real_backtester_keys_score_correctly() -> None:
    # Backtester GERÇEK anahtarlari: "sharpe" + "win_rate_pct" (0-100). Eskiden
    # compute_reward "sharpe_ratio"/"win_rate" okuyup ikisini de 0 hesapliyordu.
    metrics = {
        "n_trades": 50,
        "sharpe": 2.5,
        "max_drawdown_pct": 10.0,
        "total_return_pct": 120.0,
        "win_rate_pct": 60.0,
    }
    rc = compute_reward(metrics, verdict="pass")
    assert rc.sharpe_ok > 0.0  # "sharpe" anahtari okundu
    assert rc.win_rate_ok > 0.0  # win_rate_pct 0-100 → kesire normalize edildi
    assert not any("Düşük win_rate: 0.0%" in n for n in rc.notes)


def test_few_trades_partial_score() -> None:
    rc = compute_reward({"n_trades": 5}, verdict="pass")
    assert rc.trade_count_ok == 0.5
    assert any("Az işlem" in n for n in rc.notes)


def test_negative_sharpe_zero_score() -> None:
    rc = compute_reward({"n_trades": 20, "sharpe_ratio": -1.0}, verdict="pass")
    assert rc.sharpe_ok == 0.0


def test_high_drawdown_zero_score() -> None:
    rc = compute_reward({"n_trades": 20, "max_drawdown_pct": 60.0}, verdict="pass")
    assert rc.drawdown_ok == 0.0


def test_negative_return_zero_score() -> None:
    rc = compute_reward({"n_trades": 20, "total_return_pct": -5.0}, verdict="pass")
    assert rc.return_ok == 0.0


def test_neutral_label_between_thresholds() -> None:
    rc = compute_reward({"n_trades": 15}, verdict="pass")
    assert REWARD_REJECT < rc.composite < REWARD_PASS
    assert rc.label == "neutral"


def test_to_dict_has_all_keys() -> None:
    rc = compute_reward({"n_trades": 20, "sharpe_ratio": 1.0}, verdict="pass")
    d = rc.to_dict()
    assert {"composite", "label", "execution_ok", "notes"} <= d.keys()


# ---- build_preference_pairs ----


def test_preference_pairs_correct() -> None:
    high = RewardCriteria(
        execution_ok=1,
        trade_count_ok=1,
        sharpe_ok=1,
        drawdown_ok=1,
        return_ok=1,
        win_rate_ok=1,
    )
    low = RewardCriteria()
    pairs = build_preference_pairs([("good", high), ("bad", low)], min_gap=0.25)
    assert len(pairs) == 1
    assert pairs[0]["chosen_id"] == "good"
    assert pairs[0]["rejected_id"] == "bad"


def test_preference_pairs_gap_too_small() -> None:
    a = RewardCriteria(execution_ok=0.6)
    b = RewardCriteria(execution_ok=0.4)
    assert build_preference_pairs([("a", a), ("b", b)], min_gap=0.25) == []


def test_preference_pairs_empty() -> None:
    assert build_preference_pairs([]) == []


# ---- SqliteStore reward_signals ----


def test_save_and_get_reward_signal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rc = compute_reward(
        {
            "n_trades": 30,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 15.0,
            "total_return_pct": 50.0,
            "win_rate": 0.55,
        },
        verdict="pass",
    )
    store.save_reward_signal("sess_001", rc, raw_metrics={"n_trades": 30})
    result = store.get_reward_signal("sess_001")
    assert result is not None
    assert result["composite_score"] == rc.composite
    assert result["label"] == rc.label


def test_list_reward_signals_label_filter(tmp_path: Path) -> None:
    store = _store(tmp_path)
    good_rc = compute_reward(
        {
            "n_trades": 30,
            "sharpe_ratio": 2.0,
            "max_drawdown_pct": 10.0,
            "total_return_pct": 80.0,
            "win_rate": 0.58,
        },
        verdict="pass",
    )
    bad_rc = compute_reward({}, had_error=True)
    store.save_reward_signal("good", good_rc)
    store.save_reward_signal("bad", bad_rc)
    assert any(r["session_id"] == "good" for r in store.list_reward_signals(label="chosen"))
    assert any(r["session_id"] == "bad" for r in store.list_reward_signals(label="rejected"))


def test_reward_signal_upsert(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rc = compute_reward({"n_trades": 20}, verdict="pass")
    store.save_reward_signal("dup", rc)
    store.save_reward_signal("dup", rc)
    assert sum(1 for s in store.list_reward_signals() if s["session_id"] == "dup") == 1


# ---- score_and_save_sessions ----


def _seed_session(store: SqliteStore, session_id: str, verdict: str) -> None:
    m = (
        {
            "n_trades": 25,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 12.0,
            "total_return_pct": 40.0,
            "win_rate": 0.55,
        }
        if verdict == "pass"
        else {
            "n_trades": 2,
            "sharpe_ratio": -0.5,
            "max_drawdown_pct": 50.0,
            "total_return_pct": -5.0,
            "win_rate": 0.2,
        }
    )
    store.save_tool_use_example(
        example_id=f"{session_id}_0",
        session_id=session_id,
        question="Q?",
        step_index=0,
        step_type="think",
        content="hipotez",
    )
    store.save_tool_use_example(
        example_id=f"{session_id}_1",
        session_id=session_id,
        question="Q?",
        step_index=1,
        step_type="call",
        content="backtest()",
        tool_name="run_backtest",
        tool_output={"verdict": verdict, "metrics": m},
        verdict=verdict,
    )


def test_score_and_save_sessions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_session(store, "sp", "pass")
    _seed_session(store, "sf", "fail")
    scored = score_and_save_sessions(store=store)
    assert len(scored) == 2
    assert store.get_reward_signal("sp") is not None
    assert store.get_reward_signal("sf") is not None


def test_score_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_session(store, "idem", "pass")
    score_and_save_sessions(store=store)
    assert score_and_save_sessions(store=store) == []


# ---- build_dpo_dataset / get_dpo_stats ----


def test_build_dpo_no_signals(tmp_path: Path) -> None:
    assert build_dpo_dataset(store=_store(tmp_path)) == []


def test_get_dpo_stats_empty(tmp_path: Path) -> None:
    stats = get_dpo_stats(store=_store(tmp_path))
    assert stats["n_signals"] == 0
    assert stats["dpo_eligible_pairs"] == 0


def test_get_dpo_stats_with_signals(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_session(store, "sp2", "pass")
    _seed_session(store, "sf2", "fail")
    score_and_save_sessions(store=store)
    stats = get_dpo_stats(store=store)
    assert stats["n_signals"] == 2
    assert "chosen" in stats["label_distribution"] or "rejected" in stats["label_distribution"]
