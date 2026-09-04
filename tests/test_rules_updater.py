"""rules_updater modülü birim testleri — çevrimdışı, geçici DB."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.learning.memory import (
    init_schema,
    save_error_pattern,
    save_model_trial,
    save_system_profile,
)
from app.agents.learning.rules_updater import (
    analyze_failed_trials,
    approve_suggestion,
    dismiss_suggestion,
    generate_rule_suggestions,
    list_pending_suggestions,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "test_learning.db"
    init_schema(p)
    return p


def _seed_profile(db: Path) -> str:
    return save_system_profile(
        {"os": "Darwin", "cpu": {"name": "M1"}, "memory": {"ram_total_gb": 8.0}, "gpu": {}},
        db_path=db,
    )


def test_empty_db_returns_no_suggestions(db: Path) -> None:
    result = generate_rule_suggestions(db_path=db)
    assert result == []


def test_analyze_failed_trials_below_threshold(db: Path) -> None:
    pid = _seed_profile(db)
    save_model_trial(pid, "model_a", "ollama", "failed", db_path=db)
    save_model_trial(pid, "model_a", "ollama", "failed", db_path=db)
    assert analyze_failed_trials(min_failures=3, db_path=db) == []


def test_analyze_failed_trials_at_threshold(db: Path) -> None:
    pid = _seed_profile(db)
    for _ in range(3):
        save_model_trial(pid, "bad_model", "ollama", "failed", db_path=db)
    results = analyze_failed_trials(min_failures=3, db_path=db)
    assert len(results) == 1
    assert results[0].model_id == "bad_model"
    assert results[0].failure_rate == 1.0


def test_generate_blacklist_suggestion_for_failing_model(db: Path) -> None:
    pid = _seed_profile(db)
    for _ in range(3):
        save_model_trial(pid, "flaky_model", "ollama", "failed", db_path=db)

    suggestions = generate_rule_suggestions(db_path=db)
    patches = [json.loads(s.proposed_patch) for s in suggestions]
    blacklist = next((p for p in patches if p.get("action") == "blacklist_model"), None)
    assert blacklist is not None
    assert blacklist["model_id"] == "flaky_model"


def test_generate_throttle_suggestion_for_slow_model(db: Path) -> None:
    pid = _seed_profile(db)
    save_model_trial(pid, "slow_model", "ollama", "unstable", tokens_per_second=2.0, db_path=db)
    save_model_trial(pid, "slow_model", "ollama", "unstable", tokens_per_second=2.0, db_path=db)

    suggestions = generate_rule_suggestions(db_path=db)
    patches = [json.loads(s.proposed_patch) for s in suggestions]
    throttle = next((p for p in patches if p.get("action") == "throttle_model"), None)
    assert throttle is not None
    assert throttle["model_id"] == "slow_model"


def test_generate_error_pattern_suggestion(db: Path) -> None:
    for _ in range(2):
        save_error_pattern(
            error_signature="OOM_KILL",
            error_type="resource",
            probable_cause="Yetersiz RAM",
            recommended_fix="Daha küçük model seç",
            confidence=0.9,
            db_path=db,
        )

    suggestions = generate_rule_suggestions(db_path=db)
    patches = [json.loads(s.proposed_patch) for s in suggestions]
    known_err = next((p for p in patches if p.get("action") == "add_known_error"), None)
    assert known_err is not None
    assert known_err["error_signature"] == "OOM_KILL"
    assert known_err["occurrences"] == 2


def test_dry_run_does_not_persist(db: Path) -> None:
    pid = _seed_profile(db)
    for _ in range(3):
        save_model_trial(pid, "ghost_model", "ollama", "failed", db_path=db)

    suggestions = generate_rule_suggestions(db_path=db, dry_run=True)
    assert len(suggestions) > 0
    assert list_pending_suggestions(db_path=db) == []


def test_idempotent_suggestion_generation(db: Path) -> None:
    pid = _seed_profile(db)
    for _ in range(3):
        save_model_trial(pid, "dup_model", "ollama", "failed", db_path=db)

    first = generate_rule_suggestions(db_path=db)
    second = generate_rule_suggestions(db_path=db)
    assert len(second) == 0
    assert len(list_pending_suggestions(db_path=db)) == len(first)


def test_approve_suggestion(db: Path) -> None:
    pid = _seed_profile(db)
    for _ in range(3):
        save_model_trial(pid, "approve_model", "ollama", "failed", db_path=db)
    generate_rule_suggestions(db_path=db)

    pending = list_pending_suggestions(db_path=db)
    assert len(pending) > 0
    sid = pending[0].suggestion_id

    assert approve_suggestion(sid, db_path=db) is True
    remaining = list_pending_suggestions(db_path=db)
    assert all(s.suggestion_id != sid for s in remaining)


def test_dismiss_suggestion(db: Path) -> None:
    pid = _seed_profile(db)
    for _ in range(3):
        save_model_trial(pid, "dismiss_model", "ollama", "failed", db_path=db)
    generate_rule_suggestions(db_path=db)

    pending = list_pending_suggestions(db_path=db)
    sid = pending[0].suggestion_id

    assert dismiss_suggestion(sid, db_path=db) is True
    remaining = list_pending_suggestions(db_path=db)
    assert all(s.suggestion_id != sid for s in remaining)


def test_approve_nonexistent_returns_false(db: Path) -> None:
    assert approve_suggestion("nonexistent-id", db_path=db) is False


def test_dismiss_nonexistent_returns_false(db: Path) -> None:
    assert dismiss_suggestion("nonexistent-id", db_path=db) is False
