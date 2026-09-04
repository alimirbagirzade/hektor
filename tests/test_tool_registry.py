"""Bilimsel araç kayıt defteri testleri (keşif + parametre doğrulama + çözümleme)."""

from __future__ import annotations

import pytest

from app.tools.tool_registry import (
    get_tool,
    list_tools,
    resolve,
    validate_params,
)


def test_builtins_registered() -> None:
    ids = {t.tool_id for t in list_tools()}
    assert {"montecarlo", "stats-correlation", "stats-describe", "verify-backtest"} <= ids


def test_category_filter() -> None:
    prob = list_tools("probability")
    assert prob and all(t.category == "probability" for t in prob)


def test_get_unknown_tool_is_none() -> None:
    assert get_tool("does-not-exist") is None


def test_validate_params_missing_required_and_seed() -> None:
    problems = validate_params("montecarlo", {})
    assert any("trade_returns" in p for p in problems)
    assert any("seed" in p for p in problems)


def test_validate_params_ok() -> None:
    assert validate_params("montecarlo", {"trade_returns": [0.1], "seed": 1}) == []


def test_validate_unknown_tool_raises() -> None:
    with pytest.raises(KeyError):
        validate_params("nope", {})


def test_resolve_and_call_montecarlo() -> None:
    fn = resolve("montecarlo")
    res = fn([0.05, -0.02, 0.03], seed=1, n_paths=50)
    assert res.n_paths == 50


def test_requires_seed_contract() -> None:
    assert get_tool("montecarlo").requires_seed is True  # type: ignore[union-attr]
    assert get_tool("stats-describe").requires_seed is False  # type: ignore[union-attr]
