from __future__ import annotations

import asyncio

from app.monitoring.self_heal import SelfHealingController
from app.monitoring.sentinel import ProbeResult, SentinelReport


def _report(name: str, status: str) -> SentinelReport:
    return SentinelReport(
        overall=status,
        summary=status,
        probes=[ProbeResult(name, status, "test")],
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_requires_three_consecutive_failures(tmp_path) -> None:
    calls = {"n": 0}

    async def repair():
        calls["n"] += 1
        return {"verified": True}

    ctl = SelfHealingController(repairs={"x": repair}, state_path=tmp_path / "state.json")
    asyncio.run(ctl.run_once(_report("x", "fail")))
    asyncio.run(ctl.run_once(_report("x", "fail")))
    assert calls["n"] == 0
    result = asyncio.run(ctl.run_once(_report("x", "fail")))
    assert calls["n"] == 1
    assert result["actions"][0]["verified"] is True
    assert ctl.status()["consecutive"]["x"] == 0


def test_success_resets_failure_streak(tmp_path) -> None:
    async def repair():
        raise AssertionError("repair should not run")

    ctl = SelfHealingController(repairs={"x": repair}, state_path=tmp_path / "state.json")
    asyncio.run(ctl.run_once(_report("x", "fail")))
    asyncio.run(ctl.run_once(_report("x", "ok")))
    asyncio.run(ctl.run_once(_report("x", "fail")))
    assert ctl.status()["consecutive"]["x"] == 1


def test_failed_repair_opens_circuit_and_persists(tmp_path) -> None:
    calls = {"n": 0}

    async def repair():
        calls["n"] += 1
        return {"verified": False, "reason": "still broken"}

    path = tmp_path / "state.json"
    ctl = SelfHealingController(repairs={"x": repair}, state_path=path)
    for _ in range(3):
        asyncio.run(ctl.run_once(_report("x", "fail")))
    assert calls["n"] == 1
    assert ctl.status()["cooldown_until"]["x"]

    restored = SelfHealingController(repairs={"x": repair}, state_path=path)
    asyncio.run(restored.run_once(_report("x", "fail")))
    assert calls["n"] == 1
    assert restored.status()["history"][0]["verified"] is False
