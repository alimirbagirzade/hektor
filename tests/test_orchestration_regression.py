"""Gerileme bloklayıcı — çevrimdışı testler (enjekte metrik sağlayıcı + baseline store).

Gerçek veri/dosya YOK: saf kıyas fonksiyonu, BaselineStore (tmp dosya), RegressionGuard
(enjekte sağlayıcı) ve delege'nin verdict→StageStatus eşlemesi doğrulanır. v5 mekanizması
(top_opening_share artışı) özellikle test edilir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.orchestration.regression as reg_mod
from app.orchestration import delegates
from app.orchestration.orchestrator import RunContext
from app.orchestration.pipeline import StageStatus
from app.orchestration.regression import (
    BaselineStore,
    RegressionGuard,
    RegressionResult,
    evaluate_regression,
    metrics_from_report,
)

# Sağlıklı (geçen) baseline.
_GOOD = {
    "guaranteed_profit_hits": 0.0,
    "top_opening_share": 0.10,
    "leakage_prefix_share": 0.0,
    "ignores_costs_hits": 0.0,
    "discipline_coverage": 0.50,
    "dataset_go": 1.0,
}


def _find(r: RegressionResult, name: str):  # type: ignore[no-untyped-def]
    return next((f for f in r.findings if f.name == name), None)


# ── saf kıyas ──────────────────────────────────────────────────────────────────


def test_skip_when_no_baseline() -> None:
    r = evaluate_regression(dict(_GOOD), None)
    assert r.verdict == "skip"
    assert all(f.status == "new" for f in r.findings)


def test_pass_when_equal() -> None:
    r = evaluate_regression(dict(_GOOD), dict(_GOOD))
    assert r.verdict == "pass"


def test_pass_when_improved() -> None:
    better = dict(_GOOD)
    better["top_opening_share"] = 0.02  # daha az ezber = iyileşme
    better["discipline_coverage"] = 0.70  # daha çok kapsam = iyileşme
    r = evaluate_regression(better, dict(_GOOD))
    assert r.verdict == "pass"
    assert _find(r, "top_opening_share").status == "improved"
    assert _find(r, "discipline_coverage").status == "improved"


def test_fail_on_opening_memorization_increase() -> None:
    """v5 MEKANİZMASI: açılış ezberi oranı baseline'ı tolerans dışı aşarsa gerileme."""
    worse = dict(_GOOD)
    worse["top_opening_share"] = 0.30  # 0.10 → 0.30, abs_tol=0.05 dışı
    r = evaluate_regression(worse, dict(_GOOD))
    assert r.verdict == "fail"
    assert _find(r, "top_opening_share").status == "regressed"


def test_opening_increase_within_tolerance_passes() -> None:
    worse = dict(_GOOD)
    worse["top_opening_share"] = 0.13  # 0.10 → 0.13, abs_tol=0.05 içinde
    r = evaluate_regression(worse, dict(_GOOD))
    assert r.verdict == "pass"
    assert _find(r, "top_opening_share").status == "stable"


def test_fail_on_poison_increase_zero_tolerance() -> None:
    worse = dict(_GOOD)
    worse["guaranteed_profit_hits"] = 1.0  # 0 → 1, abs_tol=0 → herhangi artış bloklar
    r = evaluate_regression(worse, dict(_GOOD))
    assert r.verdict == "fail"
    assert _find(r, "guaranteed_profit_hits").status == "regressed"


def test_fail_on_discipline_coverage_drop() -> None:
    worse = dict(_GOOD)
    worse["discipline_coverage"] = 0.30  # 0.50 → 0.30 düşüş = gerileme
    r = evaluate_regression(worse, dict(_GOOD))
    assert r.verdict == "fail"
    assert _find(r, "discipline_coverage").status == "regressed"


def test_fail_on_go_to_nogo() -> None:
    worse = dict(_GOOD)
    worse["dataset_go"] = 0.0  # GO → NO-GO
    r = evaluate_regression(worse, dict(_GOOD))
    assert r.verdict == "fail"


def test_missing_metric_ignored() -> None:
    """Baseline'da olmayan metrik kıyaslanmaz (yeni metrik eklenmesi gerileme saymaz)."""
    base = {"top_opening_share": 0.10}
    cur = {"top_opening_share": 0.10, "discipline_coverage": 0.01}
    r = evaluate_regression(cur, base)
    assert r.verdict == "pass"


# ── metrics_from_report ─────────────────────────────────────────────────────────


def test_metrics_from_report_coverage_ratio() -> None:
    m = metrics_from_report(
        {
            "verdict": "GO",
            "guaranteed_profit_hits": 0,
            "top_opening_share": 0.2,
            "leakage_prefix_share": 0.0,
            "ignores_costs_hits": 0,
            "discipline_present": 30,
            "discipline_target": 60,
        }
    )
    assert m["discipline_coverage"] == 0.5
    assert m["dataset_go"] == 1.0


def test_metrics_from_report_zero_target_is_full_coverage() -> None:
    m = metrics_from_report({"verdict": "NO-GO", "discipline_target": 0})
    assert m["discipline_coverage"] == 1.0  # hedef yoksa kapsam cezası yok
    assert m["dataset_go"] == 0.0


# ── BaselineStore ───────────────────────────────────────────────────────────────


def test_baseline_store_roundtrip(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "baseline.json")
    assert store.load() is None  # yokken None
    store.save(dict(_GOOD), note="ilk")
    loaded = store.load()
    assert loaded == _GOOD


def test_baseline_store_corrupt_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "baseline.json"
    p.write_text("{ bozuk json", encoding="utf-8")
    assert BaselineStore(p).load() is None


# ── RegressionGuard ─────────────────────────────────────────────────────────────


def test_guard_skip_when_no_current_metrics(tmp_path: Path) -> None:
    guard = RegressionGuard(BaselineStore(tmp_path / "b.json"), metrics_provider=lambda: {})
    assert guard.run().verdict == "skip"


def test_guard_fail_on_regression(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "b.json")
    store.save(dict(_GOOD))
    worse = dict(_GOOD, top_opening_share=0.40)
    guard = RegressionGuard(store, metrics_provider=lambda: worse)
    assert guard.run().verdict == "fail"


def test_guard_commit_baseline_then_pass(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "b.json")
    guard = RegressionGuard(store, metrics_provider=lambda: dict(_GOOD))
    # İlk koşu: baseline yok → skip.
    assert guard.run().verdict == "skip"
    # Commit → baseline kurulur.
    committed = guard.commit_baseline(note="test")
    assert committed == _GOOD
    # Aynı metrik → artık pass.
    assert guard.run().verdict == "pass"


def test_guard_provider_exception_yields_skip(tmp_path: Path) -> None:
    def boom() -> dict[str, float]:
        raise RuntimeError("veri yok")

    guard = RegressionGuard(BaselineStore(tmp_path / "b.json"), metrics_provider=boom)
    assert guard.run().verdict == "skip"


# ── delege verdict → StageStatus eşlemesi ──────────────────────────────────────


def _ctx() -> RunContext:
    return RunContext(run_id="r", stage="regression", run={}, params={}, store=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("pass", StageStatus.completed),
        ("skip", StageStatus.skipped),
        ("fail", StageStatus.blocked),  # gerileme insan araştırması bekler
    ],
)
def test_delegate_maps_verdict_to_status(
    monkeypatch: pytest.MonkeyPatch, verdict: str, expected: StageStatus
) -> None:
    monkeypatch.setattr(
        reg_mod.RegressionGuard, "run", lambda self: RegressionResult(verdict, "özet", [])
    )
    res = delegates.regression(_ctx())
    assert res.status == expected
    assert res.output["verdict"] == verdict
    assert res.message == "özet"
