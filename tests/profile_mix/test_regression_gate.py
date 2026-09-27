"""Regression kapısı: overall yükselse bile domain regression'ı terfiyi engeller."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evals.profile.eval_runner import load_eval_config
from app.evals.profile.regression_eval import evaluate_regression_gate
from app.evals.profile.score_aggregator import overall_score
from app.lora.profile_registry import (
    ProfileRecord,
    ProfileRegistry,
    ProfileRegistryError,
    ProfileStatus,
)

RULES = load_eval_config()["regression_gate"]
N = {"n_by_domain": dict.fromkeys(("math", "statistics", "reasoning", "trading", "coding"), 20)}


def metrics(math: float, stats: float, reas: float, trad: float, code: float, **kw: float) -> dict:
    return {
        "math_accuracy": math,
        "statistics_accuracy": stats,
        "reasoning_accuracy": reas,
        "trading_accuracy": trad,
        "coding_accuracy": code,
        "grounding": kw.get("grounding", 0.9),
        "hallucination_rate": kw.get("hallucination_rate", 0.05),
        "abstention_accuracy": kw.get("abstention_accuracy", 0.9),
        **N,
    }


def test_spec_example_math_regression_blocks_despite_higher_overall() -> None:
    old = metrics(0.92, 0.70, 0.90, 0.85, 0.80)
    new = metrics(0.75, 0.94, 0.89, 0.85, 0.81)
    w = load_eval_config()["eval_weights"]
    o_old, o_new = overall_score(old, w)["overall"], overall_score(new, w)["overall"]
    assert o_new > o_old  # overall YÜKSELDİ …
    res = evaluate_regression_gate(
        new, old, RULES, candidate_overall=o_new, reference_overall=o_old
    )
    assert not res.passed  # … ama terfi YOK
    blocked = {(b.metric, b.rule) for b in res.blockers}
    assert ("math_accuracy", "max_drop") in blocked
    assert res.overall_delta is not None and res.overall_delta > 0
    # reasoning -0.01: izin içinde ama raporlanır
    assert any(r.metric == "reasoning_accuracy" for r in res.regressions)


def test_clean_improvement_passes() -> None:
    old = metrics(0.90, 0.70, 0.90, 0.85, 0.80)
    new = metrics(0.90, 0.75, 0.90, 0.86, 0.80, grounding=0.92, hallucination_rate=0.04)
    assert evaluate_regression_gate(new, old, RULES).passed


@pytest.mark.parametrize(
    ("override", "metric"),
    [
        ({"statistics_accuracy": 0.69}, "statistics_accuracy"),
        ({"grounding": 0.85}, "grounding"),
        ({"hallucination_rate": 0.06}, "hallucination_rate"),
        ({"abstention_accuracy": 0.7}, "abstention_accuracy"),
    ],
)
def test_each_rule_blocks(override: dict, metric: str) -> None:
    old = metrics(0.90, 0.70, 0.90, 0.85, 0.80)
    new = {**metrics(0.90, 0.70, 0.90, 0.85, 0.80), **override}
    res = evaluate_regression_gate(new, old, RULES)
    assert not res.passed and metric in {b.metric for b in res.blockers}


def test_unmeasured_metric_blocks() -> None:
    old = metrics(0.9, 0.7, 0.9, 0.85, 0.8)
    new = {**old, "coding_accuracy": None}
    res = evaluate_regression_gate(new, old, RULES)
    assert not res.passed and any(b.detail == "metrik ölçülemedi" for b in res.blockers)


def test_small_sample_blocks() -> None:
    old = metrics(0.9, 0.7, 0.9, 0.85, 0.8)
    new = {**metrics(0.95, 0.8, 0.9, 0.9, 0.85), "n_by_domain": {"math": 3}}
    res = evaluate_regression_gate(new, old, RULES)
    assert not res.passed and any(b.rule == "min_items" for b in res.blockers)


def _rec(pid: str) -> ProfileRecord:
    return ProfileRecord(
        profile_id=pid,
        profile_name=pid,
        profile_version="1",
        base_model_version="b",
        rag_version="r",
        adapter_versions={},
        adapter_weights={},
        merge_method="svd",
        merge_parameters={},
        profile_hash="h-" + pid,
    )


def test_production_requires_gate_and_explicit_approval(tmp_path: Path) -> None:
    reg = ProfileRegistry(tmp_path / "p.jsonl")
    reg.upsert(_rec("a"))
    with pytest.raises(ProfileRegistryError):  # experimental → production yasak
        reg.transition("a", ProfileStatus.PRODUCTION, gate_passed=True, user_approved=True)
    reg.transition("a", ProfileStatus.EVALUATING)
    with pytest.raises(ProfileRegistryError, match="eval"):
        reg.transition("a", ProfileStatus.VALIDATED)  # eval koşusu yok
    reg.record_eval(
        "a",
        run_id="run1",
        eval_dataset_version="golden_test@x",
        eval_score=0.8,
        domain_scores={},
        grounding_score=0.9,
        hallucination_score=0.05,
    )
    reg.transition("a", ProfileStatus.VALIDATED)
    with pytest.raises(ProfileRegistryError, match="gate"):
        reg.transition("a", ProfileStatus.PRODUCTION, user_approved=True)
    with pytest.raises(ProfileRegistryError, match="onay"):
        reg.transition("a", ProfileStatus.PRODUCTION, gate_passed=True)
    assert (
        reg.transition("a", ProfileStatus.PRODUCTION, gate_passed=True, user_approved=True).status
        is ProfileStatus.PRODUCTION
    )


def test_only_one_production_profile(tmp_path: Path) -> None:
    reg = ProfileRegistry(tmp_path / "p.jsonl")
    for pid in ("a", "b"):
        reg.upsert(_rec(pid))
        reg.transition(pid, ProfileStatus.EVALUATING)
        reg.record_eval(
            pid,
            run_id="r",
            eval_dataset_version="g",
            eval_score=0.5,
            domain_scores={},
            grounding_score=None,
            hallucination_score=None,
        )
        reg.transition(pid, ProfileStatus.VALIDATED)
        reg.transition(pid, ProfileStatus.PRODUCTION, gate_passed=True, user_approved=True)
    st = {r.profile_id: r.status for r in reg.records()}
    assert st == {"a": ProfileStatus.DEPRECATED, "b": ProfileStatus.PRODUCTION}
