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


def test_same_hash_upsert_keeps_eval_history_and_serving_model(tmp_path: Path) -> None:
    """Aynı içerikle yeniden kayıt (ör. build --run tekrarı) validated profili boşaltmamalı."""
    reg = ProfileRegistry(tmp_path / "p.jsonl")
    reg.upsert(_rec("a"))
    reg.transition("a", ProfileStatus.EVALUATING)
    reg.record_eval(
        "a",
        run_id="run1",
        eval_dataset_version="golden_test@x",
        eval_score=0.8,
        domain_scores={"math": 0.9},
        grounding_score=0.9,
        hallucination_score=0.05,
    )
    reg.transition("a", ProfileStatus.VALIDATED, note="ilk eval")
    rows = reg.records()
    rows[0].serving_model = "hektor-a:latest"
    from app.lora.mix_common import write_jsonl

    write_jsonl(reg.path, (r.to_dict() for r in rows))

    again = _rec("a")  # builder'ın ürettiği gibi: eval alanları/servis modeli boş
    again.merged_adapter_path = "models/profiles/a_yeni"
    reg.upsert(again)
    rec = reg.get("a")
    assert rec is not None
    assert rec.status is ProfileStatus.VALIDATED
    assert rec.last_eval_run_id == "run1" and rec.eval_score == 0.8
    assert rec.domain_scores == {"math": 0.9} and rec.grounding_score == 0.9
    assert rec.eval_dataset_version == "golden_test@x"
    assert rec.serving_model == "hektor-a:latest"
    assert rec.merged_adapter_path == "models/profiles/a_yeni"
    assert "ilk eval" in rec.notes
    # Durum makinesi tutarlı kalır: validated → evaluating → validated yine mümkün.
    reg.transition("a", ProfileStatus.EVALUATING)
    reg.transition("a", ProfileStatus.VALIDATED)


# ------------------------------------------------ production referansı karşılaştırılabilirliği

_MANIFEST = {
    "eval_dataset": "golden_test@x",
    "eval_dataset_hash": "ds" * 8,
    "rag_version": "rag" * 5,
    "base_model_hash": "base" * 4,
    "eval_config_hash": "cfg" * 5,
}


def _gate_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cand_over: dict) -> dict:
    from app.config import get_settings
    from app.evals.profile.eval_registry import EvalRegistry
    from app.lora.mix_cli import compute_gate

    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    get_settings.cache_clear()
    preg = ProfileRegistry()
    for pid in ("p", "c"):
        preg.upsert(_rec(pid))
    preg.transition("p", ProfileStatus.EVALUATING)
    preg.record_eval(
        "p",
        run_id="run_p",
        eval_dataset_version="golden_test@x",
        eval_score=0.8,
        domain_scores={},
        grounding_score=0.9,
        hallucination_score=0.05,
    )
    preg.transition("p", ProfileStatus.VALIDATED)
    preg.transition("p", ProfileStatus.PRODUCTION, gate_passed=True, user_approved=True)
    ereg = EvalRegistry()
    old = metrics(0.90, 0.70, 0.90, 0.85, 0.80)
    new = metrics(0.90, 0.75, 0.90, 0.86, 0.80, grounding=0.92, hallucination_rate=0.04)
    ereg.save({**_MANIFEST, "run_id": "run_p", "profile": "p"}, {"D_rag+p": old}, [])
    ereg.save({**_MANIFEST, **cand_over, "run_id": "run_c", "profile": "c"}, {"D_rag+c": new}, [])
    return compute_gate("c")


def test_gate_passes_when_production_run_is_comparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    res = _gate_setup(tmp_path, monkeypatch, {})
    assert res["passed"] is True, res
    assert res["reference_run_id"] == "run_p" and res["candidate_run_id"] == "run_c"


@pytest.mark.parametrize(
    "over",
    [
        {"eval_dataset_hash": "baska"},
        {"rag_version": "baska"},
        {"base_model_hash": "baska"},
        {"eval_config_hash": "baska"},
        {"base_model_hash": "unknown"},
        {"rag_version": ""},
    ],
)
def test_gate_fails_closed_on_incomparable_production_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, over: dict
) -> None:
    """Farklı eval seti/RAG/base/config ile koşulmuş production referansı kıyaslanamaz."""
    res = _gate_setup(tmp_path, monkeypatch, over)
    assert res["passed"] is False
    (key,) = over
    assert any(b.startswith(key) for b in res["blockers"]), res
    assert "karşılaştırılamaz" in res["error"]
