"""İnsan denetimi: dengeli kör örnek, uyum/kappa, false PASS ayrı raporu; karşılaştırma raporu."""

from __future__ import annotations

import json
from pathlib import Path

from app.evals.profile.comparison import compare_profiles
from app.evals.profile.eval_registry import EvalRegistry
from app.evals.profile.eval_runner import load_eval_config, run_eval
from app.evals.profile.generators import SystemSpec
from app.evals.profile.human_audit import compute_agreement, export_audit, stratified_sample
from app.evals.profile.schema import ItemScore

from .mix_helpers import FakeChunk, FakeRetriever
from .test_eval_runner import ITEMS, _gens, snapshot


def _scores() -> list[ItemScore]:
    out = []
    for d in ("math", "statistics", "reasoning", "trading", "coding"):
        for i in range(30):
            out.append(
                ItemScore(
                    item_id=f"{d}{i}",
                    domain=d,
                    system="D",
                    correct=i % 2 == 0,
                    score=float(i % 2 == 0),
                    answer=f"ans {d}{i}",
                )
            )
    return out


def test_stratified_sample_is_balanced_and_deterministic() -> None:
    s1 = stratified_sample(_scores(), 100, seed=42)
    s2 = stratified_sample(_scores(), 100, seed=42)
    assert [x.item_id for x in s1] == [x.item_id for x in s2]
    counts: dict[str, int] = {}
    for x in s1:
        counts[x.domain] = counts.get(x.domain, 0) + 1
    assert set(counts.values()) == {20}


def test_export_is_blind(tmp_path: Path) -> None:
    paths = export_audit("run_x", _scores(), {}, sample_size=10, seed=1, out_dir=tmp_path)
    blind = [json.loads(x) for x in paths["blind"].read_text(encoding="utf-8").splitlines()]
    for row in blind:
        assert "automatic_decision" not in row and "automatic_score" not in row
        assert {"question", "gold_answer", "rag_chunk_ids", "model_answer"} <= set(row)
    key = [json.loads(x) for x in paths["key"].read_text(encoding="utf-8").splitlines()]
    assert all("automatic_decision" in k for k in key)


def test_agreement_and_false_pass() -> None:
    key = [
        {"audit_id": "a1", "item_id": "i1", "system": "D", "automatic_decision": "PASS"},
        {"audit_id": "a2", "item_id": "i2", "system": "D", "automatic_decision": "PASS"},
        {"audit_id": "a3", "item_id": "i3", "system": "D", "automatic_decision": "FAIL"},
        {"audit_id": "a4", "item_id": "i4", "system": "D", "automatic_decision": "FAIL"},
        {"audit_id": "a5", "item_id": "i5", "system": "D", "automatic_decision": "PASS"},
    ]
    blind = [
        {"audit_id": "a1", "human_decision": "PASS"},
        {"audit_id": "a2", "human_decision": "FAIL", "error_category": "hallucination"},
        {"audit_id": "a3", "human_decision": "FAIL"},
        {"audit_id": "a4", "human_decision": "pass"},
        {"audit_id": "a5", "human_decision": "PARTIAL", "error_category": "math_failure"},
    ]
    rep = compute_agreement(blind, key)
    assert rep.n_reviewed == 5 and rep.agreement == 0.4
    assert rep.false_pass == 2 and rep.false_pass_rate == round(2 / 3, 4)
    assert {x["item_id"] for x in rep.false_pass_items} == {"i2", "i5"}
    assert rep.false_fail == 1 and rep.partial_count == 1
    assert rep.error_categories == {"hallucination": 1, "math_failure": 1}


def test_unreviewed_rows_are_ignored() -> None:
    rep = compute_agreement(
        [{"audit_id": "a1", "human_decision": ""}],
        [{"audit_id": "a1", "item_id": "i", "system": "D", "automatic_decision": "PASS"}],
    )
    assert rep.n_reviewed == 0 and rep.agreement is None


def test_comparison_report_csv_json_md(tmp_path: Path) -> None:
    base, lora = _gens()
    reg = EvalRegistry(tmp_path / "runs")
    systems = [
        SystemSpec("A_base", "Base", base, use_rag=False),
        SystemSpec("B_base_rag", "Base + RAG", base, use_rag=True),
        SystemSpec("D_rag+balanced_v1_svd", "Base + RAG + Balanced Profile", lora, use_rag=True),
        SystemSpec(
            "C_adapter_math_v1",
            "Base + math LoRA",
            None,
            use_rag=False,
            unavailable_reason="serving_model tanımlı değil",
        ),
    ]
    run_eval(
        ITEMS,
        systems,
        eval_config=load_eval_config(),
        rag_snapshot=snapshot(),
        retriever=FakeRetriever([FakeChunk("c1", "p1", "x")]),
        registry=reg,
        dataset_label="validation@x",
        model_meta={"profile": "balanced_v1"},
        resource_fn=dict,
    )
    res = compare_profiles(["balanced_v1", "statistics_v1"], registry=reg, out_dir=tmp_path / "rep")
    assert res["missing_profiles"] == ["statistics_v1"] and res["rows"] == 4
    md = Path(res["paths"]["md"]).read_text(encoding="utf-8")
    assert (
        "| Configuration | Math | Stats | Reasoning | Trading | Coding | Grounding | Hallucination"
        in md
    )
    assert "Base + RAG + Balanced Profile" in md
    assert "Koşulamayan" in md and "serving_model" in md
    assert Path(res["paths"]["csv"]).read_text(encoding="utf-8").startswith("run_id,profile")
    assert len(json.loads(Path(res["paths"]["json"]).read_text(encoding="utf-8"))) == 4
