"""Eval runner: A/B/C/D aynı koşullarda, aynı RAG bağlamı, koşulamayan sistem raporu."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evals.profile.eval_registry import EvalRegistry
from app.evals.profile.eval_runner import load_eval_config, run_eval
from app.evals.profile.generators import SystemSpec
from app.evals.profile.schema import EvalItem
from app.memory.rag_version import RagConfigDrift, RagSnapshot

from .mix_helpers import FakeChunk, FakeGenerator, FakeRetriever

ITEMS = [
    EvalItem(
        id="m1",
        domain="math",
        question="Q-MATH 2+2 kaçtır?",
        expected_numeric_value=4,
        numeric_tolerance=1e-6,
    ),
    EvalItem(
        id="s1",
        domain="statistics",
        question="Q-STAT ortalama?",
        expected_numeric_value=8,
        numeric_tolerance=1e-6,
    ),
    EvalItem(
        id="r1",
        domain="reasoning",
        question="Q-RAG kaynak sorusu",
        requires_rag=True,
        required_claims=["momentum"],
        expected_chunk_ids=["c1"],
        expected_document_ids=["p1"],
    ),
    EvalItem(
        id="a1",
        domain="trading",
        question="Q-ABSTAIN bilinmeyen",
        requires_abstention=True,
        requires_rag=True,
    ),
]


def snapshot(**over: object) -> RagSnapshot:
    base: dict[str, object] = {
        "embedding_model": "nomic-embed-text",
        "embedding_model_version": "v1.5",
        "reranker": "heuristic",
        "reranker_version": "builtin",
        "top_k": 2,
        "overfetch": 4,
        "hybrid": True,
        "rrf": False,
        "graph": False,
        "router": False,
        "contextual_embed": False,
        "chunk_size": 1200,
        "chunk_overlap": 200,
        "index_hash": "idx1",
    }
    base.update(over)
    return RagSnapshot(**base)  # type: ignore[arg-type]


def _gens() -> tuple[FakeGenerator, FakeGenerator]:
    base = FakeGenerator(
        "base",
        {
            "Q-MATH": "Cevap: 5",  # yanlış
            "Q-STAT": "Cevap: 8",
            "Q-RAG": "Bilinmiyor ama sanırım trend.",
            "Q-ABSTAIN": "Fiyat 123 olacak.",  # uydurma → halüsinasyon
        },
    )
    lora = FakeGenerator(
        "lora",
        {
            "Q-MATH": "2+2=4\nCevap: 4",
            "Q-STAT": "Cevap: 7",  # LoRA burada kötüleşiyor
            "Q-RAG": "Momentum etkisi [p1:c1] kaynağında anlatılır.",
            "Q-ABSTAIN": "Kaynaklarda bu bilgi yok, cevaplayamam.",
        },
    )
    return base, lora


def _systems(base: FakeGenerator, lora: FakeGenerator) -> list[SystemSpec]:
    return [
        SystemSpec("A_base", "Base", base, use_rag=False),
        SystemSpec("B_base_rag", "Base + RAG", base, use_rag=True),
        SystemSpec(
            "C_p",
            "Base + P",
            lora,
            use_rag=False,
            lora="p",
            lora_kind="profile",
            base_counterpart="A_base",
        ),
        SystemSpec(
            "D_rag+p",
            "Base + RAG + P",
            lora,
            use_rag=True,
            lora="p",
            lora_kind="profile",
            base_counterpart="B_base_rag",
        ),
        SystemSpec(
            "C_missing",
            "Base + Q",
            None,
            use_rag=False,
            unavailable_reason="serving_model tanımlı değil",
        ),
    ]


def test_four_systems_same_conditions(tmp_path: Path) -> None:
    base, lora = _gens()
    retriever = FakeRetriever(
        [FakeChunk("c1", "p1", "momentum metni"), FakeChunk("c9", "p9", "gürültü")]
    )
    cfg = load_eval_config()
    out = run_eval(
        ITEMS,
        _systems(base, lora),
        eval_config=cfg,
        rag_snapshot=snapshot(),
        retriever=retriever,
        registry=EvalRegistry(tmp_path / "runs"),
        resource_fn=lambda: {"ram_mb": 1000, "vram_mb": None},
    )
    # Retrieval kalem başına BİR kez (B ve D aynı bağlamı paylaşır).
    assert len(retriever.calls) == len(ITEMS)
    # Aynı system prompt + aynı üretim parametreleri tüm çağrılarda.
    all_calls = base.calls + lora.calls
    assert {c[1] for c in all_calls} == {cfg["system_prompt"]}
    assert len({c[2] for c in all_calls}) == 1
    # B ve D'ye giden prompt'lar birebir aynı (yalnız model farklı).
    b_prompts = [c[0] for c in base.calls if "BAĞLAM" in c[0]]
    d_prompts = [c[0] for c in lora.calls if "BAĞLAM" in c[0]]
    assert b_prompts == d_prompts and len(b_prompts) == len(ITEMS)

    assert out.not_run == {"C_missing": "serving_model tanımlı değil"}
    assert set(out.metrics) == {"A_base", "B_base_rag", "C_p", "D_rag+p"}
    a, d = out.metrics["A_base"], out.metrics["D_rag+p"]
    assert a["math_accuracy"] == 0.0 and d["math_accuracy"] == 1.0
    assert a["statistics_accuracy"] == 1.0 and d["statistics_accuracy"] == 0.0
    assert a["hallucination_rate"] > d["hallucination_rate"]
    assert d["grounding"] == 1.0 and d["citation_accuracy"] == 1.0
    assert out.metrics["A_base"]["grounding"] is None  # RAG'sız sistemde grounding yok
    assert d["retrieval_recall_at_k"] == 1.0 and d["retrieval_precision"] == 0.5
    # Overall ayrı; ham metrikler her zaman mevcut.
    assert out.overall["D_rag+p"]["overall"] is not None
    assert "note" in out.overall["D_rag+p"]


def test_failure_diagnosis_attributes_cause() -> None:
    base, lora = _gens()
    retriever = FakeRetriever([FakeChunk("c1", "p1", "momentum")])
    out = run_eval(
        ITEMS,
        _systems(base, lora),
        eval_config=load_eval_config(),
        rag_snapshot=snapshot(),
        retriever=retriever,
        resource_fn=dict,
    )
    by = {(s.system, s.item_id): s for s in out.items}
    # LoRA istatistikte yanlış, base doğru → LoRA kaynaklı
    assert "LoRA_reasoning_failure" in by[("C_p", "s1")].failure_categories
    assert "statistics_failure" in by[("C_p", "s1")].failure_categories
    # base çekimser kalması gerekirken uydurdu
    assert "hallucination" in by[("A_base", "a1")].failure_categories
    assert by[("A_base", "m1")].failure_categories[:1] == ["math_failure"]


def test_rag_config_drift_invalidates_run() -> None:
    base, lora = _gens()
    with pytest.raises(RagConfigDrift):
        run_eval(
            ITEMS,
            _systems(base, lora),
            eval_config=load_eval_config(),
            rag_snapshot=snapshot(),
            retriever=FakeRetriever([]),
            rag_snapshot_fn=lambda: snapshot(top_k=10),
            resource_fn=dict,
        )


def test_rag_system_without_retriever_is_reported_not_silently_zero() -> None:
    base, lora = _gens()
    out = run_eval(
        ITEMS,
        _systems(base, lora),
        eval_config=load_eval_config(),
        rag_snapshot=snapshot(),
        retriever=None,
        resource_fn=dict,
    )
    assert "B_base_rag" in out.not_run and "D_rag+p" in out.not_run
    assert "B_base_rag" not in out.metrics


def test_registry_roundtrip(tmp_path: Path) -> None:
    base, lora = _gens()
    reg = EvalRegistry(tmp_path / "runs")
    out = run_eval(
        ITEMS,
        _systems(base, lora),
        eval_config=load_eval_config(),
        rag_snapshot=snapshot(),
        retriever=FakeRetriever([FakeChunk("c1", "p1", "x")]),
        registry=reg,
        dataset_label="validation@abc",
        model_meta={"profile": "p"},
        resource_fn=dict,
    )
    stored = reg.load(out.manifest["run_id"])
    assert stored.manifest["replay_key"] == out.manifest["replay_key"]
    assert len(stored.items) == len(out.items)
    assert reg.latest_for_profile("p", "validation") is not None
    assert reg.latest_for_profile("p", "golden_test") is None


def test_cli_dry_run_calls_no_llm(capsys: pytest.CaptureFixture[str]) -> None:
    from app.evals.profile import eval_runner

    rc = eval_runner.main(["--profile", "balanced_v1", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and '"dry_run": true' in out and '"n_items": 27' in out


def test_cli_golden_requires_final_and_validated(capsys: pytest.CaptureFixture[str]) -> None:
    from app.evals.profile import eval_runner

    assert (
        eval_runner.main(["--profile", "balanced_v1", "--split", "golden_test", "--dry-run"]) == 2
    )
    assert (
        eval_runner.main(
            ["--profile", "balanced_v1", "--split", "golden_test", "--final", "--dry-run"]
        )
        == 2
    )  # profil registry'de validated değil
