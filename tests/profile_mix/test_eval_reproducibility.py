"""Tekrar-üretilebilirlik: aynı girdiler → aynı replay_key ve aynı sonuçlar."""

from __future__ import annotations

from app.evals.profile.eval_runner import load_eval_config, run_eval
from app.evals.profile.generators import SystemSpec
from app.evals.profile.manifest import RunManifest

from .mix_helpers import FakeChunk, FakeGenerator, FakeRetriever
from .test_eval_runner import ITEMS, _gens, snapshot

REQUIRED_MANIFEST_FIELDS = (
    "run_id",
    "timestamp",
    "git_commit",
    "base_model_hash",
    "tokenizer_hash",
    "quantization",
    "rag_version",
    "rag_index_hash",
    "embedding_model",
    "reranker",
    "adapter_versions",
    "profile",
    "merge_method",
    "weights",
    "eval_dataset_hash",
    "eval_config_hash",
    "generation_config",
    "seed",
    "hardware",
    "software_versions",
)


def _run(cfg: dict | None = None):
    base, lora = _gens()
    systems = [
        SystemSpec("A_base", "Base", base, use_rag=False),
        SystemSpec("D_rag+p", "RAG+P", lora, use_rag=True, base_counterpart="A_base"),
    ]
    return run_eval(
        ITEMS,
        systems,
        eval_config=cfg or load_eval_config(),
        rag_snapshot=snapshot(),
        retriever=FakeRetriever([FakeChunk("c1", "p1", "momentum")]),
        dataset_label="validation@abc",
        dataset_hash="abc",
        model_meta={"profile": "balanced_v1", "git_commit": "deadbeef", "weights": {"math": 0.3}},
        resource_fn=dict,
    )


def test_manifest_has_all_reproducibility_fields() -> None:
    m = _run().manifest
    for f in REQUIRED_MANIFEST_FIELDS:
        assert f in m, f
    assert m["seed"] == m["generation_config"]["seed"] == 42
    assert m["rag_version"] == snapshot().rag_version


def test_same_inputs_same_replay_key_and_results() -> None:
    a, b = _run(), _run()
    assert a.manifest["replay_key"] == b.manifest["replay_key"]
    strip = lambda out: [(s.system, s.item_id, s.correct, s.score, s.answer) for s in out.items]  # noqa: E731
    assert strip(a) == strip(b)
    assert a.metrics == b.metrics


def test_any_input_change_changes_replay_key() -> None:
    cfg = load_eval_config()
    cfg2 = {**cfg, "generation": {**cfg["generation"], "seed": 7}}
    assert _run().manifest["replay_key"] != _run(cfg2).manifest["replay_key"]


def test_rag_version_changes_with_any_retrieval_setting() -> None:
    assert snapshot().rag_version == snapshot().rag_version
    assert snapshot().rag_version != snapshot(reranker="flashrank").rag_version
    assert snapshot().rag_version != snapshot(index_hash="idx2").rag_version


def test_replay_key_ignores_volatile_fields() -> None:
    kw = dict.fromkeys(RunManifest.__dataclass_fields__, "x")
    kw.update(
        adapter_versions={},
        weights={},
        generation_config={},
        seed=1,
        systems=[],
        profile=None,
        merge_method=None,
    )
    m1 = RunManifest(**{**kw, "run_id": "r1", "timestamp": "t1"})  # type: ignore[arg-type]
    m2 = RunManifest(**{**kw, "run_id": "r2", "timestamp": "t2"})  # type: ignore[arg-type]
    assert m1.replay_key == m2.replay_key


def test_generator_receives_seed() -> None:
    gen = FakeGenerator("g", {"Q-MATH": "Cevap: 4"})
    run_eval(
        ITEMS[:1],
        [SystemSpec("A", "A", gen, use_rag=False)],
        eval_config=load_eval_config(),
        rag_snapshot=snapshot(),
        resource_fn=dict,
    )
    assert gen.calls[0][2].seed == 42 and gen.calls[0][2].temperature == 0.0
