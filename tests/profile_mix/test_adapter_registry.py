"""Domain adapter kayıt defteri: bağımsızlık, izlenebilirlik, production onayı."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.lora.domain_adapter_registry import (
    AdapterRegistryError,
    DomainAdapterRegistry,
    EvalStatus,
    ProductionStatus,
    read_peft_adapter_config,
)

from .mix_helpers import make_record


def test_register_and_roundtrip_keeps_all_fields(adapter_reg: DomainAdapterRegistry) -> None:
    rec = make_record("math")
    adapter_reg.register(rec)
    got = adapter_reg.get("math_lora", "v1")
    assert got is not None
    for field in (
        "adapter_id",
        "adapter_version",
        "base_model",
        "base_model_revision",
        "base_model_hash",
        "tokenizer_hash",
        "dataset_version",
        "dataset_hash",
        "training_config_hash",
        "domain",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "learning_rate",
        "epochs",
        "max_seq_length",
        "training_seed",
        "created_at",
        "git_commit",
        "eval_status",
        "production_status",
    ):
        assert getattr(got, field) == getattr(rec, field), field


def test_adapter_trained_on_top_of_another_is_rejected(adapter_reg: DomainAdapterRegistry) -> None:
    with pytest.raises(AdapterRegistryError, match="üstünden"):
        adapter_reg.register(make_record("statistics", parent_adapter="math_lora@v1"))


def test_adapter_files_must_be_independent(adapter_reg: DomainAdapterRegistry) -> None:
    adapter_reg.register(make_record("math"))
    with pytest.raises(AdapterRegistryError, match="bağımsız"):
        adapter_reg.register(make_record("statistics", adapter_path="models/adapters/math_v1"))


def test_duplicate_version_rejected(adapter_reg: DomainAdapterRegistry) -> None:
    adapter_reg.register(make_record("math"))
    with pytest.raises(AdapterRegistryError, match="zaten"):
        adapter_reg.register(make_record("math", adapter_path="elsewhere"))


def test_missing_reproducibility_hash_rejected(adapter_reg: DomainAdapterRegistry) -> None:
    with pytest.raises(AdapterRegistryError, match="dataset_hash"):
        adapter_reg.register(make_record("math", dataset_hash=""))


def test_new_record_can_not_start_as_production(adapter_reg: DomainAdapterRegistry) -> None:
    adapter_reg.register(make_record("math", production_status=ProductionStatus.PRODUCTION))
    got = adapter_reg.get("math_lora")
    assert got is not None and got.production_status is ProductionStatus.NONE


def test_production_requires_user_approval_and_passed_eval(
    adapter_reg: DomainAdapterRegistry,
) -> None:
    adapter_reg.register(make_record("math"))
    # eval geçmeden onaylı bile olsa production olamaz
    assert not adapter_reg.set_production_status(
        "math_lora", "v1", ProductionStatus.PRODUCTION, user_approved=True
    )
    adapter_reg.set_eval_status("math_lora", "v1", EvalStatus.PASSED)
    assert not adapter_reg.set_production_status("math_lora", "v1", ProductionStatus.PRODUCTION)
    assert adapter_reg.set_production_status(
        "math_lora", "v1", ProductionStatus.PRODUCTION, user_approved=True
    )


def test_single_production_per_domain(adapter_reg: DomainAdapterRegistry) -> None:
    for v in ("v1", "v2"):
        adapter_reg.register(make_record("math", v, adapter_path=f"p/{v}"))
        adapter_reg.set_eval_status("math_lora", v, EvalStatus.PASSED)
        adapter_reg.set_production_status(
            "math_lora", v, ProductionStatus.PRODUCTION, user_approved=True
        )
    statuses = {r.adapter_version: r.production_status for r in adapter_reg.records()}
    assert statuses == {"v1": ProductionStatus.DEPRECATED, "v2": ProductionStatus.PRODUCTION}
    latest = adapter_reg.latest_for_domain("math")
    assert latest is not None and latest.adapter_version == "v2"


def test_failed_eval_adapter_not_selected(adapter_reg: DomainAdapterRegistry) -> None:
    adapter_reg.register(make_record("math"))
    adapter_reg.set_eval_status("math_lora", "v1", EvalStatus.FAILED)
    assert adapter_reg.latest_for_domain("math") is None


def test_read_peft_adapter_config(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["v_proj", "q_proj"],
                "base_model_name_or_path": "Qwen/Qwen3-4B",
                "revision": "main",
            }
        ),
        encoding="utf-8",
    )
    cfg = read_peft_adapter_config(tmp_path)
    assert cfg["r"] == 8 and cfg["target_modules"] == ["q_proj", "v_proj"]
    assert cfg["base_model"] == "Qwen/Qwen3-4B"
