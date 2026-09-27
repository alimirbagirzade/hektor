"""Profil kurucu: uyumluluk kontrolleri, merge yöntemleri, dry-run, registry kaydı."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from app.lora.domain_adapter_registry import DomainAdapterRegistry
from app.lora.mix_common import MixConfigError, load_mix_config
from app.lora.profile_builder import (
    BuildPlan,
    build_profile,
    plan_profile,
    resolve_merge_parameters,
)
from app.lora.profile_registry import ProfileRegistry, ProfileRegistryError, ProfileStatus

from .mix_helpers import make_record


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[BuildPlan, Path]] = []

    def merge(self, plan: BuildPlan, output_dir: Path) -> Path:
        self.calls.append((plan, output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


def test_plan_ok_for_all_methods(full_adapter_reg: DomainAdapterRegistry) -> None:
    for method in ("svd", "ties", "dare_ties", "linear"):
        plan = plan_profile("balanced_v1", method, adapter_registry=full_adapter_reg)
        assert plan.ok, plan.errors
        assert plan.profile_id == f"balanced_v1_{method}"
        assert set(plan.adapters) == {"math", "statistics", "reasoning", "trading", "coding"}
        assert plan.weights["math"] == pytest.approx(0.30)


def test_same_profile_different_methods_have_different_hashes(
    full_adapter_reg: DomainAdapterRegistry,
) -> None:
    hashes = {
        plan_profile("balanced_v1", m, adapter_registry=full_adapter_reg).profile_hash
        for m in ("svd", "ties", "dare_ties", "linear")
    }
    assert len(hashes) == 4


def test_zero_weight_domain_is_skipped(full_adapter_reg: DomainAdapterRegistry) -> None:
    cfg = copy.deepcopy(load_mix_config())
    cfg["profiles"]["no_code"] = {**cfg["profiles"]["balanced_v1"], "coding": 0.0}
    plan = plan_profile("no_code", "svd", adapter_registry=full_adapter_reg, config=cfg)
    assert plan.ok and "coding" not in plan.adapters


def test_missing_adapter_blocks(adapter_reg: DomainAdapterRegistry) -> None:
    adapter_reg.register(make_record("math"))
    plan = plan_profile("balanced_v1", "svd", adapter_registry=adapter_reg)
    assert not plan.ok
    assert any("statistics" in e for e in plan.errors)


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"base_model_hash": "x" * 64}, "base model hash"),
        ({"base_model": "other/model"}, "base model adı"),
        ({"tokenizer_hash": "y" * 64}, "tokenizer"),
        ({"target_modules": ["q_proj", "v_proj"]}, "target_modules"),
    ],
)
def test_incompatible_adapter_blocks(
    adapter_reg: DomainAdapterRegistry, override: dict[str, Any], needle: str
) -> None:
    for d in ("math", "statistics", "reasoning", "trading"):
        adapter_reg.register(make_record(d))
    adapter_reg.register(make_record("coding", **override))
    plan = plan_profile("balanced_v1", "svd", adapter_registry=adapter_reg)
    assert not plan.ok
    assert any(needle in e for e in plan.errors), plan.errors


def test_rank_mismatch_blocks_same_rank_methods_but_not_svd(
    adapter_reg: DomainAdapterRegistry,
) -> None:
    for d in ("math", "statistics", "reasoning", "trading"):
        adapter_reg.register(make_record(d))
    adapter_reg.register(make_record("coding", r=32, lora_alpha=64))
    for method in ("linear", "ties", "dare_ties"):
        plan = plan_profile("balanced_v1", method, adapter_registry=adapter_reg)
        assert not plan.ok and any("aynı r" in e for e in plan.errors)
    svd = plan_profile("balanced_v1", "svd", adapter_registry=adapter_reg)
    assert svd.ok and svd.merge_parameters["svd_rank"] == 32 and svd.warnings


def test_merge_parameters() -> None:
    cfg = load_mix_config()
    assert resolve_merge_parameters("dare_ties", cfg)["seed"] == 42  # DARE seed'li
    assert resolve_merge_parameters("ties", cfg, {"density": 0.3})["density"] == 0.3
    with pytest.raises(MixConfigError):
        resolve_merge_parameters("ties", cfg, {"density": 1.5})
    with pytest.raises(MixConfigError):
        resolve_merge_parameters("magic", cfg)


def test_dry_run_writes_nothing(
    full_adapter_reg: DomainAdapterRegistry, profile_reg: ProfileRegistry, tmp_path: Path
) -> None:
    backend = FakeBackend()
    plan = plan_profile("balanced_v1", "svd", adapter_registry=full_adapter_reg)
    res = build_profile(
        plan, backend=backend, profile_registry=profile_reg, output_dir=tmp_path / "out"
    )
    assert res["status"] == "dry_run"
    assert backend.calls == [] and profile_reg.records() == []
    assert not (tmp_path / "out").exists()


def test_blocked_plan_never_merges(
    adapter_reg: DomainAdapterRegistry, profile_reg: ProfileRegistry
) -> None:
    backend = FakeBackend()
    plan = plan_profile("balanced_v1", "svd", adapter_registry=adapter_reg)  # boş registry
    res = build_profile(plan, run=True, backend=backend, profile_registry=profile_reg)
    assert res["status"] == "blocked" and backend.calls == []


def test_run_registers_experimental_profile(
    full_adapter_reg: DomainAdapterRegistry, profile_reg: ProfileRegistry, tmp_path: Path
) -> None:
    plan = plan_profile("statistics_v1", "ties", adapter_registry=full_adapter_reg)
    res = build_profile(
        plan,
        run=True,
        backend=FakeBackend(),
        profile_registry=profile_reg,
        output_dir=tmp_path / "statistics_v1_ties",
        rag_version="rag-abc",
    )
    assert res["status"] == "built"
    rec = profile_reg.get("statistics_v1_ties")
    assert rec is not None
    assert rec.status is ProfileStatus.EXPERIMENTAL  # kurulum = deney; validated DEĞİL
    assert rec.merge_method == "ties" and rec.merge_parameters["density"] == 0.5
    assert rec.adapter_versions["statistics"] == "statistics_lora@v1"
    assert rec.adapter_weights["statistics"] == pytest.approx(0.35)
    assert rec.rag_version == "rag-abc" and rec.profile_hash == plan.profile_hash


def test_same_profile_id_with_changed_content_is_rejected(
    full_adapter_reg: DomainAdapterRegistry, profile_reg: ProfileRegistry, tmp_path: Path
) -> None:
    plan = plan_profile("balanced_v1", "svd", adapter_registry=full_adapter_reg)
    build_profile(
        plan,
        run=True,
        backend=FakeBackend(),
        profile_registry=profile_reg,
        output_dir=tmp_path / "a",
    )
    plan2 = plan_profile(
        "balanced_v1", "svd", adapter_registry=full_adapter_reg, merge_overrides={"svd_clamp": 0.9}
    )
    with pytest.raises(ProfileRegistryError, match="sürüm"):
        build_profile(
            plan2,
            run=True,
            backend=FakeBackend(),
            profile_registry=profile_reg,
            output_dir=tmp_path / "b",
        )


def test_profile_builder_cli_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    from app.lora import profile_builder

    # Gerçek (boş) registry → plan bloklanır ama komut çökmez, dry-run'da hiçbir şey yazmaz.
    rc = profile_builder.main(["--profile", "balanced_v1", "--merge-method", "svd"])
    out = capsys.readouterr().out
    assert rc == 1 and '"status": "blocked"' in out


# ------------------------------------------------------------------ gerçek PEFT (opsiyonel)


def _tiny_llama(tmp: Path) -> Path:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg)
    out = tmp / "tiny_base"
    model.save_pretrained(out)
    return out


def _tiny_adapter(base_dir: Path, out: Path, seed: int, r: int = 4) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True)
    cfg = LoraConfig(
        r=r, lora_alpha=2 * r, target_modules=["q_proj", "v_proj"], init_lora_weights=False
    )
    peft_model = get_peft_model(base, cfg)
    peft_model.save_pretrained(out)
    return out


@pytest.mark.parametrize("method", ["svd", "linear", "ties", "dare_ties"])
def test_real_peft_weighted_merge_keeps_base_unchanged(
    tmp_path: Path, profile_reg: ProfileRegistry, method: str
) -> None:
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    from app.lora.profile_builder import PeftMergeBackend

    base_dir = _tiny_llama(tmp_path)
    base_before = load_file(str(base_dir / "model.safetensors"))
    reg = DomainAdapterRegistry(tmp_path / "ad.jsonl")
    for i, d in enumerate(("math", "statistics", "reasoning", "trading", "coding")):
        path = _tiny_adapter(base_dir, tmp_path / f"{d}_lora", seed=i + 1)
        reg.register(
            make_record(
                d,
                adapter_path=str(path),
                base_model=str(base_dir),
                target_modules=["q_proj", "v_proj"],
                r=4,
                lora_alpha=8,
            )
        )
    plan = plan_profile("balanced_v1", method, adapter_registry=reg)
    assert plan.ok, plan.errors
    res = build_profile(
        plan,
        run=True,
        backend=PeftMergeBackend(str(base_dir)),
        profile_registry=profile_reg,
        output_dir=tmp_path / "merged",
    )
    assert res["status"] == "built"
    merged = Path(res["merged_adapter_path"])
    assert (merged / "adapter_config.json").exists()
    # Base ağırlıkları bayt-bayt aynı (merge_and_unload YOK).
    base_after = load_file(str(base_dir / "model.safetensors"))
    assert base_before.keys() == base_after.keys()
    assert all(torch.equal(base_before[k], base_after[k]) for k in base_before)
    # Birleşik adapter gerçekten yüklenip üretim yapabiliyor.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    m = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True), str(merged)
    )
    out = m(torch.tensor([[1, 2, 3]])).logits
    assert out.shape == (1, 3, 64) and torch.isfinite(out).all()
