"""Qwen3-30B-A3B (MoE) LoRA hazırlığı: hedef-modül uygulaması, eşleşme kontrolü, RAM tahmini.

ÇEVRİMDIŞI: torch/peft/model yüklemez; yalnız saf yardımcıları ve profil YAML'ını sınar.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training import peft_lora_train
from app.training.peft_lora_train import (
    ATTENTION_TARGET_MODULES,
    TARGET_MODULES,
    PeftTrainConfig,
    build_lora_kwargs,
    build_training_kwargs,
    check_cpu_ram,
    estimate_cpu_train_ram_gb,
    load_lora_profile,
    normalize_target_modules,
    recipe_summary,
    unmatched_target_modules,
)


def _cfg(**kw) -> PeftTrainConfig:
    base = {
        "base_model": "dummy",
        "train_jsonl": Path("t.jsonl"),
        "valid_jsonl": Path("v.jsonl"),
        "adapter_output_path": Path("out"),
    }
    base.update(kw)
    return PeftTrainConfig(**base)  # type: ignore[arg-type]


# Qwen3-MoE (transformers 5.x) modül adlarının temsili alt kümesi: uzmanlar birleşik
# parametre (Qwen3MoeExperts) → gate_proj/up_proj/down_proj MODÜL olarak yok; router `gate`.
_MOE_MODULES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp",
    "model.layers.0.mlp.experts",
    "model.layers.0.mlp.gate",
]
_DENSE_MODULES = [*_MOE_MODULES[:4], *(f"model.layers.0.mlp.{m}" for m in TARGET_MODULES[4:])]


def test_default_targets_unchanged() -> None:
    assert build_lora_kwargs(_cfg())["target_modules"] == list(TARGET_MODULES)


def test_moe_default_targets_detected_as_unmatched() -> None:
    # Eski sessiz davranışın kökü: 7'li liste MoE'de yalnız attention'a düşer.
    assert unmatched_target_modules(_MOE_MODULES, TARGET_MODULES) == [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_router_gate_is_not_gate_proj() -> None:
    assert unmatched_target_modules(["model.layers.0.mlp.gate"], ("gate_proj",)) == ["gate_proj"]


def test_attention_targets_match_moe_and_dense() -> None:
    assert unmatched_target_modules(_MOE_MODULES, ATTENTION_TARGET_MODULES) == []
    assert unmatched_target_modules(_DENSE_MODULES, TARGET_MODULES) == []


def test_normalize_target_modules_orders_and_validates() -> None:
    assert normalize_target_modules(["o_proj", "q_proj"]) == ("q_proj", "o_proj")
    with pytest.raises(ValueError, match="lm_head"):
        normalize_target_modules(["q_proj", "lm_head"])
    with pytest.raises(ValueError):
        normalize_target_modules([])
    with pytest.raises(ValueError):
        normalize_target_modules("q_proj")


def test_moe_profile_loads_attention_only_with_checkpointing() -> None:
    prof = load_lora_profile("moe30b_attn_local")
    assert prof["target_modules"] == ATTENTION_TARGET_MODULES
    assert prof["gradient_checkpointing"] is True
    assert prof["assistant_only_loss"] is True
    prof.pop("epochs", None)
    cfg = _cfg(**prof)
    assert build_lora_kwargs(cfg)["target_modules"] == list(ATTENTION_TARGET_MODULES)
    assert recipe_summary(cfg)["target_modules"] == list(ATTENTION_TARGET_MODULES)


def test_profile_target_modules_now_applied_not_ignored() -> None:
    # small_smoke_test YAML'da yalnız attention yazıyordu ama eskiden 7 modül eğitiliyordu.
    assert load_lora_profile("small_smoke_test")["target_modules"] == ATTENTION_TARGET_MODULES


def test_profile_with_bad_target_raises(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text("bad:\n  r: 8\n  target_modules: [q_proj, embed_tokens]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_lora_profile("bad", profiles_path=p)


def test_gradient_checkpointing_training_kwargs() -> None:
    off = build_training_kwargs(_cfg(), num_epochs=1, output_dir="o", on_cuda=False)
    assert "gradient_checkpointing" not in off
    on = build_training_kwargs(
        _cfg(gradient_checkpointing=True), num_epochs=1, output_dir="o", on_cuda=False
    )
    assert on["gradient_checkpointing"] is True
    assert on["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_ram_estimate_30b_bf16_fits_fp32_does_not() -> None:
    total = 61_064_245_248  # Qwen3-30B-A3B bf16 checkpoint ≈ 61 GB (safetensors index)
    bf16 = estimate_cpu_train_ram_gb(total, src_bytes=2, dst_bytes=2)
    fp32 = estimate_cpu_train_ram_gb(total, src_bytes=2, dst_bytes=4)
    assert 60 < bf16 < 128 < fp32


def test_check_cpu_ram_uses_local_index(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 61 * 1024**3}}), encoding="utf-8"
    )

    class _VM:
        available = 100 * 1024**3

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    ok, msg = check_cpu_ram(str(tmp_path), 2)
    assert ok, msg
    ok, msg = check_cpu_ram(str(tmp_path), 4)
    assert not ok
    assert "bf16" in msg


def test_check_cpu_ram_unknown_model_skips(tmp_path: Path) -> None:
    ok, msg = check_cpu_ram(str(tmp_path / "yok"), 2)
    assert ok
    assert "atlandı" in msg


def test_keep_awake_is_noop_safe() -> None:
    with peft_lora_train._keep_awake():
        pass


# --- Kademe-2: hedef kontrolü YÜKLEMEDEN önce (meta cihaz), yerel minik config ile -------
def _tiny_config_dir(tmp_path: Path, model_type: str) -> Path:
    d = tmp_path / model_type
    d.mkdir()
    cfg: dict = {
        "model_type": model_type,
        "architectures": [
            "Qwen3MoeForCausalLM" if model_type == "qwen3_moe" else "Qwen3ForCausalLM"
        ],
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
    }
    if model_type == "qwen3_moe":
        cfg.update({"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16})
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def test_precheck_targets_moe_vs_dense(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from app.training.peft_lora_train import precheck_target_modules

    moe = _tiny_config_dir(tmp_path, "qwen3_moe")
    dense = _tiny_config_dir(tmp_path, "qwen3")
    err = precheck_target_modules(str(moe), TARGET_MODULES)
    assert err and "gate_proj" in err and "moe30b_attn_local" in err
    assert precheck_target_modules(str(moe), ATTENTION_TARGET_MODULES) is None
    assert precheck_target_modules(str(dense), TARGET_MODULES) is None


def test_precheck_targets_unknown_model_is_skipped(tmp_path: Path) -> None:
    from app.training.peft_lora_train import precheck_target_modules

    assert precheck_target_modules(str(tmp_path / "yok"), TARGET_MODULES) is None
