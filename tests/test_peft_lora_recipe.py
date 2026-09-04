"""İleri LoRA reçetesi (rsLoRA/DoRA/init/NEFTune/LoRA+/regularizasyon) plumbing testleri.

Bu testler ÇEVRİMDIŞI çalışır: yalnız saf config builder'ları sınar (torch/peft yüklemez).
Amaç: araştırma entegrasyonunun (doküman v1.2) config→PEFT kwargs köprüsünü korumak.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.training.peft_lora_train import (
    PeftTrainConfig,
    build_lora_kwargs,
    build_training_kwargs,
    init_is_gguf_unsafe,
    load_lora_profile,
    normalize_init_lora_weights,
    recipe_summary,
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


# --- normalize_init_lora_weights ---


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("", True),
        (True, True),
        ("false", False),
        (False, False),
        ("pissa", "pissa"),
        ("OLoRA", "olora"),
        ("gaussian", "gaussian"),
        ("pissa_niter_8", "pissa_niter_8"),
    ],
)
def test_normalize_init_valid(value, expected) -> None:
    assert normalize_init_lora_weights(value) == expected


def test_normalize_init_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen init_lora_weights"):
        normalize_init_lora_weights("magic")


def test_init_is_gguf_unsafe() -> None:
    assert init_is_gguf_unsafe("pissa") is True
    assert init_is_gguf_unsafe("olora") is True
    assert init_is_gguf_unsafe("corda") is True
    assert init_is_gguf_unsafe("pissa_niter_4") is True
    assert init_is_gguf_unsafe("gaussian") is False
    assert init_is_gguf_unsafe("true") is False
    assert init_is_gguf_unsafe("loftq") is False


# --- build_lora_kwargs ---


def test_build_lora_kwargs_defaults() -> None:
    kw = build_lora_kwargs(_cfg())
    assert kw["r"] == 8
    assert kw["lora_alpha"] == 16
    assert kw["bias"] == "none"
    assert kw["task_type"] == "CAUSAL_LM"
    # lm_head/embed YOK — tied-embedding/GGUF güvenliği
    assert kw["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert "lm_head" not in kw["target_modules"]
    assert kw["use_rslora"] is False
    assert kw["use_dora"] is False
    assert kw["init_lora_weights"] is True


def test_build_lora_kwargs_advanced() -> None:
    kw = build_lora_kwargs(_cfg(use_rslora=True, use_dora=True, init_lora_weights="pissa"))
    assert kw["use_rslora"] is True
    assert kw["use_dora"] is True
    assert kw["init_lora_weights"] == "pissa"


# --- build_training_kwargs ---


def test_build_training_kwargs_defaults() -> None:
    kw = build_training_kwargs(_cfg(), num_epochs=2, output_dir="out", on_cuda=False)
    assert kw["num_train_epochs"] == 2
    assert kw["weight_decay"] == 0.01
    assert kw["warmup_ratio"] == 0.03
    assert kw["lr_scheduler_type"] == "cosine"
    assert kw["max_grad_norm"] == 1.0
    assert kw["seed"] == 42
    assert kw["fp16"] is False
    # NEFTune varsayılan kapalı → anahtar HİÇ olmamalı (0.0 None'a düşmez, hook açılmaz)
    assert "neftune_noise_alpha" not in kw


def test_build_training_kwargs_neftune_and_cuda() -> None:
    kw = build_training_kwargs(
        _cfg(neftune_noise_alpha=5), num_epochs=1, output_dir="out", on_cuda=True
    )
    assert kw["neftune_noise_alpha"] == 5
    assert kw["fp16"] is True


def test_build_training_kwargs_max_steps_caps() -> None:
    # max_steps verilmezse anahtar HİÇ olmamalı (eski davranış: epoch yönetir).
    kw0 = build_training_kwargs(_cfg(), num_epochs=2, output_dir="out", on_cuda=False)
    assert "max_steps" not in kw0
    # max_steps>0 → adım sayısı TAM kapanır (kök bug fix: iterations artık epoch'a kaçmaz).
    kw = build_training_kwargs(_cfg(), num_epochs=1, output_dir="out", on_cuda=False, max_steps=200)
    assert kw["max_steps"] == 200


# --- recipe_summary ---


def test_recipe_summary_vanilla() -> None:
    rs = recipe_summary(_cfg())
    assert rs["advanced_techniques"] == ["(yok — vanilya LoRA)"]
    assert rs["gguf_unsafe_init"] is False


def test_recipe_summary_lists_techniques() -> None:
    rs = recipe_summary(
        _cfg(
            use_rslora=True, neftune_noise_alpha=5, loraplus_lr_ratio=16, init_lora_weights="pissa"
        )
    )
    joined = " ".join(rs["advanced_techniques"])
    assert "rsLoRA" in joined
    assert "NEFTune" in joined
    assert "LoRA+" in joined
    assert "init=pissa" in joined
    assert rs["gguf_unsafe_init"] is True


def test_assistant_only_loss_opt_in() -> None:
    """assistant_only_loss: varsayılan kapalı (özette yok); açıkken özette görünür.

    v5 disiplin-onarımı adayı; config-seviyesi opt-in olarak taşınır (bulut notebook
    wiring'i ayrı adım). Varsayılan False → eğitim davranışı/regresyon yok.
    """
    assert _cfg().assistant_only_loss is False
    assert all(
        "assistant_only_loss" not in t for t in recipe_summary(_cfg())["advanced_techniques"]
    )

    joined = " ".join(recipe_summary(_cfg(assistant_only_loss=True))["advanced_techniques"])
    assert "assistant_only_loss" in joined


# --- load_lora_profile (gerçek YAML) ---


def test_load_profile_discipline_safe() -> None:
    prof = load_lora_profile("discipline_safe")
    assert prof["lora_r"] == 16
    assert prof["lora_alpha"] == 32
    assert prof["learning_rate"] == 0.0001
    assert prof["neftune_noise_alpha"] == 5
    assert prof["lr_scheduler_type"] == "cosine"
    assert prof["epochs"] == 1  # config alanı değil ama bilgi olarak taşınır
    # assistant_only_loss profil→config alanına taşınır (şu an false: davranış değişmez)
    assert prof["assistant_only_loss"] is False


def test_load_profile_applies_to_config() -> None:
    prof = load_lora_profile("discipline_safe")
    prof.pop("epochs", None)
    prof.pop("max_examples", None)
    cfg = _cfg(**prof)
    assert cfg.lora_r == 16
    assert cfg.learning_rate == 0.0001
    assert cfg.neftune_noise_alpha == 5
    # Builder'a da doğru akmalı
    assert (
        build_training_kwargs(cfg, num_epochs=1, output_dir="o", on_cuda=False)[
            "neftune_noise_alpha"
        ]
        == 5
    )


def test_load_profile_unknown_raises() -> None:
    with pytest.raises(KeyError):
        load_lora_profile("yok_boyle_profil")


# --- kl_reg_beta (araştırma turu 3 — arXiv:2512.22337, KL-regularized SFT) ---


def test_kl_reg_beta_default_off() -> None:
    """Varsayılan 0.0 — recipe_summary'de görünmez, davranış değişmez (opt-in)."""
    assert _cfg().kl_reg_beta == 0.0
    assert all("kl_reg" not in t for t in recipe_summary(_cfg())["advanced_techniques"])


def test_kl_reg_beta_shows_in_recipe_summary() -> None:
    joined = " ".join(recipe_summary(_cfg(kl_reg_beta=0.01))["advanced_techniques"])
    assert "kl_reg" in joined
    assert "0.01" in joined


def test_make_trainer_cls_beta_zero_returns_plain_trainer() -> None:
    # transformers yalnız train-cpu/dev extra'sında kurulu olabilir (CI'da yok) —
    # importorskip ile bu test o ortamda sessizce atlanır (skip != fail).
    transformers = pytest.importorskip("transformers")

    from app.training.peft_lora_train import _make_trainer_cls

    assert _make_trainer_cls(0.0) is transformers.Trainer
    assert _make_trainer_cls(-1.0) is transformers.Trainer


def test_make_trainer_cls_beta_positive_returns_subclass() -> None:
    transformers = pytest.importorskip("transformers")

    from app.training.peft_lora_train import _make_trainer_cls

    cls = _make_trainer_cls(0.01)
    assert issubclass(cls, transformers.Trainer)
    assert cls is not transformers.Trainer
    assert cls.kl_reg_beta == 0.01


def test_load_profile_discipline_safe_kl() -> None:
    """discipline_safe_kl: discipline_safe_local + kl_reg_beta=0.01 (DENEYSEL profil)."""
    prof = load_lora_profile("discipline_safe_kl")
    assert prof["kl_reg_beta"] == 0.01
    assert prof["assistant_only_loss"] is True
    assert prof["learning_rate"] == 0.0001
    assert prof["neftune_noise_alpha"] == 5


def test_high_capacity_uses_rslora() -> None:
    prof = load_lora_profile("high_capacity_reasoning")
    assert prof["use_rslora"] is True
    assert prof["lora_r"] == 32


def test_discipline_safe_drives_cloud_notebook(tmp_path) -> None:
    """lora-cloud-prep --profile discipline_safe yolunun ürettiği bulut notebook'u,
    v5 reçetesini (lr=1e-4, dropout=0.1, NEFTune=5, r=16, epoch=1) gerçekten yansıtır.

    main.py CLI mantığını birebir taklit eder (profil → recipe → build_stage2_notebook),
    veri setine dokunmadan reçete köprüsünü regresyona karşı korur.
    """
    import json

    from app.training.cloud_notebook import build_stage2_notebook

    prof = load_lora_profile("discipline_safe")
    out = tmp_path / "nb.ipynb"
    build_stage2_notebook(
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        adapter_name="achilles_lora_cloud",
        hf_dataset_repo="user/achilles-lora-sft",
        max_seq_length=prof["max_seq_length"],
        lora_r=prof["lora_r"],
        learning_rate=prof["learning_rate"],
        num_epochs=prof["epochs"],
        out_path=out,
        lora_alpha=prof["lora_alpha"],
        lora_dropout=prof["lora_dropout"],
        use_rslora=prof["use_rslora"],
        neftune_noise_alpha=prof["neftune_noise_alpha"],
        weight_decay=prof["weight_decay"],
        warmup_ratio=prof["warmup_ratio"],
    )
    nb = json.loads(out.read_text(encoding="utf-8"))
    src = "".join(
        ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
        for c in nb["cells"]
    )
    assert "{" not in src.split("```")[0] or "{BASE_MODEL}" not in src  # placeholder kalmadı
    assert "LEARNING_RATE   = 0.0001" in src
    assert "LORA_DROPOUT    = 0.1" in src
    assert "NEFTUNE_ALPHA   = 5" in src
    assert "LORA_ALPHA      = 32" in src
    assert "LORA_R          = 16" in src
    assert "USE_RSLORA      = False" in src
    assert "neftune_noise_alpha = (NEFTUNE_ALPHA or None)" in src
    # v5 disiplin-onarımı REGRESYON GUARD'ı: bulut notebook asistan-only loss'u
    # train_on_responses_only ile uygular + maskeleme doğrulaması (loss token > 0).
    # Bu hücreler silinirse/bozulursa v5-tipi regresyon riski geri gelir → test korur.
    assert "train_on_responses_only" in src
    assert '"<|im_start|>assistant\\n"' in src  # response_part — yalnız asistana loss
    assert "n_loss > 0" in src  # maskeleme boş kalmasın guard'ı
