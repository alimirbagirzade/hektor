"""Gerçek küçük PEFT modelinde gradyan, base koruma ve birleşim eşitliği."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from app.training.merge_aware_lora import install_merge_aware  # noqa: E402


def test_real_peft_gradient_and_bit_exact_merge():
    torch.manual_seed(42)
    base = torch.nn.Sequential(torch.nn.Linear(8, 6, bias=True)).to(torch.bfloat16)
    original = base[0].weight.detach().clone()
    model = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["0"]))
    assert len(install_merge_aware(model)) == 1
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        model(x).float().square().mean().backward()
        assert model.base_model.model[0].lora_B.default.weight.grad.abs().sum() > 0
        if _ == 1:
            assert model.base_model.model[0].lora_A.default.weight.grad.abs().sum() > 0
        optimizer.step()
    assert torch.equal(original, model.base_model.model[0].base_layer.weight)
    model.eval()
    before = model(x).detach()
    with model.disable_adapter():
        assert torch.equal(
            model(x), torch.nn.functional.linear(x, original, base[0].base_layer.bias)
        )
    merged = model.merge_and_unload()
    assert torch.equal(before, merged(x))


def test_dropout_recipe_rejected():
    model = peft.get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(8, 6)).to(torch.bfloat16),
        peft.LoraConfig(r=2, target_modules=["0"], lora_dropout=0.1),
    )
    with pytest.raises(ValueError, match="dropout"):
        install_merge_aware(model)


def test_save_reload_reinstalls_recipe_and_preserves_merge(tmp_path):
    torch.manual_seed(71)
    original = torch.nn.Sequential(torch.nn.Linear(8, 6)).to(torch.bfloat16)
    state = {k: v.clone() for k, v in original.state_dict().items()}
    model = peft.get_peft_model(original, peft.LoraConfig(r=2, target_modules=["0"]))
    model.base_model.model[0].lora_B.default.weight.data.fill_(0.03)
    install_merge_aware(model)
    model.eval()
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    expected = model(x).detach()
    model.save_pretrained(tmp_path)
    fresh = torch.nn.Sequential(torch.nn.Linear(8, 6)).to(torch.bfloat16)
    fresh.load_state_dict(state)
    restored = peft.PeftModel.from_pretrained(fresh, tmp_path)
    install_merge_aware(restored)
    restored.eval()
    assert torch.equal(expected, restored(x))
    assert torch.equal(expected, restored.merge_and_unload(safe_merge=False)(x))


def test_autocast_rejected():
    model = peft.get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(8, 6)).to(torch.bfloat16),
        peft.LoraConfig(r=2, target_modules=["0"]),
    )
    install_merge_aware(model)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        model(torch.randn(2, 8, dtype=torch.bfloat16))


@pytest.mark.parametrize("option", ["bias", "multiple", "trainable_base", "dora"])
def test_unsupported_recipe_rejected(option):
    config = peft.LoraConfig(
        r=2,
        target_modules=["0"],
        bias="all" if option == "bias" else "none",
        use_dora=option == "dora",
    )
    model = peft.get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(8, 6)).to(torch.bfloat16), config
    )
    if option == "multiple":
        model.add_adapter("extra", peft.LoraConfig(r=2, target_modules=["0"]))
    if option == "trainable_base":
        model.base_model.model[0].base_layer.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        install_merge_aware(model)


def test_merged_disable_rejected_without_silent_base_substitution():
    model = peft.get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(8, 6)).to(torch.bfloat16),
        peft.LoraConfig(r=2, target_modules=["0"]),
    )
    install_merge_aware(model)
    layer = model.base_model.model[0]
    layer.merge(safe_merge=False)
    with pytest.raises(ValueError, match="merged layer"):
        install_merge_aware(model)
    layer.enable_adapters(False)
    with pytest.raises(ValueError, match="disable"):
        model(torch.randn(2, 8, dtype=torch.bfloat16))


def test_modules_to_save_rejected():
    model = peft.get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.Linear(6, 4)).to(torch.bfloat16),
        peft.LoraConfig(r=2, target_modules=["0"], modules_to_save=["1"]),
    )
    with pytest.raises(ValueError, match="modules_to_save"):
        install_merge_aware(model)
