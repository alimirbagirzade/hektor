"""Birleşik bf16 hesabını eğitimde kullanan deneysel vanilla LoRA sınırı."""

from __future__ import annotations

from types import MethodType
from typing import Any


def merged_weight_forward(layer: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Base'i değiştirmeden fp32 toplamı base dtype'a yuvarla; cast gradyanı korunur."""
    import torch
    import torch.nn.functional as functional

    if args or kwargs:
        raise ValueError("Merge-aware Linear ek forward argümanı desteklemez")
    base = layer.get_base_layer()
    if (
        x.dtype != base.weight.dtype
        or torch.is_autocast_enabled("cpu")
        or torch.is_autocast_enabled("cuda")
    ):
        raise ValueError("Merge-aware: base/input dtype eşit olmalı, autocast kapalı olmalı")
    if layer.disable_adapters and layer.merged:
        raise ValueError("Merge-aware merged layer disable edilemez; ayrı frozen base kullan")
    if layer.disable_adapters or layer.merged:
        return base(x)
    adapters = [name for name in layer.active_adapters if name in layer.lora_A]
    if len(adapters) != 1:
        raise ValueError("Merge-aware yalnız tek aktif adapter destekler")
    name = adapters[0]
    delta = layer.get_delta_weight(name)
    weight = (base.weight.float() + delta.float()).to(base.weight.dtype)
    return functional.linear(x, weight, base.bias)


def install_merge_aware(model: Any) -> list[str]:
    """İsimleri döndür; desteklenmeyen reçeteyi hesap başlamadan reddet."""
    import torch
    from peft.tuners.lora.layer import Linear, LoraLayer

    for config in model.peft_config.values():
        if config.bias != "none" or config.modules_to_save or config.target_parameters:
            raise ValueError("Merge-aware bias/modules_to_save/target_parameters desteklemez")
        if config.init_lora_weights not in (True, "gaussian"):
            raise ValueError("Merge-aware base'i değiştiren initialization desteklemez")
    if any(
        p.requires_grad and "lora_A" not in name and "lora_B" not in name
        for name, p in model.named_parameters()
    ):
        raise ValueError("Merge-aware yalnız A/B parametrelerini eğitir")
    for name, layer in model.named_modules():
        if isinstance(layer, LoraLayer) and not isinstance(layer, Linear):
            raise ValueError(f"{name}: yalnız LoRA Linear desteklenir")

    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, Linear)]
    if not layers:
        raise ValueError("Merge-aware için LoRA Linear bulunamadı")
    for name, layer in layers:
        base = layer.get_base_layer()
        if layer.merged:
            raise ValueError(f"{name}: kurulum sırasında merged layer desteklenmez")
        if not isinstance(base, torch.nn.Linear) or any(p.requires_grad for p in base.parameters()):
            raise ValueError(f"{name}: frozen torch Linear base gerekir")
        if base.weight.dtype != torch.bfloat16 or layer.fan_in_fan_out:
            raise ValueError(f"{name}: yalnız bf16, transpozesiz Linear desteklenir")
        if layer.lora_variant or any(layer.lora_bias.values()):
            raise ValueError(f"{name}: DoRA/variant/LoRA bias desteklenmez")
        if layer.active_adapters != ["default"] or list(layer.lora_A) != ["default"]:
            raise ValueError(f"{name}: tek adapter gerekir")
        for dropout in layer.lora_dropout.values():
            if getattr(dropout, "p", 0) != 0:
                raise ValueError(f"{name}: dropout=0 gerekir")
        for parameter in list(layer.lora_A.parameters()) + list(layer.lora_B.parameters()):
            if parameter.dtype != torch.float32:
                raise ValueError(f"{name}: LoRA parametreleri fp32 olmalı")
    for _, layer in layers:
        layer.forward = MethodType(merged_weight_forward, layer)  # type: ignore[method-assign]
    return [name for name, _ in layers]
