"""scripts/merge_adapter.py kapı mantığı (Kademe-2 D3/D6) — çevrimdışı, model yüklemez."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "merge_adapter", Path(__file__).resolve().parents[1] / "scripts" / "merge_adapter.py"
)
assert _spec and _spec.loader
merge_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_adapter)


def _m(**kw: object) -> dict:
    base = {"diff": 0.4, "effect": 13.75, "kl": 0.0016, "top10": 9, "same_top": True}
    base.update(kw)
    return base


def test_v12_measurement_passes() -> None:
    assert merge_adapter.gate_failures([_m()]) == []


def test_noop_adapter_fails() -> None:
    # Anahtar kayması → adapter hiç yüklenmez: etki 0, fark 0; eski kapı 0 ≤ 0.5·0 ile GEÇİYORDU.
    fails = merge_adapter.gate_failures([_m(diff=0.0, effect=0.0, kl=0.0, top10=10)])
    assert any("adapter etkisi" in f for f in fails)


def test_merge_error_relative_threshold_tightened() -> None:
    assert merge_adapter.gate_failures([_m(diff=3.0)])  # eski 0.5 eşiğinde geçerdi
    assert merge_adapter.gate_failures([_m(), _m(kl=0.05)])  # her prompt ayrı denetlenir


def test_max_kl_override_relaxes_only_kl() -> None:
    """v13 ölçümü (2026-09-30): KL 0.0201 varsayılanda düşer, bilinçli 0.025 ile geçer."""
    v13 = _m(diff=0.6035, effect=17.69, kl=0.0201, top10=10)
    assert merge_adapter.gate_failures([v13]) == ["prompt 0: KL 0.0201 > 0.01"]
    assert merge_adapter.gate_failures([v13], max_kl=0.025) == []
    # Gevşek KL diğer kapıları AÇMAZ.
    assert merge_adapter.gate_failures([_m(diff=3.0, kl=0.02)], max_kl=0.025)


def test_all_positions_catches_error_hidden_by_last_token() -> None:
    torch = pytest.importorskip("torch")
    before = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 10
    after = before.clone()
    after[0, 0] += 10  # son token konumu aynıdır; eski ölçüm hatayı kaçırırdı.
    metrics = merge_adapter.distribution_metrics(before, after, before - 2)
    assert metrics["positions"] == 2
    assert metrics["kl"] > 0.01
    assert metrics["kl_mean"] < metrics["kl"]
    assert metrics["same_top"] is False
    assert merge_adapter.gate_failures([metrics])


def test_all_positions_identical_distributions() -> None:
    torch = pytest.importorskip("torch")
    before = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 10
    metrics = merge_adapter.distribution_metrics(before, before.clone(), before - 2)
    assert metrics["kl"] == pytest.approx(0, abs=1e-7)
    assert metrics["top10"] == 10
    assert merge_adapter.gate_failures([metrics]) == []


@pytest.mark.parametrize("key", ["effect", "diff", "kl"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metrics_fail_closed(key, value):
    assert merge_adapter.gate_failures([_m(**{key: value})])


def test_nonfinite_logits_rejected():
    torch = pytest.importorskip("torch")
    before = torch.zeros((2, 12))
    before[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Sonlu olmayan"):
        merge_adapter.distribution_metrics(before, before, before)
