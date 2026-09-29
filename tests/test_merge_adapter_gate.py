"""scripts/merge_adapter.py kapı mantığı (Kademe-2 D3/D6) — çevrimdışı, model yüklemez."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
