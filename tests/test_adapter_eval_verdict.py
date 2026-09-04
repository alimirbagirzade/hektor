"""Adapter eval verdict mantığı — v5 disiplin-regresyon koruması (küçük-n accept yasağı).

v5 dersi: eval n=1 örnekle 'accept' demişti (istatistiksel temel yok). `_decide_verdict`
artık min_n altında 'accept' vermez; regresyon ise her n'de raporlanır (güvenli yön).
Tamamen çevrimdışı — saf karar fonksiyonu, torch gerektirmez.
"""

from __future__ import annotations

import json

from app.training.adapter_eval import _MIN_EVAL_N, _decide_verdict, _resolve_base_model


def test_regression_rejected_at_any_n() -> None:
    # adapter base'in altında → tek örnekte bile reject (güvenli yön)
    assert _decide_verdict(0.8, 0.5, n=1) == "reject"
    assert _decide_verdict(0.8, 0.5, n=100) == "reject"


def test_small_n_improvement_is_inconclusive_not_accept() -> None:
    # v5 senaryosu: tek soruda adapter daha iyi görünür → ASLA accept
    assert _decide_verdict(0.0, 1.0, n=1) == "inconclusive"
    assert _decide_verdict(0.5, 0.9, n=_MIN_EVAL_N - 1) == "inconclusive"


def test_sufficient_n_improvement_accepts() -> None:
    assert _decide_verdict(0.5, 0.9, n=_MIN_EVAL_N) == "accept"
    assert _decide_verdict(0.5, 0.6, n=20) == "accept"


def test_equality_is_inconclusive() -> None:
    assert _decide_verdict(0.7, 0.7, n=50) == "inconclusive"


def test_degenerate_adapter_always_rejected() -> None:
    """Degenerasyon (tekrar döngüsü) KATEGORİK başarısızlık — v5 dersi: skordan bağımsız veto.

    Adapter base'den daha iyi skor alsa ve n büyük olsa bile, dejenere çıktı reddedilir.
    """
    assert _decide_verdict(0.5, 0.9, n=50, adapter_degenerate=True) == "reject"
    assert _decide_verdict(0.0, 1.0, n=100, adapter_degenerate=True) == "reject"
    # degenerate False iken normal akış korunur
    assert _decide_verdict(0.5, 0.9, n=50, adapter_degenerate=False) == "accept"


def test_custom_min_n_threshold() -> None:
    assert _decide_verdict(0.5, 0.9, n=8, min_n=10) == "inconclusive"
    assert _decide_verdict(0.5, 0.9, n=10, min_n=10) == "accept"


def test_resolve_base_model_reads_adapter_config(tmp_path) -> None:
    """adapter_config.json'daki base_model_name_or_path okunmalı (küçük-model adapter'ı
    yanlış base'le yüklenmesin)."""
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"}),
        encoding="utf-8",
    )
    assert _resolve_base_model(tmp_path) == "Qwen/Qwen2.5-1.5B-Instruct"


def test_resolve_base_model_missing_or_bad_returns_none(tmp_path) -> None:
    assert _resolve_base_model(tmp_path) is None  # config yok
    (tmp_path / "adapter_config.json").write_text("{ bozuk json", encoding="utf-8")
    assert _resolve_base_model(tmp_path) is None  # parse edilemez
