"""v15 kritik hesap referansları ve dürüst kanıt kapısı; LLM başarı testi değildir."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from app.evals.v15_protocol import (
    decision,
    paired_interval,
    posterior,
    split_errors,
    transition_matrix,
    validate_ohlc,
    verify_lock,
    write_lock,
)


def test_s07_rsi_independent_gain_loss_reference() -> None:
    prices = pd.Series([100, 103, 101, 105, 104], dtype=float)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(4).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(4).mean().iloc[-1]
    actual = 100 - 100 / (1 + gain / loss)
    assert actual == pytest.approx(70)  # bağımsız: 100 * (3+4)/(3+4+2+1)
    assert 0 <= actual <= 100


def test_s09_true_range_and_volume_absence() -> None:
    high, low, previous_close = 108, 103, 100
    assert max(high - low, abs(high - previous_close), abs(low - previous_close)) == 8
    assert "volume" not in {"high": high, "low": low, "close": 106}


def test_s14_variance_units_and_gain() -> None:
    gain, estimate = posterior(100, 106, 4, 2)
    assert gain == pytest.approx(2 / 3)
    assert estimate == pytest.approx(104)
    # Birim ölçeği ×10 -> varyans ×100; kazanç değişmez.
    scaled_gain, scaled = posterior(1000, 1060, 400, 200)
    assert scaled_gain == pytest.approx(gain)
    assert scaled == pytest.approx(10 * estimate)


def test_s22_transition_counts_not_index_alignment() -> None:
    states = [0, 1, 0, 2, 1, 0]
    matrix = transition_matrix(states, 3)
    assert matrix == [[0, 0.5, 0.5], [1, 0, 0], [0, 1, 0]]
    counts = np.zeros((3, 3))
    np.add.at(counts, (np.array(states[:-1]), np.array(states[1:])), 1)
    np.testing.assert_allclose(matrix, counts / counts.sum(axis=1, keepdims=True))
    assert all(math.isnan(x) for x in transition_matrix([0], 2)[0])


def test_s25_payoff_and_break_even() -> None:
    p, win, loss, cost = 0.6, 0.01, 0.02, 0.001
    assert p * win - (1 - p) * loss == pytest.approx(-0.002)
    assert p * win - (1 - p) * loss - cost == pytest.approx(-0.003)
    assert (loss + cost) / (win + loss) == pytest.approx(0.7)


def test_s28_future_perturbation_positive_and_negative_controls() -> None:
    original = pd.Series([1, 4, 2, 7, 3, 5, 6, 8], dtype=float)
    perturbed = original.copy()
    cutoff = 3
    perturbed.iloc[cutoff + 1 :] = [900, -400, 700, -800]
    np.testing.assert_allclose(
        original.ewm(alpha=0.3, adjust=False).mean().iloc[: cutoff + 1],
        perturbed.ewm(alpha=0.3, adjust=False).mean().iloc[: cutoff + 1],
    )
    # shift(1), full-sample fit sızıntısını çözmez: karşı kontrol gerçekten başarısız.
    before = ((original - original.mean()) / original.std()).shift(1)
    after = ((perturbed - perturbed.mean()) / perturbed.std()).shift(1)
    assert not np.allclose(before.iloc[1 : cutoff + 1], after.iloc[1 : cutoff + 1])


def test_s29_funding_and_turnover() -> None:
    assert pytest.approx(0.224) == 64 * 7 * 0.0005
    assert abs(0.8 - 0.5) == pytest.approx(0.3)
    assert 1000 * abs(0.8 - 0.5) * 0.001 == pytest.approx(0.3)


def test_s20_close_to_close_execution_and_turnover() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    weights = pd.Series([0.5, 0.8, 0.3])  # kapanışta yeni hedef ağırlık
    gross = weights.shift(1) * prices.pct_change(fill_method=None)
    np.testing.assert_allclose(gross.iloc[1:], [0.05, -0.08])
    turnover = weights.diff().abs()
    assert turnover.iloc[1] == pytest.approx(0.3)
    net = gross - turnover * 0.001  # aynı kapanışta yeniden dengeleme, tek yön oran
    np.testing.assert_allclose(net.iloc[1:], [0.0497, -0.0805])
    # Bu teknik sentetik fixture gerçekleşmiş piyasa/işlem maliyeti ölçümü değildir.


def test_s24_gaussian_hmm_variances_and_filtering_oracle() -> None:
    from itertools import product

    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    initial = np.array([0.5, 0.5])
    sigmas = np.array([1.0, 3.0])
    observations = [0.0, 1.0, 3.0]

    def emission(x: float) -> np.ndarray:
        return np.exp(-0.5 * (x / sigmas) ** 2) / (sigmas * math.sqrt(2 * math.pi))

    filtered = initial.copy()
    for index, observation in enumerate(observations):
        prior = filtered @ transition if index else initial
        filtered = prior * emission(observation)
        filtered /= filtered.sum()
        # Bağımsız oracle: tüm gizli durum yollarını açıkça enumerate et.
        masses = np.zeros(2)
        for path in product(range(2), repeat=index + 1):
            mass = initial[path[0]] * emission(observations[0])[path[0]]
            for t in range(1, index + 1):
                mass *= transition[path[t - 1], path[t]] * emission(observations[t])[path[t]]
            masses[path[-1]] += mass
        np.testing.assert_allclose(filtered, masses / masses.sum())
    # Aynı sıfır ortalama, farklı varyanslı Gaussian emission gerçekten ayrışır.
    assert emission(0)[0] == pytest.approx(3 * emission(0)[1])
    assert 2 * (2 - 1) + 2 * 2 + (2 - 1) == 7  # geçiş + mean/variance + initial


def test_s29_purge_label_intervals_not_universal_gap() -> None:
    train_intervals = [(0, 2), (2, 4), (5, 8)]
    validation = (3, 6)
    kept = [
        interval
        for interval in train_intervals
        if interval[1] < validation[0] or interval[0] > validation[1]
    ]
    assert kept == [(0, 2)]  # kapalı aralıklar; veri/label ufkuna bağlı


@pytest.mark.parametrize("values", [(100, 102, 97, 98), (100, 103, 99, 102)])
def test_ohlc_valid_rising_and_falling(values: tuple[int, ...]) -> None:
    assert validate_ohlc(values)


@pytest.mark.parametrize(
    "values",
    [
        (100, 99, 97, 98),
        (100, 102, 101, 98),
        (100, math.inf, 97, 98),
        (100, math.nan, 97, 98),
        (100, "102", 97, 98),
        (True, 102, 97, 98),
    ],
)
def test_ohlc_rejects_bad_types_and_ranges(values: tuple[object, ...]) -> None:
    assert not validate_ohlc(values)


def test_istanbul_no_seasonal_dst() -> None:
    local = datetime(2026, 1, 15, 10, tzinfo=ZoneInfo("Europe/Istanbul"))
    assert local.astimezone(ZoneInfo("UTC")).hour == 7


def test_group_split_rejects_same_source_or_template() -> None:
    a = {"id": "a", "question": "bir", "source_group": "paper", "template_family": "x"}
    b = {"id": "b", "question": "iki", "source_group": "paper", "template_family": "x"}
    errors = split_errors({"train": [a], "final": [b]})
    assert any("source_group" in error for error in errors)
    assert any("template_family" in error for error in errors)


def test_locked_final_cannot_be_overwritten_or_modified(tmp_path: Path) -> None:
    source = tmp_path / "final.jsonl"
    source.write_text("ilk", encoding="utf-8")
    lock = write_lock(tmp_path / "locks", [source])
    assert verify_lock(lock) == []
    with pytest.raises(FileExistsError):
        write_lock(tmp_path / "locks", [source])
    source.write_text("değişmiş", encoding="utf-8")
    assert verify_lock(lock)


def test_empty_evidence_never_accepts() -> None:
    assert decision({})[0] == "yetersiz_kanit"


def test_unmeasured_performance_claim_alone_rejects() -> None:
    assert decision({"unmeasured_claims": 1})[0] == "reddedilen_aday"


def test_bootstrap_question_level_deterministic() -> None:
    assert paired_interval([0, 0.2, -0.1]) == paired_interval([0, 0.2, -0.1])
    assert paired_interval([0, 0.2, -0.1])["n_questions"] == 3


def complete_evidence() -> dict:
    from app.evals.v15_protocol import ACCEPTANCE

    evidence = dict.fromkeys(
        (
            "training_complete",
            "transfer_verified",
            "final_locked",
            "checkpoint_locked",
            "final_complete",
            "blind_review_complete",
            "paired_v14_complete",
            "critical_model_fixtures_pass",
            "repetition_improved",
            "first_attempt_only",
        ),
        True,
    )
    evidence.update(
        critical_errors=0,
        unmeasured_claims=0,
        core_regressions=0,
        score=108,
        maximum=120,
        merge_max_kl=0.001,
        merge_scope=ACCEPTANCE["merge_scope_required"],
        merge_kl_direction=ACCEPTANCE["merge_kl_direction"],
        checkpoint_locked_at="2026-10-04T09:00:00+00:00",
        final_started_at="2026-10-04T10:00:00+00:00",
        expected_question_ids=[f"q{i}" for i in range(30)],
        scored_question_ids=[f"q{i}" for i in range(30)],
        development_score_fraction=0.9,
        raw_answers_sha256="a" * 64,
        blind_scores_sha256="b" * 64,
        paired_v14_sha256="c" * 64,
    )
    return evidence


def test_complete_evidence_accepts() -> None:
    assert decision(complete_evidence())[0] == "kabul_edilen_arastirma_adayi"


@pytest.mark.parametrize("field", list(complete_evidence()))
def test_each_missing_gate_blocks_acceptance(field: str) -> None:
    evidence = complete_evidence()
    del evidence[field]
    assert decision(evidence)[0] != "kabul_edilen_arastirma_adayi"


def test_final_order_and_wrong_kl_direction_block() -> None:
    evidence = complete_evidence()
    evidence["final_started_at"] = "2026-10-04T08:00:00+00:00"
    assert decision(evidence)[0] == "reddedilen_aday"
    evidence = complete_evidence()
    evidence["merge_kl_direction"] = "reversed"
    assert decision(evidence)[0] == "yetersiz_kanit"


def test_empty_splits_and_lock_block(tmp_path: Path) -> None:
    assert split_errors({})
    assert split_errors({"train": [], "final": []})
    with pytest.raises(ValueError):
        write_lock(tmp_path, [])
