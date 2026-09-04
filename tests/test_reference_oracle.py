"""ReferenceOracle testleri — determinizm + compute_indicator ile birebir referans."""

from __future__ import annotations

import math

import pandas as pd

from app.trading.indicators import compute_indicator
from app.verification.exams.reference_oracle import ReferenceOracle


def test_synthetic_closes_deterministik() -> None:
    a = ReferenceOracle.synthetic_closes(8, seed=0)
    b = ReferenceOracle.synthetic_closes(8, seed=0)
    pd.testing.assert_series_equal(a, b)
    c = ReferenceOracle.synthetic_closes(8, seed=1)
    assert not a.equals(c)  # farklı seed → farklı seri


def test_indicator_series_compute_indicator_ile_ayni() -> None:
    closes = ReferenceOracle.synthetic_closes(50, seed=7)
    df = pd.DataFrame({"close": closes})
    ref = ReferenceOracle.indicator_series("SMA", df, 5)
    expected = compute_indicator("SMA", df, 5)
    pd.testing.assert_series_equal(ref, expected)


def test_formula_value_safe_eval_kullanir() -> None:
    out = ReferenceOracle.formula_value("100 - 100/(1 + rs)", {"rs": 2.0})
    assert math.isclose(out, 100 - 100 / 3, rel_tol=1e-9)
