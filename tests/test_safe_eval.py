"""safe_eval testleri — doğru hesap + eval/exec sızıntısı YOK."""

from __future__ import annotations

import math

import pytest

from app.verification.exams.safe_eval import UnsafeExpressionError, safe_eval


def test_aritmetik_dogru() -> None:
    assert safe_eval("a*b + min(a, b)", {"a": 2, "b": 3}) == 8.0  # 6 + 2
    assert safe_eval("(a + b) / 2", {"a": 4, "b": 6}) == 5.0
    assert safe_eval("2 ** 3 - 1", {}) == 7.0


def test_rsi_benzeri_formul() -> None:
    # RSI = 100 - 100/(1+RS), RS=2 → 66.666...
    out = safe_eval("100 - 100/(1 + rs)", {"rs": 2.0})
    assert math.isclose(out, 100 - 100 / 3, rel_tol=1e-9)


def test_math_fonksiyonlari() -> None:
    assert safe_eval("sqrt(x)", {"x": 16}) == 4.0
    assert math.isclose(safe_eval("log(x)", {"x": math.e}), 1.0, rel_tol=1e-9)
    assert safe_eval("abs(0 - y)", {"y": 5}) == 5.0


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "os.system('rm -rf /')",
        "open('secret.txt')",
        "eval('1+1')",
        "(1).__class__",
        "data[0]",
        "lambda: 1",
        "[i for i in range(3)]",
        "{'a': 1}",
        "a if b else c",
    ],
)
def test_guvensiz_ifade_reddedilir(expr: str) -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_eval(expr, {"a": 1, "b": 2, "c": 3, "data": 0})


def test_tanimsiz_sembol_reddedilir() -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_eval("z + 1", {"a": 1})


def test_bool_sabiti_reddedilir() -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_eval("True", {})


def test_izin_verilmeyen_fonksiyon_reddedilir() -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_eval("pow(2, 3)", {})  # pow whitelist'te yok; ** kullanılmalı
