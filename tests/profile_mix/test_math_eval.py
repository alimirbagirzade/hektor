"""Matematik/istatistik/trading/kod deterministik değerlendiricileri."""

from __future__ import annotations

import pytest

from app.evals.profile.answer_eval import evaluate_answer
from app.evals.profile.coding_eval import evaluate_coding
from app.evals.profile.math_eval import evaluate_math
from app.evals.profile.schema import EvalItem
from app.evals.profile.text_checks import exact_match, final_number, parse_number


def _m(value: float, tol: float | None = 1e-6, **kw: object) -> EvalItem:
    return EvalItem(
        id="m",
        domain="math",
        question="q",
        expected_numeric_value=value,
        numeric_tolerance=tol,
        **kw,
    )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("0,5", 0.5),
        ("1.234,5", 1234.5),
        ("1,234.5", 1234.5),
        ("1,234", 1234.0),
        ("−3", -3.0),
        ("1e-4", 1e-4),
    ],
)
def test_parse_number_locales(text: str, value: float) -> None:
    assert parse_number(text) == pytest.approx(value)


def test_final_answer_marker_wins() -> None:
    assert final_number("Adım 1: 3*4 = 12\nAdım 2: 12-4 = 8\nCevap: 8") == 8
    assert final_number("Önce 10 bulunur, sonuç 14.") == 14


def test_numeric_tolerance() -> None:
    assert evaluate_math(_m(1.469328, 5e-4), "Cevap: 1.4693")["correct"] is True
    assert evaluate_math(_m(1.469328, 5e-4), "Cevap: 1.47")["correct"] is False
    assert evaluate_math(_m(14), "Determinant 14'tür.\nCevap: 14")["correct"] is True


def test_no_percent_ratio_leniency_prevents_false_pass() -> None:
    # 2 beklenirken 200 (ya da 0.02) KABUL EDİLMEZ — sahte PASS en tehlikeli durumdur.
    assert evaluate_math(_m(2), "Cevap: 200")["correct"] is False
    assert evaluate_math(_m(2), "Cevap: 0.02")["correct"] is False


def test_missing_number_is_wrong() -> None:
    assert evaluate_math(_m(7), "Bilemiyorum")["correct"] is False


def test_multiple_choice_exact_match() -> None:
    assert exact_match("Doğru seçenek: (B)\nCevap: B", ["B"])
    assert not exact_match("Cevap: C", ["B"])
    item = EvalItem(id="mc", domain="math", question="q", accepted_answers=["B"])
    assert evaluate_math(item, "Cevap: B")["correct"] is True


def test_forbidden_claim_overrides_correct_number() -> None:
    item = _m(0.5, forbidden_claims=["garanti kazanç"])
    assert evaluate_math(item, "Garanti kazanç var. Cevap: 0.5")["correct"] is False


def test_statistics_needs_claims_as_well() -> None:
    item = EvalItem(
        id="s",
        domain="statistics",
        question="q",
        expected_numeric_value=2,
        numeric_tolerance=1e-6,
        required_claims=["örneklem küçük|küçük örneklem"],
    )
    ok = evaluate_answer(item, "Örneklem küçük olduğundan dikkat. Cevap: 2")
    half = evaluate_answer(item, "Cevap: 2")
    assert ok["correct"] is True and ok["score"] == 1.0
    assert half["correct"] is False and half["score"] == 0.5


def test_trading_advice_language_fails_but_negation_does_not() -> None:
    item = EvalItem(
        id="t",
        domain="trading",
        question="q",
        required_claims=["hipotez|test", "kesin değil|belirsiz"],
    )
    assert evaluate_answer(item, "Bu kesin değil; test edilmesi gereken bir hipotez.")["correct"]
    bad = evaluate_answer(
        item, "Kesinlikle al! Hipotez değil, garanti. Test gereksiz; kesin değil demem."
    )
    assert bad["correct"] is False and bad["checks"]["advice_language"]


def test_trading_cost_subdomain_requires_cost_mention() -> None:
    item = EvalItem(
        id="t",
        domain="trading",
        subdomain="costs",
        question="q",
        expected_numeric_value=0.01,
        numeric_tolerance=5e-4,
    )
    assert evaluate_answer(item, "Net getiri. Cevap: 0.01")["correct"] is False
    assert (
        evaluate_answer(item, "Komisyon ve slippage düşülünce net: Cevap: 0.01")["correct"] is True
    )


def test_abstention_item() -> None:
    item = EvalItem(id="a", domain="reasoning", question="q", requires_abstention=True)
    assert evaluate_answer(item, "Kaynaklarda bu bilgi yok.")["correct"] is True
    assert evaluate_answer(item, "Performans %12 olacak.")["correct"] is False


CODE_ITEM = EvalItem(
    id="c",
    domain="coding",
    question="q",
    requires_code_execution=True,
    code_tests="assert shift_signal([1,0,1]) == [0,1,0]",
)


def test_code_execution_disabled_is_unmeasured_not_pass() -> None:
    ans = "```python\ndef shift_signal(s):\n    return [0] + s[:-1] if s else []\n```"
    res = evaluate_coding(CODE_ITEM, ans, allow_execution=False)
    assert res["correct"] is None and "skipped" in res["checks"]


def test_code_execution_unit_tests() -> None:
    good = "```python\ndef shift_signal(s):\n    return [0] + s[:-1] if s else []\n```"
    bad = "```python\ndef shift_signal(s):\n    return s\n```"
    assert evaluate_coding(CODE_ITEM, good, allow_execution=True)["correct"] is True
    assert evaluate_coding(CODE_ITEM, bad, allow_execution=True)["correct"] is False


def test_unsafe_code_is_never_executed() -> None:
    evil = "```python\nimport os\nos.remove('x')\ndef shift_signal(s):\n    return s\n```"
    res = evaluate_coding(CODE_ITEM, evil, allow_execution=True)
    assert res["correct"] is False and res["checks"]["unsafe_code"]


def test_code_timeout() -> None:
    loop = "```python\ndef shift_signal(s):\n    while True:\n        pass\n```"
    from app.evals.profile import coding_eval

    res = coding_eval.run_code_tests(coding_eval.extract_code(loop), CODE_ITEM.code_tests, 1.0)
    assert res["passed"] is False and "zaman aşımı" in res["error"]
