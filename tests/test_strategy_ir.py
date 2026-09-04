import pytest

from app.trading.strategy_ir import StrategyIR, example_ir, parse_rule


def test_example_ir_valid():
    ir = example_ir()
    assert ir.name == "ema_rsi_trend_filter_v1"
    cols = ir.required_columns()
    assert "ema_20" in cols and "ema_50" in cols and "rsi_14" in cols


def test_invalid_rule_rejected():
    with pytest.raises(ValueError):
        StrategyIR(name="bad", entry_rules=["ema_20 >>> ema_50"])


def test_parse_rule():
    assert parse_rule("rsi_14 > 55") == ("rsi_14", ">", "55")


def test_roundtrip_json():
    ir = example_ir()
    js = ir.model_dump_json()
    ir2 = StrategyIR.model_validate_json(js)
    assert ir2.name == ir.name
