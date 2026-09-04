"""evaluate_model.check_flags kırmızı bayrak testleri (Kural 2 disiplini)."""

from __future__ import annotations

from app.training.evaluate_model import check_flags


def test_success_without_test_flagged() -> None:
    # Test/backtest'ten söz etmeden "çalışıyor/başarılı" demek Kural 2 ihlali.
    flags = check_flags("Bu yaklaşım çalışıyor ve başarılı sonuç verir.", [])
    assert "success_without_test" in flags


def test_success_with_backtest_not_flagged() -> None:
    flags = check_flags("Backtest sonrası bu yaklaşım başarılı görünüyor.", [])
    assert "success_without_test" not in flags


def test_guaranteed_profit_flagged() -> None:
    flags = check_flags("Bu yöntem garanti kâr sağlar.", [])
    assert "guaranteed_profit" in flags


def test_clean_answer_no_flags() -> None:
    flags = check_flags("RSI bir momentum osilatörüdür; 0-100 aralığında.", [])
    assert flags == []
