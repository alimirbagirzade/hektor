"""risk_manager._extract_trade_returns — per-trade maliyet bütünlüğü (CLAUDE.md kural 3).

Per-trade getiri dekompozisyonu giriş VE çıkış maliyetini içermeli. Eskiden bloklar
ham ``position`` ile sınırlanırken çıkış maliyeti (gecikmeli pozisyonla bir bar
sonra yüklendiği için) bloğun dışında kalıyor → trade getirisi şişiyor → Kelly
tahmini fazla iyimser. Testler maliyeti ``_net_returns`` üzerinden (gerçek backtester
mekaniği) üretip her iki maliyet turunun da yakalandığını kanıtlar.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.trading.backtester import _net_returns
from app.trading.risk_manager import _extract_trade_returns

_COST = 0.01  # komisyon + slippage (turn başına)


def test_trade_return_includes_entry_and_exit_cost() -> None:
    """Kapanan round-trip trade İKİ maliyet turu taşımalı (giriş + çıkış).

    position=[0,1,1,0,0], piyasa getirisi=0 → eff_pos=[0,0,1,1,0]:
    giriş maliyeti idx2'de, çıkış maliyeti idx4'te. Bileşik = (0.99)(1)(0.99)-1 = -0.0199.
    Eski (buggy) kod yalnız girişi yakalayıp -0.01 verirdi.
    """
    position = pd.Series([0, 1, 1, 0, 0], dtype=float)
    bar_ret = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
    net = _net_returns(position, bar_ret, _COST)

    trades = _extract_trade_returns(position, net)

    assert len(trades) == 1
    assert trades.iloc[0] == pytest.approx(-0.0199, abs=1e-9)  # iki maliyet turu
    # Regresyon koruması: tek tur (-0.01) ESKİ buggy davranıştı.
    assert trades.iloc[0] < -0.015


def test_trade_return_composes_gross_with_costs() -> None:
    """Brüt getiri + giriş/çıkış maliyeti birlikte bileşik olmalı.

    Tutuş barında +%5 piyasa getirisi (idx2, eff_pos=1) → net idx: [0,0,0.05-0.01,0,-0.01]
    = [0,0,0.04,0,-0.01]. Bileşik = (1.04)(1)(0.99)-1 = 0.0296.
    """
    position = pd.Series([0, 1, 1, 0, 0], dtype=float)
    bar_ret = pd.Series([0.0, 0.0, 0.05, 0.0, 0.0])  # getiri eff_pos=1 olan idx2'de gerçekleşir
    net = _net_returns(position, bar_ret, _COST)

    trades = _extract_trade_returns(position, net)

    assert len(trades) == 1
    assert trades.iloc[0] == pytest.approx(1.04 * 0.99 - 1.0, abs=1e-9)  # 0.0296


def test_open_position_at_end_has_no_exit_cost() -> None:
    """Seri sonunda hâlâ açık pozisyon: çıkış gerçekleşmedi → yalnız giriş maliyeti.

    position=[0,1,1] → eff_pos=[0,0,1]: giriş idx2'de, çıkış yok. Trade = -0.01.
    """
    position = pd.Series([0, 1, 1], dtype=float)
    bar_ret = pd.Series([0.0, 0.0, 0.0])
    net = _net_returns(position, bar_ret, _COST)

    trades = _extract_trade_returns(position, net)

    assert len(trades) == 1
    assert trades.iloc[0] == pytest.approx(-0.01, abs=1e-9)  # yalnız giriş maliyeti


def test_no_position_returns_empty() -> None:
    """Hiç pozisyon yoksa boş seri döner."""
    position = pd.Series([0, 0, 0], dtype=float)
    bar_ret = pd.Series([0.01, -0.02, 0.03])
    net = _net_returns(position, bar_ret, _COST)

    trades = _extract_trade_returns(position, net)
    assert trades.empty


def test_two_distinct_trades_each_carry_exit_cost() -> None:
    """İki ayrı trade'in HER biri kendi çıkış maliyetini taşımalı; sayı = giriş sayısı.

    position=[0,1,0,1,0,0] → iki ayrı round-trip; her ikisi de seri içinde kapanır
    (çıkış maliyeti eff_pos 1→0 ile bir sonraki barda yüklenir, son bar bunu içerir).
    Her biri -0.0199 (giriş+çıkış).
    """
    position = pd.Series([0, 1, 0, 1, 0, 0], dtype=float)
    bar_ret = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    net = _net_returns(position, bar_ret, _COST)

    trades = _extract_trade_returns(position, net)

    assert len(trades) == 2
    for t in trades:
        assert t == pytest.approx(-0.0199, abs=1e-9)
