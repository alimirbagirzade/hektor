"""StrategyIR regresyon testleri: maliyet doğrulaması + Pine `ta.bb` demet sırası.

İki bulgu, aynı dosya (``app/trading/strategy_ir.py``):

(A) ``CostSpec`` sıfır ve NEGATİF komisyon/slippage'i doğrulamasız kabul ediyordu.
    Negatif maliyet ``_net_returns``'te ``turnover * |maliyet|`` kadar getiri EKLER
    (maliyeti ödüle çevirir); sıfır toplam maliyetsiz backtest demektir — ikisi de
    Kural 3 ihlali.

(B) Pine ihracında ``ta.bb`` demeti yanlış sırada çözülüyordu; kanonik ``{col}``
    (ORTA bant) TradingView'de ÜST bandı alıyor, backtest ile Pine çelişiyordu.

Testler çevrimdışıdır (Ollama/veri gerektirmez).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.trading.strategy_ir import CostSpec, IndicatorSpec, StrategyIR, example_ir

# --------------------------------------------------------------------------------------
# (A) Maliyet doğrulaması — Kural 3
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("commission", "slippage"),
    [
        (-0.01, 0.0),
        (0.0, -0.01),
        (-0.0005, -0.0005),
        (-1e-9, 0.001),
    ],
)
def test_negatif_maliyet_reddedilir(commission: float, slippage: float) -> None:
    """Negatif maliyet maliyeti ödüle çevirir → kurulum anında reddedilmeli."""
    with pytest.raises(ValidationError):
        CostSpec(commission=commission, slippage=slippage)


def test_sifir_toplam_maliyet_reddedilir() -> None:
    """commission + slippage == 0 → maliyetsiz backtest (Kural 3) → reddedilir."""
    with pytest.raises(ValidationError, match="Kural 3"):
        CostSpec(commission=0.0, slippage=0.0)


def test_tek_bilesen_sifir_kabul_edilir() -> None:
    """Komisyonsuz broker + gerçek slippage meşrudur; toplam > 0 olduğu sürece geçer."""
    assert CostSpec(commission=0.0, slippage=0.001).slippage == 0.001
    assert CostSpec(commission=0.001, slippage=0.0).commission == 0.001


def test_varsayilanlar_degismedi() -> None:
    """Mevcut meşru çağrılar bozulmamalı: varsayılan 0.0005 + 0.0005."""
    c = CostSpec()
    assert c.commission == 0.0005
    assert c.slippage == 0.0005


def test_inf_ve_nan_maliyet_reddedilir() -> None:
    """inf/nan maliyet tüm metrik serisini sessizce nan'a çevirirdi."""
    with pytest.raises(ValidationError):
        CostSpec(commission=float("inf"))
    with pytest.raises(ValidationError):
        CostSpec(slippage=float("nan"))


def test_strategy_ir_json_girdisinde_negatif_maliyet_reddedilir() -> None:
    """Doğrulanmış girdi yolu: API gövdesi / LLM JSON'u → ``model_validate``.

    Bu, ``POST /api/backtest`` ve ``research.orchestrator`` hattının kullandığı yoldur;
    sentez motorları yalnız ``setdefault('costs', ...)`` yaptığı için LLM açıkça
    ``{"commission": -0.01, ...}`` verirse tek kapı burasıdır.
    """
    payload = {
        "name": "negatif_maliyet_v1",
        "indicators": [{"name": "EMA", "period": 20}],
        "entry_rules": ["ema_20 > 100"],
        "exit_rules": ["ema_20 < 100"],
        "costs": {"commission": -0.01, "slippage": 0.0},
    }
    with pytest.raises(ValidationError):
        StrategyIR.model_validate(payload)

    payload["costs"] = {"commission": 0.0, "slippage": 0.0}
    with pytest.raises(ValidationError):
        StrategyIR.model_validate(payload)

    payload["costs"] = {"commission": 0.0005, "slippage": 0.0005}
    ir = StrategyIR.model_validate(payload)
    assert ir.costs.commission + ir.costs.slippage > 0


def test_example_ir_gecerli_kalir() -> None:
    """Referans IR yeni kısıtları sağlamalı (regresyon emniyeti)."""
    ir = example_ir()
    assert ir.costs.commission > 0
    assert ir.costs.slippage > 0


# --------------------------------------------------------------------------------------
# (B) Pine `ta.bb` demet sırası — backtest/Pine paritesi
# --------------------------------------------------------------------------------------


def _bb_ir() -> StrategyIR:
    return StrategyIR(
        name="bb_parite_v1",
        indicators=[IndicatorSpec(name="BB", period=20)],
        entry_rules=["close > bb_20"],
        exit_rules=["close < bb_20"],
    )


def test_pine_bb_demeti_middle_upper_lower_sirasinda() -> None:
    """Pine v5 ``ta.bb`` [middle, upper, lower] döner; kanonik ``bb_20`` = ORTA bant."""
    pine = _bb_ir().to_pine()
    assert "[bb_20, bb_20_upper, bb_20_lower] = ta.bb(close, 20, 2)" in pine
    # Eski (hatalı) sıra bir daha üretilmemeli.
    assert "[bb_20_upper, bb_20, bb_20_lower]" not in pine


def test_pine_bb_kanonik_kolonu_python_ihraciyla_ayni_bandi_gosterir() -> None:
    """Parite: Python ihracında ``df["bb_20"] = _mid``; Pine'da da ``bb_20`` orta bant.

    Kural ``close > bb_20`` iki artefaktta AYNI bandı kıyaslamalı; aksi hâlde
    TradingView sonucu doğrulanmış verdict'ten sapar.
    """
    from app.trading.package_exporter import _ir_to_python

    ir = _bb_ir()
    py = _ir_to_python(ir)
    pine = ir.to_pine()

    # Python tarafı: kanonik kolon orta banda (_mid) atanır.
    assert 'df["bb_20"]       = _mid' in py
    # Pine tarafı: kanonik kolon demetin İLK (middle) elemanıdır.
    bb_line = next(ln for ln in pine.splitlines() if "ta.bb(" in ln)
    targets = bb_line.split("]")[0].lstrip("[").split(",")
    assert [t.strip() for t in targets] == ["bb_20", "bb_20_upper", "bb_20_lower"]


def test_pine_bb_kurallari_tanimli_degiskene_atif_yapar() -> None:
    """Sıra düzeltmesi ``defined`` kümesini bozmamalı: üst/alt bant kuralları da geçer."""
    ir = StrategyIR(
        name="bb_bantlar_v1",
        indicators=[IndicatorSpec(name="BB", period=20)],
        entry_rules=["close > bb_20_upper"],
        exit_rules=["close < bb_20_lower"],
    )
    pine = ir.to_pine()
    assert "entryCondition = close > bb_20_upper" in pine
    assert "exitCondition  = close < bb_20_lower" in pine
