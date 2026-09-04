"""Bilimsel araç kayıt defteri — keşif + parametre doğrulama + tembel çözümleme.

Her araç bir ``ToolDescriptor`` ile tanımlanır (metadata; ağır modüller import
EDİLMEZ). ``resolve()`` çağrı anında ``module:function`` yolunu yükler. Böylece
``list_tools()`` numpy/pandas dışında hiçbir şeyi import etmez (çevrimdışı + hızlı).

Determinizm sözleşmesi: ``requires_seed=True`` olan araçlar ``seed`` parametresi
olmadan çalıştırılmamalı (Kural 6). ``validate_params`` bunu zorlar.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class ToolDescriptor:
    """Tek bir bilimsel aracın tanımı (yalnız metadata + tembel giriş noktası)."""

    tool_id: str
    name: str
    category: str  # probability / statistics / trading / risk / verification
    description: str
    entrypoint: str  # "module:function" — tembel çözümlenir
    version: str = "1"
    required_params: tuple[str, ...] = ()
    requires_seed: bool = False
    deterministic: bool = True
    offline: bool = True
    network: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "version": self.version,
            "required_params": list(self.required_params),
            "requires_seed": self.requires_seed,
            "deterministic": self.deterministic,
            "offline": self.offline,
            "network": self.network,
        }


_TOOLS: dict[str, ToolDescriptor] = {}


def register_tool(desc: ToolDescriptor) -> None:
    _TOOLS[desc.tool_id] = desc


def list_tools(category: str | None = None) -> list[ToolDescriptor]:
    tools = sorted(_TOOLS.values(), key=lambda d: d.tool_id)
    if category:
        tools = [t for t in tools if t.category == category]
    return tools


def get_tool(tool_id: str) -> ToolDescriptor | None:
    return _TOOLS.get(tool_id)


def validate_params(tool_id: str, params: dict[str, Any]) -> list[str]:
    """Eksik zorunlu parametreleri ve eksik seed'i (gerekliyse) listele. Boş = geçerli."""
    desc = _TOOLS.get(tool_id)
    if desc is None:
        raise KeyError(f"Bilinmeyen araç: {tool_id}")
    problems = [f"eksik parametre: {p}" for p in desc.required_params if p not in params]
    if desc.requires_seed and "seed" not in params:
        problems.append("seed zorunlu (determinizm — Kural 6)")
    return problems


def resolve(tool_id: str) -> Callable[..., Any]:
    """Aracın çağrılabilir giriş noktasını tembel yükle (``module:function``)."""
    desc = _TOOLS.get(tool_id)
    if desc is None:
        raise KeyError(f"Bilinmeyen araç: {tool_id}")
    module_path, _, func_name = desc.entrypoint.partition(":")
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name, None)
    if fn is None:  # entrypoint yanlış yapılandırılmış (fonksiyon adı hatalı) → net hata
        raise KeyError(
            f"'{func_name}' fonksiyonu {module_path} içinde yok ({tool_id} entrypoint hatalı)"
        )
    return cast("Callable[..., Any]", fn)


# --- yerleşik araçlar ------------------------------------------------------
# Yeni (app/tools) + mevcut (app/trading) araçlar tek defterde keşfedilebilir.
_BUILTINS: tuple[ToolDescriptor, ...] = (
    ToolDescriptor(
        tool_id="montecarlo",
        name="Monte Carlo equity simülatörü",
        category="probability",
        description="İşlem getirilerinden bootstrap equity yolları + risk-of-ruin (seed zorunlu).",
        entrypoint="app.tools.probability_simulator:monte_carlo_equity",
        required_params=("trade_returns",),
        requires_seed=True,
    ),
    ToolDescriptor(
        tool_id="stats-correlation",
        name="Korelasyon + permütasyon p-değeri",
        category="statistics",
        description="Pearson+Spearman ve permütasyon testi p-değeri (seed zorunlu).",
        entrypoint="app.tools.statistics_checker:correlation_report",
        required_params=("x", "y"),
        requires_seed=True,
    ),
    ToolDescriptor(
        tool_id="stats-describe",
        name="Betimsel istatistik",
        category="statistics",
        description="Tek serinin ortalama/medyan/std + örneklem-büyüklüğü uyarıları.",
        entrypoint="app.tools.statistics_checker:describe_series",
        required_params=("values",),
    ),
    ToolDescriptor(
        tool_id="verify-backtest",
        name="Backtest sonucu doğrulayıcı",
        category="verification",
        description="Gerçekçi-olmayan metrik (Sharpe>5, inf/nan, dd<-100) uyarısı.",
        entrypoint="app.tools.result_verifier:verify_backtest_metrics",
        required_params=("metrics",),
    ),
    # Mevcut araçlar (yalnız keşif/işaret — kod tekrarı yok):
    ToolDescriptor(
        tool_id="backtest",
        name="Strateji backtest motoru",
        category="trading",
        description="StrategyIR'i geçmiş veride test eder (komisyon+slippage dahil).",
        entrypoint="app.trading.backtester:run_backtest",
        required_params=("df", "ir"),
    ),
    ToolDescriptor(
        tool_id="risk",
        name="Risk raporu (Kelly/drawdown/sabit risk)",
        category="risk",
        description="Backtest sonucundan pozisyon büyüklüğü ve risk önerisi (tavsiye değil).",
        entrypoint="app.trading.risk_manager:analyze_risk",
        required_params=("strategy_name", "equity_curve", "position", "returns"),
    ),
    ToolDescriptor(
        tool_id="overfit",
        name="Örneklem-içi/dışı overfit kontrolü",
        category="trading",
        description="IS/OOS bölünmesi + statik overfit uyarıları (look-ahead şüphesi).",
        entrypoint="app.trading.overfit_checks:in_out_of_sample",
        required_params=("df", "ir"),
    ),
)

for _desc in _BUILTINS:
    register_tool(_desc)

BUILTIN_TOOL_IDS: tuple[str, ...] = tuple(t.tool_id for t in _BUILTINS)
