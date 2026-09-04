"""app/tools — Bilimsel Araç Çalışma Zamanı (Scientific Tool Runtime).

LLM matematik/istatistik/olasılık/risk hesaplarını "kafadan" yapmak yerine
deterministik Python araçlarıyla doğrular. Her araç:
- saf numpy/pandas (scipy/statsmodels YOK),
- determinizm için seed alır (Kural 6),
- çıktısını *hipotez/test noktası* olarak çerçeveler (Kural 1 — tavsiye değil),
- ``eval``/``exec`` kullanmaz (Kural 5),
- çalıştırması ``tool_runs``/``tool_artifacts`` ile loglanabilir (denetim izi).
"""

from __future__ import annotations

from app.tools.probability_simulator import MonteCarloResult, monte_carlo_equity
from app.tools.result_verifier import (
    verify_backtest_metrics,
    verify_kelly,
    verify_probability,
)
from app.tools.statistics_checker import (
    CorrelationReport,
    StatsReport,
    correlation_report,
    describe_series,
)
from app.tools.tool_registry import (
    ToolDescriptor,
    get_tool,
    list_tools,
    resolve,
    validate_params,
)

__all__ = [
    "CorrelationReport",
    "MonteCarloResult",
    "StatsReport",
    "ToolDescriptor",
    "correlation_report",
    "describe_series",
    "get_tool",
    "list_tools",
    "monte_carlo_equity",
    "resolve",
    "validate_params",
    "verify_backtest_metrics",
    "verify_kelly",
    "verify_probability",
]
