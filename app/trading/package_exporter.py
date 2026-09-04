"""Hektor Package exporter.

Bir StrategyIR'ı Entropia arayüzüne yüklenebilir .achpkg (JSON) formatına dönüştürür.
Package, Pine Script (TradingView) ve Python modülü olmak üzere iki kod çıktısı içerir.
Entropia tarafı JS/TS ile bu dosyayı okuyup kendi strategy tester'ında çalıştırır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.trading.strategy_ir import BASE_COLUMNS, StrategyIR, is_number_literal, parse_rule

ACHPKG_VERSION = "1"


@dataclass
class HektorPackage:
    name: str
    version: str
    package_type: str  # "strategy" | "indicator"
    pine_code: str
    python_code: str
    backtest_verdict: str | None = None
    backtest_metrics: dict = field(default_factory=dict)
    source: str = "achilles_research"
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "achilles_package_version": ACHPKG_VERSION,
            "name": self.name,
            "version": self.version,
            "type": self.package_type,
            "source": self.source,
            "created_at": self.created_at,
            "backtest_verdict": self.backtest_verdict,
            "backtest_metrics": self.backtest_metrics,
            "code": {
                "pine": self.pine_code,
                "python": self.python_code,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")


def export_strategy(
    ir: StrategyIR,
    *,
    version: str = "1.0.0",
    backtest_verdict: str | None = None,
    backtest_metrics: dict | None = None,
    created_at: str = "",
) -> HektorPackage:
    """StrategyIR → HektorPackage (Pine + Python kodu içerir)."""
    import datetime as dt

    return HektorPackage(
        name=ir.name,
        version=version,
        package_type="strategy",
        pine_code=ir.to_pine(),
        python_code=_ir_to_python(ir),
        backtest_verdict=backtest_verdict,
        backtest_metrics=backtest_metrics or {},
        source="achilles_research",
        created_at=created_at or dt.datetime.now(dt.UTC).isoformat(),
    )


def _rule_to_py(rule: str) -> str:
    """IR kuralını pandas boolean ifadesine çevir.

    Sayı/kolon ayrımı ``is_number_literal`` ile yapılır — backtester ile AYNI tanım.
    Eski ``float(rhs)`` denemesi ``inf``/``nan`` gibi tokenları sayı sayıyordu; backtester
    ise onları kolon sayıyordu → üretilen modül ``NameError`` veriyordu.
    """
    lhs, op, rhs = parse_rule(rule)
    if is_number_literal(rhs):
        return f'(df["{lhs}"] {op} {rhs})'
    return f'(df["{lhs}"] {op} df["{rhs}"])'


def _ir_to_python(ir: StrategyIR) -> str:
    """StrategyIR → Entropia-uyumlu Python modülü.

    Üretilen modül standart bir `compute_signals(df) -> df` arayüzü sunar.
    Giriş: open/high/low/close/volume kolonları olan DataFrame.
    Çıkış: entry_signal ve exit_signal (0/1) kolonu eklenmiş DataFrame.
    Look-ahead bias önlemi: sinyaller shift(1) ile geciktirilir.

    Kural bir kolona atıf yapıyor ama üretilen kod o kolonu HESAPLAMIYORSA
    ``ValueError`` atar (çalıştırıldığında ``KeyError`` veren, ama üretim anında
    "başarılı" görünen paket üretmek yerine).
    """
    lines: list[str] = [
        f'"""Hektor Package: {ir.name}',
        f"Market: {ir.market} | Timeframe: {ir.timeframe}",
        f"Commission: {ir.costs.commission} | Slippage: {ir.costs.slippage}",
        '"""',
        "from __future__ import annotations",
        "import pandas as pd",
        "",
        f"PACKAGE_NAME = {ir.name!r}",
        'PACKAGE_VERSION = "1.0.0"',
        "",
        "",
        "def compute_signals(df: pd.DataFrame) -> pd.DataFrame:",
        '    """Giriş/çıkış sinyallerini hesapla.',
        "    ",
        "    Beklenen kolonlar: open, high, low, close, volume",
        "    Döndürülen kolonlar: entry_signal (1=giriş), exit_signal (1=çıkış)",
        '    """',
        "    df = df.copy()",
    ]

    # `defined` = üretilen modülün GERÇEKTEN yazdığı kolonlar; kural kapsamı bununla
    # doğrulanır (bkz. fonksiyon docstring'i).
    defined: set[str] = set(BASE_COLUMNS)
    unsupported: list[str] = []

    for ind in ir.indicators:
        col = ind.column
        name = ind.name.upper()
        p = ind.period
        if name == "EMA":
            lines.append(f'    df["{col}"] = df["close"].ewm(span={p}, adjust=False).mean()')
            defined.add(col)
        elif name == "SMA":
            lines.append(f'    df["{col}"] = df["close"].rolling({p}).mean()')
            defined.add(col)
        elif name == "RSI":
            # Wilder ewm(alpha=1/p) — backtest indicators.py ile AYNI; SMA değil
            # (aksi halde dışa aktarılan strateji doğrulanan davranıştan sapar).
            lines += [
                '    _d = df["close"].diff()',
                f"    _g = _d.clip(lower=0).ewm(alpha=1/{p}, adjust=False).mean()",
                f"    _l = (-_d).clip(lower=0).ewm(alpha=1/{p}, adjust=False).mean()",
                f'    df["{col}"] = 100 - (100 / (1 + _g / _l.replace(0, 1e-10)))',
            ]
            defined.add(col)
        elif name == "ATR":
            # Wilder ewm — backtest ATR ile AYNI (SMA rolling değil).
            lines += [
                '    _hl = df["high"] - df["low"]',
                '    _hc = (df["high"] - df["close"].shift()).abs()',
                '    _lc = (df["low"] - df["close"].shift()).abs()',
                "    _tr = pd.concat([_hl, _hc, _lc], axis=1).max(axis=1)",
                f'    df["{col}"] = _tr.ewm(alpha=1/{p}, adjust=False).mean()',
            ]
            defined.add(col)
        elif name == "MACD":
            # Kanonik `{col}` = MACD ÇİZGİSİ — compute_indicator("MACD") ile aynı, yani
            # backtest'te doğrulanan kolonla birebir. Eskiden yalnız `{col}_line` yazılıyordu;
            # kurallar `{col}`e atıf yaptığı için üretilen modül KeyError veriyordu.
            lines += [
                '    _e12 = df["close"].ewm(span=12, adjust=False).mean()',
                '    _e26 = df["close"].ewm(span=26, adjust=False).mean()',
                f'    df["{col}"] = _e12 - _e26',
                f'    df["{col}_signal"] = df["{col}"].ewm(span=9, adjust=False).mean()',
                f'    df["{col}_hist"] = df["{col}"] - df["{col}_signal"]',
            ]
            defined |= {col, f"{col}_signal", f"{col}_hist"}
        elif name in ("BB", "BOLLINGER"):
            # Kanonik `{col}` = ORTA bant — compute_indicator("BB") ile aynı.
            lines += [
                f'    _mid = df["close"].rolling({p}).mean()',
                f'    _std = df["close"].rolling({p}).std()',
                f'    df["{col}_upper"] = _mid + 2 * _std',
                f'    df["{col}"]       = _mid',
                f'    df["{col}_lower"] = _mid - 2 * _std',
            ]
            defined |= {f"{col}_upper", col, f"{col}_lower"}
        elif name in ("STOCH", "STOCHASTIC"):
            lines += [
                f'    _low_n  = df["low"].rolling({p}).min()',
                f'    _high_n = df["high"].rolling({p}).max()',
                "    _range  = (_high_n - _low_n).replace(0, 1e-10)",
                '    _raw_k  = 100 * (df["close"] - _low_n) / _range',
                f'    df["{col}_k"] = _raw_k.rolling(3).mean()',
                f'    df["{col}_d"] = df["{col}_k"].rolling(3).mean()',
            ]
            defined |= {f"{col}_k", f"{col}_d"}
        elif name == "VWAP":
            lines += [
                '    _hlc3   = (df["high"] + df["low"] + df["close"]) / 3',
                '    _cum_pv = (_hlc3 * df["volume"]).cumsum()',
                '    _cum_v  = df["volume"].cumsum().replace(0, 1e-10)',
                f'    df["{col}"] = _cum_pv / _cum_v',
            ]
            defined.add(col)
        elif name in ("SUPERTREND", "ST"):
            mult = 3.0
            lines += [
                '    _st_hl  = df["high"] - df["low"]',
                '    _st_hc  = (df["high"] - df["close"].shift()).abs()',
                '    _st_lc  = (df["low"]  - df["close"].shift()).abs()',
                "    _st_tr  = pd.concat([_st_hl, _st_hc, _st_lc], axis=1).max(axis=1)",
                f"    _st_atr = _st_tr.rolling({p}).mean()",
                '    _st_hl2 = (df["high"] + df["low"]) / 2',
                f"    _st_lower = _st_hl2 - {mult} * _st_atr",
                f'    df["{col}"]     = _st_lower',
                f'    df["{col}_dir"] = (df["close"] > _st_lower).astype(int)',
            ]
            defined |= {col, f"{col}_dir"}
        else:
            lines.append(f"    # {col}: {name} desteklenmiyor — elle ekleyin")
            unsupported.append(f"{name} ({col})")

    missing = sorted(ir.rule_columns() - defined)
    if missing:
        detail = f" Desteklenmeyen gösterge(ler): {', '.join(unsupported)}." if unsupported else ""
        raise ValueError(
            f"Python dışa aktarımı eksik: kurallar {missing} kolonlarına atıf yapıyor ama "
            f"üretilen modül bunları hesaplamıyor.{detail} "
            "Çalıştırıldığında KeyError verecek bir paket üretmek yerine durduruldu."
        )

    lines.append("")

    if ir.entry_rules:
        parts = " & ".join(_rule_to_py(r) for r in ir.entry_rules)
        lines.append(f"    entry_mask = {parts}")
    else:
        lines.append("    entry_mask = pd.Series(False, index=df.index)")

    if ir.exit_rules:
        parts = " & ".join(_rule_to_py(r) for r in ir.exit_rules)
        lines.append(f"    exit_mask = {parts}")
    else:
        lines.append("    exit_mask = pd.Series(False, index=df.index)")

    lines += [
        "",
        "    # Look-ahead bias önlemi: sinyali bir bar geciktir",
        '    df["entry_signal"] = entry_mask.shift(1).fillna(False).astype(int)',
        '    df["exit_signal"]  = exit_mask.shift(1).fillna(False).astype(int)',
        "    return df",
    ]

    return "\n".join(lines)
