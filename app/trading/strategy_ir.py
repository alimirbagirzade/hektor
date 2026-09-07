"""Strategy Intermediate Representation (IR).

A JSON-serializable description of a strategy: indicators, entry/exit rules,
risk, and cost model. Rules are simple comparison expressions over computed
indicator columns (e.g. "ema_20 > ema_50", "rsi_14 > 55").

The IR is intentionally restrictive and declarative so it is safe to evaluate
(no arbitrary code execution) and easy to translate to Pine/MQL5 later.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

_RULE_RE = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(<|<=|>|>=|==|!=)\s*([a-zA-Z_][a-zA-Z0-9_]*|-?\d+(?:\.\d+)?)\s*$"
)

# Kural sağ tarafında OHLCV kolonları her zaman vardır (gösterge tanımı gerektirmez).
# Backtest DataFrame'i, üretilen Python modülü ve Pine (builtin) için ortak taban.
BASE_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def is_number_literal(token: str) -> bool:
    """Kural sağ tarafı sayı sabiti mi? ``_RULE_RE``'nin sayı dalıyla AYNI tanım.

    Tek kaynak: backtester (``_eval_rules``), ``required_columns`` ve dışa aktarım
    kod üreteçleri bunu kullanır. Ayrı ayrı yazılmış sezgisel kontroller
    (``isdigit``, ``float()`` denemesi) birbirinden sapıyordu; ``inf``/``nan`` gibi
    tokenlar bir tarafta kolon, diğerinde sayı sayılıyordu.
    """
    return bool(_NUMBER_RE.match(token))


class IndicatorSpec(BaseModel):
    name: str
    period: int = 14

    @property
    def column(self) -> str:
        return f"{self.name.lower()}_{self.period}"


class RiskSpec(BaseModel):
    stop_loss: str | None = "2 * ATR"
    take_profit: str | None = None
    position_size: str = "fixed_fractional"


class CostSpec(BaseModel):
    """İşlem maliyeti modeli — işlem başına oransal komisyon ve slippage.

    Kural 3 ("maliyetleri yok sayma") burada alan düzeyinde uygulanır; çünkü bu
    değerler doğrudan ``_net_returns(..., cost_per_turn=commission + slippage)``a
    gider ve backtest metriklerini (Sharpe/getiri) belirler:

    * **Negatif değer REDDEDİLİR.** Negatif maliyet her pozisyon değişiminde net
      getiriye ``turnover * |maliyet|`` EKLER, yani maliyeti ödüle çevirir; çok
      işlemli bir strateji şişirilmiş Sharpe ile ``evaluate()`` kapısını geçebilir.
      Bu asla meşru bir girdi değildir.
    * **TOPLAM maliyet sıfır REDDEDİLİR** (``commission + slippage <= 0``). Sıfır
      toplam = maliyetsiz backtest; Kural 3'ün tam olarak yasakladığı şey. Tek tek
      bileşenin 0 olması meşrudur (komisyonsuz broker + gerçek slippage, ya da
      spread'in tamamı komisyona katlanmış modelleme), bu yüzden alan başına ``gt``
      değil toplam üzerinden kapı konur.
    * ``allow_inf_nan=False``: ``inf``/``nan`` maliyet tüm metrik serisini sessizce
      ``nan``a çevirirdi.

    Varsayılanlar (0.0005 + 0.0005) DEĞİŞMEDİ. Doğrulama yalnız kurulum/``model_validate``
    anında çalışır (``validate_assignment`` kapalı): sınır girdileri (API gövdesi,
    LLM'in ürettiği JSON) burada yakalanır.
    """

    commission: float = Field(default=0.0005, ge=0.0, allow_inf_nan=False)
    slippage: float = Field(default=0.0005, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _reject_costless(self) -> CostSpec:
        if self.commission + self.slippage <= 0.0:
            raise ValueError(
                "Maliyetsiz backtest yasak (Kural 3): commission + slippage > 0 olmalı; "
                f"verilen commission={self.commission}, slippage={self.slippage}. "
                "Komisyonsuz broker modelliyorsan slippage'ı gerçekçi bir değerle ver."
            )
        return self


class StrategyIR(BaseModel):
    name: str
    market: str = "XAUUSD"
    timeframe: str = "15m"
    indicators: list[IndicatorSpec] = Field(default_factory=list)
    entry_rules: list[str] = Field(default_factory=list)
    exit_rules: list[str] = Field(default_factory=list)
    risk: RiskSpec = Field(default_factory=RiskSpec)
    costs: CostSpec = Field(default_factory=CostSpec)

    @field_validator("entry_rules", "exit_rules")
    @classmethod
    def _validate_rules(cls, rules: list[str]) -> list[str]:
        for r in rules:
            if not _RULE_RE.match(r):
                raise ValueError(
                    f"Geçersiz kural: {r!r}. Biçim: '<col> <op> <col|sayı>' örn. 'ema_20 > ema_50'"
                )
        return rules

    def required_columns(self) -> set[str]:
        return {ind.column for ind in self.indicators} | self.rule_columns()

    def rule_columns(self) -> set[str]:
        """Yalnız kuralların referans verdiği kolonlar (gösterge listesi hariç).

        Dışa aktarım üreteçleri bunu "üretilen kod hangi kolonları tanımlamak
        ZORUNDA" sorusunu yanıtlamak için kullanır.
        """
        cols: set[str] = set()
        for rule in [*self.entry_rules, *self.exit_rules]:
            m = _RULE_RE.match(rule)
            if m:
                lhs, _, rhs = m.groups()
                cols.add(lhs)
                if not is_number_literal(rhs):
                    cols.add(rhs)
        return cols

    def to_pine(self) -> str:
        """StrategyIR → TradingView Pine Script v5 (taslak).

        Kural bir kolona atıf yapıyor ama üretilen kod o değişkeni TANIMLAMIYORSA
        ``ValueError`` atar. Eskiden desteklenmeyen gösterge sessizce yorum satırına
        düşüyor, ``entryCondition`` tanımsız değişkene atıf yapıyordu → TradingView'a
        yapıştırıldığında derlenmeyen, ama üretim anında "başarılı" görünen çıktı.
        """
        # commission_value in Pine expects percent; strip trailing zeros for readability
        commission_pct = self.costs.commission * 100
        commission_str = f"{commission_pct:.4f}".rstrip("0").rstrip(".")
        # Strateji adı Pine string literalinin içine girer → tırnak/ters bölü kaçışlanmalı.
        safe_name = self.name.replace("\\", "\\\\").replace('"', '\\"')
        lines: list[str] = [
            "//@version=5",
            (
                f'strategy("{safe_name}", overlay=true,'
                f" commission_type=strategy.commission.percent,"
                f" commission_value={commission_str})"
            ),
            "",
        ]
        # Indikatör tanımları. `defined` = üretilen Pine kodunun GERÇEKTEN tanımladığı
        # değişkenler; kural kapsamı bunun üzerinden doğrulanır.
        defined: set[str] = set(BASE_COLUMNS)
        unsupported: list[str] = []
        for ind in self.indicators:
            col = ind.column
            n = ind.name.upper()
            if n == "EMA":
                lines.append(f"{col} = ta.ema(close, {ind.period})")
                defined.add(col)
            elif n == "SMA":
                lines.append(f"{col} = ta.sma(close, {ind.period})")
                defined.add(col)
            elif n == "RSI":
                lines.append(f"{col} = ta.rsi(close, {ind.period})")
                defined.add(col)
            elif n == "ATR":
                lines.append(f"{col} = ta.atr({ind.period})")
                defined.add(col)
            elif n == "MACD":
                # Kanonik `{col}` = MACD ÇİZGİSİ — compute_indicator("MACD") ile aynı.
                # Eskiden yalnız `{col}_line` tanımlanıyordu; kurallar `{col}`e atıf
                # yaptığı için üretilen kod tanımsız değişken kullanıyordu.
                lines.append(f"[{col}, {col}_signal, {col}_hist] = ta.macd(close, 12, 26, 9)")
                defined |= {col, f"{col}_signal", f"{col}_hist"}
            elif n in ("BB", "BOLLINGER"):
                # Kanonik `{col}` = ORTA bant — compute_indicator("BB") ve Python
                # ihracındaki `df["{col}"] = _mid` ile aynı.
                # Pine v5 `ta.bb` demeti [middle, upper, lower] SIRASIYLA döner. Eskiden
                # `[{col}_upper, {col}, {col}_lower]` yazılıyordu; bu, ORTA bandı
                # `{col}_upper`a, ÜST bandı kanonik `{col}`e bağlıyordu. Sonuç: "close >
                # bb_20" kuralı backtest'te orta banda, TradingView'de üst banda kıyaslanıyor,
                # aynı paketin iki kod artefaktı çelişiyordu.
                lines.append(f"[{col}, {col}_upper, {col}_lower] = ta.bb(close, {ind.period}, 2)")
                defined |= {f"{col}_upper", col, f"{col}_lower"}
            elif n in ("STOCH", "STOCHASTIC"):
                # IndicatorSpec yalnız `period` taşır; %K/%D yumuşatma Pine varsayılanı 3.
                lines.append(f"{col}_k = ta.sma(ta.stoch(close, high, low, {ind.period}), 3)")
                lines.append(f"{col}_d = ta.sma({col}_k, 3)")
                defined |= {f"{col}_k", f"{col}_d"}
            elif n == "VWAP":
                lines.append(f"{col} = ta.vwap(hlc3)")
                defined.add(col)
            elif n in ("SUPERTREND", "ST"):
                # IndicatorSpec çarpan taşımaz; Pine varsayılanı 3.0 sabit.
                lines.append(f"[{col}, {col}_dir] = ta.supertrend(3.0, {ind.period})")
                defined |= {col, f"{col}_dir"}
            else:
                lines.append(f"// {col} = ???  /* {n} desteklenmiyor */")
                unsupported.append(f"{n} ({col})")
        lines.append("")

        missing = sorted(self.rule_columns() - defined)
        if missing:
            detail = (
                f" Desteklenmeyen gösterge(ler): {', '.join(unsupported)}." if unsupported else ""
            )
            raise ValueError(
                f"Pine dışa aktarımı eksik: kurallar {missing} kolonlarına atıf yapıyor ama "
                f"üretilen kod bunları tanımlamıyor.{detail} "
                "Derlenmeyen bir Pine betiği üretmek yerine durduruldu."
            )

        def _rule_to_pine(rule: str) -> str:
            lhs, op, rhs = parse_rule(rule)
            return f"{lhs} {op} {rhs}"

        entry_cond = " and ".join(_rule_to_pine(r) for r in self.entry_rules) or "false"
        exit_cond = " and ".join(_rule_to_pine(r) for r in self.exit_rules) or "false"

        lines += [
            f"entryCondition = {entry_cond}",
            f"exitCondition  = {exit_cond}",
            "",
            "if entryCondition",
            '    strategy.entry("Long", strategy.long)',
            "if exitCondition",
            '    strategy.close("Long")',
        ]
        return "\n".join(lines)


def parse_rule(rule: str) -> tuple[str, str, str]:
    m = _RULE_RE.match(rule)
    if not m:
        raise ValueError(f"Geçersiz kural: {rule!r}")
    return m.group(1), m.group(2), m.group(3)


def example_ir() -> StrategyIR:
    return StrategyIR(
        name="ema_rsi_trend_filter_v1",
        market="XAUUSD",
        timeframe="15m",
        indicators=[
            IndicatorSpec(name="EMA", period=20),
            IndicatorSpec(name="EMA", period=50),
            IndicatorSpec(name="RSI", period=14),
            IndicatorSpec(name="ATR", period=14),
        ],
        entry_rules=["ema_20 > ema_50", "rsi_14 > 55"],
        exit_rules=["ema_20 < ema_50", "rsi_14 < 45"],
        risk=RiskSpec(stop_loss="2 * ATR", position_size="fixed_fractional"),
        costs=CostSpec(commission=0.0005, slippage=0.0005),
    )
