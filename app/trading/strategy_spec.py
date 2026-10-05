"""strategy_spec.py — sohbetten çıkarılan, TEST EDİLEBİLİR strateji tanımı (Faz 2B).

``StrategyIR``'den farkı: stop, pozisyon büyüklüğü, yön ve maliyet birimleri AÇIKTIR; hiçbir
alan sessizce varsayılana düşmez. Kurallar ``StrategyIR`` ile aynı güvenli regex dilindedir
(``<kolon> <op> <kolon|sayı>``; eval/exec yok — Kural 5).

Desteklenen stop türleri (sınırlı ve açık):
- ``none``         : stop yok (risk tabanlı boyutlandırma ile kullanılamaz).
- ``fixed_pct``    : giriş fiyatından sabit yüzde (``value`` = 0.02 → %2). Pozisyon boyunca SABİT.
- ``atr_initial``  : giriş anında ``value × ATR(atr_period)``; ATR SİNYAL BARININ kapanışındaki
                     değerdir (karar anında bilinen). Pozisyon boyunca SABİT kalır.
- ``atr_trailing`` : başlangıçta ``atr_initial`` gibi; her bar KAPANIŞINDA
                     ``kapanış ∓ value × ATR`` ile yalnız lehe güncellenir, bir sonraki bardan
                     itibaren geçerlidir (takip eden stop).

Hedef (take-profit): ``none`` | ``fixed_pct`` | ``atr_multiple`` (giriş anı ATR) |
``r_multiple`` (stop mesafesinin katı; stop gerektirir).

Pozisyon büyüklüğü:
- ``fixed_fraction`` : her işlemde özsermayenin ``fraction`` katı kadar NOMİNAL (1.0 = tam
                       özsermaye, kaldıraçsız). ``fraction ≤ max_leverage``.
- ``risk_per_trade`` : stop'a kadar kayıp özsermayenin ``risk_pct``'si olacak şekilde nominal;
                       ``max_leverage × özsermaye`` ile sınırlı. Stop gerektirir.

Maliyetler (hepsi ZORUNLU, birimleri açık; eksik değer REDDEDİLİR):
- ``commission_bps_per_side`` : her dolumda nominalin baz puanı (1 bp = %0.01).
- ``slippage_bps_per_side``   : her dolumda fiyatın aleyhe kayması (bp).
- ``spread_bps``              : alış-satış makası (bp, TAM makas; her dolumda yarısı aleyhe).
- ``funding_bps_per_day``     : açık pozisyonun nominali üzerinden günlük fonlama/taşıma (bp).
Sıfır değer yalnız ``zero_reasons[alan]`` ile (≥10 karakter gerekçe) kabul edilir ve raporda
görünür. Her maliyetin pozitif olması zorunlu DEĞİLDİR (komisyonsuz broker, fonlamasız spot).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.trading.strategy_ir import _RULE_RE, BASE_COLUMNS, IndicatorSpec, is_number_literal

# compute_indicator'ın tanıdığı adlar (kolon = <ad>_<periyot>, küçük harf).
SUPPORTED_INDICATORS = frozenset(
    {
        "ema",
        "sma",
        "rsi",
        "atr",
        "macd",
        "entropy",
        "permentropy",
        "forbidden",
        "forbiddenrate",
        "complexity",
        "complexityentropy",
        "bollinger",
        "bb",
    }
)
_COL_RE = re.compile(r"^([a-z]+)_(\d{1,4})$")

COST_FIELDS = (
    "commission_bps_per_side",
    "slippage_bps_per_side",
    "spread_bps",
    "funding_bps_per_day",
)
COST_LABELS = {
    "commission_bps_per_side": "Komisyon (bp / dolum)",
    "slippage_bps_per_side": "Kayma (bp / dolum, aleyhe)",
    "spread_bps": "Makas (bp, tam; dolum başına yarısı)",
    "funding_bps_per_day": "Fonlama/taşıma (bp / gün, açık nominal)",
}

STOP_TYPES = ("none", "fixed_pct", "atr_initial", "atr_trailing")
TP_TYPES = ("none", "fixed_pct", "atr_multiple", "r_multiple")
SIZING_TYPES = ("fixed_fraction", "risk_per_trade")


def column_supported(col: str) -> bool:
    if col in BASE_COLUMNS:
        return True
    m = _COL_RE.match(col)
    return bool(m and m.group(1) in SUPPORTED_INDICATORS and int(m.group(2)) >= 1)


def rule_problem(rule: str) -> str:
    """Kural desteklenmiyorsa nedeni; destekleniyorsa ``""``."""
    m = _RULE_RE.match(rule or "")
    if not m:
        return "kural dili desteklemiyor (biçim: '<kolon> <op> <kolon|sayı>')"
    lhs, _op, rhs = m.groups()
    for col in (lhs, rhs):
        if is_number_literal(col):
            continue
        if not column_supported(col):
            return f"'{col}' hesaplanabilen bir kolon değil"
    if is_number_literal(lhs):
        return "sol taraf kolon olmalı"
    return ""


class StopSpec(BaseModel):
    type: Literal["none", "fixed_pct", "atr_initial", "atr_trailing"]
    value: float | None = Field(default=None, allow_inf_nan=False)
    atr_period: int = Field(default=14, ge=1, le=500)

    @model_validator(mode="after")
    def _check(self) -> StopSpec:
        if self.type == "none":
            if self.value not in (None, 0, 0.0):
                raise ValueError("Stop türü 'none' iken değer verilemez.")
            self.value = None
            return self
        if self.value is None or self.value <= 0:
            raise ValueError(f"Stop '{self.type}' için pozitif değer gerekli.")
        if self.type == "fixed_pct" and self.value >= 1:
            raise ValueError("fixed_pct stop oran olarak verilir (0.02 = %2); 1'den küçük olmalı.")
        return self


class TakeProfitSpec(BaseModel):
    type: Literal["none", "fixed_pct", "atr_multiple", "r_multiple"] = "none"
    value: float | None = Field(default=None, allow_inf_nan=False)
    atr_period: int = Field(default=14, ge=1, le=500)

    @model_validator(mode="after")
    def _check(self) -> TakeProfitSpec:
        if self.type == "none":
            self.value = None
            return self
        if self.value is None or self.value <= 0:
            raise ValueError(f"Hedef '{self.type}' için pozitif değer gerekli.")
        return self


class SizingSpec(BaseModel):
    type: Literal["fixed_fraction", "risk_per_trade"]
    fraction: float | None = Field(default=None, allow_inf_nan=False)
    risk_pct: float | None = Field(default=None, allow_inf_nan=False)
    max_leverage: float = Field(default=1.0, gt=0, le=20, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check(self) -> SizingSpec:
        if self.type == "fixed_fraction":
            if self.fraction is None or self.fraction <= 0:
                raise ValueError("fixed_fraction için pozitif 'fraction' gerekli (1.0 = tam).")
            if self.fraction > self.max_leverage:
                raise ValueError("fraction, max_leverage'ı aşamaz.")
        else:
            if self.risk_pct is None or not (0 < self.risk_pct <= 0.1):
                raise ValueError("risk_per_trade için 0 < risk_pct ≤ 0.1 (oran) gerekli.")
        return self


class CostModel(BaseModel):
    """Birimleri açık maliyetler; eksik değer reddedilir, sıfır yalnız gerekçeyle."""

    commission_bps_per_side: float | None = Field(default=None, allow_inf_nan=False)
    slippage_bps_per_side: float | None = Field(default=None, allow_inf_nan=False)
    spread_bps: float | None = Field(default=None, allow_inf_nan=False)
    funding_bps_per_day: float | None = Field(default=None, allow_inf_nan=False)
    zero_reasons: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> CostModel:
        missing = [f for f in COST_FIELDS if getattr(self, f) is None]
        if missing:
            names = ", ".join(COST_LABELS[f] for f in missing)
            raise ValueError(f"Maliyet eksik: {names}. Değer girin (sıfırsa gerekçesiyle).")
        for f in COST_FIELDS:
            v = float(getattr(self, f))
            if v < 0:
                raise ValueError(f"{COST_LABELS[f]} negatif olamaz (maliyet ödüle dönerdi).")
            if v > 1000:
                raise ValueError(f"{COST_LABELS[f]} {v} bp — gerçek dışı (>%10).")
            if v == 0 and len((self.zero_reasons.get(f) or "").strip()) < 10:
                raise ValueError(
                    f"{COST_LABELS[f]} sıfır: açık gerekçe gerekli (≥10 karakter; ör. "
                    "'komisyonsuz broker', 'spot, fonlama yok')."
                )
        self.zero_reasons = {
            k: v.strip() for k, v in self.zero_reasons.items() if getattr(self, k, None) == 0
        }
        return self


class UnsupportedRule(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    why: str = Field(default="", max_length=500)
    important: bool = True


class TestableStrategy(BaseModel):
    """Sohbet cevabından çıkarılmış, okunur ve test edilebilir strateji."""

    __test__ = False  # pytest bunu test sınıfı sanmasın (ad "Test" ile başlıyor)

    name: str = Field(..., min_length=1, max_length=120)
    market: str = Field(..., min_length=1, max_length=40)
    timeframe: str = Field(..., min_length=1, max_length=10)
    direction: Literal["long", "short"]
    indicators: list[IndicatorSpec] = Field(default_factory=list)
    entry_rules: list[str] = Field(..., min_length=1)
    exit_rules: list[str] = Field(default_factory=list)
    stop: StopSpec
    take_profit: TakeProfitSpec = Field(default_factory=TakeProfitSpec)
    sizing: SizingSpec
    costs: CostModel
    unsupported_rules: list[UnsupportedRule] = Field(default_factory=list)
    # Basitleştirilmiş strateji: ana stratejiden AÇIKÇA çıkarılan kurallar + ana kimlik.
    simplified_from: str = ""
    removed_rules: list[str] = Field(default_factory=list)

    @field_validator("entry_rules", "exit_rules")
    @classmethod
    def _rules(cls, rules: list[str]) -> list[str]:
        out = []
        for r in rules:
            r = (r or "").strip()
            why = rule_problem(r)
            if why:
                raise ValueError(f"Kural '{r}': {why}")
            out.append(r)
        return out

    @model_validator(mode="after")
    def _cross(self) -> TestableStrategy:
        from app.trading.backtester import _BARS_PER_YEAR

        if self.timeframe not in _BARS_PER_YEAR:
            raise ValueError(
                f"Zaman dilimi '{self.timeframe}' tanınmıyor (izinli: {', '.join(_BARS_PER_YEAR)})."
            )
        if self.sizing.type == "risk_per_trade" and self.stop.type == "none":
            raise ValueError("Risk tabanlı boyutlandırma stop olmadan tanımsız.")
        if self.take_profit.type == "r_multiple" and self.stop.type == "none":
            raise ValueError("R-katı hedef stop olmadan tanımsız.")
        if self.simplified_from and not self.removed_rules:
            raise ValueError("Basitleştirilmiş strateji çıkarılan kuralları açıkça listelemeli.")
        return self

    # ── kimlik ───────────────────────────────────────────────────────────────

    def canonical(self) -> dict[str, Any]:
        """Test sonucunu belirleyen alanlar (ad hariç) — kimlik özetinin girdisi."""
        data = self.model_dump(mode="json")
        data.pop("name", None)
        return data

    def content_hash(self) -> str:
        blob = json.dumps(self.canonical(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def strategy_id(self) -> str:
        return "st_" + self.content_hash()[:16]

    def important_unsupported(self) -> list[UnsupportedRule]:
        return [u for u in self.unsupported_rules if u.important]

    def required_columns(self) -> set[str]:
        cols: set[str] = set()
        for rule in [*self.entry_rules, *self.exit_rules]:
            m = _RULE_RE.match(rule)
            if m:
                lhs, _, rhs = m.groups()
                cols.add(lhs)
                if not is_number_literal(rhs):
                    cols.add(rhs)
        return {c for c in cols if c not in BASE_COLUMNS}

    def readable(self) -> dict[str, Any]:
        """Kullanıcının okuyacağı Türkçe özet (form yanında)."""
        stop = {
            "none": "Stop YOK",
            "fixed_pct": f"Sabit %{(self.stop.value or 0) * 100:g} (giriş fiyatından, sabit)",
            "atr_initial": f"{self.stop.value:g} × ATR({self.stop.atr_period}) — girişte "
            "hesaplanır, SABİT kalır",
            "atr_trailing": f"{self.stop.value:g} × ATR({self.stop.atr_period}) — TAKİP EDEN "
            "(bar kapanışında yalnız lehe güncellenir)",
        }[self.stop.type]
        tp = {
            "none": "Hedef yok",
            "fixed_pct": f"Sabit %{(self.take_profit.value or 0) * 100:g}",
            "atr_multiple": f"{self.take_profit.value} × ATR({self.take_profit.atr_period})",
            "r_multiple": f"{self.take_profit.value} R (stop mesafesinin katı)",
        }[self.take_profit.type]
        if self.sizing.type == "fixed_fraction":
            size = f"Özsermayenin {self.sizing.fraction:g} katı nominal (kaldıraç ≤ "
            size += f"{self.sizing.max_leverage:g})"
        else:
            size = f"İşlem başına özsermayenin %{(self.sizing.risk_pct or 0) * 100:g} riski "
            size += f"(stop'a göre; kaldıraç ≤ {self.sizing.max_leverage:g})"
        return {
            "Piyasa": self.market,
            "Zaman dilimi": self.timeframe,
            "Yön": "Uzun (long)" if self.direction == "long" else "Kısa (short)",
            "Giriş (hepsi doğru)": self.entry_rules,
            "Çıkış (hepsi doğru)": self.exit_rules or ["(kural yok — yalnız stop/hedef)"],
            "Stop": stop,
            "Hedef": tp,
            "Pozisyon büyüklüğü": size,
            "Maliyetler": {COST_LABELS[f]: getattr(self.costs, f) for f in COST_FIELDS},
            "Sıfır maliyet gerekçeleri": self.costs.zero_reasons,
            "Desteklenmeyen kurallar": [u.model_dump() for u in self.unsupported_rules],
        }
