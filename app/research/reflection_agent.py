"""Yansıma Ajanı — backtest sonucunu analiz edip sonraki iterasyon için iyileştirme önerir.

Görevi:
  1. Önceki indikatörü ve backtest sonucunu al
  2. "Neden başarısız?" sorusunu sor
  3. Spesifik iyileştirme öner (parametre değiştir, yeni bileşen ekle, kural gevşet)
  4. Yeni StrategyIR üret
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.brain.local_llm import LLMUnavailable, LocalLLM

logger = logging.getLogger(__name__)

_REFLECTION_PROMPT = """\
Bir trading araştırmacısısın. Aşağıdaki backtest başarısız oldu. Nedeni analiz et ve \
iyileştirilmiş bir strateji öner.

━━━ ÖNCEKİ İNDİKATÖR ━━━
{indicator_summary}

━━━ BACKTEST SONUÇLARI ━━━
Yargı: {verdict}
Sebepler: {reasons}
İşlem sayısı: {n_trades}  ← 30'dan az ise kurallar ÇOK KISITLAYICI demektir
Toplam getiri: {total_return:.4f}%
Sharpe: {sharpe}
Max Drawdown: {max_dd:.4f}%

━━━ İYİLEŞTİRME KURALLARI — sebepler listesine göre uygula ━━━

EĞER "az işlem" veya işlem sayısı < 30 ise:
  → RSI eşiğini DÜŞÜR: rsi > 55 → rsi > 50 (ya da RSI kuralını kaldır)
  → EMA crossover yerine sadece tek EMA kural kullan
  → Periyotları KÜÇÜLT (50 → 20, 20 → 14)
  → Kural SILME, EKLEME yok

EĞER "aşırı drawdown" veya drawdown < -60% ise:
  → RSI eşiğini ARTIR: rsi > 50 → rsi > 55 (daha seçici giriş)
  → EMA trend filtresi EKLE: ema_20 > ema_50 (sadece yükseliş trendinde long)
  → Çıkış kuralını hızlandır: rsi < 45 yerine rsi < 48

EĞER işlem sayısı > 2000 ise (çok fazla işlem = gürültü):
  → RSI eşiğini artır: rsi > 50 → rsi > 55
  → İki koşullu entry: rsi > 55 VE ema_20 > ema_50
  → Çıkışı da katılaştır

EĞER Sharpe < -1 ve işlem sayısı > 500 ise:
  → Yön mantığını tersine çevir: rsi > eşik → rsi < eşik (mean-reversion → trend)

TEMEL KURAL: Bir iterasyonda sadece BİR şeyi değiştir. Hepsini birden değiştirme.

━━━ ÇIKTI FORMATI ━━━
{{
  "failure_analysis": "Kısa sebep analizi...",
  "changes": ["kural X silindi", "periyot Y→Z küçültüldü"],
  "improvement_reasoning": "Bu değişiklikler işlem sayısını nasıl artıracak...",
  "strategy_ir": {{
    "name": "..._v2",
    "market": "XAUUSD",
    "timeframe": "1h",
    "indicators": [
      {{"name": "RSI", "period": 14}},
      {{"name": "EMA", "period": 20}}
    ],
    "entry_rules": ["rsi_14 > 50"],
    "exit_rules": ["rsi_14 < 50"],
    "risk": {{"stop_loss": "2 * ATR"}},
    "costs": {{"commission": 0.0005, "slippage": 0.0005}}
  }}
}}

Yalnız JSON döndür. Giriş/çıkış kuralları MEVCUT indikatör kolonlarını kullanmalı \
(ör. ema_20, rsi_14, atr_14). Hayali kolon adı yazma.
"""


class ReflectionAgent:
    def __init__(self, llm: LocalLLM | None = None) -> None:
        self.llm = llm or LocalLLM()

    def reflect(
        self,
        indicator: dict[str, Any],
        backtest_result: dict[str, Any],
        verdict: str,
        reasons: list[str],
    ) -> dict[str, Any] | None:
        """Backtest sonucunu analiz et, iyileştirilmiş IR döndür."""
        metrics = backtest_result.get("metrics", {})
        oos = backtest_result.get("oos_metrics", {})

        prompt = _REFLECTION_PROMPT.format(
            indicator_summary=self._format_indicator(indicator),
            verdict=verdict,
            reasons="; ".join(reasons),
            n_trades=metrics.get("n_trades", 0),
            total_return=metrics.get("total_return_pct", 0.0),
            sharpe=metrics.get("sharpe") or "—",
            max_dd=metrics.get("max_drawdown_pct", 0.0),
            oos_return=oos.get("total_return_pct", "—") if oos else "—",
        )

        try:
            raw = self.llm.generate(prompt, fmt="json", timeout=120, max_tokens=1500)
            return self._parse(raw)
        except LLMUnavailable as exc:
            logger.warning("LLM çevrimdışı: %s", exc)
        except Exception as exc:
            logger.warning("Yansıma başarısız: %s", exc)
        return None

    def _format_indicator(self, indicator: dict[str, Any]) -> str:
        name = indicator.get("indicator_name", "bilinmiyor")
        reasoning = indicator.get("combination_reasoning", "")
        ir = indicator.get("strategy_ir", {})
        entry = ir.get("entry_rules", [])
        return f"Ad: {name}\nMantık: {reasoning[:200]}\nGiriş kuralları: {entry}"

    def _parse(self, raw: str) -> dict[str, Any] | None:
        raw = raw.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return None

        if not data.get("strategy_ir"):
            return None

        ir = data["strategy_ir"]
        ir.setdefault("costs", {"commission": 0.0005, "slippage": 0.0005})
        ir.setdefault("risk", {"stop_loss": "2 * ATR"})
        return data
