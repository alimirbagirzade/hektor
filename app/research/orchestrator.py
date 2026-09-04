"""Araştırma Orkestratorü — tam döngüyü yönetir.

Döngü:
  soru → sentezle → StrategyIR → backtest → değerlendir
       → verdict == pass? dur
       → değilse: yansıt → iyileştir → tekrar

Her iterasyon SQLite `research_sessions` tablosuna yazılır.
Tüm zincir daha sonra LoRA eğitim verisi olarak kullanılır.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.agents.runtime import log_step, tracked
from app.memory.sqlite_store import SqliteStore
from app.research.reflection_agent import ReflectionAgent
from app.research.synthesis_engine import SynthesisEngine, SynthesisResult
from app.trading.backtester import run_backtest
from app.trading.evaluator import Verdict
from app.trading.evaluator import evaluate as eval_strategy
from app.trading.market_data_loader import generate_synthetic_ohlcv, load_ohlcv
from app.trading.strategy_ir import StrategyIR
from app.verification.exams.l5_composition import CompositionGate, _signature

# Gerçek veri varsa onu kullan — çok daha fazla bar = daha güvenilir istatistik
_REAL_DATA_CANDIDATES = [
    "data/market/raw/BTCUSD_1h_Binance.csv",
    "data/market/raw/BTCUSD_1h_Coinbase.csv",
    "data/market/raw/BTCUSD_1h_Combined_Index.csv",
    "data/market/raw/BTCUSD_1d.csv",
]

logger = logging.getLogger(__name__)


def _reuse_evaluator(verdict: Verdict) -> Any:
    """L5 BacktestGate'in zaten hesaplanmış verdict'i yeniden kullanmasını sağlar.

    Orchestrator backtest'i bir kez koşar; CompositionGate'e bu evaluator enjekte
    edilince ikinci kez backtest çalışmaz (determinizm + maliyet).
    """

    def _evaluate(_df: Any, _ir: Any, **_kw: Any) -> Verdict:
        return verdict

    return _evaluate


@dataclass
class IterationResult:
    session_id: str
    iteration: int
    indicator_name: str
    verdict: str
    reasons: list[str]
    metrics: dict[str, Any]
    reflection: str | None = None
    improvement_notes: str | None = None
    composition: dict[str, Any] | None = None  # L5 kompozisyon kapısı (math+novelty+backtest)


@dataclass
class ResearchResult:
    question: str
    iterations: list[IterationResult] = field(default_factory=list)
    final_verdict: str = "inconclusive"
    best_session_id: str | None = None

    @property
    def converged(self) -> bool:
        return self.final_verdict == "pass"

    def summary(self) -> str:
        lines = [f"Araştırma: {self.question}", f"İterasyon: {len(self.iterations)}"]
        for it in self.iterations:
            l5 = ""
            if it.composition:
                l5 = f" · L5:{it.composition.get('verdict', '?')}"
            lines.append(
                f"  [{it.iteration}] {it.indicator_name} → {it.verdict.upper()} "
                f"({', '.join(it.reasons[:1])}){l5}"
            )
        lines.append(f"Sonuç: {self.final_verdict.upper()}")
        return "\n".join(lines)


class ResearchOrchestrator:
    def __init__(
        self,
        store: SqliteStore | None = None,
        synthesis_engine: SynthesisEngine | None = None,
        reflection_agent: ReflectionAgent | None = None,
        composition_gate: CompositionGate | None = None,
        max_iterations: int = 3,
        n_bars: int = 2000,
        seed: int = 42,
        market: str = "XAUUSD",
        timeframe: str = "15m",
    ) -> None:
        self.store = store or SqliteStore()
        self.synthesis = synthesis_engine or SynthesisEngine(store=self.store)
        self.reflection = reflection_agent or ReflectionAgent()
        # None → her iterasyonda hesaplanan verdict'i yeniden kullanan kapı (çift backtest yok).
        self.composition_gate = composition_gate
        self.max_iterations = max_iterations
        self.n_bars = n_bars
        self.seed = seed
        self.market = market
        self.timeframe = timeframe

    def _load_data(self) -> tuple[pd.DataFrame, str]:
        """Mevcut en iyi veri setini yükle. Gerçek CSV varsa onu tercih et."""

        for path_str in _REAL_DATA_CANDIDATES:
            path = Path(path_str)
            if path.exists():
                try:
                    df = load_ohlcv(path)
                    if len(df) >= 500:
                        logger.info("Gerçek veri kullanılıyor: %s (%d bar)", path.name, len(df))
                        return df, path.name
                except Exception as exc:
                    logger.debug("CSV yüklenemedi %s: %s", path, exc)
        logger.info("Gerçek veri bulunamadı — sentetik %d bar kullanılıyor", self.n_bars)
        label = f"synthetic(n={self.n_bars})"
        return generate_synthetic_ohlcv(n=self.n_bars, seed=self.seed), label

    @tracked("research-orchestrator", trigger_type="manual")
    def run(self, question: str, paper_ids: list[str] | None = None) -> ResearchResult:
        """Bir araştırma sorusu için tam döngüyü çalıştır."""
        logger.info("Araştırma başladı: %s", question)
        result = ResearchResult(question=question)
        log_step(f"Araştırma turu başladı: {question[:80]}")
        df, data_source = self._load_data()

        current_indicator: SynthesisResult | None = None
        parent_session_id: str | None = None
        prev_failures: list[dict] = []
        seen_signatures: set[tuple[tuple[str, int], ...]] = set()  # L5 novelty: kopya tespiti

        for iteration in range(1, self.max_iterations + 1):
            logger.info("İterasyon %d/%d", iteration, self.max_iterations)
            session_id = f"rs_{uuid.uuid4().hex[:12]}"

            # ---- Sentez / İyileştirme ----
            if iteration == 1 or current_indicator is None:
                # Önceki başarısızlık bilgisini synthesis motoruna ilet
                synthesized = self.synthesis.synthesize(
                    question,
                    paper_ids=paper_ids,
                    market=self.market,
                    timeframe=self.timeframe,
                    prev_failures=prev_failures or None,
                )
            else:
                synthesized = current_indicator  # yansıma zaten IR'ı güncelledi

            if synthesized is None:
                logger.warning("Sentez başarısız, döngü durdu.")
                break

            # ---- StrategyIR doğrulama ----
            ir_dict = synthesized.strategy_ir
            try:
                ir = StrategyIR.model_validate(ir_dict)
            except Exception as exc:
                logger.warning("IR doğrulama hatası: %s — varsayılan kullanılıyor", exc)
                from app.trading.strategy_ir import example_ir

                ir = example_ir()

            # ---- Backtest (gerçek veri kaynağını timeframe'e göre ayarla) ----
            if len(df) > 10000:
                # Uzun veri → 1h timeframe varsay; IR'daki timeframe geçersiz kılınmaz
                ir = ir.model_copy(update={"timeframe": "1h"})
            bt = run_backtest(df, ir)
            ev = eval_strategy(df, ir)
            metrics = bt.metrics.to_dict()
            logger.info(
                "Backtest: %s → %s işlem, Sharpe=%.2f (%s)",
                ir.name,
                metrics.get("n_trades", 0),
                metrics.get("sharpe", 0),
                data_source,
            )

            # ---- L5 KOMPOZİSYON SINAVI (math + novelty + maliyet-dahil backtest) ----
            # "Anladı" = bilgiyi DOĞRU kullanıp test edilebilir YENİ bir şey üretebildi mi.
            # Zaten hesaplanan `ev` yeniden kullanılır → ikinci backtest koşulmaz.
            gate = self.composition_gate or CompositionGate(evaluator=_reuse_evaluator(ev))
            composition = gate.evaluate_composition(ir, df, seen_signatures=seen_signatures)
            seen_signatures.add(_signature(ir))
            logger.info(
                "L5 kompozisyon: %s → %s",
                ir.name,
                composition.verdict,
            )

            # ---- Session kaydet ----
            self.store.save_research_session(
                session_id=session_id,
                question=question,
                iteration=iteration,
                parent_session_id=parent_session_id,
                source_paper_ids_json=json.dumps(synthesized.source_papers),
                synthesis_reasoning=synthesized.combination_reasoning,
                proposed_indicator_json=json.dumps(synthesized.to_dict(), ensure_ascii=False),
                strategy_ir_json=json.dumps(ir.model_dump(), ensure_ascii=False),
                backtest_result_json=json.dumps(
                    {
                        "metrics": metrics,
                        "verdict": ev.verdict,
                        "reasons": ev.reasons,
                        "composition": composition.to_dict(),
                    },
                    ensure_ascii=False,
                ),
                verdict=ev.verdict,
                status="done",
            )

            iter_result = IterationResult(
                session_id=session_id,
                iteration=iteration,
                indicator_name=synthesized.indicator_name,
                verdict=ev.verdict,
                reasons=ev.reasons,
                metrics=metrics,
                composition=composition.to_dict(),
            )

            # Başarısızlık geçmişini güncelle (synthesis motoruna iletmek için)
            if ev.verdict != "pass":
                prev_failures.append(
                    {
                        "n_trades": metrics.get("n_trades", 0),
                        "verdict": ev.verdict,
                        "reasons": ev.reasons,
                    }
                )

            # ---- Yansıma (son iterasyon değilse ve pass değilse) ----
            if ev.verdict != "pass" and iteration < self.max_iterations:
                reflection_data = self.reflection.reflect(
                    indicator=synthesized.to_dict(),
                    backtest_result={"metrics": metrics},
                    verdict=ev.verdict,
                    reasons=ev.reasons,
                )
                if reflection_data:
                    reflection_text = reflection_data.get("failure_analysis", "")
                    improvement_notes = "; ".join(reflection_data.get("changes", []))
                    iter_result.reflection = reflection_text
                    iter_result.improvement_notes = improvement_notes

                    # Yansıma IR'ı mevcut indikatöre aktar
                    new_ir = reflection_data.get("strategy_ir")
                    if new_ir:
                        from copy import deepcopy

                        updated = deepcopy(synthesized)
                        updated.strategy_ir = new_ir
                        updated.indicator_name = new_ir.get("name", synthesized.indicator_name)
                        updated.combination_reasoning = reflection_data.get(
                            "improvement_reasoning", ""
                        )
                        current_indicator = updated
                    # Yansımayı session'a kaydet
                    self.store.save_research_session(
                        session_id=session_id,
                        question=question,
                        iteration=iteration,
                        parent_session_id=parent_session_id,
                        source_paper_ids_json=json.dumps(synthesized.source_papers),
                        synthesis_reasoning=synthesized.combination_reasoning,
                        proposed_indicator_json=json.dumps(
                            synthesized.to_dict(), ensure_ascii=False
                        ),
                        strategy_ir_json=json.dumps(ir.model_dump(), ensure_ascii=False),
                        backtest_result_json=json.dumps(
                            {
                                "metrics": metrics,
                                "verdict": ev.verdict,
                                "reasons": ev.reasons,
                                "composition": composition.to_dict(),
                            },
                            ensure_ascii=False,
                        ),
                        verdict=ev.verdict,
                        reflection=reflection_text,
                        improvement_notes=improvement_notes,
                        status="done",
                    )

            result.iterations.append(iter_result)
            parent_session_id = session_id

            if ev.verdict == "pass":
                result.final_verdict = "pass"
                result.best_session_id = session_id
                logger.info("PASS — iterasyon %d'de yakınsadı.", iteration)
                break

        if result.final_verdict != "pass" and result.iterations:
            result.final_verdict = result.iterations[-1].verdict
            result.best_session_id = result.iterations[-1].session_id

        logger.info("Araştırma tamamlandı:\n%s", result.summary())
        return result
