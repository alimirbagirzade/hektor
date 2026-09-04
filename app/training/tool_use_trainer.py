"""tool_use_trainer.py — LLM araç-kullanım eğitim döngüsü.

Her döngü:
  1. THINK   — LLM hipotez oluşturur (synthesis_engine)
  2. CALL    — backtest aracını çağırır (run_backtest)
  3. OBSERVE — sonucu okur (verdict, metrics)
  4. CONCLUDE — yansıma yapar (reflection_agent)

Her adım `tool_use_examples` tablosuna kaydedilir.
Biriktirilen kayıtlar SFT eğitim verisi olarak kullanılır.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.memory.sqlite_store import SqliteStore
from app.research.reflection_agent import ReflectionAgent
from app.research.synthesis_engine import SynthesisEngine
from app.trading.backtester import run_backtest
from app.trading.evaluator import evaluate as eval_strategy
from app.trading.market_data_loader import generate_synthetic_ohlcv, load_ohlcv
from app.trading.strategy_ir import StrategyIR

logger = logging.getLogger(__name__)

_REAL_DATA_CANDIDATES = [
    "data/market/raw/BTCUSD_1h_Binance.csv",
    "data/market/raw/BTCUSD_1h_Combined_Index.csv",
    "data/market/raw/BTCUSD_1d.csv",
]


@dataclass
class ToolUseStep:
    step_type: str  # think | call | observe | conclude
    content: str
    tool_name: str | None = None
    tool_input: dict = field(default_factory=dict)
    tool_output: dict = field(default_factory=dict)
    verdict: str | None = None


@dataclass
class ToolUseSession:
    session_id: str
    question: str
    steps: list[ToolUseStep] = field(default_factory=list)
    final_verdict: str = "inconclusive"

    def as_sft_example(self) -> dict[str, Any]:
        """Tek SFT üçlüsüne dönüştür: instruction + input + output."""
        tool_calls = [s for s in self.steps if s.step_type == "call"]
        conclusions = [s for s in self.steps if s.step_type == "conclude"]

        call_summary = "\n".join(
            f"  [{i + 1}] {s.tool_name}({json.dumps(s.tool_input)[:120]}) → "
            f"verdict={s.tool_output.get('verdict', '?')}, "
            f"n_trades={s.tool_output.get('metrics', {}).get('n_trades', '?')}"
            for i, s in enumerate(tool_calls)
        )
        output = conclusions[-1].content if conclusions else "Sonuç belirsiz."

        return {
            "instruction": (
                "Sen bir ticaret stratejisi araştırmacısısın. "
                "Verilen soruyu analiz et, backtest aracını kullan ve sonucu değerlendir."
            ),
            "input": (
                f"Araştırma Sorusu: {self.question}\n\nAraç Çağrıları:\n{call_summary or '  (yok)'}"
            ),
            "output": output,
            "metadata": {
                "session_id": self.session_id,
                "final_verdict": self.final_verdict,
                "n_tool_calls": len(tool_calls),
            },
        }


def _load_data(n_bars: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, str]:
    for path_str in _REAL_DATA_CANDIDATES:
        path = Path(path_str)
        if path.exists():
            try:
                df = load_ohlcv(path)
                if len(df) >= 500:
                    return df, path.name
            except Exception:
                pass
    return generate_synthetic_ohlcv(n=n_bars, seed=seed), f"synthetic(n={n_bars})"


class ToolUseTrainer:
    """Araştırma döngüsünü çalıştırır ve her adımı eğitim verisi olarak kaydeder."""

    def __init__(
        self,
        store: SqliteStore | None = None,
        n_bars: int = 2000,
        seed: int = 42,
        market: str = "BTCUSD",
        timeframe: str = "1h",
    ) -> None:
        self.store = store or SqliteStore()
        self.synthesis = SynthesisEngine(store=self.store)
        self.reflection = ReflectionAgent()
        self.n_bars = n_bars
        self.seed = seed
        self.market = market
        self.timeframe = timeframe

    def run_session(self, question: str, max_iterations: int = 2) -> ToolUseSession:
        """Tek bir tool-use eğitim seansı çalıştır."""
        session_id = f"tu_{uuid.uuid4().hex[:12]}"
        session = ToolUseSession(session_id=session_id, question=question)
        df, _source = _load_data(self.n_bars, self.seed)
        prev_failures: list[dict] = []

        for iteration in range(1, max_iterations + 1):
            step_base = (iteration - 1) * 4

            # THINK
            synthesized = self.synthesis.synthesize(
                question,
                market=self.market,
                timeframe=self.timeframe,
                prev_failures=prev_failures or None,
            )
            if synthesized is None:
                logger.warning(
                    "Sentez sonuç üretmedi (formül yok veya LLM çevrimdışı) — seans durduruluyor."
                )
                break
            think_step = ToolUseStep(
                step_type="think",
                content=(
                    f"Hipotez: {synthesized.indicator_name}\n"
                    f"Mantık: {(synthesized.combination_reasoning or '')[:300]}"
                ),
            )
            session.steps.append(think_step)
            self._save_step(session_id, question, step_base, think_step)

            # CALL — strategy_ir bir dict; StrategyIR'a çevir
            ir_dict: dict = synthesized.strategy_ir or {}
            ir_obj: StrategyIR | None = None
            try:
                if ir_dict:
                    ir_obj = StrategyIR.model_validate(ir_dict)
            except Exception:
                ir_obj = None

            call_step = ToolUseStep(
                step_type="call",
                content=f"backtest({synthesized.indicator_name})",
                tool_name="run_backtest",
                tool_input={"strategy_ir": ir_dict, "n_bars": self.n_bars},
            )
            try:
                if ir_obj:
                    bt = run_backtest(df.copy(), ir_obj)
                    ev = eval_strategy(df.copy(), ir_obj)
                    call_step.tool_output = {
                        "verdict": ev.verdict,
                        "metrics": bt.metrics.to_dict(),
                        "reasons": ev.reasons,
                    }
                    call_step.verdict = ev.verdict
                else:
                    call_step.tool_output = {"verdict": "fail", "error": "Geçersiz IR"}
                    call_step.verdict = "fail"
            except Exception as exc:
                call_step.tool_output = {"verdict": "fail", "error": str(exc)[:200]}
                call_step.verdict = "fail"

            session.steps.append(call_step)
            self._save_step(session_id, question, step_base + 1, call_step)

            # OBSERVE
            verdict = call_step.verdict or "fail"
            metrics = call_step.tool_output.get("metrics", {})
            observe_step = ToolUseStep(
                step_type="observe",
                content=(
                    f"Sonuç: {verdict.upper()} | "
                    f"İşlem: {metrics.get('n_trades', 0)} | "
                    f"Sharpe: {metrics.get('sharpe', 0):.2f} | "
                    f"Getiri: {metrics.get('total_return_pct', 0):.1f}%"
                ),
                tool_output=call_step.tool_output,
                verdict=verdict,
            )
            session.steps.append(observe_step)
            self._save_step(session_id, question, step_base + 2, observe_step)

            # CONCLUDE
            if verdict == "pass":
                conclude_content = (
                    f"Strateji '{synthesized.indicator_name}' backtest'i geçti. "
                    f"Sharpe {metrics.get('sharpe', 0):.2f}, "
                    f"{metrics.get('n_trades', 0)} işlem. Araştırma tamamlandı."
                )
            else:
                try:
                    if ir_obj:
                        reflection = self.reflection.reflect(
                            {
                                "indicator_name": synthesized.indicator_name,
                                "combination_reasoning": synthesized.combination_reasoning,
                                "strategy_ir": ir_dict,
                            },
                            {"verdict": verdict, "metrics": metrics},
                            verdict,
                            call_step.tool_output.get("reasons", []),
                        )
                        conclude_content = self._format_reflection(reflection, verdict)
                    else:
                        conclude_content = "Strateji IR üretilemedi."
                except Exception:
                    conclude_content = f"Yansıma başarısız — verdict: {verdict}"

            conclude_step = ToolUseStep(
                step_type="conclude", content=conclude_content, verdict=verdict
            )
            session.steps.append(conclude_step)
            self._save_step(session_id, question, step_base + 3, conclude_step)

            if verdict == "pass":
                session.final_verdict = "pass"
                break

            prev_failures.append({"indicator": synthesized.indicator_name, "verdict": verdict})

        if session.final_verdict == "inconclusive":
            session.final_verdict = (
                "pass" if any(s.verdict == "pass" for s in session.steps) else "fail"
            )

        return session

    def _format_reflection(self, reflection: object, verdict: str) -> str:
        """Yansıma çıktısını metne çevir (dict, str veya None olabilir)."""
        if reflection is None:
            return f"Yansıma boş — verdict: {verdict}"
        if isinstance(reflection, str):
            return reflection[:400]
        if isinstance(reflection, dict):
            lesson = reflection.get("lesson") or reflection.get("reasoning") or ""
            suggestion = reflection.get("suggestion") or reflection.get("improvement") or ""
            text = " ".join(part for part in (str(lesson), str(suggestion)) if part).strip()
            return (text or f"Yansıma alındı — verdict: {verdict}")[:400]
        return str(reflection)[:400]

    def _save_step(self, session_id: str, question: str, idx: int, step: ToolUseStep) -> None:
        self.store.save_tool_use_example(
            example_id=f"{session_id}_{idx}",
            session_id=session_id,
            question=question,
            step_index=idx,
            step_type=step.step_type,
            content=step.content,
            tool_name=step.tool_name,
            tool_input=step.tool_input,
            tool_output=step.tool_output,
            verdict=step.verdict,
        )

    def run_batch(self, questions: list[str], max_iterations: int = 2) -> list[ToolUseSession]:
        sessions = []
        for i, q in enumerate(questions, 1):
            logger.info("Toplu eğitim %d/%d: %s", i, len(questions), q[:60])
            try:
                sessions.append(self.run_session(q, max_iterations=max_iterations))
            except Exception as exc:
                logger.warning("Soru atlandı (%s): %s", q[:40], exc)
        return sessions
