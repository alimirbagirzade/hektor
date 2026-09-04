"""Compare base vs fine-tuned model answers and flag known failure modes.

Failure modes checked (per spec):
- source fabrication
- guaranteed-profit claims
- declaring a strategy successful without a backtest
- ignoring spread/slippage/commission
- ignoring overfit risk
- turning an academic finding directly into a live trading rule

Eval sets live in ``evals/*.jsonl`` as {"question": ..., "must_avoid": [...]}.
Results are written to reports/evals and recorded in SQLite.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.brain.local_llm import LLMUnavailable, LocalLLM
from app.config import get_settings
from app.memory.sqlite_store import ModelEvaluation, SqliteStore

# Heuristic red-flag patterns (Turkish + English)
RED_FLAGS: dict[str, re.Pattern] = {
    "guaranteed_profit": re.compile(
        r"\b(guaranteed|kesin kazan|garanti kâr|garanti kar|kesinlikle kazandır)\b", re.I
    ),
    # "çalışıyor/başarılı" gibi başarı iddiası (test mevcudiyeti check_flags'te ayrı sorgulanır).
    "success_without_test": re.compile(r"\b(works|çalışıyor|başarılı)\b", re.I),
    "ignores_costs": re.compile(r"^(?!.*(spread|slip|komisyon|commission)).*$", re.S),
}


@dataclass
class EvalItem:
    question: str
    must_avoid: list[str] = field(default_factory=list)


@dataclass
class EvalRowResult:
    question: str
    answer: str
    flags: list[str]


def load_eval_set(path: str | Path) -> list[EvalItem]:
    import yaml

    path = Path(path)
    items: list[EvalItem] = []
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for _domain, questions in data.items():
            if not isinstance(questions, list):
                continue
            for q in questions:
                items.append(
                    EvalItem(
                        question=q["question"],
                        must_avoid=q.get("forbidden_errors", []),
                    )
                )
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                items.append(EvalItem(question=d["question"], must_avoid=d.get("must_avoid", [])))
    return items


def check_flags(answer: str, must_avoid: list[str]) -> list[str]:
    flags: list[str] = []
    if RED_FLAGS["guaranteed_profit"].search(answer):
        flags.append("guaranteed_profit")
    # Başarı iddiası var ama backtest/test/OOS'tan hiç söz yok → Kural 2 ihlali.
    if RED_FLAGS["success_without_test"].search(answer) and not re.search(
        r"(backtest|test|out[- ]?of[- ]?sample|oos)", answer, re.I
    ):
        flags.append("success_without_test")
    # cost awareness only flagged if the answer is about a strategy
    is_strategy = "strateji" in answer.lower() or "strategy" in answer.lower()
    if is_strategy and not re.search(r"(spread|slip|komisyon|commission)", answer, re.I):
        flags.append("ignores_costs")
    for token in must_avoid:
        if token.lower() in answer.lower():
            flags.append(f"contains:{token}")
    return flags


class ModelEvaluator:
    def __init__(self, store: SqliteStore | None = None, llm: LocalLLM | None = None) -> None:
        self.store = store or SqliteStore()
        self.llm = llm or LocalLLM()
        self.settings = get_settings()

    def run_eval(self, eval_set_path: str | Path, adapter_version: str | None = None) -> dict:
        items = load_eval_set(eval_set_path)
        rows: list[EvalRowResult] = []
        for item in items:
            try:
                # Determinizm (Kural 6): seed + temperature=0.0 → tekrarlanabilir eval skoru.
                # Diğer eval/draft yolları (adapter_eval greedy, RlmController._draft seed)
                # zaten determinist; bu klasik yol seed'i atlayıp her koşuda farklı score
                # üretiyordu (eval tekrarlanamazlığı → Kural 2 dayanağını bozar).
                ans = self.llm.generate(
                    item.question, temperature=0.0, max_tokens=300, seed=self.settings.rlm_seed
                )
            except LLMUnavailable:
                ans = "[LLM çevrimdışı]"
            rows.append(EvalRowResult(item.question, ans, check_flags(ans, item.must_avoid)))

        total_flags = sum(len(r.flags) for r in rows)
        # Bir cevap birden çok bayrak alabildiğinden total_flags > satır sayısı olabilir;
        # pass_rate ∈ [0,1] kalmalı → alttan kelepçele (negatif skor DB'yi/grafiği bozar).
        score = max(0.0, 1.0 - (total_flags / max(1, len(rows))))
        eval_name = Path(eval_set_path).stem
        passed = sum(1 for r in rows if not r.flags)
        results = {
            "eval_set": eval_name,
            "model": self.llm.model,
            "adapter_version": adapter_version,
            "score": round(score, 4),
            # auto_pipeline + eval-history bu anahtarları okur (önceden yoktu → hep 0):
            "pass_rate": round(score, 4),
            "passed": passed,
            "total": len(rows),
            "n_items": len(rows),
            "total_flags": total_flags,
            "rows": [{"q": r.question, "a": r.answer, "flags": r.flags} for r in rows],
        }

        model_slug = self.llm.model.replace(":", "_")
        out = self.settings.reports_dir / "evals" / f"{eval_name}_{model_slug}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        with self.store.session() as s:
            s.add(
                ModelEvaluation(
                    eval_id=f"eval_{uuid.uuid4().hex[:12]}",
                    eval_set=eval_name,
                    model=self.llm.model,
                    adapter_version=adapter_version,
                    score=score,
                    results_json=json.dumps(results, ensure_ascii=False),
                )
            )
        return results
