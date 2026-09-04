"""Birleşik değerlendirme çalıştırıcısı (eval-runner) — tek giriş noktası + yayın kapısı.

Dağınık duran değerlendiricileri tek bir ``--type`` arayüzünde toplar ve sonucu
``ReleaseGate`` ile kapıdan geçirir (Kural: eval yoksa / kapı geçilmezse production'a
alınmaz). ``--strict`` modda kapı geçilemezse ``EvalGateError`` fırlatır.

Şu an SAĞLAM ve çevrimdışı-test edilebilir tipler:
- ``trading-hypothesis`` : yeni hipotez-test-edilebilirlik değerlendiricisi (self-contained)
- ``rag-retrieval``      : mevcut RetrievalEvaluator (retriever enjekte edilir)

Henüz BAĞLANMAMIŞ tipler (mevcut CLI'larla yapılır; net hata verir): ``rag-answer``,
``lora`` (bkz. ``hektor lora-eval``), ``formula``, ``rlm-reward`` (app/rlm bağımlı;
eş zamanlı oturum bitene dek ertelendi).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.reliability.release_gate import ReleaseGate

SUPPORTED_TYPES: tuple[str, ...] = ("trading-hypothesis", "rag-retrieval")
DEFERRED_TYPES: dict[str, str] = {
    "rag-answer": "tam RAG cevap hattı gerekir (RagAnswerer + verification/*); henüz bağlanmadı",
    "lora": "bkz. 'hektor lora-eval' (adapter + LLM gerektirir, çevrimdışı değil)",
    "formula": "formül doğrulayıcı henüz yok; hiç bağlanmadan ölü kaldığı için kaldırıldı",
    "rlm-reward": "RLM koşusundan ödül türetimi tanımlanmadı (bkz. app/rlm/lora_candidate §16)",
}


class EvalGateError(RuntimeError):
    """``--strict`` modda kapı geçilemediğinde fırlatılır (production engellenir)."""


@dataclass
class EvalRunResult:
    """Bir değerlendirme koşusunun sonucu (metrikler + kapı kararı + kalemler)."""

    eval_type: str
    n_items: int
    metrics: dict[str, float]
    passed: bool
    failures: list[str] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    report_path: str | None = None
    # opt-in registry entegrasyonu (Modül 6→8): rag-retrieval eval'inde indeks sürümü +
    # terfi kararı loglanırsa burada görünür (None = registry verilmedi).
    registry_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_type": self.eval_type,
            "n_items": self.n_items,
            "metrics": self.metrics,
            "passed": self.passed,
            "failures": self.failures,
            "items": self.items,
            "report_path": self.report_path,
            "registry_decision": self.registry_decision,
        }


class EvalRunner:
    """Değerlendirme tiplerini dağıtan ve kapıdan geçiren tek giriş noktası."""

    def __init__(self, *, reports_dir: Path | None = None) -> None:
        self._reports_dir = reports_dir or (get_settings().reports_dir / "evals")

    def run(
        self,
        eval_type: str,
        *,
        strict: bool = False,
        write_report: bool = True,
        registry: Any | None = None,
        **kwargs: Any,
    ) -> EvalRunResult:
        if eval_type in DEFERRED_TYPES:
            raise NotImplementedError(
                f"'{eval_type}' runner'a henüz bağlanmadı: {DEFERRED_TYPES[eval_type]}."
            )
        if eval_type == "trading-hypothesis":
            result = self._run_trading_hypothesis(**kwargs)
        elif eval_type == "rag-retrieval":
            result = self._run_rag_retrieval(**kwargs)
        else:
            raise ValueError(
                f"Bilinmeyen eval tipi: {eval_type} (geçerli: {', '.join(SUPPORTED_TYPES)})"
            )
        # opt-in: registry verilirse rag-retrieval sonucunu indeks-sürümü + terfi kararı olarak
        # logla (Modül 6→8). Varsayılan (registry=None) davranış değişmez.
        if registry is not None and eval_type == "rag-retrieval":
            result.registry_decision = self._log_rag_index_promotion(registry, result)
        if write_report:
            result.report_path = self._write_report(result)
        if strict and not result.passed:
            raise EvalGateError(
                f"{eval_type} kapısı geçilemedi: {'; '.join(result.failures) or 'eşik altı'}"
            )
        return result

    def _log_rag_index_promotion(self, registry: Any, result: EvalRunResult) -> dict[str, Any]:
        """Mevcut RAG indeksini sürümle + eval sonucunu terfi kararı olarak logla.

        Eval kapısı (``result.passed``) ne diyorsa onu yansıtır: geçti → ``approved``
        (eval_passed), kaldı → ``blocked``. Karar ``promotion_decisions``'a append-only yazılır.
        """
        idx = registry.snapshot_rag_index()
        vid = idx["rag_index_version_id"]
        if result.passed:
            decision = registry.log_decision(
                target_type="rag_index",
                target_id=vid,
                to_status="eval_passed",
                decision="approved",
                reason="retrieval eval geçti (recall eşiği)",
            )
        else:
            decision = registry.log_decision(
                target_type="rag_index",
                target_id=vid,
                to_status="blocked",
                decision="blocked",
                reason="; ".join(result.failures) or "retrieval eşik altı",
            )
        return {"rag_index_version_id": vid, "n_chunks": idx["n_chunks"], "decision": decision}

    # --- tipler ----------------------------------------------------------
    def _run_trading_hypothesis(
        self,
        *,
        hypotheses: Sequence[dict[str, Any] | str],
        min_candidate_rate: float = 1.0,
    ) -> EvalRunResult:
        from app.evals.trading_hypothesis_evaluator import evaluate_many

        evals = evaluate_many(list(hypotheses))
        n = len(evals)
        rejected = [e for e in evals if e.verdict == "rejected"]
        candidates = [e for e in evals if e.verdict == "candidate"]
        candidate_rate = round(len(candidates) / n, 4) if n else 0.0

        failures: list[str] = []
        if n == 0:
            # Boş set + min_candidate_rate=0.0 → 0.0 < 0.0 False → vacuous 'passed=True'.
            # Hiçbir şey değerlendirmeden "geçti" demek Kural 2 ihlalidir; açıkça başarısız say.
            failures.append("değerlendirilecek hipotez yok (boş set) — vacuous pass engellendi")
        if rejected:
            failures.append(
                f"{len(rejected)} hipotez REDDEDİLDİ (tavsiye/kesinlik dili ya da test-edilemez)"
            )
        if candidate_rate < min_candidate_rate:
            failures.append(f"aday oranı {candidate_rate:.0%} < eşik {min_candidate_rate:.0%}")
        metrics = {
            "candidate_rate": candidate_rate,
            "rejected": float(len(rejected)),
            "n": float(n),
        }
        return EvalRunResult(
            eval_type="trading-hypothesis",
            n_items=n,
            metrics=metrics,
            passed=not failures,
            failures=failures,
            items=[e.to_dict() for e in evals],
        )

    def _run_rag_retrieval(
        self,
        *,
        questions: Sequence[Any],
        retriever: Any,
        recall_threshold: float = 0.70,
    ) -> EvalRunResult:
        from app.evals.retrieval_eval import RetrievalEvaluator

        rows = RetrievalEvaluator(retriever).evaluate(list(questions))
        n = len(rows)

        def _avg(attr: str) -> float:
            return round(sum(getattr(r, attr) for r in rows) / n, 4) if n else 0.0

        metrics = {
            "recall_at_5": _avg("recall_5"),
            "recall_at_10": _avg("recall_10"),
            "precision_at_5": _avg("precision_5"),
            "mrr": _avg("mrr"),
            "ndcg": _avg("ndcg"),
        }
        gate = ReleaseGate({"recall_at_10": recall_threshold})
        gres = gate.check(metrics)
        return EvalRunResult(
            eval_type="rag-retrieval",
            n_items=n,
            metrics=metrics,
            passed=gres.passed,
            failures=gres.failures,
            items=[
                {
                    "question_id": r.question_id,
                    "recall_10": r.recall_10,
                    "precision_5": r.precision_5,
                    "mrr": r.mrr,
                }
                for r in rows
            ],
        )

    # --- rapor -----------------------------------------------------------
    def _write_report(self, result: EvalRunResult) -> str:
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        path = self._reports_dir / f"eval_{result.eval_type}_{ts}.json"
        path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)
