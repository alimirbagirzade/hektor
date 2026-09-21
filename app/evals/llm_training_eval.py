"""LoRA eğitimi sonrası LLM kalite değerlendirmesi — persona + format + RAG entegrasyon.

Üç eval seti:
- ``trader_persona``: Model trader gibi düşünüyor mu? (belirsizlik, kaynak atfı, maliyet)
- ``format_compliance``: Çıktı istenen bölümleri içeriyor mu?
- ``rag_integration``: Bağlam verildiğinde kullanıyor mu, yoksa uyduruyor mu?

Mevcut ``check_flags`` (evaluate_model) + yeni persona/format kontrolleri. Çevrimdışı
test edilebilir: LLM çağrılmaz, cevap dışarıdan verilir (adapter_eval veya lora_chat_service
cevabı geçirilir).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.training.evaluate_model import check_flags

# ---------------------------------------------------------------------------
# Persona sinyalleri — trader'a özgü davranış kalıpları
# ---------------------------------------------------------------------------
PERSONA_SIGNALS: dict[str, re.Pattern[str]] = {
    "belirsizlik": re.compile(
        r"(hipotez|olabilir|muhtemel|belirsiz|test\s+edil|denenm|"
        r"garanti\s+(?:edilemez|verilemez)|kesinlik\s+(?:yoktur|sağlanamaz)|"
        r"koşullara\s+bağlı|hypothesis|uncertain|may\b|might\b|could\b)",
        re.I,
    ),
    "maliyet": re.compile(
        r"(komisyon|spread|slippage|maliyet|kayma|işlem\s+(?:ücreti|maliyeti)|"
        r"commission|transaction\s+cost)",
        re.I,
    ),
    "risk": re.compile(
        r"(risk|drawdown|kayıp|zarar|stop[\s-]?loss|pozisyon\s+büyüklüğü|"
        r"risk[\s-]?yönetim|risk\s+management|max\s+loss)",
        re.I,
    ),
    "kaynak": re.compile(
        r"(kaynak|makale|çalışma|araştırma|literatur|paper|study|research|"
        r"source|reference|according\s+to)",
        re.I,
    ),
}

# ---------------------------------------------------------------------------
# Format bölüm anahtar kelimeleri — cevabın yapısal bütünlüğü
# ---------------------------------------------------------------------------
SECTION_KEYWORDS: dict[str, re.Pattern[str]] = {
    "hipotez": re.compile(r"(hipotez|hypothesis|varsayım|iddia|tez)", re.I),
    "test": re.compile(
        r"(test|backtest|out[\s-]?of[\s-]?sample|oos|walk[\s-]?forward|doğrula|valida)", re.I
    ),
    "risk": re.compile(r"(risk|drawdown|kayıp|zarar|stop[\s-]?loss)", re.I),
    "maliyet": re.compile(r"(maliyet|komisyon|spread|slippage|commission|cost|kayma)", re.I),
    "koşul": re.compile(
        r"(koşul|şart|condition|bağlı|depend|varsayım|assumption|sınırlama|limit)", re.I
    ),
    "kaynak": re.compile(r"(kaynak|referans|makale|çalışma|source|reference|paper)", re.I),
}


@dataclass
class TrainingEvalItem:
    """Genişletilmiş eval kalemi — persona + format + RAG alanlarıyla."""

    question: str
    must_avoid: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    persona_signals: list[str] = field(default_factory=list)
    expected_format: str | None = None
    required_sections: list[str] = field(default_factory=list)
    context: str | None = None
    context_mode: str | None = None
    must_contain_from_context: list[str] = field(default_factory=list)


@dataclass
class TrainingEvalResult:
    """Tek bir sorunun değerlendirme sonucu."""

    question: str
    answer: str
    flags: list[str] = field(default_factory=list)
    # ``None`` = bu kalem o boyutu ÖLÇMÜYOR (0.0 "kaldı" demek, None "ölçülmedi" demek).
    persona_score: float | None = None
    format_score: float | None = None
    context_score: float | None = None
    passed: bool = False


def load_training_eval_set(path: str | Path) -> list[TrainingEvalItem]:
    """Genişletilmiş eval JSONL'ını yükle."""
    path = Path(path)
    items: list[TrainingEvalItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                TrainingEvalItem(
                    question=d["question"],
                    must_avoid=d.get("must_avoid", []),
                    must_contain=d.get("must_contain", []),
                    persona_signals=d.get("persona_signals", []),
                    expected_format=d.get("expected_format"),
                    required_sections=d.get("required_sections", []),
                    context=d.get("context"),
                    context_mode=d.get("context_mode"),
                    must_contain_from_context=d.get("must_contain_from_context", []),
                )
            )
    return items


def check_persona(answer: str, required_signals: list[str]) -> tuple[float | None, list[str]]:
    """Cevabın trader persona sinyallerini taşıyıp taşımadığını kontrol et.

    Dönüş: (skor 0.0-1.0, eksik sinyaller listesi). Kalem persona ölçMÜyorsa skor
    ``None``'dır — 1.0 DEĞİL: hiç kontrol yapmadan "mükemmel" raporlamak vacuous pass olur
    (bkz. ``eval_runner`` boş-set koruması, Kural 2).
    """
    if not required_signals:
        return None, []
    found = 0
    missing: list[str] = []
    for sig in required_signals:
        pattern = PERSONA_SIGNALS.get(sig)
        if pattern and pattern.search(answer):
            found += 1
        else:
            missing.append(sig)
    return round(found / len(required_signals), 4), missing


def check_format(answer: str, required_sections: list[str]) -> tuple[float | None, list[str]]:
    """Cevabın istenen bölüm/kavramları içerip içermediğini kontrol et.

    Dönüş: (skor 0.0-1.0, eksik bölümler listesi). Kalem format ölçMÜyorsa skor ``None``
    (gerekçe: bkz. ``check_persona``).
    """
    if not required_sections:
        return None, []
    found = 0
    missing: list[str] = []
    for sec in required_sections:
        pattern = SECTION_KEYWORDS.get(sec)
        if pattern and pattern.search(answer):
            found += 1
        else:
            missing.append(sec)
    return round(found / len(required_sections), 4), missing


def check_context_usage(
    answer: str,
    *,
    context_mode: str | None,
    must_contain_from_context: list[str],
    must_contain: list[str],
    must_avoid: list[str],
) -> tuple[float | None, list[str]]:
    """RAG bağlam kullanımını kontrol et.

    ``with_context``: bağlamdan beklenen terimlerin cevaba geçip geçmediği.
    ``empty_context``: bağlam yoksa modelin uydurmayıp açıkça belirtip belirtmediği.

    Kalem bağlam ölçMÜyorsa (mod yok ya da mod var ama hiç kontrol tanımlı değil) skor
    ``None`` (gerekçe: bkz. ``check_persona``).
    """
    flags: list[str] = []
    answer_lower = answer.lower()

    if context_mode == "with_context":
        if not must_contain_from_context:
            return None, flags
        found = 0
        for term in must_contain_from_context:
            if term.lower() in answer_lower:
                found += 1
            else:
                flags.append(f"context_term_missing:{term}")
        return round(found / len(must_contain_from_context), 4), flags

    if context_mode == "empty_context":
        for term in must_contain:
            if term.lower() not in answer_lower:
                flags.append(f"abstention_missing:{term}")
        for term in must_avoid:
            if term.lower() in answer_lower:
                flags.append(f"fabrication:{term}")
        total_checks = len(must_contain) + len(must_avoid)
        if total_checks == 0:
            return None, flags
        return round(max(0.0, 1.0 - len(flags) / total_checks), 4), flags

    return None, flags


def evaluate_answer(item: TrainingEvalItem, answer: str) -> TrainingEvalResult:
    """Tek bir soru-cevap çiftini tüm boyutlarda değerlendir."""
    all_flags: list[str] = []

    discipline_flags = check_flags(answer, item.must_avoid)
    all_flags.extend(discipline_flags)

    for term in item.must_contain:
        if term.lower() not in answer.lower():
            all_flags.append(f"missing_required:{term}")

    persona_score, persona_missing = check_persona(answer, item.persona_signals)
    all_flags.extend(f"persona_missing:{s}" for s in persona_missing)

    format_score, format_missing = check_format(answer, item.required_sections)
    all_flags.extend(f"format_missing:{s}" for s in format_missing)

    context_score, context_flags = check_context_usage(
        answer,
        context_mode=item.context_mode,
        must_contain_from_context=item.must_contain_from_context,
        must_contain=item.must_contain,
        must_avoid=item.must_avoid,
    )
    all_flags.extend(context_flags)

    passed = not all_flags

    return TrainingEvalResult(
        question=item.question,
        answer=answer,
        flags=all_flags,
        persona_score=persona_score,
        format_score=format_score,
        context_score=context_score,
        passed=passed,
    )


@dataclass
class TrainingEvalSummary:
    """Bir eval setinin toplam sonucu."""

    eval_set: str
    n_items: int
    pass_rate: float
    # ``None`` = o boyutu ölçen HİÇ kalem yok. 1.0 yazmak vacuous pass olurdu: set persona
    # ölçmüyorken "persona mükemmel" raporlamak yanlış güven verir (Kural 2).
    avg_persona: float | None
    avg_format: float | None
    avg_context: float | None
    # Boyut başına kaç kalemin gerçekten ölçüldüğü — ortalamanın dayanağı görünür olsun.
    measured: dict[str, int]
    total_flags: int
    results: list[TrainingEvalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_set": self.eval_set,
            "n_items": self.n_items,
            "pass_rate": self.pass_rate,
            "avg_persona": self.avg_persona,
            "avg_format": self.avg_format,
            "avg_context": self.avg_context,
            "measured": self.measured,
            "total_flags": self.total_flags,
            "results": [
                {
                    "question": r.question,
                    "flags": r.flags,
                    "persona_score": r.persona_score,
                    "format_score": r.format_score,
                    "context_score": r.context_score,
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }


def run_training_eval(
    eval_set_path: str | Path,
    answers: list[str],
) -> TrainingEvalSummary:
    """Eval setini ve cevapları alıp toplam sonuç üret.

    ``answers[i]`` → ``items[i]`` eşleşmesi. LLM çağrılmaz — cevaplar dışarıdan gelir.
    """
    items = load_training_eval_set(eval_set_path)
    if len(answers) != len(items):
        raise ValueError(
            f"Cevap sayısı ({len(answers)}) eval kalemi sayısına ({len(items)}) eşit değil"
        )

    results = [evaluate_answer(item, ans) for item, ans in zip(items, answers, strict=True)]

    n = len(results)
    passed = sum(1 for r in results if r.passed)
    total_flags = sum(len(r.flags) for r in results)

    def _avg(attr: str) -> tuple[float | None, int]:
        """Yalnız o boyutu GERÇEKTEN ölçen kalemler üzerinden ortala (ölçen yoksa None)."""
        scores = [s for r in results if (s := getattr(r, attr)) is not None]
        if not scores:
            return None, 0
        return round(sum(scores) / len(scores), 4), len(scores)

    avg_persona, n_persona = _avg("persona_score")
    avg_format, n_format = _avg("format_score")
    avg_context, n_context = _avg("context_score")

    return TrainingEvalSummary(
        eval_set=Path(eval_set_path).stem,
        n_items=n,
        pass_rate=round(passed / n, 4) if n else 0.0,
        avg_persona=avg_persona,
        avg_format=avg_format,
        avg_context=avg_context,
        measured={"persona": n_persona, "format": n_format, "context": n_context},
        total_flags=total_flags,
        results=results,
    )
