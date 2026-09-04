"""Task classifier — kural-tabanlı (deterministik, LLM-free) görev sınıflandırma.

Kullanıcı sorusunu RLM görev tiplerinden birine eşler ve bir reasoning plan
üretir. LLM kullanılmaz → çevrimdışı çalışır, kural 6 (determinizm) gereği aynı
girdi her zaman aynı sonucu verir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Spec'teki görev tipleri (app/rlm talimatı §11).
GENERAL = "general_paper_question"
SINGLE = "single_paper_analysis"
MULTI = "multi_paper_synthesis"
MATH = "math_reasoning"
FORMULA = "formula_explanation"
TRADING = "trading_reasoning"
PHILOSOPHY = "philosophy_logic"
CONTRADICTION = "contradiction_check"
UNCERTAINTY = "uncertainty_check"
LITERATURE = "literature_review"

TASK_TYPES = (
    GENERAL,
    SINGLE,
    MULTI,
    MATH,
    FORMULA,
    TRADING,
    PHILOSOPHY,
    CONTRADICTION,
    UNCERTAINTY,
    LITERATURE,
)


# Anahtar kelime havuzları (TR + EN). Sıra önemlidir: daha spesifik tip önce.
_CONTRADICTION_KW = re.compile(
    r"\b(çeliş\w*|tutars\w*|contradict\w*|disagree\w*|conflict\w*|inconsist\w*)\b",
    re.IGNORECASE,
)
_FORMULA_KW = re.compile(
    r"\b(formül\w*|denklem|formula\w*|equation\w*|türet\w*|değişken\w*|variable\w*)\b",
    re.IGNORECASE,
)
_MATH_KW = re.compile(
    r"\b(hesapla\w*|hesab\w*|matematik\w*|ispat\w*|kanıtla\w*|compute|calculat\w*|"
    r"derive|proof|prove|theorem|teorem\w*)\b",
    re.IGNORECASE,
)
_TRADING_KW = re.compile(
    r"\b(trading|strateji\w*|strategy|sinyal\w*|signal\w*|al-sat|al\s*sat|buy|sell|"
    r"momentum|volatilit\w*|backtest\w*|getiri|return|sharpe|pozisyon|position)\b",
    re.IGNORECASE,
)
_PHILOSOPHY_KW = re.compile(
    r"\b(felsef\w*|epistem\w*|öncül\w*|çıkarım\w*|argüman\w*|philosoph\w*|premise\w*|"
    r"inference|argument\w*|ontolog\w*)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_KW = re.compile(
    r"\b(yeterli mi|emin mi\w*|belirsiz\w*|kesin mi|güvenilir mi|uncertain\w*|"
    r"sufficient|confiden\w*|reliable)\b",
    re.IGNORECASE,
)
_SYNTHESIS_KW = re.compile(
    r"\b(karşılaştır\w*|kıyasla\w*|sentez\w*|birleştir\w*|literatür\w*|tüm makale\w*|"
    r"compare|synthesi\w*|across|multiple papers|literature)\b",
    re.IGNORECASE,
)
_LITERATURE_KW = re.compile(
    r"\b(literatür\w*|alanyazın\w*|genel bakış|literature review|survey|state of the art|"
    r"son çalışmalar)\b",
    re.IGNORECASE,
)


@dataclass
class ReasoningPlan:
    """Görev tipine göre türeyen çalışma planı.

    NOT: `allow_trading_hypothesis` ve `allow_trading_signal` AÇIKLAYICI metadata'dır
    (run logunda görünür) — davranışı TEK BAŞINA belirlemezler. Kural 1 yaptırımı
    içerik-tabanlıdır: RlmController._apply_trading_guard, soru VEYA nihai cevap
    trading dili taşıyorsa zorunlu uyarıyı ekler ve `settings.rlm_allow_live_trading_signal`
    bayrağını gerçekten okur (canlı sinyal asla üretilmez).
    """

    task_type: str
    needs_retrieval: bool = True
    retrieval_rounds: int = 2
    must_include: list[str] = field(default_factory=list)
    verification_required: bool = True
    allow_trading_hypothesis: bool = False  # açıklayıcı: trading task hipotez üretebilir
    allow_trading_signal: bool = False  # açıklayıcı: asla True; gerçek kapı _apply_trading_guard


class TaskClassifier:
    """Soruyu görev tipine eşleyen ve reasoning plan üreten sınıf."""

    def classify(self, query: str, paper_ids: list[str] | None = None) -> str:
        """Soruyu deterministik kurallarla bir görev tipine eşle."""
        q = query.strip()
        n_papers = len(paper_ids) if paper_ids else 0

        # En spesifik sinyaller önce.
        if _CONTRADICTION_KW.search(q):
            return CONTRADICTION
        if _UNCERTAINTY_KW.search(q):
            return UNCERTAINTY
        if _FORMULA_KW.search(q):
            return FORMULA
        if _MATH_KW.search(q):
            return MATH
        if _PHILOSOPHY_KW.search(q):
            return PHILOSOPHY
        if _LITERATURE_KW.search(q):
            return LITERATURE
        if _SYNTHESIS_KW.search(q) or n_papers > 1:
            return MULTI
        if _TRADING_KW.search(q):
            return TRADING
        if n_papers == 1:
            return SINGLE
        return GENERAL

    def plan(
        self,
        task_type: str,
        *,
        n_papers: int = 0,
        max_rounds: int = 3,
    ) -> ReasoningPlan:
        """Görev tipine göre bir reasoning plan üret.

        Args:
            task_type: classify() çıktısı.
            n_papers: kullanıcının belirttiği makale sayısı (0 = tüm havuz).
            max_rounds: izin verilen üst retrieval turu sınırı (config'ten).
        """
        # Varsayılan: 2 tur retrieval, methodology+findings+limitations şart.
        must = ["methodology", "findings", "limitations"]
        rounds = min(2, max_rounds)
        allow_hypothesis = False

        if task_type in (MULTI, LITERATURE):
            rounds = min(3, max_rounds)
            must = ["methodology", "findings", "limitations", "dataset"]
        elif task_type in (FORMULA, MATH):
            must = ["formula", "methodology", "definition"]
        elif task_type == TRADING:
            # Trading muhakemesi → yalnız HİPOTEZ üretilebilir, sinyal ASLA.
            allow_hypothesis = True
            must = ["methodology", "findings", "limitations", "dataset"]
        elif task_type == CONTRADICTION:
            rounds = min(3, max_rounds)
            must = ["findings", "methodology"]
        elif task_type == SINGLE:
            rounds = min(2, max_rounds)

        return ReasoningPlan(
            task_type=task_type,
            needs_retrieval=True,
            retrieval_rounds=max(1, rounds),
            must_include=must,
            verification_required=True,
            allow_trading_hypothesis=allow_hypothesis,
            allow_trading_signal=False,
        )
