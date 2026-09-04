"""Offline, deterministik RAGAS-tarzı RAG metrikleri (LLM gerektirmez).

RAGAS'ın çekirdek fikirlerini (faithfulness, context precision, context recall) **LLM
olmadan**, yalnız token-örtüşmesiyle yaklaşık hesaplar. Amaç: golden chunk-id etiketi
gerektirmeden, canlı RAG cevaplarında ucuz, tekrarlanabilir bir kalite sinyali üretmek
("retrieval gürültüsü" + "dayanaksız cümle" tespiti). LLM-judge'ın deterministik, çevrimdışı,
sahte-pass üretmeyen tamamlayıcısıdır (CLAUDE.md Kural 2/6/7 ile uyumlu).

İlişki: `verification/grounding_verifier.py` cümle-bazlı SUPPORTED/UNSUPPORTED sınıflaması
yapar; buradaki `faithfulness` aynı fikrin tek 0–1 skoru hâlidir. `evals/metrics.py` ise
golden-id tabanlı precision/recall/MRR/nDCG sağlar (bu modül onları DEĞİŞTİRMEZ, tamamlar).

Sınır: token-örtüşmesi anlamsal değildir (eşanlamlı/parça-örtüşme yanıltabilir); bu yüzden
bir **proxy**'dir, LLM-judge'ın yerine geçmez. Mutlak değil, sürümler-arası KIYAS için kullan.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Token gürültüsünü azaltan kısa TR+EN durak-kelime kümesi (örtüşme şişmesini engeller).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "with",
        "this",
        "that",
        "have",
        "from",
        "they",
        "will",
        "would",
        "there",
        "their",
        "what",
        "which",
        "when",
        "into",
        "than",
        "then",
        "them",
        "these",
        "ve",
        "ile",
        "bir",
        "bu",
        "da",
        "de",
        "için",
        "gibi",
        "olan",
        "olarak",
        "daha",
        "çok",
        "ama",
        "veya",
        "ya",
        "ki",
        "mi",
        "mu",
        "ise",
        "göre",
        "kadar",
        "ben",
        "sen",
    }
)
_TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]{3,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _tokens(text: str) -> set[str]:
    """Küçük harf, ≥3 karakter, durak-kelimesiz benzersiz token kümesi (deterministik)."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    """|a ∩ b| / |a| — a'nın ne kadarının b tarafından kapsandığı (a boşsa 0.0)."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


@dataclass
class RagasOfflineScores:
    """Offline RAGAS-tarzı skorlar (hepsi [0,1]; yüksek = daha iyi)."""

    faithfulness: float  # cevap cümlelerinin bağlamca desteklenme oranı
    context_precision: float  # çekilen bağlam parçalarının cevaba katkı oranı (gürültü azlığı)
    context_recall: float | None  # referans cevabın bağlamca kapsanma oranı (referans yoksa None)
    n_contexts: int
    n_answer_sentences: int


def faithfulness(answer: str, contexts: Sequence[str], threshold: float = 0.3) -> float:
    """Cevap cümlelerinin, bağlam birleşimince desteklenme oranı (token-örtüşmesi ≥ eşik).

    Düşük skor → cevapta bağlamda olmayan ("uydurma" olabilecek) cümleler var.
    """
    sents = _sentences(answer)
    if not sents:
        return 0.0
    ctx_union: set[str] = set()
    for c in contexts:
        ctx_union |= _tokens(c)
    supported = sum(1 for s in sents if _overlap_ratio(_tokens(s), ctx_union) >= threshold)
    return supported / len(sents)


def context_precision(answer: str, contexts: Sequence[str], threshold: float = 0.06) -> float:
    """Çekilen bağlam parçalarının cevaba katkı (alaka) oranı.

    Her parça için (parça ∩ cevap)/parça hesaplanır; eşik üstü parça "alakalı" sayılır.
    Düşük skor → retrieval gürültülü (cevaba katkısız parçalar getirmiş).
    """
    if not contexts:
        return 0.0
    ans = _tokens(answer)
    relevant = sum(1 for c in contexts if _overlap_ratio(_tokens(c), ans) >= threshold)
    return relevant / len(contexts)


def context_recall(reference: str, contexts: Sequence[str]) -> float:
    """Referans (altın) cevabın, bağlam birleşimince kapsanma oranı (token bazında).

    Düşük skor → bağlam, beklenen cevabı taşıyacak bilgiyi getirememiş.
    """
    ref = _tokens(reference)
    if not ref:
        return 0.0
    ctx_union: set[str] = set()
    for c in contexts:
        ctx_union |= _tokens(c)
    return len(ref & ctx_union) / len(ref)


def evaluate_rag_answer(
    answer: str,
    contexts: Sequence[str],
    reference: str | None = None,
    *,
    faithfulness_threshold: float = 0.3,
    precision_threshold: float = 0.06,
) -> RagasOfflineScores:
    """Tek bir RAG cevabı için offline RAGAS-tarzı skorları hesapla (deterministik).

    Args:
        answer: Üretilen cevap metni.
        contexts: Cevaba verilen bağlam parçaları (retrieval çıktısı metinleri).
        reference: (Opsiyonel) altın cevap — verilirse context_recall hesaplanır.
        faithfulness_threshold: Cümle-destek eşiği.
        precision_threshold: Bağlam-alaka eşiği.

    Returns:
        `RagasOfflineScores`. LLM yok; sahte-pass üretmez (boş girdi → 0.0).
    """
    return RagasOfflineScores(
        faithfulness=faithfulness(answer, contexts, threshold=faithfulness_threshold),
        context_precision=context_precision(answer, contexts, threshold=precision_threshold),
        context_recall=(context_recall(reference, contexts) if reference is not None else None),
        n_contexts=len(contexts),
        n_answer_sentences=len(_sentences(answer)),
    )
