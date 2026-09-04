"""Evidence sufficiency scorer — RLM cevap vermeden önce kaynak yeterli mi?

Talimat §12: 0–100 arası kanıt yeterlilik skoru. LLM kullanmaz (deterministik,
çevrimdışı). Eşikler RLM controller'da uygulanır:

    80–100 : cevap üret (yeniden retrieval gereksiz)
    60–79  : cevap üret ama sınırlamayı belirt
    40–59  : tekrar retrieval dene
    0–39   : "yeterli kaynak yok" de
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.memory.retrieval_service import RetrievedChunk
from app.verification.contradiction_detector import Contradiction

# Bileşen ağırlıkları (talimat §12, toplam 100).
_W_RELEVANCE = 25
_W_COVERAGE = 20
_W_SECTION_DIVERSITY = 15
_W_CITATION = 15
_W_CONTRADICTION = 10
_W_METHOD_LIMIT = 10
_W_RECENCY = 5

# Coverage hedefi: bu kadar (veya daha çok) chunk → tam puan.
_COVERAGE_TARGET = 5

_METHOD_KW = re.compile(
    r"\b(method\w*|metodoloj\w*|approach|model|estimat\w*|regress\w*|sample|veri seti|"
    r"dataset|deney|experiment)\b",
    re.IGNORECASE,
)
_LIMIT_KW = re.compile(
    r"\b(limitation\w*|sınırlama\w*|caveat\w*|drawback\w*|assumption\w*|varsayım\w*|"
    r"weakness\w*|future work|gelecek çalışma)\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}", text.lower()))


@dataclass
class EvidenceReport:
    """Kanıt yeterlilik raporu (0–100)."""

    score: float
    relevance: float
    coverage: float
    section_diversity: float
    citation_availability: float
    contradiction_risk: float
    method_limit_presence: float
    recency_metadata: float
    n_chunks: int
    decision: str  # "answer" | "answer_with_limitation" | "retry" | "insufficient"
    details: dict = field(default_factory=dict)


class EvidenceSufficiencyScorer:
    """Getirilen chunk havuzunun bir soruyu cevaplamaya yeterliliğini ölçer."""

    def score(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        contradictions: list[Contradiction] | None = None,
        *,
        min_to_answer: int = 60,
        min_to_skip_retry: int = 80,
        min_to_retry: int = 40,
    ) -> EvidenceReport:
        contradictions = contradictions or []
        # Eşik değişmezini zorla: retry ≤ answer ≤ skip_retry (hepsi [0,100]).
        # Config/env'den çelişkili değerler gelse bile (ör. skip_retry<answer)
        # karar bandları tutarsızlaşmaz; 'retry' tabanı artık config'e bağlı (sabit 40 değil).
        min_to_retry = max(0, min(min_to_retry, 100))
        min_to_answer = max(min_to_retry, min(min_to_answer, 100))
        min_to_skip_retry = max(min_to_answer, min(min_to_skip_retry, 100))
        if not chunks:
            return EvidenceReport(
                score=0.0,
                relevance=0.0,
                coverage=0.0,
                section_diversity=0.0,
                citation_availability=0.0,
                contradiction_risk=float(_W_CONTRADICTION),
                method_limit_presence=0.0,
                recency_metadata=0.0,
                n_chunks=0,
                decision="insufficient",
                details={"reason": "Hiç chunk getirilmedi"},
            )

        q_tokens = _tokens(query)

        # 1) Relevance (25): sorgu token'larının chunk'larda örtüşme oranı (ortalama).
        #    distance None olabilir (fake embedding) → token örtüşmesi sağlam taban.
        if q_tokens:
            overlaps = []
            for c in chunks:
                c_tokens = _tokens(c.text)
                overlaps.append(len(q_tokens & c_tokens) / max(1, len(q_tokens)))
            # En iyi 3 chunk'ın ortalaması (kuyruk gürültüsünü bastır).
            top = sorted(overlaps, reverse=True)[:3]
            rel_frac = sum(top) / len(top) if top else 0.0
        else:
            rel_frac = 0.0
        relevance = min(1.0, rel_frac * 1.5) * _W_RELEVANCE  # hafif kalibrasyon

        # 2) Coverage (20): chunk sayısı / hedef.
        coverage = min(1.0, len(chunks) / _COVERAGE_TARGET) * _W_COVERAGE

        # 3) Section diversity (15): farklı bölüm sayısı.
        sections = {(c.section_name or "").strip().lower() for c in chunks}
        sections.discard("")
        diversity_frac = min(1.0, len(sections) / 3) if sections else 0.0
        section_diversity = diversity_frac * _W_SECTION_DIVERSITY

        # 4) Citation availability (15): geçerli paper_id+chunk_id taşıyan chunk oranı.
        valid = sum(1 for c in chunks if c.paper_id and c.paper_id != "?" and c.chunk_id)
        citation_availability = (valid / len(chunks)) * _W_CITATION

        # 5) Contradiction risk (10): çelişki yoksa tam puan, her çelişki cezalı.
        contradiction_risk = max(0.0, _W_CONTRADICTION - len(contradictions) * 5)

        # 6) Methodology / limitation presence (10): 5 method + 5 limitation.
        joined = " ".join(c.text for c in chunks)
        method_limit_presence = (5 if _METHOD_KW.search(joined) else 0) + (
            5 if _LIMIT_KW.search(joined) else 0
        )

        # 7) Recency / metadata quality (5): başlık taşıyan chunk oranı.
        titled = sum(1 for c in chunks if c.title)
        recency_metadata = (titled / len(chunks)) * _W_RECENCY

        total = (
            relevance
            + coverage
            + section_diversity
            + citation_availability
            + contradiction_risk
            + method_limit_presence
            + recency_metadata
        )
        total = round(max(0.0, min(100.0, total)), 2)

        if total >= min_to_skip_retry:
            decision = "answer"
        elif total >= min_to_answer:
            decision = "answer_with_limitation"
        elif total >= min_to_retry:
            decision = "retry"
        else:
            decision = "insufficient"

        return EvidenceReport(
            score=total,
            relevance=round(relevance, 2),
            coverage=round(coverage, 2),
            section_diversity=round(section_diversity, 2),
            citation_availability=round(citation_availability, 2),
            contradiction_risk=round(contradiction_risk, 2),
            method_limit_presence=float(method_limit_presence),
            recency_metadata=round(recency_metadata, 2),
            n_chunks=len(chunks),
            decision=decision,
            details={"n_contradictions": len(contradictions), "n_sections": len(sections)},
        )
