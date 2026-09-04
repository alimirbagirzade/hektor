"""Makale Anlama Skoru — 3 katmanlı doğrulama.

A. Kart Doluluk  (ağırlık 0.30): Alan doluluk oranı, LLM gerektirmez.
B. RAG Hassasiyeti (ağırlık 0.40): ChromaDB precision@5 — kart terimleri aranınca
   top-5 sonucun kaçı doğru paper_id'ye ait?
C. LLM Doğrulama  (ağırlık 0.30): Ollama'ya kart içeriğinden basit soru sorulur,
   cevabın kart terimleriyle örtüşme oranı ölçülür. Ollama kapalıysa 0.5 varsayılan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

WEIGHTS = {"extraction": 0.30, "retrieval": 0.40, "llm_verify": 0.30}

# Determinizm (CLAUDE.md kural 6): LLM doğrulama çağrısı sabit seed ile yapılır →
# aynı kart içeriği → aynı skor (yeniden skorlama gürültüsüz, önbelleğe uygun).
_LLM_VERIFY_SEED = 7

_CARD_FIELDS = [
    "title",
    "summary",
    "main_claim",
    "trading_relevance",
    "domain",
    "methods",
    "formulas",
]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ComprehensionScore:
    paper_id: str
    extraction: float = 0.0  # A — 0-1
    retrieval: float = 0.0  # B — 0-1
    llm_verify: float = 0.0  # C — 0-1
    total: float = 0.0  # ağırlıklı toplam × 100 → 0-100
    details: dict = field(default_factory=dict)
    computed_at: str = field(default_factory=_utcnow)


class ComprehensionScorer:
    def __init__(self) -> None:
        from app.memory.chroma_store import ChromaStore
        from app.memory.embedding_service import EmbeddingService
        from app.memory.sqlite_store import SqliteStore

        self._store = SqliteStore()
        self._chroma = ChromaStore()
        self._embedder = EmbeddingService()

    def score(self, paper_id: str, use_llm: bool = True) -> ComprehensionScore:
        """Anlama skoru hesapla.

        ``use_llm=False`` → HIZLI mod: yalnız A (doluluk) + B (RAG precision) gerçek
        hesaplanır; C (LLM doğrulama, tek yavaş kısım) nötr 0.5 alınır. Tüm korpusu
        saniyeler içinde skorlamak için (web/loop). C sonradan refine edilebilir.
        """
        card_json = self._store.get_latest_knowledge_card(paper_id)
        if not card_json:
            return ComprehensionScore(paper_id=paper_id, details={"error": "kart yok"})

        a = self._score_extraction(card_json)
        b = self._score_retrieval(paper_id, card_json)
        c = self._score_llm(card_json) if use_llm else 0.5

        total = round(
            (a * WEIGHTS["extraction"] + b * WEIGHTS["retrieval"] + c * WEIGHTS["llm_verify"])
            * 100,
            1,
        )
        return ComprehensionScore(
            paper_id=paper_id,
            extraction=round(a, 3),
            retrieval=round(b, 3),
            llm_verify=round(c, 3),
            total=total,
            details={
                "a_extraction": round(a, 3),
                "b_retrieval": round(b, 3),
                "c_llm": round(c, 3),
                "llm_verified": use_llm,
            },
        )

    def _score_extraction(self, card_json: dict) -> float:
        filled = sum(
            1
            for f in _CARD_FIELDS
            if card_json.get(f)
            and str(card_json[f]).strip() not in ("", "[]", "{}", "null", "None")
        )
        return filled / len(_CARD_FIELDS)

    def _score_retrieval(self, paper_id: str, card_json: dict) -> float:
        query_text = " ".join(
            str(card_json.get(f, "")) for f in ["title", "main_claim", "domain"] if card_json.get(f)
        ).strip()
        if not query_text:
            return 0.0
        try:
            embedding = self._embedder.embed_one(query_text)
            results = self._chroma.query(embedding, top_k=5)
            if not results:
                return 0.0
            hits = sum(1 for r in results if r.get("metadata", {}).get("paper_id") == paper_id)
            return hits / len(results)
        except Exception:
            return 0.0

    def _score_llm(self, card_json: dict) -> float:
        main_claim = str(card_json.get("main_claim", "")).strip()
        trading_rel = str(card_json.get("trading_relevance", "")).strip()
        if not main_claim:
            return 0.5

        try:
            from app.brain.local_llm import LocalLLM

            llm = LocalLLM()
            if not llm.available():
                return 0.5

            prompt = f"Aşağıdaki araştırma iddiası hakkında tek cümlelik özet yaz:\n\n{main_claim}"
            answer = llm.generate(prompt, max_tokens=120, temperature=0.0, seed=_LLM_VERIFY_SEED)
            if not answer:
                return 0.5

            keywords = _extract_keywords(main_claim + " " + trading_rel)
            answer_lower = answer.lower()
            matched = sum(1 for kw in keywords if kw in answer_lower)
            return min(1.0, matched / max(len(keywords), 1))
        except Exception:
            return 0.5


def _extract_keywords(text: str) -> list[str]:
    STOP = {
        "with",
        "that",
        "this",
        "from",
        "have",
        "been",
        "they",
        "were",
        "için",
        "olan",
        "veya",
        "ile",
        "bir",
        "the",
        "and",
        "using",
        "based",
    }
    words = re.findall(r"[a-zA-ZğüşıöçĞÜŞİÖÇ]{5,}", text.lower())
    return [w for w in words if w not in STOP][:15]
