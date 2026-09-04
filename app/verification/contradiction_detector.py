"""Contradiction detector — finds opposing claims across chunks.

Flags contradictions when two chunks about the same topic contain
antonym term pairs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.memory.retrieval_service import RetrievedChunk

# Bilinen çelişki sinyali çiftleri (A kelimesi → zıt B kelimesi)
_ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("increases", "decreases"),
    ("increase", "decrease"),
    ("improves", "worsens"),
    ("improve", "worsen"),
    ("positive", "negative"),
    ("higher", "lower"),
    ("rises", "falls"),
    ("rise", "fall"),
    ("gains", "loses"),
    ("gain", "loss"),
    ("beneficial", "harmful"),
    ("supports", "contradicts"),
    ("confirms", "refutes"),
    ("stable", "unstable"),
    ("correlated", "uncorrelated"),
    ("significant", "insignificant"),
    ("profitable", "unprofitable"),
    ("outperforms", "underperforms"),
]


def _tokenize_lower(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}", text.lower()))


# Antonim kelimeleri kelime-sınırı (\b) ile eşleştir: çıplak substring (`in`) YANLIŞ çelişki
# üretiyordu — "stable" ⊂ "unstable", "lower" ⊂ "follower", "fall" ⊂ "shortfall",
# "gain" ⊂ "against", "rise" ⊂ "arise". Bu sahte çelişkiler RLM cevabına haksız "limitasyon"
# ekleyip güveni düşürüyordu. \b ile yalnız tam kelime eşleşir.
_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _word_present(word: str, text_lower: str) -> bool:
    pat = _WORD_RE_CACHE.get(word)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(word)}\b")
        _WORD_RE_CACHE[word] = pat
    return pat.search(text_lower) is not None


def _find_topic_tokens(chunks: list[RetrievedChunk]) -> set[str]:
    """Tüm chunk'larda en az 2 kez geçen kelimeleri ortak konu olarak al."""
    counter: dict[str, int] = {}
    for chunk in chunks:
        for tok in _tokenize_lower(chunk.text):
            counter[tok] = counter.get(tok, 0) + 1
    return {tok for tok, cnt in counter.items() if cnt >= 2}


@dataclass
class Contradiction:
    """İki chunk arasındaki tespit edilen çelişki."""

    chunk_id_a: str
    chunk_id_b: str
    claim_a: str  # A chunk'ındaki çelişkili ifade
    claim_b: str  # B chunk'ındaki çelişkili ifade


class ContradictionDetector:
    """Chunk listesindeki çelişkileri tespit eden sınıf.

    Yaklaşım: aynı konuda iki chunk'ta karşıt terim çiftleri aranır.
    """

    def detect(self, chunks: list[RetrievedChunk]) -> list[Contradiction]:
        """Chunk'lar arasındaki çelişkileri tespit et.

        Args:
            chunks: Kontrol edilecek RetrievedChunk listesi.

        Returns:
            Tespit edilen Contradiction listesi.
        """
        if len(chunks) < 2:
            return []

        common_topics = _find_topic_tokens(chunks)
        contradictions: list[Contradiction] = []

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                ca = chunks[i]
                cb = chunks[j]

                # Ortak konu paylaşıyorlar mı?
                tokens_a = _tokenize_lower(ca.text)
                tokens_b = _tokenize_lower(cb.text)
                shared = (tokens_a & tokens_b) & common_topics
                if not shared:
                    continue

                a_lower = ca.text.lower()
                b_lower = cb.text.lower()

                # Karşıt terim çifti var mı? (kelime-sınırı eşleşmesi — substring değil)
                for word_a, word_b in _ANTONYM_PAIRS:
                    has_a_in_a = _word_present(word_a, a_lower)
                    has_b_in_b = _word_present(word_b, b_lower)
                    has_b_in_a = _word_present(word_b, a_lower)
                    has_a_in_b = _word_present(word_a, b_lower)

                    if (has_a_in_a and has_b_in_b) or (has_b_in_a and has_a_in_b):
                        # Bağlamı çıkar (ilgili cümleyi bul)
                        claim_a = _find_sentence_with(ca.text, [word_a, word_b])
                        claim_b = _find_sentence_with(cb.text, [word_a, word_b])
                        contradictions.append(
                            Contradiction(
                                chunk_id_a=ca.chunk_id,
                                chunk_id_b=cb.chunk_id,
                                claim_a=claim_a,
                                claim_b=claim_b,
                            )
                        )
                        break  # Çift başına bir çelişki yeterli

        return contradictions


def _find_sentence_with(text: str, words: list[str]) -> str:
    """Metinde verilen kelimelerden birini içeren ilk cümleyi bul."""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent_lower = sent.lower()
        if any(w in sent_lower for w in words):
            return sent.strip()[:200]
    return text[:100].strip()
