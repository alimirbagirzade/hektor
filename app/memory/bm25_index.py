"""BM25 search index — pure Python, no external dependencies.

Implements BM25 scoring with TF-IDF approximation; does not require sklearn.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Küçük harfe çevirip alfanümerik token listesi döndür."""
    return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text.lower())


class BM25Index:
    """Minimal BM25 arama indeksi.

    Parametreler:
        k1 (float): Term frekansı doygunluk faktörü (varsayılan 1.5).
        b (float): Belge uzunluğu normalleştirme faktörü (varsayılan 0.75).

    Kullanım::

        idx = BM25Index()
        idx.add_document("doc1", "ATR momentum volatility filter")
        idx.add_document("doc2", "Sharpe ratio risk adjusted return")
        results = idx.search("volatility momentum", top_k=5)
        # [("doc1", 2.34), ...]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        # doc_id → token Counter
        self._doc_tokens: dict[str, Counter] = {}
        # doc_id → token sayısı
        self._doc_lengths: dict[str, int] = {}
        # token → doc_id set (ters indeks)
        self._inverted: dict[str, set[str]] = {}
        # Ortalama belge uzunluğu (cache)
        self._avg_dl: float = 0.0

    def add_document(self, doc_id: str, text: str) -> None:
        """Belgeyi indekse ekle.

        Args:
            doc_id: Belge kimliği (chunk_id).
            text: Belge metni.
        """
        tokens = _tokenize(text)
        counter = Counter(tokens)
        self._doc_tokens[doc_id] = counter
        self._doc_lengths[doc_id] = len(tokens)

        for token in counter:
            if token not in self._inverted:
                self._inverted[token] = set()
            self._inverted[token].add(doc_id)

        # Ortalama uzunluğu güncelle
        total = sum(self._doc_lengths.values())
        self._avg_dl = total / len(self._doc_lengths) if self._doc_lengths else 1.0

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """BM25 skora göre en iyi belgeleri döndür.

        Args:
            query: Arama sorgusu.
            top_k: Döndürülecek maksimum sonuç sayısı.

        Returns:
            [(doc_id, bm25_score)] listesi, skora göre azalan sırada.
        """
        if not self._doc_tokens:
            return []

        q_tokens = _tokenize(query)
        n = len(self._doc_tokens)
        scores: dict[str, float] = {}

        for token in q_tokens:
            if token not in self._inverted:
                continue

            df = len(self._inverted[token])  # Belge frekansı
            # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            for doc_id in self._inverted[token]:
                tf = self._doc_tokens[doc_id][token]
                dl = self._doc_lengths[doc_id]
                avg_dl = self._avg_dl if self._avg_dl > 0 else 1.0

                # BM25 TF normalleşmesi
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm

        # Berabere skorlarda ikincil doc_id tie-break → determinizm (CLAUDE.md Kural 6).
        # set-iterasyon sırası PYTHONHASHSEED'e bağlı olduğundan tie-break olmadan
        # eşit-skorlu belgelerin sırası süreçler arası değişir (rank_fusion ile aynı sözleşme).
        sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return sorted_scores[:top_k]

    def __len__(self) -> int:
        return len(self._doc_tokens)
