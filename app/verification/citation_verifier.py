"""Citation verifier — checks source references in answer text.

Finds [paper_id:chunk_id] citations in the answer and verifies they
exist in the retrieved chunk list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.memory.retrieval_service import RetrievedChunk

# [paper_id:chunk_id] formatındaki atıf deseni
_CITATION_RE = re.compile(r"\[([^:\]]+):([^\]]+)\]")
# RetrievedChunk.citation sayfalı atıf üretir: "[paper:chunk, s.5]". Doğrulama için
# chunk_id'den ", s.N" son-ekini ayıkla (aksi halde chunk_id eşleşmesi hep başarısız olur).
_PAGE_SUFFIX_RE = re.compile(r",\s*s\.\d+\s*$")


@dataclass
class CitationCheck:
    """Tek bir atıfın doğrulama sonucu."""

    paper_id: str
    chunk_id: str
    exists: bool  # Atıf getirilen chunk'larda var mı?
    claim: str  # Atıfın geçtiği bağlam (yakın çevre)
    supported: bool  # Chunk metni iddiayı destekliyor mu (yaklaşık)


def _extract_context(text: str, match_start: int, window: int = 80) -> str:
    """Eşleşme etrafındaki metni döndür."""
    start = max(0, match_start - window)
    end = min(len(text), match_start + window)
    return text[start:end].strip()


class CitationVerifier:
    """Cevap metnindeki atıfları doğrulayan sınıf.

    Atıf formatı: [paper_id:chunk_id] (örn. [paper123:paper123_c0001])
    """

    def verify(
        self,
        answer_text: str,
        chunks: list[RetrievedChunk],
    ) -> list[CitationCheck]:
        """Cevap metnindeki tüm atıfları doğrula.

        Args:
            answer_text: Doğrulanacak cevap metni.
            chunks: Retrieval'dan gelen gerçek chunk'lar.

        Returns:
            Her atıf için CitationCheck listesi.
        """
        chunk_index: dict[str, RetrievedChunk] = {c.chunk_id: c for c in chunks}
        paper_ids: set[str] = {c.paper_id for c in chunks}

        results: list[CitationCheck] = []

        for match in _CITATION_RE.finditer(answer_text):
            paper_id = match.group(1).strip()
            chunk_id = _PAGE_SUFFIX_RE.sub("", match.group(2).strip()).strip()
            context = _extract_context(answer_text, match.start())

            # Chunk doğrudan mevcutsa
            chunk_exists = chunk_id in chunk_index
            paper_exists = paper_id in paper_ids

            # "exists" = hem paper_id hem chunk_id getirilmiş listede var
            exists = chunk_exists and paper_exists

            # Destekleme tahmini: chunk metnindeki token örtüşmesi
            supported = False
            if chunk_exists:
                chunk_text = chunk_index[chunk_id].text.lower()
                # Context'teki anlamlı kelimeleri kontrol et
                ctx_tokens = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}", context.lower()))
                overlap = sum(1 for tok in ctx_tokens if tok in chunk_text)
                supported = overlap >= 2

            results.append(
                CitationCheck(
                    paper_id=paper_id,
                    chunk_id=chunk_id,
                    exists=exists,
                    claim=context,
                    supported=supported,
                )
            )

        return results
