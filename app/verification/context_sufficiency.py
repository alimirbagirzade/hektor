"""Context sufficiency classifier.

Evaluates whether the retrieved chunk list is sufficient to answer the query;
reports missing formula continuation or incomplete argument conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.memory.contextual_chunker import ChunkQualityFlags
from app.memory.retrieval_service import RetrievedChunk


class SufficiencyLevel(Enum):
    """Bağlam yeterliliği düzeyi."""

    SUFFICIENT = "sufficient"
    PARTIALLY_SUFFICIENT = "partially_sufficient"
    INSUFFICIENT = "insufficient"
    CONTRADICTORY = "contradictory"
    MISSING_FORMULA_CONTINUATION = "missing_formula_continuation"
    MISSING_ARGUMENT_CONCLUSION = "missing_argument_conclusion"


@dataclass
class SufficiencyResult:
    """Bağlam yeterliliği değerlendirmesi sonucu."""

    level: SufficiencyLevel
    missing_items: list[str] = field(default_factory=list)
    can_answer: bool = False


class ContextSufficiencyClassifier:
    """Bağlam yeterliliğini değerlendiren sınıflandırıcı.

    Kurallar:
    - Chunk yoksa: INSUFFICIENT, can_answer=False.
    - Tüm chunk'larda eksik formül varsa ve komşu yok ise:
      MISSING_FORMULA_CONTINUATION, can_answer=False.
    - Eksik argüman tespiti: MISSING_ARGUMENT_CONCLUSION, can_answer=True (kısmi —
      güven düşürülür ama tümden reddedilmez; abstention düşük güvende tetiklenir).
    - Normal durumda: SUFFICIENT/PARTIALLY_SUFFICIENT, can_answer=True.
    """

    def classify(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        quality_flags: list[ChunkQualityFlags] | None = None,
    ) -> SufficiencyResult:
        """Bağlam yeterliliğini sınıflandır.

        Args:
            query: Kullanıcı sorgusu.
            chunks: Getirilen chunk listesi.
            quality_flags: ChunkQualityFlags listesi (opsiyonel).

        Returns:
            SufficiencyResult nesnesi.
        """
        if not chunks:
            return SufficiencyResult(
                level=SufficiencyLevel.INSUFFICIENT,
                missing_items=["Hiç chunk getirilmedi"],
                can_answer=False,
            )

        missing_items: list[str] = []

        if quality_flags:
            # Yalnız GERÇEKTEN getirilen chunk'ların bayraklarını dikkate al: quality_flags
            # chunks ile hizasız gelebilir (fazla/eksik/başka id). Aşağıdaki
            # len(incomplete_formula_ids) == len(chunks) kıyası iki AYRI koleksiyonu kıyaslayıp
            # yanlış MISSING_FORMULA_CONTINUATION (can_answer=False) üretirdi. Kesişimle sınırla:
            chunk_ids = {c.chunk_id for c in chunks}
            flags_by_id = {f.chunk_id: f for f in quality_flags if f.chunk_id in chunk_ids}

            incomplete_formula_ids = [
                cid
                for cid, f in flags_by_id.items()
                if f.has_incomplete_formula and not f.next_chunk_id
            ]
            incomplete_arg_ids = [
                cid for cid, f in flags_by_id.items() if f.has_incomplete_argument
            ]

            # Tüm chunk'larda formül eksikse ve hiç komşu yoksa: kritik sorun
            all_have_incomplete = len(incomplete_formula_ids) == len(chunks)
            if all_have_incomplete:
                return SufficiencyResult(
                    level=SufficiencyLevel.MISSING_FORMULA_CONTINUATION,
                    missing_items=[f"Eksik formül devamı: {cid}" for cid in incomplete_formula_ids],
                    can_answer=False,
                )

            if incomplete_formula_ids:
                missing_items.extend([f"Eksik formül: {cid}" for cid in incomplete_formula_ids])

            if incomplete_arg_ids:
                # Eksik argüman varsa kısmen yetersiz
                missing_items.extend([f"Eksik argüman: {cid}" for cid in incomplete_arg_ids])

        if missing_items:
            # Kısmi sorunlar var ama yanıtlanabilir
            level = (
                SufficiencyLevel.MISSING_ARGUMENT_CONCLUSION
                if any("argüman" in m for m in missing_items)
                else SufficiencyLevel.PARTIALLY_SUFFICIENT
            )
            return SufficiencyResult(
                level=level,
                missing_items=missing_items,
                can_answer=True,
            )

        # Her şey yolunda
        level = (
            SufficiencyLevel.SUFFICIENT
            if len(chunks) >= 3
            else SufficiencyLevel.PARTIALLY_SUFFICIENT
        )
        return SufficiencyResult(level=level, missing_items=[], can_answer=True)
