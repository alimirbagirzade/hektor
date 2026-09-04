"""Claim extractor — taslak cevabı iddialara böler ve dayanak durumunu işaretler.

LLM-free: mevcut GroundingVerifier çıktısını (cümle düzeyi dayanak) yapısal
Claim nesnelerine çevirir. Talimat §14'teki claim object şemasını üretir:

    {"claim": ..., "support_status": ..., "supporting_chunks": [...], "notes": ...}

Desteklenmeyen iddialar nihai cevaptan ÇIKARILIR (CLAUDE.md: desteklenmeyen
iddiayı nihai cevaba koyma).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.verification.grounding_verifier import GroundingLevel, GroundingResult

# GroundingLevel → spec support_status string eşlemesi.
_STATUS_MAP = {
    GroundingLevel.SUPPORTED: "supported",
    GroundingLevel.PARTIALLY_SUPPORTED: "partially_supported",
    GroundingLevel.SPECULATIVE: "speculative",
    GroundingLevel.UNSUPPORTED: "unsupported",
}


@dataclass
class Claim:
    """Tek bir iddia ve dayanak durumu."""

    claim: str
    support_status: str
    supporting_chunks: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_supported(self) -> bool:
        """Nihai cevaba girmeye uygun mu? (supported veya partially_supported)."""
        return self.support_status in ("supported", "partially_supported")


def extract_claims(groundings: list[GroundingResult]) -> list[Claim]:
    """Grounding sonuçlarından yapısal iddia listesi üret."""
    claims: list[Claim] = []
    for g in groundings:
        status = _STATUS_MAP.get(g.level, "unsupported")
        chunks = [g.evidence_chunk_id] if g.evidence_chunk_id else []
        notes = ""
        if status == "speculative":
            notes = "Temkinli/hedge'li ifade — desteklense de güven indirgenir."
        elif status == "unsupported":
            notes = "Kaynak chunk'larda dayanak bulunamadı; nihai cevaptan çıkarıldı."
        claims.append(
            Claim(
                claim=g.claim,
                support_status=status,
                supporting_chunks=chunks,
                notes=notes,
            )
        )
    return claims
