"""Çelişki tespiti testleri — kelime-sınırı eşleşmesi (substring false-positive yok).

Eski hâl antonim çiftlerini çıplak `in` (substring) ile arıyordu → "stable" ⊂ "unstable",
"lower" ⊂ "follower", "fall" ⊂ "shortfall" gibi YANLIŞ çelişkiler üretip RLM cevabına haksız
"limitasyon" ekleyip güveni düşürüyordu. Artık \b kelime-sınırı ile yalnız tam kelime eşleşir.
"""

from __future__ import annotations

from app.memory.retrieval_service import RetrievedChunk
from app.verification.contradiction_detector import ContradictionDetector


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        paper_id="p",
        text=text,
        page_number=1,
        section_name="Results",
        title="T",
        distance=0.1,
    )


def test_substring_does_not_trigger_false_contradiction() -> None:
    """'unstable' içindeki 'stable' substring'i sahte çelişki üretmemeli (kelime-sınırı)."""
    txt = "The volatility regime stays unstable during market crises persistently"
    chunks = [
        _chunk("p::c0", txt + "."),
        _chunk("p::c1", txt + " too."),
    ]
    # Her iki chunk yalnız 'unstable' içerir; tam kelime 'stable' YOK → çelişki olmamalı.
    assert ContradictionDetector().detect(chunks) == []


def test_genuine_antonym_pair_still_detected() -> None:
    """Gerçek karşıt çift (increases/decreases) hâlâ çelişki olarak yakalanmalı."""
    base = "Momentum {} the volatility persistence across equity markets."
    chunks = [
        _chunk("p::c0", base.format("increases")),
        _chunk("p::c1", base.format("decreases")),
    ]
    result = ContradictionDetector().detect(chunks)
    assert len(result) >= 1
    assert result[0].chunk_id_a == "p::c0"
    assert result[0].chunk_id_b == "p::c1"
