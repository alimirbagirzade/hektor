"""RAG ustalık metriği — LLM gerektirmeyen öğrenme/ilerleme yüzdesi.

"RAG kaç makaleyi anladı %" sorusunu yanıtlar: içerik taşıyan kart üreten
makale oranı (bilgi kapsamı) + (varsa) comprehension skoru + eğitim hazırlığı.
DB sayımlarından üretilir; ağır LLM çağrısı yapmaz (eğitimle RAM çakışmaz).
"""

from __future__ import annotations

from app.lora.dataset_builder import build_dataset
from app.memory.sqlite_store import SqliteStore


def compute_rag_mastery(store: SqliteStore | None = None) -> dict:
    """Bileşik RAG ustalık panosu döndürür (yüzdeler 0-100, tamsayı)."""
    store = store or SqliteStore()
    papers = store.list_papers()
    n_papers = len(papers)
    cards = store.list_approved_cards()
    n_cards = len(cards)

    # Yalnızca İÇERİK TAŞIYAN kartlar örnek üretir (boş kabuk kartlar üretmez).
    examples = build_dataset(cards)
    n_examples = len(examples)
    papers_with_real = len(
        {str(e.metadata.get("paper_id", "")) for e in examples if e.metadata.get("paper_id")}
    )
    empty_cards = n_cards - n_examples

    scored = 0
    total_comp = 0.0
    for p in papers:
        row = store.get_comprehension_score(p.paper_id)
        if row is not None:
            scored += 1
            total_comp += row.total_score
    avg_comp = (total_comp / scored) if scored else None

    coverage = (papers_with_real / n_papers * 100) if n_papers else 0.0
    train_readiness = min(1.0, n_examples / 50.0) * 100
    comp_component = avg_comp if avg_comp is not None else 0.0
    # Bileşik: bilgi kapsamı ×0.4 + anlama ×0.3 + eğitim hazırlığı ×0.3.
    mastery = 0.40 * coverage + 0.30 * comp_component + 0.30 * train_readiness

    return {
        "n_papers": n_papers,
        "n_cards": n_cards,
        "empty_cards": empty_cards,
        "n_examples": n_examples,
        "papers_with_real": papers_with_real,
        "papers_scored": scored,
        "coverage_percent": round(coverage),
        "train_readiness_percent": round(train_readiness),
        "comprehension_percent": round(avg_comp) if avg_comp is not None else None,
        "mastery_percent": round(mastery),
    }
