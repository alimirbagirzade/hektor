"""status_manager.py — Makale status geçişlerini yönetir ve tarihçeye yazar."""

from __future__ import annotations

from app.memory.mastery_store import MasteryStore

VALID_STATUSES = {
    "uploaded",
    "parsed",
    "metadata_extracted",
    "chunked",
    "indexed",
    "retrievable",
    "tested",
    "learned",
    "partially_learned",
    "usable_needs_review",
    "needs_rechunking",
    "needs_reindexing",
    "needs_human_review",
    "failed",
}


class StatusManager:
    """Makale status geçişlerini yöneten ve tarihçeye yazan sınıf."""

    def __init__(self, store: MasteryStore | None = None) -> None:
        self._store = store or MasteryStore()

    def update(self, paper_id: str, new_status: str, reason: str | None = None) -> None:
        """paper_id için new_status'a geçiş yap ve tarihçeye kaydet."""
        old_status = self._store.get_current_status(paper_id)
        self._store.record_status_change(
            paper_id=paper_id,
            new_status=new_status,
            old_status=old_status,
            reason=reason,
        )

    def get_current(self, paper_id: str) -> str:
        return self._store.get_current_status(paper_id)

    def get_history(self, paper_id: str) -> list[dict]:
        return self._store.get_status_history(paper_id)

    def status_from_score(self, total_score: float) -> str:
        """MasteryScore.total_score'a göre status belirle."""
        if total_score >= 90:
            return "learned"
        if total_score >= 75:
            return "usable_needs_review"
        if total_score >= 60:
            return "partially_learned"
        if total_score >= 40:
            return "needs_rechunking"
        return "failed"
