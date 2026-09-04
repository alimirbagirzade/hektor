"""Comprehension yeniden-skorlama (içerik değişimi) + determinizm seed testleri.

Tamamen çevrimdışı: ağır bağımlılıklar (SqliteStore/Chroma/Embedding/LLM) monkeypatch'lenir.
- _score_missing artık kartı skordan SONRA değişen makaleleri yeniden skorlar (donma fix'i).
- ComprehensionScorer._score_llm determinizm için sabit seed geçer (CLAUDE.md kural 6).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.research.rag_learning_loop import RagLearningLoop


class _Paper:
    def __init__(self, pid: str) -> None:
        self.paper_id = pid


class _Score:
    def __init__(self, computed_at: str) -> None:
        self.computed_at = computed_at
        self.total_score = 55.0  # eski donma eşiği (≤20) artık ölçüt DEĞİL


def test_score_missing_rescores_only_changed_or_unscored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "storage").mkdir()
    loop = RagLearningLoop()

    saved: list[str] = []

    class FakeStore:
        def list_approved_cards(self) -> list[dict]:
            return [
                {"paper_id": "p1", "created_at": "2026-01-02T00:00:00+00:00"},  # skordan YENİ
                {"paper_id": "p2", "created_at": "2026-01-01T00:00:00+00:00"},  # skordan ESKİ
                {"paper_id": "p4", "created_at": "2026-01-03T00:00:00+00:00"},  # skoru YOK
            ]

        def list_papers(self) -> list[_Paper]:
            return [_Paper("p1"), _Paper("p2"), _Paper("p3"), _Paper("p4")]

        def has_knowledge_card(self, pid: str) -> bool:
            return pid in ("p1", "p2", "p4")  # p3 kartsız → atlanır

        def get_comprehension_score(self, pid: str) -> _Score | None:
            return {
                "p1": _Score("2026-01-01T00:00:00+00:00"),  # kart daha yeni → rescore
                "p2": _Score("2026-01-02T00:00:00+00:00"),  # kart daha eski → atla
            }.get(pid)  # p4 → None (hiç skor yok → skorla)

        def save_comprehension_score(self, result: object) -> None:
            saved.append(str(result))

    class FakeScorer:
        def score(self, pid: str, use_llm: bool = True) -> str:
            return f"score::{pid}::{use_llm}"

    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", FakeStore)
    monkeypatch.setattr("app.verification.comprehension_scorer.ComprehensionScorer", FakeScorer)

    scored = loop._score_missing(limit=8)

    # p1 (kart yeni) + p4 (skorsuz) skorlanır; p2 (kart eski) ve p3 (kartsız) atlanır.
    assert scored == 2
    assert saved == ["score::p1::True", "score::p4::True"]


def test_score_missing_respects_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "storage").mkdir()
    loop = RagLearningLoop()

    class FakeStore:
        def list_approved_cards(self) -> list[dict]:
            return []  # kart-değişimi yok; sadece skorsuzları skorla

        def list_papers(self) -> list[_Paper]:
            return [_Paper(f"p{i}") for i in range(5)]

        def has_knowledge_card(self, pid: str) -> bool:
            return True

        def get_comprehension_score(self, pid: str):  # type: ignore[no-untyped-def]
            return None  # hepsi skorsuz

        def save_comprehension_score(self, result: object) -> None:
            pass

    class FakeScorer:
        def score(self, pid: str, use_llm: bool = True) -> str:
            return pid

    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", FakeStore)
    monkeypatch.setattr("app.verification.comprehension_scorer.ComprehensionScorer", FakeScorer)

    assert loop._score_missing(limit=2) == 2  # bütçe sınırı uygulanır


def test_is_newer_timestamp_compare() -> None:
    older = "2026-01-01T00:00:00+00:00"
    newer = "2026-06-21T18:00:00+00:00"
    assert RagLearningLoop._is_newer(newer, older) is True
    assert RagLearningLoop._is_newer(older, newer) is False
    assert RagLearningLoop._is_newer("", older) is False  # boş → güvenli False
    assert RagLearningLoop._is_newer("bozuk", older) is False  # parse hatası → False


def test_score_llm_passes_deterministic_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """_score_llm sabit seed + temperature=0 geçmeli (determinizm kuralı)."""
    captured: dict = {}

    class FakeLLM:
        def available(self) -> bool:
            return True

        def generate(self, prompt: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return "volatilite kümelenmesi momentum sinyalini güçlendirir"

    # ComprehensionScorer.__init__ ağır bağımlılıklarını etkisizleştir.
    monkeypatch.setattr("app.memory.chroma_store.ChromaStore", lambda *a, **k: object())
    monkeypatch.setattr("app.memory.embedding_service.EmbeddingService", lambda *a, **k: object())
    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", lambda *a, **k: object())
    monkeypatch.setattr("app.brain.local_llm.LocalLLM", FakeLLM)

    from app.verification.comprehension_scorer import ComprehensionScorer

    scorer = ComprehensionScorer()
    val = scorer._score_llm(
        {
            "main_claim": "Volatilite kümelenmesi momentum sinyalinin kalıcılığını artırır.",
            "trading_relevance": "Momentum stratejisi sinyal filtresi.",
        }
    )

    assert captured.get("seed") is not None  # seed geçildi
    assert captured.get("temperature") == 0.0  # determinist sıcaklık
    assert 0.0 <= val <= 1.0
