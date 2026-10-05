"""Sohbetten öğrenme testleri için ortak sahteler (çevrimdışı; Ollama/Chroma yok)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.memory.retrieval_service import RetrievedChunk

CHUNK_TEXT = (
    "Volatilite kümelenmesi yüksek oynaklık dönemlerinin ardışık olarak birbirini izlemesi "
    "anlamına gelir ve getirilerin varyansı zamanla değişir."
)
SUPPORTED_SENTENCE = (
    "Volatilite kümelenmesi yüksek oynaklık dönemlerinin ardışık olarak birbirini izlemesi "
    "anlamına gelir."
)


def chunk(cid: str = "c1", pid: str = "p1", text: str = CHUNK_TEXT) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        paper_id=pid,
        text=text,
        page_number=1,
        section_name=None,
        title="Test makalesi",
        distance=0.1,
    )


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = [chunk()] if chunks is None else chunks
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        return list(self.chunks)


class FakeLLM:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = answers or []
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, **kw: Any) -> str:
        self.calls.append({"prompt": prompt, **kw})
        if self.answers:
            return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return f"{SUPPORTED_SENTENCE} [p1:c1]"


@pytest.fixture
def iso(monkeypatch, tmp_path: Path):
    """Veri kökü + SQLite'ı teste özel tmp'ye al; sızıntı sağlayıcısını boşalt."""
    from app.config import settings as settings_mod
    from app.feedback import learning

    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    monkeypatch.setenv("HEKTOR_SQLITE_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("HEKTOR_LEARNING_TOKENIZER", "approx")
    settings_mod.get_settings.cache_clear()
    monkeypatch.setattr(learning, "leak_items_provider", lambda: [])
    yield tmp_path
    settings_mod.get_settings.cache_clear()


def send(
    store: Any,
    conv_id: str,
    question: str,
    *,
    llm: Any = None,
    retriever: Any = None,
    req: str | None = None,
) -> dict[str, Any]:
    from app.feedback.chat_service import send as _send

    turn, _ = _send(
        conv_id,
        question,
        req or f"r-{question[:20]}-{len(store.list_turns(conv_id))}",
        retriever=retriever or StubRetriever(),
        llm=llm or FakeLLM(),
        store=store,
    )
    return turn
