"""RagAnswerer tests.

Offline tests inject a stub retriever + stub LLM so the RAG orchestration logic
is verified without Ollama. One ``@pytest.mark.ollama`` integration test runs the
real local model against a deterministic (stubbed) context.
"""

from __future__ import annotations

import pytest

from app.brain.local_llm import LLMUnavailable, LocalLLM
from app.brain.rag_answerer import RagAnswerer
from app.memory.retrieval_service import RetrievedChunk


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"paper_x::c{i:04d}",
        paper_id="paper_x",
        text=f"Volatilite kümelenmesi momentum kalıcılığını etkiler ({i}).",
        page_number=i + 1,
        section_name="Results",
        title="Vol Clustering & Momentum",
        distance=0.1 * i,
    )


class _StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self._chunks


class _StubLLM:
    """Records the last prompt/system and returns a canned answer."""

    def __init__(self, response: str = "1. Kısa cevap\n2. Kaynaklar\n...") -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.model = "stub-model"

    def generate(self, prompt: str, *, system: str | None = None, **_: object) -> str:
        self.last_prompt = prompt
        self.last_system = system
        return self.response


class _UnavailableLLM:
    model = "offline-model"

    def generate(self, prompt: str, *, system: str | None = None, **_: object) -> str:
        raise LLMUnavailable("ollama down")


def test_no_sources_returns_grounded_refusal():
    rag = RagAnswerer(retriever=_StubRetriever([]), llm=_StubLLM())
    ans = rag.answer("Bilinmeyen bir konu?")
    assert ans.sources == []
    assert ans.llm_used is False
    assert "Kaynak bulunamadı" in ans.answer


def test_answer_with_sources_invokes_llm_and_passes_context():
    chunks = [_chunk(0), _chunk(1)]
    llm = _StubLLM(response="MODEL CEVABI")
    rag = RagAnswerer(retriever=_StubRetriever(chunks), llm=llm)

    ans = rag.answer("Momentum kalıcı mı?", top_k=2)

    assert ans.llm_used is True
    assert ans.answer == "MODEL CEVABI"
    assert ans.sources == chunks
    # the retrieved context (citations) must be embedded in the prompt
    assert llm.last_prompt is not None
    assert "[paper_x:paper_x::c0000" in llm.last_prompt
    assert "SORU: Momentum kalıcı mı?" in llm.last_prompt


def test_llm_unavailable_degrades_to_sources_only():
    chunks = [_chunk(0)]
    rag = RagAnswerer(retriever=_StubRetriever(chunks), llm=_UnavailableLLM())

    ans = rag.answer("Soru?")

    assert ans.llm_used is False
    assert ans.sources == chunks
    assert "LLM çevrimdışı" in ans.answer
    assert "paper_x" in ans.answer  # citation still shown


# --------------------------------------------------------------------------
# Integration: real local model (skipped unless Ollama + model are ready)
# --------------------------------------------------------------------------
def _ollama_model_ready(model: str) -> bool:
    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return any(model.split(":")[0] in n for n in names)
    except Exception:
        return False


@pytest.mark.ollama
def test_rag_real_llm_with_stub_context():
    llm = LocalLLM()
    if not _ollama_model_ready(llm.model):
        pytest.skip(f"Ollama model not ready: {llm.model}")

    rag = RagAnswerer(retriever=_StubRetriever([_chunk(0), _chunk(1)]), llm=llm)
    ans = rag.answer("Volatilite kümelenmesi momentum için ne ima eder?")

    assert ans.llm_used is True
    assert isinstance(ans.answer, str)
    assert len(ans.answer.strip()) > 0
    assert len(ans.sources) == 2
