"""Amaç bazlı dışlama + retrieval sorgu çevirisi — çevrimdışı (stub base, sahte LLM)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from app.lora.mix_common import hash_obj
from app.memory import query_translation as qt
from app.memory.doc_purpose import excluded_paper_ids, parse_purposes, purpose_of
from app.memory.rag_version import RagSnapshot
from app.memory.reranker import Reranker
from app.memory.reranking_retriever import RerankingRetriever
from app.memory.retrieval_service import RetrievedChunk

GUIDE = "paper_cd76dd9a893f"


def _chunk(cid: str, paper: str, distance: float = 0.2) -> RetrievedChunk:
    return RetrievedChunk(cid, paper, "metin", 1, "results", "T", distance)


class _Base:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, frozenset[str]]] = []

    def retrieve(
        self, query: str, top_k: int | None = None, *, exclude_papers: frozenset[str] = frozenset()
    ) -> list[RetrievedChunk]:
        self.calls.append((query, exclude_papers))
        kept = [c for c in self.chunks if c.paper_id not in exclude_papers]
        return kept[: top_k or len(kept)]


def _rr(base: _Base, **kw: object) -> RerankingRetriever:
    return RerankingRetriever(base=base, reranker=Reranker(), enabled=True, hybrid=False, **kw)


def test_purpose_config_marks_project_guides() -> None:
    assert purpose_of(GUIDE) == "proje_dokumani"
    assert purpose_of("paper_baska") == "kaynak"
    assert GUIDE in excluded_paper_ids(parse_purposes("proje_dokumani, "))
    assert excluded_paper_ids(frozenset()) == frozenset()
    with pytest.raises(ValueError, match="dışlanamaz"):
        excluded_paper_ids(frozenset({"kaynak"}))


def test_default_retriever_unchanged() -> None:
    base = _Base([_chunk("g1", GUIDE), _chunk("b1", "paper_b")])
    out = _rr(base, exclude_purposes=frozenset(), translate="off").retrieve("sorgu", top_k=2)
    assert {c.chunk_id for c in out} == {"g1", "b1"}
    assert base.calls[0][1] == frozenset()  # dışlama argümanı hiç gönderilmedi


def test_exclusion_applied_in_index_query() -> None:
    base = _Base([_chunk("g1", GUIDE), _chunk("b1", "paper_b")])
    rr = _rr(base, exclude_purposes=frozenset({"proje_dokumani"}), translate="off")
    assert [c.chunk_id for c in rr.retrieve("sorgu", top_k=2)] == ["b1"]
    assert GUIDE in base.calls[0][1]


class _FakeLLM:
    def __init__(self, answer: str) -> None:
        self.answer, self.n = answer, 0

    def generate(self, prompt: str, **kw: object) -> str:
        self.n += 1
        assert kw.get("temperature") == 0.0 and kw.get("seed") == 42
        return self.answer


def test_translation_cached_and_deterministic(tmp_path: Path) -> None:
    llm = _FakeLLM('English: "EMA formula with alpha"')
    path = tmp_path / "tr.json"
    q = "Alpha katsayısı ile EMA formülü nedir?"
    assert qt.translate_query(q, model="qwen3:x", llm=llm, path=path) == "EMA formula with alpha"
    assert qt.translate_query(q, model="qwen3:x", llm=llm, path=path) == "EMA formula with alpha"
    assert llm.n == 1  # ikinci çağrı önbellekten
    assert qt.translate_query("What is EMA?", model="qwen3:x", llm=llm, path=path) == (
        "What is EMA?"
    )  # Türkçe değil → çevrilmez


def test_translation_rejects_adapter_model_and_turkish_output(tmp_path: Path) -> None:
    with pytest.raises(qt.TranslationError, match="adapter"):
        qt.translate_query("EMA nedir ve nasıl?", model="hektor-v13-30b", path=tmp_path / "a")
    with pytest.raises(qt.TranslationError, match="Türkçe"):
        qt.translate_query(
            "EMA nedir ve nasıl?",
            model="qwen3:x",
            llm=_FakeLLM("EMA bir ortalama ve filtre"),
            path=tmp_path / "b",
        )


def test_retriever_uses_translation_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qt, "translate_query", lambda q, model: "EN " + q)
    base = _Base([_chunk("b1", "paper_b")])
    _rr(base, exclude_purposes=frozenset(), translate="en").retrieve("TR", top_k=1)
    assert [c[0] for c in base.calls] == ["EN TR"]
    base.calls.clear()
    _rr(base, exclude_purposes=frozenset(), translate="bilingual").retrieve("TR", top_k=1)
    assert [c[0] for c in base.calls] == ["EN TR", "TR"]


def test_translation_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(q: str, model: str) -> str:
        raise qt.TranslationError("yok")

    monkeypatch.setattr(qt, "translate_query", boom)
    base = _Base([_chunk("b1", "paper_b")])
    _rr(base, exclude_purposes=frozenset(), translate="en").retrieve("TR sorgu", top_k=1)
    assert base.calls[0][0] == "TR sorgu"


def test_translation_llm_unavailable_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canlıda çeviri modeli yok / Ollama kapalı → RAG düşmez, orijinal sorguyla sürer."""
    from app.brain.local_llm import LLMUnavailable

    def down(q: str, model: str) -> str:
        raise LLMUnavailable("Ollama yanıt vermedi")

    monkeypatch.setattr(qt, "translate_query", down)
    base = _Base([_chunk("b1", "paper_b")])
    out = _rr(base, exclude_purposes=frozenset(), translate="en").retrieve("TR sorgu", top_k=1)
    assert [c.chunk_id for c in out] == ["b1"] and base.calls[0][0] == "TR sorgu"


def test_invalid_translate_mode_rejected() -> None:
    with pytest.raises(ValueError, match="rag_query_translate"):
        _rr(_Base([]), translate="fr")


def _snap(**kw: object) -> RagSnapshot:
    base = {
        "embedding_model": "nomic-embed-text",
        "embedding_model_version": "v",
        "reranker": "h",
        "reranker_version": "b",
        "top_k": 6,
        "overfetch": 4,
        "hybrid": True,
        "rrf": False,
        "graph": False,
        "router": False,
        "contextual_embed": True,
        "chunk_size": 1200,
        "chunk_overlap": 200,
        "index_hash": "x",
    }
    base.update(kw)
    return RagSnapshot(**base)  # type: ignore[arg-type]


def test_rag_version_stable_for_defaults_and_split_for_new_settings() -> None:
    snap = _snap()
    legacy = {
        k: v
        for k, v in asdict(snap).items()
        if k not in {"exclude_purposes", "query_translate", "translate_model"}
    }
    assert snap.rag_version == "rag-" + hash_obj(legacy)[:12]  # eski sürüm kimliği korunur
    assert _snap(query_translate="en", translate_model="m").rag_version != snap.rag_version
    assert _snap(exclude_purposes="proje_dokumani").rag_version != snap.rag_version
