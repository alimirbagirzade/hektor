"""Cevap-kalitesi yardımcıları + RagAnswerer entegrasyon testleri (sentetik, Chroma'sız).

CRAG-lite abstain kapısı + lost-in-the-middle reorder deterministik → gerçek embedding/
Chroma gerekmez; sahte RetrievedChunk'larla tam doğrulanır.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.brain.answer_quality import (
    assess_confidence,
    is_weak_retrieval,
    reorder_lost_in_middle,
    similarity,
)
from app.brain.rag_answerer import RagAnswerer
from app.memory.retrieval_service import RetrievedChunk


def _chunk(cid: str, distance: float, text: str = "içerik") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        paper_id="p1",
        text=text,
        page_number=1,
        section_name="Methods",
        title="Paper",
        distance=distance,
    )


# ---- similarity ----
def test_similarity_maps_cosine_distance() -> None:
    assert similarity(0.0) == 1.0
    assert similarity(1.0) == 0.0
    assert similarity(0.4) == 0.6
    assert similarity(None) == 0.0
    assert similarity(2.0) == 0.0  # aralık-dışı klemp
    assert similarity(-0.1) == 1.0


# ---- assess_confidence ----
def test_assess_confidence_empty() -> None:
    c = assess_confidence([])
    assert c.n == 0 and c.best_similarity == 0.0


def test_assess_confidence_orders() -> None:
    c = assess_confidence([_chunk("a", 0.3), _chunk("b", 0.5)])
    assert c.n == 2
    assert abs(c.best_similarity - 0.7) < 1e-9
    assert abs(c.margin - 0.2) < 1e-9  # 0.7 - 0.5


# ---- is_weak_retrieval ----
def test_weak_when_empty() -> None:
    assert is_weak_retrieval(assess_confidence([]), 0.18, 0.02) is True


def test_strong_retrieval_not_weak() -> None:
    conf = assess_confidence([_chunk("a", 0.3), _chunk("b", 0.6)])  # sim 0.7 / 0.4
    assert is_weak_retrieval(conf, 0.18, 0.02) is False


def test_weak_when_best_below_floor() -> None:
    conf = assess_confidence([_chunk("a", 0.9), _chunk("b", 0.95)])  # sim 0.1 / 0.05
    assert is_weak_retrieval(conf, 0.18, 0.02) is True


# ---- reorder_lost_in_middle ----
def test_reorder_keeps_short_lists() -> None:
    cs = [_chunk("a", 0.1), _chunk("b", 0.2)]
    assert reorder_lost_in_middle(cs) == cs


def test_reorder_places_best_at_edges_same_set() -> None:
    cs = [_chunk(x, 0.1) for x in ["r0", "r1", "r2", "r3", "r4"]]
    out = reorder_lost_in_middle(cs)
    ids = [c.chunk_id for c in out]
    # rank0 başta, rank1 sonda (en güçlüler uçta); ortada en zayıflar
    assert ids[0] == "r0"
    assert ids[-1] == "r1"
    assert set(ids) == {"r0", "r1", "r2", "r3", "r4"}  # eklenmez/çıkarılmaz
    assert ids[len(ids) // 2] in {"r3", "r4"}  # orta = en zayıf


# ---- RagAnswerer entegrasyon ----
class _StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._c = chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self._c


class _StubLLM:
    def __init__(self, reply: str = "CEVAP") -> None:
        self.calls = 0
        self.last_prompt = ""
        self._reply = reply

    def generate(self, prompt: str, system: str | None = None, **kw: object) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self._reply


def _answerer(chunks: list[RetrievedChunk], llm: _StubLLM, **flags: object) -> RagAnswerer:
    a = RagAnswerer(retriever=_StubRetriever(chunks), llm=llm)  # type: ignore[arg-type]
    a.settings = SimpleNamespace(  # type: ignore[assignment]
        rag_abstain=flags.get("abstain", False),
        rag_abstain_min_similarity=flags.get("min_sim", 0.18),
        rag_abstain_min_margin=flags.get("min_margin", 0.02),
        rag_reorder_context=flags.get("reorder", True),
        rag_verify_citations=flags.get("verify_cites", False),
    )
    return a


def test_abstain_on_weak_retrieval_does_not_call_llm() -> None:
    llm = _StubLLM()
    a = _answerer([_chunk("a", 0.9), _chunk("b", 0.95)], llm, abstain=True)
    res = a.answer("alakasız sorgu")
    assert res.llm_used is False
    assert llm.calls == 0  # zayıf → LLM çağrılmadı (uydurma yok)
    assert "Yetersiz dayanak" in res.answer or "Insufficient grounding" in res.answer
    assert res.sources  # zayıf eşleşmeler yine gösterilir


def test_strong_retrieval_answers_normally() -> None:
    llm = _StubLLM()
    a = _answerer([_chunk("a", 0.3), _chunk("b", 0.5)], llm, abstain=True)
    res = a.answer("iyi sorgu")
    assert res.llm_used is True
    assert llm.calls == 1
    assert res.answer == "CEVAP"


def test_abstain_disabled_answers_even_if_weak() -> None:
    llm = _StubLLM()
    a = _answerer([_chunk("a", 0.95)], llm, abstain=False)
    res = a.answer("zayıf ama kapı kapalı")
    assert res.llm_used is True and llm.calls == 1


def test_reorder_applied_to_llm_context() -> None:
    llm = _StubLLM()
    cs = [_chunk(x, 0.2, text=f"metin-{x}") for x in ["r0", "r1", "r2", "r3", "r4"]]
    a = _answerer(cs, llm, abstain=False, reorder=True)
    a.answer("soru")
    # LLM prompt'unda en güçlü (r0) ortadan ÖNCE; r0 başta r1 sonda olmalı (reorder izi)
    p = llm.last_prompt
    assert p.index("metin-r0") < p.index("metin-r3")  # r0 ortadakilerden önce
    assert p.index("metin-r1") > p.index("metin-r3")  # r1 sona doğru


# ---- citation verification ----
def _pc(cid: str, pid: str = "p1") -> RetrievedChunk:
    return RetrievedChunk(cid, pid, "t", 1, "Methods", "T", 0.3)


def test_verify_citations_all_supported() -> None:
    from app.brain.answer_quality import verify_citations

    chunks = [_pc("c1", "pa"), _pc("c2", "pb")]
    chk = verify_citations("İddia [pa:c1]. Başka [pb:c2, s.3].", chunks)
    assert chk.n_cited == 2 and not chk.has_unsupported


def test_verify_citations_paper_match_ok_chunk_off() -> None:
    from app.brain.answer_quality import verify_citations

    # chunk_id getirilmedi ama paper_id eşleşiyor → desteklenir (yaklaşık id)
    chk = verify_citations("[pa:cX]", [_pc("c1", "pa")])
    assert not chk.has_unsupported


def test_verify_citations_flags_hallucinated() -> None:
    from app.brain.answer_quality import verify_citations

    chk = verify_citations("Gerçek [pa:c1]. Uydurma [pZ:cZ].", [_pc("c1", "pa")])
    assert chk.n_cited == 2
    assert chk.unsupported == ["pZ:cZ"]


def test_verify_citations_empty() -> None:
    from app.brain.answer_quality import verify_citations

    assert verify_citations("", [_pc("c1")]).n_cited == 0
    assert verify_citations("metin [p:c]", []).n_cited == 0


def test_citation_warning_only_when_unsupported() -> None:
    from app.brain.answer_quality import CitationCheck, citation_warning

    assert citation_warning(CitationCheck(n_cited=2)) == ""
    w = citation_warning(CitationCheck(n_cited=2, unsupported=["pZ:cZ"]))
    assert "pZ:cZ" in w and ("DİKKAT" in w or "Auto-check" in w)


def test_answerer_appends_warning_for_hallucinated_citation() -> None:
    llm = _StubLLM(reply="İddia [p1:c1]. Uydurma [pX:cX].")
    a = _answerer([_pc("c1", "p1")], llm, verify_cites=True)
    res = a.answer("soru")
    assert "pX:cX" in res.answer  # uydurma atıf uyarıda
    assert "DİKKAT" in res.answer or "Auto-check" in res.answer


def test_answerer_no_warning_when_disabled() -> None:
    llm = _StubLLM(reply="Uydurma [pX:cX].")
    a = _answerer([_pc("c1", "p1")], llm, verify_cites=False)
    res = a.answer("soru")
    assert "DİKKAT" not in res.answer and res.answer == "Uydurma [pX:cX]."
