"""RLM için GÜVENLİ tool wrapper'ları (talimat §12).

Her wrapper: girdiyi doğrular, JSON-dostu döner, mevcut Achilles servislerini (retriever,
verifier, safe_eval) sarar. HİÇBİRİ shell/network çalıştırmaz, secret/env okumaz, filesystem
yazmaz, eval/exec kullanmaz (calculator bile AST-tabanlı safe_eval'dir). Allowlist
yaptırımı `tool_registry.SafeToolRegistry`'dedir.
"""

from __future__ import annotations

from typing import Any

from app.memory.retrieval_service import RetrievedChunk
from app.rlm.tool_registry import SafeToolRegistry

_MAX_TOP_K = 20
_MAX_TEXT = 2000


def _chunk_from_dict(d: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(d.get("chunk_id", "")),
        paper_id=str(d.get("paper_id", "")),
        text=str(d.get("text", "")),
        page_number=d.get("page_number"),
        section_name=d.get("section_name"),
        title=d.get("title"),
        distance=d.get("distance"),
    )


def _chunk_to_dict(c: RetrievedChunk) -> dict[str, Any]:
    return {
        "paper_id": c.paper_id,
        "chunk_id": c.chunk_id,
        "section_name": c.section_name,
        "page_number": c.page_number,
        "title": c.title,
        "text": c.text[:_MAX_TEXT],
    }


# ---- tool implementasyonları (her biri doğrular + JSON-dostu döner) ------------


def rag_search(query: str, top_k: int = 8, **_: Any) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query boş olamaz")
    k = max(1, min(int(top_k), _MAX_TOP_K))
    from app.memory.reranking_retriever import RerankingRetriever

    chunks = RerankingRetriever().retrieve(query, top_k=k)
    return [_chunk_to_dict(c) for c in chunks]


def get_paper_metadata(paper_id: str, **_: Any) -> dict[str, Any]:
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("paper_id boş olamaz")
    from sqlalchemy import select

    from app.memory.sqlite_store import Paper, SqliteStore

    with SqliteStore().session() as s:
        p = s.scalar(select(Paper).where(Paper.paper_id == paper_id))
        if p is None:
            return {}
        return {
            "paper_id": p.paper_id,
            "title": p.title,
            "year": p.year,
            "authors": p.authors,
            "source": p.source,
            "n_pages": p.n_pages,
        }


def get_paper_chunks(
    paper_id: str, chunk_ids: list[str] | None = None, **_: Any
) -> list[dict[str, Any]]:
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("paper_id boş olamaz")
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_chunks(paper_id)
    wanted = set(chunk_ids) if chunk_ids else None
    out: list[dict[str, Any]] = []
    for c in rows:
        if wanted is not None and c.chunk_id not in wanted:
            continue
        out.append(
            {
                "paper_id": c.paper_id,
                "chunk_id": c.chunk_id,
                "section_name": c.section_name,
                "page_number": c.page_number,
                "text": (c.text or "")[:_MAX_TEXT],
            }
        )
    return out


def calculator(expression: str, **_: Any) -> float:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression boş olamaz")
    from app.verification.exams.safe_eval import safe_eval  # AST whitelist — eval/exec YOK

    return safe_eval(expression, {})


def citation_check(answer: str, evidence: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
    from app.verification.citation_verifier import CitationVerifier

    chunks = [_chunk_from_dict(d) for d in (evidence or [])]
    checks = CitationVerifier().verify(str(answer), chunks)
    valid = sum(1 for c in checks if c.exists)
    return {
        "total": len(checks),
        "valid": valid,
        "score": (valid / len(checks)) if checks else 1.0,
    }


def grounding_check(answer: str, evidence: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
    from app.verification.grounding_verifier import GroundingLevel, GroundingVerifier

    chunks = [_chunk_from_dict(d) for d in (evidence or [])]
    results = GroundingVerifier().verify(str(answer), chunks)
    unsupported = [r.claim for r in results if r.level == GroundingLevel.UNSUPPORTED]
    return {"claims": len(results), "unsupported": unsupported}


def contradiction_check(evidence: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
    from app.verification.contradiction_detector import ContradictionDetector

    chunks = [_chunk_from_dict(d) for d in (evidence or [])]
    found = ContradictionDetector().detect(chunks)
    return {"contradictions": [f"{c.chunk_id_a}↔{c.chunk_id_b}" for c in found]}


def formula_check(text: str, **_: Any) -> dict[str, Any]:
    """Hafif yapısal bütünlük: parantez/ayraç dengesi (eval YOK)."""
    t = str(text)
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    balanced = True
    for ch in t:
        if ch in "([{":
            stack.append(ch)
        # short-circuit: pop yalnız ch bir kapatıcıyken çalışır (side-effect korunur)
        elif ch in ")]}" and (not stack or stack.pop() != pairs[ch]):
            balanced = False
            break
    return {"balanced_delimiters": balanced and not stack}


def build_default_registry() -> SafeToolRegistry:
    """Varsayılan güvenli tool kayıt defteri (yalnız allowlist'teki adlar)."""
    reg = SafeToolRegistry()
    reg.register("rag_search", rag_search)
    reg.register("get_paper_metadata", get_paper_metadata)
    reg.register("get_paper_chunks", get_paper_chunks)
    reg.register("calculator", calculator)
    reg.register("citation_check", citation_check)
    reg.register("grounding_check", grounding_check)
    reg.register("contradiction_check", contradiction_check)
    reg.register("formula_check", formula_check)
    return reg
