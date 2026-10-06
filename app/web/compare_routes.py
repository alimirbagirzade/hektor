"""Aday karşılaştırması: kör inceleme ve karar web uçları (Faz 2D).

Ağır üretim (modelleri koşturma) web'den TETİKLENMEZ: ``hektor compare-run`` ile ortak ağır iş
kilidi altında yapılır. Web yalnız kör paketi gösterir (model kimliği yok), insan puanlarını
alır ve kararı hesaplar. Karar etkinleştirme DEĞİLDİR (2A'da ayrı insan eylemi).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.web.security import require_auth, require_human

compare_router = APIRouter(
    prefix="/api/compare", tags=["compare"], dependencies=[Depends(require_auth)]
)
_human = Depends(require_human)


class ReviewRequest(BaseModel):
    scores: dict[str, dict[str, dict[str, Any]]]
    reviewer: str = Field(default="insan", max_length=80)


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from app.evals.candidate_compare import CompareError

    try:
        return fn(*args, **kwargs)
    except CompareError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@compare_router.get("")
def compare_list() -> dict[str, Any]:
    from app.evals.candidate_compare import list_comparisons

    return {"items": list_comparisons()}


@compare_router.get("/{cmp_id}/blind")
def compare_blind(cmp_id: str) -> dict[str, Any]:
    """Kör paket: cevaplar etiketli (A/B/C), model kimliği YOK; kaynaklı soruda kanıt var."""
    from app.evals.candidate_compare import blind_packet

    return {"items": _call(blind_packet, cmp_id)}


@compare_router.get("/{cmp_id}/keys")
def compare_keys(cmp_id: str) -> dict[str, Any]:
    """Doğrulanmış cevap anahtarları + kaç cevabın eşleştiği (model kimliği YOK)."""
    from app.evals.candidate_compare import key_summary

    return {"items": _call(key_summary, cmp_id)}


@compare_router.get("/{cmp_id}/ai-review")
def compare_ai_review(cmp_id: str, reveal: bool = False) -> dict[str, Any]:
    """AYRI kayıtlı AI incelemesi. İnsan incelemesi bitmeden puanlar yalnız ``reveal=true`` ile
    döner (insan inceleyici çapalanmasın). İnsan puanı DEĞİLDİR; karara girmez."""
    from app.evals.candidate_compare import AI_REVIEW_NOTE, _manifest, ai_review

    rec = _call(ai_review, cmp_id)
    if rec is None:
        return {"available": False, "note": AI_REVIEW_NOTE}
    pending_human = _call(_manifest, cmp_id).get("status") == "generated"
    if pending_human and not reveal:
        return {
            "available": True,
            "hidden": True,
            "reviewer_model": rec.get("reviewer_model"),
            "at": rec.get("at"),
            "note": AI_REVIEW_NOTE + " Kendi puanlarınızı verdikten sonra görmeniz önerilir.",
        }
    return {"available": True, "hidden": False, **rec}


@compare_router.post("/{cmp_id}/review", dependencies=[_human])
def compare_review(cmp_id: str, req: ReviewRequest) -> dict[str, Any]:
    from app.evals.candidate_compare import submit_review

    return _call(submit_review, cmp_id, req.scores, req.reviewer)


@compare_router.post("/{cmp_id}/finalize", dependencies=[_human])
def compare_finalize(cmp_id: str) -> dict[str, Any]:
    from app.evals.candidate_compare import finalize

    return _call(finalize, cmp_id)


@compare_router.get("/{cmp_id}")
def compare_result(cmp_id: str) -> dict[str, Any]:
    from app.evals.candidate_compare import result

    return _call(result, cmp_id)
