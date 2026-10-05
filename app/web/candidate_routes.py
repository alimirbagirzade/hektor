"""Aday hattı web uçları (Öğrenme Havuzu → Aday hattı).

Akış: Adayı doğrula → Ollama'ya hazırla → Karşılaştır → Sonuçları incele (kör inceleme,
``/api/compare``) → Kullanıma al / Geri dön (``/api/models``). Bu uçlar iş mantığını
TEKRARLAMAZ: doğrulama ``candidate_checks``, ağır işler ``candidate_jobs`` (CLI ile aynı
betik/komut), karar ve etkinleştirme mevcut modüllerdir.

Yetki: okuma uçları ``require_auth``; başlatma / durdurma ``require_human``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.web.security import require_auth, require_human

candidate_router = APIRouter(
    prefix="/api/candidates", tags=["candidates"], dependencies=[Depends(require_auth)]
)
_human = Depends(require_human)


class PrepareRequest(BaseModel):
    ollama_tag: str = Field(..., min_length=2, max_length=61)
    template_from: str = Field(default="", max_length=80)
    request_id: str = Field(..., min_length=8, max_length=80)


class CompareRequest(BaseModel):
    ollama_tag: str = Field(..., min_length=2, max_length=61)
    active: str = Field(default="", max_length=80)
    base: str = Field(default="", max_length=80)
    question_set: str = Field(default="", max_length=300)
    request_id: str = Field(..., min_length=8, max_length=80)


class StopRequest(BaseModel):
    reason: str = Field(default="", max_length=300)


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from app.training.candidate_jobs import JobError

    try:
        return fn(*args, **kwargs)
    except JobError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def overview() -> dict[str, Any]:
    from app.config import get_settings
    from app.evals.candidate_compare import list_comparisons
    from app.evals.candidate_decisions import list_decisions, norm_tag
    from app.feedback.model_activation import resolve_chat_tag
    from app.training import candidate_jobs as cj
    from app.training.candidate_checks import recipe_for_adapter, verify_run_completion

    s = get_settings()
    items = []
    d = s.adapters_dir
    comps = list_comparisons()
    decisions = list_decisions()
    for p in sorted(d.iterdir() if d.is_dir() else [], key=lambda x: x.stat().st_mtime):
        if not p.is_dir() or not (p / "adapter_config.json").is_file():
            continue
        recipe = recipe_for_adapter(p.name)
        comp = verify_run_completion(p.name, recipe)
        jobs = cj.list_jobs(adapter=p.name, limit=10)
        tags = sorted(
            {norm_tag(str(j["params"].get("ollama_tag"))) for j in jobs if j.get("params")}
        )
        items.append(
            {
                "adapter": p.name,
                "completion": comp,
                "suggested_tag": cj.suggest_tag(p.name),
                "jobs": jobs,
                "tags": tags,
                "comparisons": [
                    c
                    for c in comps
                    if norm_tag(str((c.get("models") or {}).get("candidate"))) in tags
                ],
                "decisions": [x for x in decisions if x.get("candidate_tag") in tags],
            }
        )
    items.reverse()
    return {
        "items": items,
        "running": cj.running_job(),
        "capabilities": {k: cj.capability(k) for k in cj.KINDS},
        "defaults": {
            "template_from": cj.DEFAULT_TEMPLATE,
            "base": cj.DEFAULT_TEMPLATE,
            "active": resolve_chat_tag("main"),
            "question_set": cj.DEFAULT_SET,
        },
        "note": "Karşılaştırma bir LLM soru-cevap ölçümüdür; trading performansı değildir. Küçük "
        "koşular entegrasyon testidir, kalite üstünlüğü kanıtı değildir.",
    }


@candidate_router.get("")
def candidates() -> dict[str, Any]:
    return overview()


@candidate_router.get("/{adapter}/verify")
def candidate_verify(adapter: str, ollama_tag: str = "") -> dict[str, Any]:
    """Adayı doğrula (salt-okuma): tamamlanma + (etiket verilirse) dönüşüm bütünlüğü."""
    from app.training.candidate_checks import (
        recipe_for_adapter,
        verify_conversion,
        verify_run_completion,
    )
    from app.training.candidate_jobs import ADAPTER_RE, TAG_RE

    if not ADAPTER_RE.match(adapter) or (ollama_tag and not TAG_RE.match(ollama_tag)):
        raise HTTPException(status_code=422, detail="Geçersiz ad.")
    out: dict[str, Any] = {
        "completion": verify_run_completion(adapter, recipe_for_adapter(adapter))
    }
    if ollama_tag:
        out["conversion"] = verify_conversion(adapter, ollama_tag)
    return out


@candidate_router.post("/{adapter}/prepare", dependencies=[_human])
def candidate_prepare(adapter: str, req: PrepareRequest) -> dict[str, Any]:
    from app.training.candidate_jobs import start_job

    return _call(
        start_job,
        "conversion",
        adapter=adapter,
        request_id=req.request_id,
        params={"ollama_tag": req.ollama_tag, "template_from": req.template_from},
    )


@candidate_router.post("/{adapter}/compare", dependencies=[_human])
def candidate_compare(adapter: str, req: CompareRequest) -> dict[str, Any]:
    from app.training.candidate_jobs import start_job

    return _call(
        start_job,
        "comparison",
        adapter=adapter,
        request_id=req.request_id,
        params={
            "ollama_tag": req.ollama_tag,
            "active": req.active,
            "base": req.base,
            "question_set": req.question_set,
        },
    )


@candidate_router.get("/jobs/{job_id}")
def candidate_job(job_id: str) -> dict[str, Any]:
    from app.training.candidate_jobs import get_job

    return _call(get_job, job_id)


@candidate_router.post("/jobs/{job_id}/stop", dependencies=[_human])
def candidate_job_stop(job_id: str, req: StopRequest) -> dict[str, Any]:
    from app.training.candidate_jobs import stop_job

    return _call(stop_job, job_id, req.reason)
