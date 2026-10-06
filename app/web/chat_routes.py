"""Tek sohbet ekranı + öğrenme havuzu web uçları (Faz 1).

Yetki:
- Konuşma/tur okuma, Faydalı/Hatalı, hariç tutma, önizleme → ``require_auth``.
- Eğitim ADAYI oluşturma (Öğrensin/Düzelt), aday düzenleme, gerekçeli insan onayı, veri sürümü
  oluşturma, kontrollü ölçüm → ``require_human`` (motor kendi eğitim verisini üretemez/onaylayamaz).
- Hiçbir uç eğitim başlatmaz ve bulut çağrısı yapmaz (Kural 8). Model etkinleştirme
  (``/api/models/*``) YALNIZ insan eylemiyle ve karşılaştırma kararına bağlı kurallarla yapılır
  (``model_activation``); hiçbir şey kendiliğinden etkinleştirilmez.

GET uçları YAZMAZ; koşu gözlemi ayrı ``POST /api/learn/runs/sync`` ile yapılır.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.web.security import require_auth, require_human

chat_router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(require_auth)])
learn_router = APIRouter(prefix="/api/learn", tags=["learn"], dependencies=[Depends(require_auth)])
models_router = APIRouter(
    prefix="/api/models", tags=["models"], dependencies=[Depends(require_auth)]
)
_human = Depends(require_human)


class ConversationCreate(BaseModel):
    title: str = Field(default="", max_length=200)
    slot: str = Field(default="main", pattern="^(main|trial)$")
    is_test: bool = False


class ActivateRequest(BaseModel):
    tag: str = Field(..., min_length=1, max_length=200)
    reason: str = Field(default="", max_length=2000)


class ReasonRequest(BaseModel):
    reason: str = Field(..., min_length=10, max_length=2000)


class FeedbackRequest(BaseModel):
    label: str = Field(..., pattern="^(useful|wrong|)$")
    note: str = Field(default="", max_length=4000)
    spans: list[dict[str, Any]] | None = None


class LearnRequest(BaseModel):
    domain: str | None = Field(default=None, max_length=20)


class CorrectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    domain: str | None = Field(default=None, max_length=20)
    spans: list[dict[str, Any]] | None = None


class ExcludeRequest(BaseModel):
    excluded: bool = True
    reason: str = Field(default="", max_length=2000)


class ApproveRequest(BaseModel):
    reason: str = Field(..., min_length=10, max_length=4000)


class EditRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    domain: str | None = Field(default=None, max_length=20)


class LinkRunRequest(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=40)


class DatasetRequest(BaseModel):
    include_human: bool = True


def _svc() -> Any:
    from app.feedback.learning import LearningService

    return LearningService()


def _store() -> Any:
    from app.feedback.chat_store import ChatStore

    return ChatStore()


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from app.feedback.learning import LearningError

    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    except LearningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ── sohbet ───────────────────────────────────────────────────────────────────


def _slot(slot: str) -> str:
    if slot not in ("main", "trial"):
        raise HTTPException(status_code=422, detail=f"Geçersiz yuva: {slot}")
    return slot


@chat_router.get("/model")
def chat_model(slot: str = "main") -> dict[str, Any]:
    """Yuvanın (ana/deneme) model kimliği (etiket, digest, köken kaydı, son ölçüm)."""
    from app.feedback.model_identity import describe_chat_model

    return describe_chat_model(slot=_slot(slot))


@chat_router.get("/resources")
def chat_resources(slot: str = "main") -> dict[str, Any]:
    """Sunucu tarafı kaynak kararı (cevaplama açık/kapalı + gerekçe + ölçümler)."""
    from app.feedback.resource_guard import check_chat_resources

    return check_chat_resources(slot=_slot(slot)).to_dict()


@chat_router.post("/measure", dependencies=[_human])
def chat_measure() -> dict[str, Any]:
    """Kontrollü ilk ölçüm (eğitim YOKKEN): modeli kısa üretimle yükle, ayak izini kaydet."""
    from app.feedback.chat_service import measure

    try:
        return measure()
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@chat_router.get("/conversations")
def list_conversations(limit: int = 50) -> dict[str, Any]:
    return {"conversations": _store().list_conversations(limit=max(1, min(limit, 200)))}


@chat_router.post("/conversations")
def create_conversation(req: ConversationCreate) -> dict[str, Any]:
    return _store().create_conversation(req.title, slot=req.slot, is_test=req.is_test)


@chat_router.post("/conversations/{conversation_id}/mark-test")
def mark_test_conversation(conversation_id: str) -> dict[str, Any]:
    """Sohbeti TEST olarak işaretle (tek yönlü): adayları hiçbir veri sürümüne girmez."""
    return _call(_store().mark_test, conversation_id)


@chat_router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    store = _store()
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Konuşma bulunamadı: {conversation_id}")
    turns = store.list_turns(conversation_id)
    for t in turns:
        t["candidates"] = [
            {
                k: c[k]
                for k in (
                    "candidate_id",
                    "kind",
                    "status",
                    "status_reason",
                    "verification",
                    "target_text",
                )
            }
            for c in store.candidates_for_turn(t["turn_id"])
        ]
    return {"conversation": conv, "turns": turns}


@chat_router.post("/turns/{turn_id}/feedback")
def turn_feedback(turn_id: str, req: FeedbackRequest) -> dict[str, Any]:
    """Faydalı / Hatalı (aday ÜRETMEZ). Boş etiket işareti kaldırır."""
    return _call(_svc().set_feedback, turn_id, req.label, req.note, req.spans)


@chat_router.post("/turns/{turn_id}/exclude", dependencies=[_human])
def turn_exclude(turn_id: str, req: ExcludeRequest) -> dict[str, Any]:
    """Eğitimden hariç tut / geri al. Eski veri sürümü dosyalarını değiştirmez."""
    return _call(_svc().set_excluded, turn_id, req.excluded, req.reason)


@chat_router.post("/turns/{turn_id}/learn", dependencies=[_human])
def turn_learn(turn_id: str, req: LearnRequest) -> dict[str, Any]:
    """'Öğrensin' — model cevabı eğitim adayı olur (tekrar tıklama aynı adayı döndürür)."""
    cand, created = _call(_svc().learn, turn_id, req.domain)
    return {"candidate": cand, "created": created}


@chat_router.post("/turns/{turn_id}/correct", dependencies=[_human])
def turn_correct(turn_id: str, req: CorrectRequest) -> dict[str, Any]:
    """'Düzelt' — kullanıcının metni eğitim adayı olur (turda tek aktif düzeltme)."""
    cand, created = _call(_svc().correct, turn_id, req.text, domain=req.domain, spans=req.spans)
    return {"candidate": cand, "created": created}


# ── öğrenme havuzu ───────────────────────────────────────────────────────────


@learn_router.get("/summary")
def learn_summary() -> dict[str, Any]:
    return _svc().summary()


@learn_router.get("/candidates")
def learn_candidates(status: str = "", limit: int = 200) -> dict[str, Any]:
    return {"items": _svc().list_candidates(status or None, limit=max(1, min(limit, 1000)))}


@learn_router.get("/errors")
def learn_errors() -> dict[str, Any]:
    return {"items": _svc().error_queue()}


@learn_router.post("/candidates/{candidate_id}/approve", dependencies=[_human])
def learn_approve(candidate_id: str, req: ApproveRequest) -> dict[str, Any]:
    return _call(_svc().approve, candidate_id, req.reason)


@learn_router.post("/candidates/{candidate_id}/link-run", dependencies=[_human])
def learn_link_run(candidate_id: str, req: LinkRunRequest) -> dict[str, Any]:
    """Adayı kayıtlı strateji test koşusuna bağla (zaman alanları + strateji ailesi)."""
    return _call(_svc().link_run, candidate_id, req.run_id)


@learn_router.post("/candidates/{candidate_id}/edit", dependencies=[_human])
def learn_edit(candidate_id: str, req: EditRequest) -> dict[str, Any]:
    return _call(_svc().edit, candidate_id, req.text, domain=req.domain)


@learn_router.post("/datasets/preview")
def learn_preview(req: DatasetRequest) -> dict[str, Any]:
    """Veri sürümü özeti (yazma yok)."""
    from app.feedback.chat_dataset import preview

    return preview(include_human=req.include_human)


@learn_router.post("/datasets", dependencies=[_human])
def learn_create_dataset(req: DatasetRequest) -> dict[str, Any]:
    """Değişmez veri sürümü oluştur. Eğitim BAŞLATMAZ; kanonik dosyaya bağlamaz."""
    from app.feedback.chat_dataset import create_version

    try:
        version, created = create_version(include_human=req.include_human)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"version": version, "created": created}


@learn_router.get("/datasets")
def learn_datasets() -> dict[str, Any]:
    from app.feedback.chat_dataset import version_overview

    return version_overview(_store())


@learn_router.post("/runs/sync")
def learn_runs_sync() -> dict[str, Any]:
    """Bağlanmış veriyle başlatılan eğitim koşularını gözle (dosya okur, kayıt günceller)."""
    from app.feedback.chat_dataset import observe_runs

    return {"runs": observe_runs()}


# ── model etkinleştirme (Faz 2A; yalnız insan) ───────────────────────────────


def _act(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from app.feedback.model_activation import ActivationError

    try:
        return fn(*args, **kwargs)
    except ActivationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@models_router.get("/activation")
def models_activation() -> dict[str, Any]:
    """Ana/deneme yuvaları, tutarlılık, adaylar ve uygun oldukları yuvalar (yazma yok)."""
    from app.feedback.model_activation import overview

    return overview()


@models_router.post("/activate-main", dependencies=[_human])
def models_activate_main(req: ActivateRequest) -> dict[str, Any]:
    """Adayı ana sohbet modeli yap — yalnız 'Kabul' kararı + digest eşleşmesi."""
    from app.feedback.model_activation import activate_main

    return _act(activate_main, req.tag, req.reason)


@models_router.post("/activate-trial", dependencies=[_human])
def models_activate_trial(req: ActivateRequest) -> dict[str, Any]:
    """Adayı DENEME sohbetine al — 'Kabul' ya da 'Yetersiz kanıt'."""
    from app.feedback.model_activation import activate_trial

    return _act(activate_trial, req.tag)


@models_router.post("/clear-trial", dependencies=[_human])
def models_clear_trial() -> dict[str, Any]:
    from app.feedback.model_activation import clear_trial

    return _act(clear_trial)


@models_router.post("/rollback", dependencies=[_human])
def models_rollback(req: ReasonRequest) -> dict[str, Any]:
    """Önceki ana modele dön (Ollama'da var + digest eşleşmesi doğrulanır)."""
    from app.feedback.model_activation import rollback_main

    return _act(rollback_main, req.reason)


@models_router.post("/repair-registry", dependencies=[_human])
def models_repair_registry() -> dict[str, Any]:
    from app.feedback.model_activation import repair_registry

    return _act(repair_registry)


@learn_router.get("/legacy-echo")
def learn_legacy_echo(limit: int = 100) -> dict[str, Any]:
    """Eski Echo kayıtları — SALT OKUNUR (Faz 1'de taşınmaz)."""
    from app.feedback.store import FeedbackStore

    return {"items": FeedbackStore().list(limit=max(1, min(limit, 500))), "read_only": True}
