"""chat_service.py — tek sohbet ekranının sunucu tarafı (Ollama + RAG, geçmişli).

Bir tur:
1. Tur 'pending' olarak AYRILIR (aynı istek kimliği → aynı tur; çift cevap üretilmez).
2. Kaynak koruması SUNUCUDA uygulanır (``resource_guard``); izin yoksa tur 'blocked'.
3. Kira alınır (eğitim başlatmayla yarış), son N tur karakter bütçesiyle geçmiş olarak eklenir.
4. Cevap ``RagAnswerer`` ile üretilir; model etiketi istek başında SABİTLENİR, digest cevaptan
   önce (``/api/tags``) ve sonra (``/api/ps``) okunur → model değişimi sırasında her tur
   gerçekten kullandığı kimlikle kaydedilir. Eğitim yoksa bellek ayak izi ölçülür.

Sohbet içeriği RAG indeksine YAZILMAZ; eğitim başlatılmaz.
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.config import get_settings
from app.feedback.chat_store import ChatStore, utcnow


class ChatBusy(RuntimeError):
    """Aynı istek hâlâ işleniyor (409)."""


def select_history(
    turns: list[dict[str, Any]], before_index: int, n_turns: int, budget: int
) -> tuple[list[tuple[int, str, str]], list[str], bool]:
    """Son ``n_turns`` cevaplanmış turu karakter bütçesiyle seç → (geçmiş, tur id'leri, kesildi)."""
    eligible = [
        t
        for t in turns
        if t["turn_index"] < before_index and t["status"] == "answered" and t["llm_used"]
    ]
    picked: list[tuple[int, str, str]] = []
    ids: list[str] = []
    used = 0
    truncated = False
    for t in reversed(eligible[-max(0, n_turns) :] if n_turns > 0 else []):
        q = t["question"]
        a = t["raw_answer"] or t["answer"]
        room = budget - used - len(q)
        if room <= 80:
            truncated = True
            break
        if len(a) > room:
            a = a[: room - 20].rstrip() + " …[kısaltıldı]"
            truncated = True
        picked.append((t["turn_index"], q, a))
        ids.append(t["turn_id"])
        used += len(q) + len(a)
    picked.reverse()
    ids.reverse()
    return picked, ids, truncated


def _source_dict(c: Any) -> dict[str, Any]:
    return {
        "paper_id": c.paper_id,
        "chunk_id": c.chunk_id,
        "title": c.title,
        "page": c.page_number,
        "section": c.section_name,
        "distance": c.distance,
        "text": c.text,
    }


def send(
    conversation_id: str,
    question: str,
    client_request_id: str,
    *,
    top_k: int | None = None,
    retriever: Any = None,
    llm: Any = None,
    transport: Any = None,
    store: ChatStore | None = None,
) -> tuple[dict[str, Any], bool]:
    """Bir soru gönder. (tur, yeniden_oynatma_mı) döndürür."""
    from app.feedback.model_identity import (
        match_entry,
        model_origin,
        ollama_ps,
        ollama_tags,
        record_footprint,
    )
    from app.feedback.resource_guard import (
        acquire_chat_lease,
        check_chat_resources,
        release_chat_lease,
        training_activity,
    )

    store = store or ChatStore()
    if store.get_conversation(conversation_id) is None:
        raise KeyError(f"Konuşma bulunamadı: {conversation_id}")
    turn, is_new = store.reserve_turn(conversation_id, client_request_id, question)
    if not is_new:
        if turn["status"] == "pending":
            raise ChatBusy("Bu istek hâlâ işleniyor.")
        return turn, True
    store.touch_conversation(conversation_id, title_if_empty=question)

    s = get_settings()
    tag = s.effective_chat_model  # istek başında sabitlenir
    turn_id = turn["turn_id"]

    def _finish(**fields: Any) -> dict[str, Any]:
        updated = store.update_turn(turn_id, model_tag=tag, **fields)
        assert updated is not None
        return updated

    try:
        decision = check_chat_resources(transport)
        if not decision.allowed:
            return _finish(status="blocked", status_detail=decision.reason), False
        lease, why = acquire_chat_lease(turn_id)
    except Exception as exc:  # tur 'pending'de asılı kalmasın
        return _finish(status="error", status_detail=f"{type(exc).__name__}: {exc}"), False
    if lease is None:
        return _finish(status="blocked", status_detail=why), False
    try:
        history, history_ids, truncated = select_history(
            store.list_turns(conversation_id),
            turn["turn_index"],
            s.chat_history_turns,
            s.chat_history_char_budget,
        )
        retrieval_query = question if not history else f"{question}\n{history[-1][1]}"
        before = match_entry(ollama_tags(transport), tag)
        from app.brain.local_llm import LocalLLM
        from app.brain.rag_answerer import RagAnswerer

        model = llm or LocalLLM(model=tag, transport=transport)
        ans = RagAnswerer(retriever=retriever, llm=model).answer(
            question, top_k=top_k, history=history, retrieval_query=retrieval_query
        )
        entry = match_entry(ollama_ps(transport), tag) if ans.llm_used else None
        digest_before = str((before or {}).get("digest") or "")
        digest_after = str((entry or {}).get("digest") or "")
        digest = digest_after or digest_before
        note = ""
        if digest_before and digest_after and digest_before != digest_after:
            note = "Etiketin digest'i cevap sırasında değişti; kaydedilen digest cevap sonrası."
        elif not digest:
            note = "Digest okunamadı (Ollama /api/tags ve /api/ps yanıt vermedi)."
        footprint = None
        if entry is not None:
            act = training_activity()
            if not act["active"] and not act["starting"]:
                footprint = record_footprint(tag, entry, measured_at=utcnow())
        checks: list[dict[str, Any]] = [
            {
                "kind": "gecmis",
                "status": "aktarildi" if history else "yok",
                "scope": f"{len(history)} tur",
                "turn_ids": history_ids,
                "turn_indexes": [h[0] for h in history],
                "truncated": truncated,
                "detail": "Önceki model cevapları bağlamdır; doğrulanmış kaynak sayılmaz.",
            }
        ]
        cc = ans.citation_check
        if cc is not None and cc.n_unique:
            checks.append(
                {
                    "kind": "atif_kimligi",
                    "status": "kaldi" if cc.unsupported else "gecti",
                    "scope": f"{cc.n_unique - len(cc.unsupported)}/{cc.n_unique} kimlik",
                    "detail": "Kimliğin getirilen parçalarda olması iddianın desteklendiğini "
                    "göstermez.",
                }
            )
        prompt_sha = (
            hashlib.sha256(f"{ans.system_prompt}\x00{ans.user_prompt}".encode()).hexdigest()
            if ans.user_prompt
            else ""
        )
        return (
            _finish(
                status="answered" if ans.llm_used else "no_llm",
                status_detail=""
                if ans.llm_used
                else "Model çağrılmadı (kaynak yok/çekimser/LLM kapalı).",
                retrieval_query=retrieval_query,
                system_prompt=ans.system_prompt,
                user_prompt=ans.user_prompt,
                prompt_sha256=prompt_sha,
                history_turn_ids=history_ids if ans.llm_used else [],
                model_digest=digest,
                model_info={
                    "digest_before": digest_before,
                    "digest_after": digest_after,
                    "digest_note": note,
                    "origin": model_origin(tag),
                    "footprint": footprint,
                    "setting_source": "HEKTOR_CHAT_MODEL"
                    if s.chat_model.strip()
                    else "HEKTOR_LLM_MODEL (chat_model boş)",
                },
                sources=[_source_dict(c) for c in ans.sources],
                checks=checks,
                answer=ans.answer,
                raw_answer=ans.raw_answer,
                llm_used=ans.llm_used,
            ),
            False,
        )
    except Exception as exc:
        return _finish(status="error", status_detail=f"{type(exc).__name__}: {exc}"), False
    finally:
        release_chat_lease(lease)


def measure(transport: Any = None, llm: Any = None) -> dict[str, Any]:
    """Kontrollü ilk ölçüm: eğitim YOKKEN modeli kısa üretimle yükle, ayak izini kaydet."""
    from app.brain.local_llm import LLMUnavailable, LocalLLM
    from app.feedback.model_identity import match_entry, ollama_ps, record_footprint
    from app.feedback.resource_guard import (
        acquire_chat_lease,
        release_chat_lease,
        training_activity,
    )

    act = training_activity()
    if act["active"] or act["starting"]:
        raise RuntimeError("Eğitim sürüyor/başlatılıyor — ölçüm eğitim dışında yapılır.")
    tag = get_settings().effective_chat_model
    lease, why = acquire_chat_lease(f"measure-{hashlib.sha1(utcnow().encode()).hexdigest()[:8]}")
    if lease is None:
        raise RuntimeError(why)
    try:
        model = llm or LocalLLM(model=tag, transport=transport)
        try:
            model.generate("ölçüm", max_tokens=1, temperature=0.0, seed=42, timeout=600)
        except LLMUnavailable as exc:
            if "num_predict" not in str(exc):
                raise
        entry = match_entry(ollama_ps(transport), tag)
        if entry is None:
            raise RuntimeError("Model yüklendikten sonra /api/ps'te görünmedi; ölçüm alınamadı.")
        return {"tag": tag, "footprint": record_footprint(tag, entry, measured_at=utcnow())}
    finally:
        release_chat_lease(lease)
