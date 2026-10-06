"""A · Buluttan ikinci görüş — insan tıklamasıyla, tur başına, varsayılan KAPALI.

Akış: ``preview`` (gönderilecek metnin TAMAMI + özet + sağlayıcı/şart/kota/iptal bilgisi) →
``start`` (insan, önizlenen özetle) → arka planda sağlayıcı çağrısı → ``get`` / ``cancel``.

Her kayıt ÜÇ AYRI alan taşır ve hiçbiri diğerinin yerine geçmez:
- ``yerel``              : yerel cevap (metin + özet + model kimliği) — değişmez kopya,
- ``bulut``              : bulut önerisi (metin, sağlayıcı, model, durum),
- ``bagimsiz_dogrulama`` : bulut metnine YEREL deterministik kontroller (Kural 1, hesap, atıf
                           kimliği, kaynak benzerliği) — bulut cevabı doğru kabul EDİLMEZ.

Bulut cevabı öğrenme adayı OLMAZ, doğrulama sayılmaz; ``app.feedback`` / eğitim kodu bu modülü
kullanmaz (statik test). Gönderilmeyenler: kaynak parçaları/makale metinleri, sohbet geçmişinin
geri kalanı, ``.env``, ``data/``, ``storage/``.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.cloud.policy import policy_for
from app.cloud.providers import Provider, ProviderError, make_provider
from app.config import get_settings

NOTE = (
    "İkinci görüş — doğrulama DEĞİLDİR. Bulut cevabı otomatik doğru kabul edilmez, öğrenme "
    "adayı olmaz ve eğitime girmez."
)
CANCEL_NOTE = (
    "İptal ya da zaman aşımında alt süreç sonlandırılır; yarım cevap saklanmaz, kayıt "
    "'iptal edildi' olarak kalır. Gönderilmiş istek sağlayıcı tarafında yine kotaya sayılabilir."
)
NOT_SENT = [
    "kaynak parçaları ve makale metinleri",
    "sohbetin diğer turları",
    ".env, data/, storage/, model ağırlıkları",
]

# Testlerde değiştirilebilir (ağsız sahte sağlayıcı).
provider_factory: Callable[[str], Provider] = make_provider

_LIVE: dict[str, Provider] = {}
_LOCK = threading.Lock()


class CloudError(ValueError):
    """Kullanıcıya gösterilecek ikinci görüş hatası."""


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _dir() -> Path:
    d = get_settings().root / "storage" / "cloud" / "second_opinions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(rec_id: str) -> Path:
    return _dir() / f"{Path(rec_id).name}.json"


def _read(rec_id: str) -> dict[str, Any] | None:
    try:
        return json.loads(_path(rec_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(rec: dict[str, Any]) -> None:
    tmp = _path(rec["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_path(rec["id"]))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records() -> list[dict[str, Any]]:
    out = []
    for p in sorted(_dir().glob("so_*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def used_today() -> int:
    today = datetime.now(UTC).date().isoformat()
    return sum(1 for r in _records() if str(r.get("created_at", "")).startswith(today))


def status() -> dict[str, Any]:
    """Özellik açık mı, neden kapalı? (salt-okuma)"""
    s = get_settings()
    blockers: list[str] = []
    if not s.cloud_second_opinion:
        blockers.append("Özellik kapalı (HEKTOR_CLOUD_SECOND_OPINION=1 değil).")
    if not s.cloud_provider:
        blockers.append("Sağlayıcı seçilmedi (HEKTOR_CLOUD_PROVIDER).")
    if not s.cloud_terms_ack:
        blockers.append(
            "Kullanım şartlarının okunduğu tarih girilmedi (HEKTOR_CLOUD_TERMS_ACK=YYYY-MM-DD)."
        )
    pol = policy_for(s.cloud_provider) if s.cloud_provider else None
    if s.cloud_provider and pol is None:
        blockers.append(f"Bilinmeyen sağlayıcı: {s.cloud_provider}")
    if pol is not None and pol.second_opinion != "izinli_insan_tiklamasi":
        blockers.append(f"{pol.provider}: {pol.reason}")
    avail = (False, "")
    if not blockers:
        try:
            avail = provider_factory(s.cloud_provider).available()
        except ProviderError as exc:
            avail = (False, str(exc))
        if not avail[0]:
            blockers.append(avail[1])
    used = used_today()
    if used >= s.cloud_daily_max:
        blockers.append(f"Günlük üst sınır doldu ({used}/{s.cloud_daily_max}).")
    return {
        "enabled": not blockers,
        "blockers": blockers,
        "provider": s.cloud_provider,
        "provider_detail": avail[1],
        "policy": {
            "reason": pol.reason,
            "sources": list(pol.sources),
            "training_use": pol.training_use,
        }
        if pol
        else None,
        "terms_ack": s.cloud_terms_ack,
        "quota": {"used_today": used, "daily_max": s.cloud_daily_max},
        "note": NOTE,
    }


def _turn(turn_id: str) -> dict[str, Any]:
    from app.feedback.chat_store import ChatStore

    t = ChatStore().get_turn(turn_id)
    if t is None:
        raise CloudError(f"Tur bulunamadı: {turn_id}")
    if t.get("status") != "answered" or not t.get("answer"):
        raise CloudError("Yalnız cevaplanmış bir tur için ikinci görüş istenebilir.")
    return t


def build_payload(turn: dict[str, Any]) -> str:
    """Gönderilecek metnin TAMAMI (önizlemede aynen gösterilir)."""
    checks = [
        f"- {c.get('kind')}: {c.get('status') or c.get('label') or ''} {c.get('detail') or ''}"
        for c in (turn.get("checks") or [])
        if c.get("kind") != "gecmis"
    ]
    return (
        "Bir yerel trading-araştırma asistanının cevabını eleştirel olarak incele. Yatırım "
        "tavsiyesi verme. Hatalı, eksik ya da kanıtsız iddiaları, maliyet/look-ahead/örneklem "
        "varsayımlarını madde madde belirt; emin olmadığını söyle. Doğru bulduklarını da "
        "gerekçesiyle yaz.\n\n"
        f"SORU:\n{turn.get('question', '')}\n\n"
        f"YEREL CEVAP ({turn.get('model_tag', '')}):\n{turn.get('answer', '')}\n\n"
        "YEREL DETERMİNİSTİK KONTROLLER:\n" + ("\n".join(checks) or "- (yok)")
    )


def preview(turn_id: str) -> dict[str, Any]:
    """Tek ekran: ne gidecek, kime, kota/maliyet, iptal davranışı. Hiçbir şey GÖNDERMEZ."""
    s = get_settings()
    turn = _turn(turn_id)
    payload = build_payload(turn)
    st = status()
    blockers = list(st["blockers"])
    if len(payload) > s.cloud_max_chars:
        blockers.append(f"Gönderilecek metin {len(payload)} kr > üst sınır {s.cloud_max_chars}.")
    return {
        **st,
        "enabled": not blockers,
        "blockers": blockers,
        "turn_id": turn_id,
        "payload": payload,
        "payload_sha256": _sha(payload),
        "payload_chars": len(payload),
        "est_input_tokens": int(len(payload) / 3.5) + 1,
        "not_sent": NOT_SENT,
        "cost_note": (
            "Abonelikli CLI: istek, aboneliğinizin kullanım kotasından düşer; kalan kota bu "
            "araçtan okunamaz. Ücretli API bu dalda yok."
            if s.cloud_provider == "claude_code_cli"
            else "Sahte sağlayıcı: maliyet yok, ağ çağrısı yok."
            if s.cloud_provider == "fake"
            else "Sağlayıcı seçilmedi."
        ),
        "cancel_note": CANCEL_NOTE,
    }


def start(turn_id: str, payload_sha256: str, *, wait: bool = False) -> dict[str, Any]:
    """İnsan onayıyla gönder. Önizlenen özet tutmazsa (metin değiştiyse) GÖNDERMEZ."""
    pv = preview(turn_id)
    if not pv["enabled"]:
        raise CloudError("İkinci görüş kapalı: " + " | ".join(pv["blockers"]))
    if payload_sha256 != pv["payload_sha256"]:
        raise CloudError("Gönderilecek metin önizlemeden farklı — önizlemeyi yenileyin.")
    turn = _turn(turn_id)
    s = get_settings()
    rec_id = "so_" + secrets.token_hex(6)
    rec: dict[str, Any] = {
        "id": rec_id,
        "turn_id": turn_id,
        "created_at": utcnow(),
        "provider": s.cloud_provider,
        "terms_ack": s.cloud_terms_ack,
        "payload_sha256": pv["payload_sha256"],
        "payload": pv["payload"],
        "yerel": {
            "answer": turn.get("answer", ""),
            "answer_sha256": _sha(turn.get("answer", "")),
            "model_tag": turn.get("model_tag", ""),
            "model_digest": turn.get("model_digest", ""),
        },
        "bulut": {"status": "pending", "text": "", "model": "", "error": "", "finished_at": ""},
        "bagimsiz_dogrulama": None,
        "training_use": False,
        "note": NOTE,
    }
    _write(rec)
    provider = provider_factory(s.cloud_provider)
    with _LOCK:
        _LIVE[rec_id] = provider
    th = threading.Thread(target=_run, args=(rec_id, provider, turn), daemon=True)
    th.start()
    if wait:
        th.join(timeout=float(s.cloud_timeout_s) + 5)
    return get(rec_id)


def _run(rec_id: str, provider: Provider, turn: dict[str, Any]) -> None:
    s = get_settings()
    rec = _read(rec_id) or {}
    try:
        reply = provider.ask(rec["payload"], timeout_s=float(s.cloud_timeout_s))
        cur = _read(rec_id) or rec
        if cur["bulut"]["status"] == "cancelled":
            return  # iptal sonrası gelen cevap saklanmaz
        cur["bulut"] = {
            "status": "done",
            "text": reply.text,
            "model": reply.model,
            "error": "",
            "finished_at": utcnow(),
        }
        cur["bagimsiz_dogrulama"] = _independent_check(reply.text, turn)
        _write(cur)
    except ProviderError as exc:
        cur = _read(rec_id) or rec
        if cur["bulut"]["status"] != "cancelled":
            cur["bulut"].update(status="failed", error=str(exc)[:500], finished_at=utcnow())
            _write(cur)
    finally:
        with _LOCK:
            _LIVE.pop(rec_id, None)


def _independent_check(text: str, turn: dict[str, Any]) -> dict[str, Any]:
    """Bulut metnine YEREL deterministik kontroller (Kural 1, hesap, atıf, kaynak benzerliği)."""
    from app.feedback.verify import verify_target

    try:
        v = verify_target(
            text,
            question=str(turn.get("question", "")),
            sources=list(turn.get("sources") or []),
            domain="general",
        )
    except Exception as exc:  # doğrulama çalışmazsa bulut cevabı yine doğru SAYILMAZ
        return {"ran": False, "error": str(exc)[:300], "verdict": "dogrulanamadi"}
    return {
        "ran": True,
        "checks": v.get("checks", []),
        "coverage": v.get("coverage", {}),
        "verdict": "dogrulama_degil",
        "note": "Kaynak benzerliği doğrulama değildir; karar insanındır.",
    }


def cancel(rec_id: str) -> dict[str, Any]:
    rec = _read(rec_id)
    if rec is None:
        raise CloudError(f"Kayıt yok: {rec_id}")
    if rec["bulut"]["status"] != "pending":
        return rec
    rec["bulut"].update(status="cancelled", finished_at=utcnow(), text="")
    _write(rec)
    with _LOCK:
        prov = _LIVE.get(rec_id)
    if prov is not None:
        prov.cancel()
    return get(rec_id)


def get(rec_id: str) -> dict[str, Any]:
    rec = _read(rec_id)
    if rec is None:
        raise CloudError(f"Kayıt yok: {rec_id}")
    return rec


def list_for_turn(turn_id: str) -> list[dict[str, Any]]:
    return [r for r in _records() if r.get("turn_id") == turn_id]
