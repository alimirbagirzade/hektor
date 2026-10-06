"""model_activation.py — ana sohbet modeli ile DENEME modelinin ayrı yuvaları ve etkinleştirme.

Kurallar (Faz 2A):
- İki yuva: ``main`` (ana sohbet) ve ``trial`` (belirgin etiketli deneme sohbeti). Ayrıdırlar;
  deneme yuvasındaki model ana sohbete hiçbir koşulda cevap vermez.
- Etkinleştirme kayıt dosyası yoksa ana model, ayardaki model (``HEKTOR_CHAT_MODEL`` /
  ``HEKTOR_LLM_MODEL``) olarak bir BAŞLANGIÇ KAYDIDIR: ``evaluation = baseline_unevaluated``.
  Geriye dönük olarak "başarıyla değerlendirildi" SAYILMAZ. GET/okuma bu kaydı diske yazmaz.
- Ana yuvaya yalnız karşılaştırma kararı ``kabul`` olan aday geçer; ``yetersiz_kanit`` yalnız
  deneme yuvasına; ``ret`` / ``kritik_ret`` hiçbir yuvaya (kritik ret üretime geçirilemez).
- Karar, adayın KARŞILAŞTIRILAN digest'ine bağlıdır: Ollama'daki etiket şu an aynı digest'i
  göstermiyorsa etkinleştirme reddedilir.
- Geri dönüş: önceki ana modelin Ollama'da hâlâ bulunduğu VE digest'inin kayıttakiyle eşleştiği
  doğrulanır; değilse reddedilir.
- Etkinleştirme kaydı ile adapter kayıt defterindeki ``production`` durumu TUTARLI güncellenir:
  önce niyet günlüğü (``model_activation.pending.json``) yazılır, sonra kayıt defteri, sonra
  durum dosyası (atomik değiştirme), en son günlük silinir. Yarıda kalan işlem bir sonraki
  okumada günlükten TAMAMLANIR (ileri sarma) → iki farklı "aktif model" bilgisi kalmaz.
- Devam eden cevap, BAŞLADIĞI andaki model kimliğiyle tamamlanır (``chat_service`` etiketi
  istek başında çözer); etkinleştirme yalnız sonraki istekleri etkiler.
- Hiçbir fonksiyon kendiliğinden çağrılmaz; tümü insan eylemiyle (``require_human``) tetiklenir.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.evals.candidate_decisions import (
    DECISION_TR,
    MAIN_OK,
    TRIAL_OK,
    latest_decision,
    norm_digest,
    norm_tag,
)
from app.feedback.chat_store import utcnow

SLOTS = ("main", "trial")
BASELINE_EVAL = "baseline_unevaluated"
BASELINE_NOTE = (
    "Başlangıç kaydı: bu model etkinleştirme sistemi kurulmadan önce aktifti. Aday "
    "karşılaştırmasından geçmedi; geriye dönük olarak 'başarıyla değerlendirildi' sayılmaz."
)
TRIAL_BANNER = (
    "DENEME SOHBETİ — bu model ana model olarak kabul EDİLMEDİ (kanıt yetersiz ya da henüz "
    "etkinleştirilmedi). Cevaplar ana modelin kalitesini göstermez."
)


class ActivationError(ValueError):
    """Kullanıcıya gösterilecek (422/409) etkinleştirme kuralı ihlali."""


def _default_tags() -> list[dict[str, Any]] | None:
    from app.feedback.model_identity import ollama_tags

    return ollama_tags()


# Testlerde değiştirilebilir: Ollama /api/tags listesi (None = ulaşılamadı).
tags_provider: Callable[[], list[dict[str, Any]] | None] = _default_tags


# ── yollar ───────────────────────────────────────────────────────────────────


def _storage(root: Path | None = None) -> Path:
    return (root or get_settings().root) / "storage"


def state_path(root: Path | None = None) -> Path:
    return _storage(root) / "model_activation.json"


def journal_path(root: Path | None = None) -> Path:
    return _storage(root) / "model_activation.pending.json"


def log_path(root: Path | None = None) -> Path:
    return _storage(root) / "model_activation_log.jsonl"


def _lock_path(root: Path | None = None) -> Path:
    return _storage(root) / "model_activation.lock"


def registry_path(root: Path | None = None) -> Path:
    return (root or get_settings().root) / "registry" / "adapters" / "registry.jsonl"


# ── düşük düzey yardımcılar ──────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_log(event: dict[str, Any], root: Path | None = None) -> None:
    p = log_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": utcnow(), **event}, ensure_ascii=False, sort_keys=True) + "\n")


@contextlib.contextmanager
def _op_lock(root: Path | None = None) -> Iterator[None]:
    """Etkinleştirme işlemleri tek seferde bir tane (kısa ömürlü, pid'li kilit)."""
    from app.training.resource_lock import owner_alive

    p = _lock_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "at": time.time()}).encode())
            os.close(fd)
            break
        except FileExistsError:
            info = _read_json(p) or {}
            if info.get("pid") and not owner_alive(info):
                with contextlib.suppress(OSError):
                    p.unlink()
                continue
            raise ActivationError(
                "Başka bir etkinleştirme işlemi sürüyor; birazdan tekrar deneyin."
            ) from None
    else:
        raise ActivationError("Etkinleştirme kilidi alınamadı.")
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            p.unlink()


# ── durum ────────────────────────────────────────────────────────────────────


def _setting_tag() -> tuple[str, str]:
    s = get_settings()
    src = "HEKTOR_CHAT_MODEL" if s.chat_model.strip() else "HEKTOR_LLM_MODEL (chat_model boş)"
    return s.effective_chat_model, src


def baseline_entry() -> dict[str, Any]:
    tag, src = _setting_tag()
    return {
        "tag": norm_tag(tag),
        "digest": "",
        "source": "baseline",
        "setting_source": src,
        "evaluation": BASELINE_EVAL,
        "decision": "",
        "adapter_id": "",
        "note": BASELINE_NOTE,
    }


def load_state(root: Path | None = None) -> dict[str, Any]:
    """Durum (yarım işlem varsa önce tamamlanır). Dosya yoksa SANAL başlangıç kaydı (yazılmaz)."""
    recovery_error = ""
    if journal_path(root).exists():
        try:
            recover(root)
        except Exception as exc:  # sohbet kilitlenmesin: eski (tutarlı) durum kullanılır
            recovery_error = f"Yarım etkinleştirme tamamlanamadı: {exc}"
    st = _read_json(state_path(root))
    if st is None:
        st = {"version": 0, "main": baseline_entry(), "trial": None, "virtual": True}
    else:
        st.setdefault("trial", None)
        st["virtual"] = False
    if recovery_error:
        st["recovery_error"] = recovery_error
    return st


def resolve_chat_tag(slot: str = "main", root: Path | None = None) -> str:
    """Yuvanın şu anki etiketi. Deneme yuvası boşsa ``""`` (sohbet cevaplanmaz)."""
    if slot not in SLOTS:
        raise ActivationError(f"Geçersiz yuva: {slot}")
    st = load_state(root)
    entry = st.get(slot)
    if not entry:
        return ""
    return str(entry.get("tag") or "")


def describe_slot(slot: str = "main", root: Path | None = None) -> dict[str, Any]:
    st = load_state(root)
    entry = st.get(slot) or {}
    return {
        "slot": slot,
        "tag": entry.get("tag", ""),
        "source": entry.get("source", ""),
        "evaluation": entry.get("evaluation", ""),
        "decision": entry.get("decision", ""),
        "decision_label": DECISION_TR.get(str(entry.get("decision") or ""), ""),
        "note": entry.get("note", ""),
        "banner": TRIAL_BANNER if slot == "trial" else "",
        "record_version": st.get("version", 0),
        "virtual": st.get("virtual", False),
    }


# ── Ollama doğrulaması ───────────────────────────────────────────────────────


def _ollama_digest(tag: str) -> str | None:
    """Etiketin Ollama'daki digest'i; Ollama'ya ulaşılamazsa None, etiket yoksa ""."""
    from app.feedback.model_identity import match_entry

    tags = tags_provider()
    if tags is None:
        return None
    entry = match_entry(tags, tag)
    return norm_digest(str((entry or {}).get("digest") or "")) if entry else ""


def _require_present(tag: str, digest: str, what: str) -> str:
    cur = _ollama_digest(tag)
    if cur is None:
        raise ActivationError("Ollama'ya ulaşılamadı — model varlığı/digest'i doğrulanamadı.")
    if not cur:
        raise ActivationError(f"{what} '{tag}' Ollama'da bulunamadı.")
    if digest and cur != norm_digest(digest):
        raise ActivationError(
            f"{what} '{tag}' Ollama'da başka ağırlıklarla duruyor (digest {cur[:12]} ≠ kayıttaki "
            f"{norm_digest(digest)[:12]}). Karşılaştırılan model bu değil; etkinleştirilmedi."
        )
    return cur


# ── kayıt defteri (adapter production) ───────────────────────────────────────


def _registry(root: Path | None = None) -> Any:
    from app.lora.adapter_registry import AdapterRegistry

    return AdapterRegistry(registry_path(root))


def registry_production_id(root: Path | None = None) -> str:
    rec = _registry(root).get_production()
    return rec.adapter_id if rec else ""


def _sync_registry(adapter_id: str, root: Path | None = None) -> None:
    """Kayıt defterindeki ``production``ı etkin ana modele eşitle (idempotent)."""
    reg = _registry(root)
    if adapter_id:
        if registry_production_id(root) == adapter_id:
            return
        if not reg.promote(adapter_id, user_approved=True):
            raise ActivationError(
                f"Kayıt defterinde {adapter_id} production'a alınamadı (durumu eval_passed/"
                "approved değil ya da kayıt yok)."
            )
    else:
        reg.archive_production()


def consistency(root: Path | None = None) -> dict[str, Any]:
    """Etkinleştirme kaydı ↔ kayıt defteri production tutarlılığı (salt okuma)."""
    st = load_state(root)
    want = str((st.get("main") or {}).get("adapter_id") or "")
    have = registry_production_id(root)
    if st.get("virtual"):
        return {"consistent": True, "registry_production": have, "activation_adapter": ""}
    return {
        "consistent": want == have,
        "registry_production": have,
        "activation_adapter": want,
    }


# ── iki aşamalı güncelleme ───────────────────────────────────────────────────


def _commit(new_state: dict[str, Any], event: dict[str, Any], root: Path | None = None) -> None:
    journal = {
        "op_id": "act_" + secrets.token_hex(6),
        "to_state": new_state,
        "event": event,
        "started_at": utcnow(),
    }
    _write_atomic(journal_path(root), journal)
    _apply_journal(journal, root)


def _apply_journal(journal: dict[str, Any], root: Path | None = None) -> None:
    to_state = journal["to_state"]
    _sync_registry(str((to_state.get("main") or {}).get("adapter_id") or ""), root)
    cur = _read_json(state_path(root)) or {}
    if int(cur.get("version", 0)) < int(to_state.get("version", 0)):
        _write_atomic(state_path(root), {k: v for k, v in to_state.items() if k != "virtual"})
        _append_log({**journal["event"], "op_id": journal["op_id"]}, root)
    with contextlib.suppress(OSError):
        journal_path(root).unlink()


def recover(root: Path | None = None) -> bool:
    """Yarıda kalmış işlemi günlükten tamamla. True → bir işlem tamamlandı."""
    journal = _read_json(journal_path(root))
    if journal is None:
        with contextlib.suppress(OSError):
            journal_path(root).unlink()
        return False
    _apply_journal(journal, root)
    return True


def _next_state(st: dict[str, Any]) -> dict[str, Any]:
    nxt = {k: v for k, v in st.items() if k != "virtual"}
    nxt["version"] = int(st.get("version", 0)) + 1
    nxt["updated_at"] = utcnow()
    return nxt


def pilot_block(tag: str, adapter_id: str = "", root: Path | None = None) -> str:
    """Model reçeteye SINIRLI risk kabulüyle eğitilmiş (pilot) bir adapter'dan mı? (boş = hayır)

    Kademe 2 (2026-10-06) E-1, kullanıcı kararı: pilot hiçbir yoldan ana modele geçmez. Adapter
    adı karar/kayıt defterinden ve Modelfile kökeninden alınır; kolay akış reçetesi yalnız-reçete
    kapsamlı bir risk kabulüyle kayıtlıysa engel döner.
    """
    from app.feedback.model_identity import model_origin
    from app.training.candidate_checks import recipe_for_adapter
    from app.training.easy_train import recipe_has_limited_acceptance

    names: set[str] = set()
    if adapter_id:
        rec = _registry(root).get(adapter_id)
        if rec is not None and rec.adapter_name:
            names.add(str(rec.adapter_name))
    origin = model_origin(norm_tag(tag), root)
    if origin and origin.get("adapter"):
        names.add(str(origin["adapter"]))
    for name in sorted(names):
        recipe = recipe_for_adapter(name) or {}
        if recipe_has_limited_acceptance(str(recipe.get("recipe_sha") or "")):
            return (
                f"'{tag}' adapter'ı '{name}' YALNIZ pilot için sınırlı risk kabulüyle eğitildi — "
                "ana modele geçemez (deneme sohbetinde kullanılabilir)."
            )
    return ""


def _check_registry_target(adapter_id: str, root: Path | None = None) -> None:
    """Günlük yazılmadan ÖNCE: kayıt defteri bu adapter'ı production'a alabilir mi?"""
    if not adapter_id:
        return
    from app.lora.adapter_registry import AdapterStatus

    rec = _registry(root).get(adapter_id)
    if rec is None:
        raise ActivationError(f"Karardaki adapter {adapter_id} kayıt defterinde yok.")
    if rec.status is AdapterStatus.REJECTED:
        raise ActivationError(f"Adapter {adapter_id} kayıt defterinde reddedilmiş.")
    if rec.status not in (
        AdapterStatus.EVAL_PASSED,
        AdapterStatus.APPROVED,
        AdapterStatus.PRODUCTION,
    ):
        raise ActivationError(
            f"Adapter {adapter_id} kayıt defterinde '{rec.status.value}' — production'a alınamaz."
        )


def main_role_ok(dec: dict[str, Any]) -> bool:
    """Karar bağımsız final kanıtı mı? (karşılaştırmanın ETKİN rolü 'final')."""
    return str(dec.get("role") or "") == "final"


def _decision_for(tag: str, root: Path | None = None) -> tuple[dict[str, Any], str]:
    """Etiketin GÜNCEL digest'iyle eşleşen son karar → (karar, digest)."""
    cur = _ollama_digest(tag)
    if cur is None:
        raise ActivationError("Ollama'ya ulaşılamadı — aday digest'i doğrulanamadı.")
    if not cur:
        raise ActivationError(f"Aday '{tag}' Ollama'da bulunamadı.")
    from app.evals.candidate_decisions import list_decisions, norm_digest

    # Kademe 2 (2026-10-06) F4-2: kritik ret KALICIDIR — aynı digest için sonradan yazılan bir
    # karar onu geçersiz kılamaz (yeniden karşılaştırma ile "temize çıkarma" yok).
    if any(
        r["decision"] == "kritik_ret"
        and r["candidate_tag"] == norm_tag(tag)
        and r["candidate_digest"] == norm_digest(cur)
        for r in list_decisions(root)
    ):
        raise ActivationError(
            f"'{tag}' ({cur[:12]}) daha önce KRİTİK RET aldı; sonraki kararlar bunu kaldırmaz."
        )
    dec = latest_decision(tag, cur, root)
    if dec is None:
        any_dec = latest_decision(tag, "", root)
        if any_dec is not None:
            raise ActivationError(
                f"'{tag}' için karar başka bir digest'e ait ({any_dec['candidate_digest'][:12]}); "
                f"Ollama'daki model ({cur[:12]}) karşılaştırılmadı."
            )
        raise ActivationError(f"'{tag}' için karşılaştırma kararı yok — önce karşılaştırma.")
    return dec, cur


# ── eylemler (yalnız insan) ──────────────────────────────────────────────────


def activate_main(tag: str, reason: str, *, root: Path | None = None) -> dict[str, Any]:
    """Adayı ANA sohbet modeli yap. Yalnız ``kabul`` kararı + digest eşleşmesi."""
    reason = (reason or "").strip()
    if len(reason) < 10:
        raise ActivationError("Etkinleştirme gerekçesi en az 10 karakter olmalı.")
    tag = norm_tag(tag)
    with _op_lock(root):
        dec, digest = _decision_for(tag, root)
        if dec["decision"] == "kritik_ret":
            raise ActivationError("Kritik hatadan reddedilmiş aday üretime geçirilemez.")
        if dec["decision"] not in MAIN_OK:
            raise ActivationError(
                f"Karar '{DECISION_TR[dec['decision']]}' — ana model yalnız 'Kabul' kararıyla "
                "etkinleştirilebilir. 'Yetersiz kanıt' yalnız deneme sohbetinde kullanılabilir."
            )
        # Kademe 2 (2026-10-06) F4-5, kullanıcı kararı: ana model YALNIZ kullanılmamış gizli
        # final setiyle verilmiş 'Kabul' ile değişir. Geliştirme setindeki 'Kabul' (ya da
        # daha önce görülmüş final soruları → 'development'e düşmüş koşu) yalnız deneme
        # yuvasına yeter. İnsan onayı: web ucu require_human + gerekçe.
        pilot = pilot_block(tag, str(dec.get("adapter_id") or ""), root)
        if pilot:
            raise ActivationError(pilot)
        if not main_role_ok(dec):
            raise ActivationError(
                f"Karar '{dec.get('role') or '?'}' rollü karşılaştırmadan — ana model yalnız "
                "kullanılmamış gizli FİNAL setiyle verilmiş 'Kabul' ile değişir. Bu aday deneme "
                "sohbetinde kullanılabilir."
            )
        _check_registry_target(str(dec.get("adapter_id") or ""), root)
        st = load_state(root)
        if st.get("recovery_error"):
            raise ActivationError(st["recovery_error"])
        # F4-1: karar ŞU ANKİ ana modele karşı verilmiş olmalı (zayıf bir modele ya da artık
        # ana olmayan eski modele karşı alınmış "kabul" ana modelin yerine geçemez).
        cur_main = norm_tag(str((st.get("main") or {}).get("tag") or ""))
        vs = norm_tag(str(dec.get("active_tag") or ""))
        if vs != cur_main:
            raise ActivationError(
                f"Karar '{vs or '?'}' modeline karşı verilmiş; şu anki ana model '{cur_main}'. "
                "Aday güncel ana modelle yeniden karşılaştırılmalı."
            )
        prev = dict(st["main"])
        if prev.get("source") == "baseline" and not prev.get("digest"):
            prev["digest"] = _ollama_digest(str(prev.get("tag") or "")) or ""
        prev.pop("previous", None)
        nxt = _next_state(st)
        nxt["main"] = {
            "tag": tag,
            "digest": digest,
            "source": "activation",
            "evaluation": "compared",
            "decision": dec["decision"],
            "decision_id": dec["decision_id"],
            "comparison_id": dec.get("comparison_id", ""),
            "adapter_id": str(dec.get("adapter_id") or ""),
            "activated_at": utcnow(),
            "reason": reason,
            "previous": prev,
        }
        if (nxt.get("trial") or {}).get("tag") == tag:
            nxt["trial"] = None  # aynı model iki yuvada birden durmasın
        _commit(nxt, {"op": "activate_main", "tag": tag, "digest": digest, "reason": reason}, root)
    return load_state(root)


def activate_trial(tag: str, *, root: Path | None = None) -> dict[str, Any]:
    """Adayı DENEME yuvasına al: ``kabul`` ya da ``yetersiz_kanit`` + digest eşleşmesi."""
    tag = norm_tag(tag)
    with _op_lock(root):
        dec, digest = _decision_for(tag, root)
        if dec["decision"] not in TRIAL_OK:
            raise ActivationError(
                f"Karar '{DECISION_TR[dec['decision']]}' — reddedilmiş aday deneme sohbetinde de "
                "kullanılamaz."
            )
        st = load_state(root)
        if (st.get("main") or {}).get("tag") == tag:
            raise ActivationError("Bu model zaten ana model; deneme yuvasına alınmaz.")
        nxt = _next_state(st)
        nxt["trial"] = {
            "tag": tag,
            "digest": digest,
            "source": "trial",
            "decision": dec["decision"],
            "decision_id": dec["decision_id"],
            "comparison_id": dec.get("comparison_id", ""),
            "activated_at": utcnow(),
            "note": TRIAL_BANNER,
        }
        _commit(nxt, {"op": "activate_trial", "tag": tag, "digest": digest}, root)
    return load_state(root)


def clear_trial(*, root: Path | None = None) -> dict[str, Any]:
    with _op_lock(root):
        st = load_state(root)
        if not st.get("trial"):
            return st
        nxt = _next_state(st)
        nxt["trial"] = None
        _commit(nxt, {"op": "clear_trial"}, root)
    return load_state(root)


def _check_rollback_decision(prev: dict[str, Any], root: Path | None = None) -> None:
    """Kademe 2 (2026-10-06) F4-8: geri dönülecek modelin karar deposundaki GÜNCEL durumu.

    Önceki model etkinleştirildiğinde geçerli olan karar sonradan değişmiş olabilir:
    - aynı etiket+digest için KRİTİK RET (F4-2: kalıcı) → hiçbir yuvaya, geri dönüşle de dönmez;
    - karşılaştırmayla gelmiş (``decision_id`` taşıyan) model için en son karar artık
      'Kabul' değilse → geri dönüş eski, geçersizleşmiş kararla ana modeli kurmuş olurdu.
    Karşılaştırmasız başlangıç modeli (baseline) yalnız kritik ret ile engellenir.
    """
    from app.evals.candidate_decisions import list_decisions

    tag = norm_tag(str(prev.get("tag") or ""))
    digest = norm_digest(str(prev.get("digest") or ""))
    if any(
        r["decision"] == "kritik_ret"
        and r["candidate_tag"] == tag
        and r["candidate_digest"] == digest
        for r in list_decisions(root)
    ):
        raise ActivationError(
            f"Önceki model '{tag}' ({digest[:12]}) sonradan KRİTİK RET aldı; geri dönüş yapılmadı."
        )
    if not prev.get("decision_id"):
        return
    dec = latest_decision(tag, digest, root)
    # E-8: geri dönüş de ana model kuralına tabi (final rollü kabul).
    if dec is not None and dec["decision"] in MAIN_OK and not main_role_ok(dec):
        raise ActivationError(
            f"Önceki model '{tag}' için kabul FİNAL rollü değil — ana model yalnız gizli final "
            "setiyle verilmiş 'Kabul' ile kurulabilir; geri dönüş yapılmadı."
        )
    if dec is None or dec["decision"] not in MAIN_OK:
        now = DECISION_TR.get(dec["decision"], dec["decision"]) if dec else "karar yok"
        raise ActivationError(
            f"Önceki model '{tag}' için güncel karar '{now}' — ana model yalnız 'Kabul' kararıyla "
            "kurulabilir; geri dönüş yapılmadı."
        )


def rollback_main(reason: str, *, root: Path | None = None) -> dict[str, Any]:
    """Önceki ana modele dön: Ollama'da var mı + digest eşleşiyor mu doğrulanır."""
    reason = (reason or "").strip()
    if len(reason) < 10:
        raise ActivationError("Geri dönüş gerekçesi en az 10 karakter olmalı.")
    with _op_lock(root):
        st = load_state(root)
        prev = (st.get("main") or {}).get("previous")
        if not prev:
            raise ActivationError("Geri dönülecek önceki ana model kaydı yok.")
        tag = str(prev.get("tag") or "")
        digest = str(prev.get("digest") or "")
        if not digest:
            raise ActivationError(
                f"Önceki model '{tag}' için kayıtlı digest yok — aynı model olduğu doğrulanamaz; "
                "geri dönüş yapılmadı."
            )
        _require_present(tag, digest, "Önceki model")
        _check_rollback_decision(prev, root)
        pilot = pilot_block(tag, str(prev.get("adapter_id") or ""), root)
        if pilot:
            raise ActivationError(pilot)
        _check_registry_target(str(prev.get("adapter_id") or ""), root)
        cur = dict(st["main"])
        cur.pop("previous", None)
        nxt = _next_state(st)
        restored = {k: v for k, v in prev.items() if k != "previous"}
        restored.update({"restored_at": utcnow(), "restore_reason": reason, "previous": cur})
        nxt["main"] = restored
        _commit(nxt, {"op": "rollback_main", "tag": tag, "digest": digest, "reason": reason}, root)
    return load_state(root)


def repair_registry(*, root: Path | None = None) -> dict[str, Any]:
    """Kayıt defterini etkinleştirme kaydına eşitle (tutarsızlık görüldüğünde, insan eylemi)."""
    with _op_lock(root):
        st = load_state(root)
        if not st.get("virtual"):
            _sync_registry(str((st.get("main") or {}).get("adapter_id") or ""), root)
            _append_log({"op": "repair_registry"}, root)
    return consistency(root)


def overview(root: Path | None = None) -> dict[str, Any]:
    """Arayüz için: yuvalar, tutarlılık, adaylar ve her birinin hangi yuvaya uygun olduğu."""
    from app.evals.candidate_decisions import list_decisions

    st = load_state(root)
    latest: dict[str, dict[str, Any]] = {}
    for row in list_decisions(root):
        latest[f"{row['candidate_tag']}|{row['candidate_digest']}"] = row
    cands = []
    for row in sorted(latest.values(), key=lambda r: str(r.get("decided_at") or ""), reverse=True):
        cands.append(
            {
                **{
                    k: row.get(k)
                    for k in (
                        "decision_id",
                        "candidate_tag",
                        "candidate_digest",
                        "decision",
                        "comparison_id",
                        "adapter_id",
                        "decided_at",
                        "summary",
                    )
                },
                "decision_label": DECISION_TR.get(row["decision"], row["decision"]),
                "main_allowed": row["decision"] in MAIN_OK and main_role_ok(row),
                "role": row.get("role", ""),
                "trial_allowed": row["decision"] in TRIAL_OK,
            }
        )
    return {
        "main": describe_slot("main", root),
        "main_record": st.get("main"),
        "trial": describe_slot("trial", root) if st.get("trial") else None,
        "trial_record": st.get("trial"),
        "consistency": consistency(root),
        "candidates": cands,
        "note": "Etkinleştirme yalnız sonraki istekleri etkiler; devam eden cevap başladığı "
        "modelle tamamlanır. Karşılaştırma kararları LLM soru-cevap ölçümüdür, trading "
        "performansı değildir.",
    }
