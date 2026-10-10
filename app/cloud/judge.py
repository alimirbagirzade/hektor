"""Bulut hakem — K1 (eğitim sonrası kör paket) ve K2 (öğrenme adayı) — insan tıklamasıyla.

Tasarım: ``docs/TASARIM_SUREKLI_DONGU.md`` (Seçenek A). İkinci görüşle AYNI kurallar
(``app.cloud.second_opinion``): varsayılan kapalı, gönderilecek metnin TAMAMI önce gösterilir,
gönderim önizlenen özetle eşleşmezse reddedilir, her istek ayrı insan tıklaması, ortak günlük
kota. Toplu/otomatik gönderim YOKTUR:

- **K1** : karşılaştırmanın kör paketi (rol/model kimliği yok) + kilitli ölçüt. Paket
  ``cloud_judge_max_chars``'ı aşarsa PARÇALARA bölünür; her parça ayrı önizleme + ayrı tık.
  Tüm parçalar bitince puanlar ``candidate_compare.submit_ai_review`` ile AYRI dosyaya yazılır —
  insan puanı değildir, ``finalize``/terfi okumaz. Var olan AI incelemesi EZİLMEZ. Gizli final
  setinin paketi buluta GİTMEZ.
- **K2** : tek öğrenme adayı (soru + hedef metin + yerel kontrol özeti). Sonuç
  ``tutarli | supheli | belirsiz`` + gerekçe, AYRI kayıtta durur; adayın durumunu, verisini ya
  da eğitim satırını DEĞİŞTİRMEZ. Kabul/ret insanındır.

Bulut metni eğitim verisine girmez: bu modül ``app.feedback``/eğitim koduna yazmaz; eğitim
kapısı bulut kökenli satırı zaten NO-GO sayar (``app.cloud.policy.cloud_origin_lines``).
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.cloud import second_opinion as so
from app.cloud.providers import Provider, ProviderError
from app.config import get_settings

K1_NOTE = (
    "Bulut hakem puanı — İNSAN puanı DEĞİLDİR; karara ve terfiye girmez, kör insan incelemesinin "
    "yerine geçmez."
)
K2_NOTE = (
    "Bulut kontrolü — doğrulama DEĞİLDİR. Adayın durumunu değiştirmez, eğitime girmez; kabul/ret "
    "kararı sizindir."
)
K2_BLOCKED_STATUSES = {
    "leak": "Eval sızıntısı işaretli aday buluta gönderilmez (eval içeriği dışarı çıkmaz).",
    "excluded": "Hariç tutulan aday buluta gönderilmez.",
}
K2_VERDICTS = ("tutarli", "supheli")

_LIVE: dict[str, Provider] = {}
_LOCK = threading.Lock()

# Testlerde değiştirilebilir; varsayılan ikinci görüşle aynı fabrika.
provider_factory: Callable[[str], Provider] = lambda name: so.provider_factory(name)  # noqa: E731


class JudgeError(ValueError):
    """Kullanıcıya gösterilecek bulut hakem hatası."""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _dir() -> Path:
    d = get_settings().root / "storage" / "cloud" / "judge"
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


def records(kind: str, target: str) -> list[dict[str, Any]]:
    out = []
    for p in sorted(_dir().glob("jd_*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if r.get("kind") == kind and r.get("target") == target:
            out.append(r)
    return sorted(out, key=lambda r: str(r.get("created_at", "")))


def get(rec_id: str) -> dict[str, Any]:
    rec = _read(rec_id)
    if rec is None:
        raise JudgeError(f"Kayıt yok: {rec_id}")
    return rec


def cancel(rec_id: str) -> dict[str, Any]:
    rec = get(rec_id)
    if rec["status"] != "pending":
        return rec
    rec.update(status="cancelled", finished_at=_utcnow(), text="")
    _write(rec)
    with _LOCK:
        prov = _LIVE.get(rec_id)
    if prov is not None:
        prov.cancel()
    return get(rec_id)


def _cost_note(provider: str) -> str:
    if provider == "claude_code_cli":
        return (
            "Abonelikli CLI: her parça aboneliğinizin kotasından düşer; ikinci görüşle ORTAK "
            "günlük üst sınır."
        )
    return "Sahte sağlayıcı: maliyet yok, ağ çağrısı yok." if provider == "fake" else ""


def _launch(rec: dict[str, Any], on_done: Callable[[dict[str, Any]], None], wait: bool) -> None:
    s = get_settings()
    _write(rec)
    provider = provider_factory(s.cloud_provider)
    with _LOCK:
        _LIVE[rec["id"]] = provider
    th = threading.Thread(target=_run, args=(rec["id"], provider, on_done), daemon=True)
    th.start()
    if wait:
        th.join(timeout=float(s.cloud_timeout_s) + 5)


def _run(rec_id: str, provider: Provider, on_done: Callable[[dict[str, Any]], None]) -> None:
    s = get_settings()
    rec = _read(rec_id) or {}
    try:
        reply = provider.ask(rec["payload"], timeout_s=float(s.cloud_timeout_s))
        cur = _read(rec_id) or rec
        if cur["status"] == "cancelled":
            return  # iptal sonrası gelen cevap saklanmaz
        cur.update(status="done", text=reply.text, model=reply.model, finished_at=_utcnow())
        on_done(cur)  # ayrıştırır; ayrıştırma hatası kaydı 'failed' yapar (ham metin kalır)
        _write(cur)
    except ProviderError as exc:
        cur = _read(rec_id) or rec
        if cur["status"] != "cancelled":
            cur.update(status="failed", error=str(exc)[:500], finished_at=_utcnow())
            _write(cur)
    finally:
        with _LOCK:
            _LIVE.pop(rec_id, None)


def _new_record(kind: str, target: str, payload: str, **extra: Any) -> dict[str, Any]:
    s = get_settings()
    return {
        "id": "jd_" + secrets.token_hex(6),
        "kind": kind,
        "target": target,
        "created_at": _utcnow(),
        "provider": s.cloud_provider,
        "terms_ack": s.cloud_terms_ack,
        "payload_sha256": _sha(payload),
        "payload": payload,
        "status": "pending",
        "text": "",
        "model": "",
        "error": "",
        "finished_at": "",
        "training_use": False,
        **extra,
    }


def _json_object(text: str) -> dict[str, Any]:
    """Cevaptaki ilk ``{`` ile son ``}`` arası (``json`` çit içi de olur)."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise JudgeError("Bulut cevabında JSON nesnesi yok.")
    try:
        obj = json.loads(text[i : j + 1])
    except ValueError as exc:
        raise JudgeError(f"Bulut cevabı JSON olarak ayrıştırılamadı: {exc}") from exc
    if not isinstance(obj, dict):
        raise JudgeError("Bulut cevabı JSON nesnesi değil.")
    return obj


# ── K1 · eğitim sonrası kör paket hakemi ─────────────────────────────────────


def _k1_header(crit: dict[str, Any]) -> str:
    scale = crit.get("scale") or {}
    anchors = "\n".join(f"  {k}: {v}" for k, v in sorted((scale.get("anchors") or {}).items()))
    crits = "\n".join(f"  - {c}" for c in crit.get("critical_definitions") or [])
    return (
        "Bir yerel trading-araştırma modelinin cevaplarını KÖR olarak puanla. Hangi cevabın hangi "
        "modele ait olduğu bilinmiyor; etiketler (A, B, C) rastgeledir. Yatırım tavsiyesi verme.\n"
        f"Ölçek {scale.get('min', 0)}–{scale.get('max', 4)} (tam sayı):\n{anchors}\n"
        f"Kritik hata tanımları:\n{crits}\n"
        "Kaynaklı sorularda cevabı verilen KANIT ile karşılaştır; kanıtta olmayan iddia eksidir.\n"
        "YALNIZ şu biçimde tek bir JSON nesnesi döndür, başka metin yazma:\n"
        '{"scores": {"<soru_id>": {"<etiket>": {"score": 0, "critical": false, '
        '"note": "kısa gerekçe"}}}}\n'
        "Her sorudaki HER etiket puanlanmalı.\n\n"
    )


def _k1_block(p: dict[str, Any]) -> str:
    ev = "\n".join(f"  - {e}" for e in p.get("evidence") or []) or "  - (yok)"
    rub = "\n".join(f"  - {r}" for r in p.get("rubric") or [])
    answers = "\n".join(f"[{a['label']}]\n{a['answer']}" for a in p["answers"])
    return (
        f"### SORU {p['question_id']} (aile {p['family']}, boyut {p.get('dimension', '')})\n"
        f"{p['question']}\nKANIT:\n{ev}\n"
        + (f"RUBRİK:\n{rub}\n" if rub else "")
        + f"CEVAPLAR:\n{answers}\n\n"
    )


def k1_parts(cmp_id: str) -> list[dict[str, Any]]:
    """Kör paketi ölçüt başlığıyla parçalara böl (deterministik: paket diskte sabit)."""
    from app.evals import candidate_compare as cc

    try:
        m = cc._manifest(cmp_id)
        packet = cc.blind_packet(cmp_id)
        crit = cc._criteria(m["criteria_sha"])["criteria"]
    except cc.CompareError as exc:
        raise JudgeError(str(exc)) from exc
    header = _k1_header(crit)
    limit = get_settings().cloud_judge_max_chars
    parts: list[dict[str, Any]] = []
    cur_ids: list[str] = []
    cur_txt = ""
    for p in packet:
        block = _k1_block(p)
        if len(header) + len(block) > limit:
            raise JudgeError(
                f"{p['question_id']}: tek soru bile parça sınırını aşıyor ({len(block)} kr > "
                f"{limit}) — HEKTOR_CLOUD_JUDGE_MAX_CHARS'ı artırın."
            )
        if cur_ids and len(header) + len(cur_txt) + len(block) > limit:
            parts.append({"question_ids": cur_ids, "payload": header + cur_txt})
            cur_ids, cur_txt = [], ""
        cur_ids.append(p["question_id"])
        cur_txt += block
    if cur_ids:
        parts.append({"question_ids": cur_ids, "payload": header + cur_txt})
    labels = {p["question_id"]: sorted(a["label"] for a in p["answers"]) for p in packet}
    for i, part in enumerate(parts):
        part.update(
            index=i,
            payload_sha256=_sha(part["payload"]),
            chars=len(part["payload"]),
            labels={q: labels[q] for q in part["question_ids"]},
        )
    return parts


def _k1_part_state(cmp_id: str, part: dict[str, Any]) -> dict[str, Any] | None:
    """Bu parçanın (aynı metin özetiyle) son kaydı."""
    recs = [r for r in records("k1", cmp_id) if r.get("payload_sha256") == part["payload_sha256"]]
    return recs[-1] if recs else None


def k1_preview(cmp_id: str) -> dict[str, Any]:
    """Sıradaki parçanın TAMAMI + tüm parçaların durumu. Hiçbir şey GÖNDERMEZ."""
    from app.evals import candidate_compare as cc

    st = so.status()
    blockers = list(st["blockers"])
    try:
        m = cc._manifest(cmp_id)
    except cc.CompareError as exc:
        raise JudgeError(str(exc)) from exc
    if m.get("role_requested") == "final":
        blockers.append(
            "Gizli final setinin cevapları buluta gönderilmez (set bağımsızlığını yitirir)."
        )
    existing = cc.ai_review(cmp_id)
    written_by_k1 = bool(existing and str(existing.get("method", "")).startswith("K1"))
    if existing and not written_by_k1:
        blockers.append(
            f"Bu karşılaştırmanın AI incelemesi zaten var ({existing.get('reviewer_model')}) — "
            "ezilmez."
        )
    parts = k1_parts(cmp_id)
    rows = []
    for part in parts:
        rec = _k1_part_state(cmp_id, part)
        rows.append(
            {
                "index": part["index"],
                "question_ids": part["question_ids"],
                "chars": part["chars"],
                "payload_sha256": part["payload_sha256"],
                "status": (rec or {}).get("status", "none"),
                "record_id": (rec or {}).get("id", ""),
                "error": (rec or {}).get("error", ""),
            }
        )
    nxt = next((r for r in rows if r["status"] not in ("done", "pending")), None)
    if written_by_k1:
        nxt = None
    full = parts[nxt["index"]] if nxt else None
    return {
        **st,
        "enabled": not blockers and nxt is not None,
        "blockers": blockers,
        "comparison_id": cmp_id,
        "parts": rows,
        "n_parts": len(parts),
        "next": (
            {
                "index": full["index"],
                "payload": full["payload"],
                "payload_sha256": full["payload_sha256"],
                "chars": full["chars"],
                "est_input_tokens": int(full["chars"] / 3.5) + 1,
            }
            if full
            else None
        ),
        "ai_review_written": written_by_k1,
        "not_sent": [
            "model/rol kimliği ve mühürlü eşleme",
            "otomatik puanlanan (anahtarlı) sorular",
            ".env, data/, storage/, model ağırlıkları",
        ],
        "cost_note": _cost_note(st["provider"]),
        "cancel_note": so.CANCEL_NOTE,
        "note": K1_NOTE,
    }


def _k1_parse(text: str, labels: dict[str, list[str]]) -> dict[str, dict[str, dict[str, Any]]]:
    obj = _json_object(text)
    scores = obj.get("scores", obj)
    if not isinstance(scores, dict):
        raise JudgeError("'scores' nesnesi yok.")
    clean: dict[str, dict[str, dict[str, Any]]] = {}
    for qid, labs in labels.items():
        got = scores.get(qid)
        if not isinstance(got, dict) or sorted(got) != labs:
            raise JudgeError(f"{qid}: tüm etiketler puanlanmamış (beklenen {labs}).")
        clean[qid] = {}
        for lab in labs:
            v = got[lab] if isinstance(got[lab], dict) else {}
            score = v.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
                raise JudgeError(f"{qid}/{lab}: puan 0–4 tam sayı olmalı.")
            clean[qid][lab] = {
                "score": score,
                "critical": v.get("critical") is True,
                "note": str(v.get("note", ""))[:500],
            }
    return clean


def _k1_finish_if_complete(cmp_id: str) -> None:
    """Tüm parçalar 'done' ise birleşik puanları AYRI AI incelemesi olarak yaz (bir kez)."""
    from app.evals import candidate_compare as cc

    if cc.ai_review(cmp_id) is not None:
        return
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    models, recs = set(), []
    for part in k1_parts(cmp_id):
        rec = _k1_part_state(cmp_id, part)
        if rec is None or rec.get("status") != "done" or not rec.get("scores"):
            return
        merged.update(rec["scores"])
        models.add(f"{rec.get('provider')}/{rec.get('model')}")
        recs.append(rec["id"])
    cc.submit_ai_review(
        cmp_id,
        merged,
        reviewer_model="bulut:" + ",".join(sorted(models)),
        method=(
            f"K1 bulut hakem — kör paket + kilitli ölçüt, {len(recs)} parça, her parça ayrı "
            f"insan tıklamasıyla ({', '.join(recs)}). {K1_NOTE}"
        ),
    )


def k1_start(cmp_id: str, part_index: int, payload_sha256: str, *, wait: bool = False) -> dict:
    """(İnsan) Sıradaki parçayı gönder. Önizlenen özet tutmazsa GÖNDERMEZ."""
    pv = k1_preview(cmp_id)
    nxt = pv["next"]
    for row in pv["parts"]:  # çift tıklama: bekleyen aynı parça yeni istek açmaz
        if (
            row["index"] == part_index
            and row["status"] == "pending"
            and row["payload_sha256"] == payload_sha256
        ):
            return {**get(row["record_id"]), "replayed": True}
    if not pv["enabled"] or nxt is None:
        raise JudgeError(
            "Bulut hakem kapalı: " + (" | ".join(pv["blockers"]) or "gönderilecek parça yok")
        )
    if nxt["index"] != part_index or nxt["payload_sha256"] != payload_sha256:
        raise JudgeError("Gönderilecek metin önizlemeden farklı — önizlemeyi yenileyin.")
    part = k1_parts(cmp_id)[part_index]
    rec = _new_record(
        "k1",
        cmp_id,
        nxt["payload"],
        part_index=part_index,
        question_ids=part["question_ids"],
        labels=part["labels"],
        scores=None,
        note=K1_NOTE,
    )

    def on_done(cur: dict[str, Any]) -> None:
        try:
            cur["scores"] = _k1_parse(cur["text"], cur["labels"])
        except JudgeError as exc:
            cur.update(status="failed", error=f"ayrıştırılamadı: {exc}"[:500])
            return
        _write(cur)
        try:
            _k1_finish_if_complete(cmp_id)
        except Exception as exc:  # birleşik yazım başarısızsa parça yine 'done' kalır
            cur["error"] = f"AI incelemesi yazılamadı: {exc}"[:500]

    _launch(rec, on_done, wait)
    return get(rec["id"])


# ── K2 · öğrenme adayı kontrolü ──────────────────────────────────────────────


def _candidate(candidate_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from app.feedback.chat_store import ChatStore

    store = ChatStore()
    cand = store.get_candidate(candidate_id)
    if cand is None:
        raise JudgeError(f"Aday bulunamadı: {candidate_id}")
    turn = store.get_turn(cand["turn_id"]) or {}
    return cand, turn


def k2_payload(cand: dict[str, Any], turn: dict[str, Any]) -> str:
    """Gönderilecek metnin TAMAMI (önizlemede aynen gösterilir). Kaynak metni GİTMEZ."""
    checks = [
        f"- {c.get('kind')}: {c.get('status') or c.get('label') or ''} {c.get('detail') or ''}"
        for c in ((cand.get("verification") or {}).get("checks") or [])
    ]
    return (
        "Bir yerel trading-araştırma asistanının eğitim ADAYI cevabını eleştirel olarak incele. "
        "Yatırım tavsiyesi verme. Cevap soruyla tutarlı, kanıtlanabilir ve maliyet/look-ahead/"
        "örneklem varsayımlarını açık mı? İlk satıra YALNIZ 'KARAR: tutarli' ya da 'KARAR: "
        "supheli' yaz; sonra gerekçeyi madde madde ver, emin olmadığını söyle.\n\n"
        f"SORU:\n{turn.get('question', '')}\n\n"
        f"ADAY CEVAP ({'kullanıcı düzeltmesi' if cand.get('kind') == 'correct' else 'model cevabı'}"
        f", alan {cand.get('domain', '')}):\n{cand.get('target_text', '')}\n\n"
        "YEREL DETERMİNİSTİK KONTROLLER:\n" + ("\n".join(checks) or "- (yok)")
    )


def k2_preview(candidate_id: str) -> dict[str, Any]:
    cand, turn = _candidate(candidate_id)
    s = get_settings()
    st = so.status()
    blockers = list(st["blockers"])
    if cand.get("status") in K2_BLOCKED_STATUSES:
        blockers.append(K2_BLOCKED_STATUSES[cand["status"]])
    payload = k2_payload(cand, turn)
    if len(payload) > s.cloud_max_chars:
        blockers.append(f"Gönderilecek metin {len(payload)} kr > üst sınır {s.cloud_max_chars}.")
    return {
        **st,
        "enabled": not blockers,
        "blockers": blockers,
        "candidate_id": candidate_id,
        "payload": payload,
        "payload_sha256": _sha(payload),
        "payload_chars": len(payload),
        "est_input_tokens": int(len(payload) / 3.5) + 1,
        "not_sent": [
            "kaynak parçaları ve makale metinleri",
            "diğer adaylar (tık başına TEK aday)",
            ".env, data/, storage/, model ağırlıkları",
        ],
        "cost_note": _cost_note(st["provider"]),
        "cancel_note": so.CANCEL_NOTE,
        "note": K2_NOTE,
        "history": records("k2", candidate_id),
    }


def _k2_verdict(text: str) -> str:
    m = re.search(r"KARAR\s*:\s*(tutarl[ıi]|şüpheli|supheli)", text, re.IGNORECASE)
    if not m:
        return "belirsiz"
    v = m.group(1).lower().replace("ı", "i").replace("ş", "s").replace("ü", "u")
    return v if v in K2_VERDICTS else "belirsiz"


def k2_start(candidate_id: str, payload_sha256: str, *, wait: bool = False) -> dict[str, Any]:
    """(İnsan) TEK adayı gönder. Önizlenen özet tutmazsa GÖNDERMEZ."""
    pv = k2_preview(candidate_id)
    for r in pv["history"]:
        if r["status"] == "pending" and r["payload_sha256"] == payload_sha256:
            return {**r, "replayed": True}
    if not pv["enabled"]:
        raise JudgeError("Bulut kontrolü kapalı: " + " | ".join(pv["blockers"]))
    if payload_sha256 != pv["payload_sha256"]:
        raise JudgeError("Gönderilecek metin önizlemeden farklı — önizlemeyi yenileyin.")
    cand, _ = _candidate(candidate_id)
    rec = _new_record(
        "k2",
        candidate_id,
        pv["payload"],
        candidate_revision=cand.get("revision"),
        target_sha=cand.get("target_sha", ""),
        verdict="",
        note=K2_NOTE,
    )

    def on_done(cur: dict[str, Any]) -> None:
        cur["verdict"] = _k2_verdict(cur["text"])

    _launch(rec, on_done, wait)
    return get(rec["id"])


def k2_latest(candidate_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Aday başına son K2 kaydının özeti (liste ekranı için; salt-okuma)."""
    want = set(candidate_ids)
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(_dir().glob("jd_*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if r.get("kind") != "k2" or r.get("target") not in want:
            continue
        prev = out.get(r["target"])
        if prev is None or str(r.get("created_at")) >= str(prev.get("created_at")):
            out[r["target"]] = {
                k: r.get(k)
                for k in ("id", "status", "verdict", "created_at", "target_sha", "error")
            }
    return out
