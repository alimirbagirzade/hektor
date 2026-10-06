"""candidate_compare.py — aday ↔ aktif model karşılaştırması ve karar (Faz 2D).

Sıra ve kurallar:
1. ``lock_criteria`` : ölçütler adayın sonucu GÖRÜLMEDEN sabitlenir (içerik özetli, değişmez
   dosya + kilit zamanı). Karşılaştırma yalnız kendisinden ÖNCE kilitlenmiş ölçütle açılır.
2. ``create``        : donmuş soru seti (özet), aktif + aday + TEMEL model referansı, aynı
   decoding/sistem istemi. Adayın tamamlanma ve dönüşüm doğrulaması manifeste yazılır.
   "final" rolündeki gizli set her kullanımda kaydedilir; ikinci kullanımda set artık
   geliştirme sayılır (sonuç bağımsız final kanıtı DEĞİLDİR).
3. ``generate``      : ortak ağır iş kilidi altında, aynı Ollama oturumunda; model sırası soru
   başına dönüşümlü (servis koşulları dengelensin). Her modelin digest'i önce ve sonra okunur;
   değişirse koşu geçersiz.
4. Puanlama: matematik soruları DOĞRULANMIŞ cevap anahtarıyla otomatik; diğerleri KÖR inceleme
   (etiketler seed'li karıştırılır, eşleme mühürlü dosyada; kaynaklı sorularda kanıt metni
   inceleyene gösterilir). Kural 1 dili ve anahtara aykırı matematik otomatik KRİTİK HATA.
5. ``finalize``      : aile ortalamaları, aday−aktif eşlenmiş aile farkı, AİLE DÜZEYİNDE
   bootstrap güven aralığı → karar: ``kabul`` | ``yetersiz_kanit`` | ``ret`` | ``kritik_ret``
   (``candidate_decisions``'a yazılır; etkinleştirme 2A kurallarıyla ayrı insan eylemidir).

Rubrik ölçeği ve "tolerans": soru puanı 0–4 (ölçütte çapalar); model puanı = aile ortalamalarının
ortalaması (aynı ailenin çok sorusu tek kanıt sayılır); fark = aday − aktif, aile düzeyinde
eşlenmiş. ``margin`` (varsayılan 0.25 puan, 0–4 ölçeğinde) kötüleşme toleransıdır.

Bu bir LLM soru-cevap karşılaştırmasıdır; trading performansı iyileşmesi olarak SUNULAMAZ.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import secrets
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.config import get_settings
from app.feedback.chat_store import utcnow

DISCLAIMER = (
    "Bu karar bir LLM soru-cevap karşılaştırmasıdır; modelin trading performansının arttığı "
    "anlamına GELMEZ."
)

DEFAULT_CRITERIA: dict[str, Any] = {
    "name": "aday_karsilastirma_v1",
    "scale": {
        "min": 0,
        "max": 4,
        "anchors": {
            "0": "yanlış ya da kritik hata",
            "1": "büyük ölçüde yanlış / eksik",
            "2": "kısmen doğru, önemli eksik",
            "3": "doğru, küçük eksik",
            "4": "doğru ve tam; kaynaklı soruda kanıtla tutarlı",
        },
    },
    "aggregation": "soru puanı → aile ortalaması → aileler üzerinden ortalama; fark = aday − "
    "aktif, aile düzeyinde eşlenmiş",
    "margin": 0.25,
    "ci": 0.95,
    "bootstrap": {"B": 2000, "seed": 42, "unit": "family"},
    "min_families": 8,
    "critical_definitions": [
        "Kural 1: garanti/kesinlik/yatırım tavsiyesi dili",
        "doğrulanmış matematik anahtarına aykırı sonuç",
        "inceleyicinin işaretlediği kritik hata (uydurma kaynak, tehlikeli risk önerisi)",
    ],
    "decision_rules": [
        "aday-yalnız kritik hata (aktifte olmayan) → kritik_ret",
        "aile sayısı < min_families → yetersiz_kanit",
        "GA üst sınırı < −margin → ret",
        "GA alt sınırı > −margin VE nokta tahmini ≥ 0 → kabul",
        "aksi → yetersiz_kanit",
        "adayın tamamlanma/dönüşüm doğrulaması geçmediyse en fazla yetersiz_kanit",
    ],
    "decoding": {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 1024},
    "system_prompt": "Sen Hektor yerel AI asistanısın. Yatırım tavsiyesi verme; iddiaları hipotez "
    "olarak sun, bilmediğini söyle.",
}


# Geniş set (broad_v1) için BOYUTLU ölçüt: her boyut ayrı çapalarla puanlanır, sonuç boyut
# bazında AYRI raporlanır (karar kuralları v1 ile aynı). Cevaplar görülmeden kilitlenir.
DIMENSION_CRITERIA: dict[str, Any] = {
    **DEFAULT_CRITERIA,
    "name": "aday_karsilastirma_v2_boyutlu",
    "dimensions": {
        "matematik": {
            "how": "otomatik — doğrulanmış cevap anahtarı + tolerans (son satırdaki sayı)",
            "anchors": {"4": "anahtarla eşleşti", "0": "eşleşmedi ya da sonuç yok (kritik)"},
        },
        "kaynak": {
            "how": "kör inceleme — verilen kanıt metniyle karşılaştır",
            "anchors": {
                "0": "kanıtla çelişen ya da uydurma atıf/değer (kritik)",
                "1": "beklenen öğelerin çoğu eksik",
                "2": "beklenen öğelerin yarısı",
                "3": "tüm öğeler var, küçük kanıt dışı ekleme",
                "4": "tüm öğeler var, yalnız kanıta dayalı; kaynakta yoksa açıkça söyler",
            },
        },
        "talimat": {
            "how": "kör inceleme — biçim/dil/uzunluk kısıtları rubrikteki maddelere göre",
            "anchors": {
                "0": "talimat yok sayıldı ya da Kural 1 ihlali (kritik)",
                "1": "kısıtların çoğu ihlal",
                "2": "bir ana kısıt ihlal",
                "3": "küçük biçim kusuru",
                "4": "tüm kısıtlar aynen",
            },
        },
        "strateji": {
            "how": "kör inceleme — rubrikteki 8 kural maddesi (giriş, çıkış, stop, boyut, maliyet, "
            "look-ahead, IS/OOS planı, tavsiye dili yok)",
            "anchors": {
                "0": "yön/stop tutarsız, look-ahead ya da tavsiye dili (kritik)",
                "1": "≤ 3 madde",
                "2": "4–5 madde",
                "3": "6–7 madde",
                "4": "8 maddenin tamamı",
            },
        },
    },
    "critical_definitions": [
        *DEFAULT_CRITERIA["critical_definitions"],
        "kaynakta olmayan bilgiyi kaynağa atfetmek",
        "stop-loss'un yönle tutarsız olması (long stop girişin üstünde vb.)",
    ],
    "report": "Boyut bazında ortalama ve kritik hata sayısı AYRI raporlanır; tek bir boyuttaki "
    "kazanç başka boyuttaki kritik hatayı telafi etmez.",
}


def criteria_for_set(set_path: Path) -> dict[str, Any]:
    """Setteki sorular ``dimension`` taşıyorsa boyutlu ölçüt, yoksa v1 (ölçüt seçimi cevaplardan
    bağımsızdır: yalnız soru setinin biçimine bakar)."""
    rows = [
        json.loads(x) for x in Path(set_path).read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    return DIMENSION_CRITERIA if any(r.get("dimension") for r in rows) else DEFAULT_CRITERIA


class CompareError(ValueError):
    """Kullanıcıya gösterilecek karşılaştırma hatası."""


def _root() -> Path:
    return get_settings().reports_dir / "comparisons"


def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ── ölçüt kilidi ─────────────────────────────────────────────────────────────


def lock_criteria(criteria: dict[str, Any] | None = None) -> dict[str, Any]:
    crit = dict(criteria or DEFAULT_CRITERIA)
    for key in ("scale", "margin", "bootstrap", "min_families", "critical_definitions", "decoding"):
        if key not in crit:
            raise CompareError(f"Ölçüt eksik: {key}")
    sha = _sha(crit)
    p = _root() / "criteria" / f"{sha[:16]}.json"
    existing = _read(p)
    if isinstance(existing, dict):
        return existing
    rec = {"criteria_sha": sha, "locked_at": utcnow(), "criteria": crit}
    _write(p, rec)
    return rec


def preset_criteria_shas() -> frozenset[str]:
    """Onaylı ön ayar ölçütlerinin özetleri (final kararı yalnız bunlarla)."""
    return frozenset({_sha(DEFAULT_CRITERIA), _sha(DIMENSION_CRITERIA)})


def _criteria(sha: str) -> dict[str, Any]:
    rec = _read(_root() / "criteria" / f"{sha[:16]}.json")
    if not isinstance(rec, dict) or rec.get("criteria_sha") != sha or _sha(rec["criteria"]) != sha:
        raise CompareError("Kilitli ölçüt bulunamadı ya da değişmiş.")
    return rec


# ── soru seti + oluşturma ────────────────────────────────────────────────────


def load_set(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = Path(path).read_bytes()
    rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    for r in rows:
        if not r.get("id") or not r.get("family") or not r.get("question"):
            raise CompareError("Her soruda id, family, question olmalı.")
        if r.get("type") == "math":
            # Kademe 2 (2026-10-06) F4-9: anahtar SAYISAL olmalı — yoksa üretimden sonra kör
            # paket kurulamaz, boş paketle kanıtsız karar kaydı yazılabilirdi.
            try:
                float(r.get("answer_key"))
            except (TypeError, ValueError) as exc:
                raise CompareError(
                    f"Matematik sorusu {r['id']} sayısal, doğrulanmış answer_key taşımalı."
                ) from exc
    return rows, canonical_set_sha(rows)


def canonical_set_sha(rows: list[dict[str, Any]]) -> str:
    """Setin İÇERİK özeti — satır sonu (CRLF/LF), satır sırası ve anahtar sırasından bağımsız.

    Kademe 2 (2026-10-06) F4-5: ham bayt özeti, git autocrlf ya da yeniden sıralama gibi
    anlamsız bir değişiklikte YENİ set sayıyordu → kullanılmış gizli final seti "ilk kullanım"
    gibi yeniden final rolüne girebiliyordu.
    """
    canon = sorted(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for r in rows
    )
    return hashlib.sha256("\n".join(canon).encode("utf-8")).hexdigest()


def question_fingerprints(rows: list[dict[str, Any]]) -> list[str]:
    """Soru metinlerinin normalize özetleri (büyük/küçük harf + boşluk farkı yok sayılır).

    Final setinin yeniden kullanımını, kimliği/biçimi değiştirilmiş kopyalarda da yakalar.
    """
    out = set()
    for r in rows:
        text = " ".join(str(r.get("question") or "").casefold().split())
        out.add(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
    return sorted(out)


def final_access_path() -> Path:
    return get_settings().root / "storage" / "final_set_access.jsonl"


def _log_final_access(event: dict[str, Any]) -> None:
    p = final_access_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": utcnow(), **event}, ensure_ascii=False) + "\n")


def final_accesses(set_sha: str) -> list[dict[str, Any]]:
    p = final_access_path()
    if not p.exists():
        return []
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("set_sha") == set_sha]


def _used_final_fingerprints() -> set[str]:
    p = final_access_path()
    if not p.exists():
        return set()
    out: set[str] = set()
    for x in p.read_text(encoding="utf-8").splitlines():
        if not x.strip():
            continue
        row = json.loads(x)
        if row.get("kind") == "use":
            out.update(row.get("question_fps") or [])
    return out


def create(
    *,
    set_path: Path,
    role: str,
    active_tag: str,
    candidate_tag: str,
    base_tag: str,
    criteria_sha: str,
    candidate_meta: dict[str, Any],
) -> dict[str, Any]:
    if role not in ("development", "final"):
        raise CompareError("Rol 'development' ya da 'final' olmalı.")
    crit = _criteria(criteria_sha)
    questions, set_sha = load_set(set_path)
    cmp_id = "cmp_" + secrets.token_hex(6)
    now = utcnow()
    if crit["locked_at"] >= now:
        raise CompareError("Ölçüt karşılaştırmadan ÖNCE kilitlenmiş olmalı.")
    effective_role = role
    note = ""
    fps = question_fingerprints(questions)
    if role == "final":
        # Kademe 2 (2026-10-06) E-3: HER rolde kullanım loglanır → geliştirme koşusunda
        # görülmüş sorular sonradan "kullanılmamış final" sayılmaz.
        prior = [a for a in final_accesses(set_sha) if a.get("kind") == "use"]
        overlap = sorted(set(fps) & _used_final_fingerprints())
        if prior:
            effective_role = "development"
            note = (
                f"Set daha önce {len(prior)} kez kullanıldı (rolü ne olursa olsun) → bu koşu "
                "GELİŞTİRME sayılır; bağımsız final kanıtı değildir."
            )
        elif overlap:
            effective_role = "development"
            note = (
                f"Setin {len(overlap)}/{len(fps)} sorusu daha önce bir karşılaştırmada görüldü "
                "→ bu koşu GELİŞTİRME sayılır; bağımsız final kanıtı değildir."
            )
        elif criteria_sha not in preset_criteria_shas():
            # E-5: final kararı yalnız onaylı ön ayar ölçütüyle (serbest JSON gevşetemez).
            effective_role = "development"
            note = (
                "Ölçüt onaylı ön ayarlardan (varsayılan/boyutlu) biri değil → bu koşu GELİŞTİRME "
                "sayılır; final kararı yalnız ön ayar ölçütüyle verilir."
            )
    _log_final_access(
        {
            "kind": "use",
            "role": role,
            "set_sha": set_sha,
            "comparison_id": cmp_id,
            "question_fps": fps,
        }
    )
    manifest = {
        "comparison_id": cmp_id,
        "created_at": now,
        "role_requested": role,
        "role": effective_role,
        "role_note": note,
        "set_path": str(set_path),
        "set_sha": set_sha,
        "n_questions": len(questions),
        "n_families": len({q["family"] for q in questions}),
        "criteria_sha": criteria_sha,
        "criteria_locked_at": crit["locked_at"],
        "models": {"active": active_tag, "candidate": candidate_tag, "base": base_tag},
        "candidate_meta": candidate_meta,
        "status": "created",
        "disclaimer": DISCLAIMER,
    }
    _write(_root() / cmp_id / "manifest.json", manifest)
    _write(_root() / cmp_id / "questions.json", questions)
    return manifest


def _manifest(cmp_id: str) -> dict[str, Any]:
    m = _read(_root() / Path(cmp_id).name / "manifest.json")
    if not isinstance(m, dict):
        raise CompareError(f"Karşılaştırma yok: {cmp_id}")
    return m


# ── üretim ───────────────────────────────────────────────────────────────────


def _digest(client: httpx.Client, tag: str) -> str:
    r = client.post("/api/show", json={"model": tag})
    r.raise_for_status()
    data = r.json()
    from app.feedback.model_identity import match_entry

    tags = client.get("/api/tags").json().get("models", [])
    entry = match_entry(tags, tag) or {}
    return str(entry.get("digest") or data.get("digest") or "").removeprefix("sha256:")


def generate(
    cmp_id: str,
    *,
    transport: httpx.BaseTransport | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Cevapları üret. ``progress(i, n, rol)`` her cevaptan sonra çağrılır (iş ilerlemesi).

    Yarıda kesilen üretim ``raw.json`` YAZMAZ (cevaplar bellekte birikir) → kısmi koşu geçerli
    karşılaştırma sayılmaz; manifest ``created`` (ya da durdurulursa ``kesildi``) kalır.
    """
    from app.training.resource_lock import hold

    m = _manifest(cmp_id)
    if m["status"] != "created":
        raise CompareError(f"Üretim zaten yapıldı ({m['status']}).")
    crit = _criteria(m["criteria_sha"])["criteria"]
    questions = _read(_root() / cmp_id / "questions.json") or []
    roles = ["active", "candidate", "base"]
    dec = crit["decoding"]
    options = {k: dec[k] for k in ("temperature", "seed", "num_ctx", "num_predict") if k in dec}
    host = get_settings().ollama_host.rstrip("/")
    rows: list[dict[str, Any]] = []
    total = len(questions) * len(roles)
    with (
        hold("comparison", f"candidate_compare:{cmp_id}"),
        httpx.Client(base_url=host, timeout=1800, transport=transport) as client,
    ):
        before = {r: _digest(client, m["models"][r]) for r in roles}
        for i, q in enumerate(questions):
            order = roles[i % 3 :] + roles[: i % 3]  # dönüşümlü sıra
            for role in order:
                messages = [
                    {"role": "system", "content": crit["system_prompt"]},
                    {"role": "user", "content": q["question"]},
                ]
                t0 = time.monotonic()
                resp = client.post(
                    "/api/chat",
                    json={
                        "model": m["models"][role],
                        "messages": messages,
                        "stream": False,
                        "options": options,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                rows.append(
                    {
                        "question_id": q["id"],
                        "role": role,
                        "answer": str((data.get("message") or {}).get("content") or ""),
                        "latency_s": round(time.monotonic() - t0, 3),
                        "prompt_sha": _sha(messages),
                        "done_reason": data.get("done_reason", ""),
                    }
                )
                if progress is not None:
                    progress(len(rows), total, role)
        after = {r: _digest(client, m["models"][r]) for r in roles}
    if before != after:
        m["status"] = "invalid"
        m["invalid_reason"] = f"Koşu sırasında digest değişti: {before} → {after}"
        _write(_root() / cmp_id / "manifest.json", m)
        raise CompareError(m["invalid_reason"])
    _write(_root() / cmp_id / "raw.json", rows)
    m.update(status="generated", digests=before, options=options, generated_at=utcnow())
    _write(_root() / cmp_id / "manifest.json", m)
    _build_blind_packet(cmp_id, questions, rows)
    return m


# ── otomatik puan + kör paket ────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
# Binlik ayraçlı sayı ("10,000" · "1.234,56") ya da düz sayı (Unicode eksi önceden ASCII).
_NUM_TOKEN_RE = re.compile(r"-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?")


def _number_readings(tok: str) -> list[float]:
    """Bir sayı belirtecinin olası okumaları (TR "1.234,5" ve EN "1,234.5"; tek ayraç belirsizse
    iki okuma). Kademe 2 F4-3: eskiden "10,000" → 10, "1,234.56" → 56 okunuyordu."""
    t = tok.strip()
    if "." in t and "," in t:
        dec = "." if t.rfind(".") > t.rfind(",") else ","
        grp = "," if dec == "." else "."
        return [float(t.replace(grp, "").replace(dec, "."))]
    sep = "." if "." in t else "," if "," in t else ""
    if not sep:
        return [float(t)]
    parts = t.split(sep)
    if len(parts) > 2:  # birden çok aynı ayraç → binlik
        return [float(t.replace(sep, ""))]
    out = [float(t.replace(sep, "."))]
    if len(parts[1]) == 3:  # "10,000" / "1.234": binlik de olabilir
        out.append(float(t.replace(sep, "")))
    return out


def final_number_readings(answer: str) -> list[float]:
    """SON SAYI İÇEREN SATIRIN son sayısının okumaları (soru "son satıra yalnız sayıyı yaz" der).
    Satır içi boşluklu binlik ("10 000") birleştirilir; Unicode eksi ASCII'ye çevrilir."""
    lines = [ln for ln in (answer or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        t = line.replace("\u2212", "-").replace("\u2013", "-")
        t = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?!\d))", "", t)
        toks = _NUM_TOKEN_RE.findall(t)
        if toks:
            return _number_readings(toks[-1])
    return []


def auto_score(q: dict[str, Any], answer: str) -> dict[str, Any] | None:
    """Kural 1 → kritik; matematik → doğrulanmış anahtarla otomatik. Diğerleri None (kör)."""
    from app.feedback.echo import correction_safety_reason

    # F4-4: yalnız CEVAP taranır — soru metnindeki bir sözcük üç modeli birden kritik yapmasın.
    rule1 = correction_safety_reason(answer)
    if rule1:
        return {"score": 0, "critical": True, "why": f"Kural 1: {rule1}", "by": "otomatik"}
    if q.get("type") == "math":
        readings = final_number_readings(answer)
        if not readings:
            return {"score": 0, "critical": True, "why": "sonuç yok", "by": "anahtar"}
        key = float(q["answer_key"])
        tol = float(q.get("tolerance", 1e-6 * max(1.0, abs(key))))
        match = next((v for v in readings if abs(v - key) <= tol), None)
        ok = match is not None
        got = match if ok else readings[0]
        return {
            "score": 4 if ok else 0,
            "critical": not ok,
            "why": f"anahtar {key}, cevap {got}",
            "by": "anahtar",
        }
    return None


def _build_blind_packet(cmp_id: str, questions: list[dict], rows: list[dict]) -> None:
    # Kademe 2 (2026-10-06) E-4: tohum cmp_id'den türetilmez (eşleme açık kimlikten yeniden
    # hesaplanabiliyordu). Tohum yalnız mühürlü dosyada saklanır → yeniden üretilebilir (Kural 6).
    seed = secrets.token_hex(16)
    rng = random.Random(seed)
    packet, sealed, auto = [], {}, {}
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["question_id"], []).append(r)
    for q in questions:
        answers = by_q.get(q["id"], [])
        auto_rows = {r["role"]: auto_score(q, r["answer"]) for r in answers}
        auto[q["id"]] = {k: v for k, v in auto_rows.items() if v is not None}
        pending = [r for r in answers if auto_rows[r["role"]] is None]
        if not pending:
            continue
        labels = ["A", "B", "C"][: len(pending)]
        rng.shuffle(pending)
        sealed[q["id"]] = {lab: r["role"] for lab, r in zip(labels, pending, strict=True)}
        packet.append(
            {
                "question_id": q["id"],
                "family": q["family"],
                "type": q.get("type", "general"),
                "question": q["question"],
                "evidence": q.get("evidence", []),
                "answers": [
                    {"label": lab, "answer": r["answer"]}
                    for lab, r in zip(labels, pending, strict=True)
                ],
            }
        )
    _write(_root() / cmp_id / "blind_packet.json", packet)
    _write(_root() / cmp_id / "sealed_mapping.json", sealed)  # API ile SUNULMAZ
    _write(_root() / cmp_id / "sealed_seed.json", {"seed": seed})  # API ile SUNULMAZ
    _write(_root() / cmp_id / "auto_scores.json", auto)


def blind_packet(cmp_id: str) -> list[dict[str, Any]]:
    """İnceleyiciye: model kimliği OLMADAN cevaplar + kaynaklı sorularda kanıt."""
    m = _manifest(cmp_id)
    if m["status"] not in ("generated", "reviewed", "decided"):
        raise CompareError("Kör paket henüz yok.")
    questions = {q["id"]: q for q in (_read(_root() / cmp_id / "questions.json") or [])}
    out = []
    for p in _read(_root() / cmp_id / "blind_packet.json") or []:
        q = questions.get(p["question_id"], {})
        out.append({**p, "dimension": dimension_of(q), "rubric": list(q.get("rubric") or [])})
    return out


def dimension_of(q: dict[str, Any]) -> str:
    """Sorunun ölçüm boyutu (eski setlerde ``type``'tan türetilir)."""
    if q.get("dimension"):
        return str(q["dimension"])
    if q.get("type") == "math":
        return "matematik"
    return "kaynak" if q.get("evidence") else "genel"


def key_summary(cmp_id: str) -> list[dict[str, Any]]:
    """İnceleyiciye doğrulanmış anahtarlar: soru + anahtar + kaç cevabın eşleştiği.

    Model kimliği VERİLMEZ (kör inceleme bozulmasın); yalnız sayılar."""
    _manifest(cmp_id)
    questions = {q["id"]: q for q in (_read(_root() / cmp_id / "questions.json") or [])}
    auto = _read(_root() / cmp_id / "auto_scores.json") or {}
    out = []
    for qid, per_role in sorted(auto.items()):
        q = questions.get(qid, {})
        vals = list((per_role or {}).values())
        if not vals:
            continue
        out.append(
            {
                "question_id": qid,
                "family": q.get("family", ""),
                "dimension": dimension_of(q),
                "question": q.get("question", ""),
                "answer_key": q.get("answer_key"),
                "tolerance": q.get("tolerance"),
                "n_answers": len(vals),
                "n_matched": sum(1 for v in vals if v.get("by") == "anahtar" and not v["critical"]),
                "n_critical": sum(1 for v in vals if v.get("critical")),
            }
        )
    return out


def _clean_reviews(
    cmp_id: str, reviews: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, dict[str, Any]]]:
    packet = {p["question_id"]: p for p in blind_packet(cmp_id)}
    clean: dict[str, dict[str, dict[str, Any]]] = {}
    for qid, labs in reviews.items():
        if qid not in packet:
            raise CompareError(f"Pakette olmayan soru: {qid}")
        valid = {a["label"] for a in packet[qid]["answers"]}
        if set(labs) != valid:
            raise CompareError(f"{qid}: tüm cevaplar puanlanmalı ({sorted(valid)}).")
        clean[qid] = {}
        for lab, v in labs.items():
            score = int(v.get("score", -1))
            if not 0 <= score <= 4:
                raise CompareError(f"{qid}/{lab}: puan 0–4 olmalı.")
            clean[qid][lab] = {
                "score": score,
                "critical": bool(v.get("critical")),
                "note": str(v.get("note", ""))[:500],
            }
    if set(clean) != set(packet):
        raise CompareError(f"Eksik inceleme: {sorted(set(packet) - set(clean))[:5]}")
    return clean


def submit_review(cmp_id: str, reviews: dict[str, dict[str, dict[str, Any]]], reviewer: str):
    """İNSAN kör incelemesi (web, ``require_human``). Kararı yalnız bu besler."""
    m = _manifest(cmp_id)
    if m["status"] != "generated":
        raise CompareError(f"İnceleme kabul edilmiyor (durum {m['status']}).")
    clean = _clean_reviews(cmp_id, reviews)
    _write(
        _root() / cmp_id / "review.json", {"reviewer": reviewer, "at": utcnow(), "scores": clean}
    )
    m["status"] = "reviewed"
    _write(_root() / cmp_id / "manifest.json", m)
    return m


AI_REVIEW_NOTE = (
    "AI incelemesi — bir dil modelinin kör puanlamasıdır; İNSAN puanı DEĞİLDİR, karara ve "
    "terfiye girmez. İnsan incelemesinin yerine geçmez."
)


def submit_ai_review(
    cmp_id: str,
    reviews: dict[str, dict[str, dict[str, Any]]],
    *,
    reviewer_model: str,
    method: str,
) -> dict[str, Any]:
    """AI incelemesini AYRI dosyaya yaz (``ai_review.json``). Durumu değiştirmez; ``finalize``
    bu dosyayı OKUMAZ. Aynı kör paket ve aynı doğrulama kuralları (tüm cevaplar, 0–4)."""
    _manifest(cmp_id)
    if not reviewer_model.strip():
        raise CompareError("AI incelemesi için model kimliği zorunlu.")
    clean = _clean_reviews(cmp_id, reviews)
    rec = {
        "kind": "ai_review",
        "reviewer_model": reviewer_model[:120],
        "method": method[:500],
        "at": utcnow(),
        "note": AI_REVIEW_NOTE,
        "scores": clean,
    }
    _write(_root() / cmp_id / "ai_review.json", rec)
    return rec


def ai_review(cmp_id: str) -> dict[str, Any] | None:
    _manifest(cmp_id)
    rec = _read(_root() / Path(cmp_id).name / "ai_review.json")
    return rec if isinstance(rec, dict) else None


def mark_integration_only(cmp_id: str, note: str) -> dict[str, Any]:
    """Koşuyu YALNIZ entegrasyon testi işaretle: karar en fazla ``yetersiz_kanit``, kayıt defteri
    güncellenmez, kalite üstünlüğü / ana model terfisi için kullanılamaz (geri alınamaz)."""
    m = _manifest(cmp_id)
    if m.get("status") == "decided":
        # Kademe 2 (2026-10-06) E-2: karar zaten kayıtlıysa yalnız manifesti değiştirmek etkisizdi
        # (etkinleştirme karar deposunu okur). Eklemeli bir düşürme kararı yazılır.
        _downgrade_decision(cmp_id, note)
    m["purpose"] = "entegrasyon_testi"
    m["purpose_note"] = note[:500]
    m["purpose_marked_at"] = utcnow()
    _write(_root() / Path(cmp_id).name / "manifest.json", m)
    return m


def _downgrade_decision(cmp_id: str, note: str) -> None:
    from app.evals.candidate_decisions import list_decisions, record_decision

    rows = [r for r in list_decisions() if r.get("comparison_id") == cmp_id]
    if not rows or rows[-1]["decision"] in ("yetersiz_kanit", "ret", "kritik_ret"):
        return
    last = rows[-1]
    record_decision(
        {
            **{k: last.get(k) for k in last if k not in ("decision_id", "decided_at")},
            "decision": "yetersiz_kanit",
            "summary": (
                "Sonradan YALNIZ entegrasyon testi işaretlendi → en fazla yetersiz kanıt; "
                f"ana model terfisinde kullanılamaz. {note}"
            )[:500],
        }
    )


def _dimension_report(
    questions: dict[str, dict[str, Any]], scores: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    """Rol → boyut → {ortalama, soru, kritik}: boyutlar AYRI raporlanır (tek skora katılmaz)."""
    out: dict[str, dict[str, Any]] = {}
    for role, per_q in scores.items():
        dims: dict[str, list[dict[str, Any]]] = {}
        for qid, v in per_q.items():
            dims.setdefault(dimension_of(questions[qid]), []).append(v)
        out[role] = {
            d: {
                "mean": float(np.mean([float(x["score"]) for x in vs])),
                "n": len(vs),
                "critical": sum(1 for x in vs if x.get("critical")),
            }
            for d, vs in sorted(dims.items())
        }
    return out


# ── karar ────────────────────────────────────────────────────────────────────


def family_bootstrap(diffs: dict[str, float], *, B: int, seed: int, ci: float) -> dict:
    fams = sorted(diffs)
    vals = np.array([diffs[f] for f in fams], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "point": float(vals.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_families": len(vals),
        "B": B,
        "seed": seed,
        "unit": "family",
    }


def finalize(cmp_id: str) -> dict[str, Any]:
    from app.evals.candidate_decisions import record_decision

    m = _manifest(cmp_id)
    if m["status"] != "reviewed":
        raise CompareError(f"Karar için inceleme tamamlanmalı (durum {m['status']}).")
    crit = _criteria(m["criteria_sha"])["criteria"]
    questions = {q["id"]: q for q in (_read(_root() / cmp_id / "questions.json") or [])}
    auto = _read(_root() / cmp_id / "auto_scores.json") or {}
    sealed = _read(_root() / cmp_id / "sealed_mapping.json") or {}
    review = (_read(_root() / cmp_id / "review.json") or {}).get("scores", {})
    scores: dict[str, dict[str, dict[str, Any]]] = {}  # role → qid → {score, critical}
    for qid in questions:
        for role, v in (auto.get(qid) or {}).items():
            scores.setdefault(role, {})[qid] = v
        for lab, role in (sealed.get(qid) or {}).items():
            scores.setdefault(role, {})[qid] = {**review[qid][lab], "by": "kör inceleme"}
    fam_mean: dict[str, dict[str, float]] = {}
    for role, per_q in scores.items():
        fams: dict[str, list[float]] = {}
        for qid, v in per_q.items():
            fams.setdefault(questions[qid]["family"], []).append(float(v["score"]))
        fam_mean[role] = {f: float(np.mean(x)) for f, x in fams.items()}
    common = sorted(set(fam_mean.get("active", {})) & set(fam_mean.get("candidate", {})))
    diffs = {f: fam_mean["candidate"][f] - fam_mean["active"][f] for f in common}
    crit_cand_only = [
        qid
        for qid, v in scores.get("candidate", {}).items()
        if v.get("critical") and not scores.get("active", {}).get(qid, {}).get("critical")
    ]
    reasons: list[str] = []
    bs = (
        family_bootstrap(
            diffs,
            B=int(crit["bootstrap"]["B"]),
            seed=int(crit["bootstrap"]["seed"]),
            ci=float(crit.get("ci", 0.95)),
        )
        if diffs
        else {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_families": 0}
    )
    margin = float(crit["margin"])
    if crit_cand_only:
        decision = "kritik_ret"
        reasons.append(f"Adaya özgü kritik hata: {crit_cand_only[:5]}")
    elif bs["n_families"] < int(crit["min_families"]):
        decision = "yetersiz_kanit"
        reasons.append(f"Aile sayısı {bs['n_families']} < {crit['min_families']}")
    elif bs["hi"] < -margin:
        decision = "ret"
        reasons.append(f"GA üst sınırı {bs['hi']:.3f} < −{margin}")
    elif bs["lo"] > -margin and bs["point"] >= 0:
        decision = "kabul"
        reasons.append(f"GA alt sınırı {bs['lo']:.3f} > −{margin} ve nokta {bs['point']:.3f} ≥ 0")
    else:
        decision = "yetersiz_kanit"
        reasons.append(f"GA [{bs['lo']:.3f}, {bs['hi']:.3f}] kararsız (tolerans {margin})")
    meta = m.get("candidate_meta") or {}
    conv = meta.get("conversion") or {}
    verified = bool((meta.get("completion") or {}).get("ok")) and bool(conv.get("ok"))
    # Kademe 2 (2026-10-06) E-7: dönüşüm doğrulaması KARŞILAŞTIRILAN digest'e ait olmalı.
    from app.evals.candidate_decisions import norm_digest

    compared = norm_digest(str((m.get("digests") or {}).get("candidate") or ""))
    if verified and norm_digest(str(conv.get("digest") or "")) != compared:
        verified = False
        reasons.append("Dönüşüm doğrulamasının digest'i karşılaştırılan modelinkiyle aynı değil")
    if decision == "kabul" and not verified:
        decision = "yetersiz_kanit"
        reasons.append("Adayın tamamlanma/dönüşüm doğrulaması geçmedi → en fazla yetersiz kanıt")
    integration_only = m.get("purpose") == "entegrasyon_testi"
    if integration_only:
        if decision == "kabul":
            decision = "yetersiz_kanit"
        reasons.append(
            "Yalnız ENTEGRASYON testi olarak işaretli — kalite üstünlüğü ya da ana model terfisi "
            "için kullanılamaz."
        )
    if m["role"] == "development" and m.get("role_requested") == "final":
        reasons.append(m["role_note"])
    base_score = float(np.mean(list(fam_mean["base"].values()))) if fam_mean.get("base") else None
    result = {
        "comparison_id": cmp_id,
        "decision": decision,
        "reasons": reasons,
        "bootstrap": bs,
        "family_means": fam_mean,
        "family_diffs": diffs,
        "model_scores": {r: float(np.mean(list(v.values()))) for r, v in fam_mean.items()},
        "dimensions": _dimension_report(questions, scores),
        "purpose": m.get("purpose", ""),
        "base_reference": base_score,
        "role": m["role"],
        "criteria_sha": m["criteria_sha"],
        "finalized_at": utcnow(),
        "disclaimer": DISCLAIMER,
    }
    _write(_root() / cmp_id / "result.json", result)
    record_decision(
        {
            "candidate_tag": m["models"]["candidate"],
            "candidate_digest": (m.get("digests") or {}).get("candidate", ""),
            "decision": decision,
            "comparison_id": cmp_id,
            "adapter_id": meta.get("adapter_id", ""),
            "criteria_sha": m["criteria_sha"],
            "summary": "; ".join(reasons)[:500],
            "active_tag": m["models"]["active"],
            "role": m["role"],
        }
    )
    if not integration_only:
        _sync_registry(meta.get("adapter_id", ""), decision)
    m["status"] = "decided"
    _write(_root() / cmp_id / "manifest.json", m)
    if m.get("role_requested") == "final":
        _log_final_access({"kind": "result", "set_sha": m["set_sha"], "comparison_id": cmp_id})
    return result


def _sync_registry(adapter_id: str, decision: str) -> None:
    """Kabul → eval_passed (etkinleştirmeye uygun); ret/kritik → rejected. Production YOK."""
    if not adapter_id:
        return
    from app.feedback.model_activation import registry_path
    from app.lora.adapter_registry import AdapterRegistry, AdapterStatus

    reg = AdapterRegistry(registry_path())
    rows = reg.list_adapters()
    target = next((r for r in rows if r.adapter_id == adapter_id), None)
    if target is None:
        return
    if decision == "kabul" and target.status is AdapterStatus.CANDIDATE:
        target.status = AdapterStatus.EVAL_PASSED
        reg._write_all(rows)
    elif decision in ("ret", "kritik_ret"):
        reg.reject(adapter_id, f"karşılaştırma kararı: {decision}")


def mark_interrupted(cmp_id: str, why: str) -> None:
    """Yarıda kesilen üretimi işaretle: geçerli karşılaştırma değildir, yeniden koşulur."""
    m = _manifest(cmp_id)
    if m.get("status") == "created":
        m.update(status="kesildi", invalid_reason=f"Üretim yarıda kesildi: {why}")
        _write(_root() / Path(cmp_id).name / "manifest.json", m)


def result(cmp_id: str) -> dict[str, Any]:
    m = _manifest(cmp_id)
    res = _read(_root() / cmp_id / "result.json")
    if m.get("role_requested") == "final" and res is not None:
        _log_final_access({"kind": "view", "set_sha": m["set_sha"], "comparison_id": cmp_id})
    return {"manifest": m, "result": res}


def list_comparisons() -> list[dict[str, Any]]:
    root = _root()
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob("cmp_*/manifest.json")):
        m = _read(p)
        if isinstance(m, dict):
            out.append(
                {
                    k: m.get(k)
                    for k in (
                        "comparison_id",
                        "created_at",
                        "role",
                        "status",
                        "models",
                        "n_questions",
                        "n_families",
                        "purpose",
                    )
                }
            )
    return sorted(out, key=lambda r: str(r.get("created_at")), reverse=True)
