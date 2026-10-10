"""loop_state.py — tek tıklamalı döngünün "Bekleyen kararlar" görünümü (Seçenek A).

Tasarım: ``docs/TASARIM_SUREKLI_DONGU.md``. Döngü:

    araştırma → kart → aday veri → [eğitime hazır mı?] → Kademe 2 + İNSAN onayı → eğitim
    → (tek tık) Ollama'ya hazırla → karşılaştır → (tık) K1 bulut hakem → kör İNSAN incelemesi
    → karar → başa

Bu modül SALT-OKUMADIR: durumu ayrı bir dosyada tutmaz, her çağrıda diskteki kayıtlardan
(kolay akış hazırlığı, adapter dizini, aday işleri, karşılaştırma manifestleri, bulut hakem
kayıtları, öğrenme adayları) YENİDEN türetir — sunucu yeniden açılınca kaybolan bir durum yoktur.
Hiçbir adımı başlatmaz, onay vermez, eğitim başlatmaz (Kural 8); yalnız sıradaki insan kararını
ve tıklanacak yeri gösterir.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

from app.config import get_settings
from app.feedback.chat_store import utcnow

NOTE = (
    "Bekleyen kararlar diskteki kayıtlardan türetilir; bu ekran hiçbir işi başlatmaz. Eğitim "
    "onayı, bulut gönderimi ve terfi her zaman ayrı insan tıklamasıdır."
)


def _read(p: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _training_items() -> list[dict[str, Any]]:
    from app.training.easy_train import readiness

    r = readiness()
    bad = [i for i in r["items"] if not i["ok"]]
    if not bad:
        return [
            {
                "key": "egitim_onayi",
                "stage": "egitim",
                "who": "insan",
                "title": "Eğitim hazır — ağırlık onayı + tek tık eğitim onayı sizde",
                "detail": f"veri özeti {str(r.get('data_sha256', ''))[:12]} · tüm kapılar geçti",
                "where": "kolay_egitim",
            }
        ]
    if [i["key"] for i in bad] == ["kademe2"]:
        return [
            {
                "key": "kademe2",
                "stage": "egitim",
                "who": "insan",
                "title": "Eğitim öncesi Kademe 2 derin av gerekli (denetimli)",
                "detail": bad[0]["detail"],
                "where": "kolay_egitim",
            }
        ]
    return []  # veri/ağır iş/kod engeli: döngü bu adımda değil (ayrıntı kolay akışta)


def _latest_adapter_items() -> list[dict[str, Any]]:
    """Son eğitilen adapter tamamlanmış ama hiç Ollama'ya hazırlanmamışsa tek-tık önerisi.

    Tam tamamlanma doğrulaması (ağırlık özeti vb.) aday hattında yapılır; burada yalnız ucuz
    ``run_complete`` sinyali okunur (sayfa her açılışta büyük dosya özetlemesin)."""
    from app.training import candidate_jobs as cj

    d = get_settings().adapters_dir
    adapters = [
        p for p in (d.iterdir() if d.is_dir() else []) if (p / "adapter_config.json").is_file()
    ]
    if not adapters:
        return []
    last = max(adapters, key=lambda p: p.stat().st_mtime)
    done = _read(last / "run_complete.json") or {}
    if not done or done.get("global_step") != done.get("max_steps"):
        return []
    jobs = cj.list_jobs(adapter=last.name, limit=20)
    if any(j["kind"] == "conversion" and j["status"] in ("done", *cj.ACTIVE) for j in jobs):
        return []
    return [
        {
            "key": f"hazirla:{last.name}",
            "stage": "aday",
            "who": "insan",
            "title": f"Eğitim bitti ({last.name}) — Ollama'ya hazırla + karşılaştır (tek tık)",
            "detail": "Dönüşüm doğrulanınca karşılaştırma kendiliğinden başlar.",
            "where": "aday_hatti",
            "adapter": last.name,
            "suggested_tag": cj.suggest_tag(last.name),
        }
    ]


def _job_items() -> list[dict[str, Any]]:
    from app.training import candidate_jobs as cj

    out = []
    run = cj.running_job()
    if run:
        out.append(
            {
                "key": f"is:{run['job_id']}",
                "stage": "aday",
                "who": "sistem",
                "title": f"Sürüyor: {run.get('kind_label', run['kind'])} ({run['adapter']})",
                "detail": str((run.get("progress") or {}).get("label") or ""),
                "where": "aday_hatti",
            }
        )
    for j in cj.list_jobs(limit=20):
        chain = j.get("chain") or {}
        if j["kind"] == "conversion" and chain.get("error"):
            out.append(
                {
                    "key": f"zincir:{j['job_id']}",
                    "stage": "aday",
                    "who": "insan",
                    "title": f"Karşılaştırma kendiliğinden başlayamadı ({j['adapter']})",
                    "detail": chain["error"],
                    "where": "aday_hatti",
                }
            )
    return out


def _comparison_items(cloud_on: bool) -> list[dict[str, Any]]:
    from app.cloud import judge
    from app.evals.candidate_compare import ai_review, list_comparisons

    out = []
    for c in list_comparisons():
        cid, st = c["comparison_id"], c.get("status")
        if st == "generated":
            k1 = judge.records("k1", cid)
            has_ai = ai_review(cid) is not None
            k1_txt = (
                "AI incelemesi kayıtlı"
                if has_ai
                else f"bulut hakem: {sum(r['status'] == 'done' for r in k1)} parça bitti"
                if k1
                else "bulut hakem gönderilmedi"
                if cloud_on
                else "bulut kapalı"
            )
            out.append(
                {
                    "key": f"kor:{cid}",
                    "stage": "degerlendirme",
                    "who": "insan",
                    "title": f"Değerlendirme hazır — kör inceleme bekliyor ({cid})",
                    "detail": f"{k1_txt} · aday {(c.get('models') or {}).get('candidate', '')}",
                    "where": "karsilastirma",
                    "comparison_id": cid,
                    "k1_available": cloud_on and not has_ai and c.get("role") != "final",
                }
            )
        elif st == "reviewed":
            out.append(
                {
                    "key": f"karar:{cid}",
                    "stage": "degerlendirme",
                    "who": "insan",
                    "title": f"Kör inceleme bitti — karar (finalize) bekliyor ({cid})",
                    "detail": "Karar yalnız insan puanından hesaplanır; AI/bulut puanı girmez.",
                    "where": "karsilastirma",
                    "comparison_id": cid,
                }
            )
    return out


def _k2_items(cloud_on: bool) -> list[dict[str, Any]]:
    from app.cloud import judge
    from app.feedback.chat_store import ChatStore

    if not cloud_on:
        return []
    review = ChatStore().list_candidates(status="review")
    if not review:
        return []
    checked = judge.k2_latest([c["candidate_id"] for c in review])
    unchecked = [c for c in review if c["candidate_id"] not in checked]
    if not unchecked:
        return []
    return [
        {
            "key": "k2",
            "stage": "ogrenme",
            "who": "insan",
            "title": f"{len(unchecked)} öğrenme adayı inceleme bekliyor (bulut kontrolü yapılmadı)",
            "detail": "Bulut kontrolü aday başına ayrı tıklamadır; kabul/ret sizde.",
            "where": "ogrenme_havuzu",
        }
    ]


def _nightly_items() -> list[dict[str, Any]]:
    """Gece döngüsü (docs/TASARIM_GECE_DONGUSU.md): karantina kararları + CSV adayları."""
    from datetime import UTC, datetime

    from app.feedback.chat_store import ChatStore
    from app.orchestration import nightly

    items: list[dict[str, Any]] = []
    q = ChatStore().list_candidates(status="quarantined")
    if q:
        items.append(
            {
                "key": "karantina",
                "stage": "ogrenme",
                "who": "insan",
                "title": f"{len(q)} öğrenme adayı karantinada (yerel hakem şüphelendi)",
                "detail": "Eğitime girmezler. İnceleyin; gerekçeyle karantinayı kaldırabilirsiniz.",
                "where": "ogrenme_havuzu",
            }
        )
    last = nightly.latest()
    if last is None:
        return items
    try:
        age_h = (
            datetime.now(UTC) - datetime.fromisoformat(str(last.get("started_at")))
        ).total_seconds() / 3600
    except (TypeError, ValueError):
        age_h = 0.0
    if age_h > 36:
        items.append(
            {
                "key": "gece_gecikti",
                "stage": "gece",
                "who": "sistem",
                "title": f"Gece döngüsü {age_h:.0f} saattir koşmadı",
                "detail": "Görev zamanlayıcıyı kontrol edin (scripts/install-nightly-task.ps1).",
                "where": "ogrenme_havuzu",
            }
        )
    csv = (last.get("steps") or {}).get("csv") or {}
    fresh = [
        e
        for f in csv.get("files", [])
        for e in (f.get("report") or {}).get("selected", [])
        if e.get("verdict") == "aday_oos_tutarli"
    ]
    if fresh:
        items.append(
            {
                "key": "csv_aday",
                "stage": "gece",
                "who": "insan",
                "title": f"{len(fresh)} CSV adayı örneklem dışında tutarlı — denetim bekliyor",
                "detail": "ADAYDIR, hazır değil: /backtest-auditor + final dönem (bir kez) sizde. "
                + (last.get("report_md") or ""),
                "where": "ogrenme_havuzu",
            }
        )
    return items


def pending_decisions(section_timeout_s: float = 10.0) -> dict[str, Any]:
    """Döngünün bekleyen insan kararları (salt-okuma). Bir bölüm hata verirse diğerleri sürer."""
    from app.cloud.second_opinion import status as cloud_status

    try:
        cloud_on = bool(cloud_status()["enabled"])
    except Exception:
        cloud_on = False
    sections: list[tuple[str, Callable[[], list[dict[str, Any]]]]] = [
        ("egitim", _training_items),
        ("aday", _latest_adapter_items),
        ("isler", _job_items),
        ("degerlendirme", lambda: _comparison_items(cloud_on)),
        ("ogrenme", lambda: _k2_items(cloud_on)),
        ("gece", _nightly_items),
    ]
    # Bölümler paralel ve SÜRE SINIRLI okunur: biri takılırsa (git/SQLite/dosya kilidi) kutu yine
    # döner, takılan bölüm "zaman aşımı" diye görünür (2026-10-10: arka plan döngüleri açık web
    # sunucusunda uç bir kez 2 dk+ yanıtsız kaldı; kök neden bulunamadı).
    pool = ThreadPoolExecutor(max_workers=len(sections), thread_name_prefix="loop_state")
    futures = {name: pool.submit(fn) for name, fn in sections}
    done, _ = wait(futures.values(), timeout=section_timeout_s)
    pool.shutdown(wait=False, cancel_futures=True)
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, fut in futures.items():
        if fut not in done:
            errors.append(f"{name}: zaman aşımı ({section_timeout_s:.0f} sn) — sonra yenileyin")
            continue
        try:
            items.extend(fut.result())
        except Exception as exc:  # bir bölüm okunamazsa ekran yine açılır, sebep görünür
            errors.append(f"{name}: {exc}"[:300])
    return {
        "items": items,
        "n_human": sum(1 for i in items if i["who"] == "insan"),
        "cloud_enabled": cloud_on,
        "errors": errors,
        "generated_at": utcnow(),
        "note": NOTE,
    }
