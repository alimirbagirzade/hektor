"""local_judge.py — gece döngüsünün YEREL hakemi (Ollama): öğrenme adayını tutarlılık için okur.

Tasarım: ``docs/TASARIM_GECE_DONGUSU.md``. Bulut K2'nin (``app.cloud.judge``) gözetimsiz, yerel
karşılığıdır; bulut KULLANMAZ (Ollama Cloud etiketleri de reddedilir — uzakta çalışırlar):

- Girdi: soru + aday hedef metin + yerel deterministik kontrol özeti + (yerel olduğu için)
  turdaki kaynak parçalarının kısaltılmış metni.
- Çıktı: ``tutarli | supheli | belirsiz`` + gerekçe. ``supheli``/``belirsiz`` → aday
  KARANTİNAYA alınır (``LearningService.set_quarantine``); eğitime girmez. ``tutarli`` adayın
  durumunu DEĞİŞTİRMEZ — hakem onayı doğrulama ya da eğitim onayı sayılmaz.
- Hakemin metni hiçbir yere eğitim verisi olarak yazılmaz; yalnız ayrı kayıtta durur.
- Determinizm: ``temperature=0`` + ``seed`` (Kural 6). Aynı hedef + model + istem sürümü için
  ikinci kez sorulmaz (kayıt yeniden kullanılır).

Bilinen sınır: yerel hakem çoğu kurulumda cevabı üreten modelle AYNI modeldir; kendi cevabını
kayırma eğilimi vardır. Bu yüzden ``tutarli`` hiçbir şeyi onaylamaz, yalnız şüphe karantinaya
alır (tek yönlü, temkinli kapı).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.feedback.chat_store import ChatStore, utcnow

PROMPT_VERSION = "lj-1"
VERDICTS = ("tutarli", "supheli", "belirsiz")
QUARANTINE_VERDICTS = ("supheli", "belirsiz")
JUDGE_STATUSES = ("eligible", "review")
SEED = 42
_SOURCE_CHARS = 1200
_MAX_SOURCES = 4

SYSTEM = (
    "Sen titiz ve şüpheci bir hakemsin. Görevin bir eğitim ADAYI cevabının soruyla ve verilen "
    "kaynaklarla tutarlı olup olmadığını denetlemek. Yatırım tavsiyesi verme. Emin değilsen "
    "'supheli' de."
)


class LocalJudgeError(RuntimeError):
    """Yerel hakem çalıştırılamadı (kullanıcıya/rapora gösterilir)."""


def judge_dir() -> Path:
    d = get_settings().state_dir / "nightly" / "judge"
    d.mkdir(parents=True, exist_ok=True)
    return d


def judge_model() -> str:
    s = get_settings()
    return (s.nightly_judge_model or "").strip() or s.llm_model


def model_blocker(model: str) -> str:
    from app.cloud.policy import CLOUD_TAG_SUFFIXES

    if any(model.lower().endswith(suf) for suf in CLOUD_TAG_SUFFIXES):
        return (
            f"Hakem modeli '{model}' bir Ollama Cloud etiketi — uzakta çalışır; gözetimsiz "
            "bulut döngüsü yasak. Yerel bir model seçin (HEKTOR_NIGHTLY_JUDGE_MODEL)."
        )
    return ""


def build_prompt(cand: dict[str, Any], turn: dict[str, Any]) -> str:
    checks = [
        f"- {c.get('kind')}: {c.get('status') or c.get('label') or ''} {c.get('detail') or ''}"
        for c in ((cand.get("verification") or {}).get("checks") or [])
    ]
    sources = []
    for i, src in enumerate((turn.get("sources") or [])[:_MAX_SOURCES], 1):
        text = re.sub(r"\s+", " ", str(src.get("text") or "")).strip()[:_SOURCE_CHARS]
        sources.append(f"[{i}] {src.get('title', '')} (s. {src.get('page', '?')}): {text}")
    who = "kullanıcı düzeltmesi" if cand.get("kind") == "correct" else "model cevabı"
    return (
        "Aşağıdaki ADAY CEVAP bir trading-araştırma asistanının eğitim verisine girecek. "
        "Soruyla tutarlı mı, kaynaklarla çelişiyor mu, kaynakta olmayan kesin iddia/sayı "
        "uyduruyor mu, maliyet/look-ahead/örneklem varsayımlarını yok sayıyor mu, yatırım "
        "tavsiyesi dili var mı? İlk satıra YALNIZ 'KARAR: tutarli' ya da 'KARAR: supheli' yaz; "
        "ikinci satırdan itibaren en fazla 5 madde gerekçe ver.\n\n"
        f"SORU:\n{turn.get('question', '')}\n\n"
        f"ADAY CEVAP ({who}, alan {cand.get('domain', '')}):\n{cand.get('target_text', '')}\n\n"
        "KAYNAKLAR:\n" + ("\n".join(sources) or "- (kaynak yok)") + "\n\n"
        "YEREL DETERMİNİSTİK KONTROLLER:\n" + ("\n".join(checks) or "- (yok)")
    )


def parse_verdict(text: str) -> tuple[str, str]:
    """(karar, gerekçe). Biçim tutmazsa ``belirsiz`` (temkinli: karantina)."""
    m = re.search(r"KARAR\s*:\s*\**\s*(tutarl[ıi]|şüpheli|supheli)", text or "", re.IGNORECASE)
    if not m:
        return "belirsiz", "Hakem cevabı beklenen biçimde değil (KARAR satırı yok)."
    v = m.group(1).lower().replace("ı", "i").replace("ş", "s").replace("ü", "u")
    reason = (text[m.end() :] or "").strip()[:2000]
    return (v if v in VERDICTS else "belirsiz"), reason


def _key(cand: dict[str, Any], model: str) -> str:
    blob = f"{cand['candidate_id']}|{cand.get('target_sha', '')}|{model}|{PROMPT_VERSION}"
    return "lj_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _read(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def latest_for(candidate_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Aday başına son yerel hakem kaydı (salt-okuma; liste ekranı için)."""
    want = set(candidate_ids)
    out: dict[str, dict[str, Any]] = {}
    for p in judge_dir().glob("lj_*.json"):
        r = _read(p)
        if not r or r.get("candidate_id") not in want:
            continue
        prev = out.get(r["candidate_id"])
        if prev is None or str(r.get("created_at")) >= str(prev.get("created_at")):
            out[r["candidate_id"]] = r
    return out


Generate = Callable[[str, str], str]


def _ollama_generate(model: str) -> Generate:
    from app.brain.local_llm import LocalLLM

    llm = LocalLLM(model=model)

    def gen(prompt: str, system: str) -> str:
        return llm.generate(
            prompt, system=system, temperature=0.0, max_tokens=600, seed=SEED, timeout=300
        )

    return gen


def judge_candidate(
    cand: dict[str, Any],
    turn: dict[str, Any],
    *,
    model: str,
    generate: Generate,
) -> dict[str, Any]:
    """Tek adayı yargıla ve kaydı yaz (karantina KARARINI uygulamaz — ``run_judging`` uygular)."""
    path = judge_dir() / f"{_key(cand, model)}.json"
    cached = _read(path)
    if cached and cached.get("status") == "done":
        return {**cached, "cached": True}
    prompt = build_prompt(cand, turn)
    rec: dict[str, Any] = {
        "id": path.stem,
        "candidate_id": cand["candidate_id"],
        "target_sha": cand.get("target_sha", ""),
        "revision": cand.get("revision"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "seed": SEED,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "created_at": utcnow(),
        "note": "Yerel hakem — doğrulama/eğitim onayı DEĞİLDİR; yalnız şüpheyi karantinaya alır.",
    }
    try:
        text = generate(prompt, SYSTEM)
    except Exception as exc:  # Ollama yok/zaman aşımı: aday dokunulmadan kalır
        rec.update(status="failed", error=str(exc)[:500], verdict="", reason="")
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        return rec
    verdict, reason = parse_verdict(text)
    rec.update(status="done", verdict=verdict, reason=reason, raw=text[:8000])
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def candidates_to_judge(store: ChatStore) -> list[dict[str, Any]]:
    from app.feedback.learning import quarantine_active

    out = []
    for c in store.list_candidates():
        if c["status"] not in JUDGE_STATUSES or quarantine_active(c):
            continue
        lifted = (c.get("quarantine") or {}).get("lifted") or {}
        if lifted and lifted.get("target_sha") == c.get("target_sha"):
            continue  # insan bu hedef için karar verdi
        turn = store.get_turn(c["turn_id"])
        if turn is None or store.is_test_turn(turn):
            continue  # test sohbeti hiçbir sürüme girmez → yargılamaya gerek yok
        out.append(c)
    return out


def run_judging(
    *,
    max_candidates: int | None = None,
    generate: Generate | None = None,
    model: str | None = None,
    available: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Uygun/inceleme adaylarını yerel hakemle oku; şüpheli/belirsiz → karantina."""
    from app.feedback.learning import LearningService

    s = get_settings()
    model = model or judge_model()
    blocker = model_blocker(model)
    if blocker:
        return {"ran": False, "skipped": blocker, "model": model}
    if generate is None:
        from app.brain.local_llm import LocalLLM

        alive = available or LocalLLM(model=model).available
        if not alive():
            return {
                "ran": False,
                "skipped": "Ollama'ya ulaşılamıyor — adaylar yargılanmadı (dokunulmadı).",
                "model": model,
            }
        generate = _ollama_generate(model)
    store = ChatStore()
    svc = LearningService(store)
    limit = s.nightly_judge_max if max_candidates is None else max_candidates
    todo = candidates_to_judge(store)
    counts = dict.fromkeys(VERDICTS, 0) | {"failed": 0, "cached": 0}
    quarantined: list[dict[str, Any]] = []
    fresh = 0
    left = 0
    for cand in todo:
        cached = _read(judge_dir() / f"{_key(cand, model)}.json")
        is_cached = bool(cached and cached.get("status") == "done")
        if not is_cached:
            if fresh >= max(0, limit):
                left += 1
                continue
            fresh += 1
        turn = store.get_turn(cand["turn_id"]) or {}
        rec = judge_candidate(cand, turn, model=model, generate=generate)
        if rec.get("cached"):
            counts["cached"] += 1
        if rec["status"] != "done":
            counts["failed"] += 1
            if counts["failed"] >= 3:  # Ollama düştü: geceyi boşuna zaman aşımıyla doldurma
                break
            continue
        if not rec.get("cached"):
            counts[rec["verdict"]] += 1
        if rec["verdict"] in QUARANTINE_VERDICTS:
            after = svc.set_quarantine(
                cand["candidate_id"],
                verdict=rec["verdict"],
                reason=rec["reason"],
                judge_id=rec["id"],
                model=model,
            )
            if after["status"] == "quarantined":
                quarantined.append(
                    {
                        "candidate_id": cand["candidate_id"],
                        "verdict": rec["verdict"],
                        "reason": rec["reason"][:300],
                    }
                )
    return {
        "ran": True,
        "model": model,
        "pending_total": len(todo),
        "judged_fresh": fresh,
        "left_for_next_night": left,
        "counts": counts,
        "quarantined": quarantined,
        "note": "Yerel hakem tek yönlüdür: şüpheyi karantinaya alır, hiçbir adayı onaylamaz.",
    }
