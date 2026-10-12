"""training_loop.py — kontrollü aday eğitim döngüsünün defteri ve "sonraki eğitim" kapısı.

``storage/training_loop_policy.md`` kurallarının KOD karşılığı (eskiden yalnız Codex heartbeat'in
elle yazdığı ``training_loop_checkpoint.json`` vardı; hiçbir başlatma yolu onu okumuyordu):

1. **Kayıt**      : her başlatılan aday (``register``; detached başlatıcı çağırır) ya da koşan
                    eğitimin durum dosyasından (``sync_from_status``) defterde bir satır olur.
2. **Bitiş**      : ``record_completion`` — ``candidate_checks.verify_run_completion``
                    (süreç sonucu, adım, ağırlık dosyası + özeti, temel model, veri özeti, onay
                    kaydı) + işletim sisteminin bildirdiği ÇIKIŞ KODU (``exit_witness``). Klasör
                    ya da ``run_complete.json`` varlığı tek başına başarı SAYILMAZ.
3. **Geliştirme değerlendirmesi**: ``record_dev_eval`` — mevcut LLM-30 koşusunun (``llm30_run``)
                    rubrikle puanlanmış ``raw.jsonl``'i, ÖNCEKİ modelin aynı setteki puanlı
                    cevaplarıyla ``rubric_gate`` üzerinden eşlenmiş karşılaştırılır. Aday
                    cevaplarının bu adapter'ın AYNI ağırlıklarından geldiği manifestten
                    doğrulanır. Geçti = kapı ``insan_onayi_bekliyor`` (kritik hata yok, gerileme
                    yok, kanıt yeterli) + bayraklı cevap sayısı önceki modelden fazla değil.
                    Bu bir TERFİ kararı değildir (terfi insanda; final takımı ayrıdır).
4. **Sonraki eğitim kapısı** (``next_training_blockers``): son aday için bitiş doğrulanmamış,
                    geliştirme değerlendirmesi kaydedilmemiş ya da geçmemişse yeni eğitim
                    BAŞLAMAZ. Aynı veri özetiyle (``data_sha256``) daha önce eğitilmiş bir aday
                    varsa sıfırdan yeniden eğitim BAŞLAMAZ (yeni doğrulanmış veri gerekir).
                    Başarısızlık/regresyondan sonra devam etmek yalnız gerekçeli ve TAZE insan
                    onayıyla (``release``) mümkündür; aynı-veri kuralının istisnası yoktur.

Hiçbir fonksiyon eğitim, değerlendirme ya da dönüşüm BAŞLATMAZ; yalnız kayıt okur/yazar.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings

_ADAPTER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
RELEASE_ACTION = "train_loop_release:"


class TrainingLoopError(ValueError):
    """Kullanıcıya gösterilecek iş kuralı hatası."""


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def loop_dir() -> Path:
    d = get_settings().root / "storage" / "training_loop"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(adapter: str) -> Path:
    if not _ADAPTER_RE.match(adapter or ""):
        raise TrainingLoopError(f"Geçersiz adapter adı: {adapter!r}")
    return loop_dir() / f"{adapter}.json"


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(path: Path, value: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get(adapter: str) -> dict[str, Any] | None:
    rec = _read(_path(adapter))
    return rec if isinstance(rec, dict) else None


def entries() -> list[dict[str, Any]]:
    out = []
    for p in loop_dir().glob("*.json"):
        rec = _read(p)
        if isinstance(rec, dict) and rec.get("adapter"):
            out.append(rec)
    out.sort(key=lambda r: (str(r.get("started_at") or ""), str(r.get("registered_at") or "")))
    return out


# ── kayıt ────────────────────────────────────────────────────────────────────

_FIELDS = (
    "data_sha256",
    "recipe_sha",
    "approval_id",
    "started_at",
    "base_model",
    "profile",
    "pid",
    "pid_create_time",
)


def register(adapter: str, *, source: str, **fields: Any) -> dict[str, Any]:
    """Adayı deftere ekle (idempotent). Var olan alanlar EZİLMEZ, eksikler tamamlanır."""
    path = _path(adapter)
    rec = get(adapter) or {
        "adapter": adapter,
        "registered_at": utcnow(),
        "source": source,
        "completion": None,
        "dev_eval": None,
        "release": None,
    }
    for key in _FIELDS:
        if rec.get(key) in (None, "") and fields.get(key) not in (None, ""):
            rec[key] = fields[key]
    _write(path, rec)
    return rec


def sync_from_status() -> dict[str, Any] | None:
    """Durum dosyasındaki (koşan ya da bitmiş) adayı deftere al — bu modülden önce başlatılan
    eğitimler (ör. hektor_lora_full_20261011) de döngünün parçası olsun."""
    from app.training.detached_launch import read_detached_training_status

    st = read_detached_training_status(get_settings().root)
    name = str(st.get("adapter") or "")
    if not _ADAPTER_RE.match(name):
        return None
    return register(name, source="train_status", **{k: st.get(k) for k in _FIELDS})


# ── bitiş doğrulaması ───────────────────────────────────────────────────────


def record_completion(adapter: str) -> dict[str, Any]:
    """Bitişi doğrula ve SONUCU kaydet (başarısız sonuç da kaydedilir)."""
    from app.training.candidate_checks import recipe_for_adapter, verify_run_completion
    from app.training.exit_witness import read_exit

    rec = get(adapter) or sync_from_status()
    if rec is None or rec.get("adapter") != adapter:
        raise TrainingLoopError(f"{adapter} eğitim döngüsü defterinde yok.")
    res = verify_run_completion(adapter, recipe_for_adapter(adapter))
    rec["completion"] = {
        "verified_at": utcnow(),
        "ok": bool(res.get("ok")),
        "checks": res.get("checks", []),
        "adapter_sha256": res.get("adapter_sha256", ""),
        "exit": read_exit(adapter),
        "failed": [c["key"] for c in res.get("checks", []) if not c.get("ok")],
    }
    # Bitiş yeniden doğrulandıysa eski değerlendirme o ağırlıklara ait olmayabilir.
    dev = rec.get("dev_eval") or {}
    if dev and dev.get("adapter_sha256") != rec["completion"]["adapter_sha256"]:
        rec["dev_eval"] = {**dev, "passed": False, "stale": True}
    _write(_path(adapter), rec)
    return rec


# ── geliştirme değerlendirmesi ──────────────────────────────────────────────


def _raw_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def record_dev_eval(
    adapter: str,
    *,
    candidate_raw: Path,
    candidate_condition: str,
    reference_raw: Path,
    reference_condition: str,
    min_repeats: int = 2,
) -> dict[str, Any]:
    """Puanlanmış LLM-30 cevaplarından aday ↔ önceki model kapısı; sonucu kaydet."""
    from app.evals.rubric_gate import AWAITING_HUMAN, gate, records_from_llm30_raw

    rec = get(adapter)
    if rec is None:
        raise TrainingLoopError(f"{adapter} eğitim döngüsü defterinde yok.")
    comp = rec.get("completion") or {}
    if not comp.get("ok"):
        raise TrainingLoopError(
            "Önce bitiş doğrulanmalı (`uv run hektor egitim-dongusu bitis "
            f"{adapter}`); doğrulanmamış adayın değerlendirmesi kaydedilmez."
        )
    problems: list[str] = []
    cand_rows = [r for r in _raw_rows(candidate_raw) if r.get("condition") == candidate_condition]
    ref_rows = [r for r in _raw_rows(reference_raw) if r.get("condition") == reference_condition]
    if not cand_rows or not ref_rows:
        raise TrainingLoopError("Aday ya da referans koşulunda hiç cevap yok.")
    if {r.get("adapter_id") for r in cand_rows} != {adapter}:
        problems.append("aday cevapları bu adapter'a ait değil (adapter_id)")
    manifest = _read(candidate_raw.parent / "manifest.json") or {}
    if (manifest.get("adapter") or {}).get("adapter_sha256") != comp.get("adapter_sha256"):
        problems.append(
            "değerlendirme manifestindeki ağırlık özeti doğrulanan ağırlıklarla aynı değil"
        )
    ref_models = sorted({str(r.get("model") or "") for r in ref_rows})
    if len(ref_models) != 1 or not ref_models[0]:
        problems.append("referans koşulunda tek bir önceki model yok")
    if any(str(r.get("split")) != "validation" for r in cand_rows + ref_rows):
        problems.append("yalnız geliştirme (validation) seti kullanılabilir")
    cand_q = {r["question_id"] for r in cand_rows}
    ref_q = {r["question_id"] for r in ref_rows}
    if cand_q != ref_q:
        problems.append(f"soru kümeleri farklı ({len(cand_q)} ↔ {len(ref_q)})")
    records = [
        {**r, "model": "aday"}
        for r in records_from_llm30_raw(candidate_raw)
        if r["model"] == candidate_condition
    ] + [
        {**r, "model": "onceki"}
        for r in records_from_llm30_raw(reference_raw)
        if r["model"] == reference_condition
    ]
    res = gate(records, candidate="aday", reference="onceki", min_repeats=min_repeats)
    cand_flag = sum(1 for r in cand_rows if r.get("flags"))
    ref_flag = sum(1 for r in ref_rows if r.get("flags"))
    if cand_flag > ref_flag:
        problems.append(f"bayraklı cevap arttı ({ref_flag} → {cand_flag}: kesilme/tekrar/boş)")
    passed = res["decision"] == AWAITING_HUMAN and res.get("comparison") is not None
    passed = passed and not problems
    rec["dev_eval"] = {
        "recorded_at": utcnow(),
        "adapter_sha256": comp.get("adapter_sha256"),
        "candidate_raw": str(candidate_raw),
        "candidate_raw_sha256": _sha_file(candidate_raw),
        "candidate_condition": candidate_condition,
        "reference_raw": str(reference_raw),
        "reference_raw_sha256": _sha_file(reference_raw),
        "reference_condition": reference_condition,
        "reference_model": ref_models[0] if len(ref_models) == 1 else ref_models,
        "dataset_version": manifest.get("dataset_version", ""),
        "decision": res["decision"],
        "reasons": res["reasons"],
        "comparison": res.get("comparison"),
        "models": {
            k: {
                kk: v.get(kk)
                for kk in (
                    "n_questions",
                    "mean_score",
                    "critical_answers",
                    "n_unscored",
                    "repeats_min",
                )
            }
            for k, v in res["models"].items()
        },
        "flagged": {"aday": cand_flag, "onceki": ref_flag},
        "problems": problems,
        "passed": passed,
        "note": "Geliştirme seti kapısı — terfi kararı DEĞİLDİR; son adım insan.",
    }
    _write(_path(adapter), rec)
    return rec


# ── sonraki eğitim kapısı ───────────────────────────────────────────────────


def _trained_data_shas(exclude: str) -> dict[str, str]:
    """Daha önce eğitimi BAŞLATILMIŞ veri özetleri → adapter (defter + adapter arşivleri)."""
    from app.training.detached_launch import read_final_status

    out: dict[str, str] = {}
    for rec in entries():
        if rec["adapter"] != exclude and rec.get("data_sha256"):
            out.setdefault(str(rec["data_sha256"]), rec["adapter"])
    adapters = get_settings().adapters_dir
    if adapters.is_dir():
        for d in sorted(adapters.iterdir()):
            if d.name == exclude or not d.is_dir():
                continue
            sha = str(read_final_status(d.name, get_settings().root).get("data_sha256") or "")
            if sha:
                out.setdefault(sha, d.name)
    return out


def _released(rec: dict[str, Any]) -> bool:
    rel = rec.get("release") or {}
    comp = rec.get("completion") or {}
    return bool(rel.get("approval_id")) and rel.get("adapter_sha256", "") == comp.get(
        "adapter_sha256", ""
    )


def previous_candidate(exclude: str = "") -> dict[str, Any] | None:
    rows = [r for r in entries() if r["adapter"] != exclude]
    return rows[-1] if rows else None


def next_training_blockers(*, data_sha: str | None, adapter_name: str = "") -> list[str]:
    """Yeni (sıfırdan) eğitime engel gerekçeleri; boş liste = döngü kuralları izin veriyor."""
    problems: list[str] = []
    try:
        sync_from_status()
    except Exception as exc:  # durum okunamıyorsa döngü durumu kanıtlanamaz → kapalı kal
        return [f"Eğitim döngüsü durumu okunamadı: {exc}"]
    prev = previous_candidate(exclude=adapter_name)
    if prev is not None and not _released(prev):
        name = prev["adapter"]
        comp = prev.get("completion") or {}
        dev = prev.get("dev_eval") or {}
        if not comp:
            problems.append(
                f"Önceki aday {name} için bitiş doğrulaması yok — "
                f"`uv run hektor egitim-dongusu bitis {name}`."
            )
        elif not comp.get("ok"):
            problems.append(
                f"Önceki aday {name} bitişi doğrulanamadı ({', '.join(comp.get('failed', []))}) "
                "— başarısız koşudan sonra yeni eğitim yalnız gerekçeli insan onayıyla."
            )
        elif not dev:
            problems.append(
                f"Önceki aday {name} geliştirme değerlendirmesinden geçmedi (kayıt yok) — "
                "LLM-30 koşusu + rubrik puanı + `uv run hektor egitim-dongusu degerlendirme`."
            )
        elif not dev.get("passed"):
            why = "; ".join((dev.get("problems") or []) + (dev.get("reasons") or []))[:400]
            problems.append(
                f"Önceki aday {name} geliştirme değerlendirmesi GEÇMEDİ "
                f"({dev.get('decision')}: {why}) — başarısızlık/regresyon sonrası yeni eğitim "
                "engelli."
            )
    if data_sha:
        seen = _trained_data_shas(exclude=adapter_name)
        if data_sha in seen:
            problems.append(
                f"Bu veri özetiyle ({data_sha[:12]}) {seen[data_sha]} zaten eğitildi — aynı "
                "veriyle sıfırdan yeniden eğitim yok; yeni doğrulanmış veri gerekir."
            )
    else:
        problems.append("Eğitim verisinin özeti bilinmiyor — aynı-veri kuralı denetlenemedi.")
    return problems


# Testlerde nötrlenebilir kanca (``easy_train.kademe2_check`` deseni).
next_training_check: Callable[..., list[str]] = next_training_blockers


def release(adapter: str, reason: str) -> dict[str, Any]:
    """Başarısız/regresyonlu adaydan SONRA devam: gerekçeli + TAZE insan onayı (Kural 8)."""
    from app.agents.runtime import approvals

    reason = (reason or "").strip()
    if len(reason) < 20:
        raise TrainingLoopError("Gerekçe en az 20 karakter olmalı.")
    rec = get(adapter)
    if rec is None:
        raise TrainingLoopError(f"{adapter} eğitim döngüsü defterinde yok.")
    comp = rec.get("completion") or {}
    decision = approvals.require_fresh_approval(
        "lora-trainer",
        RELEASE_ACTION + adapter,
        "critical",
        f"Eğitim döngüsü engelini kaldır ({adapter}): {reason[:300]}",
    )
    if not decision.authorized:
        return {
            "ok": False,
            "status": "needs_approval",
            "approval_id": decision.approval_id,
            "approve_command": f"uv run hektor approval-approve {decision.approval_id}",
        }
    rec["release"] = {
        "approval_id": decision.approval_id,
        "reason": reason,
        "at": utcnow(),
        "adapter_sha256": comp.get("adapter_sha256", ""),
    }
    _write(_path(adapter), rec)
    return {"ok": True, "status": "released", "record": rec}


# ── özet (web + CLI) ─────────────────────────────────────────────────────────


def next_step(rec: dict[str, Any], running: bool) -> str:
    if running:
        return "Eğitim sürüyor — kesme; bitince `hektor egitim-dongusu bitis <ad>`."
    comp = rec.get("completion") or {}
    dev = rec.get("dev_eval") or {}
    if not comp:
        return "Bitiş doğrulaması bekliyor."
    if not comp.get("ok"):
        return "Bitiş doğrulanamadı → yeni eğitim engelli (insan incelemesi)."
    if not dev:
        return (
            "Geliştirme değerlendirmesi bekliyor: dönüşüm → LLM-30 koşusu → rubrik puanı → "
            "`hektor egitim-dongusu degerlendirme`."
        )
    if not dev.get("passed"):
        return "Değerlendirme geçmedi → yeni eğitim engelli."
    return "Değerlendirme geçti → yalnız YENİ doğrulanmış veriyle sonraki aday eğitilebilir."


def overview() -> dict[str, Any]:
    from app.training.detached_launch import is_running, read_detached_training_status

    with contextlib.suppress(Exception):
        sync_from_status()
    st = read_detached_training_status(get_settings().root)
    # "Sürüyor" yalnız süreç gerçekten canlıysa: sonuçsuz kalmış durum dosyası (ölü pid)
    # koşan eğitim sayılmaz → bitiş doğrulaması bekler (orada da başarısız çıkar).
    live = bool(st) and not (st.get("finished_at") or st.get("failed_at")) and is_running()
    running_name = str(st.get("adapter") or "") if live else ""
    rows = []
    for rec in reversed(entries()):
        comp = rec.get("completion") or {}
        dev = rec.get("dev_eval") or {}
        rows.append(
            {
                "adapter": rec["adapter"],
                "data_sha256": rec.get("data_sha256", ""),
                "recipe_sha": rec.get("recipe_sha", ""),
                "started_at": rec.get("started_at", ""),
                "running": rec["adapter"] == running_name,
                "completion": {
                    "ok": comp.get("ok"),
                    "verified_at": comp.get("verified_at"),
                    "failed": comp.get("failed", []),
                    "exit": comp.get("exit") or {},
                }
                if comp
                else None,
                "dev_eval": {
                    k: dev.get(k)
                    for k in (
                        "decision",
                        "passed",
                        "reference_model",
                        "comparison",
                        "flagged",
                        "problems",
                        "recorded_at",
                        "stale",
                    )
                }
                if dev
                else None,
                "released": _released(rec),
                "next_step": next_step(rec, rec["adapter"] == running_name),
            }
        )
    return {"candidates": rows, "running": running_name}
