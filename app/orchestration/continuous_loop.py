"""Sürekli (gözetimsiz) eğitim döngüsü — her ``tick`` TEK adım atar.

Kullanıcı kararı (2026-10-15; ``docs/TASARIM_SUREKLI_EGITIM.md``): eğitim insan onayı
beklemeden döner; ölçüt "sorular ve doğru cevaplar" (LLM-30, yerel hakem rubriği); insan
müdahalesi yalnız TÜM sorular doğru olunca (terfi + final takımı insanda). Kademe 2 yalnız
kod değişince (``kademe2_code_only``).

Akış (her tik en çok bir yan etkili adım; Görev Zamanlayıcı tikleri tekrarlar):

    kapalı / STOP_ALL / STOP_LEARNING / duraklatıldı           → hiçbir şey
    döngünün arka plan işi (LLM-30 koşusu, hakem) sürüyor     → bekle
    eğitim / dönüşüm / ağır iş kilidi                         → bekle
    son aday (defter):
      bitiş doğrulanmadı              → ``training_loop.record_completion``
      Ollama etiketi yok              → dönüşüm işi (``candidate_jobs``)
      referans LLM-30 yok             → referans koşusu (arka plan)
      aday LLM-30 yok                 → aday koşusu (arka plan)
      puansız                         → yerel hakem (arka plan)
      puanlı                          → ``record_dev_eval`` + "tüm sorular doğru mu?"
          hepsi doğru                 → DURAKLAT (insan: terfi kararı + final takımı)
          kapı geçti                  → ardışık başarısızlık = 0
          kapı geçmedi                → ardışık +1; sınırda DURAKLAT, değilse politika devam
    yeni eğitim: ``easy_train.readiness`` hazırsa (veri yeni, Kademe 2 kodu kapsıyor, kaynak boş)
      → anlık görüntü (sabit profil, merge-aware) → ``easy_train.launch(unattended=True)``

Aynı-veri kuralı, sızıntı/kalite/Kural 1 kapıları, Kademe 2, STOP_ALL ve terfi kapısı aynen
geçerlidir; bu modül yalnız insan TIKINI politika ile değiştirir. Terfi ASLA yapılmaz.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.procutil import DETACHED_HIDDEN

_TAG_SAFE = re.compile(r"[^a-z0-9._-]+")
_JOB_KINDS = ("llm30_ref", "llm30_aday", "hakem_ref", "hakem_aday")


class LoopError(RuntimeError):
    pass


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def loop_dir() -> Path:
    d = get_settings().root / "storage" / "continuous_loop"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path() -> Path:
    return loop_dir() / "state.json"


def read_state() -> dict[str, Any]:
    p = _state_path()
    try:
        data = json.loads(p.read_text("utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {"paused": False, "pause_reason": "", "consecutive_failures": 0, "candidates": {}}


def write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _event(action: str, detail: str = "", **fields: Any) -> dict[str, Any]:
    row = {"at": utcnow(), "action": action, "detail": detail, **fields}
    with (loop_dir() / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def events(limit: int = 30) -> list[dict[str, Any]]:
    p = loop_dir() / "events.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(x) for x in p.read_text("utf-8").split("\n") if x.strip()]
    return rows[-limit:]


# ── testlerde değiştirilebilir kancalar ──────────────────────────────────────


def _spawn_default(args: list[str], log_name: str) -> dict[str, Any]:
    """Arka plan Python süreci (penceresiz, ebeveynden bağımsız)."""
    from app.training.resource_lock import process_create_time

    root = get_settings().root
    log = loop_dir() / log_name
    with log.open("ab") as out:
        proc = subprocess.Popen(
            [sys.executable, *args],
            cwd=str(root),
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=DETACHED_HIDDEN,
            close_fds=True,
        )
    return {"pid": proc.pid, "create_time": process_create_time(proc.pid), "log": str(log)}


def _alive_default(job: dict[str, Any]) -> bool:
    from app.training.resource_lock import pid_alive, process_create_time

    pid = job.get("pid")
    if not pid or not pid_alive(pid):
        return False
    ct = job.get("create_time")
    return ct is None or process_create_time(int(pid)) == ct


def _ollama_digest_default(tag: str) -> str:
    """Etiketin Ollama özeti ('' = yok)."""
    from app.feedback.model_identity import match_entry, ollama_tags

    tags = ollama_tags()
    if tags is None:
        return ""
    entry = match_entry(tags, tag)
    return str((entry or {}).get("digest") or "") if entry else ""


def _busy_default() -> str:
    """Eğitim/dönüşüm/ağır iş sürüyorsa açıklama, yoksa ''."""
    from app.training import candidate_jobs, resource_lock
    from app.training.detached_launch import is_running

    if is_running():
        return "eğitim sürüyor"
    job = candidate_jobs.running_job()
    if job:
        return f"aday işi sürüyor ({job.get('kind')} · {job.get('adapter')})"
    return resource_lock.blocker() or ""


spawn: Callable[[list[str], str], dict[str, Any]] = _spawn_default
job_alive: Callable[[dict[str, Any]], bool] = _alive_default
ollama_digest: Callable[[str], str] = _ollama_digest_default
busy: Callable[[], str] = _busy_default


# ── yardımcılar ──────────────────────────────────────────────────────────────


def seeds() -> list[int]:
    return [int(x) for x in str(get_settings().continuous_loop_seeds).split() if x.strip()]


def _slug(name: str) -> str:
    return _TAG_SAFE.sub("-", name.lower().replace("_", "-")).strip("-")


def candidate_tag(adapter: str) -> str:
    return f"{_slug(adapter)}-aday"


def llm30_dir(run_id: str) -> Path:
    return get_settings().root / "reports" / "evals" / "llm30" / run_id


def _expected_rows() -> int:
    from app.evals.llm30 import QUESTION_IDS

    return len(QUESTION_IDS) * len(seeds())


def _count_rows(path: Path, *, scored: bool = False) -> int:
    if not path.exists():
        return 0
    n = 0
    for x in path.read_text("utf-8").split("\n"):
        if not x.strip():
            continue
        r = json.loads(x)
        if r.get("condition") != "B":
            continue
        if scored and "judge" not in r:
            continue
        n += 1
    return n


def run_complete(run_id: str) -> bool:
    return _count_rows(llm30_dir(run_id) / "raw.jsonl") >= _expected_rows()


def scored_complete(run_id: str) -> bool:
    from app.evals.llm30_judge import SCORED_NAME

    return _count_rows(llm30_dir(run_id) / SCORED_NAME, scored=True) >= _expected_rows()


def _llm30_args(base: str, lora: str, adapter: str, run_id: str) -> list[str]:
    return [
        "-m", "app.evals.llm30_run",
        "--base", base, "--lora", lora, "--adapter", adapter,
        "--conditions", "B", "--seeds", *[str(s) for s in seeds()],
        "--run-id", run_id,
    ]  # fmt: skip


def _start_job(state: dict[str, Any], kind: str, args: list[str], **meta: Any) -> dict[str, Any]:
    info = spawn(args, f"{kind}.log")
    state["job"] = {"kind": kind, "started_at": utcnow(), **info, **meta}
    write_state(state)
    return _event("is_baslatildi", kind, **meta)


def _pause(state: dict[str, Any], reason: str) -> dict[str, Any]:
    state["paused"] = True
    state["pause_reason"] = reason
    state["paused_at"] = utcnow()
    write_state(state)
    return _event("duraklatildi", reason)


def _fail(state: dict[str, Any], adapter: str, why: str) -> dict[str, Any]:
    """Aday başarısız: ardışık sayaç; sınırda duraklat, değilse politika serbest bırakır."""
    from app.orchestration import training_loop

    s = get_settings()
    cand = state["candidates"].setdefault(adapter, {})
    if cand.get("failed"):
        return {"action": "zaten_basarisiz", "detail": adapter}
    cand.update(failed=True, failure=why, done=True, at=utcnow())
    state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
    write_state(state)
    _event("aday_basarisiz", why, adapter=adapter, ardisik=state["consecutive_failures"])
    if state["consecutive_failures"] >= s.continuous_loop_max_failures:
        return _pause(
            state,
            f"{state['consecutive_failures']} ardışık aday kapıdan geçemedi (son: {adapter} — "
            f"{why}). İnsan incelemesi gerekli; devam için `hektor surekli-dongu devam`.",
        )
    res = training_loop.release_by_policy(
        adapter, f"Sürekli döngü: aday kapıdan geçemedi ({why}); politika ile devam."
    )
    return _event("serbest_birakildi" if res.get("ok") else "serbest_birakilamadi", adapter)


# ── tik ──────────────────────────────────────────────────────────────────────


def tick() -> dict[str, Any]:
    """Döngünün bir adımı. Dönüş: ``{action, detail}`` (her zaman olay günlüğüne de yazılır)."""
    s = get_settings()
    if not s.continuous_loop_enabled:
        return {"action": "kapali", "detail": "continuous_loop_enabled=false"}
    from app.orchestration.nightly import stop_marker

    marker = stop_marker()
    if marker:
        return {"action": "durduruldu", "detail": marker}
    state = read_state()
    state["last_tick"] = utcnow()
    if state.get("paused"):
        write_state(state)
        return {"action": "duraklatildi", "detail": state.get("pause_reason", "")}

    job = state.get("job")
    if job:
        if job_alive(job):
            write_state(state)
            return {"action": "bekliyor", "detail": f"{job['kind']} sürüyor"}
        state.pop("job", None)
        write_state(state)
        _event("is_bitti", job["kind"], run_id=job.get("run_id", ""))

    why_busy = busy()
    if why_busy:
        write_state(state)
        return {"action": "bekliyor", "detail": why_busy}

    out = _advance_candidate(state)
    if out is not None:
        return out
    return _maybe_train(state)


def _advance_candidate(state: dict[str, Any]) -> dict[str, Any] | None:
    """Son adayı bir adım ilerlet; aday yok/işlendi → None (yeni eğitime geçilir)."""
    from app.evals.llm30_judge import SCORED_NAME, all_correct
    from app.orchestration import training_loop
    from app.training import candidate_jobs

    s = get_settings()
    training_loop.sync_from_status()
    rows = training_loop.entries()
    if not rows:
        return None
    rec = rows[-1]
    name = rec["adapter"]
    cand = state["candidates"].setdefault(name, {})
    if cand.get("done"):
        return None

    # 1) bitiş
    comp = rec.get("completion")
    if not comp:
        rec = training_loop.record_completion(name)
        comp = rec["completion"]
        _event("bitis", "doğrulandı" if comp["ok"] else "başarısız", adapter=name)
        if not comp["ok"]:
            return _fail(state, name, "bitiş doğrulanamadı: " + ", ".join(comp.get("failed", [])))
        return {"action": "bitis_dogrulandi", "detail": name}
    if not comp.get("ok"):
        return _fail(state, name, "bitiş doğrulanamadı")

    # 2) dönüşüm
    tag = cand.get("tag") or candidate_tag(name)
    cand["tag"] = tag
    if not ollama_digest(tag):
        conv = cand.get("conversion_job")
        if conv:
            job = candidate_jobs.get_job(conv)
            if job.get("status") in ("failed", "lost", "error", "cancelled"):
                return _fail(state, name, f"Ollama dönüşümü başarısız ({job.get('status')})")
            if job.get("status") == "done":
                return _fail(state, name, "dönüşüm 'done' ama etiket Ollama'da yok")
            write_state(state)
            return {"action": "bekliyor", "detail": f"dönüşüm {conv} ({job.get('status')})"}
        job = candidate_jobs.start_job(
            "conversion",
            adapter=name,
            request_id=f"loop-conv-{name}"[:80],
            params={"ollama_tag": tag, "template_from": s.continuous_loop_template_from},
        )
        cand["conversion_job"] = job["job_id"]
        write_state(state)
        return _event("donusum_baslatildi", job["job_id"], adapter=name, tag=tag)

    # 3) referans LLM-30 (referans model özeti değişince yeniden)
    ref_digest = ollama_digest(s.continuous_loop_reference_tag)
    if not ref_digest:
        return _pause(state, f"Referans model Ollama'da yok: {s.continuous_loop_reference_tag}")
    ref_id = "loop_ref_" + _slug(s.continuous_loop_reference_tag) + "_" + ref_digest[:12]
    seed_tag = "s" + "-".join(str(x) for x in seeds())
    ref_id += "_" + seed_tag
    if not run_complete(ref_id):
        # Referans koşusunda base = referansın kendisi: llm30_run base/lora şablon eşitliği
        # ister; yalnız B (lora) koşulu koşulur, base kullanılmaz.
        return _start_job(
            state,
            "llm30_ref",
            _llm30_args(
                s.continuous_loop_reference_tag,
                s.continuous_loop_reference_tag,
                s.continuous_loop_reference_adapter,
                ref_id,
            ),
            run_id=ref_id,
        )

    # 4) aday LLM-30
    cand_id = f"loop_{name}_{seed_tag}"
    cand["run_id"] = cand_id
    if not run_complete(cand_id):
        return _start_job(
            state,
            "llm30_aday",
            _llm30_args(s.continuous_loop_template_from, tag, name, cand_id),
            run_id=cand_id,
            adapter=name,
        )

    # 5) hakem
    for kind, run_id in (("hakem_ref", ref_id), ("hakem_aday", cand_id)):
        if not scored_complete(run_id):
            return _start_job(
                state,
                kind,
                ["-m", "app.evals.llm30_judge", str(llm30_dir(run_id))],
                run_id=run_id,
            )

    # 6) kapı
    cand_scored = llm30_dir(cand_id) / SCORED_NAME
    ref_scored = llm30_dir(ref_id) / SCORED_NAME
    rec = training_loop.record_dev_eval(
        name,
        candidate_raw=cand_scored,
        candidate_condition="B",
        reference_raw=ref_scored,
        reference_condition="B",
        min_repeats=len(seeds()),
    )
    dev = rec["dev_eval"]
    correct = all_correct(cand_scored)
    cand.update(dev_passed=bool(dev["passed"]), correct=correct)
    _event(
        "degerlendirme",
        f"kapı={dev['decision']} geçti={dev['passed']} doğru={correct['n_correct']}/"
        f"{correct['n_questions']}",
        adapter=name,
    )
    if correct["ok"]:
        cand["done"] = True
        return _pause(
            state,
            f"TÜM SORULAR DOĞRU ({correct['n_questions']}/{correct['n_questions']}, {name}). "
            "İnsan kararı: final takımı + terfi. Döngü duraklatıldı.",
        )
    if dev["passed"]:
        cand["done"] = True
        state["consecutive_failures"] = 0
        write_state(state)
        return _event("aday_gecti", f"yanlış: {', '.join(correct['wrong'])[:300]}", adapter=name)
    return _fail(state, name, f"kapı {dev['decision']}; yanlış {len(correct['wrong'])} soru")


def _maybe_train(state: dict[str, Any]) -> dict[str, Any]:
    from app.training import easy_train

    s = get_settings()
    r = easy_train.readiness()
    blocking = [i for i in r["items"] if not i["ok"]]
    if blocking:
        detail = " | ".join(f"{i['key']}: {i['detail']}" for i in blocking)[:600]
        write_state(state)
        if any(i["key"] == "egitim_dongusu" and "zaten eğitildi" in i["detail"] for i in blocking):
            return {"action": "yeni_veri_bekleniyor", "detail": detail}
        return {"action": "egitim_hazir_degil", "detail": detail}
    if not s.unattended_training_enabled:
        write_state(state)
        return {"action": "insan_onayi_gerekli", "detail": "unattended_training_enabled=false"}
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M")
    recipe = easy_train.prepare_snapshot(
        {
            "profile": s.continuous_loop_profile,
            "base_model": s.peft_base_model,
            "mix_profile": s.continuous_loop_mix_profile,
            "mix_weights": {},
            "adapter_name": f"hektor_loop_{stamp}",
            "max_examples": 0,
        }
    )
    snap = "snap_" + recipe["recipe_sha"][:16]
    res = easy_train.launch(snap, f"loop-{snap}", unattended=True)
    _event(
        "egitim_baslatildi" if res.get("ok") else "egitim_baslatilamadi",
        res.get("message", "") or "; ".join(res.get("problems", []))[:500],
        adapter=recipe["adapter_name"],
        snapshot=snap,
    )
    return {
        "action": "egitim_baslatildi" if res.get("ok") else "egitim_baslatilamadi",
        "detail": res.get("message", "") or "; ".join(res.get("problems", []))[:500],
    }


# ── insan kontrolleri ────────────────────────────────────────────────────────


def pause(reason: str) -> dict[str, Any]:
    return _pause(read_state(), reason or "insan duraklattı")


def resume(reason: str) -> dict[str, Any]:
    """Duraklatmayı kaldır (insan). Ardışık başarısızlık sayacı sıfırlanır."""
    if len((reason or "").strip()) < 10:
        raise LoopError("Gerekçe en az 10 karakter olmalı.")
    state = read_state()
    state.update(paused=False, pause_reason="", consecutive_failures=0)
    write_state(state)
    return _event("devam", reason.strip())


def overview() -> dict[str, Any]:
    s = get_settings()
    state = read_state()
    return {
        "enabled": bool(s.continuous_loop_enabled),
        "unattended": bool(s.unattended_training_enabled),
        "kademe2_code_only": bool(s.kademe2_code_only),
        "profile": s.continuous_loop_profile,
        "reference": s.continuous_loop_reference_tag,
        "judge": s.continuous_loop_judge_model,
        "state": state,
        "events": events(15),
    }
