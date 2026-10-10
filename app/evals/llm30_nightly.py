"""llm30_nightly.py — gece döngüsünde etkin modelin LLM-30 ÖLÇÜMÜ (eğitim yok, puan yok).

Tasarım: ``docs/TASARIM_GECE_DONGUSU.md`` §LLM-30. Amaç "model sorulara gerçekçi cevap veriyor
mu?" sorusunu günden güne İZLENEBİLİR yapmak; terfi/karar ÜRETMEZ.

Protokol gereği (``docs/PROTOKOL_LORA_RAG_IYILESTIRME.md``) LLM-30 cevapları anahtar kelimeyle
PUANLANMAZ — rubrik insan/hakemindir. Bu yüzden gece yalnız DETERMİNİSTİK sinyaller kaydeder:

- **bayraklar** (``llm30.answer_flags``): token sınırında kesilme, bağlam taşması, kısa/uzun
  tekrar döngüsü, boş cevap, CJK sızıntısı — üretim sağlığı.
- **sayısal anahtar izi** (``llm30.numeric_keys``): deterministik hesaplanmış 20 sayısal alt
  maddenin değeri cevap metninde geçiyor mu. Bu bir PUAN DEĞİLDİR (sayı başka bağlamda da
  geçebilir; doğru gerekçe ölçülmez) — yalnız gerileme sinyalidir.
- süre / token.

Koşu koşulları: etkin sohbet modeli, RAG kapalı (``HEKTOR_NIGHTLY_LLM30_RAG`` ile açılabilir),
``llm30_run`` ile AYNI sistem istemi ve çözme ayarı (temperature 0, seed 42). Model özeti
(digest) değişmediyse ve son ölçüm ``nightly_llm30_every_days``'den yeni ise koşmaz (gece
boşa 30 uzun üretim yapılmaz). Ağır iş kilidi (``comparison``) altında koşar — eğitim/dönüşümle
çakışmaz. Sorular ve cevaplar eğitime/retrieval indeksine GİRMEZ; yalnız rapora yazılır.

Gerileme: aynı soru setinde önceki ölçüme göre bayraklı cevap sayısı artarsa ya da sayısal
anahtar izi düşerse ``regression`` listesi dolar ve "Bekleyen kararlar"da görünür.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.evals.llm30 import SYSTEM_PROMPT, answer_flags, load_llm30, numeric_keys, user_prompt

SEED = 42
NUM_CTX = 16384
DECODING: dict[str, float | int] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "num_predict": 4096,
    "num_ctx": NUM_CTX,
}
NOTE = (
    "LLM-30 gece ölçümü: yalnız deterministik sinyaller (bayrak + sayısal anahtar izi). Rubrik "
    "puanı DEĞİLDİR; terfi/karar üretmez. Sorular eğitime girmez."
)

Chat = Callable[[str, str, dict[str, Any]], dict[str, Any]]
Info = Callable[[str], dict[str, Any]]


# ── sayısal anahtar izi ──────────────────────────────────────────────────────

_NUM = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?![\w])")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text or ""):
        tok = m.group(0)
        # "0,6667" (Türkçe ondalık) → 0.6667; "1.000" binlik ayıracı olabilir → olduğu gibi.
        if "," in tok and "." not in tok:
            tok = tok.replace(",", ".")
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _close(a: float, b: float) -> bool:
    # Yuvarlanmış yazımı kabul et (ör. 0.6667 ↔ 2/3, 101 ↔ 101.0).
    return math.isclose(a, b, rel_tol=5e-3, abs_tol=5e-3)


def _flat(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _flat(v)]
    if isinstance(value, float) and math.isnan(value):
        return []
    return [float(value)]


def key_hits(answer: str, keys: dict[str, Any], question_id: str) -> dict[str, bool]:
    """Bu sorunun sayısal anahtarlarından hangileri cevapta geçiyor (iz, puan değil)."""
    sid = question_id.rsplit("-", 1)[-1]  # llm30-s12 → s12
    nums = _numbers(answer)
    out: dict[str, bool] = {}
    for name, value in keys.items():
        if not name.startswith(sid + "_"):
            continue
        if isinstance(value, str):  # ISO zaman: saat:dakika kısmı geçmeli
            hhmm = value[11:16]
            out[name] = bool(hhmm) and hhmm in (answer or "")
            continue
        want = _flat(value)
        out[name] = bool(want) and all(any(_close(n, w) for n in nums) for w in want)
    return out


# ── Ollama (testlerde değiştirilir) ──────────────────────────────────────────


def _ollama_chat(model: str, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
    import httpx

    from app.evals.llm30_run import chat_stream

    host = get_settings().ollama_host.rstrip("/")
    with httpx.Client(timeout=3600) as client:
        return chat_stream(
            client,
            f"{host}/api/chat",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "options": options,
            },
        )


def _ollama_info(model: str) -> dict[str, Any]:
    from app.evals.llm30_run import model_info

    try:
        return model_info(get_settings().ollama_host.rstrip("/"), model)
    except SystemExit as exc:  # model_info CLI için SystemExit fırlatır
        raise RuntimeError(str(exc)) from exc


# ── geçmiş ───────────────────────────────────────────────────────────────────


def history_path() -> Path:
    p = get_settings().state_dir / "nightly" / "llm30_history.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def history() -> list[dict[str, Any]]:
    p = history_path()
    if not p.is_file():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def _due(model: str, digest: str, rag: bool, prev: list[dict[str, Any]]) -> str:
    """Koşmak gerekiyorsa sebep; gerekmiyorsa ``""``."""
    same = [h for h in prev if h.get("model") == model and h.get("rag") == rag]
    if not same:
        return "bu model için ilk ölçüm"
    last = same[-1]
    if last.get("digest") != digest:
        return "model özeti değişti (yeni ağırlık/etiket)"
    every = get_settings().nightly_llm30_every_days
    if every <= 0:
        return ""
    try:
        age = datetime.now(UTC) - datetime.fromisoformat(str(last.get("at")))
    except (TypeError, ValueError):
        return "önceki ölçümün zamanı okunamadı"
    return f"son ölçüm {age.days} gün önce" if age >= timedelta(days=every) else ""


def _regression(cur: dict[str, Any], prev: dict[str, Any] | None) -> list[str]:
    if not prev:
        return []
    out = []
    if cur["n_flagged"] > prev.get("n_flagged", 0):
        out.append(
            f"Bayraklı cevap {prev.get('n_flagged', 0)} → {cur['n_flagged']} (kesilme/tekrar/boş)"
        )
    if cur["key_hits"] < prev.get("key_hits", 0):
        out.append(
            f"Sayısal anahtar izi {prev.get('key_hits', 0)} → {cur['key_hits']} / "
            f"{cur['key_total']}"
        )
    return out


# ── ana akış ─────────────────────────────────────────────────────────────────


def run_measurement(
    *,
    force: bool = False,
    chat: Chat | None = None,
    info: Info | None = None,
    hold_lock: bool = True,
) -> dict[str, Any]:
    s = get_settings()
    model = s.effective_chat_model
    rag = bool(s.nightly_llm30_rag)
    try:
        minfo = (info or _ollama_info)(model)
    except Exception as exc:
        return {"ran": False, "skipped": f"Ollama/model okunamadı: {exc}"[:400], "model": model}
    digest = str(minfo.get("digest", ""))
    prev_all = history()
    why = "elle zorlandı" if force else _due(model, digest, rag, prev_all)
    if not why:
        return {
            "ran": False,
            "skipped": "Model değişmedi ve son ölçüm yeni — yeniden ölçülmedi.",
            "model": model,
            "digest": digest[:12],
        }
    if hold_lock:
        from app.training.resource_lock import HeavyJobBusy, hold

        try:
            with hold("comparison", "llm30_nightly"):
                return _measure(model, minfo, rag, why, prev_all, chat or _ollama_chat)
        except HeavyJobBusy as exc:
            return {"ran": False, "skipped": f"Ağır iş sürüyor — ölçüm atlandı: {exc}"[:400]}
    return _measure(model, minfo, rag, why, prev_all, chat or _ollama_chat)


def _contexts(items: list[Any]) -> dict[str, str]:
    """llm30_run.freeze_contexts ile AYNI retrieval (geçici dosyaya dondurulur)."""
    import tempfile

    from app.evals.llm30_run import freeze_contexts

    with tempfile.TemporaryDirectory() as d:
        rows = freeze_contexts(items, Path(d) / "contexts.jsonl", None)
    return {qid: str(r.get("context", "")) for qid, r in rows.items()}


def _measure(
    model: str,
    minfo: dict[str, Any],
    rag: bool,
    why: str,
    prev_all: list[dict[str, Any]],
    chat: Chat,
) -> dict[str, Any]:
    items = load_llm30(purpose="selection")
    keys = numeric_keys()
    ctxs = _contexts(items) if rag else {}
    rows: list[dict[str, Any]] = []
    t_all = time.monotonic()
    for it in items:
        prompt = user_prompt(it.question, ctxs.get(it.id) if rag else None)
        t0 = time.monotonic()
        try:
            resp = chat(model, prompt, {**DECODING, "seed": SEED})
        except Exception as exc:
            rows.append({"question_id": it.id, "error": str(exc)[:300], "flags": ["hata"]})
            if sum(1 for r in rows if r.get("error")) >= 3:
                break  # Ollama düştü: geceyi zaman aşımıyla doldurma
            continue
        answer = str(resp.get("content", ""))
        flags = answer_flags(
            answer,
            done_reason=str(resp.get("done_reason", "")),
            prompt_tokens=int(resp.get("prompt_eval_count") or 0),
            output_tokens=int(resp.get("eval_count") or 0),
            num_ctx=NUM_CTX,
        )
        rows.append(
            {
                "question_id": it.id,
                "flags": flags,
                "keys": key_hits(answer, keys, it.id),
                "output_tokens": int(resp.get("eval_count") or 0),
                "latency_s": round(time.monotonic() - t0, 1),
                "raw_answer": answer,
            }
        )
    hits = [v for r in rows for v in (r.get("keys") or {}).values()]
    total_keys = sum(
        1 for k in keys if any(k.startswith(i.id.rsplit("-", 1)[-1] + "_") for i in items)
    )
    summary = {
        "at": datetime.now(UTC).isoformat(),
        "model": model,
        "digest": str(minfo.get("digest", "")),
        "rag": rag,
        "reason": why,
        "n": len(rows),
        "n_questions": len(items),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "n_flagged": sum(1 for r in rows if r.get("flags")),
        "flag_counts": _count(f for r in rows for f in r.get("flags", [])),
        "key_hits": sum(1 for v in hits if v),
        "key_total": total_keys,
        "avg_output_tokens": round(
            sum(r.get("output_tokens", 0) for r in rows) / max(1, len(rows))
        ),
        "seconds": round(time.monotonic() - t_all, 1),
    }
    same = [h for h in prev_all if h.get("model") == model and h.get("rag") == rag]
    prev = same[-1] if same else (prev_all[-1] if prev_all else None)
    summary["regression"] = _regression(summary, prev)
    summary["compared_to"] = (
        {k: prev.get(k) for k in ("at", "model", "digest", "n_flagged", "key_hits")}
        if prev
        else None
    )
    out_dir = get_settings().reports_dir / "evals" / "llm30_nightly"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    report = out_dir / f"{stamp}.json"
    report.write_text(
        json.dumps(
            {**summary, "note": NOTE, "decoding": DECODING, "rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary["report"] = str(report)
    with history_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return {"ran": True, **summary, "note": NOTE}


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
