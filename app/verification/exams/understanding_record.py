"""understanding_record.py — Anlama merdiveni skorunu KALICI kaydet + geçmişi oku.

``score_full_ladder`` / ``score_indicator_exams`` çıktısını hem JSON raporu
(``reports/evals/understanding/``) hem SQLite (``understanding_snapshots``) olarak
yazar → "okuduğunu anladı mı" yüzeysel %'yle değil, ZAMAN İÇİNDE sınav-geçme-oranıyla
izlenir (CLAUDE.md Kural 2: kanıtsız "başarılı" deme).

Eğitim BAŞLATMAZ; yalnız objektif anlama kanıtını kalıcılaştırır.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.verification.exams.understanding_score import UnderstandingScore

__all__ = [
    "compare_understanding",
    "load_understanding_history",
    "record_understanding",
    "understanding_report_dir",
]


def _auto_context(store: Any, base: dict[str, Any] | None) -> dict[str, Any]:
    """Snapshot bağlamını otomatik zenginleştir → zaman serisi YORUMLANABİLİR olsun.

    Çağıranın verdiği değerler korunur (setdefault). Otomatik eklenenler:
    - ``llm_model``: ölçülen base model (LocalLLM Ollama modeli). Adapter ölçülüyorsa
      çağıran ``adapter_id``/``model_kind`` geçmeli (v5-savunması: base vs adapter ayrımı).
    - ``n_papers``/``n_carded``: korpus boyutu — pass_rate değişimi korpus büyümesinden mi
      yoksa gerçek anlama değişiminden mi geldiğini ayırt etmek için.
    Hiçbir otomatik adım kaydı ASLA bozmaz (hepsi guard'lı).
    """
    ctx = dict(base or {})
    with contextlib.suppress(Exception):
        ctx.setdefault("llm_model", getattr(get_settings(), "llm_model", None))
    ctx.setdefault("model_kind", "base")  # adapter ölçümünde çağıran "adapter" geçer
    with contextlib.suppress(Exception):
        papers = store.list_papers()
        ctx.setdefault("n_papers", len(papers))
        ctx.setdefault("n_carded", sum(1 for p in papers if store.has_knowledge_card(p.paper_id)))
    return ctx


def understanding_report_dir() -> Path:
    """``reports/evals/understanding/`` (yoksa oluştur)."""
    d = get_settings().reports_dir / "evals" / "understanding"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_understanding(
    score: UnderstandingScore,
    *,
    seed: int = 0,
    store: Any = None,
    context: dict[str, Any] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Skoru DB'ye ve (opsiyonel) zaman-damgalı JSON raporuna yaz; özet döndür.

    Returns:
        ``{"snapshot_id", "report_path", "status", "pass_rate", "graded"}``.
    """
    from app.memory.sqlite_store import SqliteStore

    store = store or SqliteStore()
    ctx = _auto_context(store, context)
    snapshot_id = store.save_understanding_snapshot(score, seed=seed, context=ctx)

    report_path: str | None = None
    if write_report:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = understanding_report_dir() / f"understanding_{stamp}_{snapshot_id}.json"
        payload = {
            "snapshot_id": snapshot_id,
            "seed": seed,
            "context": ctx,
            **score.to_dict(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path = str(path)

    return {
        "snapshot_id": snapshot_id,
        "report_path": report_path,
        "status": score.status,
        "pass_rate": score.pass_rate,
        "graded": score.graded,
    }


def load_understanding_history(store: Any = None, *, limit: int = 50) -> list[dict[str, Any]]:
    """En yeni → en eski sırada kalıcı anlama anlık görüntüleri."""
    from app.memory.sqlite_store import SqliteStore

    store = store or SqliteStore()
    return store.list_understanding_snapshots(limit=limit)


def _level_pass_rate(by_level: dict[str, Any] | None, level: str) -> float | None:
    v = (by_level or {}).get(level, {})
    graded = v.get("passed", 0) + v.get("failed", 0)
    return (v.get("passed", 0) / graded) if graded else None


def compare_understanding(
    prev: dict[str, Any], curr: dict[str, Any], *, drop_threshold: float = 0.05
) -> dict[str, Any]:
    """İki snapshot'ı kıyasla → pass_rate + seviye deltası + REGRESYON bayrağı.

    v5-tipi gerilemeyi OBJEKTİF yakalamanın temeli. Kıyas yalnız AYNI ``llm_model``
    bağlamında anlamlı (model değiştiyse skor düşüşü gerçek regresyon mu farklı model mi
    belli olmaz) → ``comparable=False`` ise ``regressed`` hesaplanmaz. ``curr`` pass_rate'i
    ``prev``'ten ``drop_threshold``'tan fazla düştüyse ``regressed=True``.
    """
    pr_prev = prev.get("pass_rate")
    pr_curr = curr.get("pass_rate")
    same_model = (prev.get("context") or {}).get("llm_model") == (curr.get("context") or {}).get(
        "llm_model"
    )
    delta = round(pr_curr - pr_prev, 4) if (pr_prev is not None and pr_curr is not None) else None

    level_delta: dict[str, float] = {}
    for lvl in sorted(set(prev.get("by_level") or {}) | set(curr.get("by_level") or {})):
        rp = _level_pass_rate(prev.get("by_level"), lvl)
        rc = _level_pass_rate(curr.get("by_level"), lvl)
        if rp is not None and rc is not None:
            level_delta[lvl] = round(rc - rp, 4)

    regressed = bool(same_model and delta is not None and delta < -drop_threshold)
    return {
        "comparable": same_model,
        "pass_rate_prev": pr_prev,
        "pass_rate_curr": pr_curr,
        "delta": delta,
        "level_delta": level_delta,
        "regressed": regressed,
        "note": None if same_model else "Farklı model/bağlam — kıyas güvenilmez",
    }
