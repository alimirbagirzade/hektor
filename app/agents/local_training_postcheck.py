"""Eğitim-sonrası DOĞRULAMA scaffold'u (Phase 5E) — SALT-OKUMA; terfi/eğitim YOK.

İnsan gerçek eğitimi elle çalıştırdıktan SONRA, sonucu **read-only** denetler: handoff/
dry-run raporları, training artefaktı (storage/train_status.json), adapter metadata
(models/adapters/), adapter-eval raporu (reports/evals/), understanding-score (reports/
evals/understanding/). Sonuç yoksa `no_training_run_found`; varsa
`postcheck_ready_for_human_review`. Terfi ÖNERMEZ — yalnız `human_review_required` der.

Tasarım gereği şunları HİÇBİR ZAMAN çağırmaz: ``detached_launch.launch`` ·
``AutoLoRAPipeline.start_training`` · ``promote_to_production`` · ``require_fresh_approval``
· ``request_approval`` · ``subprocess.Popen`` · ``os.system``. Model YÜKLEMEZ, eval
ÇALIŞTIRMAZ, adapter/model YAZMAZ, korumalı yollara YAZMAZ. Yalnız dosya OKUR + stat.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)

_NOTE_NONE = "No training was started. No adapter was promoted."
_NOTE_FOUND = "No promotion was performed. (Training, if any, was run by a human — not this tool.)"
#: Terfi ASLA otomatik önerilmez.
_PROMOTION = "human_review_required"

_REVIEW_CHECKLIST = [
    "Adapter-eval base'i geçti mi (regresyon yok)?",
    "Understanding-score düştü mü (disiplin/dürüstlük)?",
    "Pretrain-gate GO idi (zehir/ezber yok)?",
    "Maliyet-dahil backtest/OOS sonucu kabul edilebilir mi?",
    "Terfi kararı: yalnız insan + ayrı taze onay (otomatik terfi YOK).",
]


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _ts_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def _default_out_dir() -> Path:
    return get_settings().root / "reports" / "local_training_orchestrator"


def _load_report(path: str | Path | None, pattern: str, out_dir: Path | None) -> dict[str, Any]:
    """Bir raporu READ-ONLY oku (verilen yol ya da out_dir'deki en son `pattern`)."""
    p: Path | None = None
    if path:
        pp = Path(path)
        p = pp if pp.is_absolute() else get_settings().root / pp
    else:
        d = out_dir or _default_out_dir()
        try:
            if d.exists():
                cands = sorted(d.glob(pattern))
                p = cands[-1] if cands else None
        except Exception:
            p = None
    if p and p.exists():
        try:
            return {"source": str(p), "data": json.loads(p.read_text(encoding="utf-8"))}
        except Exception as exc:
            return {"source": str(p), "error": str(exc)}
    return {"source": "none"}


def _probe_training(training_report: str | Path | None) -> dict[str, Any]:
    """Training artefaktını READ-ONLY ara: verilen rapor ya da storage/train_status.json."""
    if training_report:
        rep = _load_report(training_report, "*.json", None)
        if rep.get("data") is not None:
            return {"found": True, "source": rep.get("source"), "data": rep.get("data")}
        return {"found": False, "source": rep.get("source"), "error": rep.get("error")}
    s = get_settings()
    ts = s.root / "storage" / "train_status.json"
    if ts.exists():
        try:
            return {"found": True, "source": str(ts), "data": json.loads(ts.read_text("utf-8"))}
        except Exception as exc:
            return {"found": True, "source": str(ts), "error": str(exc)}
    return {"found": False, "source": "none"}


def _probe_adapter(adapter_path: str | Path | None) -> dict[str, Any]:
    """Adapter METADATA'sını READ-ONLY oku (stat). Model YÜKLEMEZ, eval ÇALIŞTIRMAZ."""
    s = get_settings()
    if adapter_path:
        p = Path(adapter_path)
        if not p.is_absolute():
            p = s.root / p
        if not p.exists():
            return {"found": False, "source": str(p), "reason": "yol yok"}
        try:
            files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
            size = sum(f.stat().st_size for f in files)
            return {
                "found": True,
                "source": str(p),
                "is_dir": p.is_dir(),
                "n_files": len(files),
                "size_bytes": size,
                "files": [f.name for f in files[:20]],
            }
        except Exception as exc:
            return {"found": False, "source": str(p), "error": str(exc)}
    # varsayılan: models/adapters/ altındaki adapter dizinlerini LİSTELE (stat).
    adapters: list[str] = []
    try:
        if s.adapters_dir.exists():
            adapters = sorted(d.name for d in s.adapters_dir.iterdir() if d.is_dir())
    except Exception:
        adapters = []
    return {"found": bool(adapters), "adapters_present": len(adapters), "adapters": adapters[:20]}


def _probe_adapter_eval() -> dict[str, Any]:
    """Adapter-eval raporlarını READ-ONLY ara (reports/evals/adapter_eval_*.json)."""
    d = get_settings().root / "reports" / "evals"
    try:
        cands = sorted(d.glob("adapter_eval_*.json")) if d.exists() else []
    except Exception:
        cands = []
    return {"found": bool(cands), "count": len(cands), "reports": [c.name for c in cands[:20]]}


def _probe_understanding() -> dict[str, Any]:
    """Understanding-score kayıtlarını READ-ONLY ara (reports/evals/understanding/*.json)."""
    d = get_settings().root / "reports" / "evals" / "understanding"
    try:
        cands = sorted(d.glob("*.json")) if d.exists() else []
    except Exception:
        cands = []
    return {"found": bool(cands), "count": len(cands), "records": [c.name for c in cands[:20]]}


def build_postcheck(
    *,
    handoff_json: str | Path | None = None,
    dryrun_json: str | Path | None = None,
    training_report: str | Path | None = None,
    adapter_path: str | Path | None = None,
    out_dir: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Eğitim-sonrası read-only doğrulama. TERFİ/EĞİTİM/ONAY-TÜKETİMİ YOK."""
    hf = _load_report(handoff_json, "*_handoff.json", out_dir)
    dr = _load_report(dryrun_json, "*_dryrun.json", out_dir)
    tr = _probe_training(training_report)
    ad = _probe_adapter(adapter_path)
    ev = _probe_adapter_eval()
    us = _probe_understanding()

    training_found = bool(tr.get("found") or ad.get("found"))
    base: dict[str, Any] = {
        "generated_at": _utcnow_iso(),
        "handoff_source": hf.get("source"),
        "handoff_status": (hf.get("data") or {}).get("status"),
        "dryrun_source": dr.get("source"),
        "training_artifacts_found": training_found,
        "adapter_eval_found": bool(ev.get("found")),
        "understanding_score_found": bool(us.get("found")),
        "training": tr,
        "adapter": ad,
        "adapter_eval": ev,
        "understanding": us,
        # Terfi ASLA otomatik önerilmez — her durumda insan incelemesi.
        "promotion_recommendation": _PROMOTION,
        "review_checklist": list(_REVIEW_CHECKLIST),
    }

    if not training_found:
        result: dict[str, Any] = {
            **base,
            "status": "no_training_run_found",
            "recommendation": "Run manual training first, then re-run postcheck.",
            "note": _NOTE_NONE,
        }
    else:
        result = {
            **base,
            "status": "postcheck_ready_for_human_review",
            "note": _NOTE_FOUND,
        }

    if write:
        try:
            written = _write_reports(result, out_dir)
            result["report_files"] = {k: str(v) for k, v in written.items()}
        except Exception as exc:  # pragma: no cover
            log.warning("Postcheck raporu yazılamadı: %s", exc)
    return result


def _write_reports(result: dict[str, Any], out_dir: Path | None) -> dict[str, Path]:
    target = out_dir or _default_out_dir()
    target.mkdir(parents=True, exist_ok=True)
    ts = _ts_compact()
    md_path = target / f"{ts}_postcheck.md"
    json_path = target / f"{ts}_postcheck.json"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"md": md_path, "json": json_path}


def render_markdown(result: dict[str, Any]) -> str:
    tr = result.get("training", {})
    ad = result.get("adapter", {})
    ev = result.get("adapter_eval", {})
    us = result.get("understanding", {})
    lines = [
        "# Eğitim-Sonrası Doğrulama (Postcheck) Raporu",
        "",
        f"> **{result.get('note', _NOTE_NONE)}**",
        "",
        "## Summary",
        f"- Üretildi: `{result.get('generated_at')}`",
        f"- **Durum:** `{result.get('status')}`",
        "",
        "## Handoff source",
        f"- `{result.get('handoff_source')}` (status: `{result.get('handoff_status')}`)",
        "",
        "## Dry-run source",
        f"- `{result.get('dryrun_source')}`",
        "",
        "## Training artifacts",
        f"- bulundu: `{result.get('training_artifacts_found')}` · "
        f"kaynak: `{tr.get('source')}` · adapter: `{ad.get('adapters_present', ad.get('found'))}`",
        "",
        "## Adapter eval",
        f"- bulundu: `{result.get('adapter_eval_found')}` · adet: `{ev.get('count', 0)}` "
        "(gerçek eval ÇALIŞTIRILMADI — yalnız rapor okundu)",
        "",
        "## Understanding score",
        f"- bulundu: `{result.get('understanding_score_found')}` · adet: `{us.get('count', 0)}`",
        "",
        "## Promotion safety",
        f"- **promotion_recommendation: `{result.get('promotion_recommendation')}`** "
        "— otomatik terfi YOK; terfi yalnız insan + ayrı taze onay.",
        "",
        "## Human review checklist",
    ]
    lines += [f"- [ ] {c}" for c in result.get("review_checklist", [])]
    if result.get("recommendation"):
        lines += ["", f"**Öneri:** {result.get('recommendation')}"]
    lines += [
        "",
        "## Final verdict",
        f"- `{result.get('status')}` — {result.get('note', _NOTE_NONE)}",
        "",
    ]
    return "\n".join(lines)
