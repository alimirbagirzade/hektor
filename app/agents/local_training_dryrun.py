"""Onaylı eğitim isteği DRY-RUN pipeline'ı (Phase 5C) — eğitim/onay-tüketimi YOK.

Akış: 5A audit (read-only) → 5B istek raporunu OKU → (varsa) approval_id'yi read-only
kontrol et (TÜKETMEDEN) → dataset/pretrain-gate read-only → adapter-eval MOCKED →
training execution PLANI üret (uygulamaz) → rapor yaz.

Tasarım gereği şunları HİÇBİR ZAMAN çağırmaz: ``detached_launch.launch`` ·
``AutoLoRAPipeline.start_training`` · ``promote_to_production`` ·
``require_fresh_approval`` (tüketim) · ``request_approval`` (oluşturma — bu faz okur).
Gerçek adapter-eval modeli ÇALIŞTIRMAZ; yalnız mocked readiness döner.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

from app.agents import local_training_orchestrator as lto
from app.config import get_settings

log = logging.getLogger(__name__)

#: Değişmez güvence — her yanıtta yer alır.
DRYRUN_BANNER = "No training was started. No approval was consumed."

#: Onaylanmış isteğin uygulanma planı (yalnız PLAN — uygulanmaz).
_EXECUTION_PLAN = [
    "validate dataset",
    "run pretrain gate",
    "prepare LoRA config",
    "wait for explicit real-training command",
]


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _ts_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def _default_out_dir() -> Path:
    return get_settings().root / "reports" / "local_training_orchestrator"


def _approval_status(approval_id: str | None) -> dict[str, Any]:
    """Onay durumunu READ-ONLY oku (get_approval). TÜKETMEZ."""
    if not approval_id:
        return {"approval_id": None, "approval_status": "none", "consumed": False}
    try:
        from app.agents.runtime import approvals

        appr = approvals.get_approval(approval_id)
        if appr is None:
            return {"approval_id": approval_id, "approval_status": "not_found", "consumed": False}
        raw = getattr(appr.status, "value", str(appr.status))
        consumed = bool(getattr(appr, "consumed_at", None))
        if raw == "approved":
            label = "approved_consumed" if consumed else "approved_not_consumed"
        else:
            label = str(raw)
        return {"approval_id": approval_id, "approval_status": label, "consumed": consumed}
    except Exception as exc:  # pragma: no cover - savunmacı
        return {"approval_id": approval_id, "approval_status": "error", "error": str(exc)}


def _load_request(request_json: str | Path | None, out_dir: Path | None) -> dict[str, Any]:
    """5B istek raporunu READ-ONLY oku (verilen yol ya da en son *_request.json)."""
    path: Path | None = None
    if request_json:
        p = Path(request_json)
        path = p if p.is_absolute() else get_settings().root / p
    else:
        d = out_dir or _default_out_dir()
        try:
            if d.exists():
                cands = sorted(d.glob("*_request.json"))
                path = cands[-1] if cands else None
        except Exception:
            path = None
    if path and path.exists():
        try:
            return {"source": str(path), "data": json.loads(path.read_text(encoding="utf-8"))}
        except Exception as exc:
            return {"source": str(path), "error": str(exc)}
    return {"source": "none"}


def _blocked_reason(report: lto.TrainingAuditReport, stop_active: bool) -> str:
    if stop_active:
        return "STOP_ALL is active"
    joined = "; ".join(report.risks) or "engelleyici risk mevcut"
    return f"Hazırlık BLOCKED — {joined}"


def build_dryrun(
    *,
    approval_id: str | None = None,
    request_json: str | Path | None = None,
    mock_adapter_eval: bool = True,
    out_dir: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Onaylı isteğin DRY-RUN pipeline'ını simüle et. EĞİTİM/ONAY-TÜKETİMİ YOK."""
    report = lto.run_audit(write=False)  # 5A read-only audit
    probes = report.probes
    stop_active = bool(probes.get("stop_all", {}).get("stop_all_active"))
    gate_verdict = probes.get("pretrain_gate", {}).get("verdict")
    appr = _approval_status(approval_id)
    request_info = _load_request(request_json, out_dir)

    base: dict[str, Any] = {
        "generated_at": _utcnow_iso(),
        "approval_id": appr.get("approval_id"),
        "approval_status": appr.get("approval_status"),
        "readiness_score": report.readiness_score,
        "readiness_verdict": report.readiness_verdict,
        "risks": list(report.risks),
        "pretrain_gate": gate_verdict or probes.get("pretrain_gate", {}).get("reason", "n/a"),
        "adapter_eval": "mocked_ready" if mock_adapter_eval else "real_eval_unsupported",
        "request": {"source": request_info.get("source")},
        "note": DRYRUN_BANNER,
    }

    if stop_active or report.readiness_verdict == "BLOCKED":
        result: dict[str, Any] = {
            **base,
            "status": "blocked",
            "reason": _blocked_reason(report, stop_active),
        }
    elif gate_verdict == "NO-GO":
        result = {**base, "status": "not_ready", "reason": "pretrain gate NO-GO"}
    elif report.readiness_verdict != "READY":
        result = {
            **base,
            "status": "not_ready",
            "reason": f"readiness {report.readiness_verdict} (skor {report.readiness_score}/100)",
        }
    elif appr.get("approval_status") != "approved_not_consumed":
        # READY ama geçerli (approved + tüketilmemiş) onay yok → onay gerekir.
        hint = (
            "Önce `uv run achilles local-training-request --create-approval` ile onay isteği "
            "oluştur ve insan onayını al."
        )
        result = {
            **base,
            "status": "needs_approval",
            "reason": f"onay durumu: {appr.get('approval_status')}",
            "hint": hint,
            "execution_plan": list(_EXECUTION_PLAN),
        }
    else:
        # READY + pretrain GO + onaylı (tüketilmemiş) → dry-run BAŞARILI (yalnız PLAN).
        result = {**base, "status": "dry_run_passed", "execution_plan": list(_EXECUTION_PLAN)}

    if write:
        try:
            written = _write_reports(result, out_dir)
            result["report_files"] = {k: str(v) for k, v in written.items()}
        except Exception as exc:  # pragma: no cover
            log.warning("Dry-run raporu yazılamadı: %s", exc)
    return result


def _write_reports(result: dict[str, Any], out_dir: Path | None) -> dict[str, Path]:
    target = out_dir or _default_out_dir()
    target.mkdir(parents=True, exist_ok=True)
    ts = _ts_compact()
    md_path = target / f"{ts}_dryrun.md"
    json_path = target / f"{ts}_dryrun.json"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"md": md_path, "json": json_path}


def render_markdown(result: dict[str, Any]) -> str:
    plan = result.get("execution_plan") or []
    score = result.get("readiness_score")
    verdict = result.get("readiness_verdict")
    lines = [
        "# Onaylı Eğitim İsteği — DRY-RUN Pipeline Raporu",
        "",
        f"> **{result.get('note', DRYRUN_BANNER)}**",
        "",
        "## Summary",
        f"- Üretildi: `{result.get('generated_at')}`",
        f"- **Durum:** `{result.get('status')}`",
        "",
        "## Approval status",
        f"- approval_id: `{result.get('approval_id')}`",
        f"- approval_status: `{result.get('approval_status')}` (read-only — TÜKETİLMEDİ)",
        "",
        "## Readiness",
        f"- skor: `{score}/100` · karar: `{verdict}`",
        "",
        "## Pretrain gate",
        f"- `{result.get('pretrain_gate')}`",
        "",
        "## Adapter eval (mock)",
        f"- `{result.get('adapter_eval')}` — gerçek model ÇALIŞTIRILMADI.",
        "",
        "## Execution plan (yalnız PLAN — uygulanmadı)",
    ]
    lines += [f"{i + 1}. {step}" for i, step in enumerate(plan)] or ["- (plan yok)"]
    if result.get("reason"):
        lines += ["", f"**Sebep:** {result.get('reason')}"]
    if result.get("hint"):
        lines += [f"**İpucu:** {result.get('hint')}"]
    lines += [
        "",
        "## Safety checks",
        "- launch / train --run / start_training / promote: ÇAĞRILMADI.",
        "- require_fresh_approval / approval consumption: YOK.",
        "- gerçek adapter-eval / model write / cloud: YOK.",
        "",
        "## Final verdict",
        f"- `{result.get('status')}` — {result.get('note', DRYRUN_BANNER)}",
        "",
    ]
    return "\n".join(lines)
