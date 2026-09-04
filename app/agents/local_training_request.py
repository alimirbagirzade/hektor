"""Onay-kapılı lokal eğitim İSTEĞİ akışı (Phase 5B) — eğitim BAŞLATMAZ.

Akış: local-training-audit (5A) → readiness değerlendir → uygunsa bir PENDING onay
isteği OLUŞTUR (tüketmeden) → kullanıcıya onay komutunu göster. Onay verilse bile bu
faz onayı TÜKETMEZ ve gerçek eğitim çalıştırmaz; akış dry-run/mocked kalır.

Tasarım gereği şunları HİÇBİR ZAMAN çağırmaz: ``detached_launch.launch`` ·
``AutoLoRAPipeline.start_training`` · ``promote_to_production`` · ``train --run`` ·
``require_fresh_approval`` (onay tüketimi). Yalnız ``request_approval`` (pending
oluşturur) + 5A audit'in read-only sondalarını kullanır.
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

#: Eğitim onayı için CLI/web ile AYNI anahtar.
_AGENT_ID = "lora-trainer"
_ACTION = "train_run"
_RISK = "critical"

#: Değişmez güvence — her yanıtta yer alır.
REQUEST_BANNER = "No training was started."
_PREVIEW_NOTE = "No approval request was created. No training was started."


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _ts_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def _blocked_reason(report: lto.TrainingAuditReport) -> str:
    if report.readiness_verdict == "BLOCKED":
        joined = "; ".join(report.risks) or "engelleyici risk mevcut"
        return f"Hazırlık BLOCKED — {joined}"
    return (
        f"Hazırlık READY değil (skor {report.readiness_score}/100, "
        f"karar={report.readiness_verdict})."
    )


def _create_pending_approval(report: lto.TrainingAuditReport) -> str:
    """PENDING onay isteği OLUŞTUR (tüketmez). approval_id döner."""
    from app.agents.runtime import approvals

    summary = (
        f"Gerçek LoRA eğitimi isteği (lokal akış) — readiness "
        f"{report.readiness_score}/100 ({report.readiness_verdict}). "
        "Onay TEK KULLANIMLIK; eğitim ayrıca taze onay + açık koşu ister."
    )
    req = approvals.request_approval(
        agent_id=_AGENT_ID, action=_ACTION, summary=summary, risk=_RISK
    )
    return req.approval_id


def build_request(
    *,
    create_approval: bool = False,
    preview: bool = False,
    out_dir: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Eğitim isteği akışını çalıştır. GERÇEK EĞİTİM BAŞLATMAZ; onay TÜKETMEZ.

    - Varsayılan (preview): audit çalıştırır, ön izleme döner; onay oluşturmaz.
    - ``create_approval=True`` ve readiness READY: PENDING onay isteği oluşturur.
    - STOP_ALL / risk / READY değil: ``blocked`` döner, onay oluşturmaz.
    """
    report = lto.run_audit(write=False)  # 5A read-only audit (eğitim başlatmaz)
    base: dict[str, Any] = {
        "generated_at": _utcnow_iso(),
        "readiness_score": report.readiness_score,
        "readiness_verdict": report.readiness_verdict,
        "risks": list(report.risks),
        "note": REQUEST_BANNER,
    }

    # Güvenli varsayılan: create_approval açıkça istenmedikçe ön izleme.
    if preview or not create_approval:
        result: dict[str, Any] = {
            **base,
            "status": "preview",
            "note": _PREVIEW_NOTE,
        }
    elif report.readiness_verdict != "READY":
        result = {**base, "status": "blocked", "reason": _blocked_reason(report)}
    else:
        approval_id = _create_pending_approval(report)
        result = {
            **base,
            "status": "approval_required",
            "approval_id": approval_id,
            "approve_command": f"uv run achilles approval-approve {approval_id}",
        }

    if write:
        try:
            written = _write_request_reports(result, out_dir)
            result["report_files"] = {k: str(v) for k, v in written.items()}
        except Exception as exc:  # pragma: no cover - rapor yazımı opsiyonel
            log.warning("İstek raporu yazılamadı: %s", exc)
    return result


def _default_out_dir() -> Path:
    return get_settings().root / "reports" / "local_training_orchestrator"


def _write_request_reports(result: dict[str, Any], out_dir: Path | None) -> dict[str, Path]:
    target = out_dir or _default_out_dir()
    target.mkdir(parents=True, exist_ok=True)
    ts = _ts_compact()
    md_path = target / f"{ts}_request.md"
    json_path = target / f"{ts}_request.json"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"md": md_path, "json": json_path}


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Lokal Eğitim İstek Raporu (approval-gated, report-only)",
        "",
        f"> **{result.get('note', REQUEST_BANNER)}**",
        "",
        f"- Üretildi: `{result.get('generated_at')}`",
        f"- **Durum:** `{result.get('status')}`",
        f"- Hazırlık: `{result.get('readiness_score')}/100` (`{result.get('readiness_verdict')}`)",
    ]
    if result.get("status") == "approval_required":
        lines += [
            f"- **Onay ID:** `{result.get('approval_id')}`",
            f"- Onayla: `{result.get('approve_command')}`",
            "- (Onay TEK KULLANIMLIK; bu akış onayı TÜKETMEZ, eğitim BAŞLATMAZ.)",
        ]
    elif result.get("status") == "blocked":
        lines += [f"- **Engel:** {result.get('reason')}"]
    else:
        lines += ["- Ön izleme — onay isteği oluşturulmadı, eğitim başlatılmadı."]

    lines += ["", "## Riskler"]
    risks = result.get("risks") or []
    lines += [f"- {r}" for r in risks] or ["- (risk yok)"]
    footer = "_Bu komut onay isteği OLUŞTURABİLİR ama onayı TÜKETMEZ ve eğitim BAŞLATMAZ._"
    lines += ["", footer, ""]
    return "\n".join(lines)
