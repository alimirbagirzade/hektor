"""İnsan-kapılı gerçek eğitim HANDOFF'u (Phase 5D) — komutu YAZDIRIR, ÇALIŞTIRMAZ.

Akış: 5C dry-run raporunu OKU → (varsa) approval_id'yi READ-ONLY kontrol et → STOP_ALL
+ dry_run_passed + approved_not_consumed ise `ready_for_human_execution` döner ve gerçek
eğitim komutunu **yalnız metin olarak** + bir son checklist verir. Komutu ÇALIŞTIRMAZ.

Tasarım gereği şunları HİÇBİR ZAMAN çağırmaz: ``detached_launch.launch`` ·
``AutoLoRAPipeline.start_training`` · ``promote_to_production`` · ``require_fresh_approval``
· ``request_approval`` · ``subprocess.Popen`` · ``os.system``. Onay TÜKETMEZ; eğitim
BAŞLATMAZ. Gerçek eğitim komutu yalnızca öneri stringi olarak raporlanır.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)

#: Gerçek eğitim komutu — YALNIZ METİN; bu modül ÇALIŞTIRMAZ.
RECOMMENDED_COMMAND = "uv run achilles train --run"
#: Onaylı web alternatifi (yalnız dokümante edilir).
WEB_ALTERNATIVE = "Onaylı web: POST /api/training/run (Phase 4D-1) — otomatik değil, insan eylemi."

_NOTE_READY = "This command was not executed. Human must run it manually."
_NOTE_NOEXEC = "No training command should be executed."

_CHECKLIST = [
    "STOP_ALL is not active",
    "Dry-run passed",
    "Approval is approved and not consumed",
    "You understand this will start real training",
    "You have enough disk/RAM/GPU",
    "You have reviewed dataset/pretrain-gate result",
]


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _ts_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def _default_out_dir() -> Path:
    return get_settings().root / "reports" / "local_training_orchestrator"


def _stop_all_active() -> bool:
    try:
        from app.agents.runtime import supervisor

        return bool(supervisor.is_stop_all_active())
    except Exception:  # pragma: no cover - savunmacı
        return False


def _approval_status(approval_id: str | None) -> dict[str, Any]:
    """Onay durumunu READ-ONLY oku (get_approval). TÜKETMEZ."""
    if not approval_id:
        return {"approval_id": None, "approval_status": "none"}
    try:
        from app.agents.runtime import approvals

        appr = approvals.get_approval(approval_id)
        if appr is None:
            return {"approval_id": approval_id, "approval_status": "not_found"}
        raw = getattr(appr.status, "value", str(appr.status))
        consumed = bool(getattr(appr, "consumed_at", None))
        if raw == "approved":
            label = "approved_consumed" if consumed else "approved_not_consumed"
        else:
            label = str(raw)
        return {"approval_id": approval_id, "approval_status": label}
    except Exception as exc:  # pragma: no cover
        return {"approval_id": approval_id, "approval_status": "error", "error": str(exc)}


def _load_dryrun(dryrun_json: str | Path | None, out_dir: Path | None) -> dict[str, Any]:
    """5C dry-run raporunu READ-ONLY oku (verilen yol ya da en son *_dryrun.json)."""
    path: Path | None = None
    if dryrun_json:
        p = Path(dryrun_json)
        path = p if p.is_absolute() else get_settings().root / p
    else:
        d = out_dir or _default_out_dir()
        try:
            if d.exists():
                cands = sorted(d.glob("*_dryrun.json"))
                path = cands[-1] if cands else None
        except Exception:
            path = None
    if path and path.exists():
        try:
            return {"source": str(path), "data": json.loads(path.read_text(encoding="utf-8"))}
        except Exception as exc:
            return {"source": str(path), "error": str(exc)}
    return {"source": "none"}


def build_handoff(
    *,
    approval_id: str | None = None,
    dryrun_json: str | Path | None = None,
    out_dir: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """İnsan-kapılı eğitim handoff'u üret. KOMUTU ÇALIŞTIRMAZ; onay TÜKETMEZ."""
    dr = _load_dryrun(dryrun_json, out_dir)
    dr_data = dr.get("data") or {}
    dr_status = dr_data.get("status")
    eff_approval_id = approval_id or dr_data.get("approval_id")
    appr = _approval_status(eff_approval_id)
    appr_status = appr.get("approval_status")

    base: dict[str, Any] = {
        "generated_at": _utcnow_iso(),
        "dryrun_source": dr.get("source"),
        "dryrun_status": dr_status,
        "approval_id": appr.get("approval_id"),
        "approval_status": appr_status,
    }

    if not dr.get("data"):
        result: dict[str, Any] = {
            **base,
            "status": "needs_dry_run",
            "reason": "dry-run raporu yok — önce `uv run achilles local-training-dry-run`.",
            "note": _NOTE_NOEXEC,
        }
    elif _stop_all_active():
        result = {**base, "status": "blocked", "reason": "STOP_ALL is active", "note": _NOTE_NOEXEC}
    elif dr_status != "dry_run_passed":
        result = {
            **base,
            "status": "blocked",
            "reason": f"Dry-run did not pass (status={dr_status}).",
            "note": _NOTE_NOEXEC,
        }
    elif appr_status == "approved_consumed":
        result = {
            **base,
            "status": "blocked",
            "reason": "Approval already consumed — request a new approval.",
            "note": _NOTE_NOEXEC,
        }
    elif appr_status != "approved_not_consumed":
        result = {
            **base,
            "status": "needs_approval",
            "reason": f"Approval not ready (status={appr_status}).",
            "hint": f"Önce onayı ver: uv run achilles approval-approve {eff_approval_id or '<id>'}",
            "note": _NOTE_NOEXEC,
        }
    else:
        result = {
            **base,
            "status": "ready_for_human_execution",
            "recommended_command": RECOMMENDED_COMMAND,
            "web_alternative": WEB_ALTERNATIVE,
            "checklist": list(_CHECKLIST),
            "note": _NOTE_READY,
        }

    if write:
        try:
            written = _write_reports(result, out_dir)
            result["report_files"] = {k: str(v) for k, v in written.items()}
        except Exception as exc:  # pragma: no cover
            log.warning("Handoff raporu yazılamadı: %s", exc)
    return result


def _write_reports(result: dict[str, Any], out_dir: Path | None) -> dict[str, Path]:
    target = out_dir or _default_out_dir()
    target.mkdir(parents=True, exist_ok=True)
    ts = _ts_compact()
    md_path = target / f"{ts}_handoff.md"
    json_path = target / f"{ts}_handoff.json"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"md": md_path, "json": json_path}


def render_markdown(result: dict[str, Any]) -> str:
    status = result.get("status")
    lines = [
        "# İnsan-Kapılı Gerçek Eğitim Handoff Raporu",
        "",
        f"> **{result.get('note', _NOTE_NOEXEC)}**",
        "",
        "## Summary",
        f"- Üretildi: `{result.get('generated_at')}`",
        f"- **Durum:** `{status}`",
        "",
        "## Dry-run source",
        f"- `{result.get('dryrun_source')}` (status: `{result.get('dryrun_status')}`)",
        "",
        "## Approval status",
        f"- approval_id: `{result.get('approval_id')}` · "
        f"durum: `{result.get('approval_status')}` (read-only — TÜKETİLMEDİ)",
        "",
        "## Human execution checklist",
    ]
    checklist = result.get("checklist") or []
    lines += [f"- [ ] {c}" for c in checklist] or ["- (uygun değil)"]
    lines += ["", "## Recommended command (YALNIZ METİN — ÇALIŞTIRILMADI)"]
    if status == "ready_for_human_execution":
        lines += [
            "```bash",
            str(result.get("recommended_command", RECOMMENDED_COMMAND)),
            "```",
            f"- Web alternatifi: {result.get('web_alternative', WEB_ALTERNATIVE)}",
        ]
    else:
        lines += ["- (hazır değil — komut önerilmedi)"]
        if result.get("reason"):
            lines += [f"- **Sebep:** {result.get('reason')}"]
        if result.get("hint"):
            lines += [f"- **İpucu:** {result.get('hint')}"]
    lines += [
        "",
        "## Safety warning",
        "- Bu komut bu araç tarafından ÇALIŞTIRILMADI; onay TÜKETİLMEDİ.",
        "- Gerçek eğitimi yalnız insan, ayrı bir adımda elle başlatır.",
        "",
        "## Final verdict",
        f"- `{status}` — {result.get('note', _NOTE_NOEXEC)}",
        "",
    ]
    return "\n".join(lines)
