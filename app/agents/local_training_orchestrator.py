"""Lokal eğitim-denetim orkestratörü (Phase 5A) — SALT RAPOR.

Bu modül Achilles'in eğitim hazırlık durumunu **okur** ve bir rapor üretir.
GERÇEK EĞİTİM BAŞLATMAZ. Tasarım gereği şunları HİÇBİR ZAMAN çağırmaz:
``detached_launch.launch`` · ``AutoLoRAPipeline.start_training`` ·
``promote_to_production`` · ``achilles train --run``. Onay TÜKETMEZ (yalnız listeler).
Cloud/Kaggle/Colab tetiklemez. Korumalı yollara (data/storage/vector_db/models/.env)
YAZMAZ; yalnız ``reports/`` altına markdown + json rapor yazar.

Her sonda (probe) savunmacıdır: bir bağımlılık yoksa/başarısızsa "unavailable"
durumu döner, audit'i çökertmez. Çıktı: eğitim-hazırlık skoru + riskler + öneriler.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)

#: Komutun ve raporun başına basılan değişmez güvence.
REPORT_ONLY_BANNER = "No training was started. This is a local report-only audit."

#: Eğitim onayı için CLI/web ile AYNI anahtar (yalnız OKUMA için kullanılır).
_TRAIN_AGENT_ID = "lora-trainer"
_TRAIN_ACTION = "train_run"


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _ts_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


@dataclass
class TrainingAuditReport:
    """Lokal eğitim-denetim raporu (salt-okuma)."""

    generated_at: str
    banner: str
    probes: dict[str, Any] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    readiness_score: int = 0
    readiness_verdict: str = "NOT_READY"  # READY | NOT_READY | BLOCKED
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Read-only probes — her biri try/except ile savunmacı; asla fırlatmaz.
# --------------------------------------------------------------------------
def _probe_stop_all() -> dict[str, Any]:
    try:
        from app.agents.runtime import supervisor

        active = bool(supervisor.is_stop_all_active())
        return {"available": True, "stop_all_active": active}
    except Exception as exc:  # pragma: no cover - savunmacı
        return {"available": False, "error": str(exc)}


def _probe_detached_training() -> dict[str, Any]:
    try:
        from app.training.detached_launch import is_detached_training_running

        return {"available": True, "running": bool(is_detached_training_running())}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}


def _probe_approvals() -> dict[str, Any]:
    """Bekleyen onayları LİSTELER + taze train_run onayı VAR MI bakar. TÜKETMEZ."""
    try:
        from app.agents.runtime import approvals

        pending = approvals.list_approvals(status="pending")
        has_fresh = approvals.has_fresh_approval(_TRAIN_AGENT_ID, _TRAIN_ACTION)
        return {
            "available": True,
            "pending_count": len(pending),
            "pending": [
                {
                    "approval_id": a.approval_id,
                    "agent_id": a.agent_id,
                    "action": a.action,
                    "risk": str(a.risk),
                }
                for a in pending
            ],
            "fresh_train_run_approval": bool(has_fresh),
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}


def _probe_auto_lora() -> dict[str, Any]:
    try:
        from app.lora.auto_pipeline import get_auto_pipeline

        return {"available": True, "status": get_auto_pipeline().get_status()}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
    except Exception:
        return 0


def _probe_data_readiness() -> dict[str, Any]:
    s = get_settings()
    sft = s.root / "data" / "lora_sft" / "lora_sft.jsonl"
    train = s.jsonl_dir / "train.jsonl"
    valid = s.jsonl_dir / "valid.jsonl"
    approved = -1
    try:
        from app.memory.sqlite_store import SqliteStore

        approved = len(SqliteStore().list_approved_cards())
    except Exception:
        approved = -1
    return {
        "available": True,
        "lora_sft_lines": _count_lines(sft),
        "train_jsonl_lines": _count_lines(train),
        "valid_jsonl_lines": _count_lines(valid),
        "approved_cards": approved,
    }


def _probe_pretrain_gate() -> dict[str, Any]:
    """pretrain-gate'i SALT-OKUMA çağır (audit_dataset) → GO/NO-GO. Eğitim başlatmaz."""
    s = get_settings()
    sft = s.root / "data" / "lora_sft" / "lora_sft.jsonl"
    if not sft.exists():
        return {"available": False, "reason": "lora_sft.jsonl yok — kapı atlandı"}
    try:
        from app.training.dataset_quality import audit_dataset

        lines = [ln for ln in sft.read_text(encoding="utf-8").splitlines() if ln.strip()]
        disc = None
        try:
            from app.training.discipline_dataset import discipline_jsonl_lines

            disc = discipline_jsonl_lines()
        except Exception:
            disc = None
        report = audit_dataset(lines, discipline_lines=disc)
        return {
            "available": True,
            "verdict": report.verdict,
            "total": report.total,
            "blockers": list(report.blockers),
            "warnings": list(report.warnings),
            "recommended_epochs": report.recommended_epochs,
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}


def _probe_adapter_eval_readiness() -> dict[str, Any]:
    s = get_settings()
    adapters: list[str] = []
    try:
        if s.adapters_dir.exists():
            adapters = sorted(p.name for p in s.adapters_dir.iterdir() if p.is_dir())
    except Exception:
        adapters = []
    return {
        "available": True,
        "adapters_present": len(adapters),
        "adapters": adapters[:20],
        "note": "Eval BAŞLATILMADI; yalnız adapter varlığı raporlandı.",
    }


# --------------------------------------------------------------------------
# Skorlama + risk sentezi (heuristik, advisory)
# --------------------------------------------------------------------------
def _score_and_risks(probes: dict[str, Any]) -> tuple[int, str, list[str]]:
    risks: list[str] = []
    score = 0

    stop = probes.get("stop_all", {})
    blocked = False
    if stop.get("stop_all_active"):
        risks.append("🛑 STOP_ALL AKTİF — tüm tehlikeli aksiyonlar (eğitim/terfi) bloklu.")
        blocked = True
    else:
        score += 15

    detached = probes.get("detached_training", {})
    if detached.get("running"):
        risks.append("Eğitim ZATEN çalışıyor (detached) — yeni koşu başlatılmamalı.")
    else:
        score += 15

    data = probes.get("data_readiness", {})
    n = int(data.get("lora_sft_lines", 0) or 0)
    if n <= 0:
        risks.append("Eğitim verisi yok (lora_sft.jsonl boş/eksik) — önce sentetik veri üret.")
        blocked = True
    elif n < 500:
        risks.append(f"Eğitim verisi az ({n} satır < 500) — overfit riski; daha çok veri öner.")
        score += 20
    else:
        score += 40

    gate = probes.get("pretrain_gate", {})
    if gate.get("available"):
        if gate.get("verdict") == "GO":
            score += 30
        else:
            for b in gate.get("blockers", []):
                risks.append(f"Kalite kapısı NO-GO: {b}")
            blocked = True
    else:
        risks.append("Ön eğitim kalite kapısı çalıştırılamadı — veri hazır değil veya eksik.")

    if blocked:
        verdict = "BLOCKED"
    elif score >= 70:
        verdict = "READY"
    else:
        verdict = "NOT_READY"
    return min(score, 100), verdict, risks


# --------------------------------------------------------------------------
# Orkestrasyon + rapor yazımı
# --------------------------------------------------------------------------
def run_audit(out_dir: Path | None = None, *, write: bool = True) -> TrainingAuditReport:
    """Tüm read-only sondaları çalıştır, raporu oluştur (ve istenirse yaz).

    GERÇEK EĞİTİM BAŞLATMAZ. Onay tüketmez. Korumalı yollara yazmaz.
    """
    probes = {
        "stop_all": _probe_stop_all(),
        "detached_training": _probe_detached_training(),
        "approvals": _probe_approvals(),
        "auto_lora": _probe_auto_lora(),
        "data_readiness": _probe_data_readiness(),
        "pretrain_gate": _probe_pretrain_gate(),
        "adapter_eval_readiness": _probe_adapter_eval_readiness(),
    }
    score, verdict, risks = _score_and_risks(probes)

    notes = [
        REPORT_ONLY_BANNER,
        "Bu denetim onay TÜKETMEZ; bekleyen onaylar yalnız listelenir.",
        "Gerçek eğitim için açık taze onay + `achilles train --run` (ya da onaylı web) gerekir.",
    ]
    report = TrainingAuditReport(
        generated_at=_utcnow_iso(),
        banner=REPORT_ONLY_BANNER,
        probes=probes,
        risks=risks,
        readiness_score=score,
        readiness_verdict=verdict,
        notes=notes,
    )

    if write:
        try:
            written = _write_reports(report, out_dir)
            report.notes.append(f"Rapor yazıldı: {written['md']} · {written['json']}")
        except Exception as exc:  # pragma: no cover - rapor yazımı opsiyonel
            log.warning("Rapor yazılamadı: %s", exc)
            report.notes.append(f"Rapor yazılamadı: {exc}")
    return report


def _default_out_dir() -> Path:
    return get_settings().root / "reports" / "local_training_orchestrator"


def _write_reports(report: TrainingAuditReport, out_dir: Path | None) -> dict[str, Path]:
    target = out_dir or _default_out_dir()
    target.mkdir(parents=True, exist_ok=True)
    ts = _ts_compact()
    md_path = target / f"{ts}_report.md"
    json_path = target / f"{ts}_report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"md": md_path, "json": json_path}


def render_markdown(report: TrainingAuditReport) -> str:
    p = report.probes
    data = p.get("data_readiness", {})
    gate = p.get("pretrain_gate", {})
    appr = p.get("approvals", {})
    lines = [
        "# Lokal Eğitim-Denetim Raporu (report-only)",
        "",
        f"> **{report.banner}**",
        "",
        f"- Üretildi: `{report.generated_at}`",
        f"- **Hazırlık skoru:** {report.readiness_score}/100",
        f"- **Karar:** `{report.readiness_verdict}`",
        "",
        "## Durum",
        f"- STOP_ALL aktif: `{p.get('stop_all', {}).get('stop_all_active')}`",
        f"- Detached eğitim çalışıyor: `{p.get('detached_training', {}).get('running')}`",
        f"- Bekleyen onay: `{appr.get('pending_count', '?')}` "
        f"(taze train_run onayı: `{appr.get('fresh_train_run_approval')}`)",
        f"- lora_sft satır: `{data.get('lora_sft_lines')}` · "
        f"train/valid: `{data.get('train_jsonl_lines')}`/`{data.get('valid_jsonl_lines')}` · "
        f"onaylı kart: `{data.get('approved_cards')}`",
        f"- Kalite kapısı: `{gate.get('verdict', gate.get('reason', 'n/a'))}` "
        f"(öneri: `{gate.get('recommended_epochs', '-')}` epoch)",
        f"- Adapter mevcut: `{p.get('adapter_eval_readiness', {}).get('adapters_present')}`",
        "",
        "## Riskler",
    ]
    lines += [f"- {r}" for r in report.risks] or ["- (risk yok)"]
    lines += ["", "## Notlar"]
    lines += [f"- {n}" for n in report.notes]
    lines += [
        "",
        "## Bekleyen onaylar (tüketilmedi)",
    ]
    pend = appr.get("pending", [])
    if pend:
        lines += [
            f"- `{a['approval_id']}` — {a['agent_id']}/{a['action']} (risk: {a['risk']})"
            for a in pend
        ]
    else:
        lines += ["- (yok)"]
    lines += ["", "_Bu araç salt rapordur; hiçbir eğitim/terfi başlatmaz._", ""]
    return "\n".join(lines)
