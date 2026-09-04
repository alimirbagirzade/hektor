"""rules_updater.py — Başarısız trial'lardan kural önerisi üretir.

Hiçbir şeyi otomatik değiştirmez: sadece `rule_suggestions` tablosuna
`status='pending_review'` kayıt ekler. Kullanıcı `achilles rules-update`
komutuyla önerileri gözden geçirir ve onaylar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agents.learning.memory import (
    _DEFAULT_DB,
    _connect,
    init_schema,
    save_rule_suggestion,
)

# Eşikler
_MIN_FAILURES_FOR_BLACKLIST = 3
_MIN_FAILURES_FOR_THROTTLE = 2
_MIN_ERROR_OCCURRENCES = 2
_MIN_TPS_THRESHOLD = 5.0


@dataclass
class TrialSummary:
    model_id: str
    total: int
    failed: int
    unstable: int
    avg_ram_gb: float
    avg_tps: float
    avg_quality: float

    @property
    def failure_rate(self) -> float:
        return round((self.failed + self.unstable) / self.total, 2) if self.total else 0.0


@dataclass
class RuleSuggestion:
    suggestion_id: str
    rule_file: str
    proposed_patch: str
    reason: str
    status: str


def _query_trial_summaries(db_path: Path) -> list[TrialSummary]:
    """model_trials tablosunu model bazında özetle."""
    conn = _connect(db_path)
    rows = conn.execute("""
        SELECT
            model_id,
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN status = 'unstable' THEN 1 ELSE 0 END) AS unstable,
            AVG(COALESCE(peak_ram_gb, 0)) AS avg_ram,
            AVG(COALESCE(tokens_per_second, 0)) AS avg_tps,
            AVG(COALESCE(quality_score, 0)) AS avg_quality
        FROM model_trials
        GROUP BY model_id
    """).fetchall()
    conn.close()
    return [
        TrialSummary(
            model_id=r["model_id"],
            total=r["total"],
            failed=r["failed"],
            unstable=r["unstable"],
            avg_ram_gb=round(r["avg_ram"] or 0.0, 2),
            avg_tps=round(r["avg_tps"] or 0.0, 1),
            avg_quality=round(r["avg_quality"] or 0.0, 2),
        )
        for r in rows
    ]


def _query_error_patterns(db_path: Path, min_occurrences: int) -> list[dict[str, Any]]:
    """Tekrar eden hata imzalarını bul."""
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT error_signature, error_type, probable_cause,
               recommended_fix, COUNT(*) AS cnt,
               AVG(confidence) AS avg_conf
        FROM error_patterns
        GROUP BY error_signature
        HAVING cnt >= ?
        ORDER BY cnt DESC
    """,
        (min_occurrences,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _existing_suggestion_keys(db_path: Path) -> set[str]:
    """Daha önce önerilmiş (pending/approved) patch'leri döndür — tekrar öneri önler."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT proposed_patch FROM rule_suggestions WHERE status IN ('pending_review','approved')"
    ).fetchall()
    conn.close()
    return {r["proposed_patch"] for r in rows}


def analyze_failed_trials(
    min_failures: int = _MIN_FAILURES_FOR_BLACKLIST,
    db_path: Path = _DEFAULT_DB,
) -> list[TrialSummary]:
    """Yeterince başarısız trial'ı olan modelleri listele."""
    init_schema(db_path)
    summaries = _query_trial_summaries(db_path)
    return [s for s in summaries if (s.failed + s.unstable) >= min_failures]


def generate_rule_suggestions(
    db_path: Path = _DEFAULT_DB,
    dry_run: bool = False,
) -> list[RuleSuggestion]:
    """Trial + hata verilerinden kural önerileri üret; DB'ye yaz (dry_run=True ise yazmaz)."""
    init_schema(db_path)
    existing = _existing_suggestion_keys(db_path)
    generated: list[RuleSuggestion] = []

    # --- 1. Model-bazlı başarısızlık önerileri ---
    for s in _query_trial_summaries(db_path):
        total_bad = s.failed + s.unstable

        if total_bad >= _MIN_FAILURES_FOR_BLACKLIST:
            patch = json.dumps(
                {
                    "action": "blacklist_model",
                    "model_id": s.model_id,
                    "failure_count": total_bad,
                    "total_trials": s.total,
                },
                ensure_ascii=False,
            )
            reason = (
                f"Model '{s.model_id}' {total_bad}/{s.total} denemede başarısız/kararsız. "
                f"Ortalama kalite: {s.avg_quality:.2f}. "
                "model_registry.yaml'a 'blacklisted: true' eklenmesi önerilir."
            )
            if patch not in existing:
                sid = _write_suggestion("model_registry.yaml", patch, reason, dry_run, db_path)
                generated.append(
                    RuleSuggestion(sid, "model_registry.yaml", patch, reason, "pending_review")
                )

        elif total_bad >= _MIN_FAILURES_FOR_THROTTLE and s.avg_tps < _MIN_TPS_THRESHOLD:
            patch = json.dumps(
                {
                    "action": "throttle_model",
                    "model_id": s.model_id,
                    "observed_avg_tps": s.avg_tps,
                    "suggested_min_tps": _MIN_TPS_THRESHOLD,
                },
                ensure_ascii=False,
            )
            reason = (
                f"Model '{s.model_id}' yavaş ({s.avg_tps:.1f} tok/sn) ve "
                f"{total_bad} kararsız trial mevcut. "
                "Advisor'a minimum TPS kısıtı eklenmesi önerilir."
            )
            if patch not in existing:
                sid = _write_suggestion("advisor_rules.yaml", patch, reason, dry_run, db_path)
                generated.append(
                    RuleSuggestion(sid, "advisor_rules.yaml", patch, reason, "pending_review")
                )

        if s.avg_ram_gb > 0 and s.total >= 2:
            patch = json.dumps(
                {
                    "action": "add_ram_warning",
                    "model_id": s.model_id,
                    "observed_avg_ram_gb": s.avg_ram_gb,
                },
                ensure_ascii=False,
            )
            reason = (
                f"Model '{s.model_id}' ortalama {s.avg_ram_gb:.1f} GB RAM kullandı. "
                "model_registry.yaml'daki 'min_ram_gb' değerinin gözden geçirilmesi önerilir."
            )
            if patch not in existing:
                sid = _write_suggestion("model_registry.yaml", patch, reason, dry_run, db_path)
                generated.append(
                    RuleSuggestion(sid, "model_registry.yaml", patch, reason, "pending_review")
                )

    # --- 2. Tekrar eden hata desenleri ---
    for p in _query_error_patterns(db_path, _MIN_ERROR_OCCURRENCES):
        patch = json.dumps(
            {
                "action": "add_known_error",
                "error_signature": p["error_signature"],
                "error_type": p["error_type"],
                "recommended_fix": p["recommended_fix"],
                "occurrences": p["cnt"],
            },
            ensure_ascii=False,
        )
        reason = (
            f"Hata '{p['error_signature']}' {p['cnt']} kez tekrar etti "
            f"(tür: {p['error_type']}). "
            f"Önerilen düzeltme: {p['recommended_fix']}. "
            "error_catalog.yaml'a eklenmesi önerilir."
        )
        if patch not in existing:
            sid = _write_suggestion("error_catalog.yaml", patch, reason, dry_run, db_path)
            generated.append(
                RuleSuggestion(sid, "error_catalog.yaml", patch, reason, "pending_review")
            )

    return generated


def _write_suggestion(
    rule_file: str,
    patch: str,
    reason: str,
    dry_run: bool,
    db_path: Path,
) -> str:
    if dry_run:
        return f"dry_{abs(hash(patch)) % 100000}"
    return save_rule_suggestion(rule_file, patch, reason, db_path)


def list_pending_suggestions(db_path: Path = _DEFAULT_DB) -> list[RuleSuggestion]:
    """Onay bekleyen önerileri listele."""
    init_schema(db_path)
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT id, rule_file, proposed_patch, reason, status "
        "FROM rule_suggestions WHERE status = 'pending_review' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [
        RuleSuggestion(r["id"], r["rule_file"], r["proposed_patch"], r["reason"], r["status"])
        for r in rows
    ]


def approve_suggestion(suggestion_id: str, db_path: Path = _DEFAULT_DB) -> bool:
    """Öneriyi 'approved' olarak işaretle (patch'i uygulamaz)."""
    init_schema(db_path)
    conn = _connect(db_path)
    cursor = conn.execute(
        "UPDATE rule_suggestions SET status = 'approved' "
        "WHERE id = ? AND status = 'pending_review'",
        (suggestion_id,),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def dismiss_suggestion(suggestion_id: str, db_path: Path = _DEFAULT_DB) -> bool:
    """Öneriyi 'dismissed' olarak işaretle."""
    init_schema(db_path)
    conn = _connect(db_path)
    cursor = conn.execute(
        "UPDATE rule_suggestions SET status = 'dismissed' "
        "WHERE id = ? AND status = 'pending_review'",
        (suggestion_id,),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0
