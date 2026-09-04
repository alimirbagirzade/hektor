from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_DB = Path(__file__).parent.parent.parent.parent / "storage" / "achilles_learning.db"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect(db_path: Path = _DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(db_path: Path = _DEFAULT_DB) -> None:
    """Tabloları oluştur (idempotent)."""
    conn = _connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS system_profiles (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            os TEXT,
            cpu TEXT,
            ram_gb REAL,
            gpu_vendor TEXT,
            gpu_name TEXT,
            vram_gb REAL,
            raw_json TEXT
        );

        CREATE TABLE IF NOT EXISTS model_trials (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            system_profile_id TEXT,
            model_id TEXT,
            backend TEXT,
            quantization TEXT,
            context_size INTEGER,
            status TEXT,
            tokens_per_second REAL,
            first_token_latency_ms REAL,
            peak_ram_gb REAL,
            peak_vram_gb REAL,
            quality_score REAL,
            raw_json TEXT,
            FOREIGN KEY (system_profile_id) REFERENCES system_profiles(id)
        );

        CREATE TABLE IF NOT EXISTS error_patterns (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            error_signature TEXT,
            error_type TEXT,
            probable_cause TEXT,
            recommended_fix TEXT,
            confidence REAL,
            raw_log TEXT
        );

        CREATE TABLE IF NOT EXISTS rule_suggestions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            rule_file TEXT,
            proposed_patch TEXT,
            reason TEXT,
            status TEXT DEFAULT 'pending_review'
        );
    """)
    conn.commit()
    conn.close()


def save_system_profile(profile_dict: dict[str, Any], db_path: Path = _DEFAULT_DB) -> str:
    """Sistem profilini kaydet. ID döndür."""
    init_schema(db_path)
    conn = _connect(db_path)
    pid = str(uuid.uuid4())
    conn.execute(
        """INSERT OR REPLACE INTO system_profiles
           (id, created_at, os, cpu, ram_gb, gpu_vendor, gpu_name, vram_gb, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pid,
            _now(),
            profile_dict.get("os"),
            profile_dict.get("cpu", {}).get("name")
            if isinstance(profile_dict.get("cpu"), dict)
            else profile_dict.get("cpu"),
            profile_dict.get("memory", {}).get("ram_total_gb", 0)
            if isinstance(profile_dict.get("memory"), dict)
            else profile_dict.get("ram_gb", 0),
            profile_dict.get("gpu", {}).get("vendor")
            if isinstance(profile_dict.get("gpu"), dict)
            else None,
            profile_dict.get("gpu", {}).get("name")
            if isinstance(profile_dict.get("gpu"), dict)
            else None,
            profile_dict.get("gpu", {}).get("vram_gb", 0)
            if isinstance(profile_dict.get("gpu"), dict)
            else 0,
            json.dumps(profile_dict, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()
    return pid


def save_model_trial(
    system_profile_id: str,
    model_id: str,
    backend: str,
    status: str,
    tokens_per_second: float = 0.0,
    first_token_latency_ms: float = 0.0,
    peak_ram_gb: float = 0.0,
    quality_score: float = 0.0,
    context_size: int = 4096,
    extra: dict[str, Any] | None = None,
    db_path: Path = _DEFAULT_DB,
) -> str:
    """Benchmark sonucunu kaydet."""
    init_schema(db_path)
    conn = _connect(db_path)
    tid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO model_trials
           (id, created_at, system_profile_id, model_id, backend, context_size,
            status, tokens_per_second, first_token_latency_ms, peak_ram_gb, quality_score, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            tid,
            _now(),
            system_profile_id,
            model_id,
            backend,
            context_size,
            status,
            tokens_per_second,
            first_token_latency_ms,
            peak_ram_gb,
            quality_score,
            json.dumps(extra or {}, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()
    return tid


def save_error_pattern(
    error_signature: str,
    error_type: str,
    probable_cause: str,
    recommended_fix: str,
    confidence: float = 0.5,
    raw_log: str = "",
    db_path: Path = _DEFAULT_DB,
) -> str:
    init_schema(db_path)
    conn = _connect(db_path)
    eid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO error_patterns
           (id, created_at, error_signature, error_type,
            probable_cause, recommended_fix, confidence, raw_log)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            eid,
            _now(),
            error_signature,
            error_type,
            probable_cause,
            recommended_fix,
            confidence,
            raw_log,
        ),
    )
    conn.commit()
    conn.close()
    return eid


def save_rule_suggestion(
    rule_file: str,
    proposed_patch: str,
    reason: str,
    db_path: Path = _DEFAULT_DB,
) -> str:
    """Kural öneri olarak kaydet (direkt uygulamaz)."""
    init_schema(db_path)
    conn = _connect(db_path)
    rid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO rule_suggestions (id, created_at, rule_file, proposed_patch, reason)
           VALUES (?, ?, ?, ?, ?)""",
        (rid, _now(), rule_file, proposed_patch, reason),
    )
    conn.commit()
    conn.close()
    return rid


def list_model_trials(limit: int = 20, db_path: Path = _DEFAULT_DB) -> list[dict[str, Any]]:
    init_schema(db_path)
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM model_trials ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_model_success_rate(model_id: str, db_path: Path = _DEFAULT_DB) -> float:
    """Bu modelin geçmişteki başarı oranı."""
    init_schema(db_path)
    conn = _connect(db_path)
    total = conn.execute(
        "SELECT COUNT(*) FROM model_trials WHERE model_id = ?", (model_id,)
    ).fetchone()[0]
    if total == 0:
        conn.close()
        return 0.5  # belirsiz → nötr
    success = conn.execute(
        "SELECT COUNT(*) FROM model_trials WHERE model_id = ?"
        " AND status IN ('excellent','usable','slow_but_usable')",
        (model_id,),
    ).fetchone()[0]
    conn.close()
    return round(success / total, 2)
