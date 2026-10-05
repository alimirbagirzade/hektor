"""v15 için kilitli deney, kaynak/aile ayrımı ve eksik kanıtta kapalı kabul kapısı."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

DECODING: dict[str, int | float] = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 16384,
    "num_predict": 4096,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "top_k": 20,
    "top_p": 0.8,
    "min_p": 0,
    "mirostat": 0,
}

ACCEPTANCE: dict[str, Any] = {
    "schema_version": 1,
    "minimum_score_fraction": 0.85,
    "critical_fixture_ids": ["S07", "S09", "S20", "S24", "S28", "S29"],
    "critical_failures_allowed": 0,
    "unmeasured_claims_allowed": 0,
    "core_regressions_allowed": 0,
    "merge_max_kl": 0.01,
    "merge_kl_direction": "KL(adapter+base || merged)",
    "merge_scope_required": "all_prompt_positions",
    "first_attempt_only": True,
    "final_after_checkpoint_lock_only": True,
    "requires_paired_v14_comparison": True,
    "requires_repetition_improvement": True,
}


def common_system_prompt() -> str:
    """Aynı tarihsel geliştirme protokolünün yalnız ortak istemini döndür."""
    from app.evals.llm30 import SYSTEM_PROMPT

    return SYSTEM_PROMPT


def sha256(path: Path) -> str:
    """Büyük ağırlıkları belleğe almadan hash hesapla."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_lock(directory: Path, paths: Sequence[Path]) -> Path:
    """Kilit bir kez oluşturulur; mevcut kilidin üzerine yazılmaz."""
    if not paths:
        raise ValueError("Boş deney kilitlenemez")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "files": {str(p.resolve()): sha256(p) for p in paths},
        "acceptance": ACCEPTANCE,
        "decoding": DECODING,
    }
    lock = directory / "lock.json"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return lock


def verify_lock(lock: Path) -> list[str]:
    """Eksik veya değişmiş dosyayı açık hata olarak döndür."""
    payload = json.loads(lock.read_text(encoding="utf-8"))
    errors = []
    if not payload.get("files"):
        errors.append("Kilit dosya kapsamı boş")
    for name, expected in payload["files"].items():
        path = Path(name)
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"Kilit ihlali: {name}")
    if payload["acceptance"] != ACCEPTANCE or payload["decoding"] != DECODING:
        errors.append("Kilitli kabul/üretim protokolü değişmiş")
    return errors


def split_errors(splits: Mapping[str, Sequence[dict[str, Any]]]) -> list[str]:
    """Kimlik, içerik, kaynak ve şablon ailesi splitler arasında ortak olamaz."""
    owners: dict[tuple[str, str], str] = {}
    errors = []
    if not splits or any(not rows for rows in splits.values()):
        errors.append("Splitler boş olamaz")
    for split, rows in splits.items():
        ids: set[str] = set()
        for row in rows:
            for field in ("id", "source_group", "template_family", "question"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    errors.append(f"{split}: eksik {field}")
            normalized_id = " ".join(str(row.get("id", "")).casefold().split())
            if normalized_id in ids:
                errors.append(f"{split}: yinelenen kimlik {row['id']}")
            ids.add(normalized_id)
            for field in ("id", "source_group", "template_family", "question"):
                value = " ".join(str(row.get(field, "")).casefold().split())
                key = (field, value)
                if key in owners and owners[key] != split:
                    errors.append(f"{field}: {owners[key]} / {split} çakışması: {value}")
                owners[key] = split
    return errors


def transition_matrix(states: Sequence[int], k: int) -> list[list[float]]:
    """Çıkışı olmayan satır NaN; sessiz prior veya düzgün dağılım eklenmez."""
    if k <= 0 or any(type(s) is not int or s < 0 or s >= k for s in states):
        raise ValueError("Durumlar [0,k) aralığında tamsayı olmalı")
    counts = [[0] * k for _ in range(k)]
    for before, after in pairwise(states):
        counts[before][after] += 1
    return [[count / sum(row) for count in row] if sum(row) else [math.nan] * k for row in counts]


def validate_ohlc(values: Sequence[object]) -> bool:
    """Tip, sonluluk ve üç OHLC eşitsizliğini ayrı uygula."""
    if len(values) != 4:
        return False
    if any(
        not isinstance(x, int | float) or isinstance(x, bool) or not math.isfinite(x)
        for x in values
    ):
        return False
    opened, high, low, close = (float(x) for x in values)  # type: ignore[arg-type]
    return high >= max(opened, close) and low <= min(opened, close) and high >= low


def posterior(prior: float, observation: float, p_prior: float, r: float) -> tuple[float, float]:
    """P_prior, Q katkısını zaten içeren toplam önsel varyanstır."""
    if p_prior < 0 or r <= 0 or not all(math.isfinite(x) for x in (prior, observation, p_prior, r)):
        raise ValueError("Sonlu değerler, P>=0 ve R>0 gerekli")
    gain = p_prior / (p_prior + r)
    return gain, prior + gain * (observation - prior)


def decision(evidence: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Yokluk başarı değildir. Teknik fixture başarısı LLM doğruluğu sayılmaz."""
    required = (
        "training_complete",
        "transfer_verified",
        "final_locked",
        "checkpoint_locked",
        "final_complete",
        "blind_review_complete",
        "paired_v14_complete",
        "critical_model_fixtures_pass",
        "repetition_improved",
        "first_attempt_only",
    )
    missing = [key for key in required if evidence.get(key) is not True]
    failures = []
    if evidence.get("transfer_gate_failed") is True:
        failures.append("aktarim_kapisi_basarisiz")
    try:
        checkpoint_time = datetime.fromisoformat(evidence["checkpoint_locked_at"])
        final_time = datetime.fromisoformat(evidence["final_started_at"])
        if checkpoint_time.tzinfo is None or final_time.tzinfo is None:
            missing.append("timezone_aware_checkpoint/final_times")
        elif final_time <= checkpoint_time:
            failures.append("final_before_checkpoint_lock")
    except (KeyError, TypeError, ValueError):
        missing.append("checkpoint/final_times")
    if evidence.get("merge_kl_direction") != ACCEPTANCE["merge_kl_direction"]:
        missing.append("merge_kl_direction")
    expected = evidence.get("expected_question_ids")
    scored = evidence.get("scored_question_ids")
    if (
        not expected
        or not scored
        or len(scored) != len(set(scored))
        or set(expected) != set(scored)
        or len(expected) != 30
    ):
        missing.append("question_coverage")
    for field in ("raw_answers_sha256", "blind_scores_sha256", "paired_v14_sha256"):
        digest = evidence.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            missing.append(field)
    for key in ("critical_errors", "unmeasured_claims", "core_regressions"):
        value = evidence.get(key)
        if type(value) is not int or value < 0:
            missing.append(key)
        elif value:
            failures.append(key)
    score, maximum = evidence.get("score"), evidence.get("maximum")
    if (
        not isinstance(score, int | float)
        or isinstance(score, bool)
        or not isinstance(maximum, int | float)
        or isinstance(maximum, bool)
        or not math.isfinite(score)
        or not math.isfinite(maximum)
        or maximum <= 0
        or not 0 <= score <= maximum
    ):
        missing.append("score/maximum")
    elif score / maximum < ACCEPTANCE["minimum_score_fraction"]:
        failures.append("puan <%85")
    elif expected and maximum != 4 * len(expected):
        missing.append("rubric_denominator")
    dev_score = evidence.get("development_score_fraction")
    if (
        not isinstance(dev_score, int | float)
        or isinstance(dev_score, bool)
        or not 0 <= dev_score <= 1
    ):
        missing.append("development_score_fraction")
    elif dev_score < ACCEPTANCE["minimum_score_fraction"]:
        failures.append("gelistirme_puani <%85")
    kl = evidence.get("merge_max_kl")
    if not isinstance(kl, int | float) or isinstance(kl, bool) or not math.isfinite(kl) or kl < 0:
        missing.append("merge_max_kl")
    elif kl > ACCEPTANCE["merge_max_kl"]:
        failures.append("KL >0.01")
    if evidence.get("merge_scope") != ACCEPTANCE["merge_scope_required"]:
        missing.append("merge_scope")
    if failures:
        return "reddedilen_aday", failures + missing
    if missing:
        return "yetersiz_kanit", missing
    return "kabul_edilen_arastirma_adayi", []


def paired_interval(differences: Iterable[float], *, seed: int = 42) -> dict[str, float | int]:
    """Soru düzeyinde eşlenmiş bootstrap; alt maddeler bağımsız örnek sayılmaz."""
    import numpy as np

    values = np.asarray(list(differences), dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("En az iki sonlu soru farkı gerekli")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n_questions": len(values),
        "mean": float(values.mean()),
        "low": float(low),
        "high": float(high),
        "seed": seed,
    }
