"""Profil router v1: yalnız validated/production profiller, düşük güvende fallback, loglama."""

from __future__ import annotations

import json
from pathlib import Path

from app.lora.profile_registry import ProfileRecord, ProfileRegistry, ProfileStatus
from app.lora.profile_router import ProfileRouter, detect_domains


def _add(reg: ProfileRegistry, name: str, status: ProfileStatus, method: str = "svd") -> str:
    pid = f"{name}_{method}"
    reg.upsert(
        ProfileRecord(
            profile_id=pid,
            profile_name=name,
            profile_version="1",
            base_model_version="base@r#h",
            rag_version="rag-x",
            adapter_versions={},
            adapter_weights={},
            merge_method=method,
            merge_parameters={},
            profile_hash=f"hash-{pid}",
        )
    )
    # Durumu doğrudan kur (geçiş kuralları registry testlerinde ayrıca sınanır).
    rows = reg.records()
    for r in rows:
        if r.profile_id == pid:
            r.status = status
            r.last_eval_run_id = "run_x"
    from app.lora.mix_common import write_jsonl

    write_jsonl(reg.path, (r.to_dict() for r in rows))
    return pid


def _router(reg: ProfileRegistry, tmp_path: Path) -> ProfileRouter:
    return ProfileRouter(registry=reg, log_path=tmp_path / "router.jsonl")


def test_detect_domains() -> None:
    assert (
        max(
            detect_domains("Sharpe oranının standart hatası ve güven aralığı").items(),
            key=lambda kv: kv[1],
        )[0]
        == "statistics"
    )
    assert detect_domains("merhaba nasılsın") == {}


def test_routes_only_to_validated_profiles(profile_reg: ProfileRegistry, tmp_path: Path) -> None:
    _add(profile_reg, "balanced_v1", ProfileStatus.VALIDATED)
    _add(profile_reg, "statistics_v1", ProfileStatus.EXPERIMENTAL)  # validated değil
    d = _router(profile_reg, tmp_path).route(
        "Bu örneklemde varyans ve korelasyon için güven aralığı nedir?"
    )
    assert d.primary_domain == "statistics"
    assert d.selected_profile == "balanced_v1_svd" and d.used_fallback


def test_validated_domain_profile_selected(profile_reg: ProfileRegistry, tmp_path: Path) -> None:
    _add(profile_reg, "balanced_v1", ProfileStatus.VALIDATED)
    _add(profile_reg, "coding_backtest_v1", ProfileStatus.VALIDATED, "ties")
    d = _router(profile_reg, tmp_path).route("Python ile pandas kullanarak backtest kodu yaz")
    assert d.selected_profile == "coding_backtest_v1_ties" and not d.used_fallback
    assert d.router_confidence >= 0.5


def test_low_confidence_falls_back_to_balanced(
    profile_reg: ProfileRegistry, tmp_path: Path
) -> None:
    _add(profile_reg, "balanced_v1", ProfileStatus.PRODUCTION)
    _add(profile_reg, "trading_analysis_v1", ProfileStatus.VALIDATED)
    d = _router(profile_reg, tmp_path).route("Genel bir değerlendirme yapar mısın?")
    assert d.router_confidence < 0.5
    assert d.selected_profile == "balanced_v1_svd" and d.used_fallback


def test_no_validated_profile_means_base_model(
    profile_reg: ProfileRegistry, tmp_path: Path
) -> None:
    _add(profile_reg, "balanced_v1", ProfileStatus.REJECTED)
    d = _router(profile_reg, tmp_path).route("türev ve integral hesapla")
    assert d.selected_profile is None and d.fallback_profile is None
    assert "base model" in d.reason


def test_router_never_invents_weights(profile_reg: ProfileRegistry, tmp_path: Path) -> None:
    _add(profile_reg, "balanced_v1", ProfileStatus.VALIDATED)
    d = _router(profile_reg, tmp_path).route("momentum stratejisi volatilite drawdown")
    ids = {r.profile_id for r in profile_reg.routable()}
    assert d.selected_profile in ids  # yalnız kayıtlı routable profiller


def test_decision_is_logged(profile_reg: ProfileRegistry, tmp_path: Path) -> None:
    _add(profile_reg, "balanced_v1", ProfileStatus.VALIDATED)
    router = _router(profile_reg, tmp_path)
    router.route("RSI ve MACD ile piyasa momentumu", query_id="q1")
    rows = [
        json.loads(x) for x in (tmp_path / "router.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    for key in (
        "query_id",
        "detected_domains",
        "selected_profile",
        "router_confidence",
        "fallback_profile",
        "created_at",
    ):
        assert key in rows[0]
    assert rows[0]["query_id"] == "q1"
