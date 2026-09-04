"""AutoDriver — çevrimdışı testler (sahte runner; gerçek `claude -p` spawn YOK).

deep-hunt'ı claude -p ile sürme + PASS→onaya ilerletme + Kural-8 (onayda durur, eğitim
asla otomatik başlamaz) doğrulanır.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.orchestration import verdict_audit
from app.orchestration.driver import (
    _REPO_ROOT,
    AutoDriver,
    build_hunt_command,
    parse_hunt_verdict,
)
from app.orchestration.orchestrator import RunContext, StageResult, TrainingOrchestrator
from app.orchestration.pipeline import StageStatus
from app.orchestration.store import OrchestrationStore


def _drive_delegates() -> dict:
    def make(status: StageStatus, msg: str):
        def d(ctx: RunContext) -> StageResult:
            return StageResult(status, msg, {})

        return d

    def deep_hunt(ctx: RunContext) -> StageResult:
        if ctx.params.get("hunt_ack"):
            return StageResult(StageStatus.completed, "ack", {})
        return StageResult(StageStatus.blocked, "hunt_ack yok", {})

    return {
        "preflight": make(StageStatus.completed, "preflight ok"),
        "deep-hunt": deep_hunt,
        "data-gate": make(StageStatus.completed, "GO"),
        "curriculum": make(StageStatus.completed, "L0-L4"),
        "dry-run": make(StageStatus.completed, "komut hazır"),
        "approval": make(StageStatus.blocked, "taze onay gerekli"),  # Kural 8 sınırı
        "train": make(StageStatus.blocked, "handoff"),
        "evaluate": make(StageStatus.skipped, "handoff"),
        "registry": make(StageStatus.skipped, "handoff"),
    }


@pytest.fixture
def driver(tmp_path: Path) -> AutoDriver:
    orch = TrainingOrchestrator(
        store=OrchestrationStore(db_path=tmp_path / "drv.db"), delegates=_drive_delegates()
    )
    return AutoDriver(orchestrator=orch)


# Gerçek bir avın üreteceği YAPILANDIRILMIŞ KANIT (P8 var-olma + P9 okuma-kanıtı): depoda
# GERÇEKTEN var olan dosyalar, ≥2 alt-sistem, her dosya için o dosyadan BİREBİR alıntılanmış
# ayırt edici bir satır. Bağımsız denetim (verdict_audit) bunu dosya sistemiyle teyit eder.
# Kanıt gerçek depo dosyalarından ÇALIŞMA ANINDA türetilir → satır numaraları değişse de sağlam.
_HUNT_FILES = [
    "app/orchestration/driver.py",
    "app/orchestration/engines.py",
    "app/orchestration/orchestrator.py",
    "app/web/security.py",
    "app/web/driver_scope.py",
]


def _distinctive_proof(rel: str, used: set[str]) -> dict:
    """Gerçek depo dosyasından, henüz kullanılmamış ayırt edici bir satır + numarasını al."""
    lines = (_REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines, start=1):
        s = ln.strip()
        if len(s) >= verdict_audit.MIN_QUOTE_LEN and s not in used:
            used.add(s)
            return {"path": rel, "line": i, "quote": ln}
    raise AssertionError(f"{rel}: ayırt edici satır bulunamadı")


def _build_valid_evidence() -> str:
    used: set[str] = set()
    scanned = [_distinctive_proof(rel, used) for rel in _HUNT_FILES]
    payload = {"scanned_files": scanned, "subsystems": ["orchestration", "web"], "findings": []}
    return f"ACHILLES_HUNT_EVIDENCE\n{json.dumps(payload, ensure_ascii=False)}\n"


_VALID_EVIDENCE = _build_valid_evidence()


def _fake_pass(
    command: list[str], timeout: int, env: dict[str, str] | None = None
) -> tuple[int, str]:
    return 0, f"denetim raporu...\nciddi bulgu yok\n{_VALID_EVIDENCE}ACHILLES_HUNT_VERDICT: PASS\n"


def _fake_pass_no_evidence(
    command: list[str], timeout: int, env: dict[str, str] | None = None
) -> tuple[int, str]:
    """Motor derin avı yapmadan sadece 'PASS' yazar (P8'in kapatması gereken sınıf)."""
    return 0, "denetim raporu...\nciddi bulgu yok\nACHILLES_HUNT_VERDICT: PASS\n"


def _fake_fail(
    command: list[str], timeout: int, env: dict[str, str] | None = None
) -> tuple[int, str]:
    return 0, "BLOCKER: x\nACHILLES_HUNT_VERDICT: FAIL bir sorun var\n"


# ── saf yardımcılar ───────────────────────────────────────────────────────────


def test_build_command_is_claude_p() -> None:
    cmd = build_hunt_command({"adapter_name": "myad"})
    assert cmd[0] == "claude" and cmd[1] == "-p"
    assert "myad" in cmd[2] and "KADEME-2" in cmd[2]
    assert "EĞİTİM BAŞLATMA" in cmd[2]  # salt-rapor güvencesi promptta


def test_parse_verdict() -> None:
    assert parse_hunt_verdict("...\nACHILLES_HUNT_VERDICT: PASS")["passed"] is True
    assert parse_hunt_verdict("...\nACHILLES_HUNT_VERDICT: FAIL x")["passed"] is False
    assert parse_hunt_verdict("verdict yok")["passed"] is False  # güvenli taraf


# ── drive ─────────────────────────────────────────────────────────────────────


def test_dry_run_does_not_spawn(driver: AutoDriver) -> None:
    run_id = driver.orch.start(model="m", profile="p", adapter_name="a")
    res = driver.drive(run_id, execute=False)
    assert res["ok"] is True
    assert res["drove"] is False
    assert res["dry_run"] is True
    assert res["command"][0] == "claude"
    # deep-hunt hâlâ blocked (sürülmedi)
    by_name = {s["name"]: s for s in res["stages"]}
    assert by_name["deep-hunt"]["status"] == StageStatus.blocked.value


def test_pass_advances_to_approval_and_stops(driver: AutoDriver) -> None:
    run_id = driver.orch.start(model="m", profile="p", adapter_name="a")
    res = driver.drive(run_id, execute=True, runner=_fake_pass)
    assert res["ok"] is True and res["drove"] is True and res["hunt_passed"] is True
    by_name = {s["name"]: s for s in res["stages"]}
    assert by_name["deep-hunt"]["status"] == StageStatus.completed.value
    assert by_name["data-gate"]["status"] == StageStatus.completed.value
    # Kural 8: onay kapısında DURUR; gerçek eğitim ASLA otomatik başlamaz
    assert res["run"]["status"] == "blocked"
    assert res["run"]["current_stage"] == "approval"
    assert by_name["train"]["status"] != StageStatus.completed.value


def test_fake_pass_without_evidence_is_caught(driver: AutoDriver) -> None:
    """P8 (Kural 8): motor derin avı yapmadan 'PASS' yazsa BAĞIMSIZ denetim yakalar.

    Yapılandırılmış kanıt yok → verdict PASS olsa bile hunt_ack YAZILMAZ, deep-hunt bloklu
    kalır. Motorun öz-beyanı artık tek kanıt değildir.
    """
    run_id = driver.orch.start(model="m", profile="p", adapter_name="a")
    res = driver.drive(run_id, execute=True, runner=_fake_pass_no_evidence)
    assert res["drove"] is True and res["hunt_passed"] is False
    assert res["audit"]["ok"] is False
    assert "kanıt" in res["audit"]["reason"].lower()
    by_name = {s["name"]: s for s in res["stages"]}
    assert by_name["deep-hunt"]["status"] == StageStatus.blocked.value
    # KRİTİK: sahte PASS eğitim kapısını AÇMADI.
    params = driver.orch.store.get_run(run_id).get("params") or {}
    assert params.get("hunt_ack") is not True


def test_pass_with_valid_evidence_reports_audit(driver: AutoDriver) -> None:
    """Gerçek av PASS'i bağımsız denetimden geçer → audit sayıları döner."""
    run_id = driver.orch.start(model="m", profile="p", adapter_name="a")
    res = driver.drive(run_id, execute=True, runner=_fake_pass)
    assert res["hunt_passed"] is True
    assert res["audit"]["ok"] is True
    assert res["audit"]["scanned_count"] >= 5 and res["audit"]["subsystem_count"] >= 2


def test_fail_keeps_hunt_blocked(driver: AutoDriver) -> None:
    run_id = driver.orch.start(model="m", profile="p", adapter_name="a")
    res = driver.drive(run_id, execute=True, runner=_fake_fail)
    assert res["drove"] is True and res["hunt_passed"] is False
    by_name = {s["name"]: s for s in res["stages"]}
    assert by_name["deep-hunt"]["status"] == StageStatus.blocked.value
    assert by_name["approval"]["status"] == StageStatus.pending.value  # ulaşmadı


def test_runner_exception_is_safe(driver: AutoDriver) -> None:
    def boom(
        command: list[str], timeout: int, env: dict[str, str] | None = None
    ) -> tuple[int, str]:
        raise RuntimeError("spawn patladı")

    run_id = driver.orch.start(model="m", profile="p", adapter_name="a")
    res = driver.drive(run_id, execute=True, runner=boom)
    assert res["ok"] is False
    assert "patladı" in res["reason"]
    # koşu çökmez; deep-hunt bloklu kalır
    assert driver.orch.store.get_stage(run_id, "deep-hunt")["status"] == StageStatus.blocked.value


def test_unknown_run_returns_error(driver: AutoDriver) -> None:
    res = driver.drive("orc_yok", execute=False)
    assert res["ok"] is False


# ── motor seçimi (kayıt tablosu) ──────────────────────────────────────────────


def test_build_command_honors_engine() -> None:
    """Varsayılan claude; başka motor seçilince o motorun argv şablonu kurulur."""
    assert build_hunt_command({"adapter_name": "a"})[:2] == ["claude", "-p"]
    assert build_hunt_command({"adapter_name": "a"}, "codex")[:2] == ["codex", "exec"]


def test_drive_rejects_unknown_engine(driver: AutoDriver) -> None:
    run_id = driver.orch.start(model="m", profile="p", adapter_name="a")
    res = driver.drive(run_id, execute=True, engine="uydurma-motor", runner=_fake_pass)
    assert res["ok"] is False and "Bilinmeyen motor" in res["reason"]
    # Motor adı HER ŞEYDEN ÖNCE doğrulanır → koşuya hiç dokunulmaz (deep-hunt pending kalır).
    assert driver.orch.store.get_stage(run_id, "deep-hunt")["status"] == StageStatus.pending.value


def test_drive_rejects_non_spawning_engine(driver: AutoDriver) -> None:
    """`local` motor yoktur (doğrudan Ollama hattı) → otonom sürüşte kullanılamaz."""
    run_id = driver.orch.start(model="m", profile="p", adapter_name="a")
    res = driver.drive(run_id, execute=True, engine="local", runner=_fake_pass)
    assert res["ok"] is False and "süreç başlatmaz" in res["reason"]


def test_dry_run_reports_engine_quota_warning(driver: AutoDriver) -> None:
    """P5 UI'ı kota uyarısını buradan gösterecek."""
    run_id = driver.orch.start(model="m", profile="p", adapter_name="a")
    res = driver.drive(run_id, execute=False, engine="codex")
    assert res["engine"] == "codex" and res["command"][:2] == ["codex", "exec"]
    assert "5 saatlik" in res["quota_warning"]
