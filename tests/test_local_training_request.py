"""Phase 5B — onay-kapılı lokal eğitim İSTEĞİ akışı testleri (offline; GERÇEK eğitim YOK).

İstek akışı bir PENDING onay isteği OLUŞTURABİLİR ama onayı TÜKETMEZ ve eğitim
BAŞLATMAZ. Bu testler doğrular: default=preview (onay yok), --create-approval+READY →
approval_required, STOP_ALL/risk/READY-değil → blocked; launch / train subprocess /
start_training / promote_to_production / require_fresh_approval HİÇBİR ZAMAN çağrılmaz.
Mock kullanılır; ağ/cloud/training yok.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agents import local_training_orchestrator as lto
from app.agents import local_training_request as ltr

_RUN_AUDIT = "app.agents.local_training_orchestrator.run_audit"
_REQUEST = "app.agents.runtime.approvals.request_approval"
_REQUIRE = "app.agents.runtime.approvals.require_fresh_approval"
_LAUNCH = "app.training.detached_launch.launch"
_START = "app.lora.auto_pipeline.AutoLoRAPipeline.start_training"
_PROMOTE = "app.lora.auto_pipeline.AutoLoRAPipeline.promote_to_production"

_MODULE_SRC = Path(ltr.__file__).read_text(encoding="utf-8")


def _report(verdict: str = "READY", score: int = 82, risks: list[str] | None = None):
    return lto.TrainingAuditReport(
        generated_at="2026-06-19T00:00:00+00:00",
        banner=lto.REPORT_ONLY_BANNER,
        probes={},
        risks=risks or [],
        readiness_score=score,
        readiness_verdict=verdict,
        notes=[],
    )


def test_default_is_preview() -> None:
    with patch(_RUN_AUDIT, return_value=_report()):
        res = ltr.build_request(create_approval=False, write=False)
    assert res["status"] == "preview"
    assert res["note"] == "No approval request was created. No training was started."


def test_preview_mode_creates_no_approval() -> None:
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_REQUEST) as m_request,
    ):
        res = ltr.build_request(create_approval=True, preview=True, write=False)
    assert res["status"] == "preview"
    m_request.assert_not_called()


def test_create_approval_when_ready_creates_pending() -> None:
    fake = SimpleNamespace(approval_id="apr_abc123")
    with (
        patch(_RUN_AUDIT, return_value=_report(verdict="READY", score=82)),
        patch(_REQUEST, return_value=fake) as m_request,
        patch(_REQUIRE) as m_require,
    ):
        res = ltr.build_request(create_approval=True, write=False)
    assert res["status"] == "approval_required"
    assert res["approval_id"] == "apr_abc123"
    assert res["approve_command"] == "uv run achilles approval-approve apr_abc123"
    m_request.assert_called_once()
    m_require.assert_not_called()  # onay TÜKETİLMEDİ


def test_create_approval_blocked_when_stop_all() -> None:
    blocked = _report(verdict="BLOCKED", score=15, risks=["🛑 STOP_ALL AKTİF — bloklu."])
    with (
        patch(_RUN_AUDIT, return_value=blocked),
        patch(_REQUEST) as m_request,
    ):
        res = ltr.build_request(create_approval=True, write=False)
    assert res["status"] == "blocked"
    assert "STOP_ALL" in res["reason"]
    m_request.assert_not_called()


def test_create_approval_blocked_when_not_ready() -> None:
    nr = _report(verdict="NOT_READY", score=40, risks=["Eğitim verisi az."])
    with (
        patch(_RUN_AUDIT, return_value=nr),
        patch(_REQUEST) as m_request,
    ):
        res = ltr.build_request(create_approval=True, write=False)
    assert res["status"] == "blocked"
    m_request.assert_not_called()


def test_no_training_started_even_when_approval_created() -> None:
    fake = SimpleNamespace(approval_id="apr_x")
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_REQUEST, return_value=fake),
        patch(_LAUNCH) as m_launch,
        patch(_START) as m_start,
        patch(_PROMOTE) as m_promote,
        patch("subprocess.Popen") as m_popen,
    ):
        res = ltr.build_request(create_approval=True, write=False)
    assert res["status"] == "approval_required"
    m_launch.assert_not_called()
    m_start.assert_not_called()
    m_promote.assert_not_called()
    m_popen.assert_not_called()


def test_launch_never_called_in_preview() -> None:
    with patch(_RUN_AUDIT, return_value=_report()), patch(_LAUNCH) as m_launch:
        ltr.build_request(create_approval=False, write=False)
    m_launch.assert_not_called()


def test_subprocess_never_spawned() -> None:
    with patch(_RUN_AUDIT, return_value=_report()), patch("subprocess.Popen") as m_popen:
        ltr.build_request(create_approval=True, write=False)
    m_popen.assert_not_called()


def test_require_fresh_approval_never_called() -> None:
    fake = SimpleNamespace(approval_id="apr_y")
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_REQUEST, return_value=fake),
        patch(_REQUIRE) as m_require,
    ):
        ltr.build_request(create_approval=True, write=False)
    m_require.assert_not_called()


def test_json_output_is_valid() -> None:
    with patch(_RUN_AUDIT, return_value=_report()):
        res = ltr.build_request(create_approval=False, write=False)
    parsed = json.loads(json.dumps(res, ensure_ascii=False))
    assert parsed["status"] == "preview"
    assert parsed["note"]


def test_markdown_and_json_request_files_written(tmp_path: Path) -> None:
    fake = SimpleNamespace(approval_id="apr_w")
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_REQUEST, return_value=fake),
    ):
        ltr.build_request(create_approval=True, out_dir=tmp_path, write=True)
    assert len(list(tmp_path.glob("*_request.md"))) == 1
    jsons = list(tmp_path.glob("*_request.json"))
    assert len(jsons) == 1
    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert data["status"] == "approval_required"
    assert data["approval_id"] == "apr_w"


def test_approval_id_and_command_format() -> None:
    fake = SimpleNamespace(approval_id="apr_deadbeef99")
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_REQUEST, return_value=fake),
    ):
        res = ltr.build_request(create_approval=True, write=False)
    assert res["approval_id"].startswith("apr_")
    assert res["approve_command"] == f"uv run achilles approval-approve {res['approval_id']}"


def test_writes_only_under_out_dir(tmp_path: Path) -> None:
    with patch(_RUN_AUDIT, return_value=_report()):
        ltr.build_request(create_approval=False, out_dir=tmp_path, write=True)
    for p in list(tmp_path.rglob("*")):
        assert tmp_path in p.parents or p == tmp_path


def test_protected_path_guard_intact() -> None:
    import importlib.util

    guard_path = Path(__file__).resolve().parents[1] / "scripts" / "check_protected_paths.py"
    spec = importlib.util.spec_from_file_location("cpp", guard_path)
    assert spec and spec.loader
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    assert cpp.is_protected("data/papers/a.pdf") is True
    assert cpp.is_protected(".env") is True
    assert cpp.is_protected("app/agents/local_training_request.py") is False


def test_source_never_references_dangerous_calls() -> None:
    # request_approval() (pending OLUŞTUR) İZİNLİDİR; tüketim/eğitim çağrıları DEĞİL.
    forbidden = (
        ".start_training(",
        ".promote_to_production(",
        "launch(",
        "require_fresh_approval(",
        "subprocess.popen(",
        "os.system(",
    )
    src_lower = _MODULE_SRC.lower()
    for token in forbidden:
        assert token not in src_lower, f"istek akışı yasak çağrı içeriyor: {token}"
    assert "import kaggle" not in src_lower
    assert "from google.colab" not in src_lower
    # request_approval (pending oluştur) bu fazda İZİNLİ ve kullanılmalı.
    assert "request_approval(" in _MODULE_SRC
