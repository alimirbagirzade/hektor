"""Phase 5D — insan-kapılı eğitim HANDOFF testleri (offline; eğitim/tüketim YOK).

Handoff 5C dry-run raporunu + onayı READ-ONLY okur ve uygunsa gerçek eğitim komutunu
YALNIZ METİN olarak verir. Bu testler doğrular: komut çalıştırılmaz, onay tüketilmez,
STOP_ALL/dry-run-fail/approval durumları doğru raporlanır. Mock; ağ/cloud yok.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agents import local_training_handoff as lth

_SUP = "app.agents.runtime.supervisor.is_stop_all_active"
_GET_APPROVAL = "app.agents.runtime.approvals.get_approval"
_REQUIRE = "app.agents.runtime.approvals.require_fresh_approval"
_REQUEST = "app.agents.runtime.approvals.request_approval"
_LAUNCH = "app.training.detached_launch.launch"
_START = "app.lora.auto_pipeline.AutoLoRAPipeline.start_training"
_PROMOTE = "app.lora.auto_pipeline.AutoLoRAPipeline.promote_to_production"

_MODULE_SRC = Path(lth.__file__).read_text(encoding="utf-8")


def _write_dryrun(
    tmp_path: Path, status: str = "dry_run_passed", approval_id: str = "apr_x"
) -> str:
    p = tmp_path / "20260101_000000_dryrun.json"
    p.write_text(json.dumps({"status": status, "approval_id": approval_id}), encoding="utf-8")
    return str(p)


def _appr(status: str = "approved", consumed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(status=status, consumed_at=("t" if consumed else None))


def test_default_no_dryrun_needs_dry_run(tmp_path: Path) -> None:
    res = lth.build_handoff(out_dir=tmp_path, write=False)
    assert res["status"] == "needs_dry_run"


def test_dryrun_not_passed_blocked(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path, status="not_ready")
    with patch(_SUP, return_value=False):
        res = lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=False)
    assert res["status"] in {"blocked", "not_ready"}


def test_no_approval_needs_approval(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path, approval_id="")
    with patch(_SUP, return_value=False), patch(_GET_APPROVAL, return_value=None):
        res = lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=False)
    assert res["status"] == "needs_approval"


def test_pending_approval_needs_approval(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path)
    with patch(_SUP, return_value=False), patch(_GET_APPROVAL, return_value=_appr("pending")):
        res = lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=False)
    assert res["status"] == "needs_approval"


def test_consumed_approval_blocked(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path)
    with patch(_SUP, return_value=False), patch(_GET_APPROVAL, return_value=_appr(consumed=True)):
        res = lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=False)
    assert res["status"] == "blocked"


def test_ready_when_passed_and_approved(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path)
    with patch(_SUP, return_value=False), patch(_GET_APPROVAL, return_value=_appr()):
        res = lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=False)
    assert res["status"] == "ready_for_human_execution"
    assert res["approval_status"] == "approved_not_consumed"
    assert res["checklist"]


def test_recommended_command_is_string(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path)
    with patch(_SUP, return_value=False), patch(_GET_APPROVAL, return_value=_appr()):
        res = lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=False)
    assert res["recommended_command"] == "uv run achilles train --run"


def test_stop_all_blocked(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path)
    with patch(_SUP, return_value=True), patch(_GET_APPROVAL, return_value=_appr()):
        res = lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=False)
    assert res["status"] == "blocked"
    assert res["reason"] == "STOP_ALL is active"


def test_command_never_executed_and_no_dangerous_calls(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path)
    with (
        patch(_SUP, return_value=False),
        patch(_GET_APPROVAL, return_value=_appr()),
        patch(_LAUNCH) as m_launch,
        patch(_START) as m_start,
        patch(_PROMOTE) as m_promote,
        patch(_REQUIRE) as m_require,
        patch(_REQUEST) as m_request,
        patch("subprocess.Popen") as m_popen,
        patch("os.system") as m_system,
    ):
        res = lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=False)
    assert res["status"] == "ready_for_human_execution"
    for m in (m_launch, m_start, m_promote, m_require, m_request, m_popen, m_system):
        m.assert_not_called()  # 8,10-16: komut çalıştırılmadı, onay tüketilmedi


def test_json_output_valid(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path)
    with patch(_SUP, return_value=False), patch(_GET_APPROVAL, return_value=_appr()):
        res = lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=False)
    parsed = json.loads(json.dumps(res, ensure_ascii=False))
    assert parsed["status"] == "ready_for_human_execution"
    assert parsed["note"]


def test_markdown_and_json_written(tmp_path: Path) -> None:
    dr = _write_dryrun(tmp_path)
    with patch(_SUP, return_value=False), patch(_GET_APPROVAL, return_value=_appr()):
        lth.build_handoff(dryrun_json=dr, out_dir=tmp_path, write=True)
    assert len(list(tmp_path.glob("*_handoff.md"))) == 1
    jsons = list(tmp_path.glob("*_handoff.json"))
    assert len(jsons) == 1
    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert data["status"] == "ready_for_human_execution"


def test_protected_path_guard_intact() -> None:
    import importlib.util

    guard_path = Path(__file__).resolve().parents[1] / "scripts" / "check_protected_paths.py"
    spec = importlib.util.spec_from_file_location("cpp", guard_path)
    assert spec and spec.loader
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    assert cpp.is_protected("models/adapters/a.bin") is True
    assert cpp.is_protected(".env") is True
    assert cpp.is_protected("app/agents/local_training_handoff.py") is False


def test_source_never_references_dangerous_calls() -> None:
    forbidden = (
        ".start_training(",
        ".promote_to_production(",
        "launch(",
        "require_fresh_approval(",
        "request_approval(",
        "subprocess.popen(",
        "os.system(",
    )
    src_lower = _MODULE_SRC.lower()
    for token in forbidden:
        assert token not in src_lower, f"handoff yasak çağrı içeriyor: {token}"
    assert "import kaggle" not in src_lower
    assert "from google.colab" not in src_lower
    # Onay yalnız READ-ONLY okunur.
    assert "get_approval(" in _MODULE_SRC
    # Gerçek eğitim komutu YALNIZ string olarak var.
    assert "uv run achilles train --run" in _MODULE_SRC
