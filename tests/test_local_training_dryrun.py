"""Phase 5C — onaylı eğitim isteği DRY-RUN pipeline testleri (offline; eğitim/tüketim YOK).

Dry-run pipeline audit + 5B isteği okur, onayı READ-ONLY kontrol eder (TÜKETMEZ),
pretrain-gate read-only + adapter-eval MOCKED çalıştırır ve bir execution PLANI üretir.
Bu testler doğrular: training başlamaz, onay tüketilmez, request_approval bile
çağrılmaz; STOP_ALL/NOT_READY/NO-GO durumları doğru raporlanır. Mock; ağ/cloud yok.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agents import local_training_dryrun as ldr
from app.agents import local_training_orchestrator as lto

_RUN_AUDIT = "app.agents.local_training_orchestrator.run_audit"
_GET_APPROVAL = "app.agents.runtime.approvals.get_approval"
_REQUIRE = "app.agents.runtime.approvals.require_fresh_approval"
_REQUEST = "app.agents.runtime.approvals.request_approval"
_LAUNCH = "app.training.detached_launch.launch"
_START = "app.lora.auto_pipeline.AutoLoRAPipeline.start_training"
_PROMOTE = "app.lora.auto_pipeline.AutoLoRAPipeline.promote_to_production"

_MODULE_SRC = Path(ldr.__file__).read_text(encoding="utf-8")


def _report(verdict="READY", score=84, gate="GO", stop_all=False, risks=None):
    return lto.TrainingAuditReport(
        generated_at="2026-06-19T00:00:00+00:00",
        banner=lto.REPORT_ONLY_BANNER,
        probes={
            "stop_all": {"available": True, "stop_all_active": stop_all},
            "pretrain_gate": {"available": True, "verdict": gate},
            "adapter_eval_readiness": {"available": True, "adapters_present": 0},
        },
        risks=risks or [],
        readiness_score=score,
        readiness_verdict=verdict,
        notes=[],
    )


def _approved(consumed=False):
    return SimpleNamespace(status="approved", consumed_at=("t" if consumed else None))


def test_default_starts_no_training(tmp_path: Path) -> None:
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_LAUNCH) as m_launch,
        patch(_START) as m_start,
        patch(_PROMOTE) as m_promote,
        patch("subprocess.Popen") as m_popen,
    ):
        ldr.build_dryrun(out_dir=tmp_path, write=False)
    for m in (m_launch, m_start, m_promote, m_popen):
        m.assert_not_called()


def test_no_approval_returns_needs_approval(tmp_path: Path) -> None:
    with patch(_RUN_AUDIT, return_value=_report()):
        res = ldr.build_dryrun(approval_id=None, out_dir=tmp_path, write=False)
    assert res["status"] in {"needs_approval", "preview_only"}


def test_approved_id_produces_dry_run_plan(tmp_path: Path) -> None:
    with (
        patch(_RUN_AUDIT, return_value=_report(gate="GO")),
        patch(_GET_APPROVAL, return_value=_approved()),
    ):
        res = ldr.build_dryrun(approval_id="apr_x", out_dir=tmp_path, write=False)
    assert res["status"] == "dry_run_passed"
    assert res["approval_status"] == "approved_not_consumed"
    assert res["execution_plan"] and "validate dataset" in res["execution_plan"]


def test_approved_id_does_not_consume(tmp_path: Path) -> None:
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_GET_APPROVAL, return_value=_approved()),
        patch(_REQUIRE) as m_require,
    ):
        ldr.build_dryrun(approval_id="apr_x", out_dir=tmp_path, write=False)
    m_require.assert_not_called()  # onay TÜKETİLMEDİ


def test_stop_all_blocks(tmp_path: Path) -> None:
    with patch(_RUN_AUDIT, return_value=_report(stop_all=True, verdict="BLOCKED")):
        res = ldr.build_dryrun(approval_id="apr_x", out_dir=tmp_path, write=False)
    assert res["status"] == "blocked"
    assert res["reason"] == "STOP_ALL is active"


def test_not_ready_blocks(tmp_path: Path) -> None:
    with patch(_RUN_AUDIT, return_value=_report(verdict="NOT_READY", score=40, gate="GO")):
        res = ldr.build_dryrun(approval_id="apr_x", out_dir=tmp_path, write=False)
    assert res["status"] in {"not_ready", "blocked"}


def test_pretrain_gate_go_in_report(tmp_path: Path) -> None:
    with (
        patch(_RUN_AUDIT, return_value=_report(gate="GO")),
        patch(_GET_APPROVAL, return_value=_approved()),
    ):
        res = ldr.build_dryrun(approval_id="apr_x", out_dir=tmp_path, write=False)
    assert res["pretrain_gate"] == "GO"


def test_pretrain_gate_nogo_is_not_ready(tmp_path: Path) -> None:
    with patch(_RUN_AUDIT, return_value=_report(verdict="READY", gate="NO-GO")):
        res = ldr.build_dryrun(approval_id="apr_x", out_dir=tmp_path, write=False)
    assert res["status"] == "not_ready"
    assert res["pretrain_gate"] == "NO-GO"


def test_adapter_eval_mocked_ready(tmp_path: Path) -> None:
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_GET_APPROVAL, return_value=_approved()),
    ):
        res = ldr.build_dryrun(
            approval_id="apr_x", mock_adapter_eval=True, out_dir=tmp_path, write=False
        )
    assert res["adapter_eval"] == "mocked_ready"


def test_no_dangerous_calls(tmp_path: Path) -> None:
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_GET_APPROVAL, return_value=_approved()),
        patch(_LAUNCH) as m_launch,
        patch(_START) as m_start,
        patch(_PROMOTE) as m_promote,
        patch(_REQUIRE) as m_require,
        patch(_REQUEST) as m_request,
        patch("subprocess.Popen") as m_popen,
    ):
        ldr.build_dryrun(approval_id="apr_x", out_dir=tmp_path, write=False)
    for m in (m_launch, m_start, m_promote, m_require, m_request, m_popen):
        m.assert_not_called()  # 10-15: hiçbir tehlikeli/oluşturucu çağrı yok


def test_json_output_valid(tmp_path: Path) -> None:
    with patch(_RUN_AUDIT, return_value=_report()):
        res = ldr.build_dryrun(out_dir=tmp_path, write=False)
    parsed = json.loads(json.dumps(res, ensure_ascii=False))
    assert parsed["note"] == ldr.DRYRUN_BANNER
    assert "status" in parsed


def test_markdown_and_json_written(tmp_path: Path) -> None:
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_GET_APPROVAL, return_value=_approved()),
    ):
        ldr.build_dryrun(approval_id="apr_x", out_dir=tmp_path, write=True)
    assert len(list(tmp_path.glob("*_dryrun.md"))) == 1
    jsons = list(tmp_path.glob("*_dryrun.json"))
    assert len(jsons) == 1
    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert data["status"] == "dry_run_passed"


def test_approval_id_and_status_format(tmp_path: Path) -> None:
    with (
        patch(_RUN_AUDIT, return_value=_report()),
        patch(_GET_APPROVAL, return_value=_approved()),
    ):
        res = ldr.build_dryrun(approval_id="apr_deadbeef", out_dir=tmp_path, write=False)
    assert res["approval_id"] == "apr_deadbeef"
    assert res["approval_status"] == "approved_not_consumed"


def test_writes_only_under_out_dir(tmp_path: Path) -> None:
    with patch(_RUN_AUDIT, return_value=_report()):
        ldr.build_dryrun(out_dir=tmp_path, write=True)
    for p in list(tmp_path.rglob("*")):
        assert tmp_path in p.parents or p == tmp_path


def test_protected_path_guard_intact() -> None:
    import importlib.util

    guard_path = Path(__file__).resolve().parents[1] / "scripts" / "check_protected_paths.py"
    spec = importlib.util.spec_from_file_location("cpp", guard_path)
    assert spec and spec.loader
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    assert cpp.is_protected("models/adapters/a.bin") is True
    assert cpp.is_protected(".env") is True
    assert cpp.is_protected("app/agents/local_training_dryrun.py") is False


def test_source_never_references_dangerous_calls() -> None:
    forbidden = (
        ".start_training(",
        ".promote_to_production(",
        "launch(",
        "require_fresh_approval(",
        "request_approval(",  # 5C OKUR; onay OLUŞTURMAZ
        "subprocess.popen(",
        "os.system(",
    )
    src_lower = _MODULE_SRC.lower()
    for token in forbidden:
        assert token not in src_lower, f"dry-run yasak çağrı içeriyor: {token}"
    assert "import kaggle" not in src_lower
    assert "from google.colab" not in src_lower
    # Onay yalnız READ-ONLY okunur.
    assert "get_approval(" in _MODULE_SRC
