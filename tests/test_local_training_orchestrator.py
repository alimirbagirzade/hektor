"""Phase 5A — lokal eğitim-denetim orkestratörü testleri (offline; GERÇEK eğitim YOK).

Orkestratör SALT RAPOR'dur: durum okur, rapor yazar. Bu testler doğrular ki audit
hiçbir tehlikeli aksiyon çağırmaz (launch / train --run subprocess / start_training /
promote_to_production), onay tüketmez, STOP_ALL'ı risk olarak raporlar, bekleyen
onayları tüketmeden listeler, geçerli JSON + markdown üretir. Ağ/cloud/training yok.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agents import local_training_orchestrator as lto

_SUP = "app.agents.runtime.supervisor.is_stop_all_active"
_LIST = "app.agents.runtime.approvals.list_approvals"
_FRESH = "app.agents.runtime.approvals.has_fresh_approval"
_REQUIRE = "app.agents.runtime.approvals.require_fresh_approval"
_LAUNCH = "app.training.detached_launch.launch"
_START = "app.lora.auto_pipeline.AutoLoRAPipeline.start_training"
_PROMOTE = "app.lora.auto_pipeline.AutoLoRAPipeline.promote_to_production"

_MODULE_SRC = Path(lto.__file__).read_text(encoding="utf-8")


def test_audit_runs_and_is_report_only(tmp_path: Path) -> None:
    report = lto.run_audit(out_dir=tmp_path, write=False)
    assert report.banner == lto.REPORT_ONLY_BANNER
    assert "No training was started" in report.banner
    assert report.readiness_verdict in {"READY", "NOT_READY", "BLOCKED"}
    md = lto.render_markdown(report)
    assert lto.REPORT_ONLY_BANNER in md


def test_audit_never_calls_launch(tmp_path: Path) -> None:
    with patch(_LAUNCH) as m_launch:
        lto.run_audit(out_dir=tmp_path, write=False)
    m_launch.assert_not_called()


def test_audit_never_spawns_train_subprocess(tmp_path: Path) -> None:
    with patch("subprocess.Popen") as m_popen:
        lto.run_audit(out_dir=tmp_path, write=False)
    m_popen.assert_not_called()


def test_audit_never_calls_start_training(tmp_path: Path) -> None:
    with patch(_START) as m_start:
        lto.run_audit(out_dir=tmp_path, write=False)
    m_start.assert_not_called()


def test_audit_never_calls_promote(tmp_path: Path) -> None:
    with patch(_PROMOTE) as m_promote:
        lto.run_audit(out_dir=tmp_path, write=False)
    m_promote.assert_not_called()


def test_stop_all_active_becomes_risk(tmp_path: Path) -> None:
    with patch(_SUP, return_value=True):
        report = lto.run_audit(out_dir=tmp_path, write=False)
    assert report.probes["stop_all"]["stop_all_active"] is True
    assert any("STOP_ALL" in r for r in report.risks)
    assert report.readiness_verdict == "BLOCKED"


def test_pending_approvals_listed_not_consumed(tmp_path: Path) -> None:
    fake = [
        SimpleNamespace(
            approval_id="apr_test", agent_id="lora-trainer", action="train_run", risk="critical"
        )
    ]
    with (
        patch(_LIST, return_value=fake),
        patch(_FRESH, return_value=False),
        patch(_REQUIRE) as m_require,
    ):
        report = lto.run_audit(out_dir=tmp_path, write=False)
    appr = report.probes["approvals"]
    assert appr["pending_count"] == 1
    assert appr["pending"][0]["approval_id"] == "apr_test"
    # Onay TÜKETİLMEDİ (require_fresh_approval hiç çağrılmadı).
    m_require.assert_not_called()


def test_json_output_is_valid(tmp_path: Path) -> None:
    report = lto.run_audit(out_dir=tmp_path, write=False)
    blob = json.dumps(report.to_dict(), ensure_ascii=False)
    parsed = json.loads(blob)
    assert parsed["banner"] == lto.REPORT_ONLY_BANNER
    assert "probes" in parsed and "readiness_score" in parsed


def test_markdown_and_json_files_written(tmp_path: Path) -> None:
    lto.run_audit(out_dir=tmp_path, write=True)
    mds = list(tmp_path.glob("*_report.md"))
    jsons = list(tmp_path.glob("*_report.json"))
    assert len(mds) == 1 and len(jsons) == 1
    # JSON dosyası geçerli olmalı.
    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert data["banner"] == lto.REPORT_ONLY_BANNER


def test_writes_only_under_out_dir_not_protected_paths(tmp_path: Path) -> None:
    # Rapor yalnız verilen out_dir altına yazılır; korumalı yollara dokunmaz.
    lto.run_audit(out_dir=tmp_path, write=True)
    written = list(tmp_path.rglob("*"))
    assert written, "rapor dosyaları out_dir altında olmalı"
    for p in written:
        assert tmp_path in p.parents or p == tmp_path


def test_protected_path_guard_classifies_orchestrator_as_allowed() -> None:
    # Orkestratör kaynak dosyası korumalı DEĞİL; korumalı yollar korunur (guard bütünlüğü).
    import importlib.util

    guard_path = Path(__file__).resolve().parents[1] / "scripts" / "check_protected_paths.py"
    spec = importlib.util.spec_from_file_location("cpp", guard_path)
    assert spec and spec.loader
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    assert cpp.is_protected("data/papers/a.pdf") is True
    assert cpp.is_protected("models/adapters/a.bin") is True
    assert cpp.is_protected(".env") is True
    assert cpp.is_protected("app/agents/local_training_orchestrator.py") is False


def test_source_never_references_dangerous_calls() -> None:
    # Statik güvence: modül kaynağı tehlikeli ÇAĞRI desenlerini içermez. (Docstring'de
    # 'train --run' GEÇEBİLİR — orası ne YAPMADIĞINI anlatır; bu yüzden yalnız gerçek
    # çağrı kalıplarını '(' ile ararız.) no-network/cloud da doğrulanır.
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
        assert token not in src_lower, f"orkestratör yasak çağrı deseni içeriyor: {token}"
    # Cloud helper IMPORT'u yok (docstring'de isim geçebilir; gerçek import aranır).
    assert "import kaggle" not in src_lower
    assert "from google.colab" not in src_lower
