"""Phase 5E — eğitim-sonrası postcheck testleri (offline; eğitim/terfi/tüketim YOK).

Postcheck SALT-OKUMA: handoff/dry-run + training artefaktı + adapter metadata +
adapter-eval + understanding-score'u read-only okur. Bu testler doğrular: terfi YOK
(her zaman human_review_required), training başlamaz, model yüklenmez/eval çalışmaz,
onay tüketilmez. Mock; ağ/cloud yok.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from app.agents import local_training_postcheck as ltp

_TR = "app.agents.local_training_postcheck._probe_training"
_AD = "app.agents.local_training_postcheck._probe_adapter"
_EV = "app.agents.local_training_postcheck._probe_adapter_eval"
_US = "app.agents.local_training_postcheck._probe_understanding"
_PROMOTE = "app.lora.auto_pipeline.AutoLoRAPipeline.promote_to_production"
_START = "app.lora.auto_pipeline.AutoLoRAPipeline.start_training"
_LAUNCH = "app.training.detached_launch.launch"
_REQUIRE = "app.agents.runtime.approvals.require_fresh_approval"
_REQUEST = "app.agents.runtime.approvals.request_approval"

_MODULE_SRC = Path(ltp.__file__).read_text(encoding="utf-8")

_NOT_FOUND = {"found": False, "source": "none"}


def _write(path: Path, data: dict) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_default_no_result_is_no_training_run_found(tmp_path: Path) -> None:
    with patch(_TR, return_value=_NOT_FOUND), patch(_AD, return_value=_NOT_FOUND):
        res = ltp.build_postcheck(out_dir=tmp_path, write=False)
    assert res["status"] == "no_training_run_found"
    assert res["training_artifacts_found"] is False


def test_handoff_report_read(tmp_path: Path) -> None:
    hf = _write(tmp_path / "20260101_000000_handoff.json", {"status": "ready_for_human_execution"})
    with patch(_TR, return_value=_NOT_FOUND), patch(_AD, return_value=_NOT_FOUND):
        res = ltp.build_postcheck(handoff_json=hf, out_dir=tmp_path, write=False)
    assert res["handoff_status"] == "ready_for_human_execution"


def test_dryrun_report_read(tmp_path: Path) -> None:
    dr = _write(tmp_path / "20260101_000000_dryrun.json", {"status": "dry_run_passed"})
    with patch(_TR, return_value=_NOT_FOUND), patch(_AD, return_value=_NOT_FOUND):
        res = ltp.build_postcheck(dryrun_json=dr, out_dir=tmp_path, write=False)
    assert res["dryrun_source"] == dr


def test_training_report_read_makes_ready(tmp_path: Path) -> None:
    tr = _write(tmp_path / "train.json", {"adapter": "achilles_lora", "iterations": 300})
    with patch(_AD, return_value=_NOT_FOUND):
        res = ltp.build_postcheck(training_report=tr, out_dir=tmp_path, write=False)
    assert res["status"] == "postcheck_ready_for_human_review"
    assert res["training_artifacts_found"] is True
    assert res["training"]["data"]["adapter"] == "achilles_lora"


def test_adapter_path_metadata_only_no_model_load(tmp_path: Path) -> None:
    adir = tmp_path / "adapter_x"
    adir.mkdir()
    (adir / "adapter_model.safetensors").write_bytes(b"not-a-real-model")
    with (
        patch(_START) as m_start,
        patch(_PROMOTE) as m_promote,
        patch(_LAUNCH) as m_launch,
    ):
        res = ltp.build_postcheck(adapter_path=str(adir), out_dir=tmp_path, write=False)
    assert res["adapter"]["found"] is True
    assert res["adapter"]["n_files"] == 1
    assert res["status"] == "postcheck_ready_for_human_review"
    for m in (m_start, m_promote, m_launch):
        m.assert_not_called()  # model YÜKLENMEDİ / eğitim başlamadı


def test_adapter_eval_found_in_report(tmp_path: Path) -> None:
    with (
        patch(_TR, return_value=_NOT_FOUND),
        patch(_AD, return_value=_NOT_FOUND),
        patch(_EV, return_value={"found": True, "count": 1, "reports": ["adapter_eval_x.json"]}),
    ):
        res = ltp.build_postcheck(out_dir=tmp_path, write=False)
    assert res["adapter_eval_found"] is True


def test_understanding_score_found_in_report(tmp_path: Path) -> None:
    with (
        patch(_TR, return_value=_NOT_FOUND),
        patch(_AD, return_value=_NOT_FOUND),
        patch(_US, return_value={"found": True, "count": 2, "records": ["a.json", "b.json"]}),
    ):
        res = ltp.build_postcheck(out_dir=tmp_path, write=False)
    assert res["understanding_score_found"] is True


def test_promotion_recommendation_always_human_review(tmp_path: Path) -> None:
    # no_training durumunda
    with patch(_TR, return_value=_NOT_FOUND), patch(_AD, return_value=_NOT_FOUND):
        r1 = ltp.build_postcheck(out_dir=tmp_path, write=False)
    # found durumunda
    tr = _write(tmp_path / "train.json", {"adapter": "x"})
    with patch(_AD, return_value=_NOT_FOUND):
        r2 = ltp.build_postcheck(training_report=tr, out_dir=tmp_path, write=False)
    assert r1["promotion_recommendation"] == "human_review_required"
    assert r2["promotion_recommendation"] == "human_review_required"


def test_no_dangerous_calls(tmp_path: Path) -> None:
    tr = _write(tmp_path / "train.json", {"adapter": "x"})
    with (
        patch(_AD, return_value=_NOT_FOUND),
        patch(_PROMOTE) as m_promote,
        patch(_START) as m_start,
        patch(_LAUNCH) as m_launch,
        patch(_REQUIRE) as m_require,
        patch(_REQUEST) as m_request,
        patch("subprocess.Popen") as m_popen,
        patch("os.system") as m_system,
    ):
        ltp.build_postcheck(training_report=tr, out_dir=tmp_path, write=False)
    for m in (m_promote, m_start, m_launch, m_require, m_request, m_popen, m_system):
        m.assert_not_called()  # 9-16: terfi/eğitim/tüketim/alt-süreç YOK


def test_json_output_valid(tmp_path: Path) -> None:
    with patch(_TR, return_value=_NOT_FOUND), patch(_AD, return_value=_NOT_FOUND):
        res = ltp.build_postcheck(out_dir=tmp_path, write=False)
    parsed = json.loads(json.dumps(res, ensure_ascii=False))
    assert parsed["status"] == "no_training_run_found"
    assert parsed["note"]


def test_markdown_and_json_written(tmp_path: Path) -> None:
    with patch(_TR, return_value=_NOT_FOUND), patch(_AD, return_value=_NOT_FOUND):
        ltp.build_postcheck(out_dir=tmp_path, write=True)
    assert len(list(tmp_path.glob("*_postcheck.md"))) == 1
    jsons = list(tmp_path.glob("*_postcheck.json"))
    assert len(jsons) == 1
    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert data["promotion_recommendation"] == "human_review_required"


def test_writes_only_under_out_dir(tmp_path: Path) -> None:
    with patch(_TR, return_value=_NOT_FOUND), patch(_AD, return_value=_NOT_FOUND):
        ltp.build_postcheck(out_dir=tmp_path, write=True)
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
    assert cpp.is_protected("app/agents/local_training_postcheck.py") is False


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
        assert token not in src_lower, f"postcheck yasak çağrı içeriyor: {token}"
    assert "import kaggle" not in src_lower
    assert "from google.colab" not in src_lower
    # Terfi metni her durumda human_review_required.
    assert "human_review_required" in _MODULE_SRC
