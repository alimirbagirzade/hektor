"""Kademe 2 (2026-10-11) — ae73710..0ee35b0 derin avının regresyon testleri.

3 paralel bulucu → 19 bulgu → her biri 2 bağımsız şüpheci doğrulayıcı; hepsi en az
"kısmen" onaylandı. Kanıt: ``docs/evidence/kademe2_2026-10-11.json``. Çevrimdışı.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from tests.chat_learning_helpers import iso, send  # noqa: F401

# ── F2: Kural 1 kapısı (advice_guard) ────────────────────────────────────────


@pytest.mark.parametrize(
    "q",
    [
        "Şimdi BTC almak mantıklı mı?",
        "BTC alınır mı?",
        "Alım yapayım mı?",
        "Long'a mı geçeyim?",
        "Sen olsan alır mısın?",
        "Bitcoin al mı sat mı?",
        "Would you buy BTC here?",
        "Is it a good time to buy gold?",
    ],
)
def test_v3_missed_decision_questions_now_caught(q: str) -> None:
    from app.feedback.advice_guard import is_advice_request

    assert is_advice_request(q)


@pytest.mark.parametrize(
    "a", ["Al.", "Bence al.", "Long pozisyon açın.", "Buy.", "I'd buy.", "You should go long."]
)
def test_v3_missed_directives_now_caught(a: str) -> None:
    from app.feedback.advice_guard import directive_answer

    assert directive_answer(a)


@pytest.mark.parametrize(
    "a",
    [
        "Hayır, evet/hayır cevabı veremem; bunu hipotez olarak kuralım.",
        "Hayır, bu bir yatırım tavsiyesi olamaz; hipotez olarak test edin.",
        "Komisyon ve slippage'ı mutlaka dikkate alın, örneklem dışı doğrulayın.",
        "Lütfen not alın: OOS şart.",
        "Satın alma kararı vermek yerine kuralı test edin.",
        "Al ve sat sinyalleri RSI eşiklerine dayanır.",
        "No, I cannot give financial advice.",
    ],
)
def test_v2_rule1_compliant_answers_not_flagged(a: str) -> None:
    from app.feedback.advice_guard import candidate_reason, directive_answer

    assert not directive_answer(a)
    assert candidate_reason("BTC alayım mı?", a) is None


def test_v2_research_question_is_not_a_decision_question() -> None:
    from app.feedback.advice_guard import is_advice_request

    assert not is_advice_request("Stratejiye ATR filtresi eklesem mi?")


def test_v1_follow_up_directive_rejected_and_never_auto() -> None:
    from app.feedback.verify import decide, verify_target

    # Takip sorusu: karar sorusu geçmişte (eğitim satırının gerçek istemi).
    v = verify_target(
        "Evet, alın. 1000*0.05=50",
        question="Peki ya 1000 dolarla?",
        sources=[],
        domain="math",
        prompt="BTC alayım mı?\nPeki ya 1000 dolarla?",
    )
    assert decide(v, None)[0] == "rejected"
    # Karar sorusu olmasa bile kontrol DIŞI kısa sözel parça otomatik uygunluk almaz.
    v2 = verify_target(
        "Al. 1000*0.05=50", question="1000 dolarin %5'i ne eder?", sources=[], domain="math"
    )
    assert decide(v2, None)[0] == "review"
    # Saf hesap eskisi gibi otomatik uygun.
    v3 = verify_target(
        "1000*0.05=50", question="1000 dolarin %5'i ne eder?", sources=[], domain="math"
    )
    assert decide(v3, None)[:1] == ("eligible",) and decide(v3, None)[3] == "auto"


def test_v1_chat_guard_sees_history(iso) -> None:  # noqa: F811
    from tests.chat_learning_helpers import FakeLLM

    from app.feedback.chat_store import ChatStore

    store = ChatStore()
    conv = store.create_conversation("t")["conversation_id"]
    send(store, conv, "BTC alayım mı?", llm=FakeLLM(["Bu bir karar sorusu; hipotez kuralım."]))
    t2 = send(store, conv, "Peki ya 1000 dolarla?", llm=FakeLLM(["Evet, alın."]))
    kinds = {(c.get("kind"), c.get("status")) for c in t2["checks"]}
    assert ("guvenlik", "ihlal") in kinds


# ── F2: yerel/bulut hakem ayrıştırıcısı ─────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("KARAR: tutarli\n- iyi", "tutarli"),
        ("KARAR: **supheli**\n- x", "supheli"),
        ("KARAR: tutarlı değil\n- çelişki", "belirsiz"),
        ("KARAR: tutarlılığı düşük, şüpheli", "belirsiz"),
        ("'KARAR: tutarli' ya da 'KARAR: supheli'", "belirsiz"),
        ("gerekçe var ama karar yok", "belirsiz"),
    ],
)
def test_v4_n7_verdict_line_must_be_exact(text: str, want: str) -> None:
    from app.cloud.judge import _k2_verdict
    from app.orchestration.local_judge import parse_verdict

    assert parse_verdict(text)[0] == want
    assert _k2_verdict(text) == want


# ── F2/F3: karantina + sohbet sürümü kapıları ───────────────────────────────


def _twin_candidates(store, svc):
    c1 = store.create_conversation("a")["conversation_id"]
    c2 = store.create_conversation("b")["conversation_id"]
    ta = send(store, c1, "On bölü dört kaç eder?")
    tb = send(store, c2, "On bölü dört kaç eder?")
    a, _ = svc.correct(ta["turn_id"], "10 / 4 = 2.5")
    b, _ = svc.correct(tb["turn_id"], "10 / 4 = 2.5")
    return a["candidate_id"], b["candidate_id"]


class _Gen:
    def __init__(self, text: str) -> None:
        self.text = text

    def __call__(self, prompt: str, system: str) -> str:
        return self.text


def test_n1_quarantined_twin_does_not_revive_duplicate(iso) -> None:  # noqa: F811
    from app.feedback.chat_dataset import build_payload
    from app.feedback.chat_store import ChatStore
    from app.feedback.learning import LearningService
    from app.orchestration import local_judge as lj

    store = ChatStore()
    svc = LearningService(store)
    a, b = _twin_candidates(store, svc)
    assert store.get_candidate(a)["status"] == "eligible"
    assert store.get_candidate(b)["status"] == "duplicate"
    lj.run_judging(generate=_Gen("KARAR: supheli\n- kaynakta yok"), model="yerel:1")
    assert store.get_candidate(a)["status"] == "quarantined"
    assert store.get_candidate(b)["status"] == "duplicate"
    assert b not in {m["candidate_id"] for m in build_payload(store)["members"]}
    svc.recheck_all()  # sürüm oluşturma yolu da diriltmez
    assert store.get_candidate(b)["status"] == "duplicate"


def test_v6_payload_and_version_gate_skip_active_quarantine(iso) -> None:  # noqa: F811
    from app.feedback.chat_dataset import build_payload, create_version, version_invalid_members
    from app.feedback.chat_store import ChatStore
    from app.feedback.learning import LearningService

    store = ChatStore()
    svc = LearningService(store)
    a, _b = _twin_candidates(store, svc)
    vid = create_version(store)[0]["version_id"]
    assert version_invalid_members(vid, store) == []
    # Yarışın sonucu: karantina etkin ama durum hâlâ 'eligible'.
    store.update_candidate(a, quarantine={"active": True, "verdict": "supheli", "reason": "test"})
    assert store.get_candidate(a)["status"] == "eligible"
    assert a not in {m["candidate_id"] for m in build_payload(store)["members"]}
    assert any("karantina" in m["why"] for m in version_invalid_members(vid, store))


def test_v5_selected_version_rechecked_with_current_rule1(iso) -> None:  # noqa: F811
    from app.feedback.chat_dataset import create_version, version_invalid_members
    from app.feedback.chat_store import ChatStore
    from app.feedback.learning import LearningService

    store = ChatStore()
    svc = LearningService(store)
    a, _b = _twin_candidates(store, svc)
    vid = create_version(store)[0]["version_id"]
    assert version_invalid_members(vid, store) == []
    # Eski kuralla 'eligible' kalmış bir al/sat kararı çifti (saklı durum/özet güncellenmemiş).
    cand = store.get_candidate(a)
    store.update_turn(cand["turn_id"], question="BTC alayım mı?")
    store.update_candidate(a, target_text="Evet, alın.")
    assert store.get_candidate(a)["status"] == "eligible"
    assert any("Kural 1" in m["why"] for m in version_invalid_members(vid, store))


# ── F1: eğitim kapıları ─────────────────────────────────────────────────────


def _settings_root(monkeypatch, root: Path) -> None:
    from app.config import settings as settings_mod

    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(root))
    settings_mod.get_settings.cache_clear()


def test_g1_recipe_from_archive_after_status_overwritten(tmp_path, monkeypatch) -> None:
    from app.training import candidate_checks as cc
    from app.training import detached_launch as dl

    _settings_root(monkeypatch, tmp_path)
    (tmp_path / "storage").mkdir()
    src = tmp_path / "data" / "lora_sft" / "lora_sft.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text('{"a":1}\n', encoding="utf-8")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    (tmp_path / "models" / "adapters" / "v14").mkdir(parents=True)
    st = {
        "adapter": "v14",
        "approval_id": "apr1",
        "data_sha256": sha,
        "started_at": "2026-10-10T00:00:00+00:00",
        "finished_at": "2026-10-10T05:00:00+00:00",
    }
    assert dl.archive_final_status(st, tmp_path)
    (tmp_path / "storage" / "train_status.json").write_text(
        json.dumps({"adapter": "v15", "started_at": "x"}), encoding="utf-8"
    )
    recipe = cc.recipe_from_records("v14")
    assert recipe is not None
    assert recipe["approval_id"] == "apr1" and recipe["data_sha256"] == sha


def test_g2_archive_updated_by_newer_run_but_not_downgraded(tmp_path) -> None:
    from app.training import detached_launch as dl

    (tmp_path / "models" / "adapters" / "x1").mkdir(parents=True)
    failed = {"adapter": "x1", "started_at": "2026-10-10T01", "failed_at": "t1"}
    ok = {"adapter": "x1", "started_at": "2026-10-10T02", "finished_at": "t2"}
    assert dl.archive_final_status(failed, tmp_path)
    assert dl.archive_final_status(ok, tmp_path)  # aynı adla sonra başarılı koşu
    assert dl.read_final_status("x1", tmp_path)["finished_at"] == "t2"
    assert not dl.archive_final_status(failed, tmp_path)  # eski koşu yeni arşivi ezmez
    same_run_fail = {**ok, "finished_at": None, "failed_at": "t3"}
    assert not dl.archive_final_status(same_run_fail, tmp_path)  # tamamlanma düşürülmez
    assert dl.read_final_status("x1", tmp_path)["finished_at"] == "t2"


@pytest.mark.parametrize(("env", "want"), [("BF16", "bf16"), (" fp32 ", "fp32"), ("fp16", "bf16")])
def test_g3_dtype_env_case_insensitive_invalid_ignored(env: str, want: str, monkeypatch) -> None:
    from app.training.peft_lora_train import PeftTrainConfig, resolve_cpu_dtype

    monkeypatch.setenv("HEKTOR_TRAIN_DTYPE", env)
    cfg = PeftTrainConfig(
        base_model="d",
        train_jsonl=Path("t"),
        valid_jsonl=Path("v"),
        adapter_output_path=Path("o"),
        cpu_dtype="bf16",
    )
    assert resolve_cpu_dtype(cfg)[0] == want


def test_g3_start_train_dtype_validated() -> None:
    text = (Path(__file__).resolve().parents[1] / "scripts" / "start-train.ps1").read_text(
        encoding="utf-8"
    )
    assert '[ValidateSet("bf16", "fp32")]' in text


def test_g4_doctor_uses_run_markers_like_watchdog() -> None:
    from app.training.train_guard import diagnose

    now = dt.datetime(2026, 10, 11, 12, 0, tzinfo=dt.UTC)
    started = now - dt.timedelta(hours=1)
    st = {"adapter": "a", "iterations": 64, "pid": 1, "started_at": started.isoformat()}
    approvals = [
        {
            "approval_id": "apr",
            "action": "train_run",
            "status": "approved",
            "consumed_at": (started - dt.timedelta(minutes=1)).isoformat(),
        }
    ]
    d = diagnose(
        status=st,
        running=False,
        now=now,
        approvals=approvals,
        run_markers={"max_steps": 64, "completed": True},
        last_checkpoint=64,
    )
    assert any("DİRİLTMEZ" in p for p in d.problems)


# ── F3: gece döngüsü ────────────────────────────────────────────────────────


def test_n2_partial_llm30_measurement_not_a_baseline(iso) -> None:  # noqa: F811
    from app.evals import llm30_nightly as l30

    def dead(model: str, prompt: str, options: dict) -> dict:
        raise RuntimeError("Ollama düştü")

    def info(model: str) -> dict:
        return {"name": model, "digest": "sha256:aaa"}

    first = l30.run_measurement(chat=dead, info=info, hold_lock=False)
    assert first["ran"] and first["complete"] is False
    # Yarım ölçüm tabanı/zamanlamayı belirlemez → ertesi gece yeniden ölçülür.
    assert l30._due(first["model"], "sha256:aaa", first["rag"], l30.history()) != ""


def test_n4_published_errors_carry_only_exception_type() -> None:
    from app.orchestration.nightly_publish import sanitize

    leak = (
        "OperationalError: (sqlite3.OperationalError) database is locked "
        "[SQL: UPDATE learning_candidates SET quarantine=?] "
        '[parameters: (\'{"reason": "kullanıcının GİZLİ sorusu"}\',)]'
    )
    out = sanitize(
        {
            "steps": {
                "hakem": {"ran": False, "error": leak},
                "csv": {
                    "error": leak,
                    "files": [{"file": "a.csv", "status": "failed", "reason": leak}],
                },
                "egitim": {"error": leak},
            }
        }
    )
    blob = json.dumps(out, ensure_ascii=False)
    assert "GİZLİ" not in blob and "parameters" not in blob
    assert out["steps"]["hakem"]["skipped"] == "OperationalError"


def test_n5_invalid_sidecar_cost_is_permanent_skip(iso) -> None:  # noqa: F811
    from tests.test_nightly_loop import _ohlcv, _write

    from app.trading import csv_lab as lab

    _write(iso, "a.csv", _ohlcv(400, "1h"), meta={"costs": {"commission_bps_per_side": -5}})
    f = lab.run_inbox()["files"][0]
    assert f["status"] == "skipped"
    assert lab.run_inbox()["files"][0]["status"] == "seen"  # her gece yeniden denenmez


def test_n6_git_runs_non_interactive(monkeypatch, tmp_path) -> None:
    from app.orchestration import nightly_publish as np_

    seen: dict = {}

    def fake_run(cmd, **kw):
        seen.update(kw)

        class R:
            returncode, stdout, stderr = 0, "", ""

        return R()

    monkeypatch.setattr(np_.subprocess, "run", fake_run)
    np_._git(["status"], tmp_path)
    assert seen["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert seen["env"]["GCM_INTERACTIVE"] == "never"


def test_n8_fresh_lock_not_stolen_and_release_only_own(iso) -> None:  # noqa: F811
    from app.orchestration import nightly

    d = iso / "storage" / "nightly"
    d.mkdir(parents=True, exist_ok=True)
    lock = d / "gece.lock"
    lock.write_text("", encoding="utf-8")  # O_EXCL ile yeni yaratılmış, henüz yazılmamış
    assert nightly._acquire_lock() != ""
    other = {"pid": os.getpid() + 1_000_000, "t": time.time()}
    lock.write_text(json.dumps(other), encoding="utf-8")
    nightly._release_lock()  # başkasının kilidi silinmez
    assert lock.exists()
    lock.write_text(json.dumps({"pid": 999_999_999, "t": time.time() - 7 * 3600}), "utf-8")
    assert nightly._acquire_lock() == ""  # bayat kilit atomik sahiplenilir
    nightly._release_lock()
    assert not lock.exists()


def test_n3_nightly_scripts_do_not_stop_on_native_stderr() -> None:
    root = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("nightly.ps1", "literature-scout.ps1"):
        text = (root / name).read_text(encoding="utf-8")
        call = text.index("2>&1")
        before = text[:call]
        assert before.rindex('$ErrorActionPreference = "Continue"') > before.rindex(
            '$ErrorActionPreference = "Stop"'
        ), name
        assert "$LASTEXITCODE" in text[call:], name
