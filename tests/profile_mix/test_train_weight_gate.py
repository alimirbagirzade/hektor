"""Her LoRA eğitiminden önce karışım ağırlığı sorulur; sızıntı varsa eğitim başlamaz."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.lora.mix_common import MixConfigError, parse_weights
from app.lora.weight_decision import (
    WeightDecisionRequired,
    WeightDecisionStore,
    finalize_decision,
    resolve_training_weights,
)
from app.main import app

runner = CliRunner()


@pytest.fixture
def store(tmp_path: Path) -> WeightDecisionStore:
    return WeightDecisionStore(tmp_path / "wd.jsonl")


def test_parse_weights_is_regex_only_and_validated() -> None:
    w = parse_weights("math=0.3, statistics=0.2,reasoning:0.3")
    assert w == {"math": 0.3, "statistics": 0.2, "reasoning": 0.3, "trading": 0.0, "coding": 0.0}
    for bad in ("math=__import__('os')", "physics=0.2", "math=5", "math=0,coding=0"):
        with pytest.raises(MixConfigError):
            parse_weights(bad)


def test_non_interactive_without_decision_refuses(store: WeightDecisionStore) -> None:
    with pytest.raises(WeightDecisionRequired):
        resolve_training_weights(mix_profile=None, mix_weights=None, interactive=False, store=store)


def test_flag_decision_is_not_persisted_until_training_starts(store: WeightDecisionStore) -> None:
    dec = resolve_training_weights(
        mix_profile="balanced_v1", mix_weights=None, interactive=False, store=store
    )
    assert dec.decision_id == "" and store.records() == []  # bloklanan eğitim iz bırakmaz
    done = finalize_decision(dec, "train:x", store)
    assert done.decision_id.startswith("wd_") and done.consumed_by == "train:x"
    assert store.pending() is None


def test_recorded_decision_is_single_use(store: WeightDecisionStore) -> None:
    rec = store.record({"math": 0.5, "coding": 0.5}, "custom", "recorded")
    dec = resolve_training_weights(
        mix_profile=None, mix_weights=None, interactive=False, store=store
    )
    assert dec.decision_id == rec.decision_id
    finalize_decision(dec, "train:a", store)
    with pytest.raises(WeightDecisionRequired):  # sonraki eğitim YENİDEN sorar
        resolve_training_weights(mix_profile=None, mix_weights=None, interactive=False, store=store)


def test_stale_decision_is_ignored(store: WeightDecisionStore) -> None:
    rec = store.record({"math": 1.0}, "custom", "recorded")
    future = dt.datetime.now(dt.UTC) + dt.timedelta(hours=25)
    assert store.pending(now=future) is None and store.pending() is not None
    assert rec.decision_id


def test_new_record_supersedes_older_pending(store: WeightDecisionStore) -> None:
    old = store.record({"math": 1.0}, "custom", "recorded")
    new = store.record({"coding": 1.0}, "custom", "recorded")
    pending = store.pending()
    assert pending is not None and pending.decision_id == new.decision_id
    rows = {r.decision_id: r for r in store.records()}
    assert len(rows) == 2  # silinmez — denetlenebilir kayıt
    assert rows[old.decision_id].superseded_by == new.decision_id
    assert rows[old.decision_id].consumed_at is None
    finalize_decision(pending, "train:a", store)
    # Eski karar sonraki eğitimde SESSİZCE kullanılmaz.
    assert store.pending() is None


def test_consuming_flag_decision_supersedes_recorded_pending(store: WeightDecisionStore) -> None:
    recorded = store.record({"math": 1.0}, "custom", "recorded")
    flag = resolve_training_weights(
        mix_profile="balanced_v1", mix_weights=None, interactive=False, store=store
    )
    done = finalize_decision(flag, "train:a", store)
    assert store.pending() is None
    rows = {r.decision_id: r for r in store.records()}
    assert rows[recorded.decision_id].superseded_by == done.decision_id
    with pytest.raises(WeightDecisionRequired):
        resolve_training_weights(mix_profile=None, mix_weights=None, interactive=False, store=store)


def test_consuming_superseded_decision_is_refused(store: WeightDecisionStore) -> None:
    old = store.record({"math": 1.0}, "custom", "recorded")
    store.record({"coding": 1.0}, "custom", "recorded")
    with pytest.raises(ValueError, match="geçersiz kılınmış"):
        store.consume(old.decision_id, "train:x")


def test_legacy_record_format_stays_pending(store: WeightDecisionStore) -> None:
    """Gerçek registry'deki eski biçimli (superseded_* alanı yok) satır okunur ve bekler."""
    legacy = {
        "decision_id": "wd_a03273ef8c",
        "weights": {
            "math": 0.2,
            "statistics": 0.3,
            "reasoning": 0.2,
            "trading": 0.25,
            "coding": 0.05,
        },
        "profile_name": "trading_analysis_v1",
        "source": "recorded",
        "created_at": "2026-09-27T21:49:25.103502+00:00",
        "consumed_at": None,
        "consumed_by": None,
    }
    store.path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    now = dt.datetime(2026, 9, 28, 12, 0, tzinfo=dt.UTC)
    pending = store.pending(now=now)
    assert pending is not None and pending.decision_id == "wd_a03273ef8c"
    assert pending.profile_name == "trading_analysis_v1"


def test_legacy_multiple_open_rows_only_newest_is_pending(store: WeightDecisionStore) -> None:
    base = {"weights": {"math": 1.0}, "profile_name": "custom", "source": "recorded"}
    rows = [
        {**base, "decision_id": "wd_old", "created_at": "2026-09-28T10:00:00+00:00"},
        {**base, "decision_id": "wd_new", "created_at": "2026-09-28T11:00:00+00:00"},
    ]
    store.path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    now = dt.datetime(2026, 9, 28, 12, 0, tzinfo=dt.UTC)
    assert store.pending(now=now).decision_id == "wd_new"  # type: ignore[union-attr]
    store.consume("wd_new", "train:a")
    assert store.pending(now=now) is None  # eski açık satır da geçersiz kılındı


def test_concurrent_records_do_not_lose_updates(store: WeightDecisionStore) -> None:
    import threading

    ids: list[str] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            ids.append(store.record({"math": 1.0}, "custom", "recorded").decision_id)
        except BaseException as exc:  # pragma: no cover - hata raporu için
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    recs = store.records()
    assert sorted(r.decision_id for r in recs) == sorted(ids)  # hiçbir kayıt kaybolmadı
    assert sum(1 for r in recs if r.is_open) == 1
    assert not list(store.path.parent.glob("*.tmp")) and not list(store.path.parent.glob("*.lock"))


def test_file_lock_times_out_when_held(tmp_path: Path) -> None:
    from app.lora.mix_common import file_lock

    target = tmp_path / "wd.jsonl"
    with file_lock(target), pytest.raises(TimeoutError), file_lock(target, timeout=0.2):
        pass
    with file_lock(target, timeout=0.2):  # bırakıldıktan sonra yeniden alınabilir
        pass


def test_interactive_prompt_shows_profiles_and_retries(store: WeightDecisionStore) -> None:
    answers = iter(["bilinmeyen_profil", "statistics_v1"])
    shown: list[str] = []
    dec = resolve_training_weights(
        mix_profile=None,
        mix_weights=None,
        interactive=True,
        store=store,
        prompt=lambda _m: next(answers),
        echo=shown.append,
    )
    assert dec.profile_name == "statistics_v1" and dec.weights["statistics"] == 0.35
    assert any("balanced_v1" in s for s in shown)
    assert any("semantik yüzde" in s for s in shown)
    assert any("Geçersiz" in s for s in shown)


def _prep_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> None:
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    monkeypatch.delenv("HEKTOR_TRAIN_SUPERVISED", raising=False)
    monkeypatch.setenv("COLUMNS", "300")
    from app.config import get_settings

    get_settings.cache_clear()
    jsonl = get_settings().jsonl_dir
    jsonl.mkdir(parents=True, exist_ok=True)
    (jsonl / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda root=None: False)
    monkeypatch.setattr(
        "app.training.detached_launch.ensure_train_split", lambda settings: (len(rows), 0)
    )

    class _Doc:
        verdict, reasons = "GO", []

    monkeypatch.setattr("app.training.train_load_doctor.run_train_doctor", lambda: _Doc())

    class _Dec:
        authorized, approval_id = True, "apr_test"

    monkeypatch.setattr(
        "app.training.unattended_policy.authorize_training_action", lambda *a, **k: _Dec()
    )

    def _no_training(*a: object, **k: object) -> None:
        # Kapı regresyonu olursa test gerçek eğitime girip ASILMASIN; hızlı başarısız olsun.
        raise AssertionError("bu testte eğitim başlamamalıydı (kapı geçildi)")

    monkeypatch.setattr("app.lora.weight_decision.finalize_decision", _no_training)


def test_cli_train_run_refuses_without_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prep_train(tmp_path, monkeypatch, [{"messages": [{"role": "user", "content": "RSI nedir?"}]}])
    r = runner.invoke(app, ["train", "--run", "--backend", "peft", "--adapter-name", "t"])
    assert r.exit_code == 5, r.output
    assert "mix weights" in r.output


def test_cli_train_run_blocks_on_golden_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.evals.profile.dataset_loader import load_split
    from app.evals.profile.schema import Split

    leaked = load_split(Split.GOLDEN_TEST, purpose="leakage_check")[0].question
    _prep_train(tmp_path, monkeypatch, [{"messages": [{"role": "user", "content": leaked}]}])
    r = runner.invoke(
        app,
        [
            "train",
            "--run",
            "--backend",
            "peft",
            "--adapter-name",
            "t",
            "--mix-profile",
            "balanced_v1",
        ],
    )
    assert r.exit_code == 6, r.output
    assert "sızıntı" in r.output.lower()
    # Kapı eğitimi durdurdu → ağırlık kararı TÜKETİLMEDİ ve kayıt kalmadı.
    assert WeightDecisionStore().records() == []


def test_cli_train_run_blocks_on_question_packed_with_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gerçek eğitim biçimi (BAĞLAM + SORU) içindeki golden sorusu da exit 6 ile durdurur."""
    from app.evals.profile.dataset_loader import load_split
    from app.evals.profile.schema import Split

    leaked = load_split(Split.GOLDEN_TEST, purpose="leakage_check")[0].question
    packed = "BAĞLAM:\nUzun bir makale pasajı burada yer alır.\n\nSORU: " + leaked
    _prep_train(tmp_path, monkeypatch, [{"messages": [{"role": "user", "content": packed}]}])
    r = runner.invoke(
        app,
        [
            "train",
            "--run",
            "--backend",
            "peft",
            "--adapter-name",
            "t",
            "--mix-profile",
            "balanced_v1",
        ],
    )
    assert r.exit_code == 6, r.output


def test_leakage_check_also_scans_valid_jsonl(tmp_path: Path) -> None:
    from app.evals.profile.dataset_loader import load_split
    from app.evals.profile.schema import Split
    from app.lora.mix_cli import run_leakage_check

    leaked = load_split(Split.VALIDATION, purpose="leakage_check")[0].question
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "RSI nedir?"}]}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "valid.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": leaked}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rep = run_leakage_check(tmp_path / "train.jsonl")
    assert rep["clean"] is False and rep["n_valid"] == 1
    assert [h["split"] for h in rep["hits"]] == ["valid"]


def test_cli_mix_weights_records_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    r = runner.invoke(app, ["mix", "weights", "--profile", "trading_analysis_v1"])
    assert r.exit_code == 0, r.output
    pending = WeightDecisionStore().pending()
    assert pending is not None and pending.profile_name == "trading_analysis_v1"
    r2 = runner.invoke(app, ["mix", "weights", "--show"])
    assert pending.decision_id in r2.output


def test_detached_launch_refuses_without_recorded_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Web butonu / auto_pipeline yolu: kayıtlı karar yoksa alt süreç HİÇ açılmaz."""
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    from app.config import get_settings
    from app.training import detached_launch

    get_settings.cache_clear()
    monkeypatch.setattr(detached_launch, "is_running", lambda: False)

    def _no_spawn(*a: object, **k: object) -> None:
        raise AssertionError("karar yokken alt süreç başlatılmamalı")

    monkeypatch.setattr(detached_launch, "ensure_train_split", _no_spawn)
    res = detached_launch.launch(adapter_name="x")
    assert res["ok"] is False and "mix weights" in res["message"]
