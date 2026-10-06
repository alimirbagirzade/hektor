"""Kademe-2 (2026-09-28) eğitim başlatma/kurtarma bulgularının regresyon testleri.

A2 kurtarmada ağırlık kaybı · A3 start-train durumunda started_at yok · A4 Auto-LoRA onayı
kimlikle bağlanamıyor · A5 veri kayması mtime ile ölçülüyor · A6 onay ön-kontrollerden
ÖNCE tüketiliyor · A8 maskeleme sonrası sessiz kısmi epoch.

Hepsi çevrimdışı: gerçek eğitim/alt süreç/Ollama YOK (Popen ve trainer sahte).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.lora.mix_common import parse_weights, validate_weights, weights_to_arg
from app.main import app
from app.training.peft_lora_train import clamp_steps_to_dataset
from app.training.train_guard import (
    diagnose,
    find_run_approval,
    recovery_allowed,
    sha256_file,
)

runner = CliRunner()
_NOW = dt.datetime(2026, 9, 28, 12, 0, tzinfo=dt.UTC)
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _approval(consumed: dt.datetime, *, aid: str = "apr_x", action: str = "train_run") -> dict:
    return {
        "approval_id": aid,
        "action": action,
        "status": "approved",
        "consumed_at": consumed.isoformat(),
    }


def _status(started: dt.datetime | None, **extra: object) -> dict:
    base: dict = {"adapter": "hektor_lora_v10_4b", "pid": 4242}
    if started is not None:
        base["started_at"] = started.isoformat()
    base.update(extra)
    return base


# --- A4: kimlik eşleşmesinde tüm eğitim aksiyonları -------------------------------
def test_a4_kimlikle_auto_lora_onayi_kabul_edilir() -> None:
    started = _NOW - dt.timedelta(hours=1)
    row = _approval(started, aid="apr_auto", action="auto_lora_start_training")
    assert find_run_approval([row], started, approval_id="apr_auto") is not None
    # Kimliksiz zaman-penceresi yedeği dar kalır (yalnız train_run).
    assert find_run_approval([row], started) is None
    yabanci = _approval(started, aid="apr_y", action="rules_apply")
    assert find_run_approval([yabanci], started, approval_id="apr_y") is None


# --- A3: started_at'sız koşan eğitim sessizce OK değildir -------------------------
def test_a3_started_at_yoksa_sorun_bildirilir() -> None:
    d = diagnose(
        status=_status(None, approval_id="apr_x"),
        running=True,
        now=_NOW,
        log_mtime=_NOW,
        cpu_percent=250.0,
        approvals=[],
    )
    assert d.verdict == "DIKKAT"
    assert any("started_at" in p for p in d.problems)


# --- A5: veri kayması içerik hash'iyle -------------------------------------------
def test_a5_hash_uyusmazsa_dirilme_yok() -> None:
    started = _NOW - dt.timedelta(hours=2)
    st = _status(started, approval_id="apr_x", data_sha256="a" * 64)
    v = recovery_allowed(st, [_approval(started)], now=_NOW, data_sha256="b" * 64)
    assert not v.allowed and "hash" in v.reason


def test_a5_hash_ayniysa_train_jsonl_mtime_yanlis_alarm_vermez() -> None:
    """Kurtarma train.jsonl'i yeniden yazar → mtime ilerler; içerik aynıysa yetki verilir."""
    started = _NOW - dt.timedelta(hours=2)
    st = _status(started, approval_id="apr_x", data_sha256="A" * 64)
    v = recovery_allowed(
        st,
        [_approval(started)],
        now=_NOW,
        data_mtime=_NOW - dt.timedelta(minutes=1),
        data_sha256="a" * 64,
    )
    assert v.allowed, v.reason


def test_a5_hash_kayitli_ama_kaynak_okunamiyorsa_dirilme_yok() -> None:
    started = _NOW - dt.timedelta(hours=2)
    st = _status(started, approval_id="apr_x", data_sha256="a" * 64)
    assert not recovery_allowed(st, [_approval(started)], now=_NOW, data_sha256=None).allowed


def test_a5_teshis_hash_kaymasini_bildirir() -> None:
    started = _NOW - dt.timedelta(hours=1)
    d = diagnose(
        status=_status(started, approval_id="apr_x", data_sha256="a" * 64),
        running=True,
        now=_NOW,
        log_mtime=_NOW,
        cpu_percent=200.0,
        approvals=[_approval(started)],
        data_sha256="c" * 64,
    )
    assert d.verdict == "DIKKAT"
    assert any("hash" in p for p in d.problems)


def test_a5_sha256_file(tmp_path: Path) -> None:
    import hashlib

    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"a": 1}\n')
    assert sha256_file(p) == hashlib.sha256(b'{"a": 1}\n').hexdigest()
    assert sha256_file(tmp_path / "yok.jsonl") is None


def _use_tmp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()


def test_a5_ayni_icerikte_split_dosyaya_dokunmaz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_tmp_root(tmp_path, monkeypatch)
    from app.config import get_settings
    from app.training import detached_launch as dl

    s = get_settings()
    src = tmp_path / "data" / "lora_sft" / "lora_sft.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"messages": [{"role": "user", "content": f"s{i}"}], "source_id": f"p{i}"}
        for i in range(30)
    ]
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    dl.ensure_train_split(s)
    train = s.jsonl_dir / "train.jsonl"
    old = 1_600_000_000
    os.utime(train, (old, old))
    dl.ensure_train_split(s)
    assert int(train.stat().st_mtime) == old  # aynı içerik → yeniden yazılmadı


# --- A6: detached launch ön-kontrol + erken çıkış --------------------------------
class _FakeProc:
    pid = 4321

    def __init__(self, rc: int | None) -> None:
        self._rc = rc

    def poll(self) -> int | None:
        return self._rc


def _prep_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rc: int | None
) -> tuple[object, Path]:
    _use_tmp_root(tmp_path, monkeypatch)
    from app.training import detached_launch as dl

    src = tmp_path / "data" / "lora_sft" / "lora_sft.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text('{"messages": []}\n', encoding="utf-8")
    monkeypatch.setattr(dl, "preflight_launch", lambda *a, **k: {"ok": True, "n_train": 10})
    monkeypatch.setattr(dl, "_find_hektor", lambda root: ["hektor"])
    monkeypatch.setattr(dl.subprocess, "Popen", lambda *a, **k: _FakeProc(rc))
    return dl, src


def test_a6_launch_erken_cikista_durum_dosyasini_siler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dl, _src = _prep_launch(tmp_path, monkeypatch, rc=5)
    res = dl.launch(adapter_name="t_a6", early_exit_wait_s=0, approval_id="apr_t")
    assert res["ok"] is False
    assert "çıkış kodu 5" in res["message"]
    assert not (tmp_path / "storage" / "train_status.json").exists()  # ölü kayıt yok
    assert not (tmp_path / "storage" / ".training_launching").exists()  # kilit bırakıldı


def test_a2_a5_launch_durum_dosyasi_agirlik_ve_hash_tasir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dl, src = _prep_launch(tmp_path, monkeypatch, rc=None)
    from app.lora.weight_decision import WeightDecisionStore

    rec = WeightDecisionStore().record({"math": 0.3, "coding": 0.7}, "custom", "recorded")
    res = dl.launch(adapter_name="t_a2", early_exit_wait_s=0, approval_id="apr_t")
    assert res["ok"] is True, res
    st = json.loads((tmp_path / "storage" / "train_status.json").read_text(encoding="utf-8"))
    assert st["mix_decision_id"] == rec.decision_id
    assert parse_weights(st["mix_weights"]) == rec.weights
    assert st["data_sha256"] == sha256_file(src)
    assert dt.datetime.fromisoformat(st["started_at"]).tzinfo is not None


def test_a6_preflight_kosan_egitimde_hicbir_sey_yapmaz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_tmp_root(tmp_path, monkeypatch)
    from app.training import detached_launch as dl

    monkeypatch.setattr(dl, "is_running", lambda: True)

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("koşan eğitimde bölme yapılmamalı")

    monkeypatch.setattr(dl, "ensure_train_split", _boom)
    res = dl.preflight_launch("t")
    assert res["ok"] is False and "Zaten" in res["message"]


def test_a6_preflight_yuk_doktoru_no_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_tmp_root(tmp_path, monkeypatch)
    from app.lora.weight_decision import WeightDecisionStore
    from app.training import detached_launch as dl
    from app.training import train_load_doctor as td

    WeightDecisionStore().record({"math": 1.0}, "custom", "recorded")
    monkeypatch.setattr(dl, "is_running", lambda: False)
    monkeypatch.setattr(dl, "ensure_train_split", lambda s=None: (5, 1))
    monkeypatch.setattr(dl, "_pretrain_gate_blockers", lambda s: [])
    monkeypatch.setattr("app.lora.mix_cli.run_leakage_check", lambda p: {"clean": True})
    monkeypatch.setattr(
        td,
        "run_train_doctor",
        lambda **kw: td.TrainDoctorReport(verdict="NO-GO", reasons=["VRAM yok"]),
    )
    res = dl.preflight_launch("t")
    assert res["ok"] is False and "NO-GO" in res["message"]
    monkeypatch.setattr(td, "run_train_doctor", lambda **kw: td.TrainDoctorReport(verdict="GO"))
    assert dl.preflight_launch("t") == {"ok": True, "message": "Ön-kontroller geçti.", "n_train": 5}


# --- A6 (CLI): bölme/sızıntı onaydan ÖNCE -------------------------------------------
def _prep_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    _use_tmp_root(tmp_path, monkeypatch)
    monkeypatch.delenv("HEKTOR_TRAIN_SUPERVISED", raising=False)
    monkeypatch.delenv("HEKTOR_TRAIN_RECOVERY", raising=False)
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda root=None: False)
    calls = {"authorize": 0}

    def _authorize(*a: object, **k: object) -> object:
        calls["authorize"] += 1
        raise AssertionError("ön-kontrol düşmüşken onay TÜKETİLMEMELİ")

    monkeypatch.setattr("app.training.unattended_policy.authorize_training_action", _authorize)
    return calls


_TRAIN = [
    "train",
    "--run",
    "--backend",
    "peft",
    "--adapter-name",
    "t",
    "--mix-profile",
    "balanced_v1",
]


def test_a6_cli_bos_bolmede_onay_tuketilmez(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _prep_cli(tmp_path, monkeypatch)
    monkeypatch.setattr("app.training.detached_launch.ensure_train_split", lambda s=None: (0, 0))
    r = runner.invoke(app, _TRAIN)
    assert r.exit_code == 1, r.output
    assert calls["authorize"] == 0


def test_a6_cli_sizintida_onay_tuketilmez(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _prep_cli(tmp_path, monkeypatch)
    monkeypatch.setattr("app.training.detached_launch.ensure_train_split", lambda s=None: (5, 1))
    monkeypatch.setattr(
        "app.lora.mix_cli.run_leakage_check", lambda p: {"clean": False, "counts": {"golden": 1}}
    )
    r = runner.invoke(app, _TRAIN)
    assert r.exit_code == 6, r.output
    assert calls["authorize"] == 0


# --- A2: kurtarma bekleyen (başka eğitime ait) kararı tüketmez ----------------------
def _prep_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    _prep_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("HEKTOR_TRAIN_SUPERVISED", "1")
    monkeypatch.setenv("HEKTOR_TRAIN_RECOVERY", "1")
    monkeypatch.setattr("app.training.detached_launch.ensure_train_split", lambda s=None: (5, 1))
    monkeypatch.setattr("app.lora.mix_cli.run_leakage_check", lambda p: {"clean": True})
    # Bu testler ağırlık kararını sınar; veri kalite kapısı ayrı testte (F3-2).
    monkeypatch.setattr("app.training.detached_launch._pretrain_gate_blockers", lambda s: [])
    monkeypatch.setattr(
        "app.training.peft_lora_train.train",
        lambda cfg: {"ok": True, "adapter_path": str(cfg.adapter_output_path), "device": "cpu"},
    )
    # Bu testler ağırlık kararını sınar; kurtarma yetkisi (L-2) test_kademe2_real_git'te.
    from app.training.train_guard import RecoveryVerdict

    monkeypatch.setattr(
        "app.main._recovery_verdict",
        lambda *a, **k: RecoveryVerdict(True, "test", {"approval_id": "t"}),
    )
    from app.lora.weight_decision import WeightDecisionStore

    return WeightDecisionStore().record({"math": 1.0}, "custom", "recorded")


def test_a2_kurtarma_bayraksiz_reddedilir_ve_bekleyen_karar_kalir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _prep_recovery(tmp_path, monkeypatch)
    r = runner.invoke(app, ["train", "--run", "--backend", "peft", "--adapter-name", "t"])
    assert r.exit_code == 5, r.output
    from app.lora.weight_decision import WeightDecisionStore

    pending = WeightDecisionStore().pending()
    assert pending is not None and pending.decision_id == rec.decision_id  # type: ignore[attr-defined]


def test_a2_kurtarma_bayrakla_bekleyen_karari_tuketmez(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _prep_recovery(tmp_path, monkeypatch)
    weights = "math=0.3,statistics=0.2,reasoning=0.2,trading=0.2,coding=0.1"
    r = runner.invoke(
        app,
        ["train", "--run", "--backend", "peft", "--adapter-name", "t", "--mix-weights", weights],
    )
    assert r.exit_code == 0, r.output
    m = re.search(r"MIX_DECISION id=(\S+) profile=(\S+) weights=(\S+)", r.output)
    assert m is not None, r.output
    assert parse_weights(m.group(3)) == parse_weights(weights)
    from app.lora.weight_decision import WeightDecisionStore

    rows = {d.decision_id: d for d in WeightDecisionStore().records()}
    assert rows[rec.decision_id].consumed_at is None  # type: ignore[attr-defined]
    assert rows[m.group(1)].consumed_by == "train-recovery:t"


def test_a2_weights_to_arg_kayipsiz_ve_bosluksuz() -> None:
    w = validate_weights({"math": 0.123456, "coding": 1.5, "trading": 0.05})
    arg = weights_to_arg(w)
    assert " " not in arg
    assert parse_weights(arg) == w


# --- A8: maskeleme sonrası adım tavanı ----------------------------------------------
@pytest.mark.parametrize(
    ("max_steps", "rows", "n_ds", "expected"),
    [
        (1616, 1616, 1606, 1606),  # 1 epoch planı, 10 satır atıldı → kısmi 2. epoch YOK
        (3232, 1616, 1600, 3200),  # 2 epoch planı korunur
        (10, 1616, 1606, 10),  # smoke: tavanın altında, dokunulmaz
        (4040, 1616, 1606, 4040),  # bilinçli 2.5 epoch: 3 epoch tavanını aşmıyor
        (100, 100, 100, 100),  # atılan yok
        (0, 100, 90, 0),
    ],
)
def test_a8_clamp_steps_to_dataset(max_steps: int, rows: int, n_ds: int, expected: int) -> None:
    assert clamp_steps_to_dataset(max_steps, rows, n_ds, 1) == expected


# --- PowerShell sözleşmeleri (statik; betik ÇALIŞTIRILMAZ) ----------------------------
def test_start_train_durum_dosyasi_started_at_hash_ve_agirlik_yazar() -> None:
    src = (_SCRIPTS / "start-train.ps1").read_text(encoding="utf-8")
    for key in ("started_at", "data_sha256", "mix_weights", "mix_profile"):
        assert re.search(rf"^\s*{key}\s*=", src, re.MULTILINE), f"durum alanı yok: {key}"
    assert "MIX_DECISION" in src
    assert "[string]$MixWeights" in src and "--mix-weights" in src
    assert "HEKTOR_TRAIN_RECOVERY" in src


def test_start_train_erken_cikis_supervised_icin_de_ve_exit4_logout() -> None:
    src = (_SCRIPTS / "start-train.ps1").read_text(encoding="utf-8")
    assert "if (-not $Supervised -and $proc)" not in src
    assert "cikis kodu $code" in src  # harici izleyicinin ayrıştırdığı satır korunur
    m = re.search(r"\$code -eq 4\)\s*\{(.*?)\}\s*elseif", src, re.DOTALL)
    assert m is not None and "$LogOut" in m.group(1)


def test_watchdog_kurtarmada_agirligi_geri_verir() -> None:
    src = (_SCRIPTS / "training-watchdog.ps1").read_text(encoding="utf-8")
    assert "-MixWeights $mw" in src and "-MixProfile $mp" in src


def test_k2_2026_10_06_f3_2_train_run_kalite_kapisindan_gecer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kademe 2 F3-2: doğrudan `train --run` da kalite/tazelik/sohbet kapısından geçer."""
    _prep_recovery(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "app.training.detached_launch._pretrain_gate_blockers",
        lambda s: ["Kural 1: garanti dili (test)"],
    )
    weights = "math=0.3,statistics=0.2,reasoning=0.2,trading=0.2,coding=0.1"
    r = runner.invoke(
        app,
        ["train", "--run", "--backend", "peft", "--adapter-name", "t", "--mix-weights", weights],
    )
    assert r.exit_code == 1 and "Kural 1" in r.output


def test_p3_launch_without_consumed_approval_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kademe 2 P-3: onaysız başlatma (ör. gözetimsiz politika) alt süreç doğmadan, açık
    mesajla reddedilir — alt süreçte sessiz çıkış 3 yerine."""
    from app.training import detached_launch as dl

    _use_tmp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(dl, "preflight_launch", lambda *a, **k: {"ok": True, "n_train": 5})
    spawned: list = []
    monkeypatch.setattr(dl.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    res = dl.launch(adapter_name="t_p3", early_exit_wait_s=0)
    assert res["ok"] is False and "insan onayı" in res["message"]
    assert spawned == []


def test_j2_stop_does_not_kill_reused_or_finished_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kademe 2 J-2: bitmiş koşunun ya da başlangıç zamanı tutmayan pid'in ağacı öldürülmez."""
    import json as _json

    from app.training import detached_launch as dl

    killed: list[int] = []
    monkeypatch.setattr(dl, "_terminate_tree", lambda pid: (killed.append(pid), (True, "x"))[1])
    monkeypatch.setattr(dl.resource_lock, "process_create_time", lambda pid: 2000.0)
    st = tmp_path / "storage" / "train_status.json"
    st.parent.mkdir(parents=True)
    for info, expect_kill in (
        ({"pid": 4242, "finished_at": "2026-10-06T10:00:00+00:00"}, False),
        ({"pid": 4242, "pid_create_time": 1000.0}, False),  # pid yeniden kullanılmış
        ({"pid": 4242, "pid_create_time": 2000.0}, True),  # aynı süreç
        ({"pid": 4242}, True),  # eski (start-train.ps1) kayıt: zaman yok → eski davranış
    ):
        killed.clear()
        st.write_text(_json.dumps(info), "utf-8")
        res = dl.request_stop_detached_training(tmp_path)
        assert bool(killed) is expect_kill, (info, res)


def test_j1_resource_lock_treats_windows_delete_pending_as_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kademe 2 J-1 deseni (eğitim kilidi): O_EXCL'in PermissionError'ı istisna fırlatmaz."""
    from app.training import resource_lock as rl

    real_open = rl.os.open
    calls = {"n": 0}

    def flaky_open(path, flags, *a):  # ilk deneme: silinmekte olan dosya (Windows)
        calls["n"] += 1
        if calls["n"] == 1 and flags & rl.os.O_EXCL:
            raise PermissionError(13, "delete pending")
        return real_open(path, flags, *a)

    monkeypatch.setattr(rl.os, "open", flaky_open)
    info, why = rl.acquire("training", "t", root=tmp_path)
    assert info is not None, why
    assert rl.release(str(info["token"]), root=tmp_path)
