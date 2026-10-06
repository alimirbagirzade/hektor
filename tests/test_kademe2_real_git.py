"""Kademe 2 kapısı — GERÇEK geçici Git deposunda, kapı AÇIK entegrasyon testleri.

Diğer testler kapıyı nötrler ya da kod durumunu sahte sağlar; burada ``code_state_provider``
gerçek ``git ls-tree`` / ``git status`` ile çalışır. Kabul şartları:
- Geçerli kayıtla ön kontrol geçer.
- Kod / veri / reçete değişirse onay TÜKETİLMEDEN (hatta istenmeden) durur.
- Web, Auto-LoRA ön kontrolü, CLI ``train --run``, web alt süreci, nöbetçi kurtarması ve
  ``start-train.ps1 -SkipGate`` kapıyı aşamaz.
- Değişmeyen koşu checkpoint'ten devam edebilir.
- Kaydın yazılması/commit'lenmesi denetlenen kod özetini değiştirmez (sonsuz döngü yok).
- "Reçete özeti yeterli" yolunda reçete gerçek veri/temel model/profil/karışımı kapsar; kontrol
  ile kullanım arasında dosya değişirse başlatma durur, eğitici okuduğu baytları doğrular.
- Yarışlar: eşzamanlı iki başlatma tek süreç; ölmüş alt sürecin kilidi bayat sayılır.
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.training import easy_train as et

runner = CliRunner()
_REAL_POPEN = subprocess.Popen


def _only_training_popen(fake):
    """``dl.subprocess`` global ``subprocess`` modülüdür: git çağrıları gerçek kalsın."""

    def popen(cmd, *a, **kw):
        if "env" not in kw:
            return _REAL_POPEN(cmd, *a, **kw)
        return fake(cmd, **kw)

    return popen


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _row(i: int) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"Soru {i}: risk yönetimi nedir?"},
                {"role": "assistant", "content": f"Cevap {i}: hipotez ve test noktası."},
            ],
            "metadata": {"source_id": f"p{i % 7}"},
        },
        ensure_ascii=False,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch, real_kademe2_gate):
    from app.config import settings as settings_mod
    from app.lora.mix_common import load_mix_config, profile_weights
    from app.lora.weight_decision import WeightDecisionStore
    from app.training import detached_launch

    root = tmp_path / "repo"
    for rel, text in {
        "app/x.py": "X = 1\n",
        "scripts/s.ps1": "Write-Host ok\n",
        "configs/c.yaml": "a: 1\n",
        "pyproject.toml": "[project]\nname='t'\n",
        ".gitignore": "data/\nstorage/\nmodels/\n*.db\n",
    }.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(root))
    monkeypatch.setenv("HEKTOR_SQLITE_PATH", str(tmp_path / "t.db"))
    for k in ("HEKTOR_TRAIN_SUPERVISED", "HEKTOR_TRAIN_RECOVERY", et.RECIPE_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HEKTOR_TRAIN_RESUME", "0")
    monkeypatch.setenv("COLUMNS", "300")
    settings_mod.get_settings.cache_clear()
    src = root / "data" / "lora_sft" / "lora_sft.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text("\n".join(_row(i) for i in range(60)) + "\n", encoding="utf-8")
    # GERÇEK git kod durumu (sahte değil).
    monkeypatch.setattr(et, "code_state_provider", et._default_code_state)
    monkeypatch.setattr(detached_launch, "_pretrain_gate_blockers", lambda s: [])
    monkeypatch.setattr("app.lora.mix_cli.leakage_eval_items", lambda: [])
    monkeypatch.setattr("app.lora.mix_cli.run_leakage_check", lambda p: {"clean": True})
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda root=None: False)
    mix = sorted((load_mix_config().get("profiles") or {}).keys())[0]
    WeightDecisionStore().record(profile_weights(mix), mix, source="test")
    yield {"root": root, "src": src, "mix": mix}
    settings_mod.get_settings.cache_clear()


def _record(**scope: str) -> dict:
    return et.record_kademe2(
        scope=scope,
        findings=[{"id": "F1", "status": "duzeltildi", "evidence": "test eklendi"}],
        closure_evidence="derin av bulguları kapandı; kapı yeşil (commit abc123)",
        reviewer="insan",
    )


def _settings(**kw):
    from app.lora.mix_common import load_mix_config

    mix = sorted((load_mix_config().get("profiles") or {}).keys())[0]
    return {"adapter_name": "hektor_lora_k2g", "mix_profile": mix, **kw}


def _data_sha() -> str:
    return et._lora_sft_lines()[1]


# ── temel kapı ──────────────────────────────────────────────────────────────


def test_valid_record_passes_and_code_state_is_real_git(repo) -> None:
    code = et._default_code_state()
    assert code["ok"] and len(code["code_sha"]) == 64 and code["head"]
    assert et.kademe2_blocker() is not None  # kayıt yok
    _record(data_sha256=_data_sha())
    assert et.kademe2_blocker() is None
    keys = {i["key"]: i["ok"] for i in et.readiness()["items"]}
    assert keys["kademe2"] and keys["kod_durumu"]


def test_recording_and_committing_record_does_not_change_audited_code(repo) -> None:
    before = et._default_code_state()["code_sha"]
    rec = _record(data_sha256=_data_sha())
    assert (repo["root"] / "reports" / "kademe2" / f"{rec['record_id']}.json").is_file()
    assert et.kademe2_blocker() is None  # izlenmeyen rapor dosyası ağacı "kirletmez"
    _git(repo["root"], "add", "-A")
    _git(repo["root"], "commit", "-qm", "kademe2 kaydı")
    assert et._default_code_state()["code_sha"] == before  # kayıt commit'i kod özetini değiştirmez
    assert et.kademe2_blocker() is None  # → yeniden denetim döngüsü yok


@pytest.mark.parametrize("commit", [False, True])
def test_code_change_blocks_before_any_approval(repo, commit) -> None:
    from app.agents.runtime import approvals

    snap = et.prepare_snapshot(_settings())
    _record(recipe_sha=snap["recipe_sha"])
    (repo["root"] / "app" / "x.py").write_text("X = 2\n", encoding="utf-8")
    if commit:
        _git(repo["root"], "commit", "-qam", "kod değişti")
    r = et.launch(snap["snapshot_id"], "req-code-0001")
    assert r["status"] == "blocked", r
    want = "Kademe 2 kaydı yok" if commit else "kod durumu doğrulanamadı"
    assert any(want in p for p in r["problems"]), r["problems"]
    assert approvals.list_approvals() == []


def test_data_change_blocks_even_with_recipe_record(repo) -> None:
    from app.agents.runtime import approvals

    snap = et.prepare_snapshot(_settings())
    _record(recipe_sha=snap["recipe_sha"])
    assert et.kademe2_blocker(recipe_sha=snap["recipe_sha"]) is None
    repo["src"].write_text(repo["src"].read_text("utf-8") + _row(999) + "\n", encoding="utf-8")
    msg = et.kademe2_blocker(recipe_sha=snap["recipe_sha"])
    assert msg and "reçetenin veri özeti" in msg  # reçete kaydı başka veriyle kullanılamaz
    r = et.launch(snap["snapshot_id"], "req-data-0001")
    assert r["status"] == "blocked" and approvals.list_approvals() == []


def test_recipe_record_covers_only_its_recipe(repo) -> None:
    a = et.prepare_snapshot(_settings())
    _record(recipe_sha=a["recipe_sha"])
    for change in ({"adapter_name": "hektor_lora_other"}, {"profile": "small_smoke_test"}):
        b = et.prepare_snapshot(_settings(**change))
        assert b["recipe_sha"] != a["recipe_sha"]
        assert et.kademe2_blocker(recipe_sha=b["recipe_sha"]) is not None
    # Reçete; veri, temel model, profil, karışım ve adapter adını kapsar.
    r = et._snapshot(a["snapshot_id"])
    for key in ("data_sha256", "base_model", "profile", "mix_weights", "adapter_name"):
        assert r[key] not in (None, "", {})
    tampered = {**r, "mix_weights": {"math": 1.0}}
    assert et._recipe_sha(tampered) != r["recipe_sha"]
    # Reçete kapsamlı kayıt, reçetesiz düz CLI eğitimine izin VERMEZ (yalnız veri kapsamı verir).
    assert et.kademe2_blocker() is not None


# ── yollar: CLI / web alt süreci / nöbetçi / web / Auto-LoRA / SkipGate ──────


def _cli(*extra: str):
    from app.main import app

    return runner.invoke(
        app,
        [
            "train",
            "--run",
            "--backend",
            "peft",
            "--adapter-name",
            "hektor_lora_k2g",
            "--skip-load-check",
            *extra,
        ],
    )


@pytest.mark.parametrize(
    "mode",
    ["manual", "web_child", "sentinel_recovery"],
)
def test_cli_paths_exit_10_without_record(repo, monkeypatch, mode) -> None:
    from app.agents.runtime import approvals
    from app.training import resource_lock

    extra: list[str] = ["--mix-profile", repo["mix"]]
    if mode != "manual":
        monkeypatch.setenv("HEKTOR_TRAIN_SUPERVISED", "1")
    if mode == "sentinel_recovery":
        monkeypatch.setenv("HEKTOR_TRAIN_RECOVERY", "1")
        monkeypatch.setenv("HEKTOR_TRAIN_RESUME", "1")
    res = _cli(*extra)
    assert res.exit_code == 10, res.output
    assert approvals.list_approvals() == [] and not resource_lock.status()["held"]


def _parent_lock(monkeypatch) -> str:
    """`launch()` gibi: kilidi üst süreç alır, belirteci alt sürece devreder."""
    from app.agents.runtime import approvals
    from app.training import resource_lock
    from app.training.detached_launch import APPROVAL_ENV

    info, why = resource_lock.acquire("training", "web:test", state="launching")
    assert info is not None, why
    monkeypatch.setenv(resource_lock.TOKEN_ENV, str(info["token"]))
    # Üst süreç onayı TÜKETİR ve kimliğini devreder (L-2: belirteç tek başına yetki değil).
    req = approvals.request_approval("lora-trainer", "train_run", "web", "high")
    approvals.approve(req.approval_id)
    assert approvals.require_fresh_approval("lora-trainer", "train_run", "high", "web").authorized
    monkeypatch.setenv(APPROVAL_ENV, req.approval_id)
    return str(info["token"])


@pytest.mark.parametrize("case", ["no_approval_env", "unconsumed", "stale"])
def test_l2_forged_lock_token_does_not_bypass_approval(repo, monkeypatch, case) -> None:
    """Kademe 2 L-2: elle yazılmış kilit + belirteç, tüketilmiş TAZE onay olmadan eğitmez."""
    import datetime as dt

    from app.agents.runtime import approvals
    from app.training.detached_launch import APPROVAL_ENV

    seen = _fake_trainer(monkeypatch)
    _record(data_sha256=_data_sha())
    monkeypatch.setenv("HEKTOR_TRAIN_SUPERVISED", "1")
    _parent_lock(monkeypatch)
    if case == "no_approval_env":
        monkeypatch.delenv(APPROVAL_ENV)
    elif case == "unconsumed":
        req = approvals.request_approval("lora-trainer", "train_run", "x", "high")
        approvals.approve(req.approval_id)
        monkeypatch.setenv(APPROVAL_ENV, req.approval_id)
    else:  # tüketilmiş ama pencere dışında (eski onayın yeniden kullanımı)
        from app.training import train_guard

        real = train_guard.launch_approval_problem
        later = dt.datetime.now(dt.UTC) + dt.timedelta(hours=2)
        monkeypatch.setattr(
            train_guard, "launch_approval_problem", lambda row, **kw: real(row, now=later)
        )
    res = _cli("--mix-weights", _W)
    assert res.exit_code == 3, res.output
    assert "onayı doğrulanamadı" in res.output and seen == []


def _fake_trainer(monkeypatch) -> list:
    seen: list = []

    def fake(cfg):
        seen.append(cfg)
        return {"ok": True, "adapter_path": str(cfg.adapter_output_path), "device": "cpu"}

    monkeypatch.setattr("app.training.peft_lora_train.train", fake)
    monkeypatch.setattr("app.main._register_manual_adapter", lambda **kw: None)
    return seen


def _crashed_run(
    root: Path, adapter: str = "hektor_lora_k2g", *, consume: bool = True, attempts: int = 1
) -> str:
    """Nöbetçinin diriltebileceği gerçek bir çöküş: tüketilmiş onay + durum dosyası."""
    import datetime as dt

    from app.agents.runtime import approvals
    from app.training.train_guard import SOURCE_DATA_REL, sha256_file

    req = approvals.request_approval("lora-trainer", "train_run", "çöken koşu", "high")
    approvals.approve(req.approval_id)
    if consume:
        assert approvals.require_fresh_approval("lora-trainer", "train_run", "high", "t").authorized
    st = root / "storage" / "train_status.json"
    st.parent.mkdir(parents=True, exist_ok=True)
    st.write_text(
        json.dumps(
            {
                "adapter": adapter,
                "iterations": 100,
                "approval_id": req.approval_id,
                "started_at": dt.datetime.now(dt.UTC).isoformat(),
                "recovery_attempts": attempts,
                "data_sha256": sha256_file(root / SOURCE_DATA_REL),
            }
        ),
        encoding="utf-8",
    )
    return req.approval_id


def _recovery_env(monkeypatch) -> None:
    monkeypatch.setenv("HEKTOR_TRAIN_SUPERVISED", "1")
    monkeypatch.setenv("HEKTOR_TRAIN_RECOVERY", "1")
    monkeypatch.setenv("HEKTOR_TRAIN_RESUME", "1")


_W = "math=0.2,statistics=0.2,reasoning=0.2,trading=0.2,coding=0.2"


@pytest.mark.parametrize(
    "case",
    ["no_status", "other_adapter", "unconsumed", "attempts_exhausted", "stopped"],
)
def test_l2_recovery_env_alone_does_not_bypass_approval(repo, monkeypatch, case) -> None:
    """Kademe 2 L-2: `SUPERVISED=1 RECOVERY=1` elle verilse de onaylı çöküş yoksa eğitim YOK."""
    seen = _fake_trainer(monkeypatch)
    _record(data_sha256=_data_sha())
    if case == "other_adapter":
        _crashed_run(repo["root"], adapter="baska_kosu")
    elif case == "unconsumed":
        _crashed_run(repo["root"], consume=False)
    elif case == "attempts_exhausted":
        _crashed_run(repo["root"], attempts=4)  # nöbetçi sayacı (3) + bu denemenin artışı aşıldı
    elif case == "stopped":
        _crashed_run(repo["root"])
        st = repo["root"] / "storage" / "train_status.json"
        data = json.loads(st.read_text("utf-8"))
        data["stop_requested_at"] = "2026-10-06T10:00:00+00:00"
        st.write_text(json.dumps(data), encoding="utf-8")
    _recovery_env(monkeypatch)
    res = _cli("--mix-weights", _W)
    assert res.exit_code == 3, res.output
    assert "Kurtarma yetkisiz" in res.output
    assert seen == []


def test_l2_recovery_of_approved_crash_allowed(repo, monkeypatch) -> None:
    seen = _fake_trainer(monkeypatch)
    _record(data_sha256=_data_sha())
    aid = _crashed_run(repo["root"], attempts=3)  # start-train.ps1'in artırdığı son deneme
    _recovery_env(monkeypatch)
    res = _cli("--mix-weights", _W)
    assert res.exit_code == 0, res.output
    assert aid in res.output and len(seen) == 1


def test_resume_of_unchanged_run_allowed_but_not_after_data_change(repo, monkeypatch) -> None:
    seen = _fake_trainer(monkeypatch)
    _record(data_sha256=_data_sha())
    _crashed_run(repo["root"])
    ad = repo["root"] / "models" / "adapters" / "hektor_lora_k2g" / "checkpoint-12"
    ad.mkdir(parents=True)
    monkeypatch.setenv("HEKTOR_TRAIN_SUPERVISED", "1")
    monkeypatch.setenv("HEKTOR_TRAIN_RECOVERY", "1")
    monkeypatch.setenv("HEKTOR_TRAIN_RESUME", "1")
    w = _W
    res = _cli("--mix-weights", w)
    assert res.exit_code == 0, res.output
    # Reçetesiz yol da (D-1) kapılardan hemen sonraki bölme dosyasına özetle bağlanır.
    assert len(seen) == 1 and len(seen[0].expect_train_sha256) == 64
    repo["src"].write_text(repo["src"].read_text("utf-8") + _row(1234) + "\n", encoding="utf-8")
    again = _cli("--mix-weights", w)
    assert again.exit_code == 10, again.output  # veri değişti → kurtarma da durur


def test_easy_flow_child_bound_to_recipe(repo, monkeypatch) -> None:
    from app.lora.weight_decision import WeightDecisionStore

    seen = _fake_trainer(monkeypatch)
    snap = et.prepare_snapshot(_settings())
    _record(recipe_sha=snap["recipe_sha"])
    WeightDecisionStore().record(snap["mix_weights"], snap["mix_label"], source="t")
    monkeypatch.setenv("HEKTOR_TRAIN_SUPERVISED", "1")
    monkeypatch.setenv(et.RECIPE_ENV, snap["recipe_sha"])
    _parent_lock(monkeypatch)  # web alt süreci: üst sürecin kilit belirteci (F1-4)
    res = _cli("--profile", snap["profile"], "--base-model", snap["base_model"])
    assert res.exit_code == 0, res.output
    cfg = seen[-1]
    assert cfg.expect_train_sha256 == snap["train_sha256"]
    assert cfg.expect_valid_sha256 == snap["valid_sha256"]
    # Reçeteden farklı profil → onay/kilit öncesi çıkış 11.
    WeightDecisionStore().record(snap["mix_weights"], snap["mix_label"], source="t")
    bad = _cli("--profile", "small_smoke_test", "--base-model", snap["base_model"])
    assert bad.exit_code == 11, bad.output
    # Reçeteden farklı karışım ağırlığı → çıkış 11.
    other = dict.fromkeys(snap["mix_weights"], 0.2)
    WeightDecisionStore().record(other, "custom", source="t")
    bad_w = _cli("--profile", snap["profile"], "--base-model", snap["base_model"])
    assert bad_w.exit_code == 11 and "karışım" in bad_w.output, bad_w.output


def test_web_endpoint_and_auto_lora_preflight_blocked(repo) -> None:
    from fastapi.testclient import TestClient

    from app.agents.runtime import approvals
    from app.training.detached_launch import preflight_launch
    from app.web.server import app

    body = (
        TestClient(app).post("/api/training/run", json={"adapter_name": "hektor_lora_k2g"}).json()
    )
    assert body["ok"] is False and "Kademe 2" in body["message"]
    pre = preflight_launch("hektor_lora_k2g", run_load_doctor=False)  # Auto-LoRA yolu
    assert not pre["ok"] and "Kademe 2" in pre["message"]
    assert approvals.list_approvals() == []


def test_skipgate_and_env_cannot_disable_kademe2() -> None:
    root = Path(__file__).resolve().parents[1]
    ps = (root / "scripts" / "start-train.ps1").read_text(encoding="utf-8")
    # -SkipGate yalnız pretrain-gate kalite kapısını atlar; eğitim yine `train --run` ile
    # başlar (çıkış 10 orada) ve betik kapıyı kapatan bir ortam değişkeni KURMAZ.
    import re

    assert "--run" in ps and "train" in ps
    assigned = set(re.findall(r"\$env:(HEKTOR_\w+)\s*=", ps))
    assert assigned, "betik ortam değişkeni atamaları okunamadı"
    assert not [v for v in assigned if "KADEME" in v or "SKIP" in v or "GATE" in v], assigned
    # Uygulama kodunda kapıyı devre dışı bırakan bir bayrak/ortam değişkeni yok: tek atama
    # easy_train'de, çağrıların hepsi aynı fonksiyona gider.
    hits = []
    for p in (root / "app").rglob("*.py"):
        t = p.read_text(encoding="utf-8")
        if "kademe2_check =" in t or "kademe2_check:" in t:
            hits.append(p.name)
        assert "HEKTOR_SKIP_KADEME" not in t and "KADEME2_DISABLE" not in t
    assert hits == ["easy_train.py"]
    main = (root / "app" / "main.py").read_text(encoding="utf-8")
    i = main.index("_k2 = _easy_train.kademe2_check(recipe_sha=_recipe_sha)")
    window = main[i - 2500 : i]
    assert "if run:" in main[:i] and "SkipGate" not in window


# ── kontrol ↔ kullanım arası değişim (TOCTOU) ───────────────────────────────


def _approve_launch(snap: dict) -> None:
    from app.agents.runtime import approvals

    r = et.launch(snap["snapshot_id"], "req-appr-" + snap["recipe_sha"][:8])
    assert r["status"] == "needs_approval", r
    approvals.approve(r["approval_id"])


def test_data_changed_between_check_and_spawn_stops_without_spawn(repo, monkeypatch) -> None:
    from app.training import detached_launch as dl

    snap = et.prepare_snapshot(_settings())
    _record(recipe_sha=snap["recipe_sha"])
    _approve_launch(snap)

    def preflight_then_mutate(*a, **k):
        # precheck geçti → onay tüketildi → tam başlatma anında veri değişir
        repo["src"].write_text(repo["src"].read_text("utf-8") + _row(77) + "\n", "utf-8")
        n, _ = dl.ensure_train_split()
        return {"ok": True, "n_train": n}

    monkeypatch.setattr(dl, "preflight_launch", preflight_then_mutate)
    monkeypatch.setattr(
        dl.subprocess,
        "Popen",
        _only_training_popen(lambda *a, **k: pytest.fail("süreç doğmamalı")),
    )
    r = et.launch(snap["snapshot_id"], "req-toctou-0001")
    assert r["status"] == "error" and "reçeteyle" in r["message"], r
    assert not (repo["root"] / "storage" / "train_status.json").exists()


def test_trainer_verifies_bytes_it_reads(tmp_path) -> None:
    import hashlib

    from app.training.peft_lora_train import DataIntegrityError, _load_jsonl

    p = tmp_path / "train.jsonl"
    p.write_text(_row(1) + "\n", encoding="utf-8")
    good = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert len(_load_jsonl(p, good)) == 1
    p.write_text(_row(2) + "\n", encoding="utf-8")  # kontrolden sonra değişti
    with pytest.raises(DataIntegrityError, match="VERİ BÜTÜNLÜĞÜ"):
        _load_jsonl(p, good)


class _Proc:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self):
        return None


def test_concurrent_launches_single_spawn_and_dead_child_lock_is_stale(repo, monkeypatch) -> None:
    from app.training import detached_launch as dl
    from app.training import resource_lock

    snap = et.prepare_snapshot(_settings())
    _record(recipe_sha=snap["recipe_sha"])
    _approve_launch(snap)
    spawned: list[dict] = []
    # Ölü bir pid: alt süreç "çökmüş" sayılır (kilit bayat olmalı).
    dead = _REAL_POPEN(["git", "--version"], stdout=subprocess.DEVNULL)
    dead.wait()

    def popen(cmd, **kw):
        spawned.append(kw["env"])
        return _Proc(dead.pid)

    def preflight(*a, **k):
        n, _ = dl.ensure_train_split()
        return {"ok": True, "n_train": n}

    monkeypatch.setattr(dl, "preflight_launch", preflight)
    monkeypatch.setattr(dl, "_find_hektor", lambda root: ["hektor"])
    monkeypatch.setattr(dl.subprocess, "Popen", _only_training_popen(popen))
    monkeypatch.setattr(dl, "_EARLY_EXIT_WAIT_S", 0.0)

    def race(fn, n: int = 2) -> list[dict]:
        out: list[dict] = []

        def go(i: int) -> None:
            try:
                out.append(fn(i))
            except Exception as exc:  # iş parçacığı hatası testte görünsün
                out.append({"status": "exception", "error": repr(exc)})

        ts = [threading.Thread(target=go, args=(i,)) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        return out

    # (A) Aynı reçete, iki farklı istek, TEK onay: onay atomik tüketilir → tek başlatma.
    out = race(lambda i: et.launch(snap["snapshot_id"], f"req-race-{i:04d}"))
    statuses = sorted(r["status"] for r in out)
    assert statuses == ["needs_approval", "started"], out
    assert len(spawned) == 1 and spawned[0][et.RECIPE_ENV] == snap["recipe_sha"]
    # L-2: alt süreç, üst sürecin TÜKETTİĞİ onayın kimliğini alır (doğrulasın diye).
    started = next(r for r in out if r["status"] == "started")
    assert spawned[0][dl.APPROVAL_ENV] == started["approval_id"]
    # Alt süreç öldü → kilit bayat; yeni başlatmayı kalıcı olarak engellemez.
    assert resource_lock.blocker() is None

    # (B) İki başlatma (ör. web + kolay akış) aynı anda kilide gelir → yalnız biri doğar.
    # Alt süreç bu bölümde CANLI (gerçekte olduğu gibi): ölü pid'e devredilen kilit bayat
    # sayılır ve geç kalan ikinci başlatma onu meşru olarak kırabilirdi → zamanlamaya bağlı test.
    import os as _os

    spawned.clear()
    gate = threading.Barrier(2)

    def popen_alive(cmd, **kw):
        spawned.append(kw["env"])
        return _Proc(_os.getpid())

    monkeypatch.setattr(dl.subprocess, "Popen", _only_training_popen(popen_alive))

    def preflight_sync(*a, **k):
        gate.wait(timeout=10)
        return preflight()

    monkeypatch.setattr(dl, "preflight_launch", preflight_sync)
    (repo["root"] / "storage" / "train_status.json").unlink(missing_ok=True)
    res = race(
        lambda i: dl.launch(
            adapter_name=snap["adapter_name"],
            base_model=snap["base_model"],
            profile=snap["profile"],
            max_examples=snap["max_examples"],
            recipe_sha=snap["recipe_sha"],
            early_exit_wait_s=0,
        )
    )
    oks = [r for r in res if r.get("ok")]
    assert len(oks) == 1 and len(spawned) == 1, res
    resource_lock.release(spawned[0][resource_lock.TOKEN_ENV])


# --- Kademe 2 (2026-10-06) F3-5: veri seçimi → karışım → anlık görüntü → reçete → alt süreç ---
# Her halka ayrı bozulur; eğitim alt süreci eğiticiye ULAŞMADAN ve onay/kilit TÜKETİLMEDEN durmalı.


def _chain_child(repo, monkeypatch):
    from app.lora.weight_decision import WeightDecisionStore
    from app.training import resource_lock

    seen = _fake_trainer(monkeypatch)
    snap = et.prepare_snapshot(_settings())
    _record(recipe_sha=snap["recipe_sha"])
    monkeypatch.setenv("HEKTOR_TRAIN_SUPERVISED", "1")
    monkeypatch.setenv(et.RECIPE_ENV, snap["recipe_sha"])

    def run(weights=None):
        from app.agents.runtime import approvals

        WeightDecisionStore().record(weights or snap["mix_weights"], snap["mix_label"], "t")
        token = _parent_lock(monkeypatch)
        before = len(approvals.list_approvals())
        res = _cli("--profile", snap["profile"], "--base-model", snap["base_model"])
        resource_lock.release(token)  # üst süreç gibi: alt süreç bitince kilidi bırak
        assert len(approvals.list_approvals()) == before  # hiçbir onay istenmedi/tüketilmedi
        return res

    return snap, seen, run


def test_f3_5_chain_each_link_mismatch_stops_before_trainer(repo, monkeypatch) -> None:
    from app.feedback.chat_dataset import selection_path

    snap, seen, run = _chain_child(repo, monkeypatch)
    # 0) Sağlam zincir → eğitici ÇAĞRILIR ve onaylı özetleri alır.
    ok = run()
    assert ok.exit_code == 0, ok.output
    assert len(seen) == 1 and seen[0].expect_train_sha256 == snap["train_sha256"]

    # 1) Karışım: reçetedekinden farklı ağırlık kararı.
    other = dict.fromkeys(snap["mix_weights"], round(1 / len(snap["mix_weights"]), 6))
    if other == snap["mix_weights"]:
        other = {**other, next(iter(other)): 0.9}
    bad = run(weights=other)
    assert bad.exit_code == 11 and "karışım" in bad.output, bad.output

    # 2) Veri seçimi: sohbet veri sürümü onaydan sonra değişti.
    sel = selection_path()
    sel.write_text(json.dumps({"version_id": "dsv_baska", "train_sha256": "e" * 64}), "utf-8")
    bad = run()
    assert bad.exit_code != 0 and "sohbet veri sürümü" in bad.output, bad.output
    sel.unlink()

    # 3) Anlık görüntü: reçete dosyası elle değiştirildi.
    rp = et.snapshots_dir() / snap["snapshot_id"] / "recipe.json"
    good = rp.read_text("utf-8")
    rp.write_text(good.replace(f'"{snap["profile"]}"', '"baska_profil"', 1), "utf-8")
    bad = run()
    # Kademe 2 kapısı reçeteyi zaten doğrularken durur (10); bağlama kontrolü de (11) aynı
    # nedenle durdururdu.
    assert bad.exit_code in (10, 11) and "doğrulanamadı" in bad.output, bad.output
    rp.write_text(good, "utf-8")

    # 4) Veri: lora_sft.jsonl onaydan sonra değişti.
    src = repo["src"]
    orig = src.read_text("utf-8")
    src.write_text(orig + _row(4242) + "\n", encoding="utf-8")
    bad = run()
    assert bad.exit_code in (10, 11) and "veri" in bad.output, bad.output
    src.write_text(orig, encoding="utf-8")

    assert len(seen) == 1  # bozuk halkaların HİÇBİRİ eğiticiye ulaşmadı


def test_f3_5_trainer_rejects_changed_split_before_model_load(tmp_path, monkeypatch) -> None:
    """Son halka: kontrol ile kullanım arasında dosya değişirse eğitici model YÜKLEMEDEN durur."""
    import sys

    from app.training import peft_lora_train as plt

    tr = tmp_path / "train.jsonl"
    tr.write_text(_row(1) + "\n", "utf-8")
    va = tmp_path / "valid.jsonl"
    va.write_text(_row(2) + "\n", "utf-8")
    monkeypatch.setattr(plt, "_check_deps", lambda: [])
    monkeypatch.setitem(sys.modules, "torch", None)  # model yükleme yoluna girerse ImportError
    cfg = plt.PeftTrainConfig(
        base_model="yok/model",
        train_jsonl=tr,
        valid_jsonl=va,
        adapter_output_path=tmp_path / "ad",
        expect_train_sha256="0" * 64,
    )
    out = plt.train(cfg)
    assert out["ok"] is False and "VERİ BÜTÜNLÜĞÜ" in out["error"]


# --- Reçeteye sınırlı risk kabulü (F3-4 pilot) başka reçeteye / veri kapsamına taşınmaz ---


def test_recipe_limited_risk_acceptance_does_not_carry_over(repo) -> None:
    limited = [
        {
            "id": "F3-4",
            "status": "risk_kabul",
            "kapsam": "yalniz_recete",
            "gerekce": "yalnız 64 örneklik teknik pilot; valid loss kalite kanıtı değil",
            "sinir": {"max_examples": 64, "max_optimizer_steps": 64},
        }
    ]
    pilot = et.prepare_snapshot(_settings(max_examples=64))
    data_sha = pilot["data_sha256"]
    with pytest.raises(et.EasyTrainError, match="YALNIZ --recipe-sha"):
        et.record_kademe2(
            scope={"recipe_sha": pilot["recipe_sha"], "data_sha256": data_sha},
            findings=limited,
            closure_evidence="pilot kapsamı; make ci yeşil (commit abc123)",
            reviewer="insan",
        )
    et.record_kademe2(
        scope={"recipe_sha": pilot["recipe_sha"]},
        findings=limited,
        closure_evidence="pilot kapsamı; make ci yeşil (commit abc123)",
        reviewer="insan",
    )
    assert et.kademe2_blocker(recipe_sha=pilot["recipe_sha"]) is None  # pilot geçer
    assert et.kademe2_blocker() is not None  # reçetesiz yol (CLI/start-train) kapsanmaz
    full = et.prepare_snapshot(_settings(max_examples=0, adapter_name="hektor_lora_tam"))
    assert full["recipe_sha"] != pilot["recipe_sha"]
    assert et.kademe2_blocker(recipe_sha=full["recipe_sha"]) is not None  # tam eğitime taşınmaz


def test_recipe_has_limited_acceptance_marks_only_pilot(repo) -> None:
    pilot = et.prepare_snapshot(_settings(max_examples=64))
    other = et.prepare_snapshot(_settings(max_examples=0, adapter_name="hektor_lora_tam"))
    et.record_kademe2(
        scope={"recipe_sha": pilot["recipe_sha"]},
        findings=[
            {
                "id": "F3-4",
                "status": "risk_kabul",
                "kapsam": "yalniz_recete",
                "gerekce": "yalnız 64 örneklik teknik pilot; kalite kanıtı değil",
                "sinir": {"max_examples": 64, "max_optimizer_steps": 64},
            }
        ],
        closure_evidence="pilot kapsamı; make ci yeşil (commit abc123)",
        reviewer="insan",
    )
    assert et.recipe_has_limited_acceptance(pilot["recipe_sha"])
    assert not et.recipe_has_limited_acceptance(other["recipe_sha"])
    assert not et.recipe_has_limited_acceptance("")


def test_limited_acceptance_is_bound_to_numeric_pilot_limits(repo) -> None:
    """İnceleme kararı: F3-4/L-4/D-2 kabulü yalnız '≤64 örnek, ≤N optimizer adımı' reçetesine."""
    pilot = et.prepare_snapshot(_settings(max_examples=64))
    size = et.recipe_training_size(pilot)
    assert size["examples"] <= 64

    def rec(snap, sinir):
        f = {
            "id": "F3-4",
            "status": "risk_kabul",
            "kapsam": "yalniz_recete",
            "gerekce": "yalnız teknik pilot; valid loss kalite kanıtı değil",
        }
        if sinir is not None:
            f["sinir"] = sinir
        return et.record_kademe2(
            scope={"recipe_sha": snap["recipe_sha"]},
            findings=[f],
            closure_evidence="pilot kapsamı; make ci yeşil (commit abc123)",
            reviewer="insan",
        )

    with pytest.raises(et.EasyTrainError, match="sayısal sınır"):
        rec(pilot, None)
    with pytest.raises(et.EasyTrainError, match="optimizer adımı"):
        rec(pilot, {"max_examples": 64, "max_optimizer_steps": size["optimizer_steps"] - 1})
    with pytest.raises(et.EasyTrainError, match="örnek"):
        rec(pilot, {"max_examples": size["examples"] - 1, "max_optimizer_steps": 999})
    full = et.prepare_snapshot(_settings(max_examples=0, adapter_name="hektor_lora_tam"))
    with pytest.raises(et.EasyTrainError, match="tavansız"):
        rec(full, {"max_examples": 64, "max_optimizer_steps": 8})
    out = rec(pilot, {"max_examples": 64, "max_optimizer_steps": size["optimizer_steps"]})
    assert out["findings"][0]["sinir_dogrulama"]["optimizer_steps"] == size["optimizer_steps"]
    assert et.recipe_has_limited_acceptance(pilot["recipe_sha"])


def test_pilot_recipe_size_matches_review_limits() -> None:
    """Gerçek pilot reçetesi: moe30b_attn_long, 64 örnek → 64 mikro-adım / birikim 8 = 8 adım."""
    size = et.recipe_training_size(
        {"n_train": 1500, "max_examples": 64, "profile": "moe30b_attn_long"}
    )
    assert size == {
        "examples": 64,
        "micro_steps": 64,
        "epochs": 1,
        "grad_accum": 8,
        "optimizer_steps": 8,
    }


def test_cloud_out_of_scope_record_lapses_when_cloud_enabled(repo, monkeypatch) -> None:
    """C-1/C-5/C-6: 'bulut kapalı' koşuluyla kapsam dışı kayıt, bulut açılınca eğitimi kapsamaz."""
    from app.config import get_settings

    finding = {
        "id": "C-1",
        "status": "risk_kabul",
        "kapsam": "kayit",
        "kosul": "bulut_kapali",
        "gerekce": "bulut ikinci görüş kapalı; açılmadan önce ayrı iş",
    }
    data_sha = _data_sha()
    et.record_kademe2(
        scope={"data_sha256": data_sha},
        findings=[finding],
        closure_evidence="bulut kapalı doğrulandı; make ci yeşil",
        reviewer="insan",
    )
    assert et.kademe2_blocker() is None
    monkeypatch.setenv("HEKTOR_CLOUD_SECOND_OPINION", "1")
    get_settings.cache_clear()
    assert et.kademe2_blocker() is not None  # koşul bozuldu → kayıt geçmez
    with pytest.raises(et.EasyTrainError, match="Koşul şu an sağlanmıyor"):
        et.record_kademe2(
            scope={"data_sha256": data_sha},
            findings=[finding],
            closure_evidence="bulut kapalı doğrulandı; make ci yeşil",
            reviewer="insan",
        )
    with pytest.raises(et.EasyTrainError, match="Tanımsız koşul"):
        et.record_kademe2(
            scope={"data_sha256": data_sha},
            findings=[{**finding, "kosul": "yok_boyle"}],
            closure_evidence="bulut kapalı doğrulandı; make ci yeşil",
            reviewer="insan",
        )
