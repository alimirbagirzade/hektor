"""Faz 2C — kolay eğitim akışı.

Kabul şartları: son ayarlar hazır gelir; hazırlık kontrolü salt okunur (hiçbir dosya yazılmaz /
bölünmez); veri anlık görüntüsü + reçete ayrı ve açık adım; onay reçeteye bağlı (biri değişirse
eski onay kullanılamaz); iptal/sızıntı/kaynak/veri kontrolleri başlatmadan hemen önce yeniden;
Kademe 2 kaydı kod durumu + kapsam + kapanmış bulgular + kanıt ister; ucuz kontroller onay
tüketilmeden biter; tekrarlanan istek çift eğitim başlatmaz.
"""

from __future__ import annotations

import json

import pytest
from tests.chat_learning_helpers import iso  # noqa: F401

from app.training import easy_train as et

CODE = {"ok": True, "code_sha": "c" * 64, "dirty": [], "head": "abc"}


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
def env(iso, monkeypatch):  # noqa: F811
    from app.config import get_settings
    from app.training import detached_launch

    root = get_settings().root
    src = root / "data" / "lora_sft" / "lora_sft.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text("\n".join(_row(i) for i in range(60)) + "\n", encoding="utf-8")
    monkeypatch.setattr(et, "code_state_provider", lambda: dict(CODE))
    monkeypatch.setattr(detached_launch, "_pretrain_gate_blockers", lambda s: [])
    monkeypatch.setattr("app.lora.mix_cli.leakage_eval_items", lambda: [])
    calls: list[dict] = []

    def fake_launch(**kw):
        calls.append(kw)
        return {"ok": True, "message": "başlatıldı (sahte)", "adapter": kw["adapter_name"]}

    monkeypatch.setattr(detached_launch, "launch", fake_launch)
    return {"root": root, "src": src, "calls": calls}


SETTINGS = {"adapter_name": "hektor_lora_t1", "mix_profile": None}


def _settings(**kw):
    from app.lora.mix_common import load_mix_config

    mix = sorted((load_mix_config().get("profiles") or {}).keys())[0]
    return {**SETTINGS, "mix_profile": mix, **kw}


def _k2(scope):
    return et.record_kademe2(
        scope=scope,
        findings=[{"id": "F1", "status": "duzeltildi", "evidence": "test eklendi"}],
        closure_evidence="tüm bulgular kapandı; make ci yeşil (commit abc)",
        reviewer="insan",
    )


def test_state_is_read_only(env) -> None:
    before = sorted(p.relative_to(env["root"]) for p in env["root"].rglob("*") if p.is_file())
    st = et.state()
    after = sorted(p.relative_to(env["root"]) for p in env["root"].rglob("*") if p.is_file())
    new = [p for p in after if p not in before and "sqlite" not in str(p) and p.suffix != ".db"]
    assert new == [], new  # jsonl bölmesi, ayar, anlık görüntü YAZILMADI
    keys = {i["key"]: i["ok"] for i in st["readiness"]["items"]}
    assert keys["veri"] and not keys["kademe2"]
    assert st["settings"]["_source"].startswith("varsayılan")


def test_snapshot_immutable_and_settings_remembered(env) -> None:
    a = et.prepare_snapshot(_settings())
    b = et.prepare_snapshot(_settings())
    assert a["snapshot_id"] == b["snapshot_id"] and a["recipe_sha"] == b["recipe_sha"]
    assert et.last_settings()["adapter_name"] == "hektor_lora_t1"
    assert et.last_settings()["_source"] == "son kullanılan"
    c = et.prepare_snapshot(_settings(adapter_name="hektor_lora_t2"))
    assert c["recipe_sha"] != a["recipe_sha"]
    env["src"].write_text(env["src"].read_text("utf-8") + _row(999) + "\n", encoding="utf-8")
    d = et.prepare_snapshot(_settings())
    assert d["recipe_sha"] != a["recipe_sha"]  # veri değişti → yeni reçete


def test_kademe2_record_requirements(env, monkeypatch) -> None:
    with pytest.raises(et.EasyTrainError, match="Kapanmamış"):
        et.record_kademe2(
            scope={"data_sha256": "x"},
            findings=[{"status": "acik"}],
            closure_evidence="x" * 30,
            reviewer="i",
        )
    with pytest.raises(et.EasyTrainError, match="Kapsam"):
        et.record_kademe2(scope={}, findings=[], closure_evidence="x" * 30, reviewer="i")
    monkeypatch.setattr(et, "code_state_provider", lambda: {"ok": False, "dirty": ["app/x.py"]})
    with pytest.raises(et.EasyTrainError, match="temiz değil"):
        _k2({"data_sha256": "x"})


def test_launch_blocked_without_kademe2_and_no_approval_consumed(env) -> None:
    from app.agents.runtime import approvals

    snap = et.prepare_snapshot(_settings())
    r = et.launch(snap["snapshot_id"], "req-0001-aaaa")
    assert r["status"] == "blocked" and any("Kademe 2" in p for p in r["problems"])
    assert approvals.list_approvals() == []  # onay isteği bile açılmadı
    assert env["calls"] == []


def test_approval_bound_to_recipe_and_idempotent_launch(env) -> None:
    from app.agents.runtime import approvals

    snap = et.prepare_snapshot(_settings())
    _k2({"recipe_sha": snap["recipe_sha"]})
    r1 = et.launch(snap["snapshot_id"], "req-0002-aaaa")
    assert r1["status"] == "needs_approval" and env["calls"] == []
    approvals.approve(r1["approval_id"])
    # Başka reçete (farklı adapter) bu onayı TÜKETEMEZ.
    other = et.prepare_snapshot(_settings(adapter_name="hektor_lora_b"))
    _k2({"recipe_sha": other["recipe_sha"]})
    r_other = et.launch(other["snapshot_id"], "req-0003-aaaa")
    assert r_other["status"] == "needs_approval" and env["calls"] == []
    # Doğru reçete → başlar; aynı istek tekrar → yeniden başlatılmaz.
    r2 = et.launch(snap["snapshot_id"], "req-0004-aaaa")
    assert r2["status"] == "started" and len(env["calls"]) == 1
    assert env["calls"][0]["adapter_name"] == "hektor_lora_t1"
    r3 = et.launch(snap["snapshot_id"], "req-0004-aaaa")
    assert r3["replayed"] and len(env["calls"]) == 1


def test_recheck_right_before_launch(env, monkeypatch) -> None:
    snap = et.prepare_snapshot(_settings())
    _k2({"recipe_sha": snap["recipe_sha"]})
    env["src"].write_text(env["src"].read_text("utf-8") + _row(500) + "\n", encoding="utf-8")
    r = et.launch(snap["snapshot_id"], "req-0005-aaaa")
    assert r["status"] == "blocked" and any("değişti" in p for p in r["problems"])
    # Sızıntı: anlık görüntüdeki soru korunan eval setinde çıkarsa.
    from app.evals.profile.schema import EvalItem

    env["src"].write_text("\n".join(_row(i) for i in range(60)) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.lora.mix_cli.leakage_eval_items",
        lambda: [EvalItem(id="g1", domain="trading", question="Soru 3: risk yönetimi nedir?")],
    )
    r2 = et.launch(snap["snapshot_id"], "req-0006-aaaa")
    assert r2["status"] == "blocked" and any("sızıntı" in p for p in r2["problems"])
    # Anlık görüntü dosyası kurcalanırsa reddedilir.
    p = et.snapshots_dir() / snap["snapshot_id"] / "train.jsonl"
    p.write_text(p.read_text("utf-8") + "\n{}", encoding="utf-8")
    with pytest.raises(et.EasyTrainError, match="bozulmuş"):
        et.launch(snap["snapshot_id"], "req-0007-aaaa")


def test_web_endpoints_human_only(env) -> None:
    from fastapi.testclient import TestClient

    from app.web.security import require_human
    from app.web.server import app

    client = TestClient(app)
    assert client.get("/api/train-flow/state").status_code == 200
    bad = client.post("/api/train-flow/snapshot", json={"adapter_name": "../x"})
    assert bad.status_code == 422
    for rt in client.app.routes:
        if getattr(rt, "path", "") in ("/api/train-flow/snapshot", "/api/train-flow/launch"):
            assert require_human in {d.call for d in rt.dependant.dependencies}
