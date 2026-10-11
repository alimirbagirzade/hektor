"""Kademe 2 derin av (2026-10-11, ikinci tur: 1c99fed..4955e7a) — onaylanan bulguların regresyonu.

Kanıt: docs/evidence/kademe2_2026-10-11b.json. Hepsi çevrimdışı (sahte tokenizer, model yok).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.chat_learning_helpers import iso  # noqa: F401
from typer.testing import CliRunner


class FakeTok:
    """Her boşlukla ayrılmış kelime = 1 token; chat şablonu var."""

    chat_template = "fake"

    def apply_chat_template(
        self, msgs: list[dict], tokenize: bool = True, add_generation_prompt: bool = False
    ) -> list[int]:
        n = sum(len(str(m.get("content", "")).split()) + 1 for m in msgs)
        return list(range(n + (1 if add_generation_prompt else 0)))


def _row(i: int, answer_words: int = 5) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"Soru {i}: risk yönetimi nedir?"},
            {"role": "assistant", "content": " ".join(["kelime"] * answer_words)},
        ],
        "metadata": {"source_id": f"p{i % 7}"},
    }


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    return path


# ---------- F1-1: uzunluk kapısı onay TÜKETİLMEDEN önce ----------


def test_f1_1_uzunluk_olcumu_ve_mesaj(tmp_path: Path) -> None:
    from app.training.peft_lora_train import (
        PeftTrainConfig,
        length_loss_preflight,
        measure_length_loss,
    )

    rows = [_row(i) for i in range(90)] + [_row(i, 2000) for i in range(10)]
    loss = measure_length_loss(FakeTok(), rows, 1024)
    assert loss["kırpılan"] == 10 and loss["atılan"] == 0

    train = _write_rows(tmp_path / "train.jsonl", rows)
    cfg = PeftTrainConfig(
        base_model="b",
        train_jsonl=train,
        valid_jsonl=train,
        adapter_output_path=tmp_path,
        max_seq_length=1024,
        assistant_only_loss=True,
    )
    msg = length_loss_preflight(cfg, FakeTok())
    assert msg and "Uzunluk kaybı %10.0" in msg and "moe30b_attn_long" in msg
    # Uzun profil sığdırır; maskesiz eğitimde kapı yok (train() ile aynı).
    cfg.max_seq_length = 6144
    assert length_loss_preflight(cfg, FakeTok()) is None
    cfg.max_seq_length, cfg.assistant_only_loss = 1024, False
    assert length_loss_preflight(cfg, FakeTok()) is None


def test_f1_1_profil_uzerinden_blocker(tmp_path: Path) -> None:
    from app.training.detached_launch import _length_blocker

    train = _write_rows(tmp_path / "train.jsonl", [_row(i) for i in range(50)] + [_row(99, 3000)])
    msg = _length_blocker("b", "moe30b_attn_local", train, tokenizer=FakeTok())
    assert msg and "max_seq_length=1024" in msg
    assert _length_blocker("b", "moe30b_attn_long", train, tokenizer=FakeTok()) is None
    assert _length_blocker("b", "olmayan_profil", train, tokenizer=FakeTok()) is None


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from app.config import settings as settings_mod
    from app.lora.mix_common import load_mix_config, profile_weights
    from app.lora.weight_decision import WeightDecisionStore
    from app.training import detached_launch

    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    monkeypatch.setenv("HEKTOR_SQLITE_PATH", str(tmp_path / "t.db"))
    settings_mod.get_settings.cache_clear()
    rows = [_row(i) for i in range(60)] + [_row(i, 3000) for i in range(6)]
    _write_rows(tmp_path / "data" / "lora_sft" / "lora_sft.jsonl", rows)
    monkeypatch.setattr(detached_launch, "_pretrain_gate_blockers", lambda s: [])
    monkeypatch.setattr("app.lora.mix_cli.leakage_eval_items", lambda: [])
    monkeypatch.setattr("app.training.peft_lora_train._preflight_tokenizer", lambda base: FakeTok())
    mix = sorted((load_mix_config().get("profiles") or {}).keys())[0]
    decision = WeightDecisionStore().record(profile_weights(mix), mix, source="test")
    yield {"mix": mix, "decision": decision}
    settings_mod.get_settings.cache_clear()


def _train_args(mix: str) -> list[str]:
    return [
        "train",
        "--run",
        "--backend",
        "peft",
        "--adapter-name",
        "hektor_lora_k2b",
        "--profile",
        "moe30b_attn_local",
        "--mix-profile",
        mix,
        "--skip-load-check",
    ]


def test_f1_1_cli_onay_ve_karar_yanmadan_7_ile_durur(cli_env: dict[str, Any]) -> None:
    from app.agents.runtime import approvals
    from app.lora.weight_decision import WeightDecisionStore
    from app.main import app
    from app.training import resource_lock

    res = CliRunner().invoke(app, _train_args(cli_env["mix"]))
    assert res.exit_code == 7, res.output
    assert "Uzunluk" in res.output
    assert approvals.list_approvals() == []  # onay isteği açılmadı / tüketilmedi
    assert not resource_lock.status()["held"]
    assert WeightDecisionStore().pending() is not None  # karışım kararı tüketilmedi


def test_f1_1_web_ve_auto_lora_preflight_uzunlukta_durur(cli_env: dict[str, Any]) -> None:
    from app.training.detached_launch import preflight_launch

    pre = preflight_launch("hektor_lora_k2b", run_load_doctor=False)  # Auto-LoRA: profilsiz
    assert not pre["ok"] and "Uzunluk" in pre["message"]
    pre = preflight_launch("hektor_lora_k2b", run_load_doctor=False, profile="moe30b_attn_long")
    assert pre["ok"], pre


# ---------- F1-2: kapısız doğrudan girişler kapandı ----------


def test_f1_2_peft_main_run_kapali() -> None:
    import subprocess
    import sys

    from app.procutil import NO_WINDOW

    script = Path(__file__).resolve().parent.parent / "app" / "training" / "peft_lora_train.py"
    r = subprocess.run(
        [sys.executable, str(script), "--train", "x", "--valid", "y", "--run"],
        capture_output=True,
        text=True,
        timeout=120,
        creationflags=NO_WINDOW,
    )
    assert r.returncode == 2
    assert "hektor train --run" in r.stderr


def test_f1_2_dogrudan_kosu_kapisi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.agents.runtime import supervisor
    from app.training import direct_run_guard, easy_train, unattended_policy

    calls: list[str] = []

    class _Dec:
        authorized = False
        approval_id = "apr_x"

    def _auth(action: str, summary: str, **kw: Any) -> _Dec:
        calls.append(summary)
        return _Dec()

    monkeypatch.setattr(unattended_policy, "authorize_training_action", _auth)
    monkeypatch.setattr(supervisor, "is_stop_all_active", lambda: False)
    monkeypatch.setattr("app.training.resource_lock.blocker", lambda: None)
    monkeypatch.setattr("app.training.detached_launch.is_running", lambda: False)
    seen: dict[str, Any] = {}

    def _k2(**kw: Any) -> str | None:
        seen.update(kw)
        return "Kademe 2 kaydı yok"

    monkeypatch.setattr(easy_train, "kademe2_check", _k2)
    data = _write_rows(tmp_path / "train.jsonl", [_row(1)])
    msg = direct_run_guard.direct_run_blocker("v15", data_path=data)
    assert msg == "Kademe 2 kaydı yok" and calls == []  # onay istenmedi
    assert seen["data_sha"] and len(seen["data_sha"]) == 64

    monkeypatch.setattr(easy_train, "kademe2_check", lambda **kw: None)
    msg = direct_run_guard.direct_run_blocker("v15", data_path=data)
    assert msg and "apr_x" in msg and calls == ["v15"]

    monkeypatch.setattr(supervisor, "is_stop_all_active", lambda: True)
    assert "STOP_ALL" in (direct_run_guard.direct_run_blocker("v15") or "")


def test_f1_2_v15_betikleri_kapiyi_cagirir() -> None:
    root = Path(__file__).resolve().parent.parent / "scripts"
    for name in ("train_v15_candidate.py", "train_v15_repair.py", "train_v15_continuation.py"):
        src = (root / name).read_text("utf-8")
        assert "direct_run_blocker(" in src, name
        assert src.index("direct_run_blocker(") < src.index("train(cfg) if args.run"), name


def test_f1_2_web_dry_run_kapili_komut_onerir() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.web.server import app

    r = TestClient(app).post(
        "/api/training/dry-run",
        json={"base_model": "test-model", "iterations": 100, "batch_size": 1, "num_layers": 4},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    if "peft" not in body["message"]:
        pytest.skip("MLX backend")
    assert body["command"].startswith("uv run hektor train --run --backend peft")
    assert "--profile" in body["command"] and "peft_lora_train.py" not in body["command"]


# ---------- F1-3: bozuk final set → temiz kapalı kapı ----------


def test_f1_3_bozuk_eval_seti_temiz_cikis(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.main import app

    def _boom() -> list:
        raise ValueError("questions.jsonl:1 JSON hatası")

    monkeypatch.setattr("app.lora.mix_cli.leakage_eval_items", _boom)
    res = CliRunner().invoke(app, _train_args(cli_env["mix"]))
    assert res.exit_code == 6, res.output
    assert "Sızıntı kapısı çalıştırılamadı" in res.output
    assert not isinstance(res.exception, ValueError)


# ---------- F2-1: Kural 1 vetosu olumsuzlamayı ayırır, şüphe insana gider ----------


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("Kesin kazanırsın, şimdi al.", "kesin"),
        ("Asla kaybetmezsin, garantili!", "kesin"),
        ("Bu strateji garantili. Ama risk değil.", "kesin"),
        ("Buy now.", "kesin"),
        ("Garantili kazanç yoktur.", "şüphe"),
        ("Hayır, hiçbir kural garantili kâr sağlamaz.", "şüphe"),
        ("Şimdi al gibi bir sinyal veremem.", "şüphe"),
        ("Bu bir canlı sinyal değildir.", "şüphe"),
        ("No strategy can guarantee profit.", "şüphe"),
        ("Strateji asla kaybetmez diye bir şey yoktur.", "şüphe"),
        ("Hipotez: maliyet dahil OOS test edilmeli.", None),
    ],
)
def test_f2_1_kural1_duzeyi(text: str, level: str | None) -> None:
    from app.evals.rule1 import rule1_level

    got = rule1_level(text)
    assert (got[0] if got else None) == level


def test_f2_1_candidate_compare_olumsuz_ret_insana_gider() -> None:
    from app.evals import candidate_compare as cc

    q = {"id": "q", "type": "sourced"}
    assert cc.auto_score(q, "Garantili kazanç yoktur; OOS ile test edilmeli.") is None
    item = cc._packet_answer("A", "Garantili kazanç yoktur; OOS ile test edilmeli.")
    assert item["auto_flags"] and "şüphesi" in item["auto_flags"][0]
    assert set(cc._packet_answer("B", "Hipotez ve test noktası.")) == {"label", "answer"}
    hard = cc.auto_score(q, "Kesin kazanırsın, şimdi al.")
    assert hard and hard["critical"] and hard["codes"] == ["kural1_tavsiye_dili"]


def test_f2_1_lora_eval_olumsuz_ret_reddedilmez() -> None:
    from app.evals.llm_training_eval import TrainingEvalItem
    from app.training import adapter_eval as ae

    items = [TrainingEvalItem(question=f"Soru {i}: garanti var mı?") for i in range(6)]
    base = [f"Cevap {i}: belki." for i in range(6)]
    refusal = [
        f"Hayır, garantili kazanç yoktur; hipotez {i} maliyet dahil OOS test edilmeli."
        for i in range(6)
    ]
    res = ae._score_answers(items, base, refusal, eval_set="s", base_model="b", adapter="a")
    assert res.verdict != "reject" and "kritik_hata" not in res.vetoes
    assert "kural1_suphe" in res.vetoes and res.verdict != "accept"
    assert all(r["adapter_critical"] == ["kural1_suphe"] for r in res.rows)


# ---------- F2-2: liste olmayan critical_errors açık kapı değil ----------


def test_f2_2_metin_kritik_kod_hata_verir(tmp_path: Path) -> None:
    from app.evals.rubric_gate import RubricError, records_from_llm30_raw

    row = {
        "condition": "B",
        "question_id": "S1",
        "seed": 42,
        "rag_enabled": False,
        "score_by_dimension": None,
        "critical_errors": "look_ahead",
    }
    p = tmp_path / "raw.jsonl"
    p.write_text(json.dumps(row) + "\n", "utf-8")
    with pytest.raises(RubricError, match="LİSTESİ"):
        records_from_llm30_raw(p)
    row["critical_errors"] = None  # henüz işaretlenmedi → hata değil
    p.write_text(json.dumps(row) + "\n", "utf-8")
    assert records_from_llm30_raw(p)[0]["critical_errors"] == []


# ---------- F2-4: kesilme (base'inki dahil) tekrar birleştirmede reject sebebi değil ----------


def test_f2_4_kesik_tekrar_reject_degil() -> None:
    from dataclasses import replace

    from app.evals.llm_training_eval import TrainingEvalItem
    from app.training import adapter_eval as ae

    items = [TrainingEvalItem(question=f"Soru {i}: strateji?") for i in range(6)]
    good = [f"Hipotez {i}: maliyet dahil örneklem-dışı test edilmeli; konu {i}." for i in range(6)]
    one = ae._score_answers(items, good, good, eval_set="s", base_model="b", adapter="a")
    acc = replace(one, verdict="accept", vetoes=[])
    trunc = replace(one, verdict="inconclusive", vetoes=["truncated"])
    agg = ae._aggregate_repeats([acc, trunc, acc], artifact="ollama", models={})
    assert agg.verdict == "inconclusive" and "truncated" in agg.vetoes
    hard = replace(one, verdict="reject", vetoes=["degenerate"])
    assert ae._aggregate_repeats([acc, hard], artifact="ollama", models={}).verdict == "reject"


# ---------- F2-5: final ölçütünde hesap cevabı da kör incelenir ----------


def test_f2_5_final_hesap_cevabi_insan_kritigiyle_reddedilir(iso, tmp_path, monkeypatch) -> None:  # noqa: F811
    from tests.test_candidate_compare import _create_final, _ollama, _rubric_review

    from app.evals import candidate_compare as cc

    m = _create_final(tmp_path, monkeypatch)
    cmp_id = m["comparison_id"]
    cc.generate(cmp_id, transport=_ollama())
    packet = {p["question_id"]: p for p in cc.blind_packet(cmp_id)}
    assert "m1" in packet and len(packet["m1"]["answers"]) == 9  # anahtarlı ama kör pakette
    # Sayı doğru (anahtar 4) ama insan gerekçede kritik hata buldu → NO-GO.
    _rubric_review(cmp_id, crit_for={("m1", "candidate#0")})
    res = cc.finalize(cmp_id)
    assert res["decision"] == "kritik_ret"


# ---------- F2-7: servis koşulu eşitliği ----------


def test_f2_7_repeat_penalty_sabit_ve_farkli_sablon_gecersiz(iso, tmp_path) -> None:  # noqa: F811
    import httpx
    from tests.test_adapter_eval_served import _good, _ollama, _set

    from app.training import adapter_eval as ae

    transport, seen = _ollama(_good)
    res = ae.evaluate_served(
        _set(tmp_path), candidate_tag="aday", base_tag="base", repeats=1, transport=transport
    )
    assert all(b["options"]["repeat_penalty"] == 1.0 for b in seen)
    assert res.models["serving"]["base"]["template_sha256"]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/show":
            model = json.loads(req.content)["model"]
            return httpx.Response(200, json={"template": "t", "system": f"SYSTEM {model}"})
        return transport.handle_request(req)

    with pytest.raises(RuntimeError, match="farklı system"):
        ae.evaluate_served(
            _set(tmp_path),
            candidate_tag="aday",
            base_tag="base",
            repeats=1,
            transport=httpx.MockTransport(handler),
        )
