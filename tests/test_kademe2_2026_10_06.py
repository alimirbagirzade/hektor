"""Kademe 2 derin av (2026-10-06, main 3ef2079) — onaylanan bulguların regresyon testleri.

Her test bir bulgunun somut tetikleme senaryosunu yeniden kurar (bkz.
docs/evidence/kademe2_2026-10-06.json). Model / eğitim / ağ YOK.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from tests.chat_learning_helpers import iso  # noqa: F401

# ── F1-1: preflight reçete kapsamlı Kademe 2 kaydını görür ─────────────────────────────


def _preflight_env(monkeypatch) -> list[dict]:
    from app.training import detached_launch as dl
    from app.training import easy_train

    seen: list[dict] = []
    monkeypatch.setattr(dl, "is_running", lambda: False)
    monkeypatch.setattr("app.feedback.resource_guard.chat_lease_blocker", lambda root: None)
    monkeypatch.setattr(dl, "_adapter_dir_blocker", lambda p: None)
    monkeypatch.setattr(
        "app.lora.weight_decision.WeightDecisionStore.pending", lambda self, now=None: object()
    )
    monkeypatch.setattr(dl, "ensure_train_split", lambda s=None: (5, 1))
    monkeypatch.setattr(dl, "_pretrain_gate_blockers", lambda s: [])
    monkeypatch.setattr("app.lora.mix_cli.run_leakage_check", lambda p: {"clean": True})
    monkeypatch.setattr(dl, "_recipe_blockers", lambda b, p: ["RAM yetersiz (test)"])

    def k2(**kw):
        seen.append(kw)
        return None

    monkeypatch.setattr(easy_train, "kademe2_check", k2)
    return seen


def test_f1_1_preflight_passes_recipe_scope_to_kademe2(iso, monkeypatch) -> None:  # noqa: F811
    from app.training.detached_launch import preflight_launch

    seen = _preflight_env(monkeypatch)
    out = preflight_launch("hektor_lora_pilot", run_load_doctor=False, recipe_sha="a" * 64)
    assert out["ok"] and seen[-1] == {"recipe_sha": "a" * 64}
    preflight_launch("hektor_lora_pilot", run_load_doctor=False)
    assert seen[-1] == {"recipe_sha": ""}  # reçetesiz yollar değişmedi


def test_f1_2_recipe_blockers_run_before_spawn_when_recipe_bound(iso, monkeypatch) -> None:  # noqa: F811
    from app.training.detached_launch import preflight_launch

    _preflight_env(monkeypatch)
    out = preflight_launch(
        "hektor_lora_pilot", run_load_doctor=False, check_recipe=True, recipe_sha="a" * 64
    )
    assert not out["ok"] and "RAM yetersiz" in out["message"]


# ── F1-3: kolay akış onay eylemi eğitim yetkisi sayılır ─────────────────────────────────


def test_f1_3_easy_flow_approval_action_is_training_action() -> None:
    from app.training.train_guard import find_run_approval, is_training_action

    assert is_training_action("train_run") and is_training_action("train_run:0123456789abcdef")
    assert not is_training_action("chat_run") and not is_training_action(None)
    now = dt.datetime.now(dt.UTC)
    rows = [
        {
            "approval_id": "apr_x",
            "action": "train_run:0123456789abcdef",
            "status": "approved",
            "consumed_at": now.isoformat(),
        }
    ]
    assert find_run_approval(rows, now, approval_id="apr_x") is not None


# ── F1-5: aynı adla birden çok başlatma kaydı → en yenisi ───────────────────────────────


def test_f1_5_recipe_for_adapter_picks_newest_launch(iso, monkeypatch) -> None:  # noqa: F811
    from app.training import candidate_checks
    from app.training.easy_train import snapshots_dir

    for sid, at, appr in (
        ("snap_ffff000000000000", "2026-10-06T01:00:00+00:00", "apr_eski"),
        ("snap_0000ffff00000000", "2026-10-06T09:00:00+00:00", "apr_yeni"),
    ):
        d = snapshots_dir() / sid
        d.mkdir(parents=True)
        (d / "recipe.json").write_text(json.dumps({"adapter_name": "p1"}), "utf-8")
        (d / "launch.json").write_text(
            json.dumps({"ok": True, "approval_id": appr, "at": at}), "utf-8"
        )
    assert candidate_checks.recipe_for_adapter("p1")["approval_id"] == "apr_yeni"


# ── F1-6 / F1-7 / F1-8: kolay akış başlatma kayıtları ───────────────────────────────────


def test_f1_6_request_record_merges_with_current_file(iso) -> None:  # noqa: F811
    from app.training import easy_train

    easy_train._set_request("req-aaaaaaaa", {"status": "starting"})
    easy_train._set_request("req-bbbbbbbb", {"status": "started"})
    reqs = json.loads(easy_train.requests_path().read_text("utf-8"))
    assert set(reqs) == {"req-aaaaaaaa", "req-bbbbbbbb"}


@pytest.mark.parametrize("bad", ["x/snap_0123456789abcdef", "..\\..\\snap_0123456789abcdef", "s"])
def test_f1_7_snapshot_id_must_be_plain(iso, bad) -> None:  # noqa: F811
    from app.training import easy_train

    with pytest.raises(easy_train.EasyTrainError, match="anlık görüntü kimliği"):
        easy_train.launch(bad, "req-12345678")


def test_f1_8_failed_launch_revokes_weight_decision(iso, monkeypatch) -> None:  # noqa: F811
    from types import SimpleNamespace

    from app.lora.weight_decision import WeightDecisionStore
    from app.training import detached_launch, easy_train

    recipe = {
        "recipe_sha": "r" * 64,
        "adapter_name": "p1",
        "base_model": "b",
        "profile": "moe30b_attn_long",
        "max_examples": 64,
        "mix_weights": {"math": 1.0},
        "mix_label": "test",
    }
    monkeypatch.setattr(easy_train, "_snapshot", lambda sid: recipe)
    monkeypatch.setattr(easy_train, "precheck", lambda r: [])
    monkeypatch.setattr(easy_train, "summary", lambda r: "özet")
    monkeypatch.setattr(
        "app.agents.runtime.approvals.require_fresh_approval",
        lambda *a, **k: SimpleNamespace(authorized=True, approval_id="apr_t"),
    )
    monkeypatch.setattr(detached_launch, "launch", lambda **k: {"ok": False, "message": "RAM"})
    out = easy_train.launch("snap_0123456789abcdef", "req-12345678")
    assert out["status"] == "error"
    assert WeightDecisionStore().pending() is None  # açık karar kalmadı → sonraki koşu sorar
    reqs = json.loads(easy_train.requests_path().read_text("utf-8"))
    assert reqs["req-12345678"]["status"] == "error"


def test_weight_decision_revoke_only_open(iso) -> None:  # noqa: F811
    from app.lora.weight_decision import WeightDecisionStore

    st = WeightDecisionStore()
    d = st.record({"math": 1.0}, "t", "test")
    st.revoke(d.decision_id, "test")
    assert st.pending() is None
    row = next(r for r in st.records() if r.decision_id == d.decision_id)
    assert row.superseded_by.startswith("geri_cekildi")


# ── F3-1: tazelik tüm başlatma yollarının kapısında ─────────────────────────────────────


def test_f3_1_freshness_is_part_of_pretrain_gate(iso, monkeypatch) -> None:  # noqa: F811
    from app.config import get_settings
    from app.training import detached_launch as dl
    from app.training.sft_assembly import FreshnessResult

    s = get_settings()
    src = s.root / "data" / "lora_sft" / "lora_sft.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    row = {"messages": [{"role": "user", "content": "soru"}, {"role": "assistant", "content": "c"}]}
    src.write_text(json.dumps(row) + "\n", "utf-8")
    monkeypatch.setattr(
        "app.training.sft_assembly.check_assembly_freshness",
        lambda lines, settings: FreshnessResult(status="BAYAT", blockers=["tazelik: BAYAT (test)"]),
    )
    assert any("BAYAT" in b for b in dl._pretrain_gate_blockers(s))


# ── F4-3 / F4-4 / F4-9: otomatik puanlama ve set doğrulaması ────────────────────────────


@pytest.mark.parametrize(
    ("answer", "key", "ok"),
    [
        ("Hesap ...\n10,000", 10000, True),  # EN binlik
        ("Hesap ...\n10.000", 10000, True),  # TR binlik
        ("Sonuç: 50 adet (≈10 000 USD)", 10000, True),  # boşluklu binlik
        ("x = 1,234.56", 1234.56, True),
        ("x = 1.234,56", 1234.56, True),
        ("Sonuç −10", -10, True),  # Unicode eksi
        ("Sonuç −10", 10, False),  # işaret yanlış okunmaz
        ("0,16", 0.16, True),  # TR ondalık
        ("Cevap 0.16\n\n(maliyet 2 yönde)", 0.16, False),  # son satıra sayıdan sonra yazılan
        ("sonuç yok", 1, False),
    ],
)
def test_f4_3_auto_score_reads_final_number(answer, key, ok) -> None:
    from app.evals.candidate_compare import auto_score

    got = auto_score(
        {"question": "q", "type": "math", "answer_key": key, "tolerance": 1e-6}, answer
    )
    assert got is not None and (got["score"] == 4) is ok


def test_f4_4_rule1_scans_answer_not_question() -> None:
    from app.evals.candidate_compare import auto_score

    q = {"question": "Bu strateji garantili kazandırır mı?", "type": "open"}
    assert auto_score(q, "Hayır; bu bir hipotezdir, örneklem dışı test edilmeli.") is None


def test_f4_9_non_numeric_math_key_rejected_at_load(tmp_path) -> None:
    from app.evals.candidate_compare import CompareError, load_set

    p = tmp_path / "s.jsonl"
    row = {"id": "m", "family": "f", "type": "math", "question": "?", "answer_key": "on bir"}
    p.write_text(json.dumps(row, ensure_ascii=False) + "\n", "utf-8")
    with pytest.raises(CompareError, match="sayısal"):
        load_set(Path(p))
