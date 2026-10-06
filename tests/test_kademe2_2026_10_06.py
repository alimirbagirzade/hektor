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


# ── F1-9: açık profil seçimi, kayıtlı eski özel ağırlıklarla ezilmez ───────────────────


def test_f1_9_explicit_mix_profile_overrides_saved_custom_weights(iso, monkeypatch) -> None:  # noqa: F811
    from app.training import easy_train

    saved = {**easy_train.default_settings(), "mix_weights": {"math": 1.0}}
    monkeypatch.setattr(easy_train, "last_settings", lambda: saved)
    # Arayüz isteği: yalnız mix_profile (mix_weights anahtarı YOK).
    norm = easy_train._normalize({"adapter_name": "p1", "mix_profile": "trading_analysis_v1"})
    assert norm["mix_label"] == "trading_analysis_v1"
    assert norm["mix_weights_resolved"] != {"math": 1.0}
    # Açıkça özel ağırlık gönderen istemci (CLI/API) hâlâ özel ağırlığı alır.
    norm = easy_train._normalize(
        {"adapter_name": "p1", "mix_profile": "trading_analysis_v1", "mix_weights": {"math": 1.0}}
    )
    assert norm["mix_label"] == "özel"


# ── F4-5: final setinin kimliği içerikten; biçim/kimlik değişikliği yeniden kullanımı gizlemez ─


def _final_create(set_path):
    from app.evals import candidate_compare as cc

    lock = cc.lock_criteria()
    import time

    time.sleep(0.01)  # ölçüt karşılaştırmadan ÖNCE kilitli olmalı
    return cc.create(
        set_path=set_path,
        role="final",
        active_tag="aktif",
        candidate_tag="aday",
        base_tag="temel",
        criteria_sha=lock["criteria_sha"],
        candidate_meta={},
    )


def test_f4_5_final_set_reuse_survives_crlf_reorder_and_id_rename(iso, tmp_path) -> None:  # noqa: F811
    rows = [
        {"id": f"q{i}", "family": f"F{i % 4}", "type": "sourced", "question": f"Kavram {i} nedir?"}
        for i in range(8)
    ]
    a = tmp_path / "a.jsonl"
    a.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    first = _final_create(a)
    assert first["role"] == "final"
    # Aynı içerik: CRLF + ters sıra + anahtar sırası farklı → AYNI set özeti.
    b = tmp_path / "b.jsonl"
    b.write_bytes(
        "\r\n".join(
            json.dumps(dict(reversed(list(r.items()))), ensure_ascii=False) for r in reversed(rows)
        ).encode("utf-8")
    )
    second = _final_create(b)
    assert second["set_sha"] == first["set_sha"] and second["role"] == "development"
    # Kimlikleri değiştirilmiş + büyük harfli kopya → farklı özet, ama soru örtüşmesi yakalanır.
    renamed = [{**r, "id": "x" + r["id"], "question": r["question"].upper()} for r in rows[:3]]
    renamed += [
        {"id": "yeni1", "family": "F9", "type": "sourced", "question": "Tamamen yeni soru?"}
    ]
    c = tmp_path / "c.jsonl"
    c.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in renamed) + "\n", "utf-8")
    third = _final_create(c)
    assert third["set_sha"] != first["set_sha"]
    assert third["role"] == "development" and "3/4" in third["role_note"]
    # Gerçekten yeni bir set final kalır.
    d = tmp_path / "d.jsonl"
    d.write_text(
        json.dumps({"id": "z", "family": "F1", "type": "sourced", "question": "Başka?"}) + "\n",
        "utf-8",
    )
    assert _final_create(d)["role"] == "final"


# ── Kayıt kapısı: gerçek kanıt dosyasıyla bugün ne olur? (açık madde → kayıt YOK) ───────

_EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "kademe2_2026-10-06.json"


def _k2_cli(iso, monkeypatch, path: Path, *extra: str):  # noqa: F811
    from typer.testing import CliRunner

    from app.main import app
    from app.training import easy_train

    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.setattr(
        easy_train, "code_state_provider", lambda: {"ok": True, "code_sha": "c" * 64, "head": "h"}
    )
    return CliRunner().invoke(
        app,
        [
            "kademe2-kayit",
            "--findings",
            str(path),
            "--data-sha",
            "d" * 64,
            "--evidence",
            "make ci yeşil; PR #37 commitleri",
            *extra,
        ],
    )


def test_real_evidence_file_with_open_findings_is_refused(iso, monkeypatch) -> None:  # noqa: F811
    from app.training import easy_train

    data = json.loads(_EVIDENCE.read_text("utf-8"))
    open_ids = {b["id"] for b in data["bulgular"] if b["status"] not in easy_train.CLOSED_FINDING}
    res = _k2_cli(iso, monkeypatch, _EVIDENCE)
    if not open_ids:  # tüm maddeler bir gün kapanırsa kayıt yazılabilir — bu da doğru davranış
        assert res.exit_code == 0, res.output
        return
    assert res.exit_code == 1, res.output
    assert "Kapanmamış" in res.output
    for fid in open_ids:
        assert fid in res.output  # hangi maddenin engellediği açıkça yazılır
    assert easy_train.list_kademe2() == []  # kayıt YAZILMADI → eğitim kapısı kapalı kalır


def test_malformed_or_empty_findings_do_not_open_gate(iso, monkeypatch, tmp_path) -> None:  # noqa: F811
    from app.training import easy_train

    other = tmp_path / "x.json"
    other.write_text(json.dumps({"bulgu": "yanlış anahtar"}), "utf-8")
    assert _k2_cli(iso, monkeypatch, other).exit_code == 1  # eskiden [] → "hepsi kapalı"
    empty = tmp_path / "e.json"
    empty.write_text("[]", "utf-8")
    assert _k2_cli(iso, monkeypatch, empty).exit_code == 1
    assert _k2_cli(iso, monkeypatch, empty, "--temiz-av").exit_code == 0  # bilinçli bulgusuz av
    bare = tmp_path / "r.json"
    bare.write_text(json.dumps([{"id": "F9", "status": "risk_kabul", "kapsam": "kayit"}]), "utf-8")
    res = _k2_cli(iso, monkeypatch, bare)
    assert res.exit_code == 1 and "gerekçesiz" in res.output
    assert len(easy_train.list_kademe2()) == 1  # yalnız --temiz-av kaydı


# ── c0d6aea avı L-1 · L-3 ───────────────────────────────────────────────────────────────


def test_l1_risk_scope_must_be_known(iso, monkeypatch, tmp_path) -> None:  # noqa: F811
    from app.training import easy_train

    for kapsam in ("yalnız_reçete", None, "pilot"):
        row = {"id": "F3-4", "status": "risk_kabul", "gerekce": "x" * 30}
        if kapsam is not None:
            row["kapsam"] = kapsam
        f = tmp_path / "k.json"
        f.write_text(json.dumps([row], ensure_ascii=False), "utf-8")
        res = _k2_cli(iso, monkeypatch, f)
        assert res.exit_code == 1 and "kapsamı tanımsız" in res.output, res.output
    assert easy_train.list_kademe2() == []


def test_l3_launch_waits_for_gates_marker_or_exit(tmp_path) -> None:
    import time

    from app.training import detached_launch as dl

    class Proc:
        def __init__(self, rc=None):
            self.rc = rc

        def poll(self):
            return self.rc

    marker = tmp_path / "gates.ok"
    t0 = time.monotonic()
    assert dl._early_exit_code(Proc(rc=10), 30, marker) == 10  # kapıda düştü → başlamadı
    marker.write_text("1", "utf-8")
    assert dl._early_exit_code(Proc(), 30, marker) is None  # kapılar geçti → başlatıldı
    assert time.monotonic() - t0 < 5  # işaret/çıkış beklemeyi hemen bitirir
    assert dl._EARLY_EXIT_WAIT_S >= 60  # 8 sn penceresi geri gelmesin


# ── c0d6aea avı C-2 · C-4 · C-7 ─────────────────────────────────────────────────────────


def test_c2_ollama_cloud_tags_are_cloud_origin() -> None:
    from app.cloud.policy import cloud_origin_lines

    rows = [
        {"metadata": {"teacher": "deepseek-v3.1:671b-cloud"}},
        {"metadata": {"source": "chat", "model_tag": "gpt-oss:120b-cloud"}},
        {"metadata": {"teacher": "kimi-k2:1t-cloud:latest"}},
        {"metadata": {"teacher": "qwen3:30b-a3b-instruct-2507-q4_K_M"}},  # yerel → temiz
        {"metadata": {"source": "chat", "model_tag": "hektor-v12-30b"}},  # yerel → temiz
    ]
    assert cloud_origin_lines([json.dumps(r) for r in rows]) == [0, 1, 2]


def test_c4_cloud_cli_child_env_has_no_api_key(monkeypatch) -> None:
    from app.cloud import providers

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-degil")
    monkeypatch.setenv("HEKTOR_API_TOKEN", "insan-sirri")
    env = providers._child_env()
    assert "ANTHROPIC_API_KEY" not in env and "HEKTOR_API_TOKEN" not in env
    assert env.get("PATH")  # geri kalan ortam korunur


def test_c7_exclude_endpoint_requires_human() -> None:
    from app.web.chat_routes import chat_router
    from app.web.security import require_human

    route = next(r for r in chat_router.routes if getattr(r, "path", "").endswith("/exclude"))
    assert require_human in {d.call for d in route.dependant.dependencies}


# ── c0d6aea avı D-3 · D-4 ───────────────────────────────────────────────────────────────


def test_d3_summary_shows_effective_examples_and_seed() -> None:
    from app.training.easy_train import summary

    text = summary(
        {
            "adapter_name": "p",
            "base_model": "b",
            "profile": "moe30b_attn_long",
            "mix_label": "x",
            "mix_weights": {"math": 1.0},
            "n_train": 1491,
            "n_valid": 78,
            "max_examples": 64,
            "data_sha256": "d" * 64,
            "recipe_sha": "r" * 64,
        }
    )
    assert "eğitilecek 64 örnek" in text and "seed 42" in text


def test_d4_leakage_extractor_sees_all_trained_formats() -> None:
    from app.evals.profile.leakage import extract_train_texts

    for row in (
        {"prompt": "p", "completion": "GIZLI-CEVAP"},
        {"text": "GIZLI-CEVAP"},
        {"user": "u", "assistant": "GIZLI-CEVAP"},
    ):
        assert any("GIZLI-CEVAP" in t for t in extract_train_texts(row, 0).texts)


def test_p5_pilot_name_reused_with_full_recipe_still_blocks(iso, monkeypatch) -> None:  # noqa: F811
    """Kademe 2 P-5: aynı adapter adı sonradan tam reçeteyle kullanılsa da eski pilot engelli."""
    from app.feedback import model_activation as ma
    from app.training import candidate_checks
    from app.training.easy_train import snapshots_dir

    for sid, at, sha in (
        ("snap_ffff000000000000", "2026-10-06T01:00:00+00:00", "p" * 64),  # pilot
        ("snap_0000ffff00000000", "2026-10-06T09:00:00+00:00", "f" * 64),  # tam, daha yeni
    ):
        d = snapshots_dir() / sid
        d.mkdir(parents=True)
        (d / "recipe.json").write_text(
            json.dumps({"adapter_name": "p1", "recipe_sha": sha}), "utf-8"
        )
        (d / "launch.json").write_text(json.dumps({"ok": True, "at": at}), "utf-8")
    assert candidate_checks.recipe_for_adapter("p1")["recipe_sha"] == "f" * 64
    assert set(candidate_checks.recipe_shas_for_adapter("p1")) == {"p" * 64, "f" * 64}
    monkeypatch.setattr(
        "app.training.easy_train.recipe_has_limited_acceptance", lambda sha: sha == "p" * 64
    )
    assert "pilot" in ma.pilot_block("x-aday", adapter_name="p1")
