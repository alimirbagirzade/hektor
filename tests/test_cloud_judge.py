"""Tek tıklamalı döngü (Seçenek A) — K1/K2 bulut hakem, bekleyen kararlar, hazırla→karşılaştır
zinciri. Sahte sağlayıcıyla, AĞ ÇAĞRISI YOK (docs/TASARIM_SUREKLI_DONGU.md).

Kabul şartları: tıklamasız (önizlenen özetsiz) CLI çağrısı yapılamaz; K1 kör paketi parçalara
böler ve her parça ayrı tık ister; bulut puanı AYRI AI incelemesi olarak yazılır, kararı açmaz,
var olan AI incelemesini ezmez; gizli final seti buluta gitmez; K2 tek aday gönderir, adayın
kaydını değiştirmez, sızıntılı aday gitmez; kota ikinci görüşle ortaktır; bulut metni eğitim
kapısında NO-GO; bekleyen kararlar diskten türetilir (yeniden açılışta korunur) ve iş başlatmaz;
dönüşüm doğrulanınca karşılaştırma aynı kurallarla kendiliğinden başlar.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from tests.chat_learning_helpers import iso, send  # noqa: F401
from tests.test_candidate_compare import _create, _ollama, _review_all

from app.cloud import judge
from app.cloud import second_opinion as so
from app.cloud.providers import FakeProvider, ProviderReply
from app.evals import candidate_compare as cc

REPO = Path(__file__).resolve().parents[1]


def _enable(monkeypatch, *, daily_max=50, judge_max=None):
    from app.config import settings as settings_mod

    monkeypatch.setenv("HEKTOR_CLOUD_SECOND_OPINION", "1")
    monkeypatch.setenv("HEKTOR_CLOUD_PROVIDER", "fake")
    monkeypatch.setenv("HEKTOR_CLOUD_TERMS_ACK", "2026-10-10")
    monkeypatch.setenv("HEKTOR_CLOUD_DAILY_MAX", str(daily_max))
    if judge_max:
        monkeypatch.setenv("HEKTOR_CLOUD_JUDGE_MAX_CHARS", str(judge_max))
    settings_mod.get_settings.cache_clear()


class Scorer(FakeProvider):
    """İstemdeki her soruyu/etiketi bulup sabit puanlı JSON döndürür (ağsız)."""

    def __init__(self, score=1, junk=False) -> None:
        super().__init__()
        self.score, self.junk = score, junk

    def ask(self, prompt: str, *, timeout_s: float) -> ProviderReply:
        self.prompts.append(prompt)
        if self.junk:
            return ProviderReply(text="Puanlayamıyorum, üzgünüm.", model="fake-1")
        scores = {}
        for block in prompt.split("### SORU ")[1:]:
            qid = block.split()[0]
            labs = re.findall(r"^\[([A-C])\]$", block, re.MULTILINE)
            scores[qid] = {lab: {"score": self.score, "note": "sahte"} for lab in labs}
        return ProviderReply(
            text="```json\n" + json.dumps({"scores": scores}) + "\n```", model="f1"
        )


@pytest.fixture
def cmp_id(iso, tmp_path):  # noqa: F811
    m = _create(tmp_path)
    cc.generate(m["comparison_id"], transport=_ollama())
    return m["comparison_id"]


def _send_all_parts(cid: str) -> list[dict]:
    recs = []
    while True:
        pv = judge.k1_preview(cid)
        if pv["next"] is None:
            return recs
        nx = pv["next"]
        recs.append(judge.k1_start(cid, nx["index"], nx["payload_sha256"], wait=True))


# ── K1 ───────────────────────────────────────────────────────────────────────


def test_k1_disabled_by_default_and_needs_previewed_hash(cmp_id, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(so, "provider_factory", lambda n: calls.append(n) or Scorer())
    pv = judge.k1_preview(cmp_id)
    assert pv["enabled"] is False and pv["next"] is not None and calls == []
    with pytest.raises(judge.JudgeError, match="kapalı"):
        judge.k1_start(cmp_id, 0, pv["next"]["payload_sha256"])
    _enable(monkeypatch)
    with pytest.raises(judge.JudgeError, match="önizlemeden farklı"):
        judge.k1_start(cmp_id, 0, "0" * 64)
    with pytest.raises(judge.JudgeError, match="önizlemeden farklı"):
        judge.k1_start(cmp_id, 1, pv["next"]["payload_sha256"])  # sıradaki parça değil
    assert not list((judge._dir()).glob("jd_*.json"))  # hiçbir şey gönderilmedi


def test_k1_parts_each_need_own_click_then_separate_ai_review(cmp_id, monkeypatch) -> None:
    _enable(monkeypatch, judge_max=1500)
    fake = Scorer(score=0)
    monkeypatch.setattr(so, "provider_factory", lambda n: fake)
    pv = judge.k1_preview(cmp_id)
    assert pv["n_parts"] >= 2 and pv["enabled"]
    first = judge.k1_start(cmp_id, 0, pv["next"]["payload_sha256"], wait=True)
    assert fake.prompts == [pv["next"]["payload"]]  # tam olarak önizlenen metin gitti
    assert first["status"] == "done" and first["training_use"] is False
    assert cc.ai_review(cmp_id) is None  # tüm parçalar bitmeden yazılmaz
    assert judge.k1_preview(cmp_id)["next"]["index"] == 1  # sıradaki parça ayrı tık bekler
    _send_all_parts(cmp_id)
    assert len(fake.prompts) == pv["n_parts"]
    blob = "".join(fake.prompts)
    for leak in ("sealed", "digest", '"role"', "candidate_tag"):
        assert leak not in blob
    rec = cc.ai_review(cmp_id)
    assert rec and rec["method"].startswith("K1") and rec["reviewer_model"].startswith("bulut:")
    assert cc._manifest(cmp_id)["status"] == "generated"  # insan incelemesi hâlâ gerekir
    with pytest.raises(cc.CompareError):
        cc.finalize(cmp_id)
    _review_all(cmp_id, score=3)
    assert cc.finalize(cmp_id)["decision"] == "kabul"  # bulutun 0 puanları karara girmedi
    assert judge.k1_preview(cmp_id)["next"] is None  # yeniden gönderilecek parça yok


def test_k1_never_overwrites_existing_ai_review_and_never_sends_final_set(cmp_id, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(so, "provider_factory", lambda n: Scorer())
    packet = cc.blind_packet(cmp_id)
    scores = {p["question_id"]: {a["label"]: {"score": 2} for a in p["answers"]} for p in packet}
    cc.submit_ai_review(cmp_id, scores, reviewer_model="claude-oturum", method="elle")
    pv = judge.k1_preview(cmp_id)
    assert not pv["enabled"] and any("ezilmez" in b for b in pv["blockers"])
    (cc._root() / cmp_id / "ai_review.json").unlink()
    m = cc._manifest(cmp_id)
    m["role_requested"] = "final"
    cc._write(cc._root() / cmp_id / "manifest.json", m)
    pv = judge.k1_preview(cmp_id)
    assert not pv["enabled"] and any("final" in b for b in pv["blockers"])


def test_k1_unparseable_reply_fails_keeps_raw_text_and_allows_retry(cmp_id, monkeypatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setattr(so, "provider_factory", lambda n: Scorer(junk=True))
    nx = judge.k1_preview(cmp_id)["next"]
    rec = judge.k1_start(cmp_id, nx["index"], nx["payload_sha256"], wait=True)
    assert rec["status"] == "failed" and "ayrıştırılamadı" in rec["error"]
    assert "Puanlayamıyorum" in rec["text"] and rec["scores"] is None
    assert cc.ai_review(cmp_id) is None
    assert judge.k1_preview(cmp_id)["next"]["index"] == nx["index"]  # yeni tıkla tekrar


def test_k1_parse_rejects_missing_labels_and_bad_scores() -> None:
    labels = {"q1": ["A", "B"]}
    with pytest.raises(judge.JudgeError, match="tüm etiketler"):
        judge._k1_parse('{"scores": {"q1": {"A": {"score": 2}}}}', labels)
    with pytest.raises(judge.JudgeError, match="0–4"):
        judge._k1_parse('{"scores": {"q1": {"A": {"score": 7}, "B": {"score": 1}}}}', labels)
    with pytest.raises(judge.JudgeError, match="0–4"):
        judge._k1_parse('{"scores": {"q1": {"A": {"score": true}, "B": {"score": 1}}}}', labels)
    ok = judge._k1_parse('{"scores": {"q1": {"A": {"score": 4}, "B": {"score": 0}}}}', labels)
    assert ok["q1"]["A"] == {"score": 4, "critical": False, "note": ""}


# ── K2 ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def candidate(iso):  # noqa: F811
    from app.feedback.chat_store import ChatStore
    from app.feedback.learning import LearningService

    store = ChatStore()
    conv = store.create_conversation("TEST")["conversation_id"]
    turn = send(store, conv, "Volatilite kümelenmesi nedir?")
    cand, _ = LearningService(store).learn(turn["turn_id"])
    return cand


def test_k2_single_candidate_report_only_never_changes_candidate(candidate, monkeypatch) -> None:
    from app.feedback.chat_store import ChatStore

    _enable(monkeypatch)
    fake = FakeProvider("KARAR: şüpheli\n- kaynak gösterilmemiş")
    monkeypatch.setattr(so, "provider_factory", lambda n: fake)
    cid = candidate["candidate_id"]
    before = ChatStore().get_candidate(cid)
    pv = judge.k2_preview(cid)
    assert pv["enabled"] and candidate["target_text"] in pv["payload"]
    assert "varyansı zamanla değişir" not in pv["payload"]  # kaynak parçası gitmez
    with pytest.raises(judge.JudgeError, match="önizlemeden farklı"):
        judge.k2_start(cid, "0" * 64)
    rec = judge.k2_start(cid, pv["payload_sha256"], wait=True)
    assert fake.prompts == [pv["payload"]] and rec["verdict"] == "supheli"
    assert ChatStore().get_candidate(cid) == before  # durum/hedef/doğrulama değişmedi
    assert judge.k2_latest([cid])[cid]["verdict"] == "supheli"


def test_k2_blocks_leak_candidate_and_shares_daily_quota(candidate, monkeypatch) -> None:
    from app.feedback.chat_store import ChatStore

    _enable(monkeypatch, daily_max=1)
    monkeypatch.setattr(so, "provider_factory", lambda n: FakeProvider("KARAR: tutarli"))
    cid = candidate["candidate_id"]
    pv = judge.k2_preview(cid)
    assert judge.k2_start(cid, pv["payload_sha256"], wait=True)["verdict"] == "tutarli"
    st = so.status()  # ikinci görüş de aynı kotayı görür
    assert not st["enabled"] and any("Günlük üst sınır" in b for b in st["blockers"])
    _enable(monkeypatch, daily_max=10)
    ChatStore().update_candidate(cid, status="leak")
    pv = judge.k2_preview(cid)
    assert not pv["enabled"] and any("sızıntı" in b for b in pv["blockers"])


def test_k2_verdict_parsing() -> None:
    assert judge._k2_verdict("KARAR: tutarlı\nx") == "tutarli"
    assert judge._k2_verdict("karar : Supheli") == "supheli"
    assert judge._k2_verdict("Bence iyi.") == "belirsiz"


# ── eğitime giremez + web ────────────────────────────────────────────────────


def test_training_code_never_imports_judge_and_cloud_rows_are_no_go() -> None:
    from app.cloud.policy import cloud_origin_lines

    offenders = []
    for sub in ("app/feedback", "app/lora", "app/training", "scripts"):
        for p in (REPO / sub).rglob("*.py"):
            if "app.cloud.judge" in p.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(p.relative_to(REPO)))
    assert offenders == []
    row = {"messages": [], "metadata": {"origin": "cloud_judge_k2"}}
    assert cloud_origin_lines([json.dumps(row)]) == [0]


def test_judge_web_routes_require_human(cmp_id, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app.web.security import require_human
    from app.web.server import app

    _enable(monkeypatch)
    monkeypatch.setattr(so, "provider_factory", lambda n: Scorer())
    client = TestClient(app)
    pv = client.get(f"/api/cloud/judge/k1/{cmp_id}/preview").json()
    nx = pv["next"]
    r = client.post(
        f"/api/cloud/judge/k1/{cmp_id}",
        json={"part_index": nx["index"], "payload_sha256": nx["payload_sha256"]},
    )
    assert r.status_code == 200 and r.json()["id"].startswith("jd_")
    bad = client.post(
        f"/api/cloud/judge/k1/{cmp_id}", json={"part_index": 0, "payload_sha256": "0" * 64}
    )
    assert bad.status_code == 422
    posts = [
        rt
        for rt in client.app.routes
        if getattr(rt, "path", "").startswith("/api/cloud/judge")
        and "POST" in getattr(rt, "methods", set())
    ]
    assert len(posts) == 3
    for rt in posts:
        assert require_human in {d.call for d in rt.dependant.dependencies}


# ── bekleyen kararlar (loop_state) ───────────────────────────────────────────


def test_pending_decisions_derived_from_disk_and_start_nothing(cmp_id, monkeypatch) -> None:
    from app.config import settings as settings_mod
    from app.orchestration import loop_state
    from app.training import candidate_jobs as cj
    from app.training import easy_train

    def boom(*a, **k):
        raise AssertionError("bekleyen kararlar iş başlatmamalı")

    monkeypatch.setattr(cj, "start_job", boom)
    monkeypatch.setattr(
        easy_train,
        "readiness",
        lambda: {
            "items": [
                {"key": "veri", "ok": True},
                {"key": "kademe2", "ok": False, "detail": "yok"},
            ],
            "data_sha256": "d" * 64,
        },
    )
    _enable(monkeypatch)
    d = loop_state.pending_decisions()
    keys = {i["key"] for i in d["items"]}
    assert f"kor:{cmp_id}" in keys and "kademe2" in keys and d["errors"] == []
    kor = next(i for i in d["items"] if i["key"] == f"kor:{cmp_id}")
    assert kor["k1_available"] and kor["who"] == "insan"
    settings_mod.get_settings.cache_clear()  # "yeniden açılış": bellek durumu yok
    assert {i["key"] for i in loop_state.pending_decisions()["items"]} == keys
    _review_all(cmp_id)
    keys2 = {i["key"] for i in loop_state.pending_decisions()["items"]}
    assert f"karar:{cmp_id}" in keys2 and f"kor:{cmp_id}" not in keys2


def test_pending_decisions_section_error_does_not_hide_others(cmp_id, monkeypatch) -> None:
    from app.orchestration import loop_state
    from app.training import easy_train

    monkeypatch.setattr(easy_train, "readiness", lambda: (_ for _ in ()).throw(OSError("git")))
    d = loop_state.pending_decisions()
    assert any(e.startswith("egitim:") for e in d["errors"])
    assert any(i["key"] == f"kor:{cmp_id}" for i in d["items"])
    assert d["cloud_enabled"] is False


# ── hazırla → karşılaştır zinciri ────────────────────────────────────────────


def test_conversion_chains_comparison_with_same_rules(iso, monkeypatch, tmp_path) -> None:  # noqa: F811
    from tests.test_candidate_jobs import _inproc, _py, _start, _wait

    from app.training import candidate_jobs as cj

    root = tmp_path
    ad = root / "models" / "adapters" / "hektor_lora_t"
    ad.mkdir(parents=True)
    (ad / "adapter_config.json").write_text("{}", encoding="utf-8")
    tags = [
        {"name": "aktif:latest", "digest": "a" * 64},
        {"name": "temel:latest", "digest": "b" * 64},
        {"name": "hektor-base-30b-q4a8:latest", "digest": "b" * 64},  # varsayılan temel
    ]
    monkeypatch.setattr("app.feedback.model_identity.ollama_tags", lambda *a, **k: tags)
    monkeypatch.setattr(
        cj, "capability", lambda kind, adapter="": {"kind": kind, "supported": True, "reasons": []}
    )
    monkeypatch.setattr(
        "app.feedback.model_activation.resolve_chat_tag", lambda slot="main", root=None: "aktif"
    )
    monkeypatch.setattr(cj, "_spawn_runner", _inproc)

    def verified(adapter, tag):
        tags.append({"name": f"{tag}:latest", "digest": "c" * 64})  # dönüşüm etiketi yarattı
        return {"ok": True, "checks": [], "digest": "c" * 64}

    monkeypatch.setattr("app.training.candidate_checks.verify_conversion", verified)
    qset = tmp_path / "set.jsonl"
    qset.write_text("{}\n", encoding="utf-8")
    j = _start(
        monkeypatch,
        _py("print('5/5 ollama create')"),
        inproc=True,
        then_compare=True,
        question_set=str(qset),
    )
    done = _wait(j["job_id"])
    assert done["status"] == "done" and done["params"]["then_compare"] is True
    chain = done["chain"]
    assert chain["job_id"] and chain["error"] == "", chain
    nxt = cj.get_job(chain["job_id"])
    assert nxt["kind"] == "comparison" and nxt["request_id"] == f"chain-{j['job_id']}"
    assert nxt["params"]["question_set"] == str(qset)
    _wait(nxt["job_id"])  # karşılaştırma işi bitsin (doğrulaması sahte günlükte başarısız)


def test_chain_failure_is_recorded_and_conversion_stays_done(iso, monkeypatch) -> None:  # noqa: F811
    from tests.test_candidate_jobs import _inproc, _py, _start, _wait

    from app.config import get_settings
    from app.training import candidate_jobs as cj

    ad = get_settings().root / "models" / "adapters" / "hektor_lora_t"
    ad.mkdir(parents=True)
    (ad / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("app.feedback.model_identity.ollama_tags", lambda *a, **k: [])
    monkeypatch.setattr(
        cj, "capability", lambda kind, adapter="": {"kind": kind, "supported": True, "reasons": []}
    )
    monkeypatch.setattr(
        "app.feedback.model_activation.resolve_chat_tag", lambda slot="main", root=None: "aktif"
    )
    monkeypatch.setattr(cj, "_spawn_runner", _inproc)
    monkeypatch.setattr(
        "app.training.candidate_checks.verify_conversion",
        lambda a, t: {"ok": True, "checks": [], "digest": "c" * 64},
    )
    j = _start(monkeypatch, _py("print('ok')"), inproc=True, then_compare=True)
    done = _wait(j["job_id"])
    assert done["status"] == "done" and done["chain"]["job_id"] == ""
    assert done["chain"]["error"]  # ör. soru seti / etiket yok → elle başlatılır


def test_pending_decisions_slow_section_times_out_others_returned(cmp_id, monkeypatch) -> None:
    import threading

    from app.orchestration import loop_state
    from app.training import easy_train

    gate = threading.Event()
    monkeypatch.setattr(easy_train, "readiness", lambda: gate.wait(5) and {"items": []})
    d = loop_state.pending_decisions(section_timeout_s=0.5)
    gate.set()
    assert any(e.startswith("egitim:") and "zaman aşımı" in e for e in d["errors"])
    assert any(i["key"] == f"kor:{cmp_id}" for i in d["items"])  # diğer bölümler yine geldi
