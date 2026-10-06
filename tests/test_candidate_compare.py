"""Faz 2D — aday doğrulama ve karşılaştırma.

Kabul şartları: run_complete.json tek başına tamamlanma kanıtı değil; yarım dönüşüm başarı değil;
ölçüt adayın sonucundan ÖNCE kilitli; aynı istem/decoding ile aktif + aday + temel referans;
matematik doğrulanmış anahtarla, diğerleri kör incelemeyle (kimlik sızmaz); bootstrap aile
düzeyinde; az kanıt → yetersiz kanıt; adaya özgü kritik hata → kritik ret; gizli final seti
yeniden kullanılırsa geliştirme sayılır ve erişim kaydedilir; karar 2A'nın deposuna yazılır.
"""

from __future__ import annotations

import json

import httpx
import pytest
from tests.chat_learning_helpers import iso  # noqa: F401

from app.evals import candidate_compare as cc

DIG = {"aktif": "a" * 64, "aday": "c" * 64, "temel": "b" * 64}
VERIFIED = {"completion": {"ok": True}, "conversion": {"ok": True}}


def _set(tmp_path, n_fam=10, math_fams=6, per_fam=2):
    rows = []
    for f in range(n_fam):
        for k in range(per_fam):
            if f < math_fams:
                rows.append(
                    {
                        "id": f"q{f}_{k}",
                        "family": f"F{f}",
                        "type": "math",
                        "question": f"{f} + {k} kaç eder?",
                        "answer_key": f + k,
                    }
                )
            else:
                rows.append(
                    {
                        "id": f"q{f}_{k}",
                        "family": f"F{f}",
                        "type": "sourced",
                        "question": f"Kavram {f}-{k} nedir?",
                        "evidence": ["kanıt metni"],
                    }
                )
    p = tmp_path / "set.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    return p


def _ollama(wrong_for=None, digest_flip=False):
    state = {"calls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            d = dict(DIG)
            if digest_flip and state["calls"] > 0:
                d["aday"] = "e" * 64
            return httpx.Response(
                200, json={"models": [{"name": k, "digest": v} for k, v in d.items()]}
            )
        if req.url.path == "/api/show":
            return httpx.Response(200, json={"template": "t"})
        body = json.loads(req.content)
        state["calls"] += 1
        q = body["messages"][-1]["content"]
        if "kaç eder" in q:
            a, b = (int(x) for x in q.split(" kaç")[0].split(" + "))
            ans = a + b + (1 if body["model"] == wrong_for else 0)
            return httpx.Response(200, json={"message": {"content": f"Sonuç {ans}."}})
        return httpx.Response(200, json={"message": {"content": f"{body['model']} cevabı"}})

    return httpx.MockTransport(handler)


def _create(tmp_path, *, role="development", meta=None, **kw):
    lock = cc.lock_criteria()
    return cc.create(
        set_path=_set(tmp_path, **kw),
        role=role,
        active_tag="aktif",
        candidate_tag="aday",
        base_tag="temel",
        criteria_sha=lock["criteria_sha"],
        candidate_meta=meta if meta is not None else {**VERIFIED},
    )


def _review_all(cmp_id, score=3):
    packet = cc.blind_packet(cmp_id)
    reviews = {
        p["question_id"]: {a["label"]: {"score": score} for a in p["answers"]} for p in packet
    }
    return cc.submit_review(cmp_id, reviews, reviewer="insan")


def test_accept_path_records_decision_for_2a(iso, tmp_path) -> None:  # noqa: F811
    from app.evals.candidate_decisions import latest_decision

    m = _create(tmp_path)
    cc.generate(m["comparison_id"], transport=_ollama())
    packet = cc.blind_packet(m["comparison_id"])
    assert packet and all("role" not in a for p in packet for a in p["answers"])
    assert all(p["evidence"] for p in packet if p["type"] == "sourced")
    blob = json.dumps(packet, ensure_ascii=False)
    for leak in ("candidate", "active", '"base"', "digest", '"model"'):
        assert leak not in blob  # paket rol/model kimliği taşımıyor (cevap metni hariç)
    assert {k for p in packet for a in p["answers"] for k in a} == {"label", "answer"}
    _review_all(m["comparison_id"])
    res = cc.finalize(m["comparison_id"])
    assert res["decision"] == "kabul" and res["bootstrap"]["n_families"] == 10
    assert res["base_reference"] is not None and "trading performansının" in res["disclaimer"]
    dec = latest_decision("aday", "c" * 64)
    assert dec["decision"] == "kabul" and dec["comparison_id"] == m["comparison_id"]


def test_candidate_only_critical_error_is_critical_reject(iso, tmp_path) -> None:  # noqa: F811
    m = _create(tmp_path)
    cc.generate(m["comparison_id"], transport=_ollama(wrong_for="aday"))
    _review_all(m["comparison_id"])
    assert cc.finalize(m["comparison_id"])["decision"] == "kritik_ret"


def test_few_families_or_unverified_is_insufficient(iso, tmp_path) -> None:  # noqa: F811
    m = _create(tmp_path, n_fam=3, math_fams=2, per_fam=6)  # 18 soru ama 3 aile
    cc.generate(m["comparison_id"], transport=_ollama())
    _review_all(m["comparison_id"])
    res = cc.finalize(m["comparison_id"])
    assert res["decision"] == "yetersiz_kanit" and res["bootstrap"]["n_families"] == 3
    (tmp_path / "b").mkdir()
    m2 = _create(tmp_path / "b", meta={"completion": {"ok": False}, "conversion": {"ok": True}})
    cc.generate(m2["comparison_id"], transport=_ollama())
    _review_all(m2["comparison_id"])
    r2 = cc.finalize(m2["comparison_id"])
    assert r2["decision"] == "yetersiz_kanit" and any("doğrulaması" in x for x in r2["reasons"])


def test_worse_candidate_is_rejected(iso, tmp_path) -> None:  # noqa: F811
    m = _create(tmp_path, math_fams=0)
    cc.generate(m["comparison_id"], transport=_ollama())
    sealed = json.loads(
        (cc._root() / m["comparison_id"] / "sealed_mapping.json").read_text("utf-8")
    )
    reviews = {}
    for p in cc.blind_packet(m["comparison_id"]):
        roles = sealed[p["question_id"]]
        reviews[p["question_id"]] = {
            a["label"]: {
                "score": 1 if roles[a["label"]] == "aday" or roles[a["label"]] == "candidate" else 4
            }
            for a in p["answers"]
        }
    cc.submit_review(m["comparison_id"], reviews, reviewer="insan")
    assert cc.finalize(m["comparison_id"])["decision"] == "ret"


def test_criteria_lock_and_tamper(iso, tmp_path) -> None:  # noqa: F811
    lock = cc.lock_criteria()
    assert cc.lock_criteria()["locked_at"] == lock["locked_at"]  # değişmez
    p = cc._root() / "criteria" / f"{lock['criteria_sha'][:16]}.json"
    data = json.loads(p.read_text("utf-8"))
    data["criteria"]["margin"] = 5.0
    p.write_text(json.dumps(data), "utf-8")
    with pytest.raises(cc.CompareError, match="değişmiş"):
        cc.create(
            set_path=_set(tmp_path),
            role="development",
            active_tag="aktif",
            candidate_tag="aday",
            base_tag="temel",
            criteria_sha=lock["criteria_sha"],
            candidate_meta={},
        )


def test_final_set_reuse_becomes_development_and_is_logged(iso, tmp_path) -> None:  # noqa: F811
    a = _create(tmp_path, role="final")
    b = _create(tmp_path, role="final")
    assert a["role"] == "final" and b["role"] == "development" and "GELİŞTİRME" in b["role_note"]
    assert len([x for x in cc.final_accesses(a["set_sha"]) if x["kind"] == "use"]) == 2


def test_digest_change_invalidates_run(iso, tmp_path) -> None:  # noqa: F811
    m = _create(tmp_path)
    with pytest.raises(cc.CompareError, match="digest değişti"):
        cc.generate(m["comparison_id"], transport=_ollama(digest_flip=True))
    assert cc._manifest(m["comparison_id"])["status"] == "invalid"


def test_family_bootstrap_counts_families_not_questions() -> None:
    bs = cc.family_bootstrap({"F1": 0.5, "F2": -0.5}, B=500, seed=1, ci=0.95)
    assert bs["n_families"] == 2 and bs["unit"] == "family" and bs["lo"] <= bs["point"] <= bs["hi"]


# ── tamamlanma + dönüşüm doğrulaması ─────────────────────────────────────────


def _adapter_files(root, name="hektor_lora_t"):
    d = root / "models" / "adapters" / name
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-30B-A3B-Instruct-2507"}), "utf-8"
    )
    (d / "adapter_model.safetensors").write_bytes(b"agirlik")
    (d / "run_plan.json").write_text(json.dumps({"max_steps": 10}), "utf-8")
    (d / "run_complete.json").write_text(json.dumps({"global_step": 10, "max_steps": 10}), "utf-8")
    return d


def test_run_complete_alone_is_not_completion(iso) -> None:  # noqa: F811
    from app.config import get_settings
    from app.training.candidate_checks import verify_run_completion

    root = get_settings().root
    _adapter_files(root)
    res = verify_run_completion("hektor_lora_t")
    assert not res["ok"]  # süreç sonucu / veri / koşu kimliği yok
    keys = {c["key"]: c["ok"] for c in res["checks"]}
    assert keys["adim"] and keys["adapter_dosyalari"] and not keys["surec_sonucu"]
    from app.agents.runtime import approvals

    # Gerçek onay kaydı: istendi → insan onayladı → eğitim tüketti.
    req = approvals.require_fresh_approval("lora-trainer", "train_run", "critical", "test eğitimi")
    approvals.approve(req.approval_id)
    assert approvals.require_fresh_approval("lora-trainer", "train_run", "critical", "t").authorized
    (root / "storage").mkdir(exist_ok=True)

    def status(apr: str) -> None:
        (root / "storage" / "train_status.json").write_text(
            json.dumps(
                {
                    "adapter": "hektor_lora_t",
                    "finished_at": "x",
                    "pid": 0,
                    "data_sha256": "d" * 64,
                    "approval_id": apr,
                }
            ),
            "utf-8",
        )

    status(req.approval_id)
    recipe = {
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "data_sha256": "d" * 64,
        "approval_id": req.approval_id,
    }
    assert verify_run_completion("hektor_lora_t", recipe)["ok"]
    assert not verify_run_completion("hektor_lora_t", {**recipe, "data_sha256": "x"})["ok"]
    # Onay deposunda olmayan (uydurma) kimlik reçeteyle eşleşse bile tamamlanma sayılmaz.
    status("apr_1")
    fake = verify_run_completion("hektor_lora_t", {**recipe, "approval_id": "apr_1"})
    assert not fake["ok"]
    assert {c["key"]: c["ok"] for c in fake["checks"]}["onay_kaydi"] is False


def test_partial_conversion_is_not_success(iso) -> None:  # noqa: F811
    from app.config import get_settings
    from app.training.candidate_checks import sha256_file, verify_conversion

    root = get_settings().root
    d = _adapter_files(root)
    w = sha256_file(d / "adapter_model.safetensors")
    merged = root / "models" / "merged" / "hektor_lora_t"
    merged.mkdir(parents=True)
    (merged / "merge_info.json").write_text(
        json.dumps(
            {"adapter_sha256": w, "kl_peft_vs_merged": 0.001, "kl_gate": 0.01, "top_same": True}
        ),
        "utf-8",
    )
    g = root / "models" / "gguf"
    g.mkdir(parents=True)
    (g / "hektor_lora_t-Q4_K_M.gguf").write_bytes(b"x")
    (g / "hektor_lora_t-Q4_K_M.gguf.src").write_text(f"adapter:{w}|Q4_K_M|attn=q8_0", "utf-8")
    (g / "Modelfile.hektor-t").write_text(
        "# hektor-t - hektor_lora_t birlesik GGUF Q4_K_M; sablon x'dan.\n"
        f"FROM {g / 'hektor_lora_t-Q4_K_M.gguf'}\n",
        "utf-8",
    )
    tags = [{"name": "hektor-t:latest", "digest": "sha256:" + "f" * 64}]
    ok = verify_conversion("hektor_lora_t", "hektor-t", tags=tags)
    assert ok["ok"], ok["checks"]
    # Kademe 2 (2026-10-06) F4-10: FROM köken anahtarı OLMAYAN aynı-önekli GGUF'u gösterirse
    # ad öneki tutsa bile dönüşüm doğrulanmış sayılmaz.
    (g / "hektor_lora_t-Q8_0.gguf").write_bytes(b"eski")
    mf = g / "Modelfile.hektor-t"
    good_mf = mf.read_text("utf-8")
    mf.write_text(good_mf.replace("Q4_K_M.gguf", "Q8_0.gguf"), "utf-8")
    stray = verify_conversion("hektor_lora_t", "hektor-t", tags=tags)
    assert not {c["key"]: c["ok"] for c in stray["checks"]}["modelfile"]
    mf.write_text(good_mf, "utf-8")
    (g / "hektor_lora_t-bf16.gguf.partial").write_bytes(b"yarim")
    bad = verify_conversion("hektor_lora_t", "hektor-t", tags=tags)
    assert not bad["ok"] and "partial" in json.dumps(bad["checks"])


def test_web_blind_review_flow(iso, tmp_path) -> None:  # noqa: F811
    from fastapi.testclient import TestClient

    from app.web.security import require_human
    from app.web.server import app

    m = _create(tmp_path)
    cc.generate(m["comparison_id"], transport=_ollama())
    client = TestClient(app)
    cid = m["comparison_id"]
    assert client.get("/api/compare").json()["items"][0]["status"] == "generated"
    blind = client.get(f"/api/compare/{cid}/blind").json()["items"]
    assert "sealed" not in str(blind) and "candidate" not in str(blind)
    early = client.post(f"/api/compare/{cid}/finalize")
    assert early.status_code == 422  # inceleme olmadan karar yok
    partial = {blind[0]["question_id"]: {a["label"]: {"score": 3} for a in blind[0]["answers"]}}
    assert client.post(f"/api/compare/{cid}/review", json={"scores": partial}).status_code == 422
    full = {p["question_id"]: {a["label"]: {"score": 3} for a in p["answers"]} for p in blind}
    assert client.post(f"/api/compare/{cid}/review", json={"scores": full}).status_code == 200
    res = client.post(f"/api/compare/{cid}/finalize").json()
    assert res["decision"] == "kabul"
    for rt in client.app.routes:
        if getattr(rt, "path", "") in (
            "/api/compare/{cmp_id}/review",
            "/api/compare/{cmp_id}/finalize",
        ):
            assert require_human in {d.call for d in rt.dependant.dependencies}


def test_records_based_recipe_for_start_train_runs(iso) -> None:  # noqa: F811
    """Kolay akış kaydı yoksa (start-train/CLI) reçete bağımsız kayıtlardan kurulur."""
    import hashlib

    from app.agents.runtime import approvals
    from app.config import get_settings
    from app.training.candidate_checks import recipe_for_adapter, verify_run_completion

    root = get_settings().root
    _adapter_files(root)
    src = root / "data" / "lora_sft" / "lora_sft.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text('{"messages": []}\n', encoding="utf-8")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    req = approvals.require_fresh_approval("lora-trainer", "train_run", "critical", "t")
    approvals.approve(req.approval_id)
    approvals.require_fresh_approval("lora-trainer", "train_run", "critical", "t")
    (root / "storage").mkdir(exist_ok=True)
    (root / "storage" / "train_status.json").write_text(
        json.dumps(
            {
                "adapter": "hektor_lora_t",
                "finished_at": "x",
                "pid": 0,
                "data_sha256": sha,
                "approval_id": req.approval_id,
            }
        ),
        "utf-8",
    )
    reg = root / "registry" / "adapters" / "registry.jsonl"
    reg.parent.mkdir(parents=True)
    row = {"adapter_id": "a1", "adapter_name": "hektor_lora_t", "status": "candidate"}
    row["base_model"] = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    reg.write_text(json.dumps(row) + "\n", encoding="utf-8")
    recipe = recipe_for_adapter("hektor_lora_t")  # anlık görüntü klasörü YOK
    assert recipe is not None and recipe["data_sha256"] == sha and "kayıtlar" in recipe["source"]
    assert verify_run_completion("hektor_lora_t", recipe)["ok"]
    # Veri bugün farklıysa (yeniden özet tutmuyor) eğitim verisi doğrulanmış sayılmaz.
    src.write_text('{"messages": [1]}\n', encoding="utf-8")
    again = verify_run_completion("hektor_lora_t", recipe_for_adapter("hektor_lora_t"))
    assert not again["ok"]
    assert {c["key"]: c["ok"] for c in again["checks"]}["veri_ozeti"] is False


# ── AI incelemesi ayrı · yalnız entegrasyon testi · boyut raporu ────────────────


def _scores_for(cmp_id, score):
    return {
        p["question_id"]: {a["label"]: {"score": score} for a in p["answers"]}
        for p in cc.blind_packet(cmp_id)
    }


def test_ai_review_is_separate_and_never_decides(iso, tmp_path) -> None:  # noqa: F811
    m = _create(tmp_path)
    cid = m["comparison_id"]
    cc.generate(cid, transport=_ollama())
    with pytest.raises(cc.CompareError, match="model kimliği"):
        cc.submit_ai_review(cid, _scores_for(cid, 0), reviewer_model=" ", method="")
    rec = cc.submit_ai_review(cid, _scores_for(cid, 0), reviewer_model="ai-x", method="kör paket")
    assert rec["kind"] == "ai_review" and "İNSAN puanı DEĞİLDİR" in rec["note"]
    assert cc._manifest(cid)["status"] == "generated"  # insan incelemesi hâlâ gerekir
    with pytest.raises(cc.CompareError):
        cc.finalize(cid)  # AI puanı kararı açmaz
    _review_all(cid, score=3)
    res = cc.finalize(cid)
    assert res["decision"] == "kabul"  # AI'nin 0 puanları karara girmedi
    assert cc.ai_review(cid)["scores"]  # ayrı kayıt korunur


def test_integration_only_caps_decision_and_skips_registry(iso, tmp_path, monkeypatch) -> None:  # noqa: F811
    calls = []
    monkeypatch.setattr(cc, "_sync_registry", lambda *a: calls.append(a))
    m = _create(tmp_path, meta={**VERIFIED, "adapter_id": "adapter_x"})
    cid = m["comparison_id"]
    cc.generate(cid, transport=_ollama())
    cc.mark_integration_only(cid, "6 aile, entegrasyon koşusu")
    _review_all(cid, score=4)
    res = cc.finalize(cid)
    assert res["decision"] == "yetersiz_kanit" and res["purpose"] == "entegrasyon_testi"
    assert any("ENTEGRASYON" in r for r in res["reasons"]) and calls == []
    assert cc.list_comparisons()[0]["purpose"] == "entegrasyon_testi"


def test_dimension_report_and_key_summary_without_identity(iso, tmp_path) -> None:  # noqa: F811
    m = _create(tmp_path)
    cid = m["comparison_id"]
    cc.generate(cid, transport=_ollama(wrong_for="aday"))
    keys = cc.key_summary(cid)
    assert keys and all(k["dimension"] == "matematik" for k in keys)
    assert all(k["n_answers"] == 3 and k["n_matched"] == 2 for k in keys)
    blob = json.dumps(keys, ensure_ascii=False)
    assert "aday" not in blob and "candidate" not in blob  # model kimliği yok
    assert {p["dimension"] for p in cc.blind_packet(cid)} == {"kaynak"}
    _review_all(cid)
    dims = cc.finalize(cid)["dimensions"]
    assert dims["candidate"]["matematik"]["critical"] == len(keys)
    assert dims["active"]["matematik"]["critical"] == 0 and "kaynak" in dims["active"]


def test_dimension_criteria_chosen_from_set_shape(tmp_path) -> None:
    plain = _set(tmp_path)
    assert cc.criteria_for_set(plain)["name"] == "aday_karsilastirma_v1"
    dim = tmp_path / "dim.jsonl"
    dim.write_text(
        json.dumps({"id": "a", "family": "f", "question": "q", "dimension": "talimat"}) + "\n",
        encoding="utf-8",
    )
    crit = cc.criteria_for_set(dim)
    assert crit["name"] == "aday_karsilastirma_v2_boyutlu"
    assert set(crit["dimensions"]) == {"matematik", "kaynak", "talimat", "strateji"}


def test_web_ai_review_hidden_until_human_review(iso, tmp_path) -> None:  # noqa: F811
    from fastapi.testclient import TestClient

    from app.web.server import app

    m = _create(tmp_path)
    cid = m["comparison_id"]
    cc.generate(cid, transport=_ollama())
    client = TestClient(app)
    assert client.get(f"/api/compare/{cid}/ai-review").json()["available"] is False
    cc.submit_ai_review(cid, _scores_for(cid, 2), reviewer_model="ai-x", method="kör")
    hidden = client.get(f"/api/compare/{cid}/ai-review").json()
    assert hidden["hidden"] is True and "scores" not in hidden
    shown = client.get(f"/api/compare/{cid}/ai-review?reveal=true").json()
    assert shown["scores"] and shown["kind"] == "ai_review"
    assert client.get(f"/api/compare/{cid}/keys").json()["items"]
    # AI incelemesi web'den YAZILAMAZ (yalnız CLI); insan uç noktası ayrı.
    paths = {getattr(rt, "path", "") for rt in client.app.routes}
    assert not any("ai-review" in p and p.endswith("/submit") for p in paths)
