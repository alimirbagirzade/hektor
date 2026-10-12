"""Kaynak bilgi kalemleri: denetim, karantina, ayrı sürüm düzeltme, insan kararı, veri sürümü.

Çevrimdışı: kaynak araması, yerel hakem ve sızıntı denetimi enjekte edilir (Ollama yok).
"""

from __future__ import annotations

import json

import pytest

from app.research import knowledge_review as kr

TEACHER = "qwen3:30b-a3b-instruct-2507-q4_K_M"
CHUNK_TEXT = (
    "Sharpe orani ortalama fazla getirinin standart sapmaya bolunmesidir. Yillik hale "
    "getirmek icin gunluk oran karekok 252 ile carpilir. Komisyon ve slippage dusulmeden "
    "hesaplanan oran yaniltici olabilir."
)


def _line(question: str, answer: str, *, mode: str = "rag", teacher: str = TEACHER) -> str:
    ctx = ["paper_a_c0001", "paper_b_c0002"] if mode == "rag" else []
    user = question
    if mode == "rag":
        user = (
            "SOURCES / KAYNAKLAR:\n[paper_a:paper_a_c0001] — A\n"
            + CHUNK_TEXT
            + "\n\n---\n\n[paper_b:paper_b_c0002] — B\n"
            + CHUNK_TEXT
            + "\n\nQUESTION / SORU: "
            + question
        )
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": "Sen Hektor yerel AI asistanısın."},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "distilled": True,
                "mode": mode,
                "teacher": teacher,
                "question": question,
                "paper_id": "paper_a",
                "chunk_id": "paper_a_c0001",
                "context_chunk_ids": ctx,
            },
        },
        ensure_ascii=False,
    )


ANSWER = (
    "Sharpe oranı, ortalama fazla getirinin getirilerin standart sapmasına bölünmesiyle "
    "hesaplanır [paper_a:paper_a_c0001]. Günlük orandan yıllık orana geçerken karekök 252 "
    "ile çarpılır [paper_b:paper_b_c0002]. Komisyon ve slippage düşülmeden hesaplanan oran "
    "yanıltıcı olabilir; bu yüzden test noktası olarak maliyet sonrası net getiri ile "
    "out-of-sample dönemde yeniden ölçülmelidir. Bu bir hipotezdir, yatırım tavsiyesi "
    "değildir; sonuç ancak maliyetli backtest ve örneklem dışı doğrulama ile sınanabilir. "
    "Ayrıca risksiz faiz oranının seçimi ve getirilerin bağımsız olmadığı durumlar oranı "
    "etkiler; otokorelasyon varsa yıllıklandırma düzeltmesi gerekir."
)


ANSWER_PLAIN = ANSWER.replace(" [paper_a:paper_a_c0001]", "").replace(
    " [paper_b:paper_b_c0002]", ""
)


def _lookup(paper_id: str):
    if paper_id not in ("paper_a", "paper_b"):
        return None, {}
    info = {
        "paper_id": paper_id,
        "title": paper_id.upper(),
        "year": "2021",
        "source": "manual",
        "ingest_status": "ready_for_rag",
        "quality_score": 90.0,
    }
    return info, {f"{paper_id}_c000{1 if paper_id == 'paper_a' else 2}": CHUNK_TEXT}


def _staging(*lines: str):
    p = kr.staging_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def clean_review():
    import shutil

    shutil.rmtree(kr.review_dir(), ignore_errors=True)
    shutil.rmtree(kr.versions_dir(), ignore_errors=True)
    kr.clear_selection()
    yield
    shutil.rmtree(kr.review_dir(), ignore_errors=True)
    shutil.rmtree(kr.versions_dir(), ignore_errors=True)
    kr.clear_selection()


def _review(item, others=None, *, leak=False, canon=None):
    return kr.deterministic_review(
        item,
        others or [],
        lookup=_lookup,
        canonical_questions=canon or set(),
        leak_check=lambda q, ln: leak,
    )


def _judge(verdict: str):
    return lambda prompt, system: f"KARAR: {verdict}\n- gerekçe"


def _add(item_id, rec):
    item = kr.load_item(item_id)
    item["reviews"].append(rec)
    return kr.save_item(item)


def _imported(*lines):
    _staging(*lines)
    assert kr.import_staging()["new"] == len(lines)
    return [it["item_id"] for it in kr.list_items()]


def test_import_is_idempotent_and_preserves_source(clean_review):
    line = _line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER)
    (iid,) = _imported(line)
    assert kr.import_staging() == {"new": 0, "existing": 1, "unreadable": 0}
    item = kr.load_item(iid)
    assert item["original"]["line"] == line
    assert item["status"] == "incelemede"  # inceleme yokken onay/eğitim yok
    assert item["origin"]["teacher"] == TEACHER and item["origin"]["mode"] == "rag"


def test_criteria_recorded_with_model_and_date(clean_review):
    (iid,) = _imported(_line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER))
    rec = _review(kr.load_item(iid))
    assert set(kr.CRITERIA) <= set(rec["criteria"])
    assert rec["at"] and rec["version"] == kr.DET_VERSION and rec["revision_sha"]
    for c in rec["criteria"].values():
        assert c["result"] and c["reason"]
    assert "2021" in rec["criteria"]["guncellik"]["reason"]


def test_unverifiable_item_is_quarantined_even_if_judge_agrees(clean_review):
    (iid,) = _imported(_line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER))
    item = _add(iid, _review(kr.load_item(iid)))
    item = _add(iid, kr.judge_item(item, model="yerel", generate=_judge("tutarli")))
    # Uzun sözel cevap deterministik kontrolle tam kapsanamaz → karantina (hakem onay değil).
    assert item["status"] == "karantina"
    assert "kapsam_kismi" in item["codes"] and item["verification_class"] == ""


def test_plain_mode_without_source_is_quarantined(clean_review):
    (iid,) = _imported(_line("Volatilite nedir?", ANSWER_PLAIN, mode="plain"))
    item = _add(iid, _review(kr.load_item(iid)))
    assert item["status"] == "karantina" and "kaynaksiz" in item["codes"]


def test_judge_suspicion_and_missing_judge_quarantine(clean_review):
    (iid,) = _imported(_line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER))
    item = _add(iid, _review(kr.load_item(iid)))
    assert "hakem_yok" in item["codes"]
    item = _add(iid, kr.judge_item(item, model="yerel", generate=_judge("supheli")))
    assert item["status"] == "karantina" and "hakem_supheli" in item["codes"]
    judge = kr.latest_review(item, "yerel_hakem")
    assert judge["model"] == "yerel" and judge["prompt_sha256"] and judge["at"]


def test_judge_failure_is_not_verification(clean_review):
    (iid,) = _imported(_line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER))
    item = _add(iid, _review(kr.load_item(iid)))

    def boom(prompt, system):
        raise RuntimeError("Ollama yok")

    item = _add(iid, kr.judge_item(item, model="yerel", generate=boom))
    assert item["status"] == "karantina" and "hakem_yok" in item["codes"]


def test_leak_and_non_base_teacher_are_rejected_and_cannot_be_human_approved(clean_review):
    a, b = _imported(
        _line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER),
        _line("Drawdown nasıl hesaplanır?", ANSWER + " Ek.", teacher="hektor-v12-30b"),
    )
    leaked = _add(a, _review(kr.load_item(a), leak=True))
    assert leaked["status"] == "reddedildi" and "sizinti" in leaked["codes"]
    foreign = _add(b, _review(kr.load_item(b)))
    assert foreign["status"] == "reddedildi" and "koken" in foreign["codes"]
    with pytest.raises(kr.KnowledgeError, match="çürütülmüş"):
        kr.decide(a, "onay", "Bizzat okudum ve doğru buldum.", decided_by="ali")


def test_duplicate_against_canonical_training_data(clean_review):
    from app.training.self_distill import norm_question

    q = "Sharpe oranı nasıl yıllıklandırılır?"
    (iid,) = _imported(_line(q, ANSWER))
    item = _add(iid, _review(kr.load_item(iid), canon={norm_question(q)}))
    assert item["status"] == "tekrar"
    with pytest.raises(kr.KnowledgeError, match="Tekrar"):
        kr.decide(iid, "onay", "Bizzat okudum ve doğru buldum.", decided_by="ali")


def test_numeric_contradiction_between_similar_items(clean_review):
    q = "Günlük Sharpe oranı yıllık orana nasıl çevrilir?"
    a, b = _imported(
        _line(q, ANSWER + " Örnek: 0.05 günlük oran yıllık 0.79 olur."),
        _line(q + " Kısaca açıkla.", ANSWER + " Örnek: 0.05 günlük oran yıllık 1.25 olur."),
    )
    items = kr.list_items()
    for iid in (a, b):
        _add(iid, _review(kr.load_item(iid), items))
    second = kr.load_item(items[-1]["item_id"])
    assert "celiski" in second["codes"] and second["status"] == "karantina"


def test_human_approval_requires_reason_and_human_identity(clean_review):
    (iid,) = _imported(_line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER))
    item = _add(iid, _review(kr.load_item(iid)))
    _add(iid, kr.judge_item(item, model="yerel", generate=_judge("tutarli")))
    with pytest.raises(kr.KnowledgeError, match="gerekçesi"):
        kr.decide(iid, "onay", "kısa", decided_by="ali")
    for agent in ("codex", "driver", "otomatik", ""):
        with pytest.raises(kr.KnowledgeError):
            kr.decide(iid, "onay", "Bizzat okudum ve doğru buldum.", decided_by=agent)
    item = kr.decide(iid, "onay", "Kaynak parçalarıyla karşılaştırdım, doğru.", decided_by="ali")
    assert item["status"] == "onaylandi" and item["verification_class"] == "insan"
    assert item["decisions"][-1]["decided_by"] == "ali"


def test_gpt_comment_only_adds_suspicion(clean_review):
    (iid,) = _imported(_line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER))
    item = _add(iid, _review(kr.load_item(iid)))
    _add(iid, kr.judge_item(item, model="yerel", generate=_judge("tutarli")))
    kr.decide(iid, "onay", "Kaynak parçalarıyla karşılaştırdım, doğru.", decided_by="ali")
    out = (
        "inceleme\n"
        + kr.GPT_MARKER
        + json.dumps({"item_id": iid, "verdict": "supheli", "reason": "sayı uydurma"})
        + "\n"
        + kr.GPT_MARKER
        + json.dumps({"item_id": "kr_00000000000000000000", "verdict": "tutarli", "reason": "x"})
        + "\nbozuk "
        + kr.GPT_MARKER
        + "{"
    )
    got = kr.parse_gpt_output(out, {iid})
    assert list(got) == [iid]
    item = kr.record_gpt_review(iid, engine="codex", prompt_sha="p" * 64, **got[iid])
    gpt = kr.latest_review(item, "gpt")
    assert "abonelik CLI" in gpt["model"] and "kanıtı DEĞİLDİR" in gpt["note"]
    # İnsan onayı vardı; GPT şüphesi kayda geçer (kod listesinde görünür).
    assert "gpt_supheli" in item["codes"]
    # Onaysız kalemde GPT 'tutarli' hiçbir şeyi onaylamaz.
    _staging(_line("Drawdown nasıl ölçülür?", ANSWER + " Ek."))
    assert kr.import_staging()["new"] == 1
    other = next(x["item_id"] for x in kr.list_items() if x["item_id"] != iid)
    _add(other, _review(kr.load_item(other)))
    o = kr.record_gpt_review(
        other, engine="codex", verdict="tutarli", reason="ok", prompt_sha="p" * 64
    )
    assert o["status"] == "karantina"


def test_correction_is_separate_revision_and_resets_review(clean_review):
    (iid,) = _imported(_line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER))
    item = _add(iid, _review(kr.load_item(iid)))
    _add(iid, kr.judge_item(item, model="yerel", generate=_judge("tutarli")))
    kr.decide(iid, "onay", "Kaynak parçalarıyla karşılaştırdım, doğru.", decided_by="ali")
    fixed = ANSWER.replace("252", "252 (işlem günü)")
    item = kr.correct(iid, fixed, "İşlem günü sayısı açıkça belirtildi.", author="ali")
    assert item["original"]["line"].count("karekök 252 ile") == 1  # kaynak metin değişmedi
    assert [r["rev"] for r in item["revisions"]] == [0, 1]
    assert kr.current_answer(item) == fixed
    assert item["status"] == "incelemede"  # eski inceleme + insan onayı yeni sürüme taşınmaz
    line = json.loads(kr.training_line(item))
    assert line["metadata"]["corrected"] is True and line["metadata"]["research_revision"] == 1


def test_version_selection_feeds_assembly_and_blocks_after_change(clean_review):
    from app.config import get_settings
    from app.training.sft_assembly import assemble_sft_lines

    settings = get_settings()
    base = assemble_sft_lines(settings, discipline=False, chat=False)
    (iid,) = _imported(_line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER))
    item = _add(iid, _review(kr.load_item(iid)))
    _add(iid, kr.judge_item(item, model="yerel", generate=_judge("tutarli")))
    with pytest.raises(kr.KnowledgeError, match="Onaylı"):
        kr.create_version()  # karantinadaki kalemle sürüm olmaz
    kr.decide(iid, "onay", "Kaynak parçalarıyla karşılaştırdım, doğru.", decided_by="ali")
    m = kr.create_version("test")
    assert kr.create_version("test")["version_id"] == m["version_id"]  # idempotent
    # Seçim yokken birleştirme değişmez.
    assert assemble_sft_lines(settings, discipline=False, chat=False).lines == base.lines
    kr.select_version(m["version_id"])
    assert kr.selection_blockers() == []
    res = assemble_sft_lines(settings, discipline=False, chat=False)
    assert res.research and res.research["version_id"] == m["version_id"]
    assert res.research["selected"] == 1
    # Sonradan düzeltilen kalem → seçili sürüm geçersiz → eğitim kapısı durdurur.
    kr.correct(iid, ANSWER + " Düzeltme.", "Ek açıklama eklendi, tekrar incelenmeli.", author="ali")
    blockers = kr.selection_blockers()
    assert blockers and "sonradan düzeltildi" in blockers[0]
    from app.training.detached_launch import _pretrain_gate_blockers

    assert any("bilgi veri sürümü" in b for b in _pretrain_gate_blockers(settings))


def test_auto_approval_requires_full_coverage_and_judge(clean_review):
    (iid,) = _imported(_line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER))
    item = kr.load_item(iid)
    rec = _review(item)
    rec["codes"] = [c for c in rec["codes"] if c != "kapsam_kismi"]
    rec["full_coverage"] = True
    item = _add(iid, rec)
    assert item["status"] == "karantina" and item["codes"] == ["hakem_yok"]
    item = _add(iid, kr.judge_item(item, model="yerel", generate=_judge("tutarli")))
    assert item["status"] == "onaylandi" and item["verification_class"] == "otomatik"


def test_review_pending_runs_det_and_judge_within_limit(clean_review, monkeypatch):
    _imported(
        _line("Sharpe oranı nasıl yıllıklandırılır?", ANSWER),
        _line("Drawdown nasıl ölçülür?", ANSWER + " Ek."),
    )
    monkeypatch.setattr(kr, "_leaks", lambda q, ln: False)
    res = kr.review_pending(
        limit=1, judge_generate=_judge("supheli"), judge_model_name="yerel", lookup=_lookup
    )
    assert res["reviewed"] == 1 and res["judged"] == 1
    res = kr.review_pending(
        limit=5, judge_generate=_judge("supheli"), judge_model_name="yerel", lookup=_lookup
    )
    assert res["karantina"] == 2


def test_gpt_connection_reported_when_not_selected(clean_review):
    gpt = kr.gpt_connection()
    assert gpt["available"] is False and gpt["detail"]
