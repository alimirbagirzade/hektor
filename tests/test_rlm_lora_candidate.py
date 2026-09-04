"""RLM → LoRA aday seçimi testleri (talimat §16, salt-okuma, offline).

İzole DB (tmp_path) kullanır → diğer testlerin RLM koşularıyla karışmaz.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rlm.lora_candidate import (
    export_candidates_jsonl,
    select_lora_candidates,
)
from app.rlm.rlm_store import RlmStore


def _make_run(
    store: RlmStore,
    *,
    status: str,
    conf: float,
    cit: float,
    gnd: float,
    unsupported: list[str],
) -> str:
    run_id = store.create_run("Bu makalenin bulgusu nedir?", "general_paper_question", "stub")
    store.finish_run(
        run_id,
        status=status,
        final_answer="Kaynaklı cevap [p:c0].",
        final_confidence=conf,
        evidence_score=85.0,
    )
    store.set_verification(
        run_id,
        supported_claims=["desteklenen iddia"],
        unsupported_claims=unsupported,
        contradictions=[],
        citation_score=cit,
        grounding_score=gnd,
        context_sufficiency_score=0.9,
        final_decision=status,
    )
    return run_id


def test_select_applies_section16_thresholds(tmp_path: Path):
    store = RlmStore(db_path=tmp_path / "rlm.db")
    good = _make_run(store, status="answered", conf=0.90, cit=0.95, gnd=0.95, unsupported=[])
    _make_run(
        store, status="answered", conf=0.50, cit=0.95, gnd=0.95, unsupported=[]
    )  # düşük güven
    _make_run(
        store, status="answered", conf=0.90, cit=0.50, gnd=0.95, unsupported=[]
    )  # düşük citation
    _make_run(
        store, status="answered", conf=0.90, cit=0.95, gnd=0.50, unsupported=[]
    )  # düşük grounding
    _make_run(
        store, status="answered", conf=0.90, cit=0.95, gnd=0.95, unsupported=["y"]
    )  # desteksiz var
    _make_run(
        store, status="abstained", conf=0.90, cit=0.95, gnd=0.95, unsupported=[]
    )  # yanlış status

    cands = select_lora_candidates(store=store)

    assert [c.run_id for c in cands] == [good]  # yalnız §16'yı geçen
    assert cands[0].requires_human_approval is True  # MUTLAK — onaysız eğitim YOK
    assert cands[0].citation_score >= 0.90 and cands[0].grounding_score >= 0.90


def test_answered_with_limitation_also_eligible(tmp_path: Path):
    store = RlmStore(db_path=tmp_path / "rlm.db")
    rid = _make_run(
        store, status="answered_with_limitation", conf=0.88, cit=0.92, gnd=0.91, unsupported=[]
    )
    cands = select_lora_candidates(store=store)
    assert [c.run_id for c in cands] == [rid]


def test_export_writes_jsonl_with_approval_flag(tmp_path: Path):
    store = RlmStore(db_path=tmp_path / "rlm.db")
    _make_run(store, status="answered", conf=0.95, cit=0.95, gnd=0.95, unsupported=[])
    cands = select_lora_candidates(store=store)
    out = tmp_path / "candidates.jsonl"
    n = export_candidates_jsonl(cands, out)

    assert n == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["requires_human_approval"] is True
    assert "eğitim verisi DEĞİL" in rows[0]["note"]


def test_count_runs_and_truncation_warning(tmp_path: Path, caplog):
    """count_runs toplam verir; limit aşılınca sessiz-kesme UYARISI loglanır (no silent cap)."""
    store = RlmStore(db_path=tmp_path / "rlm.db")
    for _ in range(3):
        _make_run(store, status="answered", conf=0.95, cit=0.95, gnd=0.95, unsupported=[])
    assert store.count_runs() == 3

    caplog.set_level("WARNING")
    cands = select_lora_candidates(store=store, limit=2)  # 3 koşu, limit 2 → kesme
    assert len(cands) == 2  # yalnız en yeni 2 tarandı (hepsi uygun)
    assert "ATLANDI" in caplog.text  # kullanıcı sessizce kaybetmedi


def test_select_rejects_nonpositive_limit(tmp_path: Path):
    """limit<=0 (SQLite'ta -1='sınırsız', 0=hiç) yanlış/çelişkili uyarı üretirdi → reddedilir."""
    store = RlmStore(db_path=tmp_path / "rlm.db")
    _make_run(store, status="answered", conf=0.95, cit=0.95, gnd=0.95, unsupported=[])
    for bad in (0, -1):
        with pytest.raises(ValueError):
            select_lora_candidates(store=store, limit=bad)
