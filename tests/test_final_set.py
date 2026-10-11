"""Final test takımı (denetim 2026-10-11 E2): biçim, deterministik anahtar, kilit, sızıntı kapısı.

Sorular burada SENTETİK yer tutuculardır (gerçek final soruları kullanıcıdan gelir, yereldir).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from app.evals import final_set as fs_mod
from app.evals.final_set import (
    CALC_INPUTS,
    CATEGORIES,
    FinalQuestion,
    NumericKey,
    compute_numeric,
    final_status,
    load_final,
    lock_final,
    to_compare_rows,
    validate_final,
)
from app.evals.profile.dataset_loader import DatasetIntegrityError, GoldenAccessError

_SHARPE = {"returns": [0.01, -0.005, 0.02, 0.0], "rf_per_period": 0.0, "periods_per_year": 252}


def _write_set(root, *, per_cat=20, numeric_ok=True, verified=True):
    root.mkdir(parents=True, exist_ok=True)
    qs, keys = [], []
    for cat, spec in CATEGORIES.items():
        subs = spec["subcategories"]
        for i in range(per_cat):
            sub = subs[i % len(subs)]
            qid = f"final-{cat}-{i:02d}"
            q = {
                "id": qid,
                "category": cat,
                "subcategory": sub,
                "family": f"{cat}-aile{i % 5}",
                "question": f"Sentetik yer tutucu {cat} {i}: zq{i}x{len(cat)} ölçüm kalıbı nedir?",
                "written_by": "test",
                "human_verified": verified,
            }
            if sub == "kaynakli_uzun_cevap":
                q["evidence"] = ["dondurulmuş kanıt metni"]
            qs.append(q)
            key: dict = {"id": qid, "rubric_points": ["madde"], "verified_by": "test"}
            if cat == "hesap" and numeric_ok:
                key["numeric"] = {
                    "calc": "sharpe",
                    "inputs": {**_SHARPE, "ddof": 1},
                    "tolerance": 0.01,
                }
            keys.append(key)
    (root / "questions.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), "utf-8"
    )
    (root / "keys.jsonl").write_text(
        "".join(json.dumps(k, ensure_ascii=False) + "\n" for k in keys), "utf-8"
    )
    return root


def test_numeric_keys_are_computed_with_stated_conventions() -> None:
    r = np.array(_SHARPE["returns"])
    assert compute_numeric("sharpe", {**_SHARPE, "ddof": 1}) == pytest.approx(
        r.mean() / r.std(ddof=1) * math.sqrt(252)
    )
    # 100 → 120 → 90 → 130: tepe 120'den 90'a = %25 (pozitif yüzde puanı)
    assert compute_numeric("max_drawdown", {"equity": [100, 120, 90, 130]}) == pytest.approx(25.0)
    # iki işlem %2 ve −%1; taraf başına komisyon %0.1 + slippage %0.05 → gidiş-dönüş %0.3
    inp = {
        "gross_trade_returns": [0.02, -0.01],
        "commission_per_side": 0.001,
        "slippage_per_side": 0.0005,
        "compounding": False,
    }
    assert compute_numeric("net_return_after_costs", inp) == pytest.approx(0.4)
    assert compute_numeric("net_return_after_costs", {**inp, "compounding": True}) == pytest.approx(
        ((1.017) * (0.987) - 1) * 100
    )
    assert compute_numeric("cagr", {"start_value": 100, "end_value": 121, "years": 2}) == (
        pytest.approx(10.0)
    )
    assert compute_numeric("profit_factor", {"trade_pnls": [30, -10, 20, -15]}) == pytest.approx(2)
    exp = {"win_rate": 0.6, "avg_win": 1.0, "avg_loss": 2.0, "cost_per_trade": 0.1}
    assert compute_numeric("expectancy", exp) == pytest.approx(-0.3)
    assert set(CALC_INPUTS) == set(fs_mod.CALC_CONVENTIONS)


def test_numeric_spec_rejects_missing_or_extra_inputs_and_undefined_values() -> None:
    with pytest.raises(ValueError, match="girdiler"):
        NumericKey(calc="sharpe", inputs=_SHARPE, tolerance=0.01)  # ddof eksik: varsayılan YOK
    with pytest.raises(ValueError, match="sıfır"):
        compute_numeric(
            "sharpe",
            {"returns": [0.01, 0.01], "rf_per_period": 0.0, "periods_per_year": 252, "ddof": 1},
        )
    with pytest.raises(ValueError):
        compute_numeric(
            "net_return_after_costs",
            {
                "gross_trade_returns": [0.01],
                "commission_per_side": 0.001,
                "slippage_per_side": 0.0,
                "compounding": "evet",
            },
        )


def test_question_schema_enforces_category_and_evidence() -> None:
    base = {"id": "final-x", "family": "f", "question": "q", "written_by": "k"}
    with pytest.raises(ValueError, match="kategori"):
        FinalQuestion(**base, category="yok", subcategory="sharpe")
    with pytest.raises(ValueError, match="alt kategori"):
        FinalQuestion(**base, category="hesap", subcategory="kural1_dili")
    with pytest.raises(ValueError, match="evidence"):
        FinalQuestion(**base, category="bicim_disiplin", subcategory="kaynakli_uzun_cevap")
    with pytest.raises(ValueError, match="final-"):
        FinalQuestion(**{**base, "id": "llm30-s01"}, category="hesap", subcategory="sharpe")


def test_selection_purpose_is_forbidden(tmp_path) -> None:
    with pytest.raises(GoldenAccessError):
        load_final(purpose="selection", root=tmp_path)
    assert load_final(purpose="leakage_check", root=tmp_path) is None
    assert final_status(tmp_path) == {"state": "yok"}


def test_validate_requires_60_questions_keys_and_human_verification(tmp_path) -> None:
    root = _write_set(tmp_path / "a", per_cat=19, numeric_ok=False, verified=False)
    fs = load_final(purpose="audit", root=root)
    errors = "\n".join(validate_final(fs, check_separation=False))
    assert "57 < 60" in errors
    assert "hesap: 19 soru < 20" in errors
    assert "insan doğrulaması yok" in errors
    assert "deterministik sayısal anahtar" in errors


def test_lock_is_written_once_and_tamper_breaks_every_purpose(tmp_path) -> None:
    root = _write_set(tmp_path / "f")
    with pytest.raises(DatasetIntegrityError, match="kilitlenmemiş"):
        load_final(purpose="final_comparison", root=root)
    manifest = lock_final(root)
    assert manifest["counts"]["n_questions"] == 60 and manifest["counts"]["n_numeric"] == 20
    with pytest.raises(DatasetIntegrityError, match="zaten kilitli"):
        lock_final(root)
    fs = load_final(purpose="final_comparison", root=root)
    assert fs.locked and fs.lock_sha == manifest["lock_sha"]
    rows = to_compare_rows(fs)
    math_rows = [r for r in rows if r.get("type") == "math"]
    assert len(math_rows) == 20 and all(isinstance(r["answer_key"], float) for r in math_rows)
    assert all(r["final_lock_sha"] == manifest["lock_sha"] for r in rows)
    # Kilitten sonra anahtar değişirse HER amaçta (sızıntı kapısı dahil) hata — sessiz değil.
    kp = root / "keys.jsonl"
    kp.write_text(kp.read_text("utf-8").replace('"madde"', '"madde2"'), "utf-8")
    for purpose in ("leakage_check", "audit", "final_comparison"):
        with pytest.raises(DatasetIntegrityError, match="uyuşmuyor"):
            load_final(purpose=purpose, root=root)
    assert final_status(root)["state"] == "bozuk"


def test_human_value_must_match_computed_key(tmp_path) -> None:
    root = _write_set(tmp_path / "h")
    kp = root / "keys.jsonl"
    rows = [json.loads(x) for x in kp.read_text("utf-8").splitlines()]
    rows[next(i for i, r in enumerate(rows) if r.get("numeric"))]["numeric"]["human_value"] = 99.0
    kp.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    errors = validate_final(load_final(purpose="audit", root=root), check_separation=False)
    assert any("insan değeri 99.0" in e for e in errors)


def test_final_must_be_separate_from_llm30(tmp_path) -> None:
    from app.evals.llm30 import load_llm30

    root = _write_set(tmp_path / "s")
    qp = root / "questions.jsonl"
    rows = [json.loads(x) for x in qp.read_text("utf-8").splitlines()]
    rows[0]["question"] = load_llm30(purpose="audit")[0].question  # geliştirme sorusu kopyası
    qp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    errors = validate_final(load_final(purpose="audit", root=root))
    assert any("llm30-s01" in e and "AYRI" in e for e in errors)


def test_leakage_gate_and_self_distill_filter_cover_final_set(tmp_path, monkeypatch) -> None:
    from app.lora.mix_cli import leakage_eval_items
    from app.training.self_distill import eval_leak_filter

    root = _write_set(tmp_path / "l")
    monkeypatch.setattr(fs_mod, "final_root", lambda: root)
    ids = {it.id for it in leakage_eval_items()}
    assert "final-hesap-00" in ids
    assert "final-hesap-00" not in {it.id for it in leakage_eval_items(include_final=False)}
    final_q = load_final(purpose="audit", root=root).questions[3].question
    assert eval_leak_filter([final_q, "Bambaşka bir eğitim sorusu, ilgisiz içerik."]) == {final_q}
    # Bozuk (kilit sonrası değişmiş) set → kapı HATA verir, sessizce taramadan geçmez.
    lock_final(root)
    qp = root / "questions.jsonl"
    qp.write_text(qp.read_text("utf-8") + "\n", "utf-8")
    with pytest.raises(DatasetIntegrityError):
        leakage_eval_items()


def test_final_files_are_gitignored() -> None:
    import subprocess

    from app.lora.mix_common import repo_root
    from app.procutil import NO_WINDOW

    for path in (
        "evals/final/questions.jsonl",
        "evals/final/keys.jsonl",
        "evals/final/manifest.json",
        "reports/comparisons/final_sets/final_x.jsonl",
    ):
        res = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=repo_root(),
            check=False,
            creationflags=NO_WINDOW,
        )
        assert res.returncode == 0, f"{path} git'e girebilir (final anahtarı sızar)"
