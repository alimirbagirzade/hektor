"""ZORUNLU sızıntı testleri: eğitim ↔ golden/validation ayrımı ve golden koruması."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.evals.profile.dataset_loader import (
    DatasetIntegrityError,
    GoldenAccessError,
    assert_not_eval_path,
    eval_root,
    load_split,
    write_manifest,
)
from app.evals.profile.leakage import check_leakage, normalize_text
from app.evals.profile.schema import EvalItem, Split

ITEM = EvalItem(
    id="e1",
    domain="statistics",
    question=(
        "Örneklem standart sapması 10, örneklem büyüklüğü 25 ise ortalamanın standart hatası "
        "kaçtır?"
    ),
    reference_answer="10/sqrt(25) = 2",
    expected_document_ids=["paper_abc"],
)


def _msg(q: str, a: str = "cevap", source: str = "") -> dict:
    ex: dict = {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
    if source:
        ex["metadata"] = {"source_id": source}
    return ex


def test_exact_duplicate() -> None:
    rep = check_leakage([ITEM], [_msg("alakasız"), _msg(ITEM.question)])
    assert rep.counts() == {"exact": 1} and rep.hits[0].train_index == 1


def test_normalized_duplicate() -> None:
    q = (
        "  örneklem STANDART sapması 10; örneklem büyüklüğü 25 ise, "
        "ortalamanın standart hatası kaçtır "
    )
    assert normalize_text(q) == normalize_text(ITEM.question)
    rep = check_leakage([ITEM], [_msg(q)])
    assert rep.counts() == {"normalized": 1}


def test_near_duplicate() -> None:
    q = (
        "Örneklem standart sapması 10, örneklem büyüklüğü 25 ise ortalamanın standart hatası "
        "kaçtır? Kısaca açıkla."
    )
    rep = check_leakage([ITEM], [_msg(q)])
    assert rep.counts() == {"near_duplicate": 1}


def test_semantic_near_duplicate_with_injected_embedding() -> None:
    def embed(texts):  # "standart hata" içeren her metin aynı yöne düşer
        return [[1.0, 0.0] if "standart hata" in t.casefold() else [0.0, 1.0] for t in texts]

    paraphrase = _msg("Standart hata nasıl bulunur: s=10, n=25 olan bir örneklem için?")
    no_embed = check_leakage([ITEM], [paraphrase])
    assert no_embed.clean  # kelime örtüşmesi düşük → n-gram yakalamaz
    rep = check_leakage([ITEM], [paraphrase], embed_fn=embed)
    assert rep.counts() == {"semantic": 1}


def test_source_id_overlap() -> None:
    rep = check_leakage([ITEM], [_msg("tamamen farklı bir soru metni burada", source="paper_abc")])
    assert rep.counts() == {"source_id": 1}


def test_answer_leak_detected_too() -> None:
    rep = check_leakage([ITEM], [_msg("farklı soru", a="10/sqrt(25) = 2")])
    assert rep.counts() == {"exact": 1}


def test_clean_training_data() -> None:
    rep = check_leakage([ITEM], [_msg("RSI nedir?", "Göreli güç endeksi.", source="paper_zzz")])
    assert rep.clean and rep.n_train == 1


def test_committed_golden_and_validation_are_disjoint() -> None:
    golden = load_split(Split.GOLDEN_TEST, purpose="leakage_check")
    validation = load_split(Split.VALIDATION, purpose="leakage_check")
    as_train = [{"question": v.question, "answer": v.reference_answer} for v in validation]
    rep = check_leakage(golden, as_train)
    assert rep.clean, rep.to_dict()["hits"][:3]
    assert {g.id for g in golden}.isdisjoint({v.id for v in validation})


def test_golden_not_readable_for_selection() -> None:
    with pytest.raises(GoldenAccessError):
        load_split(Split.GOLDEN_TEST, purpose="selection")
    assert load_split(Split.VALIDATION, purpose="selection")


def test_golden_hash_tamper_detected(tmp_path: Path) -> None:
    root = tmp_path / "profile_mix"
    shutil.copytree(eval_root(), root)
    with (root / "golden_test.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "x", "domain": "math", "question": "eklendi"}) + "\n")
    with pytest.raises(DatasetIntegrityError, match="hash"):
        load_split(Split.GOLDEN_TEST, purpose="final_comparison", root=root)
    write_manifest(root)  # bilinçli güncelleme → yeniden geçerli
    assert len(load_split(Split.GOLDEN_TEST, purpose="final_comparison", root=root)) == 28


def test_eval_dir_can_not_be_training_input() -> None:
    with pytest.raises(GoldenAccessError):
        assert_not_eval_path(eval_root() / "golden_test.jsonl")
    assert_not_eval_path(Path("data/lora_sft/lora_sft.jsonl"))


def test_training_code_never_references_eval_sets() -> None:
    """Eğitim/veri üretim kodu golden/validation dizinini okumamalı (statik koruma)."""
    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for sub in ("app/training", "app/lora", "scripts"):
        for p in (repo / sub).rglob("*.py"):
            if p.name in {"mix_cli.py"}:  # yalnız sızıntı DENETİMİ için okur
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "profile_mix" in text or "golden_test" in text:
                offenders.append(str(p.relative_to(repo)))
    assert offenders == []
