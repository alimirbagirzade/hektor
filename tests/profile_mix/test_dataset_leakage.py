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


NEAR_Q = (
    "Örneklem standart sapması 10, örneklem büyüklüğü 25 ise ortalamanın standart hatası nedir?"
)


def test_near_duplicate() -> None:
    rep = check_leakage([ITEM], [_msg(NEAR_Q)])
    assert rep.counts() == {"near_duplicate": 1}


def test_superset_of_question_is_contained() -> None:
    """Eval sorusunu birebir içeren daha uzun metin (önceden near-dup) artık 'contained'."""
    q = (
        "Örneklem standart sapması 10, örneklem büyüklüğü 25 ise ortalamanın standart hatası "
        "kaçtır? Kısaca açıkla."
    )
    rep = check_leakage([ITEM], [_msg(q)])
    assert rep.counts() == {"contained": 1}


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


PASSAGE = (
    "BAĞLAM:\nMerkezi limit teoremi, bağımsız ve özdeş dağılımlı gözlemlerin ortalamasının "
    "örneklem büyüdükçe normal dağılıma yaklaştığını söyler; varyans sonlu olmalıdır.\n\n"
)


def test_question_packed_with_context_is_flagged() -> None:
    """Gerçek eğitim biçimi: ``BAĞLAM: <pasaj>\\n\\nSORU: <soru>`` — soru bağlamla paketli."""
    rep = check_leakage([ITEM], [_msg(PASSAGE + "SORU: " + ITEM.question)])
    assert not rep.clean
    assert rep.counts() == {"exact": 1}  # SORU bölümü ayrı metin olarak eşleşir


def test_near_duplicate_question_packed_with_context_is_flagged() -> None:
    rep = check_leakage([ITEM], [_msg(PASSAGE + "Soru: " + NEAR_Q)])
    assert rep.counts() == {"near_duplicate": 1}


def test_question_embedded_verbatim_in_longer_text_is_contained() -> None:
    long_text = "Aşağıdaki problemi çöz. " + ITEM.question + " Cevabını gerekçelendir."
    rep = check_leakage([ITEM], [_msg(long_text)])
    assert rep.counts() == {"contained": 1} and rep.hits[0].train_index == 0


def test_reference_answer_embedded_in_training_answer_is_contained() -> None:
    item = ITEM.model_copy(
        update={"reference_answer": "Standart hata sigma bölü karekök n formülüyle bulunur."}
    )
    answer = "Adım adım: " + item.reference_answer + " Burada sigma=10, n=25 → 2."
    rep = check_leakage([item], [_msg("tamamen alakasız bir soru metni", a=answer)])
    assert rep.counts() == {"contained": 1}


def test_short_text_is_not_containment_checked() -> None:
    """Kısa metin bilinçli olarak yalnız exact/normalized — uzun pasajda geçmesi sızıntı değil."""
    item = ITEM.model_copy(update={"reference_answer": "Sonuç 2 olur."})
    rep = check_leakage([item], [_msg("alakasız", a="Hesap yapınca sonuç 2 olur. Bitti.")])
    assert rep.clean


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
    with pytest.raises(GoldenAccessError):
        assert_not_eval_path(eval_root().parent / "llm30" / "validation.jsonl")
    assert_not_eval_path(Path("data/lora_sft/lora_sft.jsonl"))


def test_training_code_never_references_eval_sets() -> None:
    """Eğitim/veri üretim kodu golden/validation dizinini okumamalı (statik koruma)."""
    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for sub in ("app/training", "app/lora", "scripts"):
        for p in (repo / sub).rglob("*.py"):
            # Yalnız sızıntı DENETİMİ için okurlar (eğitim verisi üretmez, eğitim başlatmaz).
            if p.name in {"mix_cli.py", "eval_set_leak_check.py"}:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if any(k in text for k in ("profile_mix", "golden_test", "llm30")):
                offenders.append(str(p.relative_to(repo)))
    assert offenders == []


MULTI = EvalItem(
    id="m1",
    domain="math",
    question=(
        "S99. Hareketli ortalama\n"
        "a) Üstel hareketli ortalamanın yeni gözlem katsayısını açıkça tanımla ve formülü yaz.\n"
        "b) Pencere uzadıkça gecikmenin nasıl değiştiğini sayısal örnekle göster."
    ),
)


def test_single_subpart_of_multipart_question_is_detected() -> None:
    """Kademe 2 F4-3: eğitim satırı tek bir alt maddeye eşitse eskiden temiz geçiyordu."""
    part_a = MULTI.question.split("\n")[1]
    assert not check_leakage([MULTI], [_msg(part_a)]).clean
    body_b = MULTI.question.split("\n")[2][3:]
    assert not check_leakage([MULTI], [_msg(body_b)]).clean
    assert check_leakage([MULTI], [_msg("Kalman filtresi nasıl çalışır?")]).clean


def test_question_segment_reads_live_rag_format() -> None:
    """Kademe 2 F1-5/F4-4: canlı RAG biçimi `QUESTION / SORU:` eskiden '' döndürüyordu."""
    from app.brain.rag_answerer import build_rag_prompt
    from app.evals.profile.leakage import question_segment

    _, user = build_rag_prompt("Standart hata nasıl hesaplanır?", [])
    assert question_segment(user) == "Standart hata nasıl hesaplanır?"
    assert question_segment("BAĞLAM:\nx\n\nSORU: eski biçim") == "eski biçim"
