"""Eğitim-öncesi dataset kalite kapısı testleri (#3) — saf, çevrimdışı (LLM/DB yok)."""

from __future__ import annotations

import json

from app.training.dataset_quality import audit_dataset, recommend_epochs
from app.training.discipline_dataset import discipline_jsonl_lines


def _line(answer: str, user: str = "soru") -> str:
    return json.dumps(
        {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]},
        ensure_ascii=False,
    )


def test_recommend_epochs_boundaries() -> None:
    assert recommend_epochs(400) == 1
    assert recommend_epochs(1500) == 2
    assert recommend_epochs(3000) == 3


def test_clean_set_is_go() -> None:
    answers = [
        "Önermem; bu bir hipotezdir. shift(1), komisyon + slippage dahil backtest + OOS gerekir.",
        "Hayır demem gerekir. Pozisyonu gecikmeli uygula, maliyet dahil ölç, OOS doğrula.",
        "Bu soruya kaynak yok; uydurmam. Backtest geçmişini sorgula.",
        "Maliyeti yok sayan rakam vermem; komisyon + slippage dahil net getiri ölçülür.",
    ]
    rep = audit_dataset([_line(a) for a in answers])
    assert rep.verdict == "GO"
    assert rep.guaranteed_profit_hits == 0
    assert not rep.blockers


def test_guaranteed_profit_blocks() -> None:
    lines = [_line("Bu kurulum garanti kâr getirir, kesin kazanırsın.")]
    rep = audit_dataset(lines)
    assert rep.verdict == "NO-GO"
    assert rep.guaranteed_profit_hits >= 1
    assert any("garanti" in b.lower() or "Kural 1" in b for b in rep.blockers)


def test_opening_memorization_blocks() -> None:
    # Tüm cevaplar aynı bigramla açılıyor → ezber riski (v5 mekanizması).
    lines = [_line(f"Pasaja göre bu {i}. cevaptır ve dayanak budur.") for i in range(12)]
    rep = audit_dataset(lines)
    assert rep.verdict == "NO-GO"
    assert rep.top_opening == "pasaja göre"
    assert rep.top_opening_share > 0.4


def test_leakage_prefix_warns_without_blocking() -> None:
    # 3/100 sızıntı öneki → uyarı (eşik %2), ama açılış ezberi yok → GO.
    # Açılış bigramları gerçekten çeşitli olmalı (yoksa ezber bloku tetiklenir).
    _varied = [
        "Önermem bu hipotezdir test gerekir.",
        "Hayır demem gerekir kaynak yok.",
        "Maliyeti dahil etmeden rakam vermem.",
        "Backtest olmadan bunu söyleyemem.",
        "Şüpheyle bakarım overfit riski var.",
        "Kaynak yetersiz uydurmam burada.",
        "Bunu yapma look-ahead olur.",
    ]
    lines = [_line("pasaja göre dayanak budur ve sonuç budur.") for _ in range(3)]
    lines += [_line(_varied[i % len(_varied)] + f" ek {i}") for i in range(97)]
    rep = audit_dataset(lines)
    assert rep.verdict == "GO", rep.blockers
    assert rep.leakage_prefix_hits == 3
    assert any("sızıntı" in w for w in rep.warnings)


def test_ignores_costs_warns() -> None:
    lines = [_line("Bu strateji iyi görünüyor ama test edilmeli.")]  # 'strateji' var, maliyet yok
    rep = audit_dataset(lines)
    assert rep.ignores_costs_hits >= 1
    assert any("maliyet" in w for w in rep.warnings)


def test_discipline_pool_passes_gate() -> None:
    """Kendi disiplin verimiz kapıyı GO geçmeli (zehir/ezber yok)."""
    disc = discipline_jsonl_lines(seed=0)
    rep = audit_dataset(disc, discipline_lines=disc)
    assert rep.verdict == "GO", rep.blockers
    assert rep.guaranteed_profit_hits == 0
    assert rep.discipline_present == rep.discipline_target == len(disc)


def test_small_dataset_warns() -> None:
    rep = audit_dataset([_line("Kısa ama temiz bir cevap; hipotez + test noktası.")])
    assert any("overfit" in w for w in rep.warnings)
    assert rep.recommended_epochs == 1


# --- Şablon tekrarı (v8 dersi, 2026-09-14) -------------------------------------------------


def _unique_words(i: int, k: int = 12) -> str:
    """Her cevaba ÖZGÜ, yalnız harflerden oluşan kelimeler (paylaşılan 8-gram üretmez)."""

    def word(n: int) -> str:
        return "".join(chr(97 + (n // 26**j) % 26) for j in range(4))

    return " ".join(word(i * 97 + j) for j in range(k))


# v8'in eğitim setindeki ortak kuyruğun birebir hâli (discipline_dataset'teki eski _TEST_TAIL).
_OLD_TAIL = (
    "Doğru test noktası: pozisyonu shift(1) ile gecikmeli uygula, komisyon ve slippage "
    "DAHİL backtest et, sonra out-of-sample doğrula. 'pass' çıksa bile bu bir ADAY'dır."
)


def test_template_repetition_blocks() -> None:
    # 200 cevabın 30'u (%15) aynı kuyrukla bitiyor → v8 mekanizması → NO-GO.
    lines = [
        _line(_unique_words(i) + (". " + _OLD_TAIL if i % 200 < 30 else ".")) for i in range(200)
    ]
    rep = audit_dataset(lines)
    assert rep.verdict == "NO-GO"
    assert rep.template_ngrams_over_block > 0
    assert rep.top_template_ngram_share == 0.15
    assert any("şablon tekrarı" in b for b in rep.blockers)


def test_template_repetition_warns_in_band() -> None:
    # 1000 cevabın 15'i (%1,5): uyarı bandı (%1 < pay ≤ %2), bloklamaz.
    lines = [_line(_unique_words(i) + (". " + _OLD_TAIL if i < 15 else ".")) for i in range(1000)]
    rep = audit_dataset(lines)
    assert rep.verdict == "GO", rep.blockers
    assert rep.template_ngrams_over_block == 0
    assert any("şablon tekrarı sınırda" in w for w in rep.warnings)


def test_template_rule_needs_absolute_count_on_small_sets() -> None:
    # 20 cevabın 5'i (%25) aynı kuyruk — ama 5 tekrar mutlak alt sınırın (10) altında: blok yok.
    lines = [_line(_unique_words(i) + (". " + _OLD_TAIL if i < 5 else ".")) for i in range(20)]
    rep = audit_dataset(lines)
    assert rep.template_ngrams_over_block == 0
    assert not any("şablon" in b for b in rep.blockers)


def test_unique_answers_have_no_template_signal() -> None:
    rep = audit_dataset([_line(_unique_words(i)) for i in range(300)])
    assert rep.template_ngrams_over_block == 0
    assert rep.top_template_ngram_share < 0.01
