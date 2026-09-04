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
