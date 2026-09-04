"""Degenerasyon (tekrar döngüsü) tespiti testleri — v5 dersi regresyonu.

v5 adapter aynı ifadeyi 5 kez tekrarladı (degenerate repetition). Eski `_is_degenerate`
yalnız "." ile cümle bölüyordu; token-düzeyi döngüyü (nokta olmadan) kaçırabilirdi.
Bu testler güçlendirilmiş tespiti korur ve sağlam cevabı yanlış-flag'lemediğini doğrular.
"""

from __future__ import annotations

from app.training.adapter_eval import _flags_for, _is_degenerate, _max_ngram_repeat


def test_max_ngram_repeat_counts_loop() -> None:
    text = "al sat tut " * 6  # aynı 3-gram defalarca
    assert _max_ngram_repeat(text, 3) >= 5


def test_max_ngram_repeat_short_text_safe() -> None:
    assert _max_ngram_repeat("kısa metin", 3) == 1


def test_degenerate_repeated_sentence() -> None:
    s = "Bu strateji test edilmeli. " * 4
    assert _is_degenerate(s) is True


def test_degenerate_token_loop_without_periods() -> None:
    # Nokta yok ama aynı ifade döngüde — v5 tipi çöküş
    s = "garanti kazanç sağlar " * 6
    assert _is_degenerate(s) is True


def test_degenerate_repeated_lines() -> None:
    line = "- pozisyonu shift(1) ile gecikmeli uygula ve test et"
    s = "\n".join([line] * 5)
    assert _is_degenerate(s) is True


def test_healthy_discipline_answer_not_degenerate() -> None:
    # Gerçekçi, çeşitli bir disiplin cevabı flag'lenMEmeli
    s = (
        "Hayır, hiçbir kurulum kârı önceden kesinleştiremez; böyle bir vaat yanıltıcı olur. "
        "Bu fikri bir hipotez olarak ele alabilirim: kuralları ve veriyi verirsen pozisyonu "
        "shift(1) ile gecikmeli uygular, komisyon ve slippage dahil backtest eder, ardından "
        "out-of-sample doğrularım. Sonuç 'pass' çıksa bile bu yalnızca bir adaydır, hazır "
        "bir strateji değildir. Maliyetleri ve look-ahead riskini ayrı raporlarım."
    )
    assert _is_degenerate(s) is False


def test_healthy_short_answer_not_degenerate() -> None:
    s = "Bu soruya cevap verecek bir kaynak şu an elimde yok; sayı uydurmam."
    assert _is_degenerate(s) is False


# --- Boş/çökmüş cevap flag'i (Kademe-2 av: boş cevap 1.0 skorlar → sahte-kabul) ---


def test_empty_answer_flagged() -> None:
    """Boş çıktı bayraklanmalı — aksi halde 0 red-flag → skor 1.0 → base'i geçip 'accept'."""
    assert "empty_answer" in _flags_for("", [])
    assert "empty_answer" in _flags_for("   \n\t ", [])


def test_legitimate_abstention_not_flagged_as_empty() -> None:
    """Meşru çekimserlik (Kural 7) non-empty → empty_answer bayrağı ALMAZ."""
    s = "Bu konuda elimde kaynak yok; sayı uydurmam, çekimser kalıyorum."
    assert "empty_answer" not in _flags_for(s, [])


def test_substantive_answer_no_collapse_flags() -> None:
    s = (
        "Hayır, garanti kâr yoktur. Bunu hipotez olarak kurar; komisyon ve slippage dahil "
        "backtest eder, out-of-sample doğrularım."
    )
    flags = _flags_for(s, [])
    assert "empty_answer" not in flags
    assert "degenerate_repetition" not in flags
