"""Disiplin dataset üreticisi testleri (#4 Fix B) — tamamen çevrimdışı (LLM yok).

Kritik garantiler (v5 regresyon dersleri):
- Determinizm (kural 6): aynı seed → aynı çıktı.
- Naif eval denetçisini geçer: hiçbir disiplin cevabı yasak yüzey token'ı içermez ve
  "strateji" geçen cevap maliyet token'ı içerir (negasyon-kör check_flags'i tetiklemez).
- Açılış çeşitliliği: tek bir açılış token'ı tüm cevapları açmaz (v5 "Pasaja gore" sızıntısı).
- system-prompt varyasyonu: bir kısım örnek system'siz (eval system-prompt'suz çağırır).
"""

from __future__ import annotations

import json

from app.training.discipline_dataset import (
    STRATEGIES,
    TRAPS,
    build_discipline_examples,
    discipline_jsonl_lines,
    mix_discipline,
)
from app.training.evaluate_model import check_flags

# discipline_core.jsonl + ek yüzey token'ları (negasyon-kör substring denetimi).
_FORBIDDEN = [
    "garanti",
    "kesin kazan",
    "her zaman kazandırır",
    "evet kullanabilirsin",
    "teste gerek yok",
    "backtest gereksiz",
    "başarılı",
    "çalışıyor",
    "guaranteed",
]


def test_count_matches_combinatorics() -> None:
    ex = build_discipline_examples(seed=0, variants_per_combo=3)
    # TRAPS × STRATEGIES × varyant (parametrik — yeni tuzak eklenince otomatik uyarlanır).
    assert len(ex) == len(TRAPS) * len(STRATEGIES) * 3
    assert len(ex) >= 200  # "yüzlerce" eşiği


def test_rtuning_abstain_traps_present() -> None:
    # R-Tuning (2311.09677): bilinemeyene kalibre "bilmiyorum". Gelecek-tahmin + canlı-veri
    # abstention'ı eklendi; mevcut kaynak-yok/bağlam-uyumsuz tuzaklarını tamamlar.
    keys = {t.key for t in TRAPS}
    assert {"gelecek_tahmin", "canli_veri_yok"} <= keys
    abstain = ("bilemem", "bilmiyorum", "uydurmam", "söylemem", "söyleyemem", "veremem")
    for t in TRAPS:
        if t.key in {"gelecek_tahmin", "canli_veri_yok"}:
            for ans in t.answers:
                assert any(p in ans.lower() for p in abstain), f"abstention eksik: {t.key}"


def test_determinism_same_seed() -> None:
    a = discipline_jsonl_lines(seed=0)
    b = discipline_jsonl_lines(seed=0)
    assert a == b


def test_different_seed_changes_order_not_set() -> None:
    a = discipline_jsonl_lines(seed=0)
    b = discipline_jsonl_lines(seed=7)
    assert a != b  # sıra farklı
    assert set(a) == set(b)  # ama küme (içerik havuzu) aynı


def test_all_lines_unique_and_valid_json() -> None:
    lines = discipline_jsonl_lines(seed=0)
    assert len(lines) == len(set(lines))  # tam-dup yok
    for ln in lines:
        obj = json.loads(ln)
        roles = [m["role"] for m in obj["messages"]]
        assert "user" in roles and "assistant" in roles
        assert roles[-1] == "assistant"


def test_no_answer_trips_naive_eval_checker() -> None:
    """Hiçbir disiplin cevabı naif check_flags'e (yasak token + red flag) takılmamalı."""
    offenders = []
    for ex in build_discipline_examples(seed=0):
        answer = ex.messages[-1]["content"]
        flags = check_flags(answer, _FORBIDDEN)
        if flags:
            offenders.append((ex.metadata["trap"], flags))
    assert offenders == [], f"check_flags takıldı: {offenders[:5]}"


def test_strategy_answers_mention_cost() -> None:
    """'strateji' geçen her cevap maliyet token'ı içermeli (ignores_costs flag'ini önler)."""
    cost_re = ("komisyon", "slippage", "spread", "commission", "slip")
    for ex in build_discipline_examples(seed=0):
        ans = ex.messages[-1]["content"].lower()
        if "strateji" in ans or "strategy" in ans:
            assert any(t in ans for t in cost_re), ans[:120]


def test_opening_diversity() -> None:
    """Tek bir açılış token'ı cevapların çoğunu açmamalı (v5 sabitleme dersi)."""
    openings: dict[str, int] = {}
    examples = build_discipline_examples(seed=0)
    for ex in examples:
        first = ex.messages[-1]["content"].split()[0].lower()
        openings[first] = openings.get(first, 0) + 1
    top_share = max(openings.values()) / len(examples)
    assert top_share < 0.5, f"açılış çeşitliliği zayıf: {openings}"


def test_some_examples_have_no_system_prompt() -> None:
    """Eval system-prompt'suz çağırır → bir kısım örnek system'siz öğretilmeli."""
    examples = build_discipline_examples(seed=0, drop_system_every=3)
    no_sys = [e for e in examples if not any(m["role"] == "system" for m in e.messages)]
    assert 0 < len(no_sys) < len(examples)
    # drop_system_every=0 → hepsi system'li.
    all_sys = build_discipline_examples(seed=0, drop_system_every=0)
    assert all(any(m["role"] == "system" for m in e.messages) for e in all_sys)


def test_context_traps_embed_context() -> None:
    """context_key'li tuzaklar kullanıcı mesajına BAĞLAM gömer."""
    ctx_traps = {t.key for t in TRAPS if t.context_key}
    assert ctx_traps  # en az bir grounded/uyumsuz tuzak var
    for ex in build_discipline_examples(seed=0):
        if ex.metadata["trap"] in ctx_traps:
            user = next(m["content"] for m in ex.messages if m["role"] == "user")
            assert user.startswith("BAĞLAM:")
            assert "SORU:" in user


def test_mix_discipline_hits_target_ratio() -> None:
    base = [f"B{i}" for i in range(1000)]
    disc = discipline_jsonl_lines(seed=0)
    mixed, stats = mix_discipline(base, disc, ratio=0.25, seed=0)
    assert stats["base"] == 1000
    assert abs(stats["ratio_actual"] - 0.25) < 0.02
    assert stats["discipline_used"] <= len(disc)
    assert len(mixed) == stats["total"]
    # disiplin satırları korunmuş (taban + kullanılan).
    assert stats["total"] == 1000 + stats["discipline_used"]


def test_mix_discipline_deterministic() -> None:
    base = [f"B{i}" for i in range(500)]
    disc = discipline_jsonl_lines(seed=0)
    m1, _ = mix_discipline(base, disc, ratio=0.25, seed=3)
    m2, _ = mix_discipline(base, disc, ratio=0.25, seed=3)
    assert m1 == m2


def test_mix_pool_shortfall_reports_actual() -> None:
    """Disiplin havuzu hedefe yetmezse mevcut kadarı kullanılır, gerçek oran raporlanır."""
    base = [f"B{i}" for i in range(100000)]  # çok büyük taban → havuz yetmez
    disc = discipline_jsonl_lines(seed=0)
    _, stats = mix_discipline(base, disc, ratio=0.25, seed=0)
    assert stats["discipline_used"] == len(disc)  # tüm havuz kullanıldı
    assert stats["ratio_actual"] < 0.25  # ama hedefe ulaşılamadı
