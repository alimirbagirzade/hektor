"""LLM-30 geliştirme benchmark'ı: bütünlük, sızıntı kapısı ve sayısal anahtar doğruluğu."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.evals.llm30 import (
    CRITICAL_BY_QUESTION,
    CRITICAL_ERRORS,
    QUESTION_IDS,
    RUBRIC,
    SYSTEM_PROMPT,
    answer_flags,
    llm30_root,
    load_llm30,
    numeric_keys,
    user_prompt,
)
from app.evals.profile.dataset_loader import DatasetIntegrityError
from app.evals.profile.leakage import check_leakage


def test_thirty_questions_hash_verified() -> None:
    items = load_llm30(purpose="selection")  # geliştirme seti: seçimde okunabilir
    assert [it.id for it in items] == list(QUESTION_IDS)
    for n, it in enumerate(items, 1):
        assert it.question.startswith(f"S{n:02d}. ")
        assert "a)" in it.question
        assert not it.human_verified  # anahtar insan doğrulaması bekliyor


def test_tampered_file_rejected(tmp_path: Path) -> None:
    root = tmp_path / "llm30"
    shutil.copytree(llm30_root(), root)
    with (root / "validation.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"id": "x", "domain": "math", "question": "eklendi"}\n')
    with pytest.raises(DatasetIntegrityError, match="hash"):
        load_llm30(purpose="selection", root=root)


def test_critical_errors_mapped_to_existing_questions() -> None:
    mapped = {e for errs in CRITICAL_BY_QUESTION.values() for e in errs}
    assert mapped == set(CRITICAL_ERRORS)  # her kritik hata en az bir soruda denetlenir
    assert set(CRITICAL_BY_QUESTION) <= set(QUESTION_IDS)
    assert RUBRIC["dogruluk"] == 4 and sum(RUBRIC.values()) == 12


def test_leakage_gate_catches_llm30_question() -> None:
    items = load_llm30(purpose="leakage_check")
    leaked = {"question": items[21].question, "answer": "cevap"}
    rep = check_leakage(items, [leaked])
    assert not rep.clean


def test_leakage_check_includes_llm30(tmp_path: Path) -> None:
    from app.lora.mix_cli import run_leakage_check

    q = load_llm30(purpose="leakage_check")[5].question
    train = tmp_path / "train.jsonl"
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": "x"}]
    train.write_text(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n", encoding="utf-8")
    report = run_leakage_check(train)
    assert report["hits"], "llm30 sorusu eğitim verisinde yakalanmadı"


def test_numeric_keys_match_hand_computation() -> None:
    k = numeric_keys()
    assert k["s03_utc"] == "2026-01-15T07:00:00+00:00"  # İstanbul sabit UTC+3
    assert k["s06_hourly_ohlcv"] == [100.0, 110.0, 99.0, 109.0, 100.0]
    assert k["s10_split_toplam_deger"] == (1000.0, 1000.0)
    assert k["s10_split_duzeltilmemis_getiri_pct"] == pytest.approx(-50.0)
    assert k["s10_temettu_toplam_getiri_pct"] == pytest.approx(0.0)
    assert k["s12_ema_alpha_0_1"] == pytest.approx(101.0)
    assert k["s12_ema_alpha_0_5"] == pytest.approx(105.0)
    assert k["s13_z"] == pytest.approx(2.0)
    assert k["s14_kalman_kazanci"] == pytest.approx(2 / 3)
    assert k["s14_guncel_durum"] == pytest.approx(104.0)
    assert k["s14_guncel_varyans"] == pytest.approx(4 / 3)
    assert k["s19_entropi_esit"] == pytest.approx(1.0)
    assert str(k["s19_entropi_deterministik"]) == "0.0"  # -0.0 değil
    assert k["s22_gecis_sayilari"] == [[0, 1, 1], [2, 0, 0], [0, 1, 0]]
    assert k["s22_satir_normalize"] == [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert k["s25_brut_beklenen_pct"] == pytest.approx(-0.2)
    assert k["s25_net_beklenen_pct"] == pytest.approx(-0.3)
    assert k["s25_basabas_olasilik"] == pytest.approx(0.7)
    assert (k["s26_serbest_parametre_k3"], k["s26_serbest_parametre_k10"]) == (6, 90)


def test_transition_matrix_ignores_index_alignment() -> None:
    """Kritik hata 'indeks_hizalama': karışık indeksli Series bile konumsal eşleşmeli."""
    import pandas as pd

    from app.evals.llm30 import _transition_matrix

    s = pd.Series([0, 1, 0, 2, 1, 0], index=[5, 3, 9, 1, 0, 7])
    counts, probs = _transition_matrix(s.tolist(), 3)
    assert counts.tolist() == numeric_keys()["s22_gecis_sayilari"]
    assert probs.sum(axis=1).tolist() == [1.0, 1.0, 1.0]  # satır normalize, global değil


def _flags(answer: str, **kw: object) -> list[str]:
    opts: dict = {"done_reason": "stop", "prompt_tokens": 100, "output_tokens": 50, "num_ctx": 8192}
    opts.update(kw)
    return answer_flags(answer, **opts)


def test_answer_flags_separates_failure_modes() -> None:
    assert _flags("EMA = 101. Varsayım: alpha=0.1.") == []
    assert _flags("x", done_reason="length") == ["kesildi_token_siniri"]
    assert _flags("x", done_reason="iptal:tekrar_limiti") == ["ollama_tekrar_iptali"]
    assert _flags("x", done_reason="iptal:baska") == ["ollama_iptal"]
    assert _flags("x", prompt_tokens=8000, output_tokens=192) == ["baglam_siniri"]
    assert _flags("   ") == ["bos_cevap"]
    assert _flags("Sonuç: " + "evet " * 8) == ["kisa_tekrar"]
    loop = "Bu durum look-ahead bias yaratır ve testi geçersiz kılar. "
    assert _flags(loop * 3) == ["uzun_tekrar"]
    assert _flags("Cevap 期权 içeriyor") == ["cjk_sizinti"]


def test_long_loop_ignores_format_headers_and_data_rows() -> None:
    """2026-09-30 yanlış pozitifleri: istenen başlıklar ve sorunun gerektirdiği CSV satırı."""
    per_part = "- **Doğrudan cevap:**\nEMA değeri burada hesaplanır ve yorumlanır, tamam.\n"
    assert (
        _flags(
            "a)\n"
            + per_part
            + "b)\n"
            + per_part.replace("EMA", "SMA")
            + "c)\n"
            + per_part.replace("EMA", "WMA")
        )
        == []
    )
    row = "2025-04-05T10:00:00,ETH/USD,2800.0,2810.0,2790.0,2805.0,150"
    assert _flags(f"{row}\n{row}\n{row}\nYinelenen satırlar incelenmeli.") == []
    # Art arda olmayan ama ≥4 kez geçen içerik cümlesi yine döngü sayılır.
    s = "Bu durum look-ahead bias yaratır ve testi geçersiz kılar."
    assert _flags(" Ara cümle. ".join([s] * 4)) == ["uzun_tekrar"]


def test_answer_flags_ignores_markdown_rules() -> None:
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n" + "-" * 40 + "\n" + "=" * 30
    assert _flags(table) == []


def test_user_prompt_rag_on_off() -> None:
    assert user_prompt("S?", None) == "S?"
    assert user_prompt("S?", "[p:1] metin") == "KAYNAKLAR:\n[p:1] metin\n\nSORU:\nS?"
    assert "RAG açıkken" in SYSTEM_PROMPT


def test_reference_answers_agree_with_numeric_keys() -> None:
    ref = {it.id: it.reference_answer for it in load_llm30(purpose="audit")}
    assert "07:00 UTC" in ref["llm30-s03"]
    assert "close=109" in ref["llm30-s06"] and "volume=100" in ref["llm30-s06"]
    assert "→ 101" in ref["llm30-s12"] and "→ 105" in ref["llm30-s12"]
    assert "= 104" in ref["llm30-s14"]
    assert "0.7" in ref["llm30-s25"]
    assert "K=10 → 90" in ref["llm30-s26"]
