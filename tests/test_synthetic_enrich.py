"""Sentetik QA zenginleştirme — çevrimdışı (sahte LLM). Kapılar + devam + uygulama."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.brain.synthetic_enrich import (
    apply_enrichment,
    enrich_line,
    run_enrichment,
    validate_enriched,
    work_paths,
)

_CTX = (
    "The Average True Range (ATR) measures volatility over 14 periods. Position size is "
    "scaled inversely to ATR so that each trade risks a similar amount; wider stops follow "
    "higher volatility. The approach assumes recent volatility persists."
)
_GOOD = (
    "ATR, 14 periyotluk volatiliteyi ölçer ve pozisyon büyüklüğü ATR ile ters orantılı "
    "ayarlanır. Böylece her işlem benzer bir risk taşır; volatilite yükseldiğinde stop "
    "mesafesi genişler ve pozisyon küçülür. Yaklaşım yakın dönem volatilitesinin süreceğini "
    "varsayar, bu varsayım rejim değişiminde zayıflayabilir."
)


def _line(answer: str = "ATR volatiliteyi ölçer.", **meta: object) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"BAĞLAM:\n{_CTX}\n\nSORU: ATR ne işe yarar?"},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {"synthetic": True, **meta},
        },
        ensure_ascii=False,
    )


class _LLM:
    model = "qwen3:test"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, prompt: str, **kw: object) -> str:
        self.calls += 1
        return json.dumps({"answer": self.answer}, ensure_ascii=False)


def test_good_answer_replaces_and_keeps_original_in_metadata() -> None:
    out = enrich_line(_line(), _LLM(_GOOD))
    assert out.status == "enriched"
    obj = json.loads(out.line)
    assert obj["messages"][-1]["content"] == _GOOD
    assert obj["metadata"]["enriched"] is True
    assert obj["metadata"]["orig_answer"] == "ATR volatiliteyi ölçer."
    assert obj["messages"][1]["content"].startswith("BAĞLAM:")  # soru/bağlam korunur


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("Kısa.", "kısa"),
        (_GOOD + " Sharpe oranı 2.35 olarak ölçülmüştür.", "grounding"),  # uydurma sayı
        (_GOOD + " 后的", "cjk"),
        ("Pasajda " + _GOOD, "düşük-değer"),
        (_GOOD + " Metinde bu konu ayrıca işlenir.", "düşük-değer"),
    ],
)
def test_rejected_answer_keeps_original_line(answer: str, reason: str) -> None:
    src = _line()
    out = enrich_line(src, _LLM(answer))
    assert out.status == reason
    assert out.line == src  # veri asla kötüleşmez


def test_repetition_rejected() -> None:
    loop = "ATR volatiliteyi 14 periyotta ölçer ve pozisyonu küçültür. " * 5
    assert validate_enriched(loop, "x", _CTX) == "tekrar"


def test_long_or_already_enriched_rows_skipped_without_llm() -> None:
    llm = _LLM(_GOOD)
    assert enrich_line(_line("x" * 250), llm).status == "skipped"
    assert enrich_line(_line(enriched=True), llm).status == "skipped"
    no_ctx = _line().replace("BAĞLAM:\\n", "")
    assert enrich_line(no_ctx, llm).status == "skipped"
    assert llm.calls == 0


def test_run_is_resumable_and_aligned(tmp_path: Path) -> None:
    src = tmp_path / "synthetic_qa.jsonl"
    src.write_text("\n".join([_line(), _line("x" * 250), _line()]) + "\n", encoding="utf-8")
    llm = _LLM(_GOOD)
    r1 = run_enrichment(src, llm, limit=2)
    assert r1["done"] == 2
    r2 = run_enrichment(src, llm)
    assert r2["done"] == 3 and r2["counts"] == {"enriched": 2, "skipped": 1}
    work, _ = work_paths(src)
    assert len(work.read_text(encoding="utf-8").splitlines()) == 3


def test_resume_refused_when_source_changed(tmp_path: Path) -> None:
    src = tmp_path / "synthetic_qa.jsonl"
    src.write_text(_line() + "\n" + _line() + "\n", encoding="utf-8")
    run_enrichment(src, _LLM(_GOOD), limit=1)
    src.write_text(_line("başka") + "\n" + _line() + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run_enrichment(src, _LLM(_GOOD))


def test_apply_requires_complete_run_and_backs_up(tmp_path: Path) -> None:
    src = tmp_path / "synthetic_qa.jsonl"
    original = _line() + "\n" + _line() + "\n"
    src.write_text(original, encoding="utf-8")
    run_enrichment(src, _LLM(_GOOD), limit=1)
    with pytest.raises(ValueError):
        apply_enrichment(src, tmp_path / "bak")
    run_enrichment(src, _LLM(_GOOD))
    res = apply_enrichment(src, tmp_path / "bak")
    assert res["applied"] == 2
    assert Path(res["backup"]).read_text(encoding="utf-8") == original
    assert all(json.loads(ln)["metadata"]["enriched"] for ln in src.read_text("utf-8").splitlines())
    assert not work_paths(src)[0].exists()


def test_adapter_model_detection() -> None:
    from app.brain.synthetic_enrich import is_adapter_model

    assert is_adapter_model("hektor-v12-30b")
    assert is_adapter_model("Hektor-v10:latest")
    assert not is_adapter_model("qwen3:30b-a3b-instruct-2507-q4_K_M")


def test_resume_refused_when_model_changed(tmp_path: Path) -> None:
    src = tmp_path / "synthetic_qa.jsonl"
    src.write_text(_line() + "\n" + _line() + "\n", encoding="utf-8")
    a = _LLM(_GOOD)
    a.model = "qwen3:30b"  # type: ignore[attr-defined]
    run_enrichment(src, a, limit=1)
    b = _LLM(_GOOD)
    b.model = "hektor-v12-30b"  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        run_enrichment(src, b)


# Kademe-2 bulucu C2-C4 (2026-09-30): eski kapılar bunları %57-99 oranında kabul ediyordu.
@pytest.mark.parametrize(
    ("tail", "reason"),
    [
        (
            " Bu yaklaşım uzun vadede yatırımcılar için önemli avantajlar sağlar. Ayrıca farklı "
            "piyasa koşullarında da benzer sonuçlar elde edilebilir.",
            "grounding",
        ),
        (" Bu nedenle yatırımcı portföyünün tamamını bu yönteme ayırmalı.", "tavsiye"),
        (" Kaynak metninde bu ayrıca vurgulanır ve ATR 14 periyotla ölçülür.", "düşük-değer"),
        (" Mekanizma, ATR yükseldiğinde pozisyonun küçülmesidir.", "etiket"),
    ],
)
def test_c2_c4_gates_reject_fabricated_tails(tail: str, reason: str) -> None:
    assert validate_enriched(_GOOD + tail, "ATR volatiliteyi ölçer.", _CTX) == reason


def test_english_answer_rejected() -> None:
    eng = (
        "The ATR measures volatility over 14 periods and position size is scaled inversely "
        "to it, so that each trade risks a similar amount of capital in the market. Wider "
        "stops follow higher volatility and the approach assumes recent volatility persists."
    )
    assert validate_enriched(eng, "ATR volatiliteyi ölçer.", _CTX) == "dil"


def test_math_wording_not_treated_as_advice() -> None:
    from app.brain.synthetic_enrich import _ADVICE_RE

    for ok in ("sıfır ortalamalı dağılım", "her zaman aynı noktada", "sonlu olmayı garanti eder"):
        assert not _ADVICE_RE.search(ok), ok


def test_truncated_trailing_line_is_dropped_on_resume(tmp_path: Path) -> None:
    src = tmp_path / "synthetic_qa.jsonl"
    src.write_text(_line() + "\n" + _line() + "\n", encoding="utf-8")
    run_enrichment(src, _LLM(_GOOD), limit=1)
    work, _ = work_paths(src)
    work.write_text(work.read_text(encoding="utf-8") + '{"messages": [', encoding="utf-8")
    res = run_enrichment(src, _LLM(_GOOD))
    assert res["done"] == 2
    assert all(json.loads(ln) for ln in work.read_text(encoding="utf-8").splitlines())
    apply_enrichment(src, tmp_path / "bak")  # hiza doğrulaması geçer
