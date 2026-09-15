"""Sentetik QA kalite düzeltmeleri (2026-09-15) — çevrimdışı (LLM/DB sahte).

Ölçüm: kitaplardan üretilen 326 örneğin %33'ü "pasajda açıklanmamıştır" diyen çekimser cevap,
%40'ı "Pasaj…" açılışlıydı (eski veride %3). Kök neden: üretici `chunks[:max_chunks]` ile
kitabın ilk chunk'larını (kapak, telif, içindekiler, şekil listesi) alıyordu.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.brain.chunk_selection import is_content_chunk as _is_content_chunk
from app.brain.chunk_selection import select_content_chunks as _select_chunks
from app.brain.synthetic_qa_builder import SyntheticQABuilder, is_low_value_answer
from app.training.dataset_quality import audit_dataset

_CONTENT = (
    "Pairs trading exploits cointegration between two assets. When the spread between the "
    "prices deviates from its long-run mean, the trader sells the relatively expensive asset "
    "and buys the cheap one, expecting mean reversion. Transaction costs, borrowing fees and "
    "the risk that the cointegrating relationship breaks down during a regime change must be "
    "estimated out-of-sample before any allocation is made; otherwise the backtest overstates "
    "the achievable Sharpe ratio considerably."
)


# --- düşük değerli cevap ---------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "Pasajda cointegrated pairların keşfedilme mekanizması açıklanmamıştır.",
        "Bu kavram, verilen bilgiler arasında yer almamaktadır ve tanımlanmamıştır.",
        "Pasaja göre spread ortalamaya döner.",
        "Bu pasaj drawdown süresini ele alır ve risk ölçüsü olarak kullanır.",
        "Metinde bu konuda bilgi bulunmamaktadır; ayrıntı belirtilmemiştir.",
    ],
)
def test_low_value_answers_detected(answer: str) -> None:
    assert is_low_value_answer(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "Pairs trading, eşbütünleşik iki varlığın fiyat farkının ortalamaya dönmesine dayanır.",
        "Bu yöntemde işlem maliyetleri ayrıca belirtilmektedir ve getiriden düşülür.",
        "Model varsayımları ekte açıklanmaktadır; rejim değişimi ayrı ele alınır.",
        "Borç alma ücreti de maliyet kalemleri arasında yer almaktadır.",
    ],
)
def test_affirmative_answers_kept(answer: str) -> None:
    assert is_low_value_answer(answer) is False


# --- içerik chunk seçimi -----------------------------------------------------------------


def _chunk(idx: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(chunk_id=f"c{idx}", text=text)


def test_front_matter_and_figure_lists_are_not_content() -> None:
    assert (
        _is_content_chunk("ADVANCES IN FINANCIAL MACHINE LEARNING BY MARCOS LÓPEZ DE PRADO")
        is False
    )
    figure_list = " ".join(f"Table {i}.1 {i + 10} Equation {i} {i + 11}" for i in range(40))
    assert _is_content_chunk(figure_list) is False
    copyright_page = "Copyright 2018 by John Wiley & Sons. All rights reserved. " + "x " * 300
    assert _is_content_chunk(copyright_page) is False
    assert _is_content_chunk(_CONTENT) is True


def test_select_chunks_skips_front_matter_and_spreads_evenly() -> None:
    front = [_chunk(0, "TITLE PAGE"), _chunk(1, "Contents " + "Chapter 1 12 " * 60)]
    body = [_chunk(i, f"{_CONTENT} Bölüm {i}.") for i in range(2, 102)]
    picked = _select_chunks(front + body, 8)
    ids = [c.chunk_id for c in picked]
    assert len(ids) == 8
    assert "c0" not in ids and "c1" not in ids
    idx = [int(i[1:]) for i in ids]
    assert idx == sorted(idx) and idx[0] < 20 and idx[-1] > 85  # belgenin tamamına yayılır
    assert _select_chunks(front + body, 8) == picked  # determinist


def test_select_chunks_falls_back_for_short_documents() -> None:
    short = [
        _chunk(i, f"RSI momentum osilatörü, parça {i}, yeterince uzun metin.") for i in range(3)
    ]
    assert [c.chunk_id for c in _select_chunks(short, 8)] == ["c0", "c1", "c2"]


# --- üretici düşük değerli cevabı eler ----------------------------------------------------


class _LLM:
    def generate(self, prompt: str, **kwargs: object) -> str:
        return json.dumps(
            {
                "pairs": [
                    {
                        "question": "Pairs trading neye dayanır?",
                        "answer": "Pairs trading, iki varlık arasındaki cointegration ilişkisine "
                        "dayanır; spread ortalamadan saptığında pozisyon açılır.",
                    },
                    {
                        "question": "Stratejinin kaldıraç oranı nedir?",
                        "answer": "Pasajda pairs trading stratejisinin kaldıraç oranı ve spread "
                        "eşiği açıklanmamıştır.",
                    },
                ]
            },
            ensure_ascii=False,
        )


def test_builder_drops_abstention_answers() -> None:
    ex = SyntheticQABuilder(llm=_LLM()).build_for_chunk(_CONTENT, paper_id="p1", chunk_id="c1", n=2)
    answers = [e.messages[-1]["content"] for e in ex]
    assert len(answers) == 1
    assert "cointegration" in answers[0]


def test_prompt_no_longer_invites_abstention() -> None:
    prompt = SyntheticQABuilder(llm=_LLM())._build_prompt(_CONTENT, ("backtester", "x"), 2, None)
    assert "acikca belirt" not in prompt
    assert "pasajda cevabi olmayan" in prompt


# --- kapı + birleştirme -------------------------------------------------------------------


def _line(answer: str) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": "soru"},
                {"role": "assistant", "content": answer},
            ]
        },
        ensure_ascii=False,
    )


def test_passage_opening_share_blocks_gate() -> None:
    varied = [
        _line(f"Açılış {i} farklı bir cevapla başlar ve {i} numaralı notla biter.")
        for i in range(85)
    ]
    leaky = [
        _line(f"Pasajda {i}. kavram ayrıntılı biçimde anlatılır ve örneklenir.") for i in range(15)
    ]
    rep = audit_dataset(varied + leaky)
    assert rep.leakage_prefix_hits == 15
    assert rep.verdict == "NO-GO"
    assert any("sızıntı" in b for b in rep.blockers)


def test_assembly_drops_low_value_synthetic_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.training.sft_assembly import assemble_sft_lines

    synth = tmp_path / "data" / "lora_sft" / "synthetic_qa.jsonl"
    synth.parent.mkdir(parents=True)
    good = _line(
        "Pairs trading, eşbütünleşik iki varlığın fiyat farkının ortalamaya dönmesine dayanır."
    )
    bad = _line("Pasajda bu stratejinin kaldıraç oranı açıklanmamıştır, bilgi bulunmamaktadır.")
    synth.write_text(good + "\n" + bad + "\n", encoding="utf-8")

    def _no_db() -> None:
        raise RuntimeError("DB yok (test)")

    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", _no_db)
    res = assemble_sft_lines(SimpleNamespace(root=tmp_path), discipline=False)
    assert res.synth_n == 2
    assert res.low_value_dropped == 1
    assert len(res.lines) == 1 and "eşbütünleşik" in res.lines[0]
