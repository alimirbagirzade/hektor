"""Hibrit kaynaklı cevap (çevrimdışı, deterministik; LLM yok).

Karar (2026-09-13): model yalnız Kısa Cevap'ı yazar; diğer 7 bölüm retrieval, bilgi kartları
ve kurallardan kurulur. Kart içeriği özgün dilinde etiketli alıntıdır (çeviri yok).
"""

from __future__ import annotations

from app.brain import hybrid_answer as ha
from app.memory.retrieval_service import RetrievedChunk

_SIM = {"min_similarity": 0.55, "min_margin": 0.02}


def _chunk(
    paper_id: str, chunk_id: str, distance: float, title: str | None = "T"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text="metin",
        page_number=4,
        section_name=None,
        title=title,
        distance=distance,
    )


_CARD = {
    "title": "Graph Signal Processing",
    "main_claim": "GSP improves RV forecasting.",
    "possible_strategy_hypotheses": ["h1 by 5-10%", "h2", "h3", "h4"],
    "risk_warnings": ["Overfitting risk", "Data leakage risk"],
}


def _by_title(sections: list[ha.AnswerSection]) -> dict[str, ha.AnswerSection]:
    return {s.title: s for s in sections}


def test_sekiz_bolum_sirasi_ve_kaynak_rozetleri() -> None:
    chunks = [_chunk("p1", "c1", 0.1), _chunk("p1", "c2", 0.3)]
    secs = ha.build_sections("Kısa cevap.", chunks, {"p1": _CARD}, **_SIM)

    assert [(s.title, s.source) for s in secs] == [
        ("Kısa Cevap", "model"),
        ("Kaynaklar", "kaynak"),
        ("Bağlam Kalitesi", "kural"),
        ("Akademik Bulgu", "kaynak"),
        ("Trading Hipotezi", "kaynak"),
        ("Test Planı", "kural"),
        ("Riskler", "kural+kaynak"),
        ("Sonraki Adım", "kural"),
    ]
    assert secs[0].body == "Kısa cevap."
    assert all(isinstance(s.to_dict(), dict) for s in secs)


def test_model_yalniz_kisa_cevapta_kullanilir_bos_cevap_uyarir() -> None:
    secs = ha.build_sections("   ", [_chunk("p1", "c1", 0.1)], {}, **_SIM)
    assert secs[0].warning is True
    assert "boş" in secs[0].body
    assert [s.source for s in secs].count("model") == 1


def test_test_plani_kural_2_4_maddelerini_garanti_eder() -> None:
    body = _by_title(ha.build_sections("x", [_chunk("p1", "c1", 0.1)], {}, **_SIM))[
        "Test Planı"
    ].body
    assert "out-of-sample" in body
    assert "Komisyon + slippage" in body
    assert "shift(1)" in body
    assert "seed" in body


def test_baglam_kalitesi_esikleri() -> None:
    def level(dist: float) -> ha.AnswerSection:
        return ha.context_quality_section([_chunk("p1", "c1", dist)], **_SIM)

    assert level(0.10).body.startswith("Güçlü")  # benzerlik 0.90 ≥ 0.55×1.5
    assert level(0.35).body.startswith("Orta")  # 0.65
    weak = level(0.60)  # 0.40 < 0.55
    assert weak.body.startswith("Zayıf")
    assert weak.warning is True
    assert "zayıf dayanağa" in weak.body


def test_kaynaklar_atif_ve_baslik() -> None:
    sec = ha.sources_section([_chunk("p1", "c1", 0.1), _chunk("p2", "c9", 0.2, title=None)])
    assert sec.body == "- [p1:c1, s.4] — T\n- [p2:c9, s.4]"


def test_kart_alintilari_etiketli_ve_sinirli() -> None:
    chunks = [_chunk("p1", "c1", 0.1)]
    secs = _by_title(ha.build_sections("x", chunks, {"p1": _CARD}, **_SIM))

    finding = secs["Akademik Bulgu"].body
    assert (
        finding == "[p1] Graph Signal Processing (kaynak, çevrilmedi)\nGSP improves RV forecasting."
    )

    hyp = secs["Trading Hipotezi"].body
    assert hyp.startswith(ha.HYPOTHESIS_NOTE)  # sayılar doğrulanmamış uyarısı HER ZAMAN başta
    assert "- h1 by 5-10%" in hyp and "- h3" in hyp and "h4" not in hyp  # en fazla 3 madde

    risks = secs["Riskler"].body
    assert risks.startswith("- Overfit")  # kural maddeleri önce
    assert "(kaynak, çevrilmedi)\n- Overfitting risk\n- Data leakage risk" in risks


def test_kart_yoksa_acik_not_ve_kural_bolumleri_yine_var() -> None:
    secs = _by_title(ha.build_sections("x", [_chunk("p1", "c1", 0.1)], {}, **_SIM))
    assert "ana iddia yok" in secs["Akademik Bulgu"].body
    assert "doğrudan trading kuralına çevrilemez" in secs["Trading Hipotezi"].body
    assert secs["Riskler"].source == "kural"


def test_load_cards_tekil_ilk_iki_makale_ve_kartsizi_atlar() -> None:
    calls: list[str] = []

    def lookup(pid: str) -> dict | None:
        calls.append(pid)
        return None if pid == "p2" else {"title": pid}

    chunks = [
        _chunk("p1", "c1", 0.1),
        _chunk("p1", "c2", 0.1),
        _chunk("p2", "c3", 0.2),
        _chunk("p3", "c4", 0.3),
    ]
    cards = ha.load_cards(chunks, lookup)
    assert calls == ["p1", "p2"]  # tekil, retrieval sırası, en fazla MAX_CARD_PAPERS
    assert cards == {"p1": {"title": "p1"}}
