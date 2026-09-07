"""Regresyon: §16 atıf kapısı ATIFSIZ cevaplarda boş yere sağlanmasın (kural 7).

Bulgu: `ConfidenceScorer._citation_score` atıf kümesi BOŞ olduğunda 1.0 döndürüyordu
(boş küme = "hepsi doğru"). Bu nötr değer `citation_score ≥ 0.90` aday kapısını boş
yere geçiriyor, hiçbir kaynağa bağlanmamış iddialı cevaplar LoRA ADAYI oluyordu.

Testler tamamen ÇEVRİMDIŞI: izole SQLite (tmp_path), sahte koşu kayıtları, LLM yok.
"""

from __future__ import annotations

from pathlib import Path

from app.rlm.lora_candidate import (
    effective_citation_score,
    has_inline_citation,
    select_lora_candidates,
)
from app.rlm.rlm_store import RlmStore

# RlmController zarfının kaynak-listesi bloğu: ATIFSIZ cevapta da bulunur.
_KAYNAK_BLOGU = "Kaynak dayanakları:\n- [p1:p1_c0001, s.3] | Giriş | Örnek Makale"

# (a) ATIFSIZ ama İDDİALI cevap: gövdede tek bir satır-içi atıf yok.
ATIFSIZ_IDDIALI = (
    "Kısa cevap:\n"
    "Momentum stratejileri düşük volatilite rejiminde daha iyi çalışır.\n\n"
    "Makalelere göre gerekçe:\n"
    "1. Sharpe oranı rejim değişiminde belirgin biçimde yükselir.\n\n"
    f"{_KAYNAK_BLOGU}\n\n"
    "Güven seviyesi: High"
)

# (b) ATIFLI cevap: aynı zarf, ama iddialar satır-içi atıf taşıyor.
ATIFLI = (
    "Kısa cevap:\n"
    "Momentum stratejileri düşük volatilite rejiminde daha iyi çalışır "
    "[p1:p1_c0001, s.3].\n\n"
    "Makalelere göre gerekçe:\n"
    "1. Sharpe oranı rejim değişiminde yükselir [p1:p1_c0001, s.3].\n\n"
    f"{_KAYNAK_BLOGU}\n\n"
    "Güven seviyesi: High"
)

# (c) MEŞRU ÇEKİMSERLİK: kural 7'nin DOĞRU davranışı — uydurma yerine "cevap veremem".
CEKIMSER = (
    "Kısa cevap:\n"
    "Bu soruya kaynaklarla desteklenen güvenilir bir cevap üretilemedi.\n\n"
    "Neden: Taslak cevaptaki hiçbir iddia kaynaklarla yeterince desteklenmedi.\n\n"
    "Kaynak dayanakları (retrieval):\n- [p1:p1_c0001, s.3] Örnek Makale\n\n"
    "Güven seviyesi: Low"
)


def _run(
    store: RlmStore,
    *,
    status: str,
    answer: str,
    claims: list[str],
    cit: float,
) -> str:
    """§16'nın DİĞER tüm eşiklerini geçen sahte koşu yaz (yalnız atıf boyutu değişken)."""
    run_id = store.create_run("Momentum ne zaman çalışır?", "general_paper_question", "stub")
    store.finish_run(
        run_id,
        status=status,
        final_answer=answer,
        final_confidence=0.95,
        evidence_score=90.0,
    )
    store.set_verification(
        run_id,
        supported_claims=claims,
        unsupported_claims=[],
        contradictions=[],
        citation_score=cit,
        grounding_score=0.95,
        context_sufficiency_score=0.95,
        final_decision=status,
    )
    return run_id


# ── (a) ATIFSIZ-İDDİALI → ADAY DEĞİL ────────────────────────────────────────────


def test_atifsiz_iddiali_kosu_aday_olamaz(tmp_path: Path, caplog):
    """Boş atıf kümesinin 1.0 nötr değeri kapıyı GEÇMEMELİ (asıl bulgu)."""
    store = RlmStore(db_path=tmp_path / "rlm.db")
    _run(
        store,
        status="answered",
        answer=ATIFSIZ_IDDIALI,
        claims=["Sharpe oranı rejim değişiminde belirgin biçimde yükselir."],
        cit=1.0,  # ← boş atıf kümesinin ürettiği sahte "mükemmel" skor
    )

    caplog.set_level("WARNING")
    cands = select_lora_candidates(store=store)

    assert cands == []  # atıfsız koşu eğitim adayı OLAMAZ (kural 7)
    assert "ATIFSIZ" in caplog.text  # sessizce elenmedi — denetlenebilir uyarı


def test_atifsiz_kosu_efektif_skoru_sifir():
    """Atıf yokken skor 1.0 DEĞİL 0.0 olmalı; kaynak-listesi bloğu atıf sayılmaz."""
    assert effective_citation_score(1.0, ATIFSIZ_IDDIALI, []) == 0.0
    assert has_inline_citation(ATIFSIZ_IDDIALI, []) is False
    # Zarfın sonundaki kaynak dökümü tek başına "iddia kaynağa bağlandı" kanıtı değil.
    assert has_inline_citation(f"Bir iddia.\n\n{_KAYNAK_BLOGU}", []) is False


# ── (b) ATIFLI-YÜKSEK SKORLU → ADAY ─────────────────────────────────────────────


def test_atifli_yuksek_skorlu_kosu_aday_olur(tmp_path: Path):
    """Gerçek atıflı koşuda davranış DEĞİŞMEZ: aday olur, ham skor korunur."""
    store = RlmStore(db_path=tmp_path / "rlm.db")
    rid = _run(
        store,
        status="answered",
        answer=ATIFLI,
        claims=["Sharpe oranı rejim değişiminde yükselir [p1:p1_c0001, s.3]."],
        cit=0.95,
    )

    cands = select_lora_candidates(store=store)

    assert [c.run_id for c in cands] == [rid]
    assert cands[0].citation_score == 0.95  # ham skor aynen korundu (indirim YOK)
    assert cands[0].requires_human_approval is True
    assert effective_citation_score(0.95, ATIFLI, []) == 0.95


def test_atif_kanidi_yalniz_iddialarda_olsa_da_yeter(tmp_path: Path):
    """Atıf, zarf gövdesinde değil kayıtlı destekli iddialarda ise de kabul edilir."""
    store = RlmStore(db_path=tmp_path / "rlm.db")
    rid = _run(
        store,
        status="answered",
        answer=f"Kısa cevap:\nMomentum çalışır.\n\n{_KAYNAK_BLOGU}",
        claims=["Momentum çalışır [p1:p1_c0001]."],
        cit=0.95,
    )
    assert [c.run_id for c in select_lora_candidates(store=store)] == [rid]


# ── (c) MEŞRU ÇEKİMSERLİK → MEVCUT DAVRANIŞ KORUNUR ─────────────────────────────


def test_mesru_cekimserlik_davranisi_degismedi(tmp_path: Path, caplog):
    """Çekimser koşu kural 7'nin DOĞRU davranışıdır; aday değildir ama cezalandırılmaz.

    Eleme gerekçesi ESKİDEN OLDUĞU GİBİ `status` filtresidir (abstained/no_llm eğitim
    adayı olamaz) — yeni atıf kapısı değil. Bu yüzden çekimser koşu için ATIFSIZ uyarısı
    da üretilmez.
    """
    store = RlmStore(db_path=tmp_path / "rlm.db")
    for status in ("abstained", "no_llm"):
        _run(store, status=status, answer=CEKIMSER, claims=[], cit=1.0)

    caplog.set_level("WARNING")
    cands = select_lora_candidates(store=store)

    assert cands == []  # değişmedi: çekimser koşu zaten aday değildi
    assert "ATIFSIZ" not in caplog.text  # çekimserlik yeni kapıyla damgalanmıyor


def test_cekimser_ve_atifsiz_ayni_kovaya_dusmuyor(tmp_path: Path):
    """İki durum AYRI: çekimser status'ten, atıfsız-iddialı atıf kapısından elenir."""
    store = RlmStore(db_path=tmp_path / "rlm.db")
    cekimser_id = _run(store, status="abstained", answer=CEKIMSER, claims=[], cit=1.0)
    atifsiz_id = _run(store, status="answered", answer=ATIFSIZ_IDDIALI, claims=["İddia."], cit=1.0)
    iyi_id = _run(
        store,
        status="answered",
        answer=ATIFLI,
        claims=["Sharpe yükselir [p1:p1_c0001, s.3]."],
        cit=0.95,
    )

    secilen = {c.run_id for c in select_lora_candidates(store=store)}

    assert secilen == {iyi_id}
    assert cekimser_id not in secilen and atifsiz_id not in secilen
