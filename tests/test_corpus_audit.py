"""Korpus bütünlük denetimi testleri — çevrimdışı, saf çekirdek üzerinde.

Her kural, 2026-09-06'da 155 makalelik korpusta ELLE bulunan gerçek bir sessiz kaybı
yeniden üretir; denetçi o kaybı sayı + kimlikle yakalamalı.
"""

from __future__ import annotations

import json

from app.memory.corpus_audit import (
    FAIL,
    PASS,
    WARN,
    KorpusGirdisi,
    cikis_kodu,
    denetle,
    kart_bos_mu,
    kural_boilerplate_baslik,
    kural_bos_kartlar,
    kural_cok_versiyon,
    kural_disk_vs_db,
    kural_gomme,
    kural_oksuz_kartlar,
    kural_sifir_metin,
)

BOS_KART = json.dumps(
    {"paper_id": "p1", "title": None, "main_claim": "", "methods": [], "risk_warnings": []}
)
DOLU_KART = json.dumps({"paper_id": "p1", "title": "X", "main_claim": "Momentum persists."})


def _temiz() -> KorpusGirdisi:
    """Her kuralın PASS verdiği sağlıklı korpus."""
    return KorpusGirdisi(
        disk_pdf_adlari={"a.pdf", "b.pdf"},
        makaleler={
            "p1": ("a.pdf", "Risk-Constrained Kelly Gambling", 40_000, 12),
            "p2": ("b.pdf", "Entropy Analysis of Financial Time Series", 30_000, 9),
        },
        sqlite_gomulu={"p1": 30, "p2": 22},
        sqlite_chunk={"p1": 30, "p2": 22},
        chroma={"p1": 30, "p2": 22},
        kartlar=[("c1", "p1", "approved", DOLU_KART)],
    )


def test_temiz_korpus_pass() -> None:
    d = denetle(_temiz())
    assert d.durum == PASS
    assert all(b.seviye == PASS for b in d.bulgular)
    assert cikis_kodu(d) == 0
    assert d.ozet["makale"] == 2 and d.ozet["chroma_vektor"] == 52


def test_diskte_olup_dbye_girmeyen_pdf_fail() -> None:
    # Gerçek vaka: 155 PDF → 153 makale (başlık-dedup'ı ReAct'i düşürdü).
    g = _temiz()
    g.disk_pdf_adlari.add("arxiv_2210.03629.pdf")
    b = kural_disk_vs_db(g, 10)
    assert b.seviye == FAIL
    assert b.sayi == 1 and b.kimlikler == ["arxiv_2210.03629.pdf"]
    assert "ingest" in b.oneri
    assert cikis_kodu(denetle(g)) == 2  # zincir kapısı


def test_dbde_olup_diskte_olmayan_warn() -> None:
    g = _temiz()
    g.disk_pdf_adlari.discard("b.pdf")
    b = kural_disk_vs_db(g, 10)
    assert b.seviye == WARN and b.kimlikler == ["b.pdf"]


def test_yarim_ingest_fail() -> None:
    # Gerçek vaka: 43.005 chunk SQLite'ta, 750'si Chroma'da.
    g = _temiz()
    g.chroma.pop("p2")
    g.sqlite_gomulu["p2"] = 0
    b = kural_gomme(g, 10)
    assert b.seviye == FAIL and b.kimlikler == ["p2"]
    assert "GÖREMEZ" in b.mesaj


def test_gomme_sayi_sapmasi_warn() -> None:
    g = _temiz()
    g.chroma["p1"] = 25  # 30 embedded ama 25 vektör
    b = kural_gomme(g, 10)
    assert b.seviye == WARN and b.kimlikler == ["p1"]


def test_kart_bos_mu() -> None:
    assert kart_bos_mu(BOS_KART)
    assert kart_bos_mu("{}")
    assert kart_bos_mu("bozuk json")
    assert kart_bos_mu(json.dumps([1, 2]))
    assert not kart_bos_mu(DOLU_KART)


def test_bos_pending_kart_warn_onayli_fail() -> None:
    # Gerçek vaka: 20 karttan 18'i boş, pending kuyruğunda.
    g = _temiz()
    g.kartlar.append(("c2", "p2", "pending", BOS_KART))
    b = kural_bos_kartlar(g, 10)
    assert b.seviye == WARN and b.kimlikler == ["c2"]

    g.kartlar.append(("c3", "p2", "approved", BOS_KART))
    b = kural_bos_kartlar(g, 10)
    assert b.seviye == FAIL and b.sayi == 2
    assert "ONAYLI" in b.mesaj


def test_reddedilmis_bos_kart_sayilmaz() -> None:
    # Temizlik yapıldı (cards reject) → denetçi artık şikayet etmemeli; rejected kart
    # eğitim verisine giremez (lora-dataset yalnız approved alır).
    g = _temiz()
    g.kartlar.append(("c2", "p2", "rejected", BOS_KART))
    b = kural_bos_kartlar(g, 10)
    assert b.seviye == PASS and b.sayi == 0


def test_boilerplate_baslik_warn() -> None:
    g = _temiz()
    g.makaleler["p3"] = ("c.pdf", "Published as a conference paper at ICLR 2023", 20_000, 8)
    g.disk_pdf_adlari.add("c.pdf")
    g.sqlite_chunk["p3"] = g.sqlite_gomulu["p3"] = g.chroma["p3"] = 5
    b = kural_boilerplate_baslik(g, 10)
    assert b.seviye == WARN and b.kimlikler == ["p3"]


def test_sifir_metin_fail_ve_taranmis_warn() -> None:
    # Gerçek vaka: `2405.09673` PDF'i boş (0 karakter), aylardır fark edilmemiş.
    g = _temiz()
    g.makaleler["p1"] = ("a.pdf", "LoRA Learns Less and Forgets Less", 0, 39)
    b = kural_sifir_metin(g, 10)
    assert b.seviye == FAIL and b.kimlikler == ["p1"]

    g.makaleler["p1"] = ("a.pdf", "LoRA Learns Less and Forgets Less", 39 * 50, 39)
    b = kural_sifir_metin(g, 10)
    assert b.seviye == WARN and b.kimlikler == ["p1"]
    assert "OCR" in b.mesaj


def test_n_chars_bilinmiyorsa_sikayet_etme() -> None:
    g = _temiz()
    g.makaleler["p1"] = ("a.pdf", "Bir Başlık", None, None)
    assert kural_sifir_metin(g, 10).seviye == PASS


def test_oksuz_ve_cok_versiyon_kart_warn() -> None:
    g = _temiz()
    g.kartlar += [("c2", "yok", "pending", DOLU_KART), ("c3", "p1", "pending", DOLU_KART)]
    assert kural_oksuz_kartlar(g, 10).kimlikler == ["c2"]
    cv = kural_cok_versiyon(g, 10)
    assert cv.seviye == WARN and cv.kimlikler == ["p1"]


def test_limit_kimlikleri_kirpar_sayiyi_kirpmaz() -> None:
    g = _temiz()
    for i in range(25):
        g.disk_pdf_adlari.add(f"kayip_{i:02d}.pdf")
    b = kural_disk_vs_db(g, limit=5)
    assert b.sayi == 25 and len(b.kimlikler) == 5
    assert b.kimlikler == sorted(b.kimlikler)  # deterministik


def test_strict_warn_cikis_kodu() -> None:
    g = _temiz()
    g.kartlar.append(("c2", "p2", "pending", BOS_KART))
    d = denetle(g)
    assert d.durum == WARN
    assert cikis_kodu(d) == 0
    assert cikis_kodu(d, strict=True) == 1


def test_to_dict_json_serilestirilebilir() -> None:
    d = denetle(_temiz())
    s = json.dumps(d.to_dict(), ensure_ascii=False)
    assert '"durum": "PASS"' in s
