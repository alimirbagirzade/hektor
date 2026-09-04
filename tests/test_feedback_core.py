"""Echo feedback çekirdeği — çevrimdışı testler (tmp SQLite + tmp export).

Kayıt → Kural-1 güvenlik reddi → onay → SFT-aday export (idempotent, kanonik veriye
dokunmaz). Eğitim ASLA tetiklenmez.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.feedback.echo import APPROVED, EXPORTED, PENDING, REJECTED, EchoCollector
from app.feedback.store import FeedbackStore


@pytest.fixture
def echo(tmp_path: Path) -> EchoCollector:
    return EchoCollector(store=FeedbackStore(db_path=tmp_path / "fb.db"))


def _rec(echo: EchoCollector, correction: str, **kw: str) -> dict:
    return echo.record(
        source=kw.get("source", "rlm"),
        question=kw.get("question", "Momentum stratejisi maliyet dahil mi test edildi?"),
        bad_answer=kw.get("bad_answer", "Evet, kesinlikle çalışıyor."),
        correction=correction,
        correction_type=kw.get("correction_type", "claim_correction"),
    )


def test_clean_correction_is_pending(echo: EchoCollector) -> None:
    c = _rec(echo, "Slippage + komisyon dahil edilmeli; net Sharpe ayrı raporlanmalı.")
    assert c["status"] == PENDING
    assert c["correction_type"] == "claim_correction"


def test_guarantee_language_correction_rejected(echo: EchoCollector) -> None:
    c = _rec(echo, "Bu strateji garanti kâr sağlar, risksizdir.")
    assert c["status"] == REJECTED
    assert "Kural 1" in c["reject_reason"]


def test_guarantee_language_in_question_rejected(echo: EchoCollector) -> None:
    """Zehir 'question' alanından sızamaz (question SFT'de user-turn olur)."""
    c = echo.record(
        source="rlm",
        question="Bu stratejinin garanti kâr sağladığını test ettiniz mi?",
        bad_answer="",
        correction="Maliyet dahil test edilmeli.",
        correction_type="claim_correction",
    )
    assert c["status"] == REJECTED
    assert "Kural 1" in c["reject_reason"]


def test_guarantee_language_in_bad_answer_rejected(echo: EchoCollector) -> None:
    """Zehir 'bad_answer' alanından da reddedilir (saklanan içerik denetlenir)."""
    c = echo.record(
        source="rlm",
        question="Strateji sağlam mı?",
        bad_answer="Evet, bu garanti kâr getirir.",
        correction="Net Sharpe ayrı raporlanmalı.",
        correction_type="claim_correction",
    )
    assert c["status"] == REJECTED


def test_export_rejects_path_traversal(echo: EchoCollector) -> None:
    """out_path izinli kök (proje + depo dizini) dışına çıkamaz (yol-geçişi savunması).

    Cross-platform 'dışarı' yol: proje kökünün üst dizini (Windows'ta da Linux'ta da kök
    veya depo dizini altında DEĞİL)."""
    from app.config import get_settings

    outside = get_settings().root.parent / "achilles_escape_test.jsonl"
    with pytest.raises(ValueError):
        echo.export_approved(out_path=str(outside))


def test_empty_correction_rejected(echo: EchoCollector) -> None:
    c = _rec(echo, "   ")
    assert c["status"] == REJECTED
    assert "Boş" in c["reject_reason"]


def test_invalid_type_falls_back_to_other(echo: EchoCollector) -> None:
    c = _rec(echo, "Maliyet kontrolü eklenmeli.", correction_type="uydurma_tip")
    assert c["correction_type"] == "other"


def test_approve_then_export_roundtrip(echo: EchoCollector, tmp_path: Path) -> None:
    c = _rec(echo, "Look-ahead'a karşı pozisyon shift(1) ile gecikmeli olmalı.")
    assert echo.approve(c["correction_id"]) is True
    assert echo.store.get(c["correction_id"])["status"] == APPROVED

    out = tmp_path / "feedback_sft.jsonl"
    res = echo.export_approved(out_path=out)
    assert res["n_exported"] == 1
    assert echo.store.get(c["correction_id"])["status"] == EXPORTED

    line = out.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    roles = [m["role"] for m in obj["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert obj["messages"][2]["content"].startswith("Look-ahead")
    assert obj["metadata"]["source"] == "feedback"
    assert obj["metadata"]["feedback_id"] == c["correction_id"]


def test_export_is_idempotent_not_data_losing(echo: EchoCollector, tmp_path: Path) -> None:
    """Export sonrası 'approved' boşalsa bile ikinci export dosyayı SİLMEZ (cumulative)."""
    c = _rec(echo, "Maliyetler (komisyon+slippage) modellenmeli.")
    echo.approve(c["correction_id"])
    out = tmp_path / "fb.jsonl"
    echo.export_approved(out_path=out)
    first = out.read_text(encoding="utf-8")
    # ikinci export: artık 'approved' yok (hepsi exported) → dosya yine aynı kalmalı
    res2 = echo.export_approved(out_path=out)
    assert res2["n_exported"] == 1
    assert out.read_text(encoding="utf-8") == first


def test_approve_rechecks_safety_after_edit(echo: EchoCollector) -> None:
    """Kayıt temizken sonradan zehir enjekte edilirse onay anında yakalanır."""
    c = _rec(echo, "Temiz düzeltme.")
    echo.store.set_status(c["correction_id"], PENDING)
    # DB'de düzeltme metnini zehirle (dış müdahale simülasyonu)
    with echo.store.session() as s:
        from app.feedback.store import FeedbackCorrection

        row = s.get(FeedbackCorrection, c["correction_id"])
        row.correction = "Bu strateji garanti kâr getirir."
    assert echo.approve(c["correction_id"]) is False
    assert echo.store.get(c["correction_id"])["status"] == REJECTED


def test_summary_counts(echo: EchoCollector) -> None:
    _rec(echo, "İyi düzeltme bir.")
    _rec(echo, "garanti kâr vaadi")  # rejected
    s = echo.summary()
    assert s["pending"] == 1
    assert s["rejected"] == 1


# ── Kademe-2 av sertleştirmeleri ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "poison",
    [
        "Risk yok, hemen pozisyon aç.",  # 'risk yok' (FINANCIAL_DIRECTIVES)
        "Bu sinyal canlı: BTC al, asla kaybetmezsin.",  # 'asla kaybet' (extra)
        "Garantili bir kurulum: %20 getiri kesin.",  # 'garantili' (extra)
        "no risk, buy now",  # scan FINANCIAL_DIRECTIVES
        "%100 kâr edersin.",  # '%100 kâr' (extra)
    ],
)
def test_advice_and_certainty_language_rejected(echo: EchoCollector, poison: str) -> None:
    """ECHO-POISON-01: yalnız 'guaranteed_profit' değil; advice/risk-free/canlı-sinyal/kesinlik
    dili de intake'te reddedilir (geniş safety_scanner + extra direktif taraması)."""
    c = _rec(echo, poison)
    assert c["status"] == REJECTED
    assert "Kural 1" in c["reject_reason"]


def test_clean_academic_correction_not_overblocked(echo: EchoCollector) -> None:
    """Akademik 'risksiz faiz oranı (risk-free rate)' yanlış-pozitif vermez (FP narrowing)."""
    c = _rec(echo, "Sharpe oranı risksiz faiz oranını (risk-free rate) çıkararak hesaplanır.")
    assert c["status"] == PENDING


def test_export_skips_late_caught_poison(echo: EchoCollector, tmp_path: Path) -> None:
    """ECHO-POISON-02: filtre güçlenmeden ÖNCE onaylanmış (bayat) zehirli kayıt, export'ta
    yeniden taranıp ATLANIR — feedback satırı geniş güvenlik tarayıcısına burada uğrar."""
    c = _rec(echo, "Temiz görünüm.")
    echo.approve(c["correction_id"])
    # DB'de zehirle (onay-anı denetimini atlayan dış müdahale simülasyonu)
    with echo.store.session() as s:
        from app.feedback.store import FeedbackCorrection

        row = s.get(FeedbackCorrection, c["correction_id"])
        row.correction = "Garantili %100 kâr, asla kaybetmezsin."
    out = tmp_path / "fb.jsonl"
    res = echo.export_approved(out_path=out)
    assert res["n_exported"] == 0
    assert res["skipped_poison"] == 1
    assert out.read_text(encoding="utf-8").strip() == ""
    # zehirli kayıt EXPORTED'a işaretlenMEZ (yazılmadı)
    assert echo.store.get(c["correction_id"])["status"] == APPROVED


def test_export_escapes_unicode_line_separators(echo: EchoCollector, tmp_path: Path) -> None:
    """ECHO-JSONL-001: U+2028/U+2029/U+0085 kaçışlanır → tek geçerli JSONL satırı (splitlines
    bölmez), json.loads roundtrip korunur."""
    sep = "satır1" + chr(0x2028) + "satır2" + chr(0x2029) + "son" + chr(0x85) + "x"
    c = _rec(echo, sep)
    echo.approve(c["correction_id"])
    out = tmp_path / "fb.jsonl"
    echo.export_approved(out_path=out)
    raw = out.read_text(encoding="utf-8")
    body = raw.rstrip("\n")
    assert chr(0x2028) not in body and chr(0x2029) not in body and chr(0x85) not in body
    assert len(body.splitlines()) == 1  # tek kayıt tek satır
    obj = json.loads(body)
    assert obj["messages"][2]["content"] == sep  # roundtrip


def test_export_order_deterministic_on_created_at_tie(echo: EchoCollector, tmp_path: Path) -> None:
    """Kural 6: created_at eşitliğinde sıra correction_id ile determinist (snapshot-bağımsız)."""
    from app.feedback.store import FeedbackCorrection

    ids = []
    for i in range(3):
        c = _rec(echo, f"Maliyet kontrolü eklenmeli varyant {i}.")
        echo.approve(c["correction_id"])
        ids.append(c["correction_id"])
    # hepsini AYNI created_at'e zorla (kaba saat / aynı tick simülasyonu)
    with echo.store.session() as s:
        for cid in ids:
            s.get(FeedbackCorrection, cid).created_at = "2026-06-29T00:00:00.000000+00:00"
    out = tmp_path / "fb.jsonl"
    echo.export_approved(out_path=out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    got = [json.loads(ln)["metadata"]["feedback_id"] for ln in lines]
    assert got == sorted(ids)  # correction_id ikincil anahtar → kararlı sıra


def test_export_excludes_record_rejected_during_window(echo: EchoCollector, tmp_path: Path) -> None:
    """Eşzamanlı /reject (snapshot ile içerik-okuma arasında) reddedilen kaydı dosyaya YAZMAZ."""
    c = _rec(echo, "Pozisyon shift(1) ile gecikmeli olmalı.")
    echo.approve(c["correction_id"])
    real_get = echo.store.get
    flipped = {"done": False}

    def get_then_reject(cid: str):  # type: ignore[no-untyped-def]
        if not flipped["done"] and cid == c["correction_id"]:
            flipped["done"] = True
            echo.store.set_status(cid, REJECTED, "eşzamanlı reddetme")
        return real_get(cid)

    echo.store.get = get_then_reject  # type: ignore[method-assign]
    try:
        out = tmp_path / "fb.jsonl"
        res = echo.export_approved(out_path=out)
    finally:
        echo.store.get = real_get  # type: ignore[method-assign]
    assert res["n_exported"] == 0
    assert out.read_text(encoding="utf-8").strip() == ""
