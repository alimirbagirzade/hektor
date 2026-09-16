"""Eğitim nöbeti testleri — 2026-09-16 gecesinin iki olayını kilitler. Çevrimdışı, saf.

1. Ölü bir koşunun `train_status.json`'ı nöbetçi için KALICI YETKİ gibi davrandı:
   web'den başlayıp 0. adımda ölen, kimsenin onaylamadığı bir eğitim 13 dakika sonra
   diriltildi ve 5,5 saat eski veriyle koştu.
2. Askıya alınmış (`NtSuspendProcess`) bir eğitim 5,5 saat 21/600 adımda dondu; süreç
   "canlı" göründüğü için fark edilmedi.
"""

from __future__ import annotations

import datetime as dt

from app.training.train_guard import (
    APPROVAL_WINDOW_MINUTES,
    LOG_STALL_MINUTES,
    MAX_STATUS_AGE_HOURS,
    diagnose,
    find_run_approval,
    recovery_allowed,
)

_NOW = dt.datetime(2026, 9, 16, 12, 0, tzinfo=dt.UTC)


def _status(started: dt.datetime, **extra: object) -> dict:
    base = {
        "adapter": "hektor_lora_v9_4b",
        "dtype": "bf16",
        "iterations": 600,
        "pid": 4242,
        "started_at": started.isoformat(),
    }
    base.update(extra)
    return base


def _approval(
    consumed: dt.datetime | None, *, status: str = "approved", aid: str = "apr_x"
) -> dict:
    return {
        "approval_id": aid,
        "action": "train_run",
        "status": status,
        "consumed_at": consumed.isoformat() if consumed else None,
    }


# --- kurtarma yetkisi ------------------------------------------------------
def test_onayli_kosu_diriltilebilir() -> None:
    started = _NOW - dt.timedelta(hours=2)
    v = recovery_allowed(_status(started), [_approval(started - dt.timedelta(minutes=2))], now=_NOW)
    assert v.allowed, v.reason
    assert v.details["approval_id"] == "apr_x"


def test_onaysiz_kosu_diriltilemez() -> None:
    """Gecenin 1. olayı: onaysız ölü koşu, durum dosyası sayesinde dirilmişti."""
    started = _NOW - dt.timedelta(hours=2)
    v = recovery_allowed(_status(started), [], now=_NOW)
    assert not v.allowed
    assert "onay" in v.reason.lower()


def test_baska_kosunun_onayi_sayilmaz() -> None:
    """Bir hafta önce başka bir eğitim için tüketilen onay bu koşuyu yetkilendirmez."""
    started = _NOW - dt.timedelta(hours=1)
    eski = _approval(started - dt.timedelta(days=7))
    assert not recovery_allowed(_status(started), [eski], now=_NOW).allowed
    # Pencere içindeki onay kabul edilir (sınır davranışı).
    sinir = _approval(started - dt.timedelta(minutes=APPROVAL_WINDOW_MINUTES - 1))
    assert recovery_allowed(_status(started), [sinir], now=_NOW).allowed


def test_tuketilmemis_onay_yetki_vermez() -> None:
    started = _NOW - dt.timedelta(hours=1)
    assert not recovery_allowed(_status(started), [_approval(None)], now=_NOW).allowed


def test_bayat_durum_dosyasi_diriltilemez() -> None:
    started = _NOW - dt.timedelta(hours=MAX_STATUS_AGE_HOURS + 1)
    v = recovery_allowed(_status(started), [_approval(started)], now=_NOW)
    assert not v.allowed
    assert "bayat" in v.reason


def test_veri_degistiyse_dirilme_yok() -> None:
    """Koşudan sonra veri yeniden kurulduysa diriltilen koşu onaylanan veriyi eğitmez."""
    started = _NOW - dt.timedelta(hours=3)
    v = recovery_allowed(
        _status(started),
        [_approval(started)],
        now=_NOW,
        data_mtime=started + dt.timedelta(hours=1),
    )
    assert not v.allowed
    assert "veri" in v.reason


def test_bos_durum_ve_bozuk_zaman_fail_closed() -> None:
    assert not recovery_allowed({}, [], now=_NOW).allowed
    assert not recovery_allowed(_status(_NOW, started_at="bozuk"), [], now=_NOW).allowed


def test_find_run_approval_yalniz_train_run_bakar() -> None:
    started = _NOW - dt.timedelta(minutes=5)
    yabanci = _approval(started, aid="apr_y")
    yabanci["action"] = "rules_apply"
    assert find_run_approval([yabanci], started) is None


# --- sağlık teşhisi --------------------------------------------------------
def test_askida_kosu_yakalanir() -> None:
    """Gecenin 2. olayı: süreç canlı, log saatlerdir ilerlemiyor, CPU ~0."""
    started = _NOW - dt.timedelta(hours=6)
    d = diagnose(
        status=_status(started),
        running=True,
        now=_NOW,
        log_mtime=_NOW - dt.timedelta(minutes=LOG_STALL_MINUTES + 60),
        cpu_percent=0.0,
        approvals=[_approval(started)],
    )
    assert d.verdict == "DIKKAT"
    assert any("ilerlemiyor" in p for p in d.problems)
    assert any("CPU" in p for p in d.problems)


def test_saglikli_kosu_temiz_rapor() -> None:
    started = _NOW - dt.timedelta(hours=2)
    d = diagnose(
        status=_status(started),
        running=True,
        now=_NOW,
        log_mtime=_NOW - dt.timedelta(minutes=3),
        cpu_percent=280.0,
        data_mtime=started - dt.timedelta(minutes=30),
        approvals=[_approval(started)],
    )
    assert d.verdict == "OK", d.problems
    assert d.problems == []


def test_onaysiz_kosan_egitim_bildirilir() -> None:
    started = _NOW - dt.timedelta(hours=1)
    d = diagnose(
        status=_status(started),
        running=True,
        now=_NOW,
        log_mtime=_NOW - dt.timedelta(minutes=2),
        cpu_percent=300.0,
        approvals=[],
    )
    assert d.verdict == "DIKKAT"
    assert any("Kural 8" in p for p in d.problems)


def test_durum_kaydi_olmayan_kosu_sessizce_ok_demez() -> None:
    """Durum kaydı yoksa koşu onaya bağlanamaz → 'OK' demek yanlış güven verir.

    2026-09-16'da CLI ile başlatılan koşu için durum dosyası bilinçli yazılmamıştı; doktor
    onu 'OK' gösteriyordu, çünkü onay kontrolü `started_at` olmadan sessizce atlanıyordu.
    """
    d = diagnose(status={}, running=True, now=_NOW, log_mtime=_NOW, cpu_percent=250.0)
    assert d.verdict == "DIKKAT"
    assert any("durum kaydı" in p for p in d.problems)


def test_olu_kosu_kaydi_bildirilir() -> None:
    """Süreç yok ama durum dosyası duruyor → nöbetçinin dirilteceği tuzak."""
    d = diagnose(status=_status(_NOW - dt.timedelta(hours=1)), running=False, now=_NOW)
    assert d.verdict == "DIKKAT"
    assert any("ölü koşu" in p for p in d.problems)


def test_bosta_durum() -> None:
    d = diagnose(status={}, running=False, now=_NOW)
    assert d.verdict == "BOSTA"
    assert d.problems == []
