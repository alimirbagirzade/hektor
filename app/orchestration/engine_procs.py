"""engine_procs.py — canlı motor (spawn edilmiş CLI ajanı) süreç kaydı.

NEDEN VAR: ⛔ DURDUR butonu STOP_ALL kill-switch'ini yazar (``storage/STOP_ALL``), ama bir
dosya yazmak **koşan bir alt-süreci kesmez**. AutoDriver motoru bloklayan bir çağrıyla
doğuruyordu ve süreç tutamacı (handle) hiçbir yerde saklanmıyordu → DURDUR'a basıldığında:

  1. `claude -p` süreci 30 dakikalık zaman aşımına kadar KOŞMAYA DEVAM ediyordu
     (abonelik kotası yanmaya devam eder),
  2. arayüz "durduruldu" diyip canlı şeridi kapatıyor ve ⚡ tekrar-giriş kilidini
     AÇIYORDU → kullanıcı ikinci bir motor doğurabiliyordu. PR#122'nin tam olarak
     kapattığı "5 eşzamanlı spawn" kazası, DURDUR yolundan geri geliyordu.

Bu modül canlı motor süreçlerini `run_id` başına kaydeder; STOP_ALL ucu ve koşu sonu
temizliği buradan gerçek `terminate()`/`kill()` yapar. Kayıt SÜREÇ-İÇİdir (kalıcı değil):
web sunucusu yeniden başlarsa zaten spawn eden ebeveyn de ölmüştür.

SÖZLEŞME: `terminate_*` fonksiyonları ASLA istisna fırlatmaz — DURDUR yolu her koşulda
ilerlemelidir (bir sürecin ölmesi diğerlerini durdurmayı engellememeli).
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading

log = logging.getLogger(__name__)

# Sonlandırma sonrası sürecin kendiliğinden ölmesi için tanınan süre; dolduğunda kill().
DEFAULT_GRACE_S = 5.0

# run_id -> canlı Popen'lar. Tüm erişimler _lock altında (route'lar threadpool'da koşar).
_live: dict[str, list[subprocess.Popen]] = {}
_lock = threading.Lock()


def register(run_id: str, proc: subprocess.Popen) -> None:
    """Canlı motor sürecini kaydet (spawn'dan hemen sonra çağrılır)."""
    if not run_id:
        return
    with _lock:
        _live.setdefault(run_id, []).append(proc)


def unregister(run_id: str, proc: subprocess.Popen) -> None:
    """Süreç bittiğinde kaydı düşür (kayıt sızıntısı olmasın)."""
    if not run_id:
        return
    with _lock:
        procs = _live.get(run_id)
        if not procs:
            return
        with contextlib.suppress(ValueError):
            procs.remove(proc)
        if not procs:
            _live.pop(run_id, None)


def live_count() -> int:
    """Şu an kayıtlı (henüz bitmemiş) motor süreci sayısı."""
    with _lock:
        return sum(1 for procs in _live.values() for proc in procs if proc.poll() is None)


def is_run_live(run_id: str) -> bool:
    """Koşuya bağlı gerçekten çalışan bir motor alt-süreci var mı?"""
    if not run_id:
        return False
    with _lock:
        return any(proc.poll() is None for proc in _live.get(run_id, ()))


def _terminate(proc: subprocess.Popen, grace_s: float) -> bool:
    """Tek süreci nazikçe sonlandır, direnirse öldür. Öldürüldüyse/ölüyse True."""
    if proc.poll() is not None:
        return False  # zaten bitmiş — sayma
    try:
        proc.terminate()
    except Exception:  # süreç arada öldüyse sorun değil
        log.debug("Motor süreci terminate edilemedi", exc_info=True)
        return False
    try:
        proc.wait(timeout=grace_s)
    except Exception:
        # Nazik istek yetmedi → sert öldür (kota yakmaya devam etmesine izin verme).
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=grace_s)
    return True


def terminate_run(run_id: str, *, grace_s: float = DEFAULT_GRACE_S) -> int:
    """Bir koşuya ait tüm motor süreçlerini kes. Kesilen süreç sayısı döner."""
    with _lock:
        procs = list(_live.get(run_id) or ())
    killed = sum(1 for p in procs if _terminate(p, grace_s))
    if killed:
        log.warning("Koşu %s: %d motor süreci kesildi.", run_id, killed)
    return killed


def terminate_all(*, grace_s: float = DEFAULT_GRACE_S) -> int:
    """TÜM canlı motor süreçlerini kes (⛔ DURDUR / STOP_ALL yolu). Kesilen sayı döner."""
    with _lock:
        procs = [p for procs in _live.values() for p in procs]
    killed = sum(1 for p in procs if _terminate(p, grace_s))
    if killed:
        log.warning("STOP_ALL: %d canlı motor süreci kesildi.", killed)
    return killed


def reset() -> None:
    """Kaydı temizle (yalnız testler için — süreçleri sonlandırmaz)."""
    with _lock:
        _live.clear()
