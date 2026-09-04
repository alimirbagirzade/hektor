"""driver_scope.py — sürücü (driver) kimlik token'ları: kısa ömürlü + koşuya bağlı.

Achilles kendi motorunu (`claude -p`) doğurur (bkz. app/orchestration/driver.py).
Bu modül, doğurulan motora **insandan daha az** yetkili bir kimlik verir: token
bir `run_id`'ye bağlıdır, TTL ile sınırlıdır ve koşu bitince iptal edilir.

TEHDİT SINIRI (dürüst ol — bkz. docs/SCOPE_ISOLATION.md):
Bu token bir *yetenek*tir (capability): motor onu YALNIZCA ebeveyn tarafından
verildiği için taşır. Ama aynı OS kullanıcısı altında çalışan bir süreç, token'ı
hiç göndermeyip "insan" istemcisi gibi davranabilir. Bu katman tek başına
kriptografik bir sınır DEĞİLDİR; asıl sınır motoru araç-seviyesinde kısıtlamaktır
(driver.py `--disallowedTools`) + `api_token` atanmış olmasıdır.

Depo süreç-içidir (kalıcı değil): web sunucusu yeniden başlarsa tüm sürücü
token'ları geçersizleşir — istenen davranış (koşu da ölmüştür).
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass

# Sürücü token'ı YALNIZ bu başlıkta kabul edilir — `Authorization` başlığında ASLA.
# Böylece bir sürücü token'ı yanlışlıkla insan yoluna düşemez.
DRIVER_TOKEN_HEADER = "x-achilles-driver-token"
RUN_ID_HEADER = "x-achilles-run-id"

# Ortam değişkeni adları (doğurulan sürece geçirilir).
DRIVER_TOKEN_ENV = "ACHILLES_DRIVER_TOKEN"
DRIVER_RUN_ID_ENV = "ACHILLES_DRIVER_RUN_ID"

# Varsayılan TTL: derin av zaman aşımıyla (HUNT_TIMEOUT_S=1800) hizalı + pay.
DEFAULT_TTL_S = 2100


@dataclass(frozen=True)
class _Record:
    run_id: str
    expires_at: float


# token_sha256 -> _Record. Tüm erişimler _lock altında (route'lar threadpool'da koşar).
_store: dict[str, _Record] = {}
_lock = threading.Lock()


def _hash(token: str) -> str:
    """Token'ı sha256 ile özetle — bellek dökümü/log sızıntısı ham token vermez."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _purge_expired_locked(now: float) -> None:
    for h in [h for h, rec in _store.items() if rec.expires_at <= now]:
        del _store[h]


def mint(run_id: str, *, ttl_s: int = DEFAULT_TTL_S) -> str:
    """Bir koşu için yeni sürücü token'ı üret (ham token YALNIZ burada döner).

    Aynı koşu için önceki token'lar iptal edilir → koşu başına tek geçerli kimlik
    ("tek-koşuluk": token bir koşuya aittir, başka koşuda kullanılamaz).
    """
    if not run_id:
        raise ValueError("run_id boş olamaz")
    token = secrets.token_urlsafe(32)
    now = time.monotonic()
    with _lock:
        _purge_expired_locked(now)
        # Aynı run_id'ye ait eski kayıtları düşür (yeniden mint = eskisini iptal).
        for h in [h for h, rec in _store.items() if rec.run_id == run_id]:
            del _store[h]
        _store[_hash(token)] = _Record(run_id=run_id, expires_at=now + max(1, ttl_s))
    return token


def verify(token: str, *, run_id: str | None = None) -> str | None:
    """Token geçerliyse bağlı olduğu ``run_id``'yi döndür, değilse ``None``.

    ``run_id`` verilirse kayıttakiyle eşleşmelidir (koşu-dışı kullanım reddedilir).

    NOT: doğrulama token'ı TÜKETMEZ. Sürücü token'ı bir *kimlik etiketi*dir, tek
    kullanımlık bir yetki bileti değil — motorun meşru salt-okuma çağrıları (ör.
    /api/healthz) kimliğini kaybettirmemelidir. Tek-kullanımlık tüketim, gerçek
    yetkinin verildiği yerde kalır: ``approvals.require_fresh_approval`` (Kural 8).
    """
    if not token:
        return None
    now = time.monotonic()
    with _lock:
        _purge_expired_locked(now)
        rec = _store.get(_hash(token))
        if rec is None or rec.expires_at <= now:
            return None
        if run_id is not None and not secrets.compare_digest(rec.run_id, run_id):
            return None
        return rec.run_id


def revoke_run(run_id: str) -> int:
    """Bir koşuya ait tüm token'ları iptal et (koşu bitince çağrılır). Silinen sayısı döner."""
    with _lock:
        stale = [h for h, rec in _store.items() if rec.run_id == run_id]
        for h in stale:
            del _store[h]
        return len(stale)


def reset() -> None:
    """Tüm token'ları temizle (yalnız testler için)."""
    with _lock:
        _store.clear()
