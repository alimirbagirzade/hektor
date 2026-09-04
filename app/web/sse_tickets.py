"""sse_tickets.py — kısa ömürlü, TEK-kullanımlık SSE erişim biletleri.

SORUN (P7, 2026-07-21 denetimi): ``GET /api/training/stream`` kimliği query string'den
alıyordu (``?token=<api_token>``) çünkü ``EventSource`` özel HTTP başlığı gönderemez. Ama
sızan şey KISA ÖMÜRLÜ bir sürücü token'ı değil, İNSAN api_token'ının KENDİSİYDİ → erişim/
proxy loglarına, tarayıcı geçmişine düşer ve TTL'i yoktur (kalıcı sır sızıntısı).

ÇÖZÜM: insan bir kez NORMAL auth ile (bearer başlığı) KISA ömürlü, TEK kullanımlık bir
bilet alır; ``EventSource`` yalnız o bileti query'de taşır. Bilet loglara düşse bile
saniyeler içinde ölür ve ikinci kez kullanılamaz — kalıcı sırra kıyasla zarar penceresi
ihmal edilebilir.

Depo süreç-içidir (kalıcı değil): web sunucusu yeniden başlarsa tüm biletler geçersizleşir
(istenen davranış — canlı SSE bağlantısı da kopmuştur). Tüm erişimler ``_lock`` altında;
route'lar threadpool'da koşar.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass

# Bilet varsayılan ömrü: bileti al → hemen EventSource aç arasındaki pencereye yeter.
# Kısa tutulur çünkü loglara düşme riski TTL ile sınırlanır.
DEFAULT_TTL_S = 60


@dataclass(frozen=True)
class _Ticket:
    expires_at: float


# token_sha256 -> _Ticket. Ham bilet SAKLANMAZ (bellek dökümü/log sızıntısı ham bilet vermez).
_store: dict[str, _Ticket] = {}
_lock = threading.Lock()


def _hash(ticket: str) -> str:
    return hashlib.sha256(ticket.encode("utf-8")).hexdigest()


def _purge_expired_locked(now: float) -> None:
    for h in [h for h, rec in _store.items() if rec.expires_at <= now]:
        del _store[h]


def mint(*, ttl_s: int = DEFAULT_TTL_S) -> str:
    """Yeni bir SSE bileti üret (ham bilet YALNIZ burada döner)."""
    ticket = secrets.token_urlsafe(32)
    now = time.monotonic()
    with _lock:
        _purge_expired_locked(now)
        _store[_hash(ticket)] = _Ticket(expires_at=now + max(1, ttl_s))
    return ticket


def consume(ticket: str) -> bool:
    """Bileti TÜKET (tek-kullanımlık): geçerli+süresi dolmamışsa sil ve ``True`` döndür.

    Aynı bilet ikinci kez kullanılamaz (bağlantı açılırken tüketilir). TTL geçmişse ``False``.
    """
    if not ticket:
        return False
    now = time.monotonic()
    with _lock:
        _purge_expired_locked(now)
        h = _hash(ticket)
        rec = _store.get(h)
        if rec is None or rec.expires_at <= now:
            return False
        del _store[h]  # TEK-kullanım: doğrulama = tüketim
        return True


def reset() -> None:
    """Tüm biletleri temizle (yalnız testler için)."""
    with _lock:
        _store.clear()
