"""Web güvenlik yardımcıları (FastAPI).

Tehdit modeli: bu yerel-öncelikli bir araştırma aracıdır. Sunucu varsayılan
olarak yalnız ``127.0.0.1``'e bağlanır. Yine de savunmayı katmanlı tutarız:

- İsteğe bağlı bearer-token kimlik doğrulama (ağa açılırsa zorunlu hale gelir).
- IP başına basit hız sınırı (bellekte, bağımlılık gerektirmez).
- Güvenlik başlıkları (CSP, X-Frame-Options, vb.).
- Katı PDF upload doğrulaması (uzantı + sihirli bayt + boyut + dosya adı temizleme).
- Yol-aşımı (path traversal) koruması.

Burada kullanıcı girdisi ASLA çalıştırılmaz; strateji kuralları yalnız güvenli
regex ile ayrıştırılır (bkz. trading/strategy_ir.py).
"""

from __future__ import annotations

import logging
import re
import secrets
import time
import unicodedata
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, Request, status

from app.config import get_settings
from app.web import driver_scope

log = logging.getLogger(__name__)

# --- Sabitler ---
_PDF_MAGIC = b"%PDF-"
_ALLOWED_UPLOAD_SUFFIX = ".pdf"
_ALLOWED_CSV_SUFFIX = ".csv"
_CSV_REQUIRED_COLS = ("open", "high", "low", "close")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Güvenlik başlıkları. CSP yalnız kendi kaynaklarımıza izin verir (fontlar self-host).
SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "font-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    ),
}


# --- Kimlik doğrulama ---
def require_auth(request: Request) -> None:
    """API token ayarlıysa geçerli bir bearer token **ya da** sürücü token'ı zorunludur.

    Token boşsa (varsayılan yerel mod) doğrulama atlanır — sunucu zaten yalnız
    localhost'a bağlıdır. Ağa açmak için ACHILLES_API_TOKEN ata.

    SÜRÜCÜ KİMLİĞİ (bkz. driver_scope + docs/SCOPE_ISOLATION.md): doğurulan motorun
    ``ACHILLES_API_TOKEN``'ı bilinçli olarak BOŞALTILIR (insan sırrı çocuğa sızmasın —
    ``driver.py:_HUMAN_SECRET_ENV``). Bu kapı yalnız insan sırrını tanısaydı, api_token
    ayarlı kurulumda motor HER çağrıda 401 alır ve MCP üzerinden hiçbir iş yapamazdı.
    Bu yüzden geçerli bir sürücü token'ı da **kimlik** olarak kabul edilir.

    ⚠️ BU BİR YETKİ DEĞİL, YALNIZ KİMLİKTİR: sürücü kimliğiyle geçen istek
    ``require_human`` kapısında yine **403** alır (Kural 8) — motor onay veremez,
    eğitim başlatamaz. Geçersiz/süresi dolmuş sürücü token'ı ``resolve_scope``
    tarafından **401** ile reddedilir; sessizce ``human``'a düşme YOKTUR.
    """
    settings = get_settings()
    token = settings.api_token.strip()
    if not token:
        return

    header = request.headers.get("authorization", "")
    provided = ""
    if header.lower().startswith("bearer "):
        provided = header[7:].strip()
    else:
        provided = request.headers.get("x-api-token", "").strip()

    # sabit-zamanlı karşılaştırma (stdlib; encode ile non-ASCII token güvenli)
    if secrets.compare_digest(provided.encode("utf-8"), token.encode("utf-8")):
        return

    # İnsan sırrı eşleşmedi → geçerli bir sürücü kimliği var mı?
    # `resolve_scope` başlık VARSA doğrular (geçersizse kendisi 401 atar), başlık
    # YOKSA "human" döner → aşağıdaki 401'e düşer. Yani bu dal yalnız DOĞRULANMIŞ
    # sürücü token'ını geçirir.
    if resolve_scope(request) == "driver":
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Geçersiz veya eksik API token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# --- Scope (kimlik seviyesi): human vs driver ---
Scope = Literal["human", "driver"]


def resolve_scope(request: Request) -> Scope:
    """İsteğin kimlik seviyesini çöz: ``"human"`` (UI/CLI) veya ``"driver"`` (motor).

    - ``X-Achilles-Driver-Token`` başlığı VARSA doğrulanır; geçersiz/süresi dolmuş ya
      da ``X-Achilles-Run-Id`` ile eşleşmiyorsa **401** atılır — sessizce ``human``'a
      DÜŞÜLMEZ (aksi halde geçersiz token göndermek yetki yükseltmesi olurdu).
    - Başlık yoksa ``human``.
    """
    raw = request.headers.get(driver_scope.DRIVER_TOKEN_HEADER, "").strip()
    if not raw:
        return "human"

    claimed_run = request.headers.get(driver_scope.RUN_ID_HEADER, "").strip() or None
    if driver_scope.verify(raw, run_id=claimed_run) is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Geçersiz, süresi dolmuş veya koşuyla eşleşmeyen sürücü token'ı. "
                "Sürücü token'ı yalnız bağlı olduğu run_id ile kullanılabilir."
            ),
        )
    return "driver"


def require_human(request: Request) -> None:
    """İnsan-yalnız uçlar için kapı: ``driver`` scope **403** alır (CLAUDE.md Kural 8).

    Achilles kendi motorunu doğurduğu için (``app/orchestration/driver.py``), motorun
    kendi eğitimini onaylaması / kill-switch'i temizlemesi engellenmelidir. Bu kapı
    yetki kararlarını insana saklar.
    """
    if resolve_scope(request) == "driver":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Bu uç yalnız insan (human) scope'una açıktır. Sürücü (driver) motoru "
                "onay veremez, STOP_ALL temizleyemez ve eğitim başlatamaz (Kural 8)."
            ),
        )


def warn_if_auth_disabled() -> bool:
    """``api_token`` boşsa GÜRÜLTÜLÜ uyarı logla. Auth açıksa ``True`` döner.

    Sessiz "auth kapalı" durumu, sürücü izolasyonunun neden kriptografik bir sınır
    olmadığının da temel sebebidir → başlangıçta açıkça görünür olmalı.
    """
    if get_settings().api_token.strip():
        return True
    log.warning(
        "GÜVENLİK: ACHILLES_API_TOKEN BOŞ — API kimlik doğrulaması KAPALI. "
        "Sunucuya erişebilen her yerel süreç insan yetkisiyle istek atabilir; "
        "sürücü (driver) scope izolasyonu bu modda yalnız DERİNLEMESİNE SAVUNMADIR, "
        "kriptografik sınır DEĞİLDİR. Gerçek sınır için ACHILLES_API_TOKEN ata."
    )
    return False


# --- Hız sınırı (bellekte, IP başına kayan pencere) ---
class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._last_sweep: float = 0.0

    def check(self, client_ip: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        window_start = now - 60.0
        # Periyodik süpürme: penceresi tamamen boşalmış IP girdilerini sil. Aksi halde ağa
        # açık bir örnekte saldırgan kaynak IP'sini döndürerek (her istek farklı IP) _hits
        # sözlüğüne sınırsız anahtar ekletir (yavaş bellek sızıntısı). En çok dakikada bir,
        # son-isabeti pencere dışına düşmüş IP'leri temizle (O(n) ama nadiren çalışır).
        if now - self._last_sweep > 60.0:
            self._last_sweep = now
            stale = [ip for ip, d in self._hits.items() if not d or d[-1] < window_start]
            for ip in stale:
                del self._hits[ip]
        dq = self._hits[client_ip]
        while dq and dq[0] < window_start:
            dq.popleft()
        if len(dq) >= self.per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Hız sınırı aşıldı; biraz sonra tekrar dene.",
            )
        dq.append(now)


def client_ip(request: Request) -> str:
    # Yerel araç: proxy başlıklarına güvenmeyiz (spoof edilebilir).
    return request.client.host if request.client else "unknown"


# --- PDF upload doğrulama ---
def sanitize_filename(name: str, *, suffix: str = ".pdf", fallback: str = "paper") -> str:
    """Dosya adını güvenli bir tabana indirger (yol-aşımı / kontrol karakteri yok)."""
    name = unicodedata.normalize("NFKD", name)
    stem = Path(name).name  # dizin bileşenlerini at
    stem = _SAFE_NAME_RE.sub("_", stem).strip("._-")
    if not stem:
        stem = fallback
    if not stem.lower().endswith(suffix):
        stem = f"{stem}{suffix}"
    return stem[:128]


def validate_pdf_upload(filename: str, content: bytes) -> str:
    """Uzantı + sihirli bayt + boyut doğrular; temiz dosya adını döndürür.

    Hata durumunda HTTPException (400/413) fırlatır.
    """
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024

    if not filename or not filename.lower().endswith(_ALLOWED_UPLOAD_SUFFIX):
        raise HTTPException(status_code=400, detail="Yalnız .pdf dosyaları kabul edilir.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Boş dosya.")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Dosya çok büyük (> {settings.max_upload_mb} MB).",
        )
    if not content.startswith(_PDF_MAGIC):
        raise HTTPException(status_code=400, detail="Dosya içeriği geçerli bir PDF değil.")
    return sanitize_filename(filename)


def validate_csv_upload(filename: str, content: bytes) -> str:
    """OHLCV CSV upload'ını doğrular (uzantı + boyut + metin + başlık sniff).

    Başlık satırında open/high/low/close kolonlarını arar (load_ohlcv'nin
    zorunlu kıldıkları). Hatada HTTPException (400/413) fırlatır; temiz adı döner.
    """
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024

    if not filename or not filename.lower().endswith(_ALLOWED_CSV_SUFFIX):
        raise HTTPException(status_code=400, detail="Yalnız .csv dosyaları kabul edilir.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Boş dosya.")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Dosya çok büyük (> {settings.max_upload_mb} MB)."
        )
    try:
        head = content[:8192].decode("utf-8")
    except UnicodeDecodeError:
        try:
            head = content[:8192].decode("latin-1")
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="CSV metin olarak çözülemedi (ikili dosya?)."
            ) from exc
    first_line = next((ln for ln in head.splitlines() if ln.strip()), "").lower()
    if not all(col in first_line for col in _CSV_REQUIRED_COLS):
        raise HTTPException(
            status_code=400,
            detail="CSV başlığında open/high/low/close kolonları bulunamadı.",
        )
    return sanitize_filename(filename, suffix=".csv", fallback="ohlcv")


def safe_destination(dest_dir: Path, filename: str) -> Path:
    """Hedefin dest_dir içinde kaldığını garanti eder (path traversal koruması)."""
    dest_dir = dest_dir.resolve()
    candidate = (dest_dir / filename).resolve()
    if dest_dir not in candidate.parents and candidate.parent != dest_dir:
        raise HTTPException(status_code=400, detail="Geçersiz hedef yol.")
    return candidate
