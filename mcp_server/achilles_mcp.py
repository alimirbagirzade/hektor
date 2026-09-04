"""Achilles Web MCP sunucusu — web API'sini MCP tool'larına çevirir.

Tasarım (otomatik senkron):
- OpenAPI spec'i Achilles FastAPI uygulamasından IN-PROCESS üretilir
  (`app.web.server.app.openapi()`), böylece web'e her yeni route eklendiğinde
  MCP yeniden başlatıldığında otomatik yansır — elle güncelleme gerekmez.
- Tool çağrıları ise ÇALIŞAN web sunucusuna (http://127.0.0.1:8765) httpx ile
  proxy'lenir; uygulama MCP içinde yeniden init edilmez, SQLite kilit çakışması olmaz.

Gereksinim: ``fastmcp`` — opsiyonel ``mcp`` extra'sındadır::

    uv sync --extra mcp

Bu extra kurulmadan aşağıdaki komutlar ``ModuleNotFoundError`` ile düşer.

Çalıştırma (stdio MCP):
    uv run python mcp_server/achilles_mcp.py

Kayıt:
    claude mcp add achilles -- uv run --project <repo> python mcp_server/achilles_mcp.py

Senkron protokolü: web (app/web/server.py) değişince → MCP'yi yeniden başlat
(veya Claude Code'u). Spec her başlangıçta taze üretildiği için tool listesi güncellenir.
Ayrıntı: docs/PROTOKOL_MCP.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo kökünü path'e ekle (script doğrudan çalıştırılabilsin)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BASE_URL = os.environ.get("ACHILLES_WEB_URL", "http://127.0.0.1:8765")

# Sürücü kimliği başlıkları — app/web/driver_scope.py ile AYNI adlar (tek kaynak orada).
# Buraya elle yazılır çünkü bu modül `app` paketi kurulu olmadan da içe aktarılabilmelidir.
_DRIVER_TOKEN_ENV = "ACHILLES_DRIVER_TOKEN"
_DRIVER_RUN_ID_ENV = "ACHILLES_DRIVER_RUN_ID"
_DRIVER_TOKEN_HEADER = "x-achilles-driver-token"
_RUN_ID_HEADER = "x-achilles-run-id"


def driver_headers(env: dict[str, str] | None = None) -> dict[str, str]:
    """Ortamdaki sürücü kimliğini HTTP başlıklarına çevir (yoksa BOŞ sözlük).

    ⚠️ GÜVENLİK — KİMLİK AKLAMA (identity laundering) ÖNLEMİ:
    Bu sunucu MCP çağrılarını çalışan web'e httpx ile proxy'ler. Başlıklar
    taşınmazsa, doğurulan motorun ``driver`` kimliği proxy'de KAYBOLUR ve istek
    sunucuya ``human`` gibi ulaşır (``app/web/security.py:resolve_scope`` başlık
    yoksa ``"human"`` döner). O durumda motor, MCP üzerinden ``/api/training/run``
    ve ``/api/approvals/{id}/approve`` gibi ``human_only`` uçları çağırıp KENDİ
    eğitimini onaylayıp başlatabilirdi → CLAUDE.md Kural 8 delinir.

    İnsan bu MCP sunucusunu kendi oturumunda kullandığında bu değişkenler ortamda
    YOKTUR → başlık eklenmez → davranış eskisiyle birebir aynı (``human``).
    """
    src = os.environ if env is None else env
    token = (src.get(_DRIVER_TOKEN_ENV) or "").strip()
    run_id = (src.get(_DRIVER_RUN_ID_ENV) or "").strip()
    if not token:
        return {}
    headers = {_DRIVER_TOKEN_HEADER: token}
    if run_id:
        # run_id da gönderilir: token yalnız bağlı olduğu koşuda geçerlidir.
        headers[_RUN_ID_HEADER] = run_id
    return headers


def auth_headers() -> dict[str, str]:
    """Web API'sine gidecek kimlik başlıkları.

    ``ACHILLES_API_TOKEN`` ayarlıysa proxy istekleri ``Authorization: Bearer ...``
    ile imzalanır. Bu olmadan token açıkken TÜM MCP tool çağrıları 401 alırdı →
    "token aç, MCP kırılsın / MCP çalışsın, kapı açık kalsın" kısır döngüsü.
    Token boşsa (varsayılan yerel mod) başlık gönderilmez; web tarafı da doğrulamaz.

    Ayar `.env` üzerinden de gelebildiği için önce ayarlar, sonra ham env okunur.
    """
    token = ""
    try:
        from app.config import get_settings

        token = get_settings().api_token.strip()
    except Exception:  # ayar katmanı yüklenemezse ham env'e düş
        token = ""
    if not token:
        token = os.environ.get("ACHILLES_API_TOKEN", "").strip()
    if token:
        _warn_if_token_leaves_loopback()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _warn_if_token_leaves_loopback() -> None:
    """``ACHILLES_WEB_URL`` loopback dışıysa stderr'e uyar (token o host'a gider).

    Sert hata DEĞİL bilinçli olarak: token ayarlamanın amacı zaten ağa açmaktır ve
    web sunucusu meşru biçimde başka bir makinede olabilir. Ama bearer token'ın
    yabancı bir host'a gideceği sessiz kalmamalı — stdio MCP'de stderr güvenlidir
    (protokol stdout'u kullanır).
    """
    from urllib.parse import urlparse

    host = (urlparse(BASE_URL).hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1", "[::1]"}:
        print(
            f"GÜVENLİK UYARISI: ACHILLES_WEB_URL loopback değil ({BASE_URL}) — "
            "API token'ı bu host'a gönderilecek. Kasıtlı değilse ACHILLES_WEB_URL'i düzelt.",
            file=sys.stderr,
        )


def build_mcp():
    """Achilles OpenAPI'sinden FastMCP sunucusu kur (proxy → çalışan web).

    Spec, FastMCP'ye verilmeden ÖNCE ``allowlist.filter_spec`` ile budanır:
    yalnız açıkça izin verilen salt-okuma uçları tool olur (varsayılan kapalı).
    """
    import httpx
    from fastmcp import FastMCP
    from mcp_server.allowlist import filter_spec

    from app.web.server import app as achilles_app

    spec = filter_spec(achilles_app.openapi())  # varsayılan-kapalı budama
    # Kimlik başlıkları: insan bearer token'ı + (varsa) sürücü kimliği.
    # İkisi ÇAKIŞMAZ (farklı başlık adları) ve birlikte doğru davranırlar: bearer
    # `require_auth`'u geçirir, sürücü başlığı ise `require_human` kapısında 403'e yol
    # açar (Kural 8). Yani sürücü kimliği, insan sırrı sızsa BİLE yetki vermez.
    client = httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=120.0,
        headers={**auth_headers(), **driver_headers()},
    )
    return FastMCP.from_openapi(
        openapi_spec=spec,
        client=client,
        name="achilles-web",
    )


def __getattr__(name: str):
    """``mcp`` niteliğini TEMBEL kur (modül import'u yan etkisiz kalsın).

    Önceden modül seviyesinde ``mcp = build_mcp()`` vardı: modülü sadece import etmek
    tüm FastMCP sunucusunu kuruyor ve ``fastmcp``'yi zorunlu kılıyordu. Bu yüzden
    ``auth_headers``'ı test etmek bile opsiyonel ``mcp`` extra'sını gerektiriyordu
    (CI ``--extra dev`` ile kurduğu için `ModuleNotFoundError: fastmcp` veriyordu).

    ``mcp`` niteliği hâlâ erişilebilir (``from mcp_server.achilles_mcp import mcp``)
    — yalnız ilk erişimde kurulur. PEP 562 modül düzeyi ``__getattr__``.
    """
    if name == "mcp":
        return build_mcp()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    build_mcp().run()
