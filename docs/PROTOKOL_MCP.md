# MCP SENKRON PROTOKOLÜ — Achilles Web → MCP

> Achilles web API'sinin **salt-okuma alt kümesini** MCP tool'ları olarak dışarı açar;
> başka Claude oturumları / araçlar Achilles ile **tool** üzerinden konuşabilir.
> Sunucu: `mcp_server/achilles_mcp.py` · Skill: `/achilles-web`

## Nasıl çalışır (otomatik senkron + allow-list)

```
app/web/server.py (FastAPI, 116 operasyon)
        │  app.openapi()  (IN-PROCESS, her başlangıçta taze)
        ▼
mcp_server/allowlist.py  ──filter_spec (VARSAYILAN KAPALI)──►  21 operasyon
        ▼
mcp_server/achilles_mcp.py  ──FastMCP.from_openapi──►  21 MCP tool
        │  httpx proxy + Authorization: Bearer <ACHILLES_API_TOKEN>
        ▼
çalışan web sunucusu (http://127.0.0.1:8765)  ──►  gerçek iş
```

- **Spec in-process üretilir** (`achilles_app.openapi()`) — web'e yeni route
  eklendiğinde MCP **yeniden başlatıldığında** tool listesi otomatik güncellenir.
- **Çağrılar çalışan web'e proxy'lenir** — MCP içinde uygulama yeniden init edilmez,
  SQLite kilit çakışması olmaz. (Web sunucusu açık olmalı: `uv run achilles-web`.)

## Tool yüzeyi: varsayılan kapalı allow-list

`FastMCP.from_openapi()` kendisine verilen spec'teki HER operasyonu tool yapar. Achilles
116 operasyon sunar; çoğu dış bir ajanın görmemesi gereken yazma/tetikleme ucudur. Bu
yüzden spec, FastMCP'ye verilmeden **önce** `mcp_server/allowlist.py` ile budanır.

- **Tek kaynak:** izin listesi yalnız `allowlist.ALLOWED` içinde durur.
- **Varsayılan kapalı:** listede olmayan hiçbir uç tool olmaz. Web'e yeni route
  eklemek MCP yüzeyini **otomatik genişletmez** (bilinçli tercih).
- **Fail-closed:** `FORBIDDEN_SUBSTRINGS` ile çakışan bir giriş eklenirse ya da
  allow-list'teki bir uç spec'te bulunamazsa MCP sunucusu `AllowlistError` ile
  **kurulmaz** (sessizce yanlış yüzey açılmaz).

**Neden `route_maps` değil:** FastMCP'nin `route_maps`'i desen-sıralı ve *varsayılan
açık* çalışır — hiçbir desene uymayan yeni route tool olur. Spec budama tersini garanti
eder ve FastMCP sürüm/API değişimlerinden bağımsızdır.

Yüzeyi genişletmek için: `ALLOWED`'a **açıkça** ekle + `tests/test_mcp_allowlist.py`
yeşil kalsın. Yazma/tetikleme uçları (onay, eğitim, kill-switch, autodrive) **eklenmez**.

## Kimlik doğrulama (token kısır döngüsü çözüldü)

`ACHILLES_API_TOKEN` ayarlıysa proxy istemcisi istekleri `Authorization: Bearer ...`
ile imzalar (`achilles_mcp.auth_headers()`). Bu olmadan token açıkken **tüm** MCP tool
çağrıları 401 alırdı → "token aç, MCP kırılsın / MCP çalışsın, kapı açık kalsın".

⚠️ **Sınırın nerede olduğunu karıştırma.** MCP proxy'si token'ı **insan (human)**
scope'uyla taşır — sürücü (driver) scope'uyla değil. Yani MCP üzerinden erişen bir ajanı
`require_human` kapıları **durdurmaz**. Gerçek sınır **allow-list'in kendisidir**:
onay/eğitim/kill-switch uçları tool olarak hiç sunulmaz. Bu yüzden allow-list'e yazma
ucu eklemek, Kural 8'i doğrudan delmek demektir.

Token boşsa (varsayılan yerel mod) başlık gönderilmez ve web tarafı da doğrulamaz.

## Keşif yüzeyi: `/api/docs` + `/api/openapi.json`

Bu ikisi FastAPI'nin yerleşik uçlarıdır ve route-başına `api_auth` **almazlar** →
kimlik doğrulamasızdırlar. Karar: **token ayarlıysa kapatılır** (404), boşsa açık kalır.

- Token atamak ağa açma niyetidir; o modda kimliksiz tam route/şema envanteri
  gereksiz keşif yüzeyidir.
- Yerel varsayılan modda geliştirici kolaylığı korunur.
- **Kırılma yok** (doğrulandı): web arayüzü bu uçlara hiç başvurmaz, MCP spec'i
  HTTP'den değil in-process üretir.
- Bu bir **derinlemesine savunma**dır, erişim kontrolü değil — uçların kendisi zaten
  `api_auth` ile korunur; şema gizlemek korumanın yerine geçmez.

## 2026-07-28 MCP spec geçişi — durum tespiti (2026-07-21)

⚠️ **Spec HENÜZ YAYINLANMADI.** Yürürlükteki sürüm **2025-11-25**. `2026-07-28`
şu an **Release Candidate** (2026-05-21'de donduruldu), yayın tarihi 2026-07-28.

**Kurulu sürümlerimiz:** `fastmcp 3.4.4` + `mcp 1.28.1` →
`LATEST_PROTOCOL_VERSION = "2025-11-25"`. FastMCP kaynağında hiçbir `2026-*` referansı
yok; yani **2025-11-25 dönemi** bir implementasyondayız.

**Bu kurulum RC'den etkilenmiyor:**
- stdio taşıması + durumsuz OpenAPI proxy'si kullanıyoruz; `Mcp-Session-Id`'ye,
  session deposuna veya SSE akışına bağlı hiçbir şey yok — `from_openapi` zaten
  istek/yanıt proxy'sidir, tam da stateless spec'in istediği şekil.
- Roots / Sampling / Logging **kullanmıyoruz** (deprecate edilseler de en erken
  2027-07-28'de kalkıyorlar — ≥12 ay geçiş süresi var).

**Brief'teki iki varsayımın düzeltmesi:**
- *"SSE ölüyor"* → SSE taşıması zaten **2025-03-26**'da deprecate edilmişti; bu
  revizyonun işi değil. 2026-07-28'in yaptığı, Streamable HTTP **içindeki** SSE
  akışlarını durumsuz `InputRequiredResult` ile değiştirmek (SEP-2322).
- *"Roots/Sampling/Logging deprecation"* → doğru, ama **kaldırma değil**; yalnız
  anotasyon (SEP-2577), 12+ ay runway.

**Kural (değişmedi): session tabanlı YENİ bağımlılık eklenmeyecek.**

`fastmcp` 3.4.x kendi 2.x→3.x sağlayıcı göçü için deprecation uyarıları veriyor
(`fastmcp.server.openapi` → `fastmcp.server.providers.openapi`) — bunlar spec ile
İLGİSİZ. Yeniden değerlendirme: spec yayınlandıktan ve FastMCP `mcp`'yi 1.28.x'in
üstüne çektikten sonra.

## Senkron kuralı (ZORUNLU)

**`app/web/server.py` her değiştiğinde → MCP yeniden başlatılmalı** ki yeni/değişen
endpoint'ler tool olarak yansısın. Elle spec güncellemeye gerek YOK (in-process üretilir).

```bash
# Yeniden senkronla (web değişikliğinden sonra):
bash scripts/sync-mcp.sh
```

> CLAUDE.md doğrulama akışına ek madde: `app/web/server.py` değiştiyse
> `make test`'ten sonra `bash scripts/sync-mcp.sh` çalıştır (MCP'yi tazele).

## Kurulum / kayıt

```bash
claude mcp add achilles -- uv run --project <repo-yolu> python mcp_server/achilles_mcp.py
claude mcp list            # 'achilles' bağlı mı kontrol
```

## Kullanım
Kayıt sonrası `achilles-*` tool'ları erişilebilir olur (örn. `api_status`,
`api_ask`, `api_rag_mastery`, `api_synthesis_reports`...). Detaylı kullanım: `/achilles-web` skill.

## Doğrulama
```bash
uv run python -c "import asyncio; from mcp_server.achilles_mcp import mcp; \
import asyncio as a; print(a.run(mcp.list_tools().__await__().__next__) if False else len(a.run(mcp.list_tools())))"
# beklenen: 68 (web route sayısıyla eşleşmeli)
```
