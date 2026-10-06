# Motor Bağlama — Claude Code'u Hektor web arayüzüne bağlamak

_2026-10-06 · Eski `ROADMAP_MOTOR_BAGLAMA.md`'nin (P1-P9 paket planı + kopyala-yapıştır
prompt'lar; hepsi 2026-07-22'de kapandı) yerine geçer. Tarihçe git geçmişindedir._

Hektor iki yerde yerel **abonelikli** `claude` CLI'sini alt süreç olarak doğurur:

| Yüzey | Ne yapar | Kod |
|---|---|---|
| ⚡ RUN / Otonom AV (12·ORKESTRASYON, 15·AJAN HARİTASI) | `claude -p` ile veri hattını sürer (sür modu, MCP'li) ya da derin av yapar (av modu, `--safe-mode`) | `app/orchestration/driver.py`, `engines.py` |
| İkinci görüş (Faz 3, varsayılan KAPALI) | Tur başına, insan tıklamasıyla tek `claude -p` çağrısı | `app/cloud/providers.py` |

**API anahtarı yok.** Hektor kimlik bilgisi toplamaz/saklamaz; motor kullanıcının kendi CLI
oturumunu kullanır. `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` alt sürece geçirilmez.

## "Bağlanamıyor" — iki ayrı sebep

### 1. CLI bulunamıyor → motor "kurulu değil" görünür

Eskiden Hektor `claude`'u **yalnız PATH'te** arıyordu. Claude masaüstü uygulamasıyla kurulan
makinelerde CLI PATH'te değildir; uygulamanın kendi klasöründedir. Sonuç: motor seçicide
"kurulu değil", ⚡ RUN gri, ikinci görüş "`claude` CLI güvenilir PATH dizinlerinde yok".

**Düzeltme (`app/orchestration/executable.py::resolve_cli`):** önce PATH (çalışma dizini
hariç — taklitçi ikili koruması), sonra bilinen konumlar, sırasıyla:

1. `HEKTOR_CLAUDE_BIN` — açık geçersiz kılma (mutlak yol).
2. `~/.local/bin/claude[.exe]` (native kurulum), `~/.claude/local/claude[.exe]`.
3. Masaüstü uygulamasının indirdiği CLI: `%APPDATA%\Claude\claude-code\<sürüm>\<hash>\claude.exe`
   (macOS: `~/Library/Application Support/Claude/claude-code/...`). Yalnız yanında `.verified`
   olan (uygulamanın bütünlük denetiminden geçmiş) kopya; en yeni sürüm önce.
4. Windows MSIX (Store) paketi `%APPDATA%`'yı sanallaştırır; uygulama dışından başlatılan
   `hektor-web` kopyayı `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\claude-code\`
   altında görür — o da taranır.

`.cmd/.bat` sarmalayıcı ikinci görüşte hâlâ reddedilir (cmd.exe argüman yeniden yorumlar).

### 2. CLI bulundu ama oturum açık değil

Masaüstü uygulaması kimliği **kendi içinde** taşır; aynı makinedeki bağımsız `claude` CLI'si
ayrıca giriş ister. Girişsiz çağrı `Not logged in · Please run /login` ile çıkış 1 verir.
Eskiden Hektor yalnız stderr'i gösterdiği için bu sebep gizleniyor, yerine ilgisiz bir
PowerShell uyarısı görünüyordu; artık açık Türkçe mesaj döner.

**Tek seferlik adım (insan yapar — Hektor giriş yapmaz):**

```powershell
& (uv run python -c "from app.orchestration.executable import resolve_cli; print(resolve_cli('claude'))")
```

Açılan oturumda `/login` → abonelik hesabı → çık. Doğrulama:
`<yol> auth status` çıktısında `"loggedIn": true`.

Giriş durumu web'de **yoklanmaz** (`logged_in` daima `null`): yoklamak için motoru doğurmak
gerekir — kota yakar ve salt-okuma sözleşmesini bozar (Kural 7: bilinmiyorsa "bilinmiyor").

## Teşhis listesi

```bash
uv run python -c "from app.orchestration import engines; print(engines.describe('claude'))"
```

| Belirti | Sebep | Çözüm |
|---|---|---|
| `installed: False` | CLI hiçbir bilinen konumda yok | Claude Code'u kur ya da `HEKTOR_CLAUDE_BIN` ver |
| "oturum açık değil" | Bağımsız CLI girişsiz | Yukarıdaki tek seferlik `/login` |
| `claude çıkış 1: ...` (başka) | CLI hatası | Mesajın son 300 karakteri gösterilir |

## Sertleştirme (değişmeyen sözleşme)

- **Av modu / ikinci görüş:** `--safe-mode --strict-mcp-config --disallowedTools
  Bash,PowerShell,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task` (ikinci görüş ayrıca
  Read/Grep/Glob'u kapatır, boş geçici cwd'de koşar). `PowerShell` açıkça yasak: Bash yasağı
  onu yalnız `CLAUDE_CODE_USE_POWERSHELL_TOOL` yoksa kapatır.
- **Sür modu:** `--safe-mode` kullanılamaz (MCP'yi kapatır); `--setting-sources ""
  --disable-slash-commands --strict-mcp-config --tools Read,Grep,Glob --mcp-config <yol>`.
  `--bare` yasak (kimliği API anahtarına indirger).
- **Kimlik ayrımı:** motor `driver` scope'lu kısa ömürlü token alır; onay ve STOP_ALL temizleme
  uçları ona 403 döner (`docs/SCOPE_ISOLATION.md`). Gerçek eğitim taze insan onayında durur
  (Kural 8).
- **Ortam süzgeci:** `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `HEKTOR_API_TOKEN` ve
  ayar-ezme yolları (`CLAUDE_CODE_*_SETTINGS_PATH`) alt sürece geçmez.
- **Bağımsız verdict:** av PASS'i yalnız dosya sistemiyle doğrulanan kanıt + okuma-kanıtı
  (`{path, line, quote}`) ile kabul edilir (`app/orchestration/verdict_audit.py`).

## Testler

`tests/test_executable_resolve.py` (bilinen konum taraması, `.verified` şartı, sürüm sırası,
PATH önceliği), `tests/test_cloud_second_opinion.py` (girişsiz CLI mesajı), `tests/test_engines*.py`,
`tests/test_scope_isolation.py`. Hepsi çevrimdışı; gerçek `claude` doğurmaz.
