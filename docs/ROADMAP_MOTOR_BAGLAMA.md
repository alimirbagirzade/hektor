# ROADMAP — Yerel Motor Bağlama & Tek-Tık RUN

_Oluşturma: 2026-07-21 · Kaynak oturum: `claude/local-system-research-ed8535` (araştırma, kod yazılmadı)_

## Amaç

Kullanıcı Achilles'i yerelde açar, **RUN**'a basar; makinede kurulu ve **aboneliğiyle
girişli** bir motor (Claude Code / Codex / Gemini CLI) alt-süreç olarak doğar, MCP
üzerinden Achilles'in ajanlarını sürer. Eğitim, Kural-8 taze insan onayında durur.

**API anahtarı YOK** — motorlar kendi CLI oturumlarını kullanır. Achilles hiçbir
kimlik bilgisi toplamaz, saklamaz, göstermez.

## Mevcut durum (2026-07-21 taraması)

> ✅ **GÜNCELLEME (2026-07-22):** Aşağıdaki tablo 2026-07-21 anının fotoğrafıdır. O günden
> beri **P1-P9 KAPANDI**: onay/kill-switch izolasyonu (P1), MCP allow-list + token iletimi
> (P4), sür-modu prompt + MCP geçişi (P3 yazıldı, **P7'de fişe takıldı**) ve ⚡ RUN (artık
> dry-run değil — **sür modu**, P5-P7) tamam. Bu tablodaki "❌ yok / 🟡 dry-run" satırları
> ARTIK GEÇERLİ DEĞİL. Güncel durum için "FAZ 2 → Doğrulanmış durum" ve "FAZ 3" bölümlerine bak.

| Parça | Durum | Konum |
|---|---|---|
| MCP sunucusu (OpenAPI→tool proxy) | ✅ var | `mcp_server/achilles_mcp.py` |
| `claude -p` alt-süreç sürücüsü | ✅ var, salt-rapor | `app/orchestration/driver.py:37,56,78` |
| Autodrive ucu | ✅ var, `execute=false` kilidi | `app/web/orchestration_routes.py:132` |
| Onay kapısı + taze onay TTL | ✅ var | `app/agents/runtime/approvals.py:162` |
| STOP_ALL kill-switch | ✅ var | `app/agents/runtime/supervisor.py:70` |
| Çok-sağlayıcılı LLM katmanı | ✅ var | `app/brain/local_llm.py:27` |
| ⚡ RUN butonu | 🟡 dry-run | 15·AJAN HARİTASI sekmesi |
| **Onay/kill-switch izolasyonu** | ❌ **YOK — bloklayıcı** | aşağıda P1 |
| Motor kayıt tablosu (çok motor) | ✅ **P2 TAMAM** (PR #112) | `app/orchestration/engines.py` |
| Sür-modu prompt | ❌ yok (yalnız av-modu) | P3 |
| MCP allow-list + token iletimi | ❌ yok | P4 |

## 🔴 Bloklayıcı güvenlik açığı (P1'in gerekçesi)

Bugün RUN açılırsa: doğan motor, Achilles API'sine **insanla aynı yetkiyle** erişir.
Yani kendi eğitimini kendisi onaylayabilir (`POST /api/approvals/{id}/approve`) ve
kill-switch'i temizleyebilir (`POST /api/supervisor/clear-stop-all`).
Üstelik `api_token` varsayılan boş → doğrulama tümüyle atlanıyor
(`app/web/security.py:66`). **Kural-8 bu kurulumda kâğıt üstünde kalır.**

Çözüm: iki kimlik. Sürücü motoruna verilen kimlik onay/stop-all uçlarını *görmez*;
onay yalnız insan yüzeyinden (UI / CLI) gelir.

---

## Faz planı ve paralellik

```
        ┌──────────────── ŞERİT A (güvenlik) ────────────────┐
Faz 0 → │ P1 onay izolasyonu → P4 MCP allow-list + token     │ ┐
        └────────────────────────────────────────────────────┘ ├→ P5 UI/RUN → P6 kapanış
        ┌──────────────── ŞERİT B (motor) ───────────────────┐ │
        │ P2 motor tablosu → P3 sür-modu prompt + MCP geçişi │ ┘
        └────────────────────────────────────────────────────┘
```

- **P1 ve P2 aynı anda başlatılabilir** (farklı dosyalar, çakışma yok).
- **P4, P1'i bekler** (token modeli netleşmeden allow-list yazılamaz).
- **P3, P2'yi bekler.**
- **P5 birleşme noktası** — A ve B şeridinin ikisi de bitmeden başlamaz.
- **P6 kapanış** — hepsinden sonra, tek başına.

Tahmini net kod: ~150-200 satır. Yeni bağımlılık yok, yeni protokol yok.

---

## Her pakette geçerli ortak kurallar

Her prompt'un başına şunlar zaten gömülü, ama insan gözüyle de bil:

1. **Gerçek worktree kur** — `git rev-parse --show-toplevel` cwd'yi vermiyorsa
   ana repo'da çalışıyorsundur, DUR. Bkz `HANDOFF.md` worktree hazard bölümü.
2. **`uv sync --extra dev`** — pytest/ruff/mypy `dev` extra'sında.
3. **Kapı zorunlu:** `make format && make lint && make typecheck && make test`
4. **PR aç, CI yeşilse kendin merge et** (`gh pr merge --merge --delete-branch`).
5. **`git add -A` ASLA** — eşzamanlı oturumların WIP'ini süpürür. Dar `git add <dosya>`.
6. **Kural-8:** hiçbir paket gerçek eğitim başlatmaz.

---

# PAKETLER

## P1 — Onay & kill-switch izolasyonu 🔴 BLOKLAYICI

**Şerit:** A · **Bağımlılık:** yok · **Paralel:** P2 ile aynı anda başlatılabilir

```
Achilles'te sürücü-motor ile insan yetkisini ayıran "scope" katmanını tasarla ve uygula.

ÖNCE OKU: HANDOFF.md worktree hazard bölümü. `git rev-parse --show-toplevel` cwd'ni
vermiyorsa gerçek worktree kur:
  git worktree add "<ayrı-dizin>" -b claude/scope-isolation origin/main
Orada `uv sync --extra dev` çalıştır.

SORUN (doğrulanmış):
- app/web/security.py:66 — settings.api_token boşsa doğrulama TAMAMEN atlanıyor.
- POST /api/approvals/{id}/approve (server.py:1917) ve
  POST /api/supervisor/clear-stop-all (server.py:1971) insanla aynı yetkide.
- Sonuç: Achilles'in kendi doğurduğu `claude -p` süreci kendi eğitimini onaylayabilir
  ve kill-switch'i temizleyebilir. CLAUDE.md Kural-8 bu kurulumda etkisiz.

İSTENEN:
1. İki kimlik seviyesi: "human" (UI/CLI) ve "driver" (spawn edilen motor).
2. Onay uçları + stop-all temizleme + eğitim başlatma YALNIZ human scope'a açık.
   Driver scope bu uçlarda 403 alır — ve OpenAPI'de görünmemesi tercih edilir.
3. Driver token'ı kısa ömürlü, run_id'ye bağlı, tek-koşuluk olsun.
4. api_token boşken davranış: en azından bir başlangıç uyarısı logla; sessiz
   "auth kapalı" durumunu belirgin hale getir.
5. Mevcut insan akışlarını (web UI, CLI) BOZMA — geriye dönük uyumlu kal.

KULLAN:
- `rlm-security-reviewer` ajanı — tasarımı ve sonra uygulamayı PASS/FAIL denetlesin.
  Kendi yazdığın kodu kendin onaylama; ajanın raporu olmadan PR açma.
- `/codegen-review` skill'i — ruff+mypy+test kapısı.

TESTLER (zorunlu, çevrimdışı):
- driver scope ile approve → 403
- driver scope ile clear-stop-all → 403
- human scope ile ikisi de → 200
- driver token'ın run_id dışında kullanımı → reddedilir
- api_token boşken uyarı loglanıyor

KAPI: make format && make lint && make typecheck && make test
Sonra PR aç, CI yeşilse merge et. Gerçek eğitim BAŞLATMA.
```

---

## P2 — Motor kayıt tablosu ✅ TAMAMLANDI (PR #112, 2026-07-21)

> **Teslim:** `app/orchestration/engines.py` — `Engine` frozen dataclass + `_ENGINES` tablosu
> (claude / codex / gemini / local); yeni motor = **tek satır**. `driver.py` tabloya bağlandı:
> `build_hunt_command(run, engine)` + `engine_available(engine)`; `claude_available()` geriye
> dönük korundu. Prompt, `PROMPT` sentinel'inin yerine **tek argv öğesi** olarak konur (shell yok).
> PATH yoklaması TTL'li (`PROBE_TTL_S=60`); `which`/`clock` enjekte edilebilir → offline test.
> Kota uyarısı her motorda taşınıyor → **P5 UI bunu `describe_all()`'dan okuyacak.**
> Kimlik bilgisi alanı yok; yalnız API-key'le çalışan motor tabloya alınmadı (test bekçiliğinde).
> +27 test. **P3 artık başlatılabilir.**

**Şerit:** B · **Bağımlılık:** yok · **Paralel:** P1 ile aynı anda

```
Achilles'in birden fazla yerel "motor"u (abonelikli CLI ajanı) tanımasını sağla.

ÖNCE OKU: HANDOFF.md worktree hazard bölümü; gerçek worktree kur
(`-b claude/engine-registry`), `uv sync --extra dev`.

BAĞLAM: app/orchestration/driver.py şu an SADECE `claude`'u biliyor
(build_hunt_command():55 → ["claude","-p",prompt]; claude_available():78 → which).
Bunu küçük bir kayıt tablosuna genelleştir.

İSTENEN — app/orchestration/engines.py (YENİ, küçük tut):
Her motor için: ad, probe komutu, argv şablonu, insan-okur etiket.
  claude  → `claude -p <prompt>`        (abonelik OAuth)
  codex   → `codex exec <prompt>`       (ChatGPT plan girişi)
  gemini  → `gemini -p <prompt>`        (Google hesabı)
  local   → motor yok, doğrudan Ollama hattı (spawn yok)
Yeni motor eklemek TEK SATIR olmalı.

KRİTİK KISITLAR:
- Achilles kimlik bilgisi TOPLAMAZ/SAKLAMAZ/İSTEMEZ. Mail, şifre, API key yok.
  Motorlar kendi CLI oturumlarıyla girişli. Bizim işimiz sadece "kurulu mu / girişli mi"
  tespiti. Kimlik formu tasarlama.
- API key yolu kalıcı olarak yasak (CLAUDE.md + memory: no-api-local-subscription-only).
  Bir motor yalnız API key ile çalışıyorsa onu tabloya EKLEME.
- shell=True YOK — argv listesi. Prompt asla shell'e string olarak geçmez.
- Determinizm: probe sonuçları cache'lenecekse TTL açık olsun.

AYRICA: her motor için "abonelik kotası uyarısı" metni taşı — headless koşular
interaktif kullanımla aynı pencereyi tüketiyor (Codex'te 5 saatlik yuvarlanan pencere).
P5'te UI bunu gösterecek.

KULLAN: `/codegen-review` skill'i.

TESTLER: motor bulunamadığında davranış; argv şablonu doğru kuruluyor;
bilinmeyen motor adı reddediliyor; shell enjeksiyonu imkânsız.

KAPI: make format && make lint && make typecheck && make test → PR → merge.
```

---

## P3 — Sür-modu prompt + MCP geçişi

**Şerit:** B · **Bağımlılık:** P2 · **Paralel:** P4 ile

```
Spawn edilen motora "sür" modu prompt'u ve Achilles MCP araçlarına erişim ver.

ÖNCE: worktree kontrolü (HANDOFF.md), `-b claude/drive-mode`, `uv sync --extra dev`.
P2 (app/orchestration/engines.py) merged olmalı — üzerine kur.

BAĞLAM: app/orchestration/driver.py:37 build_hunt_prompt() SABİT ve SALT-RAPOR
(bug avı için). Kod değiştirmeyi/eğitimi açıkça yasaklıyor. RUN akışı için ikinci
bir mod gerekiyor.

İSTENEN:
1. build_drive_prompt() — "sür" modu şablonu. İçeriği:
   - Achilles MCP araçlarını kullan, doğrudan dosya düzenleme yapma
   - hedef: veri hattı adımlarını sırayla ilerlet (carding → RLM → curate → assemble)
   - EĞİTİM BAŞLATMA; taze insan onayı gerektiren her adımda DUR ve raporla
   - çıktının son satırı makine-okunur verdict olsun (mevcut
     parse_hunt_verdict():60 desenini AYNALA, yeni bir format icat etme)
2. Alt sürece MCP erişimi: mcp_server/achilles_mcp.py'yi --mcp-config ile geçir.
   Kullanıcı-düzeyi `claude mcp add` kaydına BAĞIMLI OLMA — spawn kendi kendine yetsin.
3. Timeout: mevcut HUNT_TIMEOUT_S=1800 sür-modu için yeniden değerlendir, sabiti ayır.
4. Alt sürece P1'in "driver" scope token'ı geçirilir — human token ASLA.

KULLAN:
- `rlm-integration-agent` — MCP geçiş yolunu gözden geçirsin.
- `/codegen-review`.

TESTLER: prompt şablonu eğitim-yasağı ibaresini içeriyor; verdict parse'ı
bozulmamış; MCP config yolu üretiliyor; driver token geçiyor, human token geçmiyor.

KAPI: make format && make lint && make typecheck && make test → PR → merge.
Gerçek spawn ile canlı deneme YAPMA (P6'da).
```

---

## P4 — MCP araç allow-list + token iletimi

**Şerit:** A · **Bağımlılık:** P1 · **Paralel:** P3 ile

```
Achilles MCP yüzeyini daralt ve token kısır döngüsünü çöz.

ÖNCE: worktree kontrolü (HANDOFF.md), `-b claude/mcp-allowlist`, `uv sync --extra dev`.
P1 (scope izolasyonu) merged olmalı.

İKİ SORUN (doğrulanmış):
1. mcp_server/achilles_mcp.py:43 — httpx.AsyncClient hiçbir Authorization başlığı
   set etmiyor. ACHILLES_API_TOKEN ayarlıysa TÜM MCP tool çağrıları 401 alır.
   Yani "token aç → MCP kırılır / MCP çalışsın → kapı açık kalır" kısır döngüsü.
2. FastMCP.from_openapi() ~110 endpoint'in HEPSİNİ tool yapıyor — eğitim başlatma,
   onay verme, stop-all dahil. Dış bir ajanın görmemesi gereken uçlar görünüyor.

İSTENEN:
1. Token iletimi: MCP proxy'si scope'lu token'ı Authorization başlığında geçirsin.
2. Allow-list: from_openapi bir filtreden geçsin.
   - SERBEST (okuma): rag/ask, cards, backtest okuma, status, sentinel, agents/graph
   - HİÇ SUNULMAZ: approvals/*, supervisor/stop-all, training/run, autodrive execute
   Liste açık ve tek yerde dursun; "varsayılan kapalı, açıkça izin ver" yaklaşımı.
3. /api/openapi.json ve /api/docs auth'suz (server.py:145) — dış keşif yüzeyi.
   Bunu daraltmayı DEĞERLENDİR, ama web UI'yi kırıyorsa dokunma; kararı yaz.
4. 2026-07-28 MCP spec'i (stateless çekirdek, session'ların kaldırılması, SSE'nin
   ölmesi, Roots/Sampling/Logging deprecation) yakın. Session-tabanlı YENİ bir şey
   EKLEME. Mevcut FastMCP sürümünün spec durumunu kontrol et ve bulguyu yaz.

KULLAN:
- `rlm-security-reviewer` — allow-list'i denetlesin, PASS almadan PR açma.
- `/codegen-review`.

TESTLER: yasak uçlar tool listesinde YOK; izinli uçlar VAR; token başlığı geçiyor;
token'lı modda MCP çağrısı 200 dönüyor.

KAPI: make format && make lint && make typecheck && make test → PR → merge.
```

---

## P5 — Motor durum API'si + ⚡ RUN butonu

**Şerit:** birleşme · **Bağımlılık:** P1 + P2 + P3 + P4 (hepsi merged)

```
Kullanıcının "RUN'a bas, çalışsın" deneyimini tamamla.

ÖNCE: worktree kontrolü (HANDOFF.md), `-b claude/run-button`, `uv sync --extra dev`.
P1-P4 merged olmalı.

İSTENEN — BACKEND:
GET /api/engines → her motor için: ad, etiket, kurulu mu, girişli mi, kota uyarısı.
Salt-okuma, hiçbir şey tetiklemez. Kimlik bilgisi DÖNDÜRMEZ (token/mail/key asla).

İSTENEN — FRONTEND (15·AJAN HARİTASI sekmesi):
1. Motor seçici — kurulu olmayanlar gri, "nasıl kurulur" ipucu (kimlik formu DEĞİL).
2. Mevcut ⚡ butonu şu an execute=false dry-run. Onay diyaloğuyla execute=true'ya bağla.
3. Diyalogda AÇIKÇA göster:
   - hangi motor çalışacak
   - abonelik kotası uyarısı (headless koşu interaktif kullanımınla aynı pencereyi yer)
   - "eğitim BAŞLAMAZ, taze insan onayında durur" güvencesi
4. Koşu sırasında canlı durum + görünür DURDUR (stop-all) butonu.
5. Tek-tık ile geri alınamaz iş başlamasın — onay diyaloğu atlanamaz olsun.

TASARIM KISITI: mevcut kart yerleşimini (PR#105 derli-toplu şeritler) BOZMA.
Dekoratif öğe ekleme — PR#103 declutter kararına sadık kal.

KULLAN: `/achilles-web` skill'i; UI doğrulaması için preview araçları (dev server →
ekran görüntüsü). Kullanıcıya "sen kontrol et" deme, kendin doğrula.

TESTLER: /api/engines kimlik sızdırmıyor; execute=true onaysız çağrılamıyor;
kurulu-değil motor seçilemiyor.

KAPI: make format && make lint && make typecheck && make test → PR → merge.
```

---

## P6 — Uçtan uca duman testi + Kademe-2 derin av ✅ TAMAMLANDI (2026-07-21)

**Şerit:** kapanış · **Bağımlılık:** P5 · **Paralel:** yok

### Sonuç — 5 iddiadan 4'ü doğrulandı, 1'i YANLIŞ ÇIKTI

| İddia | Sonuç |
|-------|-------|
| eğitim adımına gelince DURUYOR (taze onay yok) | ✅ doğrulandı |
| driver scope onay veremiyor (403), stop-all temizleyemiyor (403) | ✅ doğrulandı |
| motor kurulu değilken temiz hata, sessiz başarısızlık yok | ✅ doğrulandı (503 + Türkçe sebep) |
| ⛔ DURDUR koşan motoru gerçekten kesiyor | ❌ **YANLIŞTI → düzeltildi** |
| RUN → motor spawn → **MCP araçları görünüyor → ajanlar sürülüyor** | ❌ **KISMEN YANLIŞ** (aşağıya bak) |

**⛔ DURDUR kusuru (düzeltildi):** `AutoDriver` motoru bloklayan `subprocess.run` ile
doğuruyordu; süreç tutamacı saklanmadığı için STOP_ALL yalnız bir **bayrak dosyası**
yazıyordu. Motor 30 dk zaman aşımına kadar koşup **abonelik kotası yakmaya** devam
ediyordu — üstelik arayüz "durduruldu" deyip ⚡ kilidini açtığı için ÜSTÜNE ikinci motor
doğurulabiliyordu (PR#122'nin kapattığı "5 eşzamanlı spawn" kazasının geri dönüşü).
Düzeltme: `app/orchestration/engine_procs.py` süreç kaydı + `_default_runner`'ın
Popen+yoklama döngüsü + `/api/supervisor/stop-all`'ın gerçek `terminate_all()` çağrısı.

**Sür modu bağlı değil (bilinen boşluk, kapatılmadı):** `build_drive_command` **hiçbir
spawn yolundan çağrılmıyor**. ⚡ RUN yalnız **av** modunu doğuruyor; av modu
`--safe-mode` ile başlar ve o bayrak **MCP'yi de kapatır** → motor Achilles MCP
araçlarını GÖRMEZ, veri hattını İLERLETEMEZ. Duman testi bunu `≈ drive-mode-wiring`
uyarısıyla açıkça raporlar. **Sür modunu bağlamak P7'ye kalır** (kapsam kararı: bu paket
doğrulama paketiydi; eksik özelliği sessizce "tamam" göstermek yerine görünür kılındı).

### Kademe-2 derin av — onaylanan bulgular

3 bulgu adversarial doğrulamadan geçti (hepsi **aynı sınıf**: allow-list'te "okuma"
etiketli ama kalıcı YAZAN GET uçları) → üçü de allow-list'ten çıkarıldı:

| Uç | Ne yapıyordu |
|----|--------------|
| `GET /api/backtest/{id}/risk` | `rr_<id>` sabit anahtarıyla risk raporunu **EZİYORDU**; içerik motorun sorgu parametrelerinden türüyordu |
| `GET /api/understanding-score` | `record=true` ile kalıcı snapshot + JSON yazıyordu |
| `GET /api/sentinel/overview` | `run(persist=True)` ile her çağrıda geçmişe yazıyordu |

Kök sebep: allow-list sözleşmesi **HTTP metoduna** göre denetleniyordu
(`test_yazma_metodlari_tamamen_elenir` yalnız POST/PUT/DELETE/PATCH'e bakıyordu) →
yan etkili GET sessizce geçiyordu. Sınıf-düzeyi kapı:
`tests/test_mcp_allowlist_side_effects.py` handler **kaynak kodunu** tarar (üç ucu da
yakaladığı elle doğrulandı).

```
Yeni RUN hattını uçtan uca doğrula ve eğitim öncesi zorunlu derin avı çalıştır.

ÖNCE: worktree kontrolü (HANDOFF.md), `-b claude/run-e2e`, `uv sync --extra dev`.
P1-P5 merged olmalı.

BÖLÜM 1 — DUMAN TESTİ:
`uv run achilles orchestrate-smoke` hattını yeni RUN akışını kapsayacak şekilde
kullan/genişlet. Kanıtlanacaklar:
- RUN → motor spawn → MCP araçları görünüyor → ajanlar sürülüyor
- eğitim adımına gelince DURUYOR (taze onay yok)
- driver scope onay veremiyor (403) ve stop-all temizleyemiyor (403)
- DURDUR butonu koşan motoru gerçekten kesiyor
- motor kurulu değilken temiz hata, sessiz başarısızlık yok

BÖLÜM 2 — KADEME-2 DERİN AV (CLAUDE.md kadansı gereği zorunlu):
Alt-sistem başına paralel finder → her bulgu için adversarial doğrulama
(şüpheci, varsayılan "çürütülmüş") → yalnız ONAYLANANLARI düzelt.
Odak alanları: scope izolasyonu bypass'ları, MCP allow-list kaçakları,
alt-süreç enjeksiyonu, token sızıntısı (log/hata mesajı/OpenAPI dahil).
KULLAN: `lora-safety-secret-scanner` (sır/PII taraması), `rlm-security-reviewer`.

BÖLÜM 3 — DOKÜMAN:
README motor bağlama bölümü (sıfır-varsayım, numaralı, kopyala-yapıştır — memory:
readme-beginner-friendly). HANDOFF.md güncelle. Bu roadmap'i "TAMAMLANDI" işaretle.

ÇIKTI: raporda "başarılı" demeden önce KANIT göster (test çıktısı, log).
Test geçmiyorsa geçmiyor de.

KAPI: make format && make lint && make typecheck && make test → PR → merge.
Gerçek LoRA eğitimi BU PAKETTE BAŞLATILMAZ (Kural-8, insan onayı ayrı).
```

---

# FAZ 2 — RUN'ı GERÇEKTEN ÇALIŞTIR (P1-P6 doğrulama sonrası)

> ✅ **DURUM (2026-07-22): BU FAZ KAPANDI.** P7 sür modunu fişe taktı, P8 bağımsız verdict'i
> ekledi. ⚡ RUN artık **sür (drive) modunda** doğuruyor (varsayılan `mode="drive"`,
> `orchestration_routes.py:145`); `build_drive_command` sür yolundan çağrılıyor
> (`driver.py:720`); motor MCP araçlarını görüyor, veri hattını ilerletiyor; eğitim onay
> kapısında duruyor. Aşağısı bu fazı BAŞLATAN tarihsel gerekçedir:
>
> **Neden bu faz vardı:** 2026-07-21 doğrulama denetimi (3 paralel ajan + tam test paketi,
> 1700+ test yeşil) P1-P6'nın 6 paketinden **5'inin gerçekten kapandığını**, ama **P3'ün
> yarım kaldığını** buldu: sür modu eksiksiz yazılmıştı ama HİÇBİR spawn yolundan
> çağrılmıyordu (**P7 bunu bağladı**). O an ⚡ RUN yalnız **av modunu** doğuruyordu, o da
> `--safe-mode` ile → MCP KAPALI → motor Achilles araçlarını görmüyordu. Yani asıl hedef
> ("RUN → eğitim ajanları devreye girsin") o an henüz gerçek değildi — **P7'de gerçek oldu.**
>
> Ayrıca denetim 3 gerçek yan-kusur buldu (aşağıda P7'de). Hepsi "beyan edilmiş eksik" —
> repo kendi dokümanlarında dürüstçe yazmış, gizlenmemiş. Bu faz onları kapattı.

## Doğrulanmış durum (2026-07-21 denetim özeti)

| Paket | Hüküm | Not |
|---|---|---|
| P1 scope izolasyonu | ✅ gerçekten tamam | testler yetki-fn'e HİÇ ulaşılmadığını de kanıtlıyor |
| P2 motor tablosu | ✅ gerçekten tamam | sertleştirilmemiş motor spawn'ı reddediliyor |
| P3 sür modu | ✅ **P7'de FİŞE TAKILDI** | `_drive_pipeline` çağırıyor; ⚡ RUN varsayılan drive |
| P4 MCP allow-list | ✅ gerçekten tamam | 116→19 uç; test dependency GRAFİĞİNE bakıyor |
| P5 RUN deneyimi | ✅ gerçekten tamam | 3-kapılı onay; `logged_in` uydurulmuyor |
| P6 DURDUR + E2E | ✅ tamam | gerçek `terminate_all()`; canlı adımlar bilerek skip |

---

## P7 — Sür modunu fişe tak + 3 yan-kusuru kapat ✅ TAMAMLANDI (2026-07-22)

**Şerit:** tek · **Bağımlılık:** P1-P6 (hepsi merged, ✅) · **Paralel:** yok — bu tek iş

### Teslim özeti (kanıt: tam test paketi yeşil + orchestrate-smoke drive-mode-wiring PASS)

- **[SORUN 1 — ANA HEDEF] Sür modu FİŞE TAKILDI.** `AutoDriver.drive()` artık `mode`
  parametresi alır: `mode="drive"` (yeni `_drive_pipeline`) MCP'li sür komutunu
  (`build_drive_command` → `--mcp-config` + `--strict-mcp-config`, `--safe-mode` YOK)
  doğurur. ⚡ RUN ucu (`/api/orchestration/autodrive`, 15·AJAN HARİTASI) **varsayılan
  `mode="drive"`**; AV modu **ayrı tetikleyicide** korundu (12·ORKESTRASYON → Otonom AV,
  `mode="hunt"`). Sür PASS'i `ACHILLES_DRIVE_VERDICT` okur ve `hunt_ack` YAZMAZ → zorunlu
  Kademe-2 av kapısı bağımsız kaldı (Kural 8).
- **[SORUN 2] TTL düzeltildi.** `_drive_pipeline` token'ı `mint(run_id,
  ttl_s=DRIVE_TOKEN_TTL_S)` ile mint eder; test mint ÇAĞRISINI assert eder (sabit değil).
- **[SORUN 3] SSE query-token kaldırıldı.** `app/web/sse_tickets.py` (kısa ömürlü,
  TEK-kullanımlık bilet) + `POST /api/training/stream-ticket`; `/api/training/stream`
  artık insan api_token'ını query'de KABUL ETMEZ. Frontend `startSSE` bilet alır.
- **Canlı doğrulama:** `achilles orchestrate-drive-live --allow-live-spawn` (varsayılan
  KAPALI, CI'da ASLA koşmaz) — elle tek motor doğurup MCP görünürlüğünü kanıtlar.
- **Denetim:** `rlm-security-reviewer` + `rlm-integration-agent` PASS; `/codegen-review`
  kapısı (ruff+mypy+test) yeşil.

---

**(orijinal prompt — referans için korunur)**

Bu paket senin 5. adımını ("RUN → ajanlar sürülüyor") **gerçek** yapar. Denetimde bulunan
4 somut kusuru kapatır. Kapsamı DAR tut — yeni özellik ekleme, sadece yazılmış-ama-bağlanmamış
olanı bağla ve kusurları düzelt.

```
Achilles'te "sür" modunu gerçek spawn yoluna bağla ve doğrulama denetiminin bulduğu
3 yan-kusuru kapat. DAR KAPSAM: yeni özellik değil, mevcut ölü kodu fişe takmak + fix.

ÖNCE OKU:
- HANDOFF.md worktree hazard bölümü. `git rev-parse --show-toplevel` cwd'ni vermiyorsa
  gerçek worktree kur: git worktree add "<ayrı-dir>" -b claude/drive-mode-wiring origin/main
- docs/ROADMAP_MOTOR_BAGLAMA.md P7 bölümü (bu dosya) + HANDOFF.md P6 doğrulama özeti
- docs/SCOPE_ISOLATION.md (P1 sözleşmesi) — bozmayacaksın
Orada `uv sync --extra dev --extra mcp` çalıştır.

DOĞRULANMIŞ SORUNLAR (2026-07-21 denetimi):

[SORUN 1 — ANA HEDEF] Sür modu ölü kod.
  app/orchestration/driver.py:111 build_drive_prompt() + build_drive_command() eksiksiz
  yazılmış AMA hiçbir spawn yolundan çağrılmıyor. AutoDriver.drive() (driver.py:448) yalnız
  build_hunt_command (AV modu) çağırıyor. Av argv'si _CLAUDE_ARGV (engines.py:128) --safe-mode
  içeriyor → --safe-mode MCP'yi de kapatır → motor sıfır MCP sunucusuyla başlar.
  Sür argv şablonu _CLAUDE_DRIVE_ARGV (engines.py:166) DOĞRU yazılmış: --safe-mode YOK;
  --setting-sources "" + --disable-slash-commands + --strict-mcp-config + --tools Read,Grep,Glob
  + --mcp-config <path>. build_mcp_config() (driver.py:144) kendine yeten config üretir.

[SORUN 2] Ölü TTL sabiti → koşu 35. dakikada 401.
  DRIVE_TOKEN_TTL_S (driver.py:66) tanımlı ama driver_scope.mint(run_id) çağrısı (driver.py:510)
  onu GEÇMİYOR → varsayılan 2100s (driver_scope.py:36) devrede. 3600s'lik bir sür koşusu
  ~35. dakikada token ölür, MCP çağrıları 401 alır. Test (test_drive_mode.py:216) yalnız
  sabitin BÜYÜKLÜĞÜNÜ ölçüyor, KULLANILDIĞINI değil → sahte güvence.

[SORUN 3] İnsan token'ı URL sorgu dizesinde.
  app/web/server.py:1508 GET /api/training/stream token'ı request.query_params.get("token")
  ile alıyor; frontend app/web/static/assets/app.js:1214 canlı kullanıyor. Sızan şey
  KISA ÖMÜRLÜ sürücü token'ı DEĞİL, İNSAN SIRRININ KENDİSİ → erişim/proxy loglarına,
  tarayıcı geçmişine düşer, TTL'i yok. Gerekçe meşru (EventSource özel başlık gönderemez)
  ama çözüm var: kısa-ömürlü tek-kullanımlık SSE bileti.

İSTENEN:

1. [SORUN 1] drive() sür komutunu çağıran yolu bağla.
   - ⚡ RUN akışı (execute=true) SÜR modunu doğursun; AV modu ayrı bir tetikleyicide kalsın
     (av, Kademe-2 bug taraması için hâlâ gerekli — silme).
   - build_mcp_config() ile üretilen config --mcp-config olarak geçsin; --strict-mcp-config
     kullanıcı-düzeyi `claude mcp add` kaydını yok saysın (kendine yeten olsun).
   - Sür koşusuna P1 sürücü scope token'ı geçsin (build_child_env, driver.py:210); insan
     token'ı ASLA. Bu zaten kurulu — koru.
   - Verdict: sür modu ACHILLES_DRIVE_VERDICT işaretçisini kullanır (driver.py:50) — sür
     PASS'i AV hunt_ack'ini AÇMASIN. Bu ayrım zaten var — bozma.

2. [SORUN 2] mint çağrısına ttl_s=DRIVE_TOKEN_TTL_S geçir. Sabiti canlı sür koşusu
   süresiyle uyumlu yap (HUNT/DRIVE timeout'undan büyük olmalı). Test: token'ın
   GERÇEKTEN o TTL ile mint edildiğini doğrula (sabit değerini değil, mint çağrısını assert et).

3. [SORUN 3] /api/training/stream'i sorgu-dizesi token'ından kurtar.
   - Kısa ömürlü (ör. 60s), tek-kullanımlık SSE bileti üret; insan bir kez normal auth'la
     bilet alır, EventSource o bileti query'de taşır (bilet ≠ kalıcı sır; log'a düşse de
     60s sonra ölü). VEYA başka temiz çözüm öner ama İNSAN api_token'ını query'den ÇIKAR.
   - Frontend app.js:1214 buna göre güncellensin.

KRİTİK KISITLAR (denetimden çıkan dersler):
- "GET = salt-okuma" BU DEPODA YANLIŞ. Sür moduna yeni bir uç/araç eklersen yan-etkisini
  handler kaynak kodundan doğrula (tests/test_mcp_allowlist_side_effects.py deseni).
- Sür modunda --tools yalnız YERLEŞİK araçları kapsar, mcp__* araçlarını KAPSAMAZ
  (engines.py:155). Yani MCP yüzeyinin tek sınırı P4 allow-list'i + sürücü token'ı. Sür
  moduna açılan MCP araç setini allow-list'in 19 ucuyla SINIRLA; genişletme.
- Kural-8: sür modu veri hattını ilerletir AMA gerçek eğitimi taze insan onayında DURDURUR.
  Sür promptu bunu zaten yazıyor (driver.py:126) — koru, gevşetme.
- API key yolu yasak; abonelik CLI'si. --safe-mode'u sür modunda GERİ GETİRME.

⚠️ CANLI DOĞRULAMA (bu paketin en riskli kısmı — dikkatli):
  Denetim şunu vurguladı: --setting-sources "", --tools, --disable-slash-commands
  kombinasyonunun `claude` CLI'de İDDİA EDİLDİĞİ GİBİ davrandığı HİÇ test edilmedi —
  yalnız argv string'i kontrol ediliyor. PR#122'de kazara 5 gerçek `claude -p` doğup
  kota yaktı; bu yüzden E2E canlı adımlar bilerek skip'li.
  - Otomatik testte GERÇEK motor SPAWN ETME (kota yakar, CI'da claude yok).
  - Bunun yerine: TEK, KONTROLLÜ, elle-tetiklenen bir canlı duman adımı ekle
    (env bayrağı veya --allow-live-spawn ile kapılı, varsayılan KAPALI). Bu adım tek koşuluk
    bir sür motoru doğurur, MCP araçlarının GERÇEKTEN göründüğünü kanıtlar, sonra DURUR.
    Kullanıcı bunu bir kez elle çalıştırıp kanıtı görsün. Otomatik CI'da ASLA koşmaz.
  - DURDUR'un bu canlı motoru gerçekten kestiğini de aynı adımda doğrula (P6 engine_procs).

KULLAN:
- `rlm-security-reviewer` ajanı — sür modu bağlamasını + SSE bilet çözümünü PASS/FAIL
  denetlesin (yeni delik açtın mı: token sızıntısı, scope kaçağı, MCP yüzey genişlemesi).
  Kendi kodunu kendin onaylama; ajan raporu olmadan PR açma.
- `rlm-integration-agent` — MCP geçiş yolunun (build_mcp_config + --mcp-config) doğru
  kurulduğunu gözden geçirsin.
- `/codegen-review` skill'i — ruff+mypy+test kapısı.
- `/achilles-web` skill'i — SSE bilet değişikliği frontend'i etkiliyorsa preview ile doğrula.

TESTLER (zorunlu, çevrimdışı — gerçek spawn YOK):
- drive() çağrıldığında SÜR argv'si kuruluyor (av değil); --safe-mode YOK; --mcp-config VAR
- mint çağrısı ttl_s=DRIVE_TOKEN_TTL_S ile yapılıyor (mint'i mock'la, argümanı assert et)
- /api/training/stream artık insan api_token'ını query'den KABUL ETMİYOR; bilet yolu çalışıyor
- bilet kısa ömürlü + tek-kullanımlık (ikinci kullanım reddedilir; TTL sonrası reddedilir)
- sür moduna açık MCP araç seti allow-list'in 19 ucunu AŞMIYOR
- canlı-spawn adımı varsayılan KAPALI (env bayrağı olmadan skip)

KAPI: make format && make lint && make typecheck && make test → PR aç → CI yeşilse merge.
Gerçek LoRA eğitimi BAŞLATMA (Kural-8). Otomatik canlı motor spawn ETME (kota).

ÇIKTI: "çalışıyor" demeden önce KANIT göster. Sür modunun bağlandığını test çıktısıyla,
canlı doğrulamayı (elle koşulduysa) log'la kanıtla. Bağlanmadıysa "bağlanmadı" de.
```

---

## P8 — Motorun kendi karnesine güvenme (bağımsız verdict) ✅ TAMAMLANDI (2026-07-22)

**Şerit:** tek · **Bağımlılık:** P7 · **Paralel:** yok

### Teslim özeti (kanıt: tam test paketi yeşil + rlm-security-reviewer PASS)

Uygulanan tasarım: **Seçenek C (yapılandırılmış kanıt) + Seçenek A (dış-kanıt eşleştirme)**
deterministik biçimde birleştirildi. Seçenek B (ikinci bağımsız LLM doğrulayıcı) bilinçli
ERTELENDİ — çevrimdışı test edilemez + kota yakar; dürüstçe "gelecek katman" olarak belgelendi.

- **Yeni `app/orchestration/verdict_audit.py`** — deterministik, çevrimdışı, LLM YOK. Motor
  artık serbest bir "PASS" değil, `ACHILLES_HUNT_EVIDENCE` JSON kanıt bloğu (taranan dosyalar
  + alt-sistemler + bulgular) üretmek zorunda. Denetim bu kanıtı **motordan bağımsız bir
  oracle** ile — DOSYA SİSTEMİYLE — doğrular: uydurma/yok yollar sayılmaz, kapsama tabanı
  (`MIN_SCANNED_FILES=5`, `MIN_SUBSYSTEMS=2`), yol-geçişi reddi, ve PASS derken HIGH/BLOCKER
  bulgu listelemek = iç-tutarsız → reddedilir.
- **`driver.py`** — `build_hunt_prompt` kanıt bloğu ister; `AutoDriver.drive()` hunt yolu
  `hunt_ack=true`'yu YALNIZ `verdict.passed AND audit.ok` iken yazar. Kanıtsız serbest "PASS"
  deep-hunt'ı bloklu bırakır (`test_fake_pass_without_evidence_is_caught`).
- **Kural-8 SIKILAŞTI, gevşemedi:** eskiden son-satır regex'i tek dayanaktı; şimdi (a) yapısal
  kanıt zorunlu, (b) dosya sistemiyle bağımsız teyit, (c) iç-tutarsız beyan reddi.
- **rlm-security-reviewer denetimi:** genel PASS; 1 MEDIUM bulgu (derinden iç-içe JSON
  `RecursionError` fırlatır — `ValueError` değil `RuntimeError` alt sınıfı → dar `except`
  kaçırıyordu → sürücüyü çökertirdi) **düzeltildi** (`extract_evidence` fail-closed geniş
  yakalama + `drive()` denetim çağrısı try/except + regresyon testi). Path-traversal, keyfi
  dosya okuma, bilgi sızıntısı: temiz. Eğitim-kapısı bypass'ı YOK.
- **Dürüstlük sınırı (belgelendi):** denetim dosyaların VAR olduğunu doğrular, motorun onları
  GERÇEKTEN okuduğunu değil — bunu tam kapatmak ikinci bağımsız LLM doğrulayıcı gerektirir
  (Seçenek B, çevrimdışı test edilemez). "Hiç bakmadan PASS yaz" ve iç-tutarsız beyan sınıfları
  kapatıldı.

---

**(orijinal prompt — referans için korunur)**

Denetimin bulduğu en derin zayıflık; P7'den ayrı çünkü tasarım kararı gerektiriyor,
mekanik fix değil.

```
Achilles'te motorun av/sür verdict'inin BAĞIMSIZ doğrulamasını ekle.

ÖNCE: worktree kontrolü (HANDOFF.md), -b claude/independent-verdict, uv sync --extra dev.

SORUN (doğrulanmış, driver.py:544):
parse_hunt_verdict(output) verdict'i TAMAMEN motorun kendi stdout'unun son satırından
okuyor (ACHILLES_HUNT_VERDICT: PASS). Motor derin avı HİÇ yapmadan "PASS" yazsa hat açılır.
Bu, ZORUNLU Kademe-2 av kapısının (Kural-8, her eğitimden önce) TEK dayanağı → motor kendi
başarı raporunun tek kaynağı. Fail-closed korumalar var (satır yoksa FAIL, sahte-binary
savunması, Write/Edit yasak) ama BAĞIMSIZ kanıt yok.

İSTENEN — tasarım seçeneklerini DEĞERLENDİR, birini öner+uygula:
A) Dış-kanıt eşleştirme: av "bulgu" iddiasını gerçek artefaktla doğrula (motor bir bulgu
   raporu dosyası bıraktıysa onu ayrı bir salt-okuma ajanı/kod re-parse etsin; boşsa PASS reddedilir).
B) İkinci bağımsız doğrulayıcı: verdict PASS ise ayrı bir `rlm-security-reviewer`/hafif
   kod-tarama ADIMI çalışıp "gerçekten tarandı mı" sorusuna bağımsız cevap versin
   (adversarial, varsayılan çürütülmüş — CLAUDE.md av deseni).
C) Av çıktısının yapısal kanıtı: motordan serbest "PASS" değil, YAPILANDIRILMIŞ kanıt iste
   (taranan dosya sayısı, bulgu listesi hash'i) ve bunu bağımsızca teyit et.

KISIT: Kural-8. Kendi kararını kendine onaylatma zincirini KIRMAK bu paketin amacı —
gevşetme, sıkılaştır. Motor kendi verdict'ini yazabilir ama TEK KANIT olmasın.

KULLAN: `rlm-security-reviewer` (tasarımı + uygulamayı denetle), `/codegen-review`.

TESTLER: motor sahte "PASS" yazsa bağımsız doğrulayıcı bunu YAKALIYOR; gerçek av PASS'i
geçiyor; boş/eksik kanıt reddediliyor.

KAPI: make format && make lint && make typecheck && make test → PR → merge.
```

---

# FAZ 3 — Kalan iki iş (2026-07-22 kabul denetimi sonrası)

> P7/P8 bağımsız denetimi (Explore ajanı + 1700+ test yeşil) her ikisinin de kabul
> kriterlerini karşıladığını doğruladı — ⚡ RUN artık gerçekten SÜR modunda doğuruyor,
> motor MCP araçlarını görüyor. **Ama denetim P8'de dar bir boşluk ve dokümanlarda kayma
> buldu.** İkisi de küçük; RUN akışını bloklamıyor.

## P9 — Gelişmiş sahte-PASS'i kapat ✅ TAMAMLANDI (2026-07-22)

**Şerit:** tek · **Bağımlılık:** P8 (merged) · **Paralel:** DOK ile aynı anda

### Teslim özeti (kanıt: tam test paketi yeşil 1755 passed + rlm-security-reviewer PASS)

Uygulanan tasarım: **Seçenek A (deterministik okuma-kanıtı)**. Seçenek B (ikinci LLM) reddedildi
(kota + çevrimdışı test edilemez); ayrıca içerik-hash'i de reddedildi — `--safe-mode` avında
motorun yalnız Read/Grep/Glob'u var, sha256 hesaplayamaz → meşru avlar YANLIŞ reddedilirdi.

- **`verdict_audit.py` — OKUMA-KANITI katmanı (P8 var-olma kapısının ÜSTÜNE eklendi).** Motor artık
  her "taranan dosya" için `{path, line, quote}` verir: 1-tabanlı satır no + o satırın BİREBİR
  metni. Denetim dosyayı **bağımsızca** okuyup `lines[line-1].strip() == quote.strip()` doğrular.
  En az `MIN_READ_PROVEN`(=5) FARKLI dosya için geçerli kanıt gerekir. Uydurulamaz çünkü belirli
  bir dosyanın GÜNCEL durumundaki belirli satırın içeriği ancak o dosya okunarak bilinir.
- **Jenerik-satır forgery savunması:** aynı alıntı-satırı ve aynı dosya bir kez sayılır
  (`from __future__ import annotations`i 5 dosyaya kanıt diye tekrar kullanmak `read_proven_count=1`'e
  düşer); `MIN_QUOTE_LEN=12` tek-karakterlik satırları eler; `line: true` (bool, int alt sınıfı) kanıt
  sayılmaz.
- **Kapı SIKILAŞTI, gevşemedi:** P8'in var-olma/alt-sistem/iç-tutarsızlık kapıları AYNEN ilk gate;
  okuma-kanıtı en son ve en sıkı gate olarak eklendi. `driver.py:618` hâlâ `verdict.passed AND audit.ok`.
- **rlm-security-reviewer denetimi:** PASS. 1 LOW/MEDIUM sertleştirme (DoS: büyük artefaktı
  kanıt hedefi göstererek denetim belleğini tüketme) DÜZELTİLDİ → `MAX_PROOF_FILE_BYTES` sınırı +
  satır-satır okuma (dosya belleğe alınmaz). Yol-geçişi/mutlak-yol, içerik sızıntısı, tip-karışıklığı,
  ikili-dosya çökmesi: temiz.
- **Dürüstlük sınırı (belgelendi):** okuma-kanıtı "hiç açmadan PASS" sınıfını kapatır; "açtı ama
  düşünmedi" sınıfını değil — onu tam kapatmak ikinci bağımsız LLM doğrulayıcı gerektirir (ertelendi).

---

**(orijinal prompt — referans için korunur)**

Denetimin bulduğu boşluk: P8 motorun "PASS"ini dosya sistemiyle teyit ediyor — motor
UYDURMA dosya adı yazarsa yakalanıyor (test var). AMA motor 5 GERÇEK var-olan dosya adını
(driver.py, engines.py…) listeleyip **hiç okumadan** PASS yazsa denetim GEÇER; kontrol
dosyanın *var olduğuna* bakıyor, *okunduğuna* değil. Kural-8 av kapısının tek dayanağı
bu olduğu için kapatmaya değer.

```
Achilles av verdict denetimine "motor dosyaları GERÇEKTEN okudu mu" katmanını ekle.

ÖNCE OKU: HANDOFF.md worktree hazard bölümü. Gerçek worktree kur
(git worktree add "<ayrı-dir>" -b claude/verdict-read-proof origin/main),
uv sync --extra dev --extra mcp. docs/ROADMAP_MOTOR_BAGLAMA.md P8 + P9 bölümlerini oku.

DOĞRULANMIŞ BOŞLUK (2026-07-22 denetimi):
app/orchestration/verdict_audit.py motorun ACHILLES_HUNT_EVIDENCE JSON'undaki "taranan
dosyalar"ı depoda VAR MI diye teyit ediyor (audit_hunt_evidence, driver.py:602-611 çağırıyor).
Ama VAR-olma ≠ OKUNDU. Motor gerçek dosya adları listeleyip hiç okumadan PASS yazabilir.
Modülün kendi docstring'i (verdict_audit.py:22-25) bu sınırı zaten kabul ediyor.

İSTENEN — tasarım seçeneklerini DEĞERLENDİR, birini öner+uygula:
A) Okuma-kanıtı: motordan her "taranan dosya" için, o dosyanın İÇERİĞİNDEN türeyen doğrulanabilir
   bir işaret iste (ör. dosya içi belirli satır/sembol alıntısı, ya da içerik hash'inin bir
   parçası) ve bunu bağımsızca dosyayı okuyup teyit et. Uydurulamaz çünkü içeriğe bağlı.
B) İkinci bağımsız doğrulayıcı adım: PASS ise ayrı bir hafif kod-tarama/ajan, avın
   iddia ettiği bulguların gerçekten o dosyalarda olup olmadığını adversarial kontrol etsin
   (varsayılan çürütülmüş — CLAUDE.md av deseni). NOT: ikinci LLM kota/çevrimdışı-test
   maliyeti getirir; deterministik A seçeneği tercih edilir, B'yi ancak A yetmezse öner.

KISITLAR:
- Kural-8: kapıyı SIKILAŞTIR, gevşetme. Motor kendi verdict'ini yazabilir ama TEK KANIT olmasın.
- Determinizm: rastgelelik varsa seed'li. eval/exec YOK — kanıt ayrıştırma güvenli regex/JSON.
- Çevrimdışı test: gerçek claude -p SPAWN ETME (kota). Kanıtı sentetik motor-çıktısıyla test et.
- Mevcut P8 yapısal denetimini BOZMA — üstüne ekle (var-olma kontrolü hâlâ ilk kapı kalsın).

KULLAN:
- `rlm-security-reviewer` ajanı — okuma-kanıtı şemasının uydurulamaz olduğunu adversarial
  denetlesin (motor bunu dosyayı okumadan üretebilir mi?). Ajan PASS'i olmadan PR açma.
- `/codegen-review` skill'i — ruff+mypy+test kapısı.

TESTLER (zorunlu, çevrimdışı):
- gerçek-dosya-adı-listeleyip-okumayan sahte motor çıktısı → audit REDDEDER (yeni test)
- gerçekten okuyup doğru içerik-kanıtı veren çıktı → GEÇER
- naif sahte (kanıt yok / uydurma dosya) hâlâ REDDEDİLİYOR (P8 regresyonu korunur)

KAPI: make format && make lint && make typecheck && make test → PR → CI yeşilse merge.
Gerçek eğitim BAŞLATMA (Kural-8). Kanıt göster; kapatamadıysan "kapatılmadı" de.
```

---

## DOK — Doküman senkronu 🧹 MEKANİK (5 dk)

**Şerit:** tek · **Bağımlılık:** yok · **Paralel:** P9 ile aynı anda

```
P7/P8 sonrası eskimiş doküman ifadelerini kodun gerçeğiyle senkronla. SADECE doküman.

ÖNCE: worktree kontrolü (HANDOFF.md hazard). Küçük iş, ayrı worktree şart değil ama
git add -A YAPMA — yalnız dokunduğun .md dosyalarını dar ekle.

DOĞRULANMIŞ KAYMA (2026-07-22 denetimi): aşağıdaki ifadeler artık YANLIŞ, kod çürüttü:
- "build_drive_command hiçbir spawn yolundan çağrılmıyor" → ARTIK çağrılıyor (driver.py:714)
- "DRIVE_TOKEN_TTL_S ölü sabit" → ARTIK mint'e geçiyor (driver.py:774)
- "SÜR MODU BAĞLI DEĞİL / RUN yalnız av modu" → ARTIK sür modu varsayılan (orchestration_routes.py:145)

İSTENEN:
1. HANDOFF.md — "sür modu bağlı değil", "P7'ye kaldı", "ölü sabit" gibi eskimiş uyarıları
   güncelle. P7/P8'in TAMAMLANDIĞINI, ⚡ RUN'ın artık gerçekten sür modunda doğurduğunu yaz.
   P8'de KALAN boşluğu (gerçek-dosya-listeleyen gelişmiş sahte-PASS → P9) not düş.
2. docs/ROADMAP_MOTOR_BAGLAMA.md — FAZ 2 giriş metnindeki/durum tablosundaki "sür-modu prompt
   ❌ yok", "build_drive_command çağrılmıyor" gibi fiks-öncesi satırları düzelt.
3. README motor bağlama bölümü varsa: ⚡ RUN'ın artık ne yaptığını (sür modu, MCP'li,
   eğitim onayda durur) sıfır-varsayım/kopyala-yapıştır anlat (memory: readme-beginner-friendly).

KISIT: yalnız GERÇEĞE uydur — yeni özellik/iddia UYDURMA. Emin olmadığın bir durumu
"tamam" yazma; koddan doğrula (driver.py, orchestration_routes.py, engines.py).

KAPI: doküman-only ise test şart değil ama `make lint` çalıştır (markdown/format bozmadın).
PR aç, merge et.
```

---

## Sonraki chat'e devir notu

Yeni bir oturuma şunu yapıştır:

```
docs/ROADMAP_MOTOR_BAGLAMA.md dosyasını oku. FAZ 3'teki açık paketi (P9 ve/veya DOK) uygula.
İkisi paralel; bağımsızlar. Denetim kanıtları prompt'ların içinde dosya:satır ile yazılı.
```

**Durum:** P1-P9 ✅ KAPANDI — ⚡ RUN gerçekten sür modunda doğuruyor, motor MCP araçlarını
görüyor; av verdict'i artık dosya-var-olma + OKUMA-KANITI ile bağımsız doğrulanıyor
(2026-07-22). **Açık:** DOK (doküman senkronu, mekanik) — RUN akışını bloklamıyor.

## Kapsam dışı (bilinçli erteleme)

- **A2A protokolü** — kurumlar-arası/çok-makineli. Local-first kurulumda gereksiz.
- **AGENTS.md + SKILL.md taşınabilirliği** — ucuz kazanç ama RUN akışını bloklamıyor;
  P6'dan sonra ayrı küçük paket.
- **LiteLLM ağ geçidi** — `openai_base_url` (`app/config/settings.py:52`) kancası
  zaten var; ihtiyaç doğmadan katman ekleme.
- **MCP 2026-07-28 stateless göçü** — P4'te sadece "yeni session bağımlılığı ekleme"
  kısıtı var; asıl göç FastMCP sürümü hazır olunca ayrı paket.
