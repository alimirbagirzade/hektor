# HANDOFF — Achilles Trader AI

_Son güncelleme: 2026-07-22 (P9 — okuma-kanıtı: av "PASS"i artık dosya-var-olma + İÇERİK-alıntısıyla bağımsız doğrulanır) · Branch: `claude/p9-fake-pass-closure-784123` · Repo: https://github.com/alimirbagirzade/achilles_
_Açık PR: P9 (aşağıda). Veri hattı KAPALI (carding✅ RLM✅ curate✅ assemble✅). Dataset 2000 örnek, pretrain-gate **GO**. Sıradaki: LoRA eğitimi (Kural-8 kapılı, insan onayı bekliyor). Motor bağlama roadmap'i P1-P9 KAPANDI; kalan tek iş DOK (doküman senkronu, mekanik)._

Yerel-öncelikli (local-first) AI **trading araştırma** sistemi (macOS Apple Silicon + Windows).
**Canlı bot değil, yatırım tavsiyesi değil.**

---

## 🔒 2026-07-22 — P9: Gelişmiş sahte-PASS'i kapat (var-olma ≠ okundu)

Branch `claude/p9-fake-pass-closure-784123`. P8'in bıraktığı dar boşluk kapatıldı.

- **BOŞLUK (P8 denetimi):** `verdict_audit` motorun beyan ettiği "taranan dosyalar"ın depoda
  VAR olduğunu teyit ediyordu ama OKUNDUĞUNU değil. Motor 5 GERÇEK var-olan dosya adını
  (driver.py, engines.py…) **hiç okumadan** listeleyip PASS yazsa denetim GEÇERDİ.
- **ÇÖZÜM — Seçenek A (deterministik okuma-kanıtı), P8'in ÜSTÜNE.** Motor artık her dosya için
  `{path, line, quote}` verir (1-tabanlı satır no + o satırın BİREBİR metni). `verdict_audit`
  dosyayı **bağımsızca** okuyup `lines[line-1].strip() == quote.strip()` doğrular; ≥5 FARKLI
  dosya için geçerli kanıt şart. Uydurulamaz — belirli dosyanın güncel satır içeriği ancak
  okuyarak bilinir. İçerik-hash'i reddedildi (safe-mode avında sha256 hesaplanamaz →
  meşru avlar yanlış reddedilirdi); Seçenek B (ikinci LLM) reddedildi (kota).
- **Jenerik-satır forgery savunması:** aynı alıntı-satırı + aynı dosya bir kez sayılır;
  `MIN_QUOTE_LEN=12`; `line: true` (bool) kanıt sayılmaz.
- **Kural-8 SIKILAŞTI:** P8 var-olma/alt-sistem/iç-tutarsızlık kapıları AYNEN ilk gate;
  okuma-kanıtı en son+en sıkı gate. `test_listed_but_not_read_rejected` gelişmiş sahte-PASS'i yakalar.
- **Denetim (rlm-security-reviewer):** PASS. 1 LOW/MEDIUM DoS sertleştirme DÜZELTİLDİ
  (`MAX_PROOF_FILE_BYTES` + satır-satır okuma → büyük artefakt kanıt hedefi gösterilip denetim
  belleği tüketilemez). Yol-geçişi/içerik-sızıntısı/tip-karışıklığı/ikili-dosya: temiz.
- **⚠️ DÜRÜSTLÜK SINIRI:** okuma-kanıtı "hiç açmadan PASS" sınıfını kapatır, "açtı ama
  düşünmedi" sınıfını değil (o, ikinci bağımsız LLM doğrulayıcı ister — bilinçli ertelendi).
- **Kanıt:** tam test paketi yeşil (1755 passed, 4 skipped); ruff+mypy temiz.

---

## 🛡️ 2026-07-22 — P8: Motorun kendi karnesine güvenme → BAĞIMSIZ verdict (FAZ 2 KAPANDI)

Branch `claude/p7-sur-mode-fixes-323262`. **Not:** P7 zaten merged (PR#125) çıktı → bu oturum
P8'e geçti. Denetimin bulduğu en derin zayıflık kapatıldı.

- **SORUN (driver.py):** zorunlu Kademe-2 av kapısı (Kural 8, her eğitimden önce) TAMAMEN
  motorun kendi stdout son satırındaki `ACHILLES_HUNT_VERDICT: PASS`'ine güveniyordu. Motor
  derin avı HİÇ yapmadan "PASS" yazsa `hunt_ack=true` yazılır, eğitim hattı açılırdı →
  **motor kendi başarısının tek kaynağıydı.**
- **ÇÖZÜM — yapısal kanıt + dosya-sistemi teyidi (Seçenek C+A).** Yeni
  `app/orchestration/verdict_audit.py`: motor artık `ACHILLES_HUNT_EVIDENCE` JSON bloğu
  (taranan dosyalar + alt-sistemler + bulgular) üretmek zorunda. `audit_hunt_evidence()`
  bunu **motordan bağımsız** olarak DOSYA SİSTEMİYLE doğrular: uydurma/yok yollar sayılmaz,
  kapsama tabanı (5 dosya / 2 alt-sistem), yol-geçişi reddi, PASS+HIGH bulgu = iç-tutarsız →
  reddedilir. `drive()` hunt yolu `hunt_ack`'i YALNIZ `verdict.passed AND audit.ok` iken yazar.
- **Kural-8 SIKILAŞTI:** kanıtsız/uydurma/iç-tutarsız "PASS" artık av kapısını AÇAMAZ
  (`test_fake_pass_without_evidence_is_caught`). Sür (drive) modu ayrı işaretçi kullandığı
  için etkilenmez (P7 ayrımı korundu).
- **Denetim (rlm-security-reviewer):** genel PASS. 1 MEDIUM bulgu DÜZELTİLDİ: derinden iç-içe
  JSON `RecursionError` fırlatır (`ValueError` DEĞİL) → dar `except` kaçırıp sürücüyü
  çökertiyordu → `extract_evidence` fail-closed geniş yakalama + `drive()` denetim çağrısı
  try/except + regresyon testi. Eğitim-kapısı bypass'ı YOK (çökme `hunt_ack`'ten ÖNCE = fail-crash,
  fail-open değil). Path-traversal / keyfi dosya okuma / bilgi sızıntısı: temiz.
- **⚠️ DÜRÜSTLÜK SINIRI:** denetim dosyaların VAR olduğunu doğrular, motorun onları GERÇEKTEN
  okuduğunu değil. "Hiç bakmadan PASS yaz" + iç-tutarsız beyan sınıfları kapatıldı; tam
  kapatma ikinci bağımsız LLM doğrulayıcı (Seçenek B) ister — çevrimdışı test edilemez +
  kota yakar, bilinçli ertelendi. Sahte güvenlik hissi verilmedi (modül docstring + roadmap).
- **Kanıt:** tam test paketi yeşil (verdict_audit 11 + driver 2 yeni test dahil); ruff+mypy temiz.

---

## ⚡ 2026-07-22 — P7: Sür modunu fişe tak + 3 yan-kusur (TAMAMLANDI)

Branch `claude/motor-baglama-p7-2e8141`. **Senin asıl hedefin ("⚡ RUN → ajanlar sürülüyor")
artık GERÇEK.** Tam test paketi yeşil; `orchestrate-smoke` `drive-mode-wiring` PASS.

- **SORUN 1 (ana hedef) — sür modu FİŞE TAKILDI.** `AutoDriver.drive()` `mode` alır:
  `mode="drive"` yeni `_drive_pipeline` ile MCP'li sür komutunu doğurur (`--mcp-config`
  var, `--safe-mode` YOK). **⚡ RUN ucu varsayılan `drive`** (15·AJAN HARİTASI). **AV modu
  ayrı tetikleyicide korundu** (12·ORKESTRASYON → "Otonom AV", `mode="hunt"`). Sür PASS'i
  `ACHILLES_DRIVE_VERDICT` okur, `hunt_ack` YAZMAZ → zorunlu Kademe-2 av kapısı bağımsız
  (Kural 8; P8 bunu daha da sıkılaştıracak).
- **SORUN 2 — token TTL.** `mint(run_id, ttl_s=DRIVE_TOKEN_TTL_S)` artık çağrılıyor; test
  mint ÇAĞRISINI assert ediyor (sabit büyüklüğünü değil). Sür token'ı ~35. dk'da ölmüyor.
- **SORUN 3 — SSE sır sızıntısı.** İnsan `api_token`'ı ARTIK `/api/training/stream`
  query'sinde KABUL EDİLMİYOR. Yeni `app/web/sse_tickets.py` (kısa ömürlü, TEK-kullanımlık)
  + `POST /api/training/stream-ticket`. Frontend `startSSE` bilet alıp query'de taşıyor.
- **Canlı doğrulama (elle, kapılı):** `achilles orchestrate-drive-live --allow-live-spawn`
  — varsayılan KAPALI, CI'da ASLA koşmaz; tek motor doğurup MCP görünürlüğünü kanıtlar.
- **⚠️ Hâlâ KANITLANMADI:** sür motorunun MCP araçlarını GERÇEKTEN gördüğü canlı olarak
  yalnız yukarıdaki elle adımla doğrulanır (otomatik test argv'yi kontrol eder, gerçek
  spawn YAPMAZ — kota güvenliği). Kullanıcı bir kez elle koşmalı.
- **Denetim:** `rlm-security-reviewer` + `rlm-integration-agent` çalıştırıldı.
- **Sıradaki:** P8 (motorun kendi verdict'ine bağımsız doğrulama — driver.py:544 kökü).

---

## ⚡🔍 2026-07-21 — P6: RUN hattı E2E doğrulama + eğitim-öncesi Kademe-2 av (TAMAMLANDI)

Motor bağlama roadmap'i (P1–P6) **KAPANDI**. Bu paket doğrulama paketiydi:
"çalışıyor" demeden önce kanıt aramak. **5 iddiadan 1'i yanlış çıktı, 1'i kısmen yanlış.**

### ⛔ DURDUR koşan motoru KESMİYORDU (gerçek kusur — düzeltildi)
`AutoDriver` motoru bloklayan `subprocess.run` ile doğuruyordu ve süreç tutamacı hiçbir
yerde saklanmıyordu. STOP_ALL yalnız `storage/STOP_ALL` **bayrak dosyasını** yazıyordu →
motor **30 dk zaman aşımına kadar koşup abonelik kotası yakmaya devam ediyordu**. Üstelik
arayüz "durduruldu" deyip ⚡ tekrar-giriş kilidini açtığı için ÜSTÜNE ikinci motor
doğurulabiliyordu — PR#122'nin kapattığı **"5 eşzamanlı spawn"** kazasının DURDUR
yolundan geri dönüşü.

Düzeltme (3 parça):
- `app/orchestration/engine_procs.py` — canlı motor süreç kaydı (`run_id` başına).
- `driver._default_runner` — `Popen` + yoklama döngüsü; STOP_ALL etkinleşince gerçekten
  `terminate()`/`kill()`. Kesilen koşu `STOPPED_RC` → `drive()` `stopped=True` raporlar
  ("av FAIL" demez; av koşmadı, kesildi).
- `/api/supervisor/stop-all` — bayrakla BİRLİKTE `terminate_all()`; yanıtta
  `engines_terminated` sayısı döner (sessiz "durduruldu" yok).

### ✅ SÜR MODU (P6'da bağlı değildi) → P7'de FİŞE TAKILDI
> **GÜNCELLEME (P7, 2026-07-22):** Bu P6 bulgusu KAPATILDI. `build_drive_command` artık
> sür yolundan çağrılıyor (`driver.py:720`) ve ⚡ RUN ucu **varsayılan `mode="drive"`**
> (`orchestration_routes.py:145`) → motor Achilles MCP araçlarını GÖRÜR, veri hattını
> İLERLETİR. "RUN → ajanlar sürülüyor" iddiası artık GERÇEK (bkz. yukarıdaki P7 bölümü).
> Aşağısı P6 anındaki tarihsel durumdur (referans için korunur):

`build_drive_command` **hiçbir spawn yolundan çağrılmıyor**. ⚡ RUN yalnız **av** modunu
doğurur; av modu `--safe-mode` ile başlar ve o bayrak **MCP'yi de kapatır** → motor
Achilles MCP araçlarını **GÖRMEZ**, veri hattını **İLERLETEMEZ**.
👉 **"RUN → MCP araçları görünüyor → ajanlar sürülüyor" iddiası BUGÜN DOĞRU DEĞİL.**
Duman testi bunu `≈ drive-mode-wiring` satırıyla açıkça uyarır. Sür modunu bağlamak
**P7**'ye kaldı (bu paket doğrulama paketiydi; eksik özelliği sessizce "tamam"
göstermektense görünür kılındı).

### 🔍 Kademe-2 av — 3 onaylanan bulgu, hepsi AYNI SINIF
Allow-list'te "okuma" etiketli ama **kalıcı YAZAN GET** uçları (üçü de çıkarıldı):

| Uç | Ne yapıyordu |
|----|--------------|
| `GET /api/backtest/{id}/risk` | `rr_<id>` sabit anahtarıyla risk raporunu **EZİYORDU**; içerik motorun sorgu parametrelerinden türüyordu (drawdown uyarısını susturup pozisyonu şişirmek mümkündü) |
| `GET /api/understanding-score` | `record=true` ile kalıcı snapshot + JSON yazıyordu |
| `GET /api/sentinel/overview` | `run(persist=True)` ile her çağrıda geçmişe yazıyordu |

**KÖK SEBEP (ders):** allow-list sözleşmesi **HTTP METODUNA** göre denetleniyordu —
`test_yazma_metodlari_tamamen_elenir` yalnız POST/PUT/DELETE/PATCH'e bakıyordu, yan
etkili GET sessizce geçiyordu. **"GET = salt-okuma" bu depoda YANLIŞ bir varsayımdır.**
Sınıf-düzeyi kapı: `tests/test_mcp_allowlist_side_effects.py` handler **kaynak kodunu**
tarar (üç ucu da yakaladığı elle doğrulandı). Yerine salt-okuma muadilleri eklendi
(`/api/understanding-score/history`, `/api/sentinel/history`).

### Duman testi genişletildi
`uv run achilles orchestrate-smoke` artık İKİ bölüm: **runtime** (eskisi) +
**⚡ RUN sözleşmeleri** (10 yoklama). `--skip-runtime` / `--skip-run-pipeline`.
⛔ **Gerçek motor doğurmaz** (kota yakmaz): `live-spawn` bilinçli `skip` ve raporda
"KANITLANMAZ" der.

### 🔍 Kademe-2 av — 2. tur (düşen 3 finder yeniden koşturuldu): 8 onaylanan bulgu

İlk turda 6 finder'ın 3'ü API hatasıyla düşmüştü (tam da odak alanları: token sızıntısı,
scope izolasyonu, alt-süreç enjeksiyonu) → **yeniden koşturuldu, 63/63 ajan tamamlandı.**

**BU PAKETTE DÜZELTİLDİ:**

| Sev | Bulgu | Düzeltme |
|-----|-------|----------|
| 🔴 high (3/3) | **Motor binary'si mutlak yol olmadan + cwd pinlenmeden doğuruluyordu** → çalışma dizinine bırakılan sahte `claude.exe` gerçek CLI yerine koşar; taklitçi son satıra `PASS` yazıp **zorunlu derin av kapısını düşürürdü** (Kural 8). Sertleştirme bayrakları sahte binary'ye hiçbir şey yaptıramaz. | `_resolve_executable` (PATH'ten cwd atılır) + `cwd=_REPO_ROOT` pinlendi; 3 regresyon testi |
| 🟡 low | `/api/understanding-score/history` `limit` **sınırsız** (allow-list'e bu pakette eklendi) | `min(max(1,limit),200)` |
| 🟡 low | `DRIVE_TOKEN_TTL_S` **ölü sabit**; testi yalnız büyüklüğünü ölçüp sahte güvence veriyor | Açık "HENÜZ KULLANILMIYOR" uyarısı + P7 talimatı → **P7'de mint'e geçirildi** (`driver.py:780` `mint(run_id, ttl_s=DRIVE_TOKEN_TTL_S)`); sabit artık ölü değil |

**⚠️ ONAYLANDI AMA DÜZELTİLMEDİ — tasarım kararı gerektiriyor (SIRADAKİ İŞ):**

> **GÜNCELLEME (2026-07-22):** İlk iki satır KAPATILDI — insan API token'ı SSE query'sinden
> P7'de **tek-kullanımlık bilete** taşındı (`app/web/sse_tickets.py`; `/api/training/stream`
> artık api_token'ı query'de kabul etmez); `hunt_ack` öz-beyanı P8 (bağımsız `verdict_audit`
> dosya-sistemi teyidi) + P9 (okuma-kanıtı: `{path,line,quote}` bağımsız doğrulama) ile
> bağımsız kanıta bağlandı. Son iki satır (av'ın `.env` okuyabilmesi, koşu-başına autodrive
> kilidi) HÂLÂ AÇIK.

| Sev | Bulgu | Neden ertelendi |
|-----|-------|-----------------|
| 🔴 high (3/3) | **İnsan API token'ı URL sorgu dizesinde** (`/api/training/stream`, SSE) → uvicorn erişim log'una **düz metin** düşer. EventSource özel başlık gönderemediği için böyle yapılmış. | Doğru çözüm kısa ömürlü **tek-kullanımlık SSE bileti** (ayrı authenticated POST → ticket). Auth akışı değişikliği; ayrı paket. |
| 🔴 high (2/3) | **`hunt_ack` motorun KENDİ stdout'undan yazılıyor.** HTTP'de `hunt_ack=true` insan-yalnızdır (v5 regresyon gerekçesiyle), ama `AutoDriver` aynı bayrağı motorun son satırındaki `ACHILLES_HUNT_VERDICT: PASS` metnine bakarak **doğrulamasız** kalıcı yazar. Bağımsız kanıt (rapor artefaktı, bulgu sayısı) aranmaz. Prompt-injection ya da tembellik → **denetlenmemiş denetim kapısı**. | Fail-closed tarafı sağlam (işaretçi yoksa FAIL) ve eğitim yine taze insan onayı ister; ama av kapısı motorun öz-beyanına dayanıyor. Çözüm: PASS için rapor artefaktı şartı + panelde "motor beyanı" rozeti. Tasarım kararı senin. |
| 🔴 high (2/3) | Av motoru `Read/Grep/Glob` ile **`.env` gibi sır dosyalarını okuyabilir** — `build_child_env` yalnız SÜREÇ ORTAMINI temizler, dosya sistemini değil. | Av'ın işlevi için Read şart; araç seviyesinde dosya kısıtı yok. Azaltım: `.env`'i repo dışına almak ya da motoru izole çalışma alanında koşturmak. |
| 🟠 medium (2/3) | Aynı `run_id`'de **eşzamanlı autodrive**: `mint()` kardeş motorun token'ını koşu ORTASINDA iptal eder (uçta koşu-başına kilit yok). | Koşu-başına kilit gerekiyor; UI'daki tekrar-giriş kilidi güvenlik sınırı değil. |

**Reddedilen 11 iddia** (adversarial 3-lens, çoğunluk çürüttü) — ayrıntı workflow
journal'ında. Örn. "sürücü STOP_ALL basabiliyor ama kaldıramıyor" = **kasıtlı** asimetri.

### ⚠️ Bu pakette KANITLANMAYANLAR (dürüstlük notu)
- Gerçek `claude -p` spawn'ı **koşturulmadı** (kota koruması). MCP araçlarının canlı
  görünürlüğü ve ajanların gerçekten sürülmesi **kanıtlanmadı**.
- Av bir **statik kod incelemesidir**; çalışma zamanı davranışı ayrıca doğrulanmadı.

---

## 🔐 2026-07-21 — SCOPE İZOLASYONU: sürücü-motor ≠ insan yetkisi (PR #116 + #117)

**Kapatılan zafiyet:** Achilles kendi `claude -p` motorunu doğuruyor ve bu motor insanla
**aynı yetkiyle** kendi eğitimini onaylayabiliyor, kill-switch'i temizleyebiliyordu.
`app/web/security.py` — `api_token` boşsa (varsayılan) doğrulama tamamen atlanıyordu.
**Kural-8 bu kurulumda etkisizdi.**

İki kimlik seviyesi: `human` (UI/CLI) · `driver` (motor). `require_human` kapısı olan
uçlarda driver **403**. Sürücü token'ı sha256-hash'li, `run_id`-bağlı, TTL'li, koşu
bitince iptal. Doküman: **`docs/SCOPE_ISOLATION.md`**.

### ⚠️ ANA DERS — doğurulan ajana karşı HTTP auth TİYATRODUR
`rlm-security-reviewer` 3 turda HTTP katmanını atlayan **4 kanal** buldu:
1. **CLI auth'suz** — `achilles approval-approve` doğrudan `approvals.approve()` çağırır.
2. **`.env` fallback** — env'den token SİLMEK yetmez, dotenv geri okur → **boşa EZ**.
3. **MCP** — `--disallowedTools` `mcp__*`'ı kapsamaz; `achilles` MCP proxy'si 8765'e kanal açar.
4. **Hook'lar** — `.claude/settings.json` hook'ları **doğrudan kabukta** koşar (`Bash` aracı DEĞİL).

**Asıl sınır = ARAÇ kısıtı**, kanal kovalamaca değil sınıf-düzeyi çözüm:
`--safe-mode` + `--strict-mcp-config` + `--disallowedTools` (üçü **birlikte**).

### Üç spawn noktası, bilinçli farklı rejim
| Spawn | Rejim |
|-------|-------|
| `AutoDriver` | Tam kısıt + sürücü token'ı |
| `weekly-bug-scan.ps1` | Tam kısıt (kısıt eskiden yalnız PROMPT'taydı); diff'i script hesaplar |
| `rag-research-loop.ps1` | **KUŞATILAMAZ** — yazma/push işlevsel şart. Kendi eğitimini onaylayabilir; sahte kısıt EKLENMEDİ, test bunu sabitler. Yalnız güvendiğin içerikle koştur. |

### Ayrıca
- **Çok-motorlu tuzak:** sertleştirme motora ÖZGÜ. `Engine.hardened` varsayılan `False`;
  AutoDriver yalnız hardened motoru doğurur (**fail-closed**). codex/gemini doğrulanmadı.
- **Eğitim VERİSİ kararları da insan-yalnız:** kart approve/reject + feedback
  approve/reject/export (korpus = v5'in kökü). `automation/tasks`: sürücünün açtığı görev
  zorla `requires_approval=True`.
- **Dürüstlük:** `api_token` boşken bu katman kriptografik sınır **DEĞİL**.
  **Öneri: `ACHILLES_API_TOKEN` ata.** `include_in_schema=False` koruma değildir.

---

## 🆕 2026-07-21 — MOTOR BAĞLAMA ROADMAP'İ (araştırma, kod yok)

Yerel abonelikli motorları (Claude Code / Codex / Gemini CLI) Achilles'e bağlayıp
tek-tık **RUN** ile ajanları sürdürme yolu araştırıldı → **[docs/ROADMAP_MOTOR_BAGLAMA.md](docs/ROADMAP_MOTOR_BAGLAMA.md)**
(6 paket P1-P6, kopyala-yapıştır prompt'lar, 2 paralel şerit).

**Bulgu 1 — hattın %80'i zaten var, fişi takılı değil:**
MCP sunucusu (`mcp_server/achilles_mcp.py`, OpenAPI→tool proxy) · `claude -p` alt-süreç
sürücüsü (`app/orchestration/driver.py:56`, abonelik CLI'si, API key değil) · autodrive ucu
(`app/web/orchestration_routes.py:132`, `execute=false` dry-run kilidinde) · ⚡ buton (dry-run).

**Bulgu 2 — 🔴 BLOKLAYICI güvenlik açığı (P1):**
`api_token` boşsa doğrulama tamamen atlanıyor (`app/web/security.py:66`); onay
(`server.py:1917`) ve stop-all temizleme (`:1971`) insanla aynı yetkide. Yani Achilles'in
kendi doğurduğu motor **kendi eğitimini onaylayabilir + kill-switch'i temizleyebilir**
→ Kural-8 bu kurulumda etkisiz. RUN açılmadan ÖNCE scope izolasyonu şart.

**Bulgu 3 — MCP token kısır döngüsü (P4):** proxy Authorization iletmiyor
(`achilles_mcp.py:43`) → token açarsan MCP 401, MCP çalışsın diye kapatırsan kapı açık.

**Bulgu 4 — "motor bağlama" ekranı YOK:** CLI'lar kendi OAuth'uyla girişli; Achilles
kimlik toplamaz/saklamaz. Sadece "kurulu mu/girişli mi" tespiti. ⚠️ headless koşu
interaktif kullanımla **aynı abonelik penceresini** tüketir — UI'da uyarı gerekli.

**Ekosistem (2026):** MCP kazandı (Linux Foundation/Agentic AI Foundation, 97M aylık SDK
indirmesi; Claude/GPT/Gemini/Cursor/Hermes hepsi istemci). A2A tamamlayıcı ama local-first'te
gereksiz. ⚠️ **2026-07-28 spec'i stateless** — session/SSE/Sampling gidiyor; yeni session
bağımlılığı EKLEME.

**Sıradaki:** P1 + P2 paralel başlatılabilir (bkz roadmap prompt'ları).
## 🧹 GENEL TEMİZLİK — 2026-07-21 (PR #106 + #107, ikisi de MERGE EDİLMEDİ)

Dört paralel salt-okuma tarama ajanı; **muhafazakâr** kriter (hiç import edilmiyor **+**
test kapsamıyor **+** entry-point'ten erişilmiyor). Kullanıcı onayıyla uygulandı.

**PR #106 — temizlik:** 17 ölü Python modülü (~1.865 satır) + 5 ölü prompt + 7 kullanılmayan
bağımlılık (transitifle 20 paket) + 4 ölü config YAML + 4 artık script/dosya. 6 eskimiş
doküman `docs/arsiv/` altına **taşındı** (silinmedi).

**PR #107 — 3 gerçek hata** (temizlik değil, düzeltme):
1. `make install` KIRIKTI — tanımsız `--extra web` (Makefile + update.sh + update.ps1 + README).
   `|| true` ve `Out-Null` hatayı yuttuğu için görünmez kalmıştı.
2. `mcp_server/` çalışmıyordu — `fastmcp` hiçbir bağımlılık dosyasında yoktu. `mcp` extra'sı
   eklendi; `sync-mcp.sh` artık hatayı `2>/dev/null` ile yutmuyor.
3. `cors_origins` SAHTE-GUARD'dı — ayar vardı, `CORSMiddleware` hiç kurulmamıştı. Artık
   gerçekten kuruluyor + 2 regresyon testi.

### ⚠️ DERS — "referans 0" iddiası test dizinini kaçırabilir
`docs/PHASE4B_DRYRUN.md` + `PHASE4C_ACTIVATION.md` "referanssız" sanılıp arşivlenmişti;
`tests/test_phase4b_*.py` / `test_phase4c_*.py` onları **yönetişim belgesi** olarak zorunlu
kılıyor (auto-merge yasağı, rollback, insan onayı maddeleri). Test paketi yakaladı, geri alındı.
**Doküman silmeden önce `tests/` içinde de ara.**

### Git durumu (2026-07-21)
- Uzakta artık yalnız `main` + 2 PR dalı. 12 dal silindi (6 merged + 4 içeriği main'de + 2 onaylı).
- Stale ref uyarısı: `git branch -r` 62 dal gösteriyordu, `fetch --prune` sonrası gerçek sayı 13'tü.
  **Dal sayımından önce `git fetch --prune` çalıştır.**
- `git branch --no-merged` içerik değil **commit erişilebilirliği** ölçer — 6 "merge edilmemiş"
  dalın 4'ünün içeriği `git cherry` + byte-diff ile main'de doğrulandı.
- **ARŞİV ETİKETLERİ** (dal silindi, commit kalıcı korundu):
  - `arsiv/salvage-system32-cpu-lora` → `b8220cb` (9 dosya/490 satır; `cpu_lora_train.py` 304
    satır main'de yok — halefi `peft_lora_train.py`)
  - `arsiv/local-claude-operator-dry-run` → `a9ebc2e` (95 satır doküman)
- **ESKİ UYARI GEÇERSİZ:** `rag-chains-work` / `fix/rag-scoring-approval-cas` için "STRANDED,
  auto-merge etme" notu artık geçerli değil — içerik main'de doğrulandı, dal silindi.
- GitHub issue #1 KAPATILDI (kriterleri kodda doğrulandı). **#2 AÇIK BIRAKILDI** — kısmen
  farklı isimlerle yapılmış (`SynthesisEngine` var ama `StructuredIndicatorGenerator` ve
  `POST /api/research/synthesize` YOK); "tamamlandı" demek yanlış kayıt olurdu.

### Bekleyen / dokunulmayan
- **Worktree'ler kaldırılamadı** — çalışan `achilles-web` süreci `.venv` hardlink'lerini
  kilitliyor (Windows açık dosyanın hardlink'ini silmiyor). Web'i durdurup tekrar dene.
- *"Test var ama üretim yolu yok"* 6 dosya (~650 satır): `query_expander →
  multi_query_retriever → self_refining_rag` zinciri, `hybrid_retriever`, `regression_runner`,
  `ollama_installer`. Muhafazakâr kriteri sağlamadıkları için DOKUNULMADI.
- `reports/bug-scan/` `.gitignore`'da DEĞİL ama tüm komşu rapor dizinleri ignore'lu;
  `weekly-bug-scan.ps1` haftalık üretiyor → repoya birikiyor. Karar bekliyor.
- `pyarrow` kodda hiç import edilmiyor ama chromadb transitif isteyebilir → bırakıldı.

---

## 🚨 YENİ SEANS BAŞLANGICI — BUNU OKU

### ⛔ KALICI KISIT (2026-06-24) — API ASLA KULLANILMAZ
Kullanıcı direktifi: **pay-per-token API hiçbir zaman kullanılmayacak.** Çalışma-zamanı LLM
hattı YEREL (Ollama / native RLM Controller + RAG). Geliştirme/AI yardımı aylık **abonelikli**
araçlarla (Claude Code, Codex vb.) — API anahtarı/faturalı endpoint DEĞİL. Sonuç: alexzhang
RLM `backend="anthropic"` (API) yolu OPSİYONEL + KAPALI kalır (`rlm_alexzhang_enabled=False`,
`provider=native`); varsayılan/zorunlu YAPILMAZ. Yeni özelliklerde bulut API'sini varsayılan
bağımlılık yapma → opt-in + native fallback şart. Bkz memory [[no-api-local-subscription-only]].

### ⚠️ KRİTİK — WORKTREE ORPHAN HAZARD (git ANA repo'yu etkileyebilir)
Bir oturumun cwd'si `.claude/worktrees/<ad>` GÖRÜNÜR ama GERÇEK worktree OLMAYABİLİR
(pruned/orphan: `.git` yok, `git rev-parse --show-toplevel` ANA repo'yu verir, `.claude/worktrees`
gitignore'da → dosyalar commit edilemez, git komutları ANA checkout'un dalını DEĞİŞTİRİR).
2026-07-03'te tam bu yaşandı: `git checkout -b` ana checkout'u eşzamanlı oturumun parklı dalından
kopardı (kurtarıldı). **BAŞLARKEN kontrol:** `git rev-parse --show-toplevel` cwd'ni vermiyorsa
GERÇEK izole worktree kur: `cd C:/Users/sevinc/Development/achilles && git worktree add
"…/achilles-<iş>" -b <branch> origin/main` → orada `uv sync --extra dev` + gate + commit + PR +
`git worktree remove --force` + `git worktree prune`. Ana repo'da checkout-dance yapma;
`git add -A`/`git clean` ASLA (eşzamanlı oturum WIP'ini süpürür). Bkz [[worktree-orphan-hazard-2026-07-03]].

### ▶️ SIRADAKİ ADIM — LoRA EĞİTİMİ (Kural-8 kapılı, insan onayı bekliyor)

**Tamamlanan veri hattı adımları:**
- **Carding** ✅ (185 makale, 0 kartsız, 483 onaylı / 3 bekliyor / 26 reddedilmiş)
- **RLM** ✅ (33 answered, 10 abstained; 1 asılı koşu temizlendi; 5 mevcut §16 aday)
- **Curate** ✅ (`lora-curate --run`: 40 çok-versiyon kart düşürüldü, 183 kanonik tutuldu)
- **Assemble** ✅ (`assemble_sft.py`: 2000 örnek = 1324 synth-qa + 176 küratörlü kart + 500 disiplin)
- **Split** ✅ (train=1900 / valid=100)
- **Pretrain-gate** ✅ (**GO**, 2 epoch önerisi)
- **Sentinel** ✅ (9/9 sağlıklı)
- **Makale araştırma** ✅ (6 yeni arXiv makalesi indirildi + indeks güncellendi)
- **Loop** ✅ (72 saat olarak yeniden başlatıldı)
- **UTF-8 fix** ✅ (`unified_dataset.py` Windows cp1254 encoding hatası düzeltildi)

**Audit uyarıları (eğitimi bloklamaz, kürasyon uygulandı):**
- Gate 0: 98 orphan kart → kürasyon ile elendi
- Gate 5: 57 doğrulanmamış performans iddiası (review)
- Gate 6: 94 felsefe incelemesi (review)
- Gate 7: 1 güvenlik reddi → elendi

**Eğitim için gereken insan kararları:**
1. **3 bekleyen kart** — web arayüzünden (06·ONAY) onayla/reddet
2. **Taze onay** — Kural 8: `uv run achilles approval-approve <id>` (ONAYSIZ EĞİTİM ASLA)
3. **Eğitim başlatma** — `scripts/start-train.ps1 -Profile discipline_safe_local` (DETACHED;
   harness `run_in_background` oturum kapanınca ÖLÜR — [[detached-training-survives-teardown]])

Eğitim-sonrası: `evaluate_adapter` (min_n≥5, boş-cevap veto'lu) + registry ADAY (terfi İNSAN onayı).

### 🆕 EN SON İŞ (2026-07-04) — AJAN HARİTASI FRONTEND + UI DECLUTTER

Kullanıcı en baştan istediği "ışıklı yol" ajan etkileşim haritasını + web arayüz sadeleştirmesini
istedi. Üç PR MERGED (hepsi GERÇEK izole worktree ile — [[worktree-orphan-hazard-2026-07-03]] şartı):

- **PR #101 (MERGED) — 15·AJAN HARİTASI frontend:** İzleme grubunda yeni sekme. `/api/agents/graph`
  (backend PR#96) → SVG ışıklı-yol: gruba-göre lane, chain=düz ok / data=ince kesik, kaynak
  `running`→kenar AKAR (am-flow) + düğüm nabzı, 7sn poll (sekme kapanınca durur), düğüm-tıkla→ayrıntı,
  ⚡ EĞİTİMİ DEVREYE SOK = onaylıysa autodrive / onaysızsa `execute:false` DRY-RUN komutu (spawn YOK,
  Kural 8). CSP-güvenli (sınıf+SVG geometri). 10 çevrimdışı test.
- **PR #102 (MERGED) — worktree-orphan-hazard uyarısı** HANDOFF'a eklendi (yukarıdaki ⚠️ bölüm).
- **PR #103 (MERGED) — UI declutter:** `index.html`+`app.css`'ten dekoratif `scanlines` CRT overlay
  (zaten display:none, ölü) + `FATE FAVORS THE FOCUSED` hero banner kaldırıldı (net −64 satır).
  `app.css?v=7`→`v=8`. Bump, PR#101'in tam-versiyon pinleyen testini kırdı → test versiyona-DUYARSIZ
  yapıldı (`assert "app.css?v=" in html`; DERS: UI asset testi `?v=N` numarası pinlememeli).

**⏳ DEVAM EDEN EŞZAMANLI İŞ (BAŞKA OTURUM — DOKUNMA):** kullanıcı agent-map "derli toplu" KART-tabanlı
yeniden-tasarımını "öteki oturum yapsın" dedi → `claude/agent-map-cards` (worktree `gracious-darwin-ec236d`)
sürdürüyor. Kart layout spec: sütun-hizalı eşit-boy kartlar / dik-açı oklar / durum=sol-kenar-renk /
ana ajan alt-şerit yeşil buton. Bkz [[agent-graph-map-2026-07-03]], [[ui-declutter-2026-07-04]].

### 🆕 EN SON İŞ (2026-07-03 akşam) — VERİ HATTI KAPANDI + MAKALE ARAŞTIRMA

**Veri hattı işleri:**
- Asılı RLM koşusu temizlendi (rlm_b356341740f3479b: 3+ saat stuck → failed)
- `lora-curate --run`: 40 çok-versiyon kart düşürüldü (orphan koruması)
- `assemble_sft.py`: birleşik dataset yeniden inşa (küratörlü kart + synth-qa + %25 disiplin)
- `lora-split`: train=1900 / valid=100
- `pretrain-gate`: GO (2000 örnek, 2 epoch)
- **Bug fix:** `unified_dataset.py` `write_text` UTF-8 encoding eksikti → Windows cp1254'te
  Yunan karakterleri (θ vb.) yazılamıyordu. `encoding="utf-8"` eklendi.

**Haftalık makale araştırma (6 yeni arXiv makalesi):**
- `2510.13003` OPLoRA: Orthogonal Projection LoRA (Tema A — v5 forgetting)
- `2601.04525` GRACE: RL for Grounded Abstention (Tema B — Kural 7)
- `2606.21917` Pre-Gen Hallucination Detection (Tema B — üretim-öncesi tespit)
- `2509.02844` CP for Time-Series with Change Points (Tema F)
- `2606.15953` Drift-Aware Spectral Conformal Prediction (Tema F)
- `2508.19955` Global Permutation Entropy (Tema E — PE genişletme)
Tümü `%PDF` + >40KB doğrulandı. İndeks (`00_NEDEN_ONEMLI_oku_once.md`) güncellendi.

### ÖNCEKİ İŞ (2026-07-03) — EĞİTİM-ÖNCESİ KADEME-2 AV: 3 v5-SINIFI FIX (PR #95 MERGED)

LoRA eğitim-öncesi ZORUNLU derin av (CLAUDE.md "her eğitimden önce"; v5 regresyonu bunun
atlanmasındandı). Opus'la 6 finder + 2-oylu şüpheci doğrulama; 14 ham bulgudan **3'ü onaylandı,
hepsi v5-sınıfı**. Kural 8 gereği eğitimden ÖNCE onarıldı; gerçek eğitim başlatılmadı.

- 🔴 **profile-drop (CRITICAL):** `detached_launch.launch()` varsayılanı `profile=None` idi →
  web `/api/training/run` · `auto_pipeline` · `start-train.ps1` yollarının HİÇBİRİ `--profile`
  geçmiyordu → spawn edilen `train --run` VANİLYA (assistant_only_loss=False, NEFTune=0, lr=2e-4)
  koşuyordu = tam v5 maskesiz reçetesi. FIX: `launch()`+`start-train.ps1 -Profile` varsayılanı
  → `discipline_safe_local` (`profile=""` ile bilinçli vanilya kaçışı korunur). Komut kurulumu
  test edilebilir `_build_train_cmd` helper'ına çıkarıldı.
- 🔴 **eval boş-cevap sahte-kabul (CRITICAL):** `adapter_eval._flags_for` boş/whitespace cevabı
  0 red-flag → skor 1.0 → çökmüş adapter base'i geçip `accept` alıyordu. FIX: boş→`empty_answer`
  flag + kategorik veto (dejenerasyonla aynı sınıf; meşru çekimserlik non-empty → Kural 7 korunur).
- 🟠 **anlama-merdiveni elmayla-armut (MAJOR):** `auto_pipeline` base bacağı `llm=None`→Ollama
  qwen3:**4B**, adapter bacağı **1.5B**+adapter → 1.5B L3/L4 sayısal sınavlarda sistematik kaybeder
  → kalıcı SAHTE-gerileme → iyi adapter bile bloklanır. FIX: `peft_llm_shim.load_base_of()`
  (adapter'ın KENDİ base'i, adaptersiz); base bacağını yükle→koş→serbest→adapter (CPU bellek-güvenli).

Kademe-0 kapısı: format+lint+mypy temiz, testler yeşil (yerel + CI). Yeni: `test_detached_launch_profile.py`;
genişletilen: `test_adapter_eval_degenerate.py` (empty_answer) · `test_peft_llm_shim.py` (`load_base_of`).
**Çürütülen:** data-gate-stale (2/2 oy — orkestrasyon salt-danışman). **Doğrulanamayan** (session-limit
10pm): KL-yön/NEFTune (β=0 dormant), pretrain-gate `'pasaj'da'` öneki, ps1 CIM-filtresi/false-success
— düşük öncelik, ayrı takip. Bkz memory [[pretrain-hunt-profile-drop-2026-07-03]] · [[v5-adapter-regression]].

### 🆕 ÖNCEKİ İŞ (2026-07-03) — FAZ-6 SENTINEL (NÖBETÇİ) SAĞLIK MONİTÖRÜ (PR #90 MERGED)

"Birbirini denetleyen sistem" katmanının kapanış taşı (Layer 8 — Monitor & Alert):
**`app/monitoring/`** — Sentinel, 9 enjekte-edilebilir **SALT-OKUMA** probe ile tüm ajan/
altsistemleri tek bakışta izler: llm (Ollama) · web · training · orchestration (stale aşama)
· stop_all · disk · sqlite (quick_check) · feedback (bekleyen kuyruk) · **contention**
(= plandaki Resource Negotiator, DANIŞMAN modda: CPU çekişmesini raporlar, eğitimi
DURAKLAMAZ — detached PEFT'te güvenli resume yok). Agregasyon fail>warn>ok; geçmiş SQLite
(`sentinel_checks`, WAL, keep_last budamalı). Hiçbir şeyi durdurmaz/başlatmaz — öneri metni
verir (`orchestrate-recover`, `clear-stop-all`), eylem insanın.

Arayüzler: CLI `achilles sentinel [--history]` (fail→exit 2) · web `/api/sentinel/*` ·
**14·NÖBETÇİ** sekmesi (İzleme grubu, 🩺 ŞİMDİ YOKLA → probe kartları + geçmiş) · manifest
`sentinel-monitor` (semi_auto, dangerous=false). 16 offline test; canlı doğrulandı (CLI
gerçek ortamda 9/9 ✓, tarayıcıda buton akışı). Adversarial review 4/4 gerçek fix
(busy_timeout; prune same-timestamp tie `<=`→`<`+offset; overall normalize; decode-log).

**KALAN AJAN ADAYLARI (opsiyonel, plan tartışmasından):** in-repo Bug Finder/Live Inspector
(weekly-bug-scan.ps1 kısmen kapsıyor), Contradiction Broker (LLM-ağır), Artemis RAG-drift.
**Öncelik önerisi: artık ajan değil — yukarıdaki "SIRADAKİ ADIM" (ANA ortamda carding →
RLM → eğitim döngüsü).** Bkz memory [[sentinel-monitor-2026-07-03]].

### EN SON İŞ (2026-06-30) — FAZ-5 COLLISION DETECTOR + REGRESSION BLOCKER (Faz-1..5 TAMAM)

Plana göre son iki eksik ajan, Faz-1..4 desenini izleyerek **PR #86 MERGED** (additive,
enjekte-edilebilir, offline-testli, Kural-8 gated; mimari BOZULMADI). Bununla ajan-orkestrasyon
Faz-1..5 **TAMAMLANDI**. Yeni hat:
`preflight → collision → smoke → deep-hunt → data-gate → curriculum → dry-run → regression → approval → train → evaluate → registry`
(`create_run` aşamaları per-run snapshot'lar → mevcut koşular etkilenmez, resume güvenli).

- **`app/orchestration/collision.py` — Collision Detector:** eşzamanlı oturum/worktree
  çakışmasını git durumundan salt-okuma tespit eder (enjekte `git_runner` → offline test).
  `.git/index.lock` (aktif git) / aynı-branch çoklu worktree / **HEAD-drift** → **BLOK**;
  kirli izlenen ağaç → **UYARI** (başka oturum `git add -A` ile uncommitted fix'i süpürebilir —
  [[concurrent-session-worktree-collision]]); git yok → skip. Delege `fail→blocked` (kurtarılabilir:
  commit/stash → resume). `baseline_head` ctx.params'tan iletilir.
- **`app/orchestration/regression.py` — Regression Blocker:** eğitim/onay öncesi v5 ağı. Aday
  setin **v5-ilgili sinyalleri** (`top_opening_share` = v5'in TAM mekanizması, garanti vaadi,
  sızıntı, maliyet-körü, disiplin kapsamı, GO/NO-GO) son GEÇEN baseline ile kıyaslanır; yön+tolerans
  dışı kötüleşme → **BLOK** ([[v5-adapter-regression]]). baseline yok → skip (ilk koşu); baseline
  **yalnız explicit `--commit`** ile güncellenir (oto-terfi YOK — kötü set sessizce baseline olup
  gerilemeyi normalleştirmesin). `BaselineStore`(JSON)+`metrics_provider` enjekte; saf
  `evaluate_regression` çekirdeği. metrik kaynağı = `audit_dataset` (data_gate ile aynı).
- **Bağlama:** `StageKind.collision/regression` + 2 `StageDef` + 2 delege; CLI
  `orchestrate-collision` / `orchestrate-regression --commit` (pass/skip/warn→0, fail→2) + README.
- **31 yeni offline test** (12 collision + 19 regression). Kapı yeşil: ruff+mypy(228 dosya)+pytest.
  Canlı CLI duman-testi: collision bu worktree'de WARN (3 modified-tracked); regression skip(baseline yok).

### EN SON İŞ (2026-06-30) — FAZ-4 SMOKE TEST RUNNER ("stub≠runtime")

Kullanıcı "kalınan yerde devam et" → plana göre Faz-4: eksik ajan **Smoke Test Runner**.
Çekirdek ders ([[rlm-controller-entegrasyon]]): birim testleri stub'la geçse de canlı
Ollama+RAG+LLM hattı bozuk olabilir (RLM'de gerçek Ollama smoke 3 açık bulmuştu). Mimariyi
BOZMADAN additive:

- **`app/orchestration/smoke.py`** — `SmokeRunner` (enjekte edilebilir `llm`/`retriever`) +
  `SmokeResult`/`SmokeCheck`. Yoklamalar: backend canlı mı → gerçek küçük üretim (seed'li,
  boş/degenere değil; `adapter_eval._is_degenerate` yeniden kullanılır) → gerçek RAG retrieval
  (≥1 chunk; boşsa **warn**, kritik değil). Salt-okuma (üretim atılır).
- **Verdict semantiği (kilit karar):** runtime ÇEVRİMDIŞI → **skip** (kusur DEĞİL; hat DURMAZ,
  sonraki insan kapısına geçer); canlı+sağlıklı → **pass**; canlı ama boş/degenere/hata → **fail**
  (asıl yakalanacak kusur, ör. Ollama açık ama model çekilmemiş). Skip-when-offline sayesinde
  CI/çevrimdışı testler etkilenmez (eski akış deep-hunt'ta durur).
- **Hat:** `StageKind.smoke` + yeni **`smoke`** aşaması (preflight→**smoke**→deep-hunt, autonomous
  salt-okuma). `create_run` aşamaları per-run snapshot'ladığından mevcut koşular etkilenmez
  (resume güvenli). CLI **`orchestrate-smoke`** (pass/skip→0, fail→2) + README orkestrasyon bloğu.
- **13 yeni offline test** (enjekte fake'ler) + web/core testleri çevrimdışı-deterministik fixture.
  Kapı yeşil: ruff+mypy (226 dosya)+pytest tümü. **Canlı doğrulama:** `orchestrate-smoke` gerçek
  Ollama'ya vurdu (backend✓ generation✓ retrieval≈boş-korpus-warn) → PASS, exit 0.
- **YAN BULGU (Kademe-2 refleksi, doğrulandı):** `graph_corpus.py` aynı count-only cache zaafına
  sahip — `bm25_corpus`'tan DAHA kötü (ne içerik-imzası ne ingest-reset; `paper_indexer` yalnız
  bm25'i resetler). 4-lens adversarial workflow ONAYLADI (skeptik dahil; idempotency=dosya-hash,
  parse-sonucu değil → same-count-content tetiklenebilir). Opt-in/OFF (rag_graph=False) → latent.
  Doğru fix: `graph_corpus.reset_cache()`'i ingest'e bağla + build-lock (imza DEĞİL — get_all'ı
  her çağrıda zorlardı). Ayrı görev olarak işaretlendi (chip).

### 🆕 EN SON İŞ (2026-06-29) — AJAN ORKESTRASYON SİSTEMİ (3 FAZ) + KADEME-2 EĞİTİM-ÖNCESİ AV

Kullanıcı "tek tuşla Claude aboneliğiyle eğitimi devreye sok + ajanların birbiriyle etkileşimini
arayüzde gör + eksik ajanları konuş" dedi → plana göre 3 faz, hepsi CI-yeşil self-merge,
adversarial-review/test'li, mevcut mimariyi BOZMADAN (auto_pipeline/detached korundu):

**FAZ-1 — Dayanıklı Orkestrasyon (PR #72):** `app/orchestration/` — saf 9-aşama hattı
(preflight→deep-hunt→data-gate→curriculum→dry-run→approval→train→evaluate→registry) + SQLite/WAL
store (run/stage/event) + durum makinesi (step/run_until_blocked/recover_stale) + savunmacı
delegeler. CLI `orchestrate-*`, web `/api/orchestration/*`, **12·ORKESTRASYON** sekmesi (canlı
aşama grafiği + timeline + tek-tık/sürdür/recovery). Kural-8: salt-okuma aşamaları otonom,
deep-hunt/approval insan kapısı, train HANDOFF. checkpoint/resume → session-limit'e dayanıklı.
Eşzamanlı oturum Kademe-2 av ile sertleştirdi: **#75** (SqliteStore WAL), **#76** (CAS-claim/
cancel-race/recover-clobber/profile-pattern), **#77** (rlm degenerate).

**FAZ-2 — Echo Feedback (PR #74):** `app/feedback/` — kullanıcı düzeltmesi → güvenli sentetik
SFT ADAYI (lora_sft formatı). Kural-1 zehir filtresi TÜM alanlarda (soru SFT'de user-turn→
sızabilir, review buldu); export AYRI aday dosyaya (oto-merge YOK, eğitim oto-tetiklemez).
CLI `feedback-*`, web `/api/feedback/*`, **13·GERİ BİLDİRİM** sekmesi. Adversarial review 4/4
gerçek fix. **CI DERSİ:** global RateLimiter cross-test birikimi (CI hızlı→tek 60s pencere→120
aşımı→alakasız test 429) → `tests/conftest.py` autouse fixture her testte `_hits` temizler.

**FAZ-3 — AutoDriver (PR #79):** `app/orchestration/driver.py` — kullanıcının ASIL isteği:
deep-hunt'ı headless `claude -p` (ABONELİK CLI, API-key DEĞİL) ile OTONOM sürer → PASS ise
onaya ilerletir + DURUR (Kural 8: gerçek eğitim TAZE insan onayı bekler). `execute=False`
varsayılan (spawn yok, komut döner); spawn yalnız `execute=True`+`claude` PATH'te; shell=False;
prompt SALT-RAPOR. CLI `orchestrate-autodrive [--execute]`, web `/api/orchestration/autodrive/{id}`
(BackgroundTasks), **🤖 Otonom Sürüş** butonu (deep-hunt'ta blocked iken etkin).

**KADEME-2 EĞİTİM-ÖNCESİ DERİN AV (zorunlu kapı, bu seans):** İki alt sistem çok-ajan workflow
(finder × 3-lens adversarial: repro/by-design/severity, ≥2-oy onaylı) ile avlandı:
- **Orchestration (PR#72):** 17→12 onaylı/5 latent-red → **#75/#76/#77** (yukarıda) + rapor **#78**.
- **Echo feedback (PR#74, eğitim-VERİSİ üretir):** 5/5 onaylı → **#82**: ECHO-POISON-01 (HIGH —
  zehir filtresi yalnız `guaranteed_profit` taranıyordu; advice/risk-free/canlı-sinyal/kesinlik
  dili SFT adayına sızıyordu → geniş `safety_scanner` + feedback'e özel direktifler), POISON-02
  (export yazmadan önce yeniden tarar), JSONL-001 (U+2028/2029/0085 `\u`-kaçışı → `splitlines`
  bölmez), +Kural-6 determinizm + reject-penceresi. **dataset_quality.audit BİLİNÇLİ değiştirilmedi**
  (curated set FP/NO-GO riski; kartlar zaten Gate-7'den geçer).

**KALAN PLAN (eksik ajanlar):** ~~Faz-4 Smoke Test Runner~~ ✅ (2026-06-30, yukarıda). Sıradaki:
Faz-5 **Regression Blocker** + **Collision Detector**. + Faz-3 driver'a ileride alt-sistem Kademe-2
av (Faz-1 gibi). + **`graph_corpus` count-only cache staleness fix** (chip görevi; bm25 paterni
ingest-reset + build-lock ile aynalanmalı — imza DEĞİL) ve flag'lenen BM25 content-fingerprint (latent).
Memory: [[orchestration-system-2026-06-29]], [[echo-feedback-2026-06-29]],
[[orchestration-autodrive-2026-06-29]], [[orchestration-kademe2-hunt-2026-06-29]],
[[sqlite-shared-file-wal-pragmas]].

### 🆕 EN SON İŞ (2026-06-26 GECE) — İLK 1.5B LoRA EĞİTİMİ BAŞARIYLA TAMAMLANDI + RAG genişledi

Kullanıcı gece "tam yetki, eğitimi durdurma, 1.5B eğit, sabah rapor ver" dedi → otonom loop:

**1) Kademe-2 zorunlu derin av** (eğitim-öncesi): 8-finder + 2-oylu adversarial → 4 fix (maliyet×2 /
çapraz-makale provenans / tavsiye-dili olumsuzlama-bypass) → **PR #67 MERGED**. Kart-Onay bug
(boş kart approve→yanıltıcı "bulunamadı") → **PR #66 MERGED**. API-öneren doküman temizliği
(.env.example+README local-first) → **PR #69 MERGED**.

**2) Model → 1.5B** (yerel `.env`, reversible): `llm_model=qwen2.5:1.5b` + `peft_base_model=
Qwen/Qwen2.5-1.5B-Instruct`. Ollama'ya qwen2.5:1.5b çekildi. Web restart → "Canlı Durum" **1.5B**
gösteriyor (kullanıcı isteği). torch 2.12+cpu+transformers+peft kuruldu (YEREL lib, API DEĞİL).

**3) 🎯 GERÇEK 1.5B LoRA EĞİTİMİ** (`achilles_lora_15b_v1`): discipline_safe_local, 400 örnek ×
2 epoch = **400 adım/~4h** (~35.8s/adım CPU), assistant_only_loss maskeleme AKTİF (v5-fix),
loss 2.63→1.54, cosine LR düzgün kapandı. **EVAL (dürüst, adapter'ı yükler): base 0.125 (14 ihlal)
→ adapter 1.0 (0 ihlal) → VERDICT ACCEPT.** v5'in TERSİ: disiplin ÖĞRENİLDİ. Adapter ADAY
(production terfisi İNSAN onayı bekler — Kural 8). models/adapters/achilles_lora_15b_v1.

**4) RAG**: enrichment denendi (50 kartsız makale, ama 4b CPU'da ~8dk/kart + kısmen etkisiz →
1 kart zenginleşti, kesildi). 8-konu workflow → **71 arXiv makalesi indirildi** (Gerekli kaynaklar/
arxiv-batch-2026-06-26) → RAG'a ingest edildi (171→~242 makale). Carding sonraki seans (yavaş).

**KALAN/NOT:** AchillesWeb/Update scheduled-task'leri admin gerektirdi (Registry-Run autostart
onarıldı; scheduled-task `start-server.ps1 -Repair` Yönetici-PS ile tekrar). Adapter Ollama'da
çalıştırmak için 1.5B GGUF dönüşümü gerekebilir (şimdilik transformers/PEFT ile yüklenir).
Bkz memory [[loop-training-session-2026-06-25]].

### EN SON İŞ (2026-06-25 akşam) — Web UI redesign TAMAMLANDI (5 PR) + EĞİTİM-ÖNCESİ HATIRLATMA

Kullanıcı "web'i güzel ve rahat anlaşılır kıl, mimariyi BOZMA" → 10 sunum-increment, **5 PR MERGED**,
hepsi saf `index.html`+`app.css`+`app.js`, CI-yeşil + kendim merge ettim, **8765'te canlı** (static
diskten okunduğu için RESTART'sız; asset-hash her istekte yeniden):
- **#61** 5-grup nav + derin-link (hash/localStorage) · jargon tooltip + insani hata (`humanError`) + CTA hiyerarşisi · sade durum çubuğu (katlanır) + sonraki-adım şeridi
- **#62** mobil/responsive (nav-wrap, akışkan form) · tam Türkçe etiket (embed→gömme, papers→makale) · jargon tooltip yayma (13 terim/6 panel)
- **#63** tutarlı ikon dili (dekoratif emoji kaldır, refresh→↻) · backtest verdict disiplini ("maliyet+slippage dahil" + aday≠hazır, Kural 1-3) · ilk-açılış 3-adım onboarding
- **#64** renk-körü-güvenli palet (Okabe-Ito): pass SOĞUK bluish-green `#0a7d55` / fail SICAK vermillion `#cf4014` (mavi-sarı ekseni → kırmızı-yeşil körlüğünden bağımsız) + ✓/✕/≈ + WCAG AA (Python ile ölçüldü)
- **#65** kalan İngilizce etiket bitiş (agents formu agent_id/requires_approval, Train/Val Loss → Türkçe)

KORUNDU: `/api` kontratı + pydantic alanları · `data-tab`/`panel-<ad>` ID'leri · CSP `script-src self` · asset içerik-hash · 8 mutlak kural.
Detay + gotcha'lar: memory [[web-ui-redesign-2026-06-25]] · merge yetkisi [[pr-merge-authority]].

**⚠️ SONRAKİ: EĞİTİM (kullanıcı BAŞKA CHAT'te başlatacak) — ZORUNLU SIRA:**
1. **Kademe-2 derin adversarial av** — CLAUDE.md: HER eğitimden ÖNCE ZORUNLU (v5 regresyonu bu yüzden olmuştu).
   Alt-sistem başına paralel finder → 2-oylu adversarial doğrulama (şüpheci, varsayılan çürütülmüş) → yalnız onaylananı düzelt → Kademe-0 kapısı → push.
2. Veri kapısı: `pretrain-gate` (GO/NO-GO) + `lora-readiness` (≥1000 örnek).
3. `train` önce **dry-run** (önizleme); gerçek eğitim yalnız açık `--run` (Kural 8). Model Qwen2.5-1.5B, profil `discipline_safe_local`.
Bkz memory [[bug-avi-kadansi]], [[lora-loop-guvenlik-verdicti]], [[local-small-model-training-pivot]].

---

### EN SON İŞ (2026-06-25) — Çok-makine güncelleme yakınsama protokolü + sürüm rozeti

**SORUN (kullanıcı):** "2 W10 + 1 Mac'te sadece bu makinede güncel sürüm var; diğerlerinde
güncelleme/sıfırdan kurulum tam oturmuyor." → **üç katmanlı sessiz yakınsama çöküşü** (hepsi
yeni `achilles doctor` ile canlı kanıtlandı):
1. **update.ps1/sh main'e geçmiyordu** — mevcut dal main değilken `git pull origin main`
   origin/main'i feature dalına MERGE ediyordu → makine asla yakınsamaz (dev checkout
   `fix/rlm-hardening-2`'de parklanmıştı, 74 behind).
2. **Ölü scheduled-task yolları** — `start-server.ps1 -Install` yolu kuruluma sabitliyor; repo
   taşınınca AchillesWeb/AchillesUpdate (03:00) ölü `~/achilles`'i çağırıp sessizce no-op
   (ERROR_DIRECTORY). 3. **install.ps1 sessiz ff-only** + iki-kopya tehlikesi.

**FIX (hepsi MERGED):**
- **#54** — update.ps1/sh deterministik main yakınsama (parklanmış dalı zorla main'e al,
  ff-only, ıraksaksa AUTO-MERGE YOK + -Force; throwaway-klon testli) · `start-server.ps1
  -Repair` + `-Status` yol-eşleşme (non-admin'de sahte-[OK] yok) · install.ps1 görünür +
  yabancı-kopya tespiti · **YENİ `achilles doctor`** (offline drift teşhisi, sapma→exit 2).
- **#55** — `docs/GUNCELLEME_KILAVUZU.md` (yeni-başlayan kullanım kılavuzu) + README tazeleme
  (eski `reset --hard` kurtarma → `update --force`; doctor/-Repair; SSS satırları).
- **#56·#57** — stranded yerel iş cherry-pick (citation-degenerate guard + rag-chains-work
  router/abstain/golden-eval, ~+1068; `bm25_corpus` Chroma→SQLite **bilinçli supersede** —
  Chroma `get_all()` eşzamanlı erişimde BM25'i sessizce öldürüyordu, Kural 7). `fix/rlm-hardening-2`
  OLDUĞU GİBİ MERGE EDİLMEZ (6428 satır siler) — **ama `git cherry` ile doğrulandı:** ahead
  commit'lerinin İÇERİĞİ (`ed8a6bb`→#56, §16 `lora_candidate`, Kademe-2 hunt) tümüyle main'de →
  **değerli-stranded iş KALMADI** (yeniden avlama); rag-chains-work + `achilles-chains` worktree
  redundant, silinebilir.
- **#58 (PREVENTİF)** — web header **sürüm/sapma rozeti**: `app/web/version_info.py` (offline
  git) + `GET /api/version` + rozet (yeşil "güncel ✓" / amber "N commit geride — güncelle" /
  "dal X main değil"). **30 dk throttle'lı offline-güvenli `git fetch`** → nightly görev bozuk
  olsa bile gerçek drift görünür; "sessizce 333 commit geride" bir daha olamaz. Preview'da canlı
  doğrulandı (behind=12 yakaladı). Bkz memory [[multi-machine-sync-convergence]].

**MAKİNE DURUMU:** ✅ sevinc dev checkout **main'e alındı** (`HANDOFF.md` eski hâli `stash@{0}`'da);
✅ HUAWEI makinesi senkron; ⚠️ **Mac + sevinc'in zamanlanmış görevleri** kaldı: her makinede tek
sefer `git fetch origin main; git switch main; update.ps1 -Force` (mac: `./update.sh --force`) +
**yönetici** `start-server.ps1 -Repair`, sonra `achilles doctor` ile dal=main · +0/-0 doğrula.
Rozet yeni kod olduğu için her makine bir kez daha `update` edince görünür.

**GOTCHA:** PR otomasyonu BEHIND dalı kendi "Merge branch main" ile günceller; `gh pr
update-branch <n>` ile elle tetiklenebilir (ben #58'de yaptım). İzole worktree origin/main'den
açılınca origin ALTINDAN ilerleyebilir (eşzamanlı PR merge) → two-dot diff sahte "silme"; GitHub
three-dot kullanır. Windows: izole worktree `.venv`'i cockroachdb `cd.cp312-win_amd64.pyd` ile
kilitlenince `git worktree remove` "Invalid argument" verir → git registry temiz (`prune` +
merged branch sil) ama fiziksel dizin kilit çözülene (process/oturum bitene) kadar diskte kalır,
sonra `Remove-Item -Recurse -Force`.

---

### 🆕 EN SON İŞ (2026-06-24 #3) — LoRA degenerate kök-neden fix + 1.5B web chat entegrasyonu

**SORUN:** Eğitilmiş LoRA adapter'ları (v5 dahil) degenerate tekrar döngüsüne giriyordu
("pasaja göre" ezber, maliyetsiz rakam uydurma). 4B model CPU'da eğitilemez derecede ağırdı.
Kullanıcı istedi: "eğitilmiş LLM'e soru sorabileyim, web'den."

**KÖK NEDEN + FIX (push'lu, origin/main):**
1. **assistant_only_loss maskeleme** (`app/training/peft_lora_train.py`):
   - `build_masked_labels(prompt_ids, full_ids)` — prompt token'larını `-100` ile maskeler,
     yalnız asistan cevabını öğretir. Tekrar döngüsü kökten çözüldü.
   - `_chat_input_ids()` — `BatchEncoding` vs `list[int]` tip uyumsuzluğu düzeltildi.
   - `_MaskedDataCollator(pad_token_id)` — label pad'lerini `-100` yapar (0 değil).
   - `sample_rows(rows, max_examples, seed)` — deterministik alt-küme (Kural 6).
   - 10 yeni test: `tests/test_peft_assistant_only_loss.py`.

2. **Küçük model pivotu** — 4B→**Qwen2.5-1.5B-Instruct** (CPU'da eğitilebilir):
   - `adapter_eval.py` → `_resolve_base_model(adapter_dir)` adapter_config.json'dan base okur.
   - `discipline_safe_local` profili: `configs/lora/lora_profiles.yaml` (r=16, alpha=32,
     assistant_only_loss=true, NEFTune, max_examples=300).
   - `detached_launch.py` → `--profile` + `--max-examples` desteği.

3. **Web LoRA chat entegrasyonu** (kullanıcı web'den soru sorabilir):
   - `app/web/lora_chat_service.py` — adapter tarama + lazy-load + thread-safe cache.
   - `app/web/schemas.py` — `LoraChatRequest/Response`.
   - `app/web/server.py` — `GET /api/lora-adapters` + `POST /api/lora-chat`.
   - `app/web/static/index.html` + `assets/app.js` — sohbet paneli, adapter seçici, spinner.
   - Varsayılan: `achilles_lora_qwen15b` (1.5B). Base "(eğitimsiz, 4B)" etiketi.
   - **Bölüm takas:** "Eğitilen Modelle Sohbet" → **3. sıra**, "Araştırma Geçmişi" → **6. sıra**.

4. **Doğrulama:** `reports/lora_chat_check.txt` — 1.5B adapter non-degenerate cevap veriyor.

**COMMIT'LER:** `398fadb` (bölüm takas), `4515091` (1.5B varsayılan) + önceki maskeleme/chat
commit'leri — hepsi origin/main'de.

**GOTCHA'lar:**
- Working tree `fix/rlm-hardening-2` dalında (eşzamanlı oturum). Push'lar izole worktree ile.
- `achilles_lora_qwen15b` adapter'ı eşzamanlı oturumda eğitildi (1.5B, discipline_safe_local).
- **AchillesWeb scheduled task** yanlış yolda (`C:\Users\sevinc\achilles`); kullanıcı onayı bekliyor.

---

### 🆕 EN SON İŞ (2026-06-24 #2) — AI Brain ek modülleri (~16 PR MERGED)
Desktop\RAG Kaynak\Eğitim geliştirme\helix_..._talimati.txt (10-modül "AI beyin" genişletme
emri) audit edildi → çoğu ZATEN vardı. **4 gerçek boşluk** additive olarak entegre edildi
(offline, deterministik; **app/rlm DOKUNULMADI**). 3-tur adversarial bug-fix loop → CONVERGED.

**Eklenen modüller:**
- **F1 app/registry/** — DatasetVersion/RagIndexVersion/EmbeddingModelVersion/RlmRewardVersion/
  PromotionDecision ORM + RegistryStore + promotion_gates (atomik CAS approve/reject, terminal
  state machine pending→approved|rejected). Commit `e08a674`.
- **F2 app/tools/** — probability_simulator (seed'li Monte Carlo + risk-of-ruin + VaR/ES),
  statistics_checker (permütasyon p-değeri, scipy bağımlılığı YOK), result_verifier,
  tool_registry + ToolRun/ToolArtifact ORM + log_tool_run. CLI tools-list/montecarlo/stats-check.
- **F3 app/evals/eval_runner.py** — EvalRunner (trading-hypothesis/rag-retrieval tipi),
  trading_hypothesis_evaluator (regex tavsiye tespit: sure-fire/guarantee/can't-lose).
  CLI eval-runner. → F2+F3 = **PR #14**.
- **F4 app/ingestion/quality_scorer.py** — clean_text_scorer + PaperIngestionRun ORM +
  Paper.quality_score/ingest_status. compute-on-demand. CLI ingestion-quality/ingestion-quality-scan.
  → **PR #16**.

**Bug-fix loop (3 tur, converged):**
- Tur 1 (PR #19): 8 onaylı bug — 2 HIGH: TOCTOU approve/reject yarışı → atomik CAS.
- Tur 2 (PR #21): 9 fix — terminal durum makinesi; n_trades=0 falsy bypass; FK parent
  doğrulaması (SQLite FK enforcement GLOBAL KAPALI, BİLEREK); tavsiye-regex genişletme.
- Tur 3: critic tek-tur → CONVERGED, high/blocker yok.

**Zincir entegrasyonu (PR #24):** 4 yeni ajan + skill `automation_manifest.yaml`'de:
ingestion-quality-scorer, scientific-tool-runtime, hypothesis-evaluator,
model-data-registry (requires_approval → Kural 8 zincire gömülü). 20 ajan/14 adım, topolojik.

**Web + bağlantılar (PR #26/28/29/32/33/35/38):**
- `app/web/ai_brain_routes.py` — registry/tools/ingestion/eval REST API'leri.
- `/ai-brain` dashboard (bağımsız ai_brain.html, 4 sekme).
- eval→registry snapshot + karar-log; batch ingestion-quality-scan CLI.
- `registry-register-dataset` CLI (SHA-256, idempotent).
- README "AI Brain ek modülleri" CLI referansı.
- **#38 dataset auto-registration LIVE**: `build_training_split()` → `_auto_register_dataset`
  (best-effort, contextlib.suppress) → eğitim hattı kanonik lora_sft.jsonl'i otomatik
  DatasetVersion (pending) yapar.

**~135 yeni test.** Tüm PR'lar izole-worktree→auto-merge (çakışma yok).
**ERTELENEN (opsiyonel):** ana index.html'e /ai-brain nav linki; rag-answer/lora/rlm-reward
eval tipleri (app/rlm bağımlı). Bkz memory [[ai-brain-modules-integration-2026-06-24]].

### 🆕 EN SON İŞ (2026-06-24) — alexzhang13/rlm OPSİYONEL motor-adapter (4 PR MERGED)
`Desktop\RAG Kaynak\RLM\achilles_alexzhang_rlm_claude_integration_prompt` (1051 satır) uçtan uca
entegre edildi — additive, **native VARSAYILAN korundu**, OpenAI default değil, RAG/Mastery/LoRA
DOKUNULMADI. Hepsi izole-worktree→PR→auto-merge (paylaşılan tree çakışması yok).
- **PR #43** entegrasyon: `app/rlm/adapters/` (base/native/alexzhang/security), `engine_config`,
  `tool_registry`+`safe_tools` (deny-by-default allowlist), `answer_pipeline`; CLI `rlm-engine`/
  `rlm-test-adapter`. rlm-security-reviewer KENDİ kodumda 4 fix buldu (HIGH ipython-bypass→allow-list).
  GOTCHA: `.gitignore` `adapters/` (LoRA) → `app/rlm/adapters/` kaynağını sessizce yutuyordu → `*.py` negasyonu.
- **PR #44** wiring: `rlm-answer --engine`, web `/api/rlm/config`+`/test-adapter`, `rlm-tools` CLI
  (ölü-kod allowlist'i canlı bağladı), alexzhang run-log, source `support_level`.
- **PR #45** lock: pyproject `rlm` extra ↔ `uv.lock` drift kapandı (openai yalnız opsiyonel-transitive).
- **PR #46** Level 3 + observability: trajektori logging (rlm_store+JSON, `/runs/{id}/trajectory`,
  `rlm-trajectory`), docker preflight (CLI+daemon probe), web motor dashboard paneli (preview-verified),
  GERÇEK `rlms` kurulum doğrulaması. Bug-avı 5 fix (orphan-run→failed, determinizm temperature=0,
  daemon-probe, relevance clamp, traj 50MB+OOM).
- **Doğrulama:** her PR ruff+format+mypy+pytest yeşil (main CI success); dashboard preview_eval ile.
- **AÇIK KARAR (kullanıcı):** gerçek Claude-destekli alexzhang inference API+key+docker ister → API
  YOK direktifi gereği KOŞTURULMAZ; sistem native (Ollama) ile tam çalışır. Memory [[rlm-alexzhang-adapter-2026-06-24]].
- **NOT:** `rlms` paketi dev `.venv`'de kurulu kaldı (#46 doğrulaması; eşzamanlı achilles-web.exe
  `uv sync`'i kilitledi). Zararsız; temizlemek için web sunucusunu kapat + `uv sync`.

### 🆕 EN SON İŞ (2026-06-23) — Kademe-2 derin av (8 finder) → 5 fix push, 2 devir
Kullanıcı "tüm projeyi tara + pushla". 8-finder Kademe-2 workflow + adversarial doğrulama; hedef
`f14fd33` (sabah Kademe-1'den sonra geldiği için kapsanmamıştı). **Rapor: `reports/bug-scan/scan-2026-06-23_0030.md`.**
**DÜZELTİLDİ + PUSH (her biri kapı yeşil: ruff+mypy+pytest tümü):**
- **A** `b99c4a0` (HIGH): `is_detached_training_running(root)` log-tazeliği yedeği `root`'u yok sayıp
  gerçek makine logunu okuyordu → ölü-pid testi yanlış `True`. ("Windows pid bug" sanılan = izolasyon
  sızıntısı.) `root` verilince yedek de yalnız o root'u sorgular.
- **E** `ed44bb5` (BLOCKER-sınıf): `gates._card_text` `limitations`/`datasets`/`risk_warnings` alanlarını
  toplamıyordu → bu alanlardaki sır/PII Gate 7 (BLOCKER) taramasını atlıyordu. 3 alan eklendi + 2 test.
- **C-kısmi** `9453a5a` (kural 6): `concept_graph._extract_links` seedsiz LLM → graf kararsız; seed=42.
- **B** `a672ce0` (kural 3): `risk_manager._extract_trade_returns` blokları HAM position'la sınırlıyordu
  → çıkış maliyeti + son tutuş-barı bloğun dışında → Kelly şişiyordu (headline doğru). Bloklar eff_pos'tan
  sınırlanır + çıkış-maliyeti barı dahil; +5 test.
- **D** `e4fb9fa` (kural 2): `aggregate()` min-graded korumasızdı → çevrimdışı graded=1 (yalnız L5) →
  pass_rate=%100 'scored' → `auto_pipeline` TERFİ kapısını şişirir. `_MIN_GRADED_FOR_SCORE=3` guard; +3 test.

**DENETİMLİ SEANSA DEVİR (gözetimsiz fix YOK):**
- **C-cross_paper**: `cross_paper_synthesizer` sentez yolu seed — YARATICI yolda determinizm-vs-yaratıcılık
  tasarım kararı (per-input türetilmiş seed seçeneği). Backtest/eval/terfi'yi etkilemez (LOW).
- **safety FN×2**: `safety_scanner.py` eş zamanlı oturumun aktif WIP'i → çakışmamak için bırakıldı.

**GOTCHA'lar:** (1) Büyük çok-ajan av 03:00 (Istanbul) **session-limit** reset'ine takıldı → `synthesize`
+3 verify düştü; bulgular ana-döngüde elle sentez/doğrulandı. (2) Eş zamanlı oturum (RLM/PR #12) çok aktifti;
collision 2× (Fix A süpürüldü → b99c4a0; LORA_log benim B commit'ime süpürüldü). ÇÖZÜM: `git reset`+dar-add.
Bkz memory [[kademe2-hunt-2026-06-23]].

### 🆕 EN SON İŞ (2026-06-22) — İki-hat train.jsonl drifti kökten kapatıldı (Kademe-2)
**Bulgu:** İki ayrı veri hattı aynı `data/training/jsonl/train.jsonl`'i yarışıp habersiz
ezerdi. **A (zengin):** `assemble_sft`/`lora-cloud-prep` → `lora_sft.jsonl` (~2020, `{messages}`)
→ `ensure_train_split` → ~1919/101. **B (cılız):** web uçları `DatasetBuilder.build()` →
`training_examples` SQLite → ~29/4 satır, `{prompt,completion}`. `train --run`/`launch()`
eğitimden hemen önce `ensure_train_split` çağırıp onarıyordu; AMA çağırmayan bir yol (doğrudan
backend/harici) 29 satırlık bayat/yanlış-formatlı veriyle eğitebilirdi.
**Fix (bu seans):**
- `app/training/detached_launch.py` → yeni KANONİK yardımcı `build_training_split()`
  (lora_sft.jsonl → ensure_train_split; kaynak yok/boşsa `assemble_sft_lines` ile bir kez
  üret, boşsa dokunma=clobber guard). `DatasetResult`-uyumlu `TrainingSplit` döner.
- `app/web/server.py` → 3 uç (`/api/training/dataset`, `/dry-run`, `/colab-notebook`) artık
  `DatasetBuilder` yerine `build_training_split()` çağırıyor → format `{messages}`, sayı kanonik
  kaynakla AYNI. `DatasetBuilder` yalnız manuel `achilles dataset` (SQLite inceleme) için kaldı.
- Regresyon testi: `tests/test_web_api.py::test_dataset_endpoint_uses_canonical_lora_sft`
  (hermetik, monkeypatch settings → SQLite'a dokunmaz; train+valid sayı=kaynak, format={messages},
  valid⊥train).
- Doğrulama: format+lint+typecheck (179 dosya)+test (tümü yeşil).
**Not:** CLI `achilles dataset` ve `/api/training/run`→`launch()` zaten kanonik yolu kullanıyordu;
yalnız 3 web "veri oluştur/dry-run/colab" ucu cılız hatta düşüyordu. Branch/commit: bekliyor
(insan onayı ile). Memory: `training-data-pipeline` + `kademe2-pretrain-hunt-2026-06-22` güncellendi.

### 🆕 EN SON İŞ (2026-06-21) — Web upload "bir kısım gelmedi" düzeltildi
**Şikâyet:** Başka bilgisayarda çok sayıda PDF yüklenince web arayüzünde ~10 dosya görünmemiş.
**Kök neden:** Upload uçunda kayan-60sn hız sınırı (`upload_rate_limit_per_min`) **20/dk** idi;
sürükle-bırak ile >20 dosya gelince fazlası **429** yiyor, frontend bunu sessizce "hata" sayıp
dosyayı **diske bile kaydetmeden** düşürüyordu (429 middleware'de handler'dan ÖNCE).
**Fix (PR [#10](https://github.com/alimirbagirzade/achilles/pull/10), branch `fix/upload-rate-limit-retry`, oto-merge):**
- `app/web/static/assets/app.js` → toplu yükleme artık 429'da dosyayı düşürmüyor; bekleyip aynı
  dosyayı yeniden deniyor (≤40 deneme, 3.5sn ara). Kök çözüm.
- `app/config/settings.py` → `upload_rate_limit_per_min` 20→**60** (env: `ACHILLES_UPLOAD_RATE_LIMIT_PER_MIN`).
- Doğrulama: format+lint+typecheck+test **yeşil**.
**Not (kullanıcıya söylenenler):** (a) JS değişikliği için web **yeniden başlatılmalı**/sayfa yenilenmeli;
(b) zaten diske inmiş ama indekslenememiş dosyalar için **"⟳ TÜMÜNÜ İNDEKSLE"** / `uv run achilles ingest`
(idempotent kurtarma); (c) makineler arası senkron YOK — A'da yükleyip B'ye bakmak boş gelir.
**Henüz YAPILMADI (aday):** arka-plan indeksleme sessiz hatası hâlâ kullanıcıya yüzeye çıkmıyor
(Ollama kapalı/parse hatası → log'a yazılır, UI'da "alındı, indeksleniyor…" der). İleride bir
indeksleme-durumu/hata rozeti eklenebilir. Memory: yok (bu işten yeni).
**Gotcha (test):** Bash tool alt-kabuğu `PYTEST_DEBUG_TEMPROOT` user env var'ını miras almıyor →
pytest `PermissionError` verir. Çözüm: `PYTEST_DEBUG_TEMPROOT="C:/Users/sevinc/pytest-tmp" uv run pytest -q`.

### Proje amacı
LLM'i "trader gibi düşünen" bir araştırma motoru yapmak:
1. Makalelerden formül ve kavramları hafızaya al
2. Bunları birleştirip daha önce denenmemiş indikatör/algoritma öner
3. Otomatik backtest et → sonuçtan öğren → LoRA eğitim verisi üret
4. 3B modeli test eder; gerçek çıktı için 120B kullanılacak

### Mevcut durum (2026-06-21) — ✅ TOPARLAMA + TAŞINABİLİRLİK + OTOMATİK PR + YEŞİL CI

> Seans hedefi (kullanıcı): repoyu topla, pull/push çöz, kullanım kılavuzu yaz, sistemi
> **kiralık CPU / daha güçlü makineye taşımaya hazırla** ("kur-çalıştır, orada geliştirme YOK"),
> PR'ları **otomatikleştir**. Hepsi LANDED + doğrulandı. **main YEŞİL + senkron** (`dd46378`).

**1) Git toparlama (pull/push çözüldü):** 15→2 worktree, 18→5 branch (main + korunan 4:
`fix/rag-scoring-approval-cas`, `feat/web-training-gate-fix`, `feat/local-claude-operator-dry-run`,
`salvage/system32-cpu-lora`), ~780MB çöp silindi + `.gitignore` sertleşti (`.pytest_tmp_*`/`*.pid`/
`logs/`/`*.bak`/`vector_db/chroma.corrupt-*`/`storage/_*.py`/`scripts/*-autostart.vbs`). Commit `d940bdb`.
İş kaybı YOK (her dal/worktree önce denetlendi). Memory: `git-repo-durumu`.

**2) Taşınabilirlik katmanı (commit `72f4de4`):** çekirdek zaten CPU-first/taşınabilirdi; otomasyon
katmanı eklendi: `scripts/verify-install.sh` (offline kurulum kapısı, ps1'in bash portu, GECTI doğrulandı),
`scripts/install-autostart.sh` (Linux systemd/cron autostart), `setup.sh` +yerel/uzaktan erişim sorusu
(uzaktan→`0.0.0.0`+otomatik API token)+verify kapısı+opsiyonel autostart, `continuous-learning.sh` HIGH
blocker fix (gömülü powershell→`pgrep`). **KARAR:** RAG+LoRA her makinede SIFIRDAN (vector_db/sqlite/
adapter KOPYALANMAZ; sadece PDF→`ingest`→Kaggle). Memory: `bulut-tasima-protokolu`.

**3) Kullanım kılavuzu (kullanıcı için, Desktop):** `Desktop\RAG Kaynak\Kulanım talimatı\` →
`00_GENEL_BAKIS` / `01_KURULUM_ve_CALISTIRMA` / `02_PUSH_PULL_GIT` / `03_BULUT_TASIMA_PROTOKOLU`.
`README.md` eskimiş bilgileri yenilendi (`d4338b7`: bulut/9-sekme/vektör-yolu/eğitim-süresi/uzaktan-mod).

**4) OTOMATİK PR akışı CANLI + UÇTAN UCA KANITLANDI:** `gh auth login` (kullanıcı yaptı) →
`scripts/setup-pr-automation.sh` çalıştırıldı (repo PUBLIC+ADMIN → `allow_auto_merge` + main branch
koruması, required ctx=`lint · types · tests (offline)`, `enforce_admins=false` → **owner doğrudan
`git push origin main` yapabilir**, loop'lar kırılmaz). `scripts/open-pr.{sh,ps1}` = push+PR+CI-yeşilse
oto squash-merge (varsayılan; `--no-merge`/`-NoMerge` ile kapat). **PR #9 PowerShell'den oto-merge oldu
(kanıt).** `.github/pull_request_template.md` eklendi. Commit'ler `5603c74`,`dd46378`.

**5) main CI RED→YEŞİL:** seans öncesinden kırmızıydı (kimse fark etmemişti). İki arıza: format drift
(`72f4de4`'te `ruff format .`) + `test_flashrank_reranker` opsiyonel `flashrank` paketsiz FAIL (`5d3eee4`
SimpleNamespace shim). + `open-pr.ps1` PS 5.1 native-stderr bug (`dd46378`, sadece ÇALIŞTIRINCA çıktı).

**KALAN (sonraki seans — hiçbiri bloke değil):**
- **Node.js 20 deprecation** CI uyarısı (`actions/checkout@v4`, `setup-uv@v5` Node24'e zorlanıyor) →
  ileride aksiyon sürümlerini bump et. CI'ı bozmuyor.
- **GOTCHA:** açık PR'a yeni commit push'u (synchronize) CI'ı tetiklemeyebilir (PR BLOCKED kalır);
  TAZE PR sorunsuz. Tekrarsa: PR'ı close/reopen.
- **Eşzamanlı oturum `rag-chains-work`** worktree'si AKTİF (RAG "Zincir" işi: keyword-eval/router/
  CRAG-lite/FlashRank) — DOKUNMA. Not: ben `scripts/rag_keyword_eval.py`'yi main'e (eski kopya)
  commit'ledim; o oturum merge ederken küçük çakışma çıkabilir (onların çözeceği).
- Proje backlog'u (bu toparlamanın DIŞINDA): RAG turları sonrası backtest/eval ölçümü, sıradaki RAG
  adayları — bkz. memory `session-devir-2026-06-21`, `hiz-kalite-makale-uygulama`.

---

### Mevcut durum (2026-06-20) — ✅ OTONOM BAŞLANGIÇ ZİNCİRİ (branch `feat/otonom-baslangic-zinciri` · PR #5)

> Hedef: "yeni makinede git clone sonrası tertipli bir SIRALAMA ZİNCİRİ olarak otonom ayağa kalksın."
> 5 alt-sistem repo-taramasıyla tasarlandı (ayrıntı: memory `otonom-baslangic-zinciri`); sistemin %80'i
> zaten taşınabilir çıktı → eksik halkalar eklendi. **Kararlar:** tetikleyici = mevcut hibrit (HKCU Run +
> Task Scheduler yedek), otonomi = **varsayılan KAPALI**, executor = hibrit ince. Kural 8 korunur
> (gerçek eğitim/terfi yine tek-kullanımlık taze onay; executor o kapıyı zayıflatmaz).

**LANDED — 6 commit (hepsi ruff+mypy temiz · tüm offline suite YEŞİL · CLI duman-testi exit 0):**
- `verify-install.ps1` — autostart ÖNCESİ çevrimdışı duman testi kapısı (`start-server.ps1 -Install` artık önce doğrular; kalırsa autostart kurmaz; `-SkipVerify` kaçış).
- hibrit **executor** (`app/agents/runtime/executor.py`) — allow-list handler (bilinmeyen agent çalışmaz), `run_task`/`run_pending(--retry-blocked)`, STOP_ALL + taze-onay kapısını korur; + `task_queue.requeue_task` + CLI `tasks-run`. (9 test)
- `synth_qa_chain.ps1` taşınabilir (`$PSScriptRoot`+`Find-Uv`) + UTF-8 BOM (PS 5.1 parse fix).
- **runtime-init** ön-uçuş (`app/agents/runtime/preflight.py`) — manifest + 4 Phase-2 tablosu + STOP_ALL doğrula; CLI kapı. (2 test)
- **chain** (`automation_manifest.yaml` 'chain' bölümü + `app/agents/runtime/chain.py` Kahn topo-sort, döngü/eksik-step doğrulama) + CLI `chain-status [--live]`. (8 test)
- README ajan-runtime/otomasyon komut bölümü.

**KALAN (sonraki seans):**
- **PR #5** → `main` merge: github.com/alimirbagirzade/achilles/pull/5 (branch eşzamanlı oturum commit'leri `06e048b`/`3de5082`/`c420d3b`'yi de içeriyor — collision; main'e ayrı yoldan girerlerse düşer).
- **`.claude/settings.json`** SessionStart hook'unda yabancı macOS yolu (`/Users/mirbagirzade`) → **self-modification guardrail** otomatik düzeltmeyi engelliyor. Kullanıcı AÇIKÇA "o hook'u düzelt" demeli. Fix: `cd "${CLAUDE_PROJECT_DIR:-.}"`.
- Executor **per-agent handler**'ları kayıtlı değil (allow-list bilinçli boş) → `tasks-run`/`chain-status` altyapısı hazır ama henüz ajan çalıştırmaz; her ajan için handler eklemek doğal sonraki adım (tehlikeli zincir dikkatli).
- ✅ **ÇÖZÜLDÜ (`52d305c`, 2026-06-20):** `app/main.py` `__main__` bloğu (line ~1327, komutların yarısından önce) dosya SONUNA taşındı → `python -m app.main chain-status / tasks-list` artık çalışıyor (önce "no such command" veriyordu). Saf yer-değiştirme (4+/4-); ruff+mypy temiz, offline pytest yeşil. Console-script entry-point zaten etkilenmiyordu.
- **gh ipucu:** `gh auth login` interaktif; ama GCM'deki `gho_` token `git credential fill` → `GH_TOKEN` ile `gh.exe`'ye verilerek PR açılabilir (token yazdırmadan).

---

### Mevcut durum (2026-06-20 · web anlama rozeti) — ✅ CACHE-BUST + OBJ.ANLAMA 500 FIX

> Kullanıcı: "RAG anladı: %54 (63/116) · anlama %32 (114 makale) — web'te canlı mı? değilse çöz+fix+push."

**Teşhis:** Rozet SAYILARI canlıydı (`/api/rag-mastery` her 30sn DB'den, poll'lu), ama kullanıcı
DONMUŞ/önbellekten eski sayfayı görüyordu. Kök neden: `index.html` `app.js?v=2` SABİT etiketiyle
yükleniyordu; app.js değişince etiket elle bump edilmemiş (CSS ?v=4'e çıkmış, JS ?v=2'de kalmış) +
asset'lerde Cache-Control yok → tarayıcı eski JS'i süresiz önbellekten servis ediyordu (eski "anlama %"
etiketi = kullanıcının gördüğü; güncel kod "öz-değ. %").

**LANDED (2 commit — origin/main + `feat/agent-runtime-phase2` (PR #3); `feat/agent-runtime-observer`'da da var):**
- `38ed997` **cache-bust**: `index()` dinamikleşti → `/assets/app.(js|css)?v=` içerik sha256 hash'iyle
  servis + HTML `no-cache`. app.js/app.css değişince URL otomatik değişir → manuel `?v=` bump bir daha
  gerekmez. (+`test_index_cache_busting`)
- `3b67ed1` **exam timeout**: `/api/understanding-score` 500 veriyordu (yavaş CPU'da `httpx.ReadTimeout`
  yalnız `LLMUnavailable` yakalandığı için sızıyordu). `local_llm` httpx hataları→`LLMUnavailable`;
  `score_indicator_exams` try/except + fail-fast. (+`tests/test_local_llm.py`, +2 understanding_score testi)
  - NOT: eşzamanlı oturum `e359aa6` ile DAHA RAFİNE etti (bayat-skor recompute + 2-ardışık-fail bail +
    timeout 60→**240sn**). İkisi uyumlu; **e359aa6 nihai**.

**Doğrulama (canlı, restart sonrası):** index hashed URL + `no-cache` ✓; rag-mastery canlı & hareketli
(kart 237→239, içerikli makale 63→64); obj.anlama **HTTP 200** (önceden 500), ~102sn'de dürüst skor
(1 graded, pass_rate 0.0). ~590 offline test yeşil; ruff+mypy temiz.

**Araştırma sonucu (KAPANDI — bu konu için harici loop GEREKMEDİ):** obj.anlama düşük (~%0–15) çünkü
qwen3:4b held-out indikatör hesaplama sınavlarını CPU'da ya geçemiyor ya zaman aşımına uğruyor. Projenin
tezini DOĞRULUYOR: kaba öz-değ. %32 iyimser; objektif sınav geçme oranı çok düşük. Sorun model-kapasitesi
+ CPU, kod değil → makale/LoRA dış araştırma rutinleri (ayrı/zamanlanmış) bu konuya çözüm değil.

**KALAN (sonraki seans):**
- obj.anlama'yı e359aa6'nın 240sn timeout'uyla TEKRAR ölç → gerçek pass-rate (kaç sınav graded). Running
  server bayatsa web RESTART (rotalar/kod başlangıçta yüklenir; statik diskten canlı).
- Kullanıcı tarayıcısında bir kez `Ctrl+Shift+R` (eski cached index.html kırılsın) — sonrası kalıcı.
- gh CLI bu makinede kurulu DEĞİL → PR'lar GCM token (`git credential fill`) + GitHub API/`GH_TOKEN` ile
  açılıyor (token yazdırmadan; bkz. yukarıdaki "gh ipucu").

---

### Mevcut durum (2026-06-19) — ✅ KAD-2 TAMAMLANDI + 🔄 SYNTH-QA ÜRETİMİ DEVAM EDİYOR

> **Kademe-2 derin bug-avı TAMAMLANDI** (Sprint 1-5, toplam 18 onaylanan fix, commit `ceae006`).
> Şu an kritik tek bekleyen: `synthetic_qa.jsonl` → 362'den 1300'e (seed=100 CPU'da sürüyor).
> 1300'e ulaşınca: `achilles lora-split` yenile → Kaggle "Run All" tıkla (tek manuel adım).

**KAD-2 SPRINT FİX ÖZETİ (hepsi committer ve test edildi):**
- Sprint-1: rag_exam_runner sahte-geçme, bollinger registry, entropy bar-0 NaN, adapter peft_base_model, rag_answerer seed, significant_numbers binlik ayraç
- Sprint-2: BM25 tie-break (determinizm), citation_score gerçek parse, dataset_quality false-positive (192→6), entropy warmup=period (7 test yeşil)
- Sprint-3: paper_indexer embedded erken yazım (BUG-M6), sqlite_store mark_chunks_embedded, auto_pipeline eval_pass_threshold (BUG-M9)
- Sprint-4: peft_llm_shim.py (PEFT adapter → LocalLLM), auto_pipeline anlama-merdiveni kıyası (v5 savunma dikişi)
- Sprint-5: server.py BUG-H3 (komisyon+slippage eksikti), agents/runtime Phase 2 (approvals/supervisor/task_queue), overfit_checks BUG-M7 IS+OOS

**SYNTH-QA DURUMU:**
- Mevcut: 362/1300 (`data/lora_sft/synthetic_qa.jsonl`)
- Aktif: `logs/synth_qa_seed100.log` — PID 8044 çalışıyor (~3 dk/chunk, CPU-only)
- Hedef ulaşmazsa: `powershell scripts/synth_qa_chain.ps1 -Target 1300 -StartSeed 200`
- **1300'e ulaşınca:** `uv run achilles lora-split` → Kaggle "Run All"

**KALAN (KAD-2 sonrası):**
- grounding_verifier markdown sentence splitter (BUG-M8, ertelendi — büyük refactor)
- PR: `feat/agent-runtime-phase2` → `main` merge (Kad-2 tamamlanınca)

---

### Mevcut durum (2026-06-17) — 🔒 ANLAMA MERDİVENİ KALICI + 📚 MAKALE LOOP + 🐛 `\r` BUG

> Kullanıcı: "L2/L3/L4/L5'i kalıcı yap + push; bug'ları loop'la çöz; faydalı makaleleri
> sürekli indir; sonra eğitime devam." Eğitim kararı DEĞİŞMEDİ: **önce bitir, sonra bulut-GPU**
> (lokal CPU = v5 çıkmazı; veri kapısı temiz-regen bitince GO → manuel Kaggle "Run All").

**KALICILIK (L2-L5) ✅ — bu seansta inşa edildi:**
- `understanding_snapshots` SQLite tablosu + `SqliteStore.save_/list_/latest_understanding_snapshot`.
- `app/verification/exams/understanding_record.py`: `record_understanding` (DB + zaman-damgalı
  `reports/evals/understanding/*.json`), `load_understanding_history`.
- `understanding_score.py`: `score_full_ladder` (L5 deterministik—**çevrimdışı bile notlanır**
  + L3/L4 LLM + opsiyonel RAG Taban/L1/L2), `l5_example_result`, `run_rag_ladder_answers`;
  `score_indicator_exams` geri-uyumlu (refactor → `_indicator_exam_results`).
- CLI: `understanding-score --full --with-rag --record` + yeni `understanding-history`.
- Web: `/api/understanding-score`'a `full/with_rag/record` query (**VARSAYILAN DAVRANIŞ DEĞİŞMEDİ**)
  + yeni `/api/understanding-score/history`; "obj. anlama" rozeti tıklayınca tam merdiven + KALICI kayıt.
- **6 yeni test + tüm offline paket yeşil · ruff+mypy temiz.** Uçtan uca denendi (snapshot DB + CLI history).

**🔬 GENİŞ DENETİM SONRASI SAĞLAMLAŞTIRMA (2026-06-18) — kullanıcı "acele etmeden geniş bakalım ve çözelim":**
Çok-ajan denetim (5 boyut, 24 doğrulanmış bulgu) → 11 "now" çözüldü, hepsi offline test edildi:
1. **L5 yanlış-negatif BUG fix** — `composition_to_result`: backtest YALNIZ "çok az işlem/veri yok/belirsiz"
   yüzünden düştüyse artık `failed` değil `skipped` (test edilemedi). Sahte ~%0 sinyali bitti.
2. **L5 gerçek sinyal** — `l5_results_from_sessions`: `score_full_ladder` sabit `example_ir` yerine sistemin
   KENDİ ürettiği kompozisyonların (`research_sessions.verdict`) gerçek sonucunu okur (`use_sessions_l5`).
3. **Bağlam otomatik kaydı** — snapshot context'ine `llm_model`/`model_kind`/`n_papers`/`n_carded` otomatik
   yazılır (zaman serisi yorumlanabilir; base vs adapter ayrımı için temel).
4. **Merdiven sırası** — `Taban→L1..L5` sabit sıra (alfabetik sort Taban'ı sona atıyordu); CLI + web ortak.
5. **Regresyon kıyası** — `compare_understanding(prev,curr)` (yalnız aynı `llm_model` → `regressed` bayrağı) +
   CLI `understanding-history --compare` + "Bağlam" sütunu. v5-tipi gerilemeyi yakalamanın temeli.
6. **Web görünürlük** — Öğrenme paneline "Objektif Anlama Geçmişi" kartı (sparkline + tablo, `/history`
   tüketilir, XSS-güvenli esc) + açılışta son skor rozette pasif gösterilir. **Canlı sunucuda doğrulandı (HTTP 200).**
7. **Adapter-ölçüm dikişi** — `score_full_ladder(llm=...)` + `_indicator_exam_results(llm=...)` → base yerine
   ADAPTER ölçülebilir (sahte-LLM ile offline test edildi). v5-savunmasının altyapısı.
- **12 yeni test (toplam) + tüm offline paket yeşil (exit 0) · ruff+mypy temiz.** 145/145 makale artık kartlı.

**🔴 "NEXT" (eğitilmiş adapter gerektirir / daha büyük — denetim doğruladı, henüz YAPILMADI):**
- **Adapter promosyonunu anlama-merdivenine BAĞLA (en kritik):** `auto_pipeline._run_eval` base-vs-adapter
  `score_full_ladder` koşsun; adapter pass_rate base'in altındaysa promosyonu BLOKLA. (#7 dikişi hazır; PEFT
  LLM shim + eğitilmiş adapter gerekir — v6 daha yok, o yüzden offline doğrulanamaz.)
- **Disiplin/dürüstlük sınav basamağı:** merdiven "maliyetsiz getiri / garanti / pasaja-göre" v5 patolojilerine
  KÖR (onlar adapter_eval + pretrain-gate'te). `discipline_core` red-flag'lerini bir ExamResult'a sar.
- **pretrain-gate → auto_pipeline zinciri:** `dataset_quality.audit_dataset` Gate 0-8 sonrası çağrılmalı (NO-GO → bloke).

**🐛 `\r` BUG (kart üretimi OSError 22):** Windows Python stdout CRLF → bash `for pid` döngüsüne `\r`
sızıyor → `paper_xxx\r_card.json`. Kök sebep doğrulandı (DB id'leri TEMİZ: 138 makale, 0 bozuk).
`continuous-learning.sh` (satır 71/101) **eşzamanlı oturum/makine** tarafından düzeltildi; `auto-chain.sh`
de düzeltiliyor → scriptlere DOKUNMADIM (çakışmamak için). `continuous-learning` loop synth-yazma
yarışı için duraklatıldı (`storage/STOP_LEARNING`); bulk-regen bağımsız sürüyor. **KALAN:** mac-loop.sh
aynı `\r` desenine sahip (macOS'ta zararsız ama robustluk için bir gün düzeltilebilir).

**Synth temiz yeniden-üretim (AKTİF):** `synth-qa-bulk → 1300` arka planda; yeni veride "pasaja göre"
oranı **%0** (eski %68 = v5'i batıran). 1300'e ulaşınca kapı GO beklenir.

**📚 Makale loop:** 6 yeni gerçek arXiv makalesi `Desktop\RAG Kaynak\Gerekli kaynaklar\` köküne indi
(1905.05023 backtest-OVF, 2512.12924 walk-forward, 2309.15217 RAGAS, 2401.15884 CRAG, 2601.05716
Kalman-Markov, 2605.17117 geometrik rejim) + `00_NEDEN_ONEMLI_oku_once.md` güncellendi. DSR/PBO arXiv'de
yok (uydurulmadı, Kural 7). Yeni `lora-arastirma` ajanı da mevcut (LoRA tekniği araştırması için).

---

### Mevcut durum (2026-06-16 SON) — 🧠 ANLAMA DOĞRULAMA SİSTEMİ İNŞA EDİLDİ ✅

> Bu seans (06-16 akşam/gece): RAG vs LoRA netleşti + **"Anlama Doğrulama" merdiveni —
> _anlama yüzdeyle değil SINAVLA kanıtlanır_ — hem belgelendi hem KOD olarak inşa edildi.**
> Çekirdek fikir: "anladı" = bilgiyi DOĞRU KULLANIP ondan TEST EDİLEBİLİR yeni bir şey ÜRETEBİLDİ.

**İNŞA EDİLEN (hepsi push'lu · ~60 sınav testi + tüm offline paket yeşil · ruff+mypy temiz):**
- `app/verification/exams/` paketi:
  - `safe_eval.py` (whitelist AST — eval/exec YOK, Kural 5), `reference_oracle.py`, `registry.py`
  - `l3_application.py` — **L3:** formül + tutulan sayı → model hesapla → `np.allclose` referansla
  - `l4_counterfactual.py` — **L4:** parametre yön değişimi KODDAN türetilir, model puanlanır
  - `l5_composition.py` — **L5:** yeni-formül 3 kapı (math + novelty + maliyet-dahil backtest/OOS) → aday/red
  - `understanding_score.py` — objektif geçme-oranı (skipped/no_data paydaya GİRMEZ) +
    `rag_answers_to_results` (mevcut RAG sınavını Taban/L1/L2 olarak merdivene bağlar)
- **CLI:** `achilles exam-l3 / exam-l4 / exam-l5 / understanding-score`
- **ENTROPY göstergesi** (`indicators.py` + registry) — yönsel ikili Shannon entropisi [0,1],
  look-ahead'siz; entropi vizyonunun ilk indikatörü, L5 Markov+entropi yapı taşı.
- Belgeler: README "🧠 Achilles okuduğunu *anladı* mı?" + `docs/PROTOKOL_RAG_LORA_ZINCIR.md`
  "ANA FİKİR" + `docs/examples/raft_discipline_seed.jsonl` (RAFT disiplin seed örneği).
- Hafıza: `memory/anlama-dogrulama-ana-fikir.md`, `memory/arastirma-makale-kaynagi.md`.
- **Makaleler indirildi** → `C:\Users\sevinc\Desktop\RAG Kaynak\Gerekli kaynaklar`:
  RAFT(2403.10131), HMM Intraday(2006.08307), Transfer Entropy(2507.09554),
  Entropy Analysis(1807.09423) + `00_NEDEN_ONEMLI_oku_once.md`. **Kullanıcı inceleyip web'den
  RAG'a yükleyecek.** Bu indirme LOOP — yeni makaleler aynı klasöre eklenecek.

**BEKLEYEN BACKLOG (yeni seans buradan devam — kullanıcı: "loop olarak çözene kadar"):**
1. ✅ **L5 → synthesis_engine bağlandı** (`18cabd4`) — orchestrator her iterasyonda L5
   CompositionGate koşar (math+novelty+backtest); sonuç IterationResult + session + web API.
2. **Registry'yi genişlet** — ✅ permütasyon entropi (Bandt-Pompe, `PERMENTROPY`) eklendi
   (indicators + exams registry + L5 _REGISTRY + query_expander); KALAN: daha çok Markov/entropi
   göstergesi (ör. transfer entropi, rejim/HMM tabanlı) + yeni makale indir.
3. ✅ **Web objektif anlama skoru** — kaba "anlama %" dürüstçe "öz-değ. %" diye yeniden
   adlandırıldı; header'a tıklanabilir "obj. anlama" rozeti + `GET /api/understanding-score`
   (L3/L4 sınav geçme oranı; LLM yoksa insufficient_data). Ortak `score_indicator_exams`
   helper (CLI + web). KALAN: web'de L5 kompozisyon sonucunu da göstermek (opsiyonel).
4. **RAFT reçetesini düzelt** (seed'i yüzlerce örneğe ölçekle) → SONRA eğit (körlemesine 47h retrain YOK).
   - 🔬 **TEŞHİS (v5 batış kök sebebi, koddan doğrulandı):** (1) `synthetic_qa_builder._ONESHOT_EXAMPLE`
     her cevaba "Pasaja gore" öneki → model koşulsuz açılış ezberledi (bağlamsız evalde de). (2) Üretici
     yalnız "pasajdan cevapla" örneği üretiyor; adversarial disiplin örneği YOK (`raft_discipline_seed.jsonl`'de
     6 var ama ölçeklenip karıştırılmamış). (3) overfit/tekrar. → adapter maliyetsiz %20 getiri uydurdu (REJECT).
   - ✅ **Fix A yapıldı:** `_ONESHOT_EXAMPLE` açılışı çeşitlendi, "Pasaja gore" sızıntı öneki kaldırıldı + test.
   - ✅ **Fix B yapıldı (ASIL):** `app/training/discipline_dataset.py` — 9 tuzak (garanti/backtest'siz/
     maliyetsiz/kaynak-yok/bağlam-uyumsuz/look-ahead/overfit/kaldıraç/grounded-belirsizlik) × 16 strateji
     × 3 varyant = **432 deterministik adversarial örnek**. `lora-cloud-prep` bunları DEDUP'TAN SONRA
     ~%25 karıştırır (`--discipline-ratio` / `--no-discipline`); CLI `discipline-dataset` önizleme/export.
     v5 dersleri kodlandı: açılışlar çeşitli (sabitleme yok), 1/3 örnek system-prompt'suz (eval öyle
     çağırır), cevaplar naif `check_flags`'i geçer (yasak yüzey token'ı yok + maliyet token'ı var).
     12 yeni test + tüm offline paket yeşil · ruff+mypy temiz.
   - ✅ **Fix C yapıldı:** `dataset_quality.recommend_epochs(n)` (boyuta göre 1-3); mix oranı zaten flag.
   - ✅ **#3 OFFLINE KAPI yapıldı:** `app/training/dataset_quality.py` + `achilles pretrain-gate` —
     birleşik SFT setini LLM'siz tarar, **GO/NO-GO** verir (garanti-vaadi zehiri / açılış-ezberi → blok).
     `app/training/sft_assembly.py` ortak birleştirme (lora-cloud-prep + gate DRY). 11 yeni test yeşil.
   - 🔴 **KAPI İLK KOŞUDA NO-GO VERDİ (2026-06-17, beklenen):** mevcut `synthetic_qa.jsonl` (1289 satır)
     Fix A'dan ÖNCE üretildi → cevapların **%68'i "pasaja gore" ile açılıyor** (v5'i batıran tam mekanizma).
     Disiplin karışımı (432/432) ve garanti-zehiri (0) temiz; sorun ESKİ synth verisi. **ÇÖZÜM = synth-qa'yı
     temiz üreticiyle YENİDEN ÜRET** (gece loop'u bunu yapıyor). Rapor: `reports/evals/pretrain_gate.json`.
5. **Gece otonom loop (2026-06-17, AKTİF):** synth-qa'yı temiz yeniden-üret → `pretrain-gate` GO olana kadar
   → `lora-cloud-prep` paketi tazele → raporla. Eğitim: kapı GO + kullanıcı sabah Kaggle "Run All" (manuel tık).

**🎯 EĞİTİM LOOP KARARI (2026-06-16, kullanıcı):** Donanım = **Bulut GPU (Kaggle T4×2)**
(önceki bulut-reddi geri alındı; ~30 dk/koşu → loop fizibıl). Otonomi = **tam otonom loop**
(reçete→dataset→eğit→eval→koşullu terfi→düzelt→tekrar; sonuçlar raporlanır).
**Sıralama:** önce dokümanlar → #4 reçete → #3 dataset'i OFFLINE sınavdan geçir (L3/L4/L5 +
understanding-score, eğitMEDEN kalite kanıtı) → Kaggle eğit → eval → koşullu terfi.
**Caveat:** Kaggle "Run All" manuel (headless değil) — çevresi otomatik, sadece o tık kullanıcıda.

**Eğitim kararı:** detached tek-tık eğitim TEKNİK olarak hazır AMA başlatma — önce reçete düzelt
(v5 aynı sebepten battı). Kullanıcı onayı: **"önce bitir, sonra eğit."**

**Komut notu:** testleri `--basetemp=.pytest_tmp` ile çalıştır (stale `pytest-of-sevinc` izin
sorunu = WinError 5). Push döngüsü: `git fetch + rebase origin/main` sonra push (eşzamanlı makine).

**Son commit:** `d54ab68` (ENTROPY). Bu seans zinciri: `5820ab5 → 1c5f349 → da98e6e →
135cc66`(L3)`→ 8c721c9`(L4)`→ 65feac1`(L5)`→ 0ec5854`(CLI)`→ d0e139f`(L1/L2)`→ d54ab68`(ENTROPY).

**Otonom nöbet:** bu seansta ScheduleWakeup loop aktifti (sağlık + backlog ilerletme + makale indirme).
Yeni seansta kendiliğinden devam ETMEZ — istenirse `/loop` ile yeniden kur.

---

### Önceki durum (2026-06-16 erken) — LOKAL EĞİTİM BİTTİ, ADAPTER REJECT

> Kullanıcı Kaggle/bulut REDDETTİ → eğitim **lokal CPU**'da yapıldı (kendi imkanlarıyla;
> ileride uzaktan kiralık CPU). Bu seans gece+gündüz otonom yürüdü (eğitim nöbeti + post-training).

- **🔴 achilles_lora_v5 EĞİTİLDİ ama REJECT:** lokal CPU PEFT, 1203 adım, **46.75 saat**,
  loss 2.66→0.60. Adapter: `models/adapters/achilles_lora_v5/`. **AMA base-vs-adapter
  karşılaştırması (gerçek PEFT yükleme) gösterdi: adapter base'den DAHA KÖTÜ** — tekrar
  döngüsü (overfit), "pasaja göre" uydurma, maliyetsiz rakam uydurma. **TERFİ EDİLMEDİ**
  (Kural 2). RAG hâlâ base modelle çalışıyor. Detay: `memory/v5-adapter-regression.md`,
  `reports/evals/adapter_smoke_compare.json`.
- **KÖK SEBEP:** sentetik-QA recipe ("pasaja göre cevapla" + adversarial disiplin örneği yok).
- **🔧 EVAL HARNESS FIX (push'lu):** `app/training/adapter_eval.py` + `achilles lora-eval
  <adapter> --eval-set <jsonl> --n <k>` — adapter'ı transformers/PEFT ile GERÇEKTEN yükler,
  base ile kıyaslar (eski `ModelEvaluator` base Ollama'yı ölçüyordu, adapter'ı YÜKLEMİYORDU).
  UYARI: red-flag sezgisi negasyon-kör → otomatik verdict güvenilmez; LLM-judge gerekli.
- **▶️ ÖĞRENME DÖNGÜSÜ ÇALIŞIYOR** (2026-06-16 restart, 72h, `continuous-learning.sh`):
  48 kartsız makaleyi işliyor (kart/anlama/synth-qa). **AYRI OS SÜRECİ — yeni seansta da sürer.**
  `keep_alive=5m` (eğitim bitti). Yeni LoRA eğitimi başlatınca `.env: ACHILLES_OLLAMA_KEEP_ALIVE=0` yap.
- **Veri:** `data/lora_sft/lora_sft.jsonl` ~1266 örnek · `lora-split` → `data/training/jsonl/train.jsonl`
  (1203 train+63 valid). **DİKKAT:** `DatasetBuilder.build()` train.jsonl'i ezer (clobber) —
  `detached_launch.launch()` her başlatmada lora_sft'den yeniden böler. Detay: `memory/training-data-pipeline.md`.
- **RAG:** Ollama qwen3:4b + nomic-embed-text · **109 makale / 11341 parça** · kart kapsamı %56 (artıyor).
- **Bu seansta push'lananlar:** detached tek-tık eğitim + "EĞİTİME HAZIR" rozeti · CVD-safe renkler ·
  web bug avı fix'leri (XSS/auth/hata yönetimi) · çekirdek denetim fix'leri (backtester metrik:
  Sortino+trade-bazlı win-rate/PF, RAG retrieval) · güvenlik sertleştirme (TrustedHost/HSTS/pip-audit/
  gitleaks) · `update.sh` (Mac/Linux) + `update.ps1` sağlamlaştırma · Ollama `keep_alive` OOM fix.
- **🔒 BEKLEYEN (kullanıcı yönü):** (1) eğitim-veri reçetesini düzelt — adversarial disiplin
  örnekleri ekle, "pasaja göre" sızıntısını gider — SONRA yeniden eğit (körlemesine 47h retrain YOK).
  (2) eval judge'ı iyileştir (negasyon-farkında / LLM-judge).
- **Base model:** `Qwen/Qwen3-4B-Instruct-2507` (Ollama qwen3:4b ile birebir).
- **754 cybersecurity skill** `~/.claude/skills/`'e (global) kuruldu (kullanıcı isteği, alimirbagirzade
  fork) — savunma+ofansif spektrum; her oturuma yüklenir (context maliyeti); plugin'e çevrilebilir.
- **Otonom nöbet (ScheduleWakeup loop):** yeni seansta KENDİLİĞİNDEN devam ETMEZ — istenirse
  `/loop` ile yeniden kur. Öğrenme döngüsü (OS süreci) bağımsız sürer.
- **(O günkü commit:** `9115fd5` — güncel durum yukarıdaki "SON" bloğunda, `d54ab68`.)

---

## ✅ Bu Seansta Tamamlananlar (2026-06-13 → 06-14) — BÜYÜK PİVOT

**Tema:** "RAG eğitiminin sağlamlığını araştır, yeniden yaz, push." → 4-ajan
araştırma → dürüst teşhis → RAG-first robust + aşamalı (lokal-veri → bulut-GPU) eğitim.

### Commit'ler (sırayla)
- `2422c5d` RAG: Reranker'ı canlı yola bağla (over-fetch + rerank) — A2
- `ab1220b` Sentetik QA motoru (`synthetic_qa_builder.py`) + sürekli CPU-eğitimi durdur
- `ca39070` Aşamalı eğitim protokolü (Stage 1/2 doc + skill + `cloud_notebook.py`)
- `5d31657` **fix:** 4B base-model 1.5B hardcode (şema/sunucu/UI/CLI 5 katman)
- `96e55b7` RAG: hybrid BM25 (A3) + prompt birleştirme (A4)
- `ae2b998` RAG: cross-encoder reranker (A8, opt-in)
- `ec2fd4c` RAG: contextual retrieval (P2, opt-in + `reindex-contextual`)
- `4212b30` LoRA: near-duplicate dedup (A7)
- `85d61de` LoRA: `synth-qa-bulk` (checkpoint'li bulk üretim)
- `383860c` fix: upload limiti 100 MB tutarlı
- `e24cd80` feat: paper-düzeyi dedup (başlık-normalize)
- `61df8dd` fix: gece döngüsü `lora-dataset` clobber'ı kaldır (Stage 2 dataset korunsun)

### Önemli bilgiler
- **Gerçek 4B eğitimi CPU'da YAPILMAZ** (haftalar + overfit). Yalnız bulut-GPU.
- Sentetik veri synthetic_qa.jsonl'de **birikir**; birleşik Stage 2 dataset'i
  `lora-cloud-prep` üretir (lora_sft.jsonl). Döngü artık bu dosyayı ezmiyor.
- `.env`: `ACHILLES_MAX_UPLOAD_MB=100` · Fable 5 erişimi YOK (hesap/plan).

### ▶️ SABAH SIRADAKİ İŞ: Stage 2 gerçek eğitim (~30 dk, bulut-GPU)
1. `huggingface-cli upload <kullanıcı>/achilles-lora-sft data/lora_sft/lora_sft.jsonl lora_sft.jsonl --repo-type dataset`
2. HF READ token → Kaggle Secrets (ad: `HF_TOKEN`)
3. Kaggle T4×2 + Internet ON → `notebooks/achilles_lora_stage2.ipynb` → `HF_DATASET_REPO`'ya kullanıcı adı → Run All
4. İndir GGUF + Modelfile → `ollama create achilles -f Modelfile`
5. Eval: `$env:ACHILLES_LLM_MODEL='achilles'; uv run achilles evaluate evals/discipline_core.jsonl`
Detay: `docs/PROTOKOL_BULUT_EGITIM.md` · skill: `/bulut-egitim-protokolu`

### Opsiyonel (kullanıcı tetikler)
- `reindex-contextual` → P2 aktive (korpusu yeniden-embed, sonra `.env` `ACHILLES_RAG_CONTEXTUAL_EMBED=true`)
- Cross-encoder: `uv pip install sentence-transformers` + `ACHILLES_RAG_CROSS_ENCODER=true`

---

## ✅ Bu Seansta Tamamlananlar (2026-06-11)

### 1. LoRA Gate Pipeline Düzeltmesi — `cf66c63`
**Sorun:** Windows'ta yüklenen 8 knowledge card, içeriksiz (title=None, main_claim boş) halde onaylanmıştı.
Bu kartlar Gate 0/3/4'ü blokluyordu → auto-LoRA pipeline gate_failed durumunda kalıyordu.

**Çözüm:**
- `app/lora/control_plane.py` → `_run_card_gates`: gate'lerden önce `_card_text()==""` kartları filtrele
- DB'deki 8 içeriksiz kart `rejected` yapıldı (`lora_eligible=0`)
- `storage/auto_lora_state.json` → `ready_to_train`'e getirildi
- **Sonuç:** 9/9 gate PASS, 26 temiz kart pipeline'da

### 2. Windows PEFT Backend Düzeltmesi — `c2205d2`
**Sorun:** `auto_pipeline.start_training()` her platformda `python -m mlx_lm.lora` çalıştırıyordu.
MLX macOS'a özel olduğundan Windows'ta eğitim anında çöküyordu.

**Çözüm:** `app/lora/auto_pipeline.py` → `detect_lora_backend()` ile platform tespiti:
- macOS ARM64 → `mlx_lm.lora` (değişiklik yok)
- Windows/Linux → `app.training.peft_lora_train --run`

**Windows eğitim ön koşulu:**
```
uv pip install torch transformers peft datasets accelerate
```

### 3. Eğitim UI — Açıklamalar + Auto-LoRA Konfig — `1f5ca50`
- Her eğitim ayarına (model, adapter, iterasyon, batch, katman) Türkçe açıklama eklendi
- Auto-LoRA bölümüne kendi adapter adı + iterasyon inputları eklendi (`#autoLoraAdapterName`, `#autoLoraIters`)
- JS validation: boş ad ve 50–5000 dışı iter engellendi
- CSS: `.setting-group`, `.setting-desc` sınıfları eklendi

### 4. Genel Sağlık Kontrolü + Bug Fix'leri — `1aef7eb` `572b605` `47ffd4c`

**Tespit edilen ve düzeltilen hatalar:**

| Dosya | Hata | Düzeltme |
|-------|------|----------|
| `server.py:17` | `HTTPException` import eksik → runtime `NameError` | Import eklendi |
| `server.py:455` | `PaperComprehension.total` yok → 500 error | → `total_score` |
| `comprehension_scorer.py:47` | `list_knowledge_cards` yok | → `get_latest_knowledge_card` |
| `comprehension_scorer.py:117` | `llm.is_available()` yok | → `llm.available()` |
| `formula_extractor.py` | `available()` guard eksik → test isolation bug | LLM çağrısından önce `available()` kontrolü |
| `comprehension_scorer.py` | unused `json` import | ruff auto-fix |
| `sqlite_store.py` | quoted type annotations | ruff auto-fix |

**Sonuç:** 405 test PASS · ruff CLEAN · mypy CLEAN (app/ üzerinde)

### Önceki Seans Detayları (2026-06-11 akşam)

#### Batch Comprehension Skor Butonu
- `GET /api/papers/comprehension/all` — tüm skorları tek çağrıda döner (N+1 fix)
- `POST /api/papers/comprehension/batch` — kartı olan tüm makaleler için skor hesapla
- Frontend: `🧪 TÜM SKORLARI HESAPLA` butonu, client-side cache reset

#### Math-Aware Chunker + Formül Pipeline
- `app/ingestion/chunker.py` → `_MATH_BLOCK_RE` ile `$...$` / `\[...\]` / `\begin{equation}` bloklarını korur
- `app/memory/paper_indexer.py` → ingestion sonrası otomatik: formül çıkarma → kavram grafiği → çapraz sentez
- `app/research/cross_paper_synthesizer.py` → 8 kategori kombinasyonu, SHA256 idempotency, 8 fallback template

#### 27 Makale Yeniden İndekslendi
- 4194 chunk · 142 formül (7 kategori) · 19 çapraz sentez örneği · 121 toplam eğitim örneği
- Her eğitim ayarına (model, adapter, iterasyon, batch, katman) Türkçe açıklama eklendi
- Auto-LoRA bölümüne kendi adapter adı + iterasyon inputları eklendi (`#autoLoraAdapterName`, `#autoLoraIters`)
- JS validation: boş ad ve 50–5000 dışı iter engellendi
- CSS: `.setting-group`, `.setting-desc` sınıfları eklendi

---

## 📁 Kritik Dosyalar

| Dosya | Görev |
|-------|-------|
| `app/lora/auto_pipeline.py` | Otomatik pipeline + platform tespiti (MLX vs PEFT) |
| `app/lora/control_plane.py` | Gate 0-8 orkestrasyonu, boş kart filtresi |
| `app/lora/gates.py` | 9 kalite kapısı (source/schema/domain/quality/math/…) |
| `app/training/peft_lora_train.py` | Windows/Linux PEFT eğitimi (CLI: `--run`) |
| `app/training/mlx_lora_train.py` | macOS MLX eğitimi |
| `app/training/backend.py` | Platform tespiti: `detect_lora_backend()` |
| `app/training/dataset_builder.py` | `training_examples` tablosundan JSONL üretir |
| `app/memory/sqlite_store.py` | Ana DB (kartlar, örnekler, adapter'lar) |
| `storage/auto_lora_state.json` | Pipeline anlık durumu (stage, gate_summary, …) |

## 🏛️ Mimari Kararlar (değişmez)

- **API key entegrasyonu planlanmıyor.** OpenAI/Anthropic/Google key desteği kodda mevcut ama aktif olarak geliştirilmeyecek. Sistem tamamen lokal-öncelikli.
- **Uzun vadeli hedef: lokal 120B OSS model.** (ör. Llama 3.1 405B, Qwen2.5 72B+) Ollama üzerinden çalışacak. Geçiş için tek değişiklik: `.env` dosyasında `ACHILLES_LLM_MODEL` ve `ACHILLES_OLLAMA_HOST`. Kod değişikliği gerekmez.
- **Ollama host:** `127.0.0.1:11434` (localhost değil — Windows IPv6 sorunundan dolayı).

---

## ⚠️ Bilinen Sınırlamalar / Dikkat Noktaları

- **Dataset builder vs LoRA dataset builder:** `app/training/dataset_builder.py` → `training_examples` tablosunu okur. `app/lora/dataset_builder.py` → `knowledge_cards` tablosunu okur. İkisi farklı sistem.
- **Windows'ta CPU eğitimi yavaş** (~2-4 saat). Hızlı eğitim için Eğitim sekmesindeki "Colab Notebook İndir" butonunu kullan.
- **Gate tekrar çalıştırılırsa** `auto_lora_state.json` `checking` → `ready_to_train` veya `gate_failed`'a geçer.
- **İçeriksiz kart oluşursa:** ingestion sırasında LLM cevabı boş gelirse kart DB'ye boş kaydedilir. Gate pipeline bunu filtreler ama kart DB'de `approved` olarak kalabilir — `control_plane` bunu tolere eder.

## 🔧 Sonraki Olası Görevler

- [ ] Windows'ta PEFT eğitim progress'ini web UI'da SSE ile yayınla
- [ ] Boş kart oluşmasını önlemek için ingestion'a `title` validasyonu ekle
- [ ] Gate özet raporunu web UI'da göster (hangi kartlar reddedildi, neden)
- [ ] Eğitim süresi tahmini: iterasyon × batch × donanım → dakika bilgisi

---

## 🗂 Önceki Seanslar (referans)

### 2026-06-09 (öğleden sonra)
- Windows kalıcı kurulum (`install.ps1`), Task Scheduler entegrasyonu
- macOS LaunchAgent (`com.achilles.web.plist`)
- PEFT/PyTorch install fix, `update.ps1` encoding düzeltmesi
- Qwen3 thinking-mode response fix, test suite (405 test)

### 2026-06-10
- PDF yükleme event loop blocking fix (BackgroundTasks)
- Makale başlığı fallback (dosya adından)
- UI CSS fix: Risk modal, Pine Script, Backtest grid, Training light mode

---

## 🔴 Sıradaki Görevler (öncelik sırasıyla)

### 1. Windows'ta Son Güncellemeyi Al (5 dakika)
```powershell
cd "$env:USERPROFILE\achilles"
git pull
.\scripts\start-server.ps1 -Install
.\scripts\start-server.ps1 -Status
```
Ollama + web server'ın birlikte başladığını doğrula.

### 2. Daha Fazla Makale + Kart → LoRA
```bash
uv run achilles arxiv "momentum volatility regime" --max 10
uv run achilles lora-audit && uv run achilles lora-dataset
```
Hedef: 50+ onaylı kart.

### 3. 500-iter LoRA Eğitimi
```bash
uv run achilles train --run --iters 500
```

### 4. Paper Mastery Testi
```bash
uv run achilles mastery-queue --enqueue-all && uv run achilles mastery-queue --run-all
```

### 5. DPO Hazırlığı (uzun vadeli)
500+ onaylı kart gerekiyor.

---

## 📋 CLI Komut Referansı (tam liste)

```bash
# Sistem
uv run achilles init / status

# Makaleler
uv run achilles ingest / arxiv "sorgu" / arxiv-sync / papers

# Araştırma
uv run achilles ask "soru" / card <id> / extract-formulas / research "soru"

# Backtest
uv run achilles backtest <csv> / pine [strateji]

# Eğitim
uv run achilles dataset / chain-dataset / unified-dataset
uv run achilles mastery-to-sft
uv run achilles train / train --run
uv run achilles tool-use-train / tool-use-dataset
uv run achilles reward-analyze / auto-research

# Paper Mastery
uv run achilles mastery-run <paper_id>
uv run achilles mastery-queue [--enqueue-all|--run-next|--run-all]
uv run achilles mastery-score <paper_id>
uv run achilles mastery-report <paper_id>

# LoRA Control Plane
uv run achilles lora-status       # pipeline genel durumu
uv run achilles lora-audit        # Gate 0-8 denetle
uv run achilles lora-dataset      # dataset oluştur (--dry-run varsayılan)
uv run achilles lora-registry     # adapter kayıtları listele

# Web UI
uv run achilles-web   →  http://127.0.0.1:8765
```

---

## 🏗️ Mimari Özeti

```
app/
├── ingestion/    PDF okuma, metadata, chunklama, arXiv fetcher
├── memory/       SQLite + ChromaDB + embedding + MasteryStore
├── brain/        RAG, bilgi kartı, model routing
├── learning/     Paper Mastery Agent (0-100 skor)
├── lora/         LoRA Control Plane — Gate 0-8 + adapter registry
├── training/     Dataset builder, LoRA, reward, DPO, unified
├── trading/      StrategyIR, backtest, indikatörler, evaluator
├── verification/ Citation, grounding, context sufficiency
├── evals/        Eval framework, metrics
├── agents/       OSS Learning Agent, research orchestrator
└── main.py       CLI (Typer)

.claude/agents/
├── lora-control-orchestrator.md
├── lora-dataset-auditor.md
├── lora-curriculum-classifier.md
├── lora-domain-verifier.md
├── lora-math-physics-statistics-verifier.md
├── lora-logic-philosophy-reviewer.md
├── lora-safety-secret-scanner.md   ← BLOCKER gate
├── lora-trainer-configurator.md
├── lora-evaluation-reviewer.md
└── lora-adapter-registry-manager.md
```

---

## 🧪 Test Komutu

```bash
uv run pytest                    # 407 test
uv run pytest -x -q              # hızlı, ilk hatada dur
make format && make lint && make typecheck && make test
```

---

## 🔑 Önemli Dosyalar

| Dosya | Ne içerir |
|-------|-----------|
| `TRAINING_ROADMAP.md` | Eğitim stratejisi + tamamlanan/bekleyen |
| `app/main.py` | Tüm CLI komutları |
| `app/lora/control_plane.py` | LoRA Gate 0-8 orchestrator |
| `app/lora/safety_scanner.py` | Blocker gate — secrets/PII/finansal tavsiye |
| `app/lora/adapter_registry.py` | Adapter yaşam döngüsü yönetimi |
| `app/learning/paper_mastery_agent.py` | Ana mastery pipeline |
| `app/training/unified_dataset.py` | Faz 2 dataset birleştirici |
| `configs/lora/lora_profiles.yaml` | 3 LoRA eğitim profili |
| `configs/eval/lora_eval_questions.yaml` | 50 eval sorusu |
| `.env.example` | Tüm ayar değişkenleri |

> [bug-scan 2026-06-22_0900] Weekly Tier-1 scan done -> reports/bug-scan/scan-2026-06-22_0900.md
