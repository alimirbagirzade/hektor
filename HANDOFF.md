# HANDOFF — Hektor

_Depo: https://github.com/alimirbagirzade/hektor · Son güncelleme: 2026-09-17 (kurtarma yetkisi artık approval_id ile — zaman penceresi yalnız yedek · aynı gün: start-train.ps1/watchdog Kural 8 boşlukları kapandı · 2026-09-15: tekrar patolojisinin kökü)_

Yerel-öncelikli AI **trading araştırma** sistemi (Windows · macOS Apple Silicon · Linux).
**Canlı bot değil, yatırım tavsiyesi değil.**

---

## Bu depo nedir?

`alimirbagirzade/achilles` (v1) deposunun **temizlenmiş ve onarılmış** hâlidir. v1'in tüm
çalışan sistemi taşındı; ölü kod, bulut-API kalıntıları ve bayat oturum geçmişi taşınmadı.
Ne çıkarıldığı ve neden: **[docs/MIGRASYON_2.0.md](docs/MIGRASYON_2.0.md)**.

v1 geçmişi arşiv olarak eski depoda durur; bu depo tek "initial commit" ile başlar.

**2026-09-04 — proje `achilles2.0` → `hektor` olarak yeniden adlandırıldı.** CLI `hektor`
/ `hektor-web`, ortam öneki `HEKTOR_`. Mevcut kurulumlar bozulmasın diye iki geriye dönük
uyum kancası korunur (bkz. `app/config/settings.py`, `tests/test_legacy_env_migration.py`):

- Eski `ACHILLES_*` ortam değişkenleri ve `.env` satırları hâlâ okunur (uyarı loglar);
  açık `HEKTOR_*` ayarı her zaman kazanır. Bu destek **geçicidir**.
- Yalnız eski `storage/sqlite/achilles_trader_ai.db` varsa ona düşülür — korpus/kart
  geçmişi öksüz kalmaz. Dosyayı (WAL/SHM ile birlikte) yeniden adlandırmak yeterlidir.

Bilinçli olarak **değişmeyen** dış sözleşme: `.achpkg` uzantısı, JSON'daki
`achilles_package_version` anahtarı ve `source: achilles_research` değeri — bunları
Entropia tarafı okur, kırılmasınlar diye korundu.

---

## Durum

| Alan | Durum |
|---|---|
| Kapı (`make ci`) | **CI (Linux):** ✅ main'de yeşil — `1ceb867`, `e99a3bb`, `5841f7c` ve `4f32768` push koşuları success. 2026-09-07 ile 2026-09-13 arası kırmızıydı (`test_sentinel_autostart_probe` ×2; bkz. 2026-09-13 kaydı). **Yerel (Windows):** ✅ ruff format --check (429 dosya, app+tests) + ruff check + mypy (219 dosya) + pytest **2179 passed, 5 skipped, 4 deselected** (2026-09-16, `-m "not ollama"`, `4db170d` + Kademe 2 düzeltmeleri). **Yerel ✅ tek başına kapı sayılmaz.** |
| Eğitim yığını | `train-cpu` extra'sı kilitte **sabit**: torch 2.14.0 · transformers 5.16.1 · tokenizers 0.23.2 · peft 0.20.0 · accelerate 1.14.0. Kilit = kurulu ortam (birebir). Yükseltmek açık karardır → ardından adapter yeniden değerlendirilmeli |
| Son adapter | `hektor_lora_v8_4b` (600 adım, 39s 32dk, 2026-09-10 23:27) → **REJECT**, terfi ETMEDİ. Gerekçe aşağıda (2026-09-11 seansı). `hektor_lora_v9_4b` 2026-09-15'te başlatıldı ama 21/600 adımda askıya alınıp sonlandırıldı (checkpoint YOK) — veri düzeltildikten sonra baştan koşacak (2026-09-16 kaydı) |
| LLM | Yalnız yerel Ollama (`qwen3:4b-instruct-2507-q4_K_M` varsayılan, 2026-09-13'ten beri; önceki `qwen3:4b` = Thinking-2507). Bulut API istemcisi YOK. |
| Gözetimsiz eğitim | **KAPALI** (`unattended_training_enabled=false`) → her gerçek eğitim tek-kullanımlık insan onayı ister (Kural 8) |
| Arka plan döngüleri | Web açılışında çalışır; `HEKTOR_BACKGROUND_LOOPS_ENABLED=false` ile kapatılır (testlerde kapalı). **Bu makinede `.env` şu an `false`** — 2026-09-06 sunucu yeniden başlatmasında döngüler kapalı açıldı; açmak bilinçli karar ister |
| Bilgi kartı tanımı | "Kartı var" = canlı (`rejected` değil) **ve içerikli** (`card_has_content`: title veya main_claim alfanümerik). Boş kart = kart yok → makale yeniden kartlanabilir (`has_knowledge_card` / `get_latest_knowledge_card`) |
| Test izolasyonu | Testler gerçek `data/` · `storage/` ağacına **yazamaz**; ihlal ederse paket FAIL verir |

---

## Yeni seansta ilk 5 dakika

```bash
uv sync --extra dev            # bağımlılıklar (pytest/ruff/mypy 'dev' extra'sındadır)
uv run hektor status         # Ollama + korpus + model durumu
uv run hektor doctor         # bu makine origin/main'de mi (salt-okuma teşhis)
make ci                        # format + lint + typecheck + test
uv run hektor-web            # http://127.0.0.1:8765
```

Ollama kapalıysa: `ollama serve` → `ollama pull qwen3:4b-instruct-2507-q4_K_M` → `ollama pull nomic-embed-text`.

---

## Sekiz mutlak kural (CLAUDE.md)

1. Yatırım tavsiyesi üretme — çıktı daima hipotez + test noktası.
2. Test edilmeden "başarılı" deme — backtest + out-of-sample şart.
3. Maliyetleri yok sayma — komisyon + slippage her backtest'te.
4. Look-ahead yasak — pozisyon `shift(1)` ile gecikmeli.
5. `eval`/`exec` yok — strateji kuralları yalnız güvenli regex ile.
6. Determinizm — rastgelelik daima `seed` ile.
7. Kaynak uydurma — retrieval boşsa açıkça söyle.
8. Otomatik ağır eğitim yok — `train` varsayılan dry-run; gerçek eğitim `--run` + taze insan onayı.

---

## Sıradaki adım — LoRA eğitimi (insan onayı bekliyor)

Veri hattı v1'de kapanmıştı; bu depoda **veri taşınmadı** (`data/`, `storage/`, `models/`,
`vector_db/` git'te izlenmez). Yeni makinede sıfırdan üretilir:

```bash
uv run hektor ingest                 # PDF'leri data/papers/raw_pdf/ altına koy, sonra indeksle
uv run hektor synth-qa-bulk --target 1000
uv run hektor lora-curate --run
uv run python scripts/assemble_sft.py  # → data/lora_sft/lora_sft.jsonl (KANONİK)
uv run hektor lora-audit             # Gate 0-7 (--run ile 0-8) · --json ile makine-okunabilir
uv run hektor pretrain-gate          # GO / NO-GO
uv run hektor lora-split
# Kural 8 kapısı:
uv run hektor approval-approve <id>
.\scripts\start-train.ps1 -Profile discipline_safe_local   # DETACHED
```

**Kapılar artık betikte ZORUNLU** (2026-09-10, `139a4bd`): `start-train.ps1`,
`lora-split`'ten önce `pretrain-gate` + `lora-audit` koşar; NO-GO / FAIL / girdi=0 ise
**eğitim başlamaz**, kapı çalıştırılamazsa da başlamaz (Kural 2). Bilinçli insan
override'ı: `-SkipGate`. Yukarıdaki elle çağrılar artık teşhis içindir, ön koşul değil.

Eğitim sonrası: `lora-eval` (min_n≥5, degenerasyon + boş-cevap vetolu) → adapter **ADAY**;
production terfisi ayrı insan onayı ister.

> **v8 örneği (2026-09-11):** eval **REJECT** verdi — skor base'i açık ara geçmesine
> rağmen tek bir dejenere cevap kategorik veto. "Skor iyi" terfi gerekçesi değildir.

---

## Son seans — 2026-09-17: K8-b'nin geri kalanı — kurtarma artık ZAMAN PENCERESİ değil kimlik

Dal: `claude/k8b-approval-id-write-0dbedb`. **Eğitim başlatılmadı.**

Bu seans başladığında `main` zaten **paralel bir seansın** iki commit'ini içeriyordu
(aynı gece, `claude/burda-rag-qlora-training-ce0521` → PR #16): `51c634b` ("K8-b:
start-train.ps1 artık taze onay tüketiyor") `-Supervised` anahtarını ekleyip betiğin
koşulsuz `HEKTOR_TRAIN_SUPERVISED=1` vermesini kapatmıştı (§3 kapandı); `5b367c0`
("Eğitim nöbeti") `app/training/train_guard.py`'yi ekleyip nöbetçinin diriltmeden önce
`train-recovery-check` ile yetki doğrulamasını zorunlu kılmıştı (§4 kapandı). **Ama**
o modülün kendi sınırı belgeliydi (yukarıdaki "2026-09-16" kaydı §7): *"kurtarma
yetkisi, onayı koşu başlangıcına ZAMAN penceresiyle (±20 dk) bağlar; çünkü onayı
tüketen katman onay kimliğini durum dosyasına yazmıyor. Kimliği de yazmak daha sağlam
olur — açık iş."* Bu seansın tek işi **tam olarak bu açık iş**.

**Neden zaman penceresi zayıf:** `find_run_approval` tüketilen bir onayı yalnızca
`consumed_at`'in koşunun `started_at`'ine ≤20 dk yakın olmasına bakarak eşliyordu —
doğru onay biraz geç tüketilirse (yavaş model yükleme) reddedilir, ya da nadir bir
yarış durumunda pencere içindeki BAŞKA bir onay yanlışlıkla eşleşebilirdi. Kimlik
doğrudan yazılırsa tahmin gerekmez.

**Değişiklik (önceki iki commit'in üzerine, onları TEKRARLAMADAN):**
- `detached_launch.launch()` / `_status_payload()` artık `approval_id` parametresi
  alır ve `storage/train_status.json`'a yazar; web `/api/training/run` ve
  `auto_pipeline.start_training()` zaten tükettikleri `decision.approval_id`'yi geçirir.
- `start-train.ps1`: taze (`-Supervised` olmayan) başlatmada alt süreç onayı kendi
  tüketir (değişmedi — §3'ün çözümü); başarı sonrası log'dan tüketilen `apr_...`
  kimliği okunup durum dosyasına **işlenir** (`Add-Member approval_id`). Kurtarmada
  (`-Supervised -Resume`) nöbetçinin `train-recovery-check`'ten aldığı kimlik
  `-ApprovalId` ile geçirilip aynen durum dosyasına yazılır.
- `training-watchdog.ps1`: `train-recovery-check --json` çıktısının `details.approval_id`
  alanını okuyup `-ApprovalId` ile `start-train.ps1`'e iletir (önceden atılıyordu).
- `app/training/train_guard.py`: `find_run_approval` artık `approval_id` verilmişse
  ÖNCE tam kimlik eşleşmesi dener; bulunamaz/onaylı-tüketilmiş değilse zaman
  penceresine **düşmez** (sahte pozitif riski — geçersiz bir kimlik şüphelidir).
  Yalnız bu alan hiç YOKSA (eski/harici durum dosyaları, `mac-loop.sh` gibi) zaman
  penceresi yedek olarak kalır — geriye dönük uyum.
- Yeni CLI `approval-status <id>` (READ-ONLY tekil onay sorgusu); `train-authorize`
  adlı ayrı bir "üst katman tüketir" komutu YAZILMADI — zaten var olan `-Supervised`
  deseni (alt süreç kendi onayını tüketir) korunarak üstüne minimum yama yapıldı.

**Kapı:** ruff format --check + ruff check + mypy (repo geneli, 0 hata) + hedefli
pytest (`test_train_guard.py`, `test_start_train_approval_gate.py`,
`test_train_recipe_persistence.py`, `test_agent_phase2_cli.py`) yeşil; tam paket
`-m "not ollama"` ayrıca koşuldu.

**Ders (süreç):** aynı "K8-b" adı iki ayrı seansta bağımsız kullanıldı ve biri diğerini
main'e girdikten SONRA fark etti — `git fetch` + `merge-base` kontrolü olmadan PR
açmak sessiz çakışmaya yol açabiliyor. PR açmadan/merge etmeden önce `origin/main`'i
taze çekmek ve aynı alanda (burada: eğitim onay kapısı) yakın zamanda commit var mı
diye bakmak ucuz bir kontrol.

**Sıradaki:** v9 eğitimi hâlâ insan onayı bekliyor (bkz. altta "2026-09-16" kaydı §6).

---

## Son seans — 2026-09-16: v9 öncesi Kademe 2 + iki Kural 8 boşluğu

Dal: `claude/burda-rag-qlora-training-ce0521`. Tam rapor bu makinede **yerel**:
`reports/bug-scan/kademe2-2026-09-16.md` (bu klasör `.gitignore`'da — tarama raporları commit
edilmez, özet buraya yazılır). Aşağısı o raporun özetidir.

### 1. Yarım kalan v9 koşusu sonlandırıldı
2026-09-15 14:06'da başlatılan `hektor_lora_v9_4b` (600 adım) 17:54'te RAG üretimine CPU
açmak için **askıya alınmış** (`NtSuspendProcess`), 21/600 adımda donmuştu; checkpoint yok
(adapter klasörü boş). Kullanıcı kararıyla sonlandırıldı → v9 temiz veriyle baştan koşacak.
Kaynak verisi (`lora_sft.jsonl`, 1743 satır) güncel kapıda GO alıyordu; asıl sorun aşağıdaki
veri bulgularıydı.

### 2. Kademe 2 (2 finder + her bulguya 2 bağımsız doğrulayıcı)
İki oyla onaylanıp **düzeltilenler**: disiplin rotasyonu soruyla cevabı yanlış eşliyordu
(gerçek veride 147-175 satır; **uyumsuz çift 175 → 0**) · çekimser tuzaklarda maliyet/OOS
dayanağı kaybolmuştu · grup e-postaları (`{a,b}@uni.edu`) maskelenmiyordu (**28 adres**) ·
disiplin ikizleri train/valid'e dağılıyordu (`skeleton_id` ile gruplandı) · boş/null cevap
kapıdan GO alıyordu · `lora-cloud-prep` kanonik birleştirmeyi atlıyordu · `pretrain-gate`
NO-GO'da çıkış kodu 0 veriyordu. Düşük etkili 6 bulgu raporda açık bırakıldı.

### 3. İki Kural 8 boşluğu (biri düzeltildi, biri AÇIK)
- **Bayat onay (düzeltildi):** 2026-09-08'de `hektor_lora_v8_4b` için verilip hiç tüketilmemiş
  onay, 2026-09-15 20:36 UTC'de **başka** bir eğitim (`hektor_lora`, 500 adım) için tüketildi.
  Artık `APPROVAL_TTL_HOURS = 12`; bayat onay bulunamaz, tüketilemez, damgalanmaz. Karar zamanı
  okunamıyorsa taze SAYILMAZ. (Düzeltmenin kendi regresyonunu test yakaladı: toplu UPDATE
  sonrası ORM nesnesi tazelenmeyince çağıran, tükettiği onayı "tüketilmemiş" sanıyordu.)
- **`start-train.ps1` onay TÜKETMİYOR (AÇIK):** betik `HEKTOR_TRAIN_SUPERVISED=1` geçerek CLI'nın
  Kural 8 kapısını atlatıyor ve log'a "üst katman onayı kullanıldı" yazıyor; oysa betik hiçbir
  onay isteği açmıyor. Web endpoint'i sözleşmeyi tutuyor, betik tutmuyor — v7/v8 koşuları da
  böyle başlamıştı. Önerilen düzeltme raporda (`-Supervised` anahtarı + nöbetçi muafiyeti);
  nöbetçinin kurtarma yolunu değiştirdiği için gözetimli seansa bırakıldı.

### 4. Nöbetçi, onaysız bir eğitimi diriltti (canlı gözlem)
2026-09-15 23:36'da web arayüzünden `hektor_lora` (500 adım) başlatıldı ve 0. adımda öldü.
`training-watchdog.ps1` durum dosyasını görüp 23:49'da **aynı koşuyu yeniden başlattı**; 5,5 saat
eski veriyle, onaysız ve fark edilmeden eğitti (8,2 GB RAM, kart üretimini yavaşlattı).
`start-train.ps1 -Stop` ile durduruldu ve durum dosyası temizlendi. Ders: ölü bir koşunun durum
dosyası = nöbetçi için kalıcı yetki; §3'teki boşlukla birleşince onay kapısı fiilen devre dışı.

### 5. RAG
`kaynak-tamamla`: düşük anlama skorlu 5 makale için arXiv'den 9 PDF indirildi (alaka kapısı
elemeleri raporda; 1 aday 404), ingest edildi → korpus **233 makale**. 9 yeni makalenin kartı
`read-all` ile üretiliyor. Eval kirlenmesi ölçüldü: `discipline_core`'un 16 sorusunun disiplin
verisiyle birebir eşleşmesi **0** (en yüksek benzerlik 0,32).

### 6. Veri yeniden kuruldu, kapılar GEÇİLDİ — v9 **insan onayı bekliyor**

Veri, düzeltilmiş kodla (bu dal) yeniden kuruldu; veri ağacı ana checkout'ta kaldı
(`HEKTOR_ROOT_PATH` ile worktree kodu + ana `data/`+`storage/`). PR #14 **merge edilmedi**;
eğitim de bu dalın kodundan koşacak ki düzeltmeler fiilen eğitilen veriye girsin.

| Adım | Sonuç |
|---|---|
| `assemble_sft.py` | **1937 örnek** (synth 1283 + kart 233 → dedup 1453 + disiplin 484) |
| `pretrain-gate` | **GO** — PII 0, sır 0, şablon 8-gram bloğu 0, okunamayan 0, boş cevap 0. Tek uyarı: 91 "strateji" cevabında maliyet token'ı yok (sentetik QA kaynaklı) |
| `lora-audit` | **passed** — 236/236 kart onaylı, 0 red (42 "gözden geçir") |
| Veri doğrulaması | grup e-postası **0** (öncesi 28) · disiplin 484/484 `skeleton_id` · uyumsuz soru-cevap **0** (öncesi ~147) |
| Onay | `apr_2410dd477207` — istek `train --run` (SUPERVISED'sız) kapısıyla açıldı, **insan onayladı** (2026-09-16 07:30 UTC), eğitim başlarken **tüketildi** (07:40:54 UTC) |
| Eğitim | **KOŞUYOR** — `hektor_lora_v9_4b`, 600 örnek × 1 epoch, `discipline_safe_local`, bf16/CPU; 2026-09-16 10:40 (yerel) başladı, train=1820 valid=117. Log: `logs/train-v9.log` + `logs/train-v9-err.log` |

**Eğitimi başlatmak için (insan):**
```bash
uv run hektor approval-approve apr_2410dd477207
```
Sonra bu dalın kodundan ayrık başlat (worktree `.venv`'inde train-cpu kurulu):
`HEKTOR_ROOT_PATH=<ana checkout>` + `train --run --backend peft --adapter-name hektor_lora_v9_4b
--iterations 600 --profile discipline_safe_local --max-examples 600` (SUPERVISED **verme** —
onayı CLI tüketsin). `scripts/start-train.ps1` kullanılmadı: §3'teki açık boşluk yüzünden
onay tüketmiyor.

**Not:** ana `storage/train_status.json` bilinçli olarak YAZILMADI — nöbetçi o dosyayı görünce
çöken koşuyu eski kodla ve onaysız diriltiyor (§4). Yani bu koşuda otomatik kurtarma yok.
`train-doctor` bu yüzden koşan v9 için **DİKKAT** verir ("durum kaydı yok → onaya bağlanamıyor");
bu doğru davranıştır. PR #14 main'e girdikten sonra (nöbetçi kurtarma kapısı orada olunca)
durum dosyası güvenle yazılabilir.

### 7. Tekrarı önleyen sistem: eğitim nöbeti (`train_guard`)

Gecenin iki olayının ortak kökü — *koşan eğitimin sağlığını ve yetkisini kimse sorgulamıyordu* —
koda bağlandı. Yeni modül `app/training/train_guard.py` (saf fonksiyonlar: zaman/süreç/dosya
bilgisi dışarıdan verilir → test gerçek süreç istemez).

| Ne | Nasıl |
|---|---|
| `uv run hektor train-doctor [--json]` | Koşan eğitimin sağlığı: log ilerlemiyor (>45 dk) · CPU ~0 (askıda) · koşuya bağlı **tüketilmiş onay yok** (Kural 8) · veri koşudan sonra değişti · süreç yok ama durum dosyası duruyor (ölü koşu kaydı) · **durum kaydı olmayan koşu**. Çıkış 1 = DİKKAT |
| `uv run hektor train-recovery-check [--json]` | Nöbetçi diriltmeye yetkili mi? Koşu başlangıcına denk gelen tüketilmiş onay + durum dosyası tazeliği (≤72 s) + veri değişmemiş. Çıkış 3 = yetkisiz |
| `scripts/training-watchdog.ps1` | Diriltmeden ÖNCE bu kontrolü çağırır; kontrol koşturulamazsa da **dirilme yok** (fail-closed, Kural 2) |
| `.claude/agents/egitim-nobetcisi.md` | Komutu kullanan ince ajan (salt-okuma; eğitim başlatmaz/durdurmaz, onay vermez) |
| `tests/test_train_guard.py` | Gecenin iki senaryosu test: onaysız ölü koşunun dirilmesi ve askıdaki koşunun fark edilmemesi artık kırmızı |

Ayrıca `approval-approve` "bulunamadı" mesajı artık **bakılan veri kökünü** yazıyor: komut
worktree'den koşulduğunda (kendi boş `data/storage` ağacı) onay bulunamıyordu ve sebep
görünmüyordu — kullanıcı bunu canlı yaşadı.

**Sınır (kapandı — bkz. üstte "2026-09-17" kaydı):** kurtarma yetkisi, onayı koşu başlangıcına
ZAMAN penceresiyle (±20 dk) bağlıyordu; çünkü onayı tüketen katman onay kimliğini durum
dosyasına yazmıyordu. Artık yazıyor — `find_run_approval` kimlik varsa ZAMAN PENCERESİNE
düşmeden doğrudan eşleşir; pencere yalnız kimliksiz (eski/harici) durum dosyaları için yedek.

### 8. Eğitimi yavaşlatan gizli yük: web sunucusunun formül çıkarımı (ölçüldü, giderildi)

v9 adımları beklenen ~3,2 dk yerine 4-6,5 dk sürüyordu. İlk şüphe benim test koşularımdı; asıl
sebep başkaydı:

| Kanıt | Bulgu |
|---|---|
| 55 sn kesintisiz bağlantı örneklemesi (port 11434) | Ollama'nın **tek** istemcisi `hektor-web` (pid 20576, 15.09 12:41'den beri) |
| Ollama `server.log` | Gece boyunca saatte 130-290 `/api/generate`, her biri 25-60 sn, bazıları 60 sn'de 500 |
| Web günlüğü | Saat 11'de 980 `httpx` satırı; `formula_extractor` makale bitince tek satır yazıyor |
| `formulas` tablosu | 233 makalenin 26'sı işlenmiş → kalan iş günler sürerdi |
| CPU / bellek | `llama-server` %394-533 CPU; model KV önbelleğiyle **9,7 GB** RAM, boş RAM 0,4 GB'a düştü |

Kök: dün geceki bir web ingest çağrısı zenginleştirmeli (`enrich=True`) yolu tetikledi ve sunucu
korpusu chunk chunk formül çıkarımına soktu. Arka plan döngüleri `.env`'de **kapalıydı**; iş
döngüden değil istekten doğmuştu. Kodda **iptal kancası yok**, görev kuyruğu boş.

Tanıda elenenler (tekrar aranmasın): panonun `/api/status` yoklaması yalnız `/api/tags` çağırır ·
sentinel `probe_llm` da öyle, periyodik iş parçacığı yok · `enrich_corpus()` formül çıkarmaz ·
orkestrasyon koşuları `blocked` durumda durmuştu.

**Giderildi (kullanıcı onayıyla):** web sunucusu ağacı yükseltilmiş çalıştığı için UAC istemli
`taskkill /T` ile kapatıldı (sunucu `RunLevel=Highest` görevlerle başlıyor; `update.ps1` eğitim
koşarken kasıtlı olarak hiçbir şey yapmıyor) → `ollama stop` ile model boşaltıldı → boş RAM
0,4 → 10,0 GB. Eğitim hiç kesilmedi. Sonraki adımlar **3:33** ve **3:01** sürdü.

**Ders:** uzun CPU eğitimi sırasında web panosu kapalı olmalı ya da en azından ingest/formül
çıkarımı tetiklenmemeli. `train-doctor` bu yükü görmedi — yalnız koşunun kendi sağlığına
bakıyor; "makinedeki başka bir LLM işi eğitimi yavaşlatıyor" kontrolü açık iş. Pano, eğitim
bitince `HektorWeb` göreviyle açılır; formül çıkarımı kendiliğinden geri gelmez
(`uv run hektor extract-formulas` ile bilinçli başlatılır).

---

## Son seans — 2026-09-15: tekrar patolojisinin kökü → şablon iskeleti ezberi

v8'in eval'de ~%19 (16'da 3) tekrar döngüsünün kök nedeni araştırıldı. **Eğitim
başlatılmadı** (Kural 8); düzeltmeler veri üreticisi + eğitim-öncesi kapıdadır.

### 1. Ölçülenler

| Hipotez / ölçüm | Sonuç |
|---|---|
| Kesilen örnek (`max_seq_length=1024`, `<|im_end|>` kaybı → "durmayı öğrenmeme") | **ELENDİ** — v7 (300) ve v8 (600) alt kümelerinin tamamı sığıyor, bitiş token'ı hep yerinde (cevap medyanı 68 token; trainer'ın kendi `sample_rows`/`build_masked_labels`'ı + gerçek tokenizer ile) |
| Eval soruları eğitimde birebir var mı (ezber) | **Hayır** — `discipline_core` 16 sorunun 0'ı; bozulma genellemede |
| Disiplin verisi yapısı | 528 örnek = **33 cevap iskeleti**, her biri strateji adı dışında birebir **16 kopya**; iki ortak "test noktası" kuyruğu cevapların ~%70'inde |
| v8'in gerçek 600'lük alt kümesi | 141 cevap (%24) bu 33 iskeletten; tek iskelet 7× |
| v7 → v8 kuyruk maruziyeti ↔ tekrar oranı | 18/13 → 32/26 kez ↔ 1/16 → **3/16** (örnek sayısı da 2× → *karışık*, kanıt değil) |
| Cevaplar arası 8-gram doküman frekansı (>%2 cevapta geçen ifade) | `synthetic_qa` **0** · v8 `train.jsonl` **31-37** · eski disiplin havuzu 486 |
| "kural kuralı…" döngüsü | Kuyruktan değil, `look_ahead`/`backtest_yok` cevaplarındaki "(kural 4)", "bu bir kuraldır" kalıplarından |

**Güçlü hipotez:** model yüksek frekanslı cevap iskeletlerini ezberliyor, yeni sorularda
iskeletleri karıştırıp döngüye giriyor. Kesin kanıt yalnız düzeltilmiş veriyle yeniden eğitim
+ `lora-eval` (aynı reçete) ile gelir.

### 2. Yapılanlar (kullanıcı kararları: LLM'siz şablon çoğaltma · kapı NO-GO %2 / uyarı %1)

- **`app/training/discipline_dataset.py`:** ortak kuyruklar (`_TEST_TAIL/_ALT`) kaldırıldı;
  test adımı her cevapta kendi cümlesiyle. Tuzak başına **6 cevap** (3'tü) → **66 iskelet**,
  iskelet başına ≤**9** kopya (16'ydı); örnek sayısı aynı (528, rotasyon `answers[(v+si)%6]`).
  "kural N" atıfları çıkarıldı. Kural 1-4 içeriği (shift(1), maliyet, OOS, aday/hipotez,
  çekimserlik) mevcut testlerle korunuyor.
- **`app/training/dataset_quality.py` (`pretrain-gate`):** yeni **şablon tekrarı** kuralı —
  aynı 8-kelimelik ifade cevapların **>%2**'sinde → **NO-GO**; %1-%2 → uyarı; küçük setlerde
  mutlak alt sınır 10 tekrar. Rapor alanları: `top_template_ngram(_share)`,
  `template_ngrams_over_block`; CLI panelinde "En sık şablon ifadesi". `start-train.ps1`,
  orkestratör ve delege aynı `audit_dataset`'i çağırdığı için kural her eğitim yolunda geçerli.
- **Testler:** `test_discipline_dataset.py` (6 cevap/tuzak, 66 iskelet, ≤9 kopya, hiçbir 8-gram
  iki iskelette birden yok, "kural N" yok) · `test_dataset_quality.py` (blok, uyarı bandı,
  küçük-set koruması, benzersiz cevapta sinyal yok).

### 3. Gerçek veriyle kalibrasyon (yeni kod, salt-okuma)

| Veri | Karar | >%2 8-gram | En sık 8-gram |
|---|---|---|---|
| v8 `train.jsonl` (1616) | **NO-GO** | 37 | %5,7 |
| eski `lora_sft.jsonl` (1701) | **NO-GO** | 37 | %5,9 |
| `synthetic_qa.jsonl` (1117) | GO | 0 | %0,4 |
| eski disiplin havuzu (528) | NO-GO | 486 | %24,2 |
| **yeni** disiplin havuzu (528) | **GO** | 0 | %1,7 (9 tekrar < alt sınır) |
| `synthetic_qa` + yeni disiplin %25 (1489) | **GO** | 0 | %0,5 |

`hektor pretrain-gate --jsonl data/training/jsonl/train.jsonl` (gerçek CLI) → NO-GO paneli +
"En sık şablon ifadesi" satırı. CLI NO-GO'da da çıkış 0 verir; kapıyı tüketen üç yol
(`start-train.ps1` `--json`→`verdict`, orkestrasyon delegesi, yerel eğitim orkestratörü)
kararı `verdict`'ten okuduğu için eğitim gerçekten başlamaz (doğrulandı).

**Kapı:** ruff format --check (424) + ruff check + mypy (218) + pytest **2143 passed,
3 skipped, 4 deselected** (`-m "not ollama"`, PR #10 dalı PR #9 ile birleştirildikten sonra,
`6054975`). PR #10 CI (Linux) "lint · types · tests (offline)" yeşil. Birleşme öncesi
(`1bd51cc` tabanı): 2097 passed.

### 4. PR #9 ile birleşme (aynı gün, paralel seans)

Başka bir seans aynı kökü paralel ele alıp PR #9'u main'e aldı (aşağıdaki kayıt): 2 sabit kuyruk
→ **16 kuyruk varyantı** + `pretrain-gate`'e **kapanış ezberi** (>%4), boş/okunamayan satır ve
sır/PII kuralları. PR #10 main'le birleştirilirken iki çözüm çapraz ölçüldü:

| Disiplin havuzu | İskelet / en çok kopya | 8-gram kuralı (PR #10) | Kapanış + PII kuralı (PR #9) |
|---|---|---|---|
| PR #9 (16 kuyruk, tuzak başına 3 cevap) | 134 / **16** | **NO-GO** — 611 ifade >%2, en sık %6,1 | GO (kapanış %3,0) |
| PR #10 (kuyruk yok, tuzak başına 6 cevap) | 66 / **9** | **GO** — 0 | **GO** (kapanış %1,7) |
| v8 `train.jsonl` | — | NO-GO (37) | NO-GO (kapanış %5,7) |

16 kuyruk yalnız cümle SONUNU çeşitlendiriyor; cevap gövdeleri hâlâ 16 kopya — kapanış kuralı
bunu görmüyor, 8-gram kuralı görüyor. **Karar:** disiplin verisi PR #10'un (66 iskelet;
`_TEST_TAILS`/`_tail_for` kaldırıldı — başka kullanıcısı yoktu), kapıda **iki kural birlikte**
(kapanış + 8-gram), PR #9'un okunamayan satır / sır-PII / e-posta maskeleme / eval
düzeltmeleri aynen korundu. `test_dataset_quality` sızıntı testinin fixture'ı iki kuralı da
bozmayacak biçimde benzersizleştirildi.

### 5. Sıradaki (eğitimden ÖNCE, sırayla)

1. **Veriyi yeniden üret** — kanonik sıra aşağıdaki PR #9 kaydının "Sıradaki" bölümüdür
   (`read-all` → `synth-qa-bulk --since` → `assemble_sft.py` → `pretrain-gate` + `lora-audit`).
   `assemble_sft.py` ve `lora-cloud-prep` disiplin satırlarını KODDAN alır → yeni şablonlar
   otomatik girer. **Dikkat:** `assemble_sft.py` `lora_sft.jsonl`'ı, `lora-split`
   `train/valid.jsonl`'ı YEDEKSİZ ezer — önce kopyala.
2. `uv run hektor pretrain-gate` → yeni veride **GO** beklenir; eski veri artık NO-GO verir
   (bu kasıtlı: v9 aynı kalıplarla başlatılamaz).
3. **Kademe 2:** PR #9'da koşuldu, ama disiplin verisi o avdan SONRA değişti (PR #10) — eğitim
   öncesi yeniden değerlendirilmeli (en azından disiplin + kapı değişikliklerinin hedefli avı).
4. Aynı reçeteyle (v8: `discipline_safe_local`, 600 örnek, 1 epoch) yeniden eğitim —
   **insan onayı** (Kural 8). Değişen tek şey veri olsun ki sonuç yoruma açık olmasın.
5. `lora-eval` → tekrar oranı düşüyor mu? **Not:** PR #9 `discipline_core`'un 4 sorusunu
   yeniden yazdı (kirlenme) → v8'in 3/16'sıyla doğrudan kıyas geçersiz; v8'i yeni setle yeniden
   ölçmek gerekir. Tekrar sürerse sıradaki aday reçete tarafı (NEFTune kapalı; açık iş v8-3c).

---

## Son seans — 2026-09-15: trading kaynakları + Kademe 2 derin av (eğitim öncesi)

Dal: `claude/trading-dosya-makaleler-462f10` (henüz push YOK). Eğitim BAŞLATILMADI.

### 1. Trading korpusu
- `Desktop\RAG Kaynak\tader kitapları\` → 39 açık-erişim PDF (arXiv / yazar sayfası / NBER;
  `%PDF` + başlık doğrulamalı) + kullanıcının koyduğu 25 kitap. İndeks: klasördeki `00_OKU_ONCE.md`.
- RAG: 38 makale ingest (1 tekrar atlandı) → 199 makale / 17 350 chunk; 38 kart + skor (%100 okunmuş).
  Kitaplar `data/papers/raw_pdf/trading-kitaplar-2026-09-14/` → ingest SÜRÜYORDU (12/25).
- **Makine uykuya geçince ingest 6,5 saat durdu** (22:56–05:24, Kernel-Power 506/507) — Ollama
  kilitlenmesi DEĞİL. Uzun işlerde uyku kapatılmalı.
- İlk kitap ingest'i `WinError 10013` (Ollama bağlantı reddi, geçici) ile düştü; yeniden koşu
  idempotent devam etti.

### 2. Kademe 2 — 4 finder + adversarial doğrulama (0 çürütülen iddia)
Düzeltildi (bu dal):
- **Eval:** çok-setli eval'de dejenerasyon `reject`'i kayboluyordu (auto_pipeline → EVAL_PASSED);
  `_is_degenerate` araya cümle giren 3× tekrarı ve `!`/satır tekrarını kaçırıyordu (v8 #5);
  `accept` için ≥2 bayrak farkı; `must_avoid` olumsuzlamaya duyarlı ("garanti kâr yoktur");
  otomatik hat tüm eval sorularını koşar (ilk-8 kırpması v8'in #11 çöküşünü hiç görmüyordu);
  LLM çevrimdışı cevap `llm_unavailable` bayrağı; registry REJECTED/CANDIDATE'i terfi ettirmez;
  gözetimsiz politika terfiyi ASLA yetkilendirmez; merdiven kıyası sessiz atlanmaz.
- **Veri (v8 tekrar patolojisinin veri kökü):** disiplin cevaplarının 208/528'i iki SABİT kuyrukla
  bitiyordu ("Ölçülmesi gereken bir hipotez var…" v8'de 3× tekrarlanan cümle) → 16 kuyruk
  varyantı. pretrain-gate yeni NO-GO'lar: kapanış ezberi (>%4), boş/okunamayan satır,
  prompt/completion biçimi, **sır/PII** (eğitilen dosyanın kendisi). Yazar e-postaları
  (1701 satırın 92'sinde 191 adres) `sft_assembly` birleştirmesinde maskelenir.
- **Kapı yolları:** `launch()` (web/auto) artık pretrain-gate'ten geçer; boş kanonik kaynakta
  bayat split sayılmaz; `start-train.ps1` SUPERVISED/BASE_MODEL env'ini sızdırmaz, status'a pid yazar.
- **Kart/QA:** eşik altı pending kart reddedilip yeniden denenir; kart üretimi seed'li;
  `synth-qa-bulk` hedef aşılmışken sahte başarı basmaz + `--since` filtresi.
- **Eval seti değişti:** `evals/discipline_core.jsonl`'daki 4 soru eğitim şablonlarının neredeyse
  birebiriydi (kirlenme) → yeniden yazıldı. **v8 skorlarıyla doğrudan kıyas artık geçersiz.**

**Mevcut `data/lora_sft/lora_sft.jsonl` yeni kapıda NO-GO** (kapanış %5.9 + 191 PII) — beklenen;
veri yeniden kurulmalı (aşağı).

Açık kalanlar (düşük): synth-qa `--resume` kısmen işlenen makaleyi atlıyor; synth-qa yazma
kilitsiz; `auto-chain.sh`/`mac-loop.sh` zayıf kart onayı; `lora-dataset` `lora_sft.jsonl`'ı ezer;
`lora-split` satır-düzeyi karıştırır (train --run kaynak-gruplu yeniden böler); maskelemede atlanan
örnek max_steps'i düşürmez. **Backtest bulguları** (Sharpe yıllıklandırma etikete bağlı → sahte
pass, NaN `!=` al-tut, risk komutu sentetik veri, MACD period yok sayılır…) ayrı göreve bırakıldı.

### 3. Sıradaki (insan onayı bekliyor — Kural 8)
```bash
uv run hektor read-all --cards 25 --scores 25          # kitap kartları (ingest bitince)
uv run hektor synth-qa-bulk --since 2026-09-13 --target 5000   # yalnız yeni trading kaynakları
uv run python scripts/assemble_sft.py                  # e-posta maskeli birleşik set
uv run hektor pretrain-gate && uv run hektor lora-audit
.\scripts\start-train.ps1 -Adapter hektor_lora_v9_4b   # onay isteği → approval-approve <id>
```
Not: sentetik QA bugün 211 makalenin yalnız 40'ını kapsıyor; tüm korpus CPU'da onlarca saat.

### 4. Kitap soru-cevaplarında kalite çöküşü → üretim durduruldu, düzeltildi (2026-09-15)
- İlk 326 kitap örneğinde **%33 çekimser** ("pasajda açıklanmamıştır") ve **%40 "Pasaj…"
  açılışı** ölçüldü (v8'e giren eski veride %3). Kök neden: `build_for_paper`
  `chunks[:max_chunks]` alıyordu → kitabın kapak/telif/içindekiler/şekil listesi chunk'ları.
  Prompt'taki "trading kuralına çevrilemiyorsa belirt" talimatı çekimserliği besliyordu.
- Düzeltme: `_select_chunks` ön sayfaları eler ve içerik chunk'larını belgeye eşit yayar;
  `is_low_value_answer` "pasaj" atıflı / "içermiyor" cevaplarını üreticide ve
  `sft_assembly`'de atar; pretrain-gate "Pasaj…" açılışı >%10 → NO-GO (eski önek listesi
  yalnız "pasaja göre"yi tanıyordu).
- 326 bozuk örnek çıkarıldı (yedek: `data/lora_sft/synthetic_qa.before_booksfix_2026-09-15.jsonl.bak`).
  Pilot (López de Prado, 2 chunk): içerik chunk #79/#186, 0 düşük-değerli, ~86 sn/çağrı.
  Üretim yeniden başlatıldı (`--since 2026-09-13 --max-chunks 8 --resume`, ~12 saat).
- Kitapların bir kısmı `OceanofPDF.com` filigranı taşıyor (kullanıcının koyduğu dosyalar).

### 5. Bilgi kartında aynı ön sayfa sorunu → düzeltildi (2026-09-15, `claude/kart-on-sayfa-baslik`)
- López de Prado (paper_4179e6720bae) için 2 kart (06:08 rejected, 14:30 pending) `title` BOŞ
  → `is_substantive_card` (title≥8, main_claim≥40) geçmiyor. Kök neden: `build()`
  `full_text[:6000]` = kapak + #1–#10 şekil/denklem listesi.
- Düzeltme: ön sayfa ölçütü `app/brain/chunk_selection.py`'ye taşındı (synth-qa + kart ortak).
  Baştaki içerik-dışı blok >1500 krk ise kart alıntısı = ilk içerik chunk'ları (yarı bütçe) +
  eşit yayılmış 3 kesit; temiz makalede eski `full_text[:max_chars]` aynen. LLM başlığı
  boş/kısaysa (yalnız main_claim doluyken) `papers.title` → dosya adından doldurulur.
- Gerçek veri (salt-okuma): 256 chunk'ın 117'si içerik, ilk içerik #11; 5.996 krk ön sayfa atlandı.
- ⚠ `papers.title` bu kitapta KESİK: `"ADVANCES IN FINANCIAL"` → kart bu başlığı alır (eşik
  geçer ama eksik). Kökü ingest başlık çıkarımı; açık iş.
- **Karar bekliyor:** pending kart dokunulmadı. Birleşmeden sonra yeniden üretim
  (`uv run hektor card paper_4179e6720bae`) ve eski pending kartın reddi kullanıcıya bırakıldı.

> **Birleştirme notu:** `main`'deki 2026-09-14 kaydının v8-3 açık işi (şablon
> çeşitlendirme / frekans tavanı / pretrain-gate kuralı) bu dalla karşılandı; dedektör
> iki çözümün birleşimidir (`_SENTENCE_REPEAT_MIN` + genişletilmiş cümle ayırıcı).
> NEFTune-kapalı kontrol koşusu hâlâ açık.
> **Güncelleme (PR #10 birleşmesi):** bu kaydın 16 kuyruk varyantı, üstteki kayıttaki 66 iskeletli
> disiplin verisiyle değiştirildi (gövde tekrarı 8-gram kuralında NO-GO veriyordu); kapanış
> kuralı ve buradaki diğer tüm kapılar korunuyor.

---

## Son seans — 2026-09-14: gece doğrulaması geçti · onay kapısı canlı sınamada ısırdı

### 1. Gece senkronu düzeltmesi gerçek koşuda doğrulandı

`HektorUpdate` (RunLevel=Highest) 2026-09-14 03:00 koşusu, `logs/update.log`:

```
[2026-09-14 03:00] Guncelleme basliyor (Force=False)...
[2026-09-14 03:00] DURUM dal=main HEAD=e99a3bb ahead=0 behind=0
[2026-09-14 03:00] Sunucu baslatildi (PID 5388).
[2026-09-14 03:00] SONUC: OK
```

- **tokenizers damgası değişmedi** (2026-09-13 20:09:32). Web `uv run --no-sync` ile yeniden
  başladı ve ortamı senkronlamadı; üç gece üst üste bozan tetikleyici tetiklenmedi.
- `import transformers` çalışıyor (5.16.1 / 0.23.2); görev sonucu 0.
- Yükseltilmiş görev eski yükseltilmiş web'i (PID 18760) durdurabildi; web artık yeni kodla
  koşuyor (PID 4548, 03:00:07). Yeni sağlık kontrolü port sahibinin başlatılan süreç
  olduğunu doğruladı. 2026-09-13 kaydındaki Yönetici adımına **gerek kalmadı**.
- Senkron adımı koşmadı (kod değişmedi, `-Force` yok). Kalan üç fark (regex, setuptools,
  proje paketinin editable yeniden kurulumu) hâlâ uygulanmadı; kod değiştiren ilk gece koşusu
  uygular. Aciliyeti yok.
- 2026-09-13 21:01 ve 21:08'deki yükseltilmemiş `-Force` koşuları yeni betikle beklendiği gibi
  açıkça `SONUC: HATA (web durdurulamadi)` verdi (21:08'i başka biri koşturdu).

### 2. `needs-approval` onay kapısı fail-closed — canlı sınamada ısırdı (PR #5, `5841f7c`)

`dependency-approval-label` işi etiket eklenemezse `::warning::`'e düşüp yeşil dönüyordu ve
etiket repoda hiç yoktu. Etiket oluşturuldu, iş fail-closed yapıldı:

- etiket eklenemez ya da PR'da doğrulanamazsa `::error::` + kırmızı;
- `git diff … || true` kaldırıldı (fark hesaplanamazsa sessizce geçmez);
- `printf | grep -q` yerine here-string (pipefail altında SIGPIPE → yanlış negatif → yeşil).

Sahte `git`/`gh` ile 6/6 senaryo; `tests/test_github_workflows_static.py`'de 4 yeni kilit.

**Canlı sınama:** PR #5 iş akışı dosyasını değiştirdiği için kapsamdaydı. Etiketi
`github-actions[bot]` ekledi; iş çıktısı: "needs-approval etiketi eklendi ve PR'da
doğrulandı". PR #2 aynı durumda etiketsiz geçmişti.

---

## Son seans — 2026-09-13 (akşam): `qwen3:4b` düşünmesi kapatılamıyor → LocalLLM düzeltmesi

**Belirti:** `think:false` + `/no_think` verilse de qwen3:4b cevaba etiketsiz düz metin
düşünme yazıyordu ("Okay, let's tackle…"); `LocalLLM` ile "2+2" 240 sn'de bitmedi.
RAG kutusu, `hektor ask`, RLM ve kart üretimi aynı yolu kullanır.

**Kök neden (ölçüldü, Ollama 0.34.0):** yereldeki `qwen3:4b` manifest özeti
(`359d7dd4bcda`) = `qwen3:4b-thinking-2507-q4_K_M` → hibrit değil, **yalnız-düşünen**
model. Şablon asistan turunu koşulsuz `<think>` ile açıyor, `.Think` dalı yok. Canlı
denendi, HEPSİ etkisiz: `/api/generate` think:false, `/api/chat` think:false, raw prompt +
boş `<think></think>` ön-dolgusu, chat asistan ön-dolgusu. `think:true` ise düşünmeyi
`thinking` alanına doğru ayırıyor (2+2 → `"4"`, 43 sn, ~4 tok/s). `format=json` +
`think:false` hızlı ve temiz (15 token); `format=json` + `think:true` JSON'u `thinking`'e
yazıp cevabı BOŞALTIYOR. Eski kod boş cevapta `thinking`'i döndürüyordu → sızıntı yolu.

**Düzeltme (`app/brain/local_llm.py`):** `/api/show` ile kip sınıflandırması
(`none`/`toggle`/`forced`); `forced` + serbest metin → `think:true` + ek bütçe
(`HEKTOR_LLM_THINKING_EXTRA_TOKENS=1024`), format'lı çağrı → `think:false`;
`num_predict` artık her zaman sınırlı (`HEKTOR_LLM_DEFAULT_MAX_TOKENS=1024`,
tavan `HEKTOR_LLM_MAX_TOKENS_CAP=4096`); `thinking` alanı ASLA cevap değil; `<think>`
etiketleri temizlenir; bütçe cevaba geçmeden biterse `LLMUnavailable` (çağıranlar
zaten ele alıyor). Testler: `tests/test_local_llm_thinking.py` (sahte HTTP) + 2 canlı
`@pytest.mark.ollama` (2+2 temiz cevap 74 sn · JSON 4.6 sn).

**Varsayılan model değişti (kullanıcı kararı):** `qwen3:4b-instruct-2507-q4_K_M` çekildi
(ID `0edcdef34593` = registry manifesti) ve varsayılan yapıldı: `settings.llm_model`,
`.env.example`, `setup.ps1`/`setup.sh` [1], README, `docs/MIMARI_REFERANS.md`,
`model_registry.yaml` (id `qwen3_4b_q4` aynı, yalnız ollama adı), ana checkout `.env`.
Bu, PEFT base'i (`Qwen/Qwen3-4B-Instruct-2507`) ile Ollama modelini İLK KEZ birebir eşler
(`settings.py`'daki "qwen3:4b = Instruct-2507" yorumu yanlıştı, düzeltildi).

| Aynı çağrı (CPU, temperature 0) | `qwen3:4b` (Thinking-2507) | `qwen3:4b-instruct-2507-q4_K_M` |
|---|---|---|
| "Tek kelimeyle: 2+2" | 43 sn (think:true) · think:false → düşünme sızıntısı | **1.5 sn**, `"4"` |
| EN→TR çeviri ("5-10%") | 1536 token düşünme, cevap YOK (450 sn) | **9.5 sn**, doğru, "5-10%" korundu |
| `format=json` | 6.8 sn | 3.9 sn |

**Sınıflandırıcı tuzağı (düzeltildi):** Instruct-2507 de `thinking` yeteneği ilan ediyor ve
şablonunda (geçmiş turlar için) kapalı `<think>…</think>` var → ilk kural onu `forced`
sayardı. Kural artık "şablondaki son `<think>` kapanmıyorsa forced"; prompt token'larıyla
doğrulandı (thinking modelde `<think>`=151667 var, instruct'ta yok).

**Eski etiket hâlâ yerel diskte** (`qwen3:4b`, 2.5 GB) — `ollama rm qwen3:4b` kullanıcı kararı.

**Canlı doğrulama durumu (2026-09-14):** `LocalLLM().think_mode()` → `qwen3:4b`=`forced`,
instruct=`toggle` ✅; varsayılan modelle çeviri doğru ve "5-10%" korundu ✅. Canlı
`test_canli_ollama_dusunme_cevaba_sizmaz` ✅; `test_canli_ollama_json_hizli_ve_gecerli` ❌ —
**kuyruk yüzünden**: aynı anda ana checkout'ta `hektor read-all --cards 38 --scores 38`
(başlangıç 09-14 17:52) Ollama'nın tek slotunu kullanıyordu; istekler 2-5 dk bekledi, 120 sn
timeout düştü (server.log'da 1m59s `500`). Boşta ölçüm 3.9 sn idi. Test timeout'u 600 sn'ye
çekildi; **read-all bitince yeniden koşulmalı** (`uv run pytest -m ollama
tests/test_local_llm_thinking.py`). O read-all koşusu `.env` değişikliğinden SONRA
başladığı için kartları zaten yeni instruct modelle üretiyor.

---

## Son seans — 2026-09-13 (akşam): "Eğitilen model istediğimiz gibi cevap vermiyor" → hibrit kaynaklı cevap

**Belirti.** Web'deki "3 · Eğitilen Modelle Sohbet" istenen tarzda cevap vermiyor; "model
bağlanmıyor mu?" şüphesi. Bulgular ölçülerek doğrulandı:

1. **Bağlantı gerçekten kopuktu:** `tokenizers 0.22.2` ↔ `transformers 5.16.1` →
   `/api/lora-chat` 503. Kök neden ve kalıcı düzeltme bir alttaki kayıttadır (örtük senkron,
   `4dec640` / `9439355` / `b24c355`). Bu makinenin ana venv'ine `tokenizers==0.23.2` uv
   önbelleğinden elle geri kuruldu; `import transformers/peft` OK.
2. **İki farklı "4B":** ARAŞTIRMA RAG kutusu / `hektor ask` / RLM = Ollama `qwen3:4b` (base,
   adapter'sız). Adapter Ollama'ya hiç aktarılmadı; `/api/ask`'in adapter yolu MLX (Windows'ta
   çalışamaz). Adapter base'i `Qwen3-4B-Instruct-2507` ≠ Ollama `qwen3:4b`.
3. **Eğitim verisi istenen formatı içermiyor:** `train.jsonl` (1616) içinde 9 bölümlü
   `Kısa Cevap/Test Planı` biçimi **%0**; cevap medyanı 209 karakter; %72 "BAĞLAM/SORU".
4. **Çıkarım ↔ eğitim uyuşmazlığı:** sohbet system prompt'suz ve bağlamsız soruyordu (eğitimde
   %92 system, %72 bağlam); arayüz varsayılanı alfabetik ilk = **v7**.
5. **Hız (ölçüldü):** PEFT bf16 CPU ~0,37 token/sn; Ollama Q4 4,1 token/sn (~11×).
   Canlı v8: çıplak soru → 110 token kalıp metin; eğitim formatı → 43 token, bağlama sadık cümle.

**Düzeltmeler (`be47281`).** Sohbet (web + CLI `lora-chat`) SYSTEM_PROMPT gönderir; "kaynaklı"
mod retrieval + eğitimle BİREBİR `BAĞLAM:\n…\n\nSORU: …` (≤2400 karakter), retrieval boşsa
model çağrılmaz; adapter listesi en yeni önce; 503 mesajı sürüm çatışmasını da söyler;
`adapter_eval._generate` opsiyonel `system` (eval bilinçli olarak system'siz kalır).

**Format kararı (kullanıcı): HİBRİT · yalnız Türkçe · kart içeriği özgün alıntı (`7f31696`).**
Gerekçe: tam raporu modele yazdırmak soru başına 30-60 dk; kartların ~%98'i İngilizce ve
`qwen3:4b` ile otomatik çeviri güvenilmez (düşünme sızıntısı + "5-10%"→"5-1.0%").
Reddedilenler: yeniden eğitim (Kademe 2 + ~33 sa + yavaş çıkarım), yalnız RAG yolu (LoRA devre
dışı), tek seferlik kart çevirisi (önce sızıntı çözülmeli).

`app/brain/hybrid_answer.py` (saf, deterministik) — kaynaklı modda 8 bölüm, her biri rozetli:

| # | Bölüm | Kaynak |
|---|---|---|
| 1 | Kısa Cevap | **model** (tek üretim, eğitim formatı) |
| 2 | Kaynaklar | kaynak (retrieval atıfları) |
| 3 | Bağlam Kalitesi | kural (`assess_confidence`, CRAG eşikleri 0,55/0,02 → Güçlü/Orta/Zayıf+uyarı) |
| 4 | Akademik Bulgu | kaynak (kart `main_claim`, ≤2 makale, "(kaynak, çevrilmedi)") |
| 5 | Trading Hipotezi | kaynak (kart hipotezleri ≤3 + sabit "test edilmemiş, sayılar doğrulanmamış") |
| 6 | Test Planı | kural (OOS, komisyon+slippage, `shift(1)`, seed, `/backtest-auditor`) |
| 7 | Riskler | kural (+ kart `risk_warnings` alıntısı) |
| 8 | Sonraki Adım | kural |

Formül bölümü yok (kartlarda formül alanı yok). Kaynaksız modda bölüm şablonu yok. Kartlar
üretimden ÖNCE çekilir. Gerçek retrieval + kartlarla uçtan uca denendi (model üretimi taklit):
8 bölüm doğru kaynaklardan doldu; "look-ahead bias" sorusunda Bağlam Kalitesi dürüstçe "Zayıf".
**Gerçek v8 ile canlı deneme (2026-09-14, web değil servis katmanı):** gerçek model + retrieval +
kartlar, "Trend takip stratejisinde look-ahead bias nasıl önlenir?". 8 bölüm doldu, Bağlam
Kalitesi "Zayıf" (0,66 / marj 0,02) uyardı. **Kısa Cevap zayıf ve yanıltıcı:** _"…geçmiş verileri
kullanarak gelecekteki performansı tahmin etmek yerine, stratejiyi geçmişteki verilerle test
ederek önlenir."_ — `shift(1)` gecikmesini hiç anmıyor; dejenere değil. Doğru disiplini yalnız
kural bölümleri taşıdı (hibrit tasarımın gerekçesi). Süre 2545 sn — CPU başka seansın kart
üretimi + test paketiyle paylaşıldı; boş CPU'da beklenti 2-5 dk (ölçülmedi). v8 zaten
REJECT; bu çıktı "adapter'a dayanma, kural/kaynak bölümlerine bak" kararını destekliyor.

**Kapı (rebase + tekrar koruması sonrası):** ruff format + ruff check + mypy (218) + pytest
**2085 passed, 4 skipped, 2 deselected**. PR #4 CI (Linux, ilk 3 commit): "lint · types ·
tests (offline)" yeşil. Dal ilk hâlinde `update.ps1` / `uv.lock` için aynı amaçlı değişiklik
taşıyordu; main'deki düzeltme (bir alttaki kayıt) üst küme olduğu için rebase'te bırakıldı.

**Tekrar koruması (2026-09-14).** v8'in ~%19 tekrar patolojisi eğitimle çözülmeden sohbette
görünür kalsın diye: `lora_chat_service` her çıktıyı eval'deki AYNI `_is_degenerate` ile
denetler → API `degenerate` bayrağı + arayüzde "tekrar döngüsü — cevaba dayanma" rozeti;
hibrit modda Kısa Cevap uyarılı, `collapse_repetition` ile tekrarları kesilmiş ve notlu.
Ham çıktı `answer`'da korunur; `repetition_penalty` gibi üretim ayarı KULLANILMADI (kusur
ölçülebilir kalsın). Aynı değişiklikle dedektör boşluğu kapandı ve kök neden sıraya kondu —
bkz. "Bilinen açık işler → v8 sonrası" 1 ve 3.

**Açık işler.**
- ~~Ollama 0.34.0 `qwen3:4b` `think:false` + `/no_think`'e rağmen düz metin düşünüyor~~ →
  **çözüldü** (bir üstteki "LocalLLM düzeltmesi" kaydı): `qwen3:4b` = Thinking-2507,
  varsayılan artık `qwen3:4b-instruct-2507-q4_K_M`. Kart çevirisi önündeki sızıntı engeli kalktı.
- Eğitilmiş modeli ana soru-cevap hattına bağlamak (GGUF→Ollama, ~11× hız).

---

## Son seans — 2026-09-13: CI 6 gündür kırmızıydı · düzeltme makineye ulaşmamıştı · asıl tetikleyici örtük senkron

### 1. 2026-09-11 düzeltmesi makineye hiç ulaşmadı → ortam yine bozuldu

İki gece sonra ölçüldü: tokenizers yine **0.22.2**, `import transformers` yine kırık
(dist-info damgası 12 Eyl 07:55). `logs/update.log`:

```
[2026-09-12 07:55] ff-only IRAKSAK HATASI.  HEAD=3a551b0 ahead=1 behind=2
[2026-09-13 08:40] ff-only IRAKSAK HATASI.  HEAD=3a551b0 ahead=1 behind=2
```

PR merge edilmemişti; ana checkout push edilmemiş `3a551b0`'de takılı olduğu için
`origin/main`'e yakınsayamıyordu — yeni kilit ve betik oraya hiç inmedi. **"Commit
edildi" ≠ "makinede".** Önceki seansın "görev bir kez koşmadan doğrulanmış sayma"
uyarısı tuttu: görev koştu, düzeltme orada değildi.

### 2. Önceki iki teşhis YANLIŞ adımı suçluyordu — asıl tetikleyici örtük senkron

| Kanıt | Ne gösteriyor |
|---|---|
| `update.log` 09-09 (`behind=0`), 09-11 (`behind=0`), 09-12 (iraksama) | Üçünde de "Kod x → y" satırı YOK → açık `uv sync` adımı (`$updated -or $Force`) **üç seferde de atlandı** |
| `update.ps1` 4. adım | Web `uv run --project … hektor-web` ile başlıyordu — `--no-sync` yok, `UV_NO_SYNC` yok |
| `uv run --help` (uv 0.11.19) | Varsayılan **inexact senkron**; `--exact` ayrı bayrak, `--no-sync` senkronu kapatır |
| dist-info damgaları | "Sunucu başlatıldı" satırıyla **aynı dakika** |
| Kurulu sürümler | Yalnız taban kümedeki tokenizers değişti; extra'daki torch/transformers/peft dokunulmadı |

Yani tokenizers'ı düşüren, web yeniden başlatmasının örtük senkronuydu: taban küme
kilide çekildi, extra'daki transformers yerinde kaldı → çift bozuldu. `3a551b0`'in commit
mesajı ("`uv sync --extra dev` düşürdü") ve aşağıdaki 2026-09-11 kaydı olayı yanlış
adıma bağlamıştı.

**Düzeltme:** `update.ps1` web başlatma `--no-sync` → tek senkron noktası açık 3. adım.
09-11'deki kilit hizalaması + `--extra train-cpu` doğru kalıyor (açık senkron koştuğunda
yığını korur), ama tek başına tetikleyiciyi kapatmıyordu. Kilit:
`test_update_web_baslatma_ortuk_senkron_yapmaz`.

### 3. CI (Linux) 2026-09-07'den beri kırmızıydı — "Kapı ✅" hep yerel Windows ölçümüydü

main'e her push (`57ad396` → `41407dd`) başarısız; **PR #1 kırmızıyken merge edildi.** Son
koşularda düşen yalnız iki test: `test_sentinel_autostart_probe.py::test_eski_kayit_duruyorsa_warn`
ve `::test_tam_kurulum_ok`.

**Kök neden:** fixture `sentinel.os.name = "nt"` yamalıyordu — o **global** `os` modülü.
Python 3.12'de `Path.__new__` sınıfı `os.name`'e göre seçer → Linux'ta `Path` →
`WindowsPath`, `/tmp/x.ps1` → `\tmp\x.ps1`, `is_file()` False → "fail". Windows'ta yama
no-op olduğundan yerel kapı hiç görmedi. Betiğin **var olduğu** iki senaryo düşüyordu;
olmadığı senaryo tesadüfen geçiyordu.

**Düzeltme:** `sentinel._is_windows()` dikişi; testler onu yamalar, global `os.name`'e
dokunmaz. Kilit: `test_fixture_global_os_name_degistirmez`. Linux'ta ancak CI doğrular.

**Ders:** Durum tablosundaki "Kapı" satırı bundan sonra yerel sonucun yanında **CI
durumunu** da yazar.

### 4. Bağımlılık onay kapısı hiç ısırmamış

`.github/workflows/claude-code-task.yml` → `dependency-approval-label` işi pyproject/uv.lock
değişince `needs-approval` etiketi ekler (_"insan onayı şart"_). PR #2 ikisini de
değiştiriyor, iş SUCCESS — ama PR'da etiket yok. Sebep: **`needs-approval` etiketi repoda
yok** → `gh pr edit --add-label` düşüyor, `::warning::`'e iniyor, iş yine yeşil.
Bu PR'daki bağımlılık değişikliğinin insan onayı sohbette açıkça verildi (relock için
"Yap", merge için "gerekiyorsa push yap"). Etiketin oluşturulması ayrı karar — açık işler.

### 5. `update.ps1` üç yerde SESSİZCE başarı bildiriyordu (merge sonrası ilk koşuda görüldü)

PR #2 merge edilip ana checkout sıfırlandıktan sonra `update.ps1 -Force` koşuldu: ekranda
"[OK] Web çalışıyor" — ama **exit 2**. Üç ayrı sessiz başarısızlık:

| Adım | Ne oldu | Neden görünmedi |
|---|---|---|
| 1. web durdurma | 08:40'taki `HektorUpdate` koşusunun başlattığı web (PID 18760) durmadı: **yükseltilmiş süreç** (görev `RunLevel=Highest`; açık `Stop-Process` → "Erişim engellendi", doğrulandı) | Üç yöntem de `-ErrorAction SilentlyContinue`; port boşaldı mı bakılmıyordu |
| 3. senkron | `uv sync` exit 2; o koşuda site-packages'ta hiçbir şey değişmedi | Çıktı `Out-Null`'a yutuluyor, çıkış kodu kontrol edilmiyordu |
| 5. sağlık | Yeni sunucu `[Errno 10048]` ile port'a bağlanamayıp kapandı | Yalnız "port dinliyor mu"ya bakılıyordu → eski süreci başarı saydı |

**Senkronu ne kilitledi (güçlü çıkarım, kesin değil):** o sırada ana `.venv`'in python'u ile
**başka bir oturumun** (`hektor-rag-config-607e2c` · "4b LLM soru cevapları") `repro_lora_chat.py`
betiği koşuyordu (19:59'dan itibaren); yükseltilmiş web de ana venv'den koşuyor. Windows'ta
yüklü `.pyd` silinemez. Aynı oturum 20:09'da ana venv'e **elle `tokenizers==0.23.2`** kurmuş
(transkriptte doğrulandı) — ortamın merge'ten önce doğru olmasının sebebi o, benim senkronum
değil. **Ders: birden çok oturum aynı ana `.venv`'i paylaşıyor**; senkron/kurulum öncesi venv'i
kullanan süreçlere bakılmalı.

**Düzeltme (`update.ps1`):** durdurma port boşalana kadar doğrulanır, boşalmazsa loglanıp
**exit 1** (senkrona ve başlatmaya geçilmez); senkron `Start-Process` ile gerçek çıkış koduyla
koşar, çıktı `logs\uv-sync*.log`'a, sonuç `update.log`'a yazılır; venv'i kullanan başka python
süreci varsa senkron **ertelenir**; sağlık kontrolü port sahibinin **başlatılan süreç ağacı**
olduğunu doğrular; betik yalnız her adım başarılıysa **exit 0** (görev sonucu gerçeği yansıtır).
Kilitler: `tests/test_legacy_scheduled_tasks.py` içindeki beş `test_update_*` testi.

**Makinenin bu kayıt anındaki durumu:** HEAD = `origin/main` (`1ceb867`); altı eğitim paketi
kilitle birebir, `import transformers` çalışıyor. Web **reset öncesi kodla** (yükseltilmiş PID 18760)
koşuyor; senkronun kalan üç farkı (regex, setuptools, proje paketinin editable yeniden kurulumu)
uygulanmadı. Bu düzeltme main'e girdikten sonraki yükseltilmiş `HektorUpdate` koşusu web'i
durdurup senkronu tamamlayabilir (kod değiştiği için senkron koşar). Beklemeden: Yönetici
PowerShell'den `Stop-Process -Id 18760 -Force`, ardından `.\update.ps1 -Force`.

---

## Son seans — 2026-09-11: v8 REJECT (tekrar patolojisi) + gece senkronu (teşhis 2026-09-13'te düzeltildi)

### 1. v8 değerlendirildi → **REJECT**, terfi etmedi

`hektor_lora_v8_4b` 2026-09-10 23:27'de bitti (600/600 adım, 39s 32dk).
`lora-eval` · `discipline_core` · n=16:

| | skor | bayrak |
|---|---|---|
| base (Qwen3-4B) | −0.125 | 18 |
| adapter (v8) | **+0.8125** | **3** |

Verdict **REJECT** — skor ve bayrak her ikisinde de büyük iyileşmeye rağmen. Sebep
`adapter_degenerate` vetosu: `app/training/adapter_eval.py` bunu açıkça _"skordan
BAĞIMSIZ veto"_ olarak tanımlar (v5 dersi). Veto doğru çalıştı.

**Bayrakların tek tek okunması, sayının gösterdiğinden kötü bir tablo verdi:**

- **#11 — gerçek çöküş.** `"Hayır — bu kural kural kural kural…"` token sınırına
  kadar ~70 kez. Tartışmasız dejenerasyon.
- **#5 — aynı patoloji, eşiğin ALTINDA kaldı.** `"Ölçülmesi gereken bir hipotez var."`
  üç kez birebir. `_is_degenerate`'in cümle-tekrarı eşiği (`unique <= len//2`) bunu
  yakalamadı; bayrak `ignores_costs` olarak düştü. **Dedektör boşluğu** — bkz. açık işler.
- **#3 — dedektör yanlış alarmı, ama masum değil.** `"komisyon + spread dahil getiri,
  komisyon + spread dahil max drawdown ve komisyon + spread dahil Sharpe"` → 3-gram
  döngüsü tetikledi. Çöküş değil; anlamı sağlam ama şablonvari. Aynı eğilimin hafifi.

Yani tekrar patolojisi **16 cevabın 3'ünde (~%19)**, 1'inde değil. v5 disiplin
regresyonunun aynı sınıfı, düşük oranda.

**Madalyonun öteki yüzü güçlü.** Base'in bayrak dağılımı projenin yasakladığı
davranışların listesi: `ignores_costs` ×10, `guaranteed_profit`, "garanti", "kesin
kazan", `success_without_test`, "net rakam", "bu sefer farklı", "agresif gir". v8
bunların hepsini bıraktı — düzgün çekimser kalıyor, maliyet token'larını adlandırıyor,
test noktası çerçeveliyor. **Ne söyleyeceğini öğrendi; nasıl söyleyeceğinde bozuluyor.**

Rapor: `reports/evals/adapter_eval_hektor_lora_v8_4b_discipline_core.json`.
Diğer iki eval seti (`overfit_awareness`, `risk_management`) **koşulmadı** — açık iş.

### 2. Gece senkronu ve eğitim yığını (⚠ teşhis 2026-09-13'te DÜZELTİLDİ — üstteki kayda bak)

**Belirti.** v8 bitti, `lora-eval` `ImportError: tokenizers>=0.23.1 … found 0.22.2`
ile düştü. Yani 39 saatlik koşu, tam değerlendirileceği anda ölçülemez hâldeydi.

**Zincir (bu seansta yazıldığı hâliyle — YANLIŞ ADIM).** Burada tetikleyici açık
`uv sync --extra dev` sanıldı: `train-cpu` senkron kümesinin dışında kaldığı için ortak
bağımlılıkların kilide çekildiği düşünüldü. 2026-09-13'te `update.log` gösterdi ki
olayların üçünde de açık senkron **atlanmıştı**; düşüren, web başlatmanın `--no-sync`'siz
`uv run`'ıydı. Aşağıdaki kilit hizalaması ve `--extra train-cpu` yine doğru, ama
tetikleyiciyi kapatmıyordu.

**`3a551b0`'deki koruma neden yetmedi.** O koruma yalnız eğitim **CANLIYKEN** devreye
giriyor. v8 23:27'de bitince 03:00 görevi kendini serbest sandı. Koruma koşuyu
koruyordu, koşunun **DEĞERLENDİRİLMESİNİ** değil.

**Düzeltme (`9439355`).** pyproject'te `train-cpu` sürümleri v8'i eğiten ve
değerlendiren yığına sabitlendi (taban+tavan); `uv.lock` altı pakette de kurulu ortamla
birebir; `update.ps1` senkronu `--extra dev --extra train-cpu` koşuyor.

> **Sıra önemliydi:** `--extra train-cpu` TEK BAŞINA yeni bir bozulma yaratırdı — kilit
> torch 2.12.0 derken makinede 2.14.0 kuruluydu, gece senkronu torch'u DÜŞÜRÜRDÜ.
> Önce kilit kurulu ortama hizalandı, extra sonra eklendi.

Ayrıca (repo kendi dersi, `test_verify_install_uv_sync_inexact_kullanir`): `uv sync`
istenen küme dışındaki paketleri **siler**; 2026-09-07'de torch/transformers/peft böyle
silinmişti. `update.ps1` `--inexact` kullanmıyordu — `train-cpu`'yu kümeye almak bunu da
kapatır. Diğer extra'lar (fastmcp/markdown-pdf/mlx-lm) makinede kurulu değil (doğrulandı).

İki regresyon testi eklendi (`tests/test_legacy_scheduled_tasks.py`): senkron
`train-cpu`'yu kapsıyor mu, eğitim koruması duruyor mu.

### 3. Zamanlayıcı + git hijyeni

- `start-server.ps1 -Repair` koşuldu: Registry autostart ✅, eğitim watchdog ✅.
  **`HektorWeb` ve `HektorUpdate` kaydedilemedi — Yönetici PowerShell gerekiyor.**
- Yerel `main` `3a551b0`'de takılıydı ve `origin/main`'den sapmıştı (`update.ps1`
  ff-only yakınsar → güncelleme oturmaz). Commit dala cherry-pick edildi (`b0c0266`);
  **PR merge edilince** yerel main sıfırlanmalı:
  `git fetch origin && git reset --hard origin/main`.

---

## Son seans — 2026-09-10: 31 ajan envanteri + eğitim yolu artık kalite kapısından geçiyor

**Envanter (salt-okuma).** `reports/agent-inventory/envanter-2026-09-09.md` — depodaki üç
ajan popülasyonu ayrıştırıldı (31 runtime modül / 21 Claude alt-ajanı / 18 skill; hepsi
sayımla doğrulandı), determinizm sınıflaması yapıldı ve beş yapısal bulgu kanıta bağlandı.
Raporun tüm yük taşıyan iddiaları depoya karşı yeniden çalıştırıldı, hepsi tuttu (§6).

**Kapatılan bulgu — B2 (en kritik).** 31 ajanlık denetim mimarisi ile *fiilen eğitilen veri*
arasında zorunlu bağ yoktu: v7/v8 koşuları `scripts/start-train.ps1` ile başlatıldı ve
`pretrain-gate`/`lora-audit`'in hiçbirinden geçmedi. Artık betik, `lora-split`'ten **önce**
her iki kapıyı da çalıştırır:

- `pretrain-gate --json` → `verdict != GO` ise **eğitim başlamaz**.
- `lora-audit --json` → `passed=False` ise **eğitim başlamaz**.
- Kapı *çalıştırılamazsa* da başlamaz (Kural 2 — doğrulanmadan devam etme).
- Bilinçli insan override'ı: `-SkipGate` (görünür uyarı basar).

**Yeni CLI sözleşmesi.** `lora-audit` artık `--json` kabul ediyor (`pretrain-gate`'in mevcut
deseniyle aynı); rich tablosu ayrıştırılabilir arayüz değildi. Alanlar
`tests/test_lora_audit_json_cli.py` ile sabitlendi — alan adı değişirse kapı sessizce
"okunamadı"ya düşüp eğitimi hiç başlatmaz.

**Test sırasında bulunan ek açık.** Boş dataset kapıyı boşuna geçiyordu: `passed` tüm
kapıların AND'i olduğundan sıfır kartla `True` dönüyor. Betik artık `total_input` ya da
`total_approved` sıfırsa da engelliyor.

**Dikkat — nöbetçi etkileşimi.** `scripts/training-watchdog.ps1` çöken eğitimi diriltmek
için `start-train.ps1`'i çağırır, yani kapı diriltme yolunda da geçerlidir (bilinçli:
fail-closed). Geçici bir kapı arızası uzun bir koşuyu kurtarılamaz hale getirirse
`-SkipGate` ile elle diriltilir.

**Manifest drift (B1) kısmen kapandı.** `dataset-quality-gate` girdisi "CLI komutu KAYIP"
diyordu; gerçek trigger ve `ZORLANIR:` satırları yazıldı (`lora-control-plane` için de).
Genel drift testi hâlâ açık — bkz. açık işler.

---

## Son seans — 2026-09-08 (gece): v8 koşusu DURDURULDU — 600 adım ≠ 600 örnek

**Belirti.** `hektor_lora_v8_4b` koşusu "1616 örneğin yalnız 300'ü kullanıldığı için
dejenerasyon oldu → 600 örnekle eğit" gerekçesiyle başlatılmıştı. Trainer log'u aksini
söylüyordu: `max_examples=300 → 300/1616 örnek (determinist, seed=42)`. Yani koşu
**300 örnek × 2 epoch**'tu — v7 ile AYNI alt-küme (seed=42, aynı havuz), yalnız iki kat
uzun. Az-veri hipotezi test edilmiyor, ezber riski artıyordu. Koşu 13/600 adımda (2 saat)
durduruldu; hiç checkpoint yazılmamıştı (`save_steps=25`), kaybedilen tek şey 2 saat CPU.

**Kök sebep — kod, insan hatası değil.** `scripts/start-train.ps1`'de **`-MaxExamples`
parametresi yoktu**: betik `train --run`'a `--iterations` + `--profile` geçiyor, örnek
tavanı olarak profildeki `discipline_safe_local: max_examples: 300` yürürlükte kalıyordu.
Bu betikten başlatılan hiçbir koşu 300 örneğin üstüne çıkamazdı; `-Iterations` büyütmek
yalnız **epoch** büyütüyordu — `plan_iterations` docstring'inin "profilin `epochs: 1` vaadi
SESSİZCE ihlal edilir" diye adlandırdığı v5-sınıfı hata, bu kez başlatıcı tarafında.
`train_status.json` da yalnız (adapter, dtype, iterations, base_model) taşıdığı için
nöbetçinin kurtarma koşusu profili/tavanı UNUTUYORDU.

**Düzeltme (3 nokta, kapı yeşil).**

| Dosya | Ne değişti |
|---|---|
| `scripts/start-train.ps1` | `-MaxExamples` eklendi (→ `--max-examples`); `-Iterations 0` artık **profil planından** hesaplar (kanonik `plan_iterations`'a sorar, PowerShell'de tekrar etmez); adım plandan büyükse görünür **çok-epoch uyarısı**; durum dosyasına reçetenin tamamı yazılır |
| `scripts/training-watchdog.ps1` | profil + `max_examples` reçeteden geri okunur (profil artık sabit yazılı değil) |
| `app/training/detached_launch.py` | `_status_payload()` — web/auto_pipeline yolu da `profile` + `max_examples` + `base_model` yazar |
| `tests/test_train_recipe_persistence.py` | reçetenin bütün taşınmasını kilitleyen regresyon testleri |

**Sıradaki koşu (insan onayı BEKLİYOR — Kural 8).** Düzeltme `main`'e alındıktan sonra:

```powershell
.\scripts\start-train.ps1 -Adapter hektor_lora_v8_4b -MaxExamples 600 -Profile discipline_safe_local
```

Onay hakkında: bu betik `HEKTOR_TRAIN_SUPERVISED=1` yazar, yani `train --run`'ın CLI onay
kapısı ATLANIR (`app/main.py:460`) — komutu çalıştıran insan onayın kendisidir; tüketilecek
bekleyen istek yoktur (`approvals-list --status pending` boş). Bir ajan/otomasyon bu betiği
insan talimatı olmadan çalıştırırsa Kural 8 fiilen devre dışı kalır.

`-Iterations` VERME: plan `600 örnek × 1 epoch = 600 adım` üretir (doğrulandı). Ölçülen hız
~200 sn/adım → **~33 saat**. Başlatmadan önce boş RAM > 15 GB ve `ollama stop qwen3:4b`.

---

## Son seans — 2026-09-08: Kapıyı kıran `warmup_ratio` testi + araç sürüm hizalaması

**Belirti.** Bu makinede kapı KIRMIZI, tek düşen test:
`tests/test_peft_lora_recipe.py::test_build_training_kwargs_defaults` → `KeyError: 'warmup_ratio'`.

**Kök sebep — kod değil, testin beklentisi.** Kurulu transformers **5.16.1**,
`TrainingArguments.__init__` imzasından `warmup_ratio`yu KALDIRMIŞ. `f1c0bae`'deki uyumluluk
katmanı bunu zaten doğru ele alıyor (parametre desteklenmiyorsa kwargs'tan çıkar; `max_steps`
biliniyorsa `warmup_steps`e çevirir, bilinmiyorsa uyarır). Test ise hâlâ sabit
`kw["warmup_ratio"] == 0.03` bekliyordu → **shim'in ta kendisi testi düşürüyordu.**

**Düzeltme (`1d02ddd`) — uyumluluk davranışına DOKUNULMADI** (bilinçli; bkz. f1c0bae).
Yalnız test iki rejimi de kapsar oldu: destekleniyorsa oran 0.03 aynen geçer; desteklenmiyorsa
anahtar HİÇ geçmez ve varsayılan oran adıma çevrilir (0.03 × 200 = 6) — üstelik dönüşüm log'a
düşer, yani ısınma sessizce kaybolmuyor (Kural 2). Katmanın birim testleri zaten
`tests/test_warmup_ratio_compat.py`'dedir; kopya test yazılmadı.

**Araç zinciri hijyeni (`7ef46f4`, `3d9e60c`).** pre-commit ruff `v0.8.4`'te sabitliyken kurulu
ruff `0.15.15`'ti; ikisi iki test dosyasını TERS biçimlendiriyordu (`make format` bir hâli
yazıyor, `git commit` geri çeviriyordu). rev hizalandı, hook id `ruff` → `ruff-check` (yeni
sürümde `ruff` yalnızca legacy alias). Ardından `pre-commit run --all-files`'ın biriktirdiği
boşluk/satır-sonu düzeltmeleri 17 dosyada tek seferde kapatıldı; artık hiçbir hook dosya
değiştirmiyor. Dikkat gereken tek yer `TRAINING_ROADMAP.md` idi: 5 satır, sondaki iki boşluğu
markdown hard-line-break olarak kullanıyordu — körlemesine silinseydi paragraflar sessizce
birleşecekti; yerine görünür ve hook-güvenli CommonMark satır sonu kullanıldı.

**Kapı (bu makine, ana kopya):** ruff check temiz · mypy 217 dosya temiz · pytest **2041 passed,
2 skipped** (12dk42sn). Aynı ağaç, transformers'ın KURULU OLMADIĞI bir venv'de 2039 passed /
4 skipped verir — fark yalnız `importorskip("transformers")` ile korunan iki testtir. Yani
düzeltme hem "destekli" hem "desteksiz" rejimde ölçüldü; sayı farkı rejim farkıdır.

**AÇIK İŞ:** transformers 5.16.1 ile GERÇEK eğitim koşulmadı; doğrulanan yalnız config→kwargs
köprüsü. Sürüm sıçraması bağımlılığın üst sınırsız olmasından (`transformers>=4.40`) geldi —
eğitime dönmeden önce 1.5B ile kısa bir duman koşusu şart (Kural 2).

---

## Yerel eğitim denemesi — 2026-09-07: onay yarışı + bellek darboğazı

**Yerel eğitim yolu ÇALIŞIR durumda ve hiçbir eksiği yok.** Ölçüldü: `train` dry-run
`missing_packages: []`; torch 2.12+cpu / peft 0.19.1 / transformers 5.9.0 kurulu
(**2026-09-08: transformers 5.16.1'e sıçradı** — üstteki seansa bak);
`Qwen3-4B-Instruct-2507` (7.6 GB) ve `Qwen2.5-1.5B-Instruct` (2.9 GB, ChatML şablonlu)
HF önbelleğinde indirilmiş; veri bölünmüş (train=1447, valid=76). Disk 158 GB boş.

**Yerel eğitim ABONELİK KULLANMAZ.** Abonelik ajanların kod/araştırma işi içindir;
LoRA eğitimi yalnız yerel CPU'da Python hesabıdır — API çağrısı ve ücret yoktur.

**Bellek gerçeği (bu makine):** 32 GB toplam. 4B model bf16'da ~8 GB ağırlık + aktivasyon
ile pratikte **~18-20 GB**'a çıkıyor (ölçüldü) ve Ollama'nın llama-server'ı ayrıca 4-7 GB
tutuyor → boş RAM ~0.8-1 GB'a iniyor. Yerel koşu için **Qwen2.5-1.5B-Instruct** seçilmeli
(profil notu da ≤1.5B diyor). `scripts/start-train.ps1 -BaseModel ...` bunun içindir ve
seçim `train_status.json`'a yazılıp watchdog tarafından geri okunur (`cf7e893`).

**AÇIK BULGU — onay yarışı (Kural 8 hijyeni).** İki eşzamanlı `train --run` çağrısında
`consume_fresh_approval` CAS'i doğru çalışıyor (yalnız biri tüketir) ama **kaybeden taraf
YENİ bir pending onay üretip kuyrukta bırakıyor**. Gözlendi: kullanıcının verdiği onay
(`apr_7f612c303431`) başka bir oturumun `hektor_smoke_olcum` koşusu tarafından tüketildi;
bizim koşumuz yetki alamayıp `apr_042388be85de`'yi üretti ve durdu. Kuyrukta bu yüzden
kullanılmayan pending istekler birikiyor (şu an birkaç adet).
Öneri: yetki alamayan çağrı, aynı agent+action için **zaten bekleyen** bir istek varsa
yenisini üretmesin (idempotent istek); ya da kuyruk temizliği için `approval-prune`.

**Sıradaki adım:** `hektor_smoke_olcum` koşusu bitip RAM boşalınca, bekleyen onay
onaylanıp gerçek koşu başlatılacak:
`hektor train --run --backend peft --adapter-name hektor_lora_v6_local --iterations 300
--profile discipline_safe_local` (HEKTOR_PEFT_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct).
Reçete doğrulandı: 300 örnek = tam 1 epoch, r=16, lr=1e-4, NEFTune 5, assistant_only_loss
(v5 ezber-regresyonunun panzehiri), seed 42.

---

## Kademe-2 derin av — 2026-09-07: FAIL → 12 bulgu düzeltildi (`ea03b04`)

Eğitim öncesi **zorunlu** Kademe-2 avı çalıştırıldı (8 alt-sistem paralel bulucu →
HIGH/BLOCKER için 3 bağımsız şüpheci oy). **Verdict FAIL:** 45 ham bulgudan 32'si
onaylandı, 10'u bloklayan. Hepsi kodda ayrı ayrı doğrulanıp düzeltildi, her biri kendi
regresyon testiyle kilitlendi. Kapı: pytest **1999 passed**.

| Ciddiyet | Dosya | Neydi |
|---|---|---|
| BLOCKER | `market_data_loader` | CSV zaman sırasına göre **sıralanmıyordu** → ters sıralı dosya ters yönde backtest, fiili look-ahead (Kural 4), OOS dilimi en eski veri |
| HIGH | `peft_lora_train` | Checkpoint'ten koşulsuz devam; eski adım ≥ hedef ise **0 adım eğitip `ok=True`** (Kural 2) |
| HIGH | `detached_launch` | "1 epoch" fiilen **~4.8 epoch** (profil 300'e kırpıyor); satır-düzeyi bölme **18 makaleyi** train+valid'e dağıtıyordu |
| HIGH | `evaluate_model` | Garanti-vaadi deseni 17 varyantın 13'ünü kaçırıyordu |
| HIGH | `delegates` | `approval` aşaması **tek-kullanımlık onayı tüketiyordu** + her resume'da yeni pending (4 birikmişti) |
| HIGH | `engines` | codex av motoru `hardened=True` iddiasına rağmen **çıplak argv** (av, sürüşten gevşek) |
| HIGH | `knowledge_card_builder` | Tip sapmasında **içerikli kart kaydedilmeden çöküyordu** (8 sapma ölçüldü) |
| HIGH | `rag_learning_loop` + `paper_reader` | Başarısız deneme "üretildi" sayılıyor → bütçe tükeniyor, diğer makaleler **açlığa** düşüyordu |
| HIGH | `rlm/lora_candidate` | §16 atıf kapısı **atıfsız** koşularda boş yere sağlanıyordu (Kural 7) |
| HIGH | `confidence_scorer` | Aynı kök: atıfsız cevap ağırlıklı ortalamada **bedava 0.30** puan alıyordu |
| MEDIUM | `rlm_controller` | Zorunlu trading uyarısının idempotens kontrolü serbest cümleye bakıyordu → uyarının kalan 3 satırı eklenmiyordu (Kural 1) |
| — | `start-train.ps1` + `training-watchdog.ps1` | Resume varsayılanı kapanınca **çökme-kurtarma kırıldı**; `-Resume` switch'i eklendi |

**Yanlış-pozitif disiplini:** garanti deseni genişletilirken gerçek veri setindeki 7 meşru
akademik "garanti" (konformal tahmin aralığı, drawdown olasılık sınırı, FDP sınırı) elle
doğrulandı — hepsi hâlâ GO alıyor. Aynı hafta Gate 7'de bir GitHub URL'i API anahtarı
sanılıp 161 kartlık veri setini kilitlemişti; o sınıf hata tekrarlanmadı.

**Not:** Bu avdan sonra düzeltmelerin kendisi ayrıca şüpheci denetimden geçirilmelidir
(düzeltme yeni hata üretmiş olabilir). Eğitimden önce av YENİDEN koşturulmalı.

---

## Son seans — 2026-09-07: Eğitim hattı kapıları + Gate 7 yanlış pozitifi (kapandı)

**Gate 7 (BLOCKER) meşru veri setini kilitliyordu.** Gece üretimiyle onaylı kart 14 → 161
olunca `lora-audit` Gate 7 (safety) BAŞARISIZ verdi. Sebep sır DEĞİL, dedektör hatasıydı:
`_API_KEY_CANDIDATE` aday regex'i `/` ve `-` içerdiğinden bir GitHub bağlantısının host+yolu
TEK token olarak eşleşti (`com/AThreeH1/Global-Permutation-Entropy`: 3 karakter sınıfı,
4.41 entropi > 3.5 eşiği). Kartta (`card_ddb93d79f6b9`, paper_044fec06f4ff) hiçbir kimlik
bilgisi yok — kart okundu ve doğrulandı.

**Düzeltme (`2b0c263`)** veriye değil dedektöre: genel entropi sezgisi artık URL'in
şema+host+YOL bölümünde uygulanmaz (`_url_path_spans`). Yanlış-negatif korunur:
bilinen sır ön-ekleri (`ghp_`/`AKIA`/`sk-`/`xox…`) TÜM metinde — URL yolu dahil — önce aranır,
URL'in query/fragment bölümü maskelenmez (`?api_key=<sır>` hâlâ yakalanır). 3 regresyon testi;
fikstür dizgeleri parça parça kurulur (gitleaks pre-commit kancası bir kez tetiklendi —
kanca ATLANMADI, fikstür düzeltildi).

**Eğitim hattı durumu (2026-09-07 08:00):**

| Kapı | Durum |
|---|---|
| Stage 1 eşiği | 867/1000 (synth 708 + kart 159) — `synth-qa-bulk` canlı üretiyor |
| `pretrain-gate` | **GO** (blocker 0; uyarı: 57 maliyet-token'sız cevap, disiplin 289/528) |
| `lora-audit` Gate 0-7 | **GEÇTİ** (161/161 onaylandı, 20 inceleme işaretli) |
| `lora-curate` | 159 kanonik kart (orphan 0, çok-versiyon 0) |
| `lora-split` | train/valid ayrımı hazır |
| Kademe-2 derin av | **ÇALIŞIYOR** — bitmeden `hunt_ack` YOK |
| Kural 8 taze insan onayı | **BEKLİYOR** — ajan tüketmez |

**Kural 8 sınırı korundu:** gerçek eğitim başlatılmadı, `approval-approve` / `train --run` /
`/api/training/run` ÇAĞRILMADI. Orkestrasyon `deep-hunt` kapısında bloke
(`orc_f8fd6d720df34c24`, `orc_0de294252b6341e2`).

---

## Son seans — 2026-09-06: Boş bilgi kartı arayüz hatası (kapandı)

**Belirti.** Kütüphane'de bazı makaleler "✓ KARTI GÖR" gösteriyor, kart "(başlıksız)" açılıyor
ve "BİLGİ KARTI ÜRET" düğmesi kaybolduğu için makale bir daha kartlanamıyordu.

**Kök sebep.** Eski builder'ın (6dd6214 öncesi) yazdığı 7 **boş `pending`** kart
`has_knowledge_card` tarafından "kart var" sayılıyordu; `get_latest_knowledge_card` de en yeni
kartı içerik bakmadan döndürüyordu. Arayüz kodunda hata yoktu.

**Düzeltme (`75652df`).**
- `app/memory/sqlite_store.py`: iki erişimci de reddedilmiş VE içeriksiz kartı atlar; en yeni
  kart boş olsa bile daha eski içerikli canlı kart döner. Tek tanım: `card_has_content`.
- `app/web/static/assets/app.js` `renderCard`: içeriksiz kart açıkça "Bilgi kartı içeriksiz"
  + "↻ YENİDEN ÜRET" düğmesi (artık "(başlıksız)" yok).
- `tests/test_has_knowledge_card_rejected.py`: boş pending kart sayılmaz; içerikli kart tercih edilir.

**Canlı doğrulama.** Sunucu main'den yeniden başlatıldı (ayrık `uv run hektor-web`, çıktı
`logs/hektor-web.log` / `logs/hektor-web.err.log`). Boş kartlı makalede `GET /api/card/<id>` 404;
yedi makale yeniden "BİLGİ KARTI ÜRET" gösteriyor. Aynı gün başka oturumun `hektor read-all`
koşuları 5 yeni içerikli kart üretti → kartlı makale 10/159.

**Temizlik.** Aynı gün 7 boş kart `rejected` yapıldı (bu seans dışından). Ardından kullanıcı kararıyla
veritabanındaki **28 `rejected` satırın tamamı silindi** (hepsi içeriksizdi; `knowledge_cards`'a FK veren
tablo yok). Silme öncesi tutarlı yedek: `storage/sqlite/backups/hektor_trader_ai.pre-rejected-delete-20260906-225927.db`
(42 kart). Kalan: 11 approved + 3 pending, hepsi içerikli. Aynı kararla `reports/papers/` altındaki
eski builder kalıntısı **31 içeriksiz `*_card.json`** de silindi (önce zip arşivi:
`storage/sqlite/backups/reports_papers_empty_cards-<zaman>.zip`); 17 içerikli dosya kaldı.

---

## Bilinen açık işler

- **v8 sonrası (2026-09-11):**
  1. ~~**Dedektör boşluğu**~~ **KAPANDI (2026-09-14, PR #4).** `_is_degenerate`'e "aynı
     cümle ≥3 kez birebir" sinyali eklendi. Mutasyon testi v8 eval #4'ün (HANDOFF'taki
     "#5", sıfır tabanlı 4) birebir metniyle yazıldı: sinyal kaldırılırsa kırmızı. Kalibrasyon
     (v7+v8 eval, 64 gerçek cevap): yalnız bu vakayı ekledi, 32 base cevabından hiçbirini
     bayraklamadı. Sonuç: v8'in dedektörle ölçülen tekrar oranı 2/16 → **3/16** (HANDOFF'un
     elle saydığı ~%19 artık otomatik ölçülüyor). `grounding_verifier` mutasyon testi
     (envanter §5 sıra 3) ayrı iş olarak açık.
  2. **Diğer iki eval seti koşulmadı** — `overfit_awareness` + `risk_management`
     (24 soru, CPU'da ~6 saat). v8'in disiplin kazanımı orada da duruyor mu, tekrar
     patolojisi oranı ne?
  3. **Tekrar patolojisinin kökü — SIRADA (bir sonraki eğitimden ÖNCE; Kademe 2 kapsamında).**
     Reçete mi (NEFTune/lr/epoch) yoksa veri mi? **Veri tarafı ölçüldü (2026-09-14):**
     v8'in `train.jsonl`'ında (1616) cevap İÇİNDE tekrar eden cümle **0**; ama cevaplar
     ARASINDA birebir kalıp cümleler çok sık: "'pass' çıksa bile bu bir ADAY'dır." 92 cevap
     (%5,7) · "Doğru test noktası: pozisyonu shift(1) ile gecikmeli uygula…" 92 · "Ölçülmesi
     gereken bir hipotez var: shift(1)…" 58 · "Sonuç 'pass' değilse aday değildir." 58.
     v8'in eval'de döngüye soktuğu ifade tam bu 58'lik kalıp → **hipotez** (kanıt değil):
     yüksek frekanslı şablon cümle ezberi. Yapılacaklar: (a) `discipline_dataset`
     cevap şablonlarını çeşitlendir / aynı cümlenin cevaplar arası frekansına tavan koy,
     (b) `pretrain-gate`'e "cevaplar arası birebir cümle frekansı" kuralı (açılış-ezberi
     kuralı bunu görmüyor), (c) reçete tarafını ayırmak için aynı veriyle NEFTune kapalı
     kontrol koşusu. Doğrulama ancak yeniden eğitim + `lora-eval` ile (Kural 8, insan onayı).
     Kullanıcıya dönük geçici koruma PR #4'te: web sohbeti dejenere çıktıyı aynı dedektörle
     bayraklar, tekrarları gösterimde keser, ham çıktıyı korur — üretim ayarıyla gizlemez.
     **2026-09-15 ilerleme:** (a) **YAPILDI** — ortak kuyruklar kaldırıldı, 33 → 66 iskelet,
     iskelet başına 16 → ≤9 kopya. (b) **YAPILDI** — birebir cümle yerine daha sağlam
     **8-gram doküman frekansı** kuralı (NO-GO >%2, uyarı >%1); v8 verisi NO-GO, yeni karışım
     GO. Kesilen-örnek hipotezi ölçülerek elendi. (c) **AÇIK** — önce yalnız veri değişmiş
     yeniden eğitim; tekrar sürerse NEFTune kapalı kontrol. Ayrıntı: 2026-09-15 kaydı.
  4. **Yönetici `-Repair`** — `HektorWeb` + `HektorUpdate` görevleri hâlâ eski yolda.
- **2026-09-13 sonrası:**
  1. ~~**Uçtan uca doğrulama**~~ — **kapandı (2026-09-14).** 03:00 gece koşusu `SONUC: OK`,
     tokenizers damgası değişmedi, `import transformers` çalışıyor, web yeni kodla yeniden
     başladı. Kalan tek şey: senkronun üç küçük farkı kod değiştiren ilk gece koşusunda
     uygulanacak (bkz. 2026-09-14 kaydı).
  2. ~~**`needs-approval` etiketi repoda yok**~~ — **kapandı (2026-09-13).** Etiket
     oluşturuldu; `dependency-approval-label` işi artık fail-closed: etiket eklenemez ya
     da PR'da doğrulanamazsa KIRMIZI döner, fark hesaplanamazsa sessizce geçmez. İlk canlı
     sınama bu düzeltmenin kendi PR'ıydı (iş akışı dosyasını değiştirdiği için etiketlenmesi
     gerekir).
  3. **CI yeşil olmadan merge — dal koruması DIŞARIDAN kaldırılıyor (AÇIK).** main'e
     2026-09-14'te iki kez dal koruması açıldı: zorunlu `lint · types · tests (offline)`
     (yalnız `github-actions`), `strict`, `enforce_admins`, force-push ve silme kapalı. İkisi de
     API'den bayt düzeyinde geri okunarak doğrulandı; ikincisinde kontrol PR #8'in zorunlu
     kontrol listesinde de göründü. **İkisinde de sonradan kaldırıldı** (2026-09-15
     kontrolünde `protected=false`). Bu makinedeki Claude oturumu transkriptlerinde kaldırma
     izi yok; arada main'e giren tek değişiklik PR #7 merge'ü ve o merge koruma açıkken de
     `CLEAN` olurdu. Kimin kaldırdığı bilinmiyor → github.com/settings/security-log içinde
     `protected_branch.destroy`. Üçüncü kez açmadan önce kaldıran bulunmalı (yoksa çekişilir).
     Durum kontrolü: `gh api repos/alimirbagirzade/hektor/branches/main --jq .protected`.
- **2026-09-13 akşam (LLM düşünme):**
  1. **Yeni baseline** — varsayılan model `qwen3:4b-instruct-2507-q4_K_M` oldu (2026-09-13,
     KAPANDI: çekildi + varsayılan). `understanding_record` kıyası aynı `llm_model` şartı
     arar → yeni modelde anlama/sınav baseline'ı yeniden ölçülmeli; eski `qwen3:4b`
     skorlarıyla kıyaslanmaz. Diğer makinelerde `.env` + `ollama pull` elle yapılmalı.
  2. **Baseline karşılaştırılabilirliği** — bu düzeltmeden ÖNCE qwen3:4b ile alınan serbest
     metin ölçümleri (sınav/eval cevapları) düşünme sızıntısı + kesik cevap içerebilir;
     aynı `llm_model` adına rağmen yeni ölçümlerle birebir kıyaslanmamalı.
- **Ajan envanteri §5, sıra 2-6** (`reports/agent-inventory/envanter-2026-09-09.md`).
  Sıra 1 (eğitim kapısı) 2026-09-10'da kapandı. Kalanlar:
  2. Manifest `safety_gates` ↔ test kimliği eşlemesi + drift testi (B1'in genel hâli).
  3. `grounding_verifier` mutasyon testi — detektörün yakalama/yanlış-alarm oranı hiç
     ölçülmedi (karar kuralı 3/5 token örtüşmesi; olumsuzlamaya ve sayıya kör).
  4. Kart + SFT hattını `grounding_verifier`'dan geçir — doğrulayıcı şu an **eğitilen
     veriye bağlı değil** (B5-1); 3'e bağımlı.
  5. K3 normlarının (rubrik/eşik) tek dosyada toplanması + referans/ratifikasyon — insan
     kararı gerekir.
  6. Eval setini büyüt (`discipline_core.jsonl` = 16 soru); verdict'i güven aralığıyla yaz.
- `docs/MIGRASYON_2.0.md` §"Kalan adaylar" — Phase-4 GitHub otomasyonu (hiç aktive edilmedi),
  `training/dataset_builder.py` ikinci veri hattı, bulut-GPU protokol dokümanları.
- `docs/MIMARI_REFERANS.md` v1 temizliğinden ÖNCE yazıldı; kaldırılan modülleri hâlâ anlatır
  (dosya başında uyarı vardır).

## Önemli dosyalar

| Dosya | Ne |
|---|---|
| `CLAUDE.md` | Çalışma kuralları (bağlayıcı) |
| `docs/MIGRASYON_2.0.md` | v1 → 2.0 farkları |
| `docs/MIMARI_REFERANS.md` | Alt sistem alt sistem mimari referansı (v1 dönemi) |
| `automation_manifest.yaml` | Runtime ajanlarının tek bildirimsel kaynağı + zincir |
| `configs/lora/lora_profiles.yaml` | LoRA eğitim profilleri (`discipline_safe_local` varsayılan) |
| `docs/SCOPE_ISOLATION.md` | Sürücü motor ≠ insan yetkisi |
| `SECURITY.md` | Tehdit modeli + ağa açma checklist'i |
