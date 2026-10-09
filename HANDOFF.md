# HANDOFF — Hektor

_Depo: https://github.com/alimirbagirzade/hektor · Son güncelleme: 2026-10-09 (v14 eğitimi 2026-10-03'te 187/187 BİTTİ, aday — karşılaştırmalı değerlendirme bekliyor · araştırma otomasyonunu bekleten bayat kilit giderildi · #34–#45 main'de · aktif model hâlâ `hektor-v12-30b` · 4B dönemi kayıtları `docs/arsiv/4b_donemi/HANDOFF_2026-09_4B_donemi.md`'de)_


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

### 2026-10-09 — araştırma otomasyonunu bekleten kilit giderildi · durum kaydı düzeltildi

- **Kilit teşhisi:** araştırma paketi `storage/.training_launching` dosyasını engel sayıyordu.
  Ana kurulumdaki dosya **2026-09-29 08:10** tarihli (v14 başlamadan önce); hiçbir canlı sürece
  ait değil. Eğitim başlatıcı bu dosyadan ortak `storage/heavy_job.lock`'a (`resource_lock`)
  geçmişti; eski dosyayı artık kimse yazmıyor/silmiyordu → v14 bitse de araştırma süresiz
  bekledi. **Düzeltme:** `research_package.blockers` ortak ağır iş kilidine bakar (ölü sahip /
  süresi geçmiş başlatma bayat sayılır); web ipucu metni güncellendi; test eklendi. Yeni kodla
  ana kurulumda salt-okuma kontrol: **engel yok**. Bayat dosya SİLİNMEDİ (artık etkisiz).
  Etkinleşmesi için ana kurulum güncellenip 8765 web süreci yeniden başlatılmalı.
- **Durum düzeltmesi:** #34 (Faz 1–2) ve #35 (Faz 3, varsayılan kapalı) **main'e birleşti**
  (2026-10-05/06); ardından #36–#45 (Kademe 2 düzeltmeleri, aday-işleri yarışları, Windows
  konsol penceresi kilitlenmesi) de birleşti. Aşağıdaki 2026-10-06 notundaki "açık/taslak"
  ifadeleri o günün durumudur.
- **Sıradaki (öncelik):** (1) güncelle + web'i yeniden başlat, araştırma → RAG işleme → aday
  veri döngüsünü bir kez uçtan uca gözle (her aşamanın kaydı ve durma nedeni); (2) v14 / v12 /
  temel modeli kilitli sorular + kör insan incelemesi + kaynak/hesap doğruluğu ile karşılaştır
  (`cmp_ab63a45247fe` kör inceleme hâlâ bekliyor); strateji iddialarında maliyetli OOS
  backtest; (3) ancak sonra kontrollü pilot (güncel kod+veri için Kademe 2, reçeteye bağlı
  insan onayı, kısa koşu); (4) günlük işletim: `doctor` bu makinede `HektorWeb` /
  `HektorUpdate` zamanlanmış görevlerini kayıtlı bulmadı — kurulum kullanıcı kararı.
- Eğitim başlatılmadı, `.env`/STOP/zamanlayıcı değiştirilmedi.

### 2026-10-06 — Faz 1–2 gerçek kullanıma geçiş (o gün PR #34 açık · Faz 3 taslak PR #35 — ikisi de sonra birleşti)

Kanıt: `docs/evidence/faz12_uretim_2026-10-06.md`, `docs/evidence/kademe2_2026-10-06.json`.

- **Ana kurulum** `C:\HP\hektor` a56b635 → 3ef2079 (`update.ps1`, yedek
  `storage/backups/pre-update-20261005T231219Z/`). Aktif model `hektor-v12-30b` (`acebe185…`)
  DEĞİŞMEDİ. Tarayıcıda gerçek modelle: iki tur + geçmiş, Düzelt/Öğrensin/hariç, havuz
  sayaçları, veri sürümü kapalı kalması, strateji ÇELİŞKİ + onaysız test yok, aday hattı ve kör
  inceleme ekranları doğrulandı (TEST konuşması hariç tutuldu).
- **cmp_ab63a45247fe:** yalnız entegrasyon testi işaretli; AI incelemesi ayrı dosyada; **insan
  kör incelemesi bekliyor**. broad_v1 (27 aile) boyutlu ölçüt cevaplar üretilmeden kilitlendi.
- **Kademe 2 (main 3ef2079):** 26 bulgu → 17 düzeltildi (#34), 1 çürütüldü, 8 kısmi = kullanıcı
  kararı. Pilotu engelleyen F2-1 (1024 token profili) + F1-1 (reçete kapsamlı kayıt).
- **Pilot BAŞLATILMADI:** reçete hazır (`moe30b_attn_long`, 64 örnek, 8 adım, ayrı adapter);
  #34 birleşmesi + 8 kısmi bulgu kararı + uygulamadaki insan onayı bekleniyor.
- **Faz 3:** #35 taslak — ikinci görüş varsayılan kapalı, sahte sağlayıcıyla test edildi;
  bulut çıktısıyla eğitim kapalı (şartlar izin vermiyor). Gerçek bulut çağrısı yapılmadı.

### 2026-10-05 (akşam) — Faz 1–2 açıkları kapatıldı (dal `claude/hektor-faz-1-2-completion-297c12`, PR #33 — main'e birleşmedi)

Kanıt: `docs/evidence/faz2_tamamlama_2026-10-05.md`, `docs/evidence/gercek_veri_backtest_denetimi_2026-10-05.md`.
Faz 3 (bulut) yalnız tasarım: `docs/TASARIM_FAZ3_BULUT.md` (talimat DEĞİŞTİRİLMEDİ).

- **Strateji çevirisi:** orijinal öneri / model taslağı / onaylı nihai strateji ayrı kayıt;
  çelişki ya da dayanaksız model değeri taslakta boş ("karar gerekli"); fark listesi her öğesi
  işaretlenmeden (çelişkide gerekçesiz) test yok; öğrenme adayı test edilen nihai stratejiye
  bağlı, orijinalden farklıysa iddia stratejiyi kimliğiyle anmalı. Gerçek modelle yeniden
  üretildi: model "fiyat + 2 ATR" long stop'unu yine girişin altına çevirdi → artık ÇELİŞKİ.
- **Gerçek veri:** BTCUSDT 1h spot (Binance kamu arşivi, teknik doğrulama piyasası) 17 543 bar;
  `data/market/raw/` + köken kaydı (ana kurulumda). 683 işlem bağımsız hesapla aynı.
- **Aday hattı (web + CLI):** Öğrenme Havuzu → Aday hattı; `candidate-prepare/-compare/-jobs/
  -job-stop`. v14 kayıtlarla çapraz doğrulandı; `hektor-v14-30b-aday` oluşturuldu (digest =
  `hektor-v14-30b`); `cmp_ab63a45247fe` entegrasyon koşusu üretildi → **kör inceleme sizi
  bekliyor** (9 açık uçlu cevap). Ana model değişmedi (`hektor-v12-30b`).
- **Kademe 2:** reçete kapsamlı kayıt gerçek veri/temel model/profil/karışımı kapsar; alt
  süreç reçeteye bağlı (çıkış 11), eğitici okuduğu baytları doğrular; gerçek git deposunda
  kapı açık testler. Bu görevde Kademe 2 kaydı ÜRETİLMEDİ, eğitim BAŞLATILMADI.
- **Bekleyen (insan / ağır iş):** kör inceleme + karar; kabul olursa ana model etkinleştirme;
  pilot eğitim reçetesi kanıt belgesinde (§5). 8765 web süreci yeni uçlar için birleştirme
  sonrası yeniden başlatılmalı.

### 2026-10-05 — Faz 2 (#31 ile main'e birleşti)

Commit'ler: Faz 1 tamamlama `c993b78` · 2A `306aa61` · 2B `061df56` · zaman/iddia `90453a6` ·
2C `ec8157e` · 2D `a9c1496`. Tam paket **3091 passed / 13 skipped**; ruff + mypy temiz.
Belgeler: `docs/PROTOKOL_SOHBETTEN_OGRENME.md`, `docs/PROTOKOL_STRATEJI_TESTI.md`,
`docs/PROTOKOL_ADAY_KARSILASTIRMA.md`.

- **Faz 1 tamamlama:** ortak ağır iş kilidi (`storage/heavy_job.lock`; web, start-train.ps1,
  CLI, dönüşüm, karşılaştırma); "Kaynak benzerliği" artık otomatik uygunluk vermez; arayüz içi
  düzenleyici. Gerçek 30B (hektor-v12-30b) ile iki tur + Kaynak ölç + tarayıcı akışları doğrulandı.
- **2A:** ana / deneme yuvası; mevcut model "başlangıç kaydı — değerlendirilmemiş"; ana model
  yalnız `kabul` + digest; `yetersiz_kanit` yalnız deneme sohbeti; iki aşamalı günlüklü kayıt.
- **2B:** sohbet cevabı → yerel model taslağı → okunur form → veri denetimi → geliştirme /
  doğrulama / final (tek) dönem testi; stop, boyut ve birimli maliyetler gerçekten uygulanır.
- **Zaman:** veri / strateji / koşu / bilgi zamanı ayrı; `as_of` = bilgi zamanı; strateji
  aileleri; performans iddiası parmak izi + dönem + sayılarla "kayıtlı hesapla eşleşti".
- **2C:** son ayarlar hazır; salt-okunur hazırlık; anlık görüntü + reçete; reçeteye bağlı onay;
  Kademe 2 kaydı (`hektor kademe2-kayit`); idempotent başlatma.
- **2D:** `candidate-verify` (run_complete tek başına yetmez); kilitli ölçüt; kör inceleme;
  aile düzeyinde bootstrap; gizli final erişim kaydı; karar → 2A deposu.

**Henüz doğrulanmadı (gerçek sistem):** gerçek piyasa CSV'siyle strateji testi (makinede OHLCV
yok); eğitim sürerken sohbet; gerçek `adapter_to_ollama.ps1` koşusunda kilit; gerçek veriyle
anlık görüntü → onay → başlatma; gerçek 30B aday/aktif/temel karşılaştırması ve gerçek
adapter'da `candidate-verify`; gerçek kararla ana model etkinleştirme / geri dönüş.
Kademe 2 kaydı TÜM eğitim yollarında zorunlu: web butonu / Auto-LoRA / kolay akış
(`preflight_launch`), `hektor train --run` (start-train.ps1 — `-SkipGate` ile de aşılamaz —,
doğrudan CLI, nöbetçi kurtarması; çıkış kodu 10) ve `pretrain-gate`. Kayıt bu kod özeti
(temiz ağaç) + güncel `lora_sft.jsonl` özeti için olmalı: `hektor kademe2-kayit --data-sha …`.

### 2026-10-05 — Sohbetten Öğrenme Faz 1 (#30 ile main'e birleşti)

Tek ana sohbet (**00 · SOHBET**, Ollama+RAG, geçmişli, model kimlikli) + **16 · ÖĞRENME
HAVUZU**. Yalnız **Öğrensin/Düzelt** aday üretir; Faydalı/Hatalı üretmez. Kapsamlı LLM'siz
kontroller (hesap / kaynak / atıf kimliği / Kural 1; backtest ve kod testi "yapılamadı"),
gerekçeli insan onayı ayrı sınıf. Aile düzeyinde kalıcı train/eval, köprü = sızıntı çatışması,
trading zaman sıralı. Değişmez sürümler `data/learning/chat_datasets/chat_vN/` (eval ayrı
dosya, eğitime girmez); kanonik birleştirme yalnız `assemble_sft.py --chat-dataset` seçimiyle,
satır+hedef token payı ≤ %10. Seçili sürümde sonradan geçersizleşen kayıt → pretrain-gate +
web/CLI başlatma kapısı NO-GO. Sunucu tarafı kaynak koruması (RAM/VRAM ayrı, ölçüme dayalı) +
sohbet kirası ↔ başlatma kilidi. Eğitim/etkinleştirme/bulut YOK (Faz 2/3). Protokol:
`docs/PROTOKOL_SOHBETTEN_OGRENME.md`. Yeni tablolar yalnız eklenir; geri alma
`scripts/chat_learning_rollback.py`. **8765'teki web süreci yeniden başlatılmadı**; yeni uçlar
için birleştirme sonrası güvenli bir zamanda yeniden başlatma gerekir. Canlı 30B tur, "Kaynak
ölç" ve eğitim sırasındaki davranış gerçek Ollama ile ölçülmedi.

### 2026-10-04 — araştırma motorları ve sürüş görünürlüğü

Araştırma ana motoru eğitim sürücüsünden ayrıldı: Codex, Claude Code, sürümü
kısıtlı Gemini CLI ve yerel Ollama plan incelemesi. Kurulu olmayan motorlar gerekçeli
kapalıdır; ikinci inceleyici kapalı metni artık bağlantı yok izlenimi vermez.
Autodrive HTTP kabulü ile süreç kaydı arasındaki yarış kapatıldı; aynı koşuya ikinci
istek 409 alır, sayfa yenilemesi sürüş göstergesini geri yükler. Bu bir kesintisizlik
veya otomatik eğitim yeniden başlatma garantisi değildir.
Öğrenme'den doğrudan model sohbetine bağlantı var; API v14/v13/v12/v11 adapter'larını
listeliyor. PEFT sohbeti Ollama gerektirmez; model değişiminde eski önbellek yeni
model yüklenmeden bırakılır. Aktif eğitim sırasında sohbet 409 ile bekletilir;
bu kontrol tüm süreçler arasında atomik kaynak kilidi değildir.
Canlı araştırma eğitim başlatma kilidini bekliyor; kilit/STOP/onay değiştirilmedi.
Doğrulama: tam koşuda 2907 geçti; eski limit=1 varsayımını kontrol eden tek test yeni
koşu seçimine uyarlandı ve ilgili 73 test geçti. Ruff (ilgisiz tmp hariç), format,
mypy 271 kaynak, JS sözdizimi ve MCP 22 araç üretimi geçti. Canlı web yeniden açıldı;
motor seçenekleri ve Öğrenme → V14 seçili sohbet geçişi tarayıcıda doğrulandı.
Gerçek CLI/Ollama araştırma incelemesi ve PEFT cevap üretimi bu doğrulamaya dahil değil.


### 2026-10-03 — araştırma paketinin Öğrenme sekmesine entegrasyonu

Öğrenme sekmesinde paket durumu, motorlar, sıradaki aşama, engeller ve son yönetici
kontrolü görünür; 15 saniyelik salt-okunur yenileme ve Ajan Haritası yönetim bağlantısı
eklendi. Doğrudan `#sekme=learning` ile açılışta eğitim grafikleri/sayaçların yüklenmemesi
düzeltildi. Eski RAG döngüsünün web kapanınca da süreceği yönündeki metin düzeltildi.
Yerel web 8765 açık, araştırma yöneticisi Codex ile etkin fakat eğitim başlatma kilidi
nedeniyle bekliyor. Kilit/STOP/.env değiştirilmedi; gerçek araştırma turu başlamadı.
v14 eğitim günlüğü 3 Ekim 20:35'te 187/187 adım ve adapter kaydını bildiriyor;
adayın karşılaştırmalı değerlendirmesi ayrı iştir.
Doğrulama: çevrimdışı tam paket 2900 geçti / 4 dışlandı; araştırma testleri ayrıca
38 geçti. Mypy (269 kaynak), JS sözdizimi ve tarayıcıda doğrudan bağlantı + yönetim
geçişi doğrulandı. Ruff kaynak kontrolleri geçti; depo-geneli komutun karşılaştığı
ilgisiz yerel `tmp/pdfs/build_v13.py` biçim/lint hataları bu değişikliğe dahil edilmedi.

### 2026-10-01 — tek tuş araştırma paketi (doğrulandı; eğitim sonrası canlı pilot bekliyor)

Kullanıcı kapsamı: araştırma + kontrollü RAG işleme + aday veri hazırlığı; yeni ağır
eğitim yalnız önerilir. `hektor research-package` salt-okunur plan verir; `--run`
yalnız vadesi gelen tek aşamayı çalıştırır. Reçete `configs/research_package.json`,
protokol `docs/PROTOKOL_ARASTIRMA_PAKETI.md`. Ayrı temel model işçisi; `.env`, mevcut
web döngüleri ve kanonik v14 eğitim verisi değiştirilmedi. Çıktı ayrı
`data/research_package/staging/` altında birikir. Eğitim/onay/terfi/git işlemi yok.

Web: **15 · AJAN HARİTASI → Motor seç → Araştırma döngüsünü çalıştır**. Kalıcı
web yöneticisi, seçilen Codex/Claude CLI'ına sınırlı planı inceletip yerel aşamaları
seri yürütür; ikinci inceleme motoru isteğe bağlıdır. CLI kararı kod olarak
çalıştırılmaz; stdout kararı stderr'deki istem/günlükten ayrılır. Eksik/yanlış karar
ve ret işi durdurur. Yeni LoRA eğitimi başlatılmaz. Eğitim sürerken CLI de doğmaz.
Araştırmayı durdur yalnız paketin süreçlerini kapatır. Web sunucusu açık kalmalıdır;
yeniden açılışta etkinlik kaydı korunur. Bu makinede Codex kurulu, Claude yok;
Gemini mevcut kısıtlama profili olmadığı için kapalıdır.

Codex sohbet heartbeat'i **Hektor eğitim sonrası araştırma**
(`hektor-e-itim-sonras-ara-t-rma`) saatlik kayıtlı. Eğitim sürerken yalnız plan/engelleri
okur; ilk gerçek işi tüm kayıtlı koşular tamamlandıktan, 15 dk beklemeden ve bellek
kapısından sonra yapar. Değişmeyen durumda sessizdir. Eski zamanlayıcıları ayrıca
açmayın. Web yönetimi etkinse heartbeat yalnız izler ve dış `--run` engellenir.
Paket durdurma: `hektor research-package --pause`; devam: `--resume`.

Son doğrulama: **2900 geçti, 4 dışlanan, 0 hata** (`not ollama and not slow`);
ruff format/lint, mypy (269 kaynak), `node --check` ve diff kontrolü geçti.
Önceki Windows CWD sahte CLI hatası mutlak PATH adaylarıyla düzeltildi; ilgili
regresyon testi artık geçiyor. MCP 22 araç üretiyor; yeni durum GET'i var,
başlat/durdur uçları MCP dışı ve insan scope'una bağlı.

Yalıtılmış boş veri kökü ve 8771 test sunucusunda tarayıcıyla Çalıştır → Bekliyor,
yenilemede durumun korunması ve Durdur → Kapalı doğrulandı; test sunucusu kapatıldı.
Gerçek dry-run hâlâ v14 + eğitim kilidi/süreçleri + RAM nedeniyle engelli.
Gerçek CLI/Ollama/keşif işi başlatılmadı; eğitim sonrası ilk sınırlı tur canlı pilottur.
**8765'teki mevcut web süreci yeniden başlatılmadı**: yeni Python uçları için güvenli
bir zamanda yeniden başlatma gerekir. `.env` ve v14 süreçleri değiştirilmedi.

İlk sürümün sınırları: mevcut canlı RAG indeksine kontrollü ekleme; ayrı aday indeks
terfisi yok. Kaynak alıntılı yöntem önerisi bağımsız derin adversarial doğrulama
değildir. Tüm Hektor süreçleri için ortak kaynak kilidi henüz yok; paket yeni eğitim
görünce kendi işçisini durdurur, Ollama sunucu isteğinin anında iptalini garanti etmez.
Bu konular protokolde kullanıcıyla tartışılacak geliştirmeler olarak listelendi.

| Alan | Durum |
|---|---|
| Kapı (`make ci`) | **CI (Linux) ✅** PR #27 (`claude/lora-30b-a3b-prep` → main) `9cf1034` "lint · types · tests (offline)" success. #26 main'e SQUASH ile birleşmişti → #27 "dirty" idi; squash ağacı `f92a22d` ile birebir aynı olduğu doğrulanıp `-s ours` ile birleştirildi (içerik kaybı yok). **Yerel (Windows):** ruff + mypy (264) + pytest **2854+ passed**. Not: arka planda `storage/`'a yazan süreç (synth-distill/web) varken tam test koşusunda izolasyon koruması nadiren ERROR verebilir — tek başına tekrar koş. |
| Varsayılan model | **LLM:** `qwen3:30b-a3b-instruct-2507-q4_K_M` · **PEFT base:** `Qwen/Qwen3-30B-A3B-Instruct-2507` · **profil:** `moe30b_attn_local` (`app.config.DEFAULT_TRAIN_PROFILE`; attention-only, maskeli, bf16 ~61 GB RAM). Düşük RAM: `qwen3:4b-instruct-2507-q4_K_M` + `Qwen/Qwen3-4B-Instruct-2507` + `discipline_safe_local`. Çıplak `qwen3:30b`/`qwen3:4b` = Thinking-2507 (yavaş). **Bu makinenin `.env`'i: `HEKTOR_LLM_MODEL=hektor-v12-30b`** (LoRA'lı; arka plan döngüleri bu yüzden KAPALI). |
| Eğitim yığını | `train-cpu` extra'sı kilitte **sabit**: torch 2.14.0 · transformers 5.16.1 · tokenizers 0.23.2 · peft 0.20.0 · accelerate 1.14.0. Yükseltmek açık karardır → ardından adapter yeniden değerlendirilmeli |
| Koşan eğitim | **`hektor_lora_v14_30b`** — başladı 2026-10-01 11:45 (`start-train.ps1 -Profile moe30b_attn_long`, onay `apr_9e27b2e343f3`, mix `trading_analysis_v1` / `wd_db97219915`). 1489 örnek → **187 optimizer adımı** (GA 8), eval+checkpoint her 12 adımda (32 valid), load_best, kayıp **token başına** (kullanıcı kararı). **Ölçüm:** 1. optimizer adımı 13 dk 10 sn → ~42-45 sa (bitiş ≈ 3 Ekim öğleden sonra; tek adımdan genişletme). **RAM SINIRDA:** süreç private 120 GB, boş RAM 12.7 GB, boş sanal (commit) 12.1 GB → en uzun örneklerle ayırma hatası riski; çökmede nöbetçi checkpoint'ten sürdürür (her 12 adım ≈ 2.6 sa), log artık arşivlenir. Önlem: Windows sayfa dosyasını büyütmek (kullanıcı). Eğitim sürerken Ollama/web/LoRA sohbeti AÇMA. |
| Son adapter | **`hektor_lora_v13_30b`** — TAMAMLANDI 2026-09-30 11:44 (210 optimizer adımı, GA 8, val_loss en iyi 0.5769 @175, son adapter = checkpoint-175). Kayıt `adapter_ee06d5a4dfa8` **candidate**, terfi YOK. Ollama: `hektor-v13-30b` (Q4_K_M + attn q8_0). Önceki: v12 (candidate; Ollama şablonunda CR — D1), v11 (candidate). |
| LLM | Yalnız yerel Ollama. Bulut API istemcisi YOK. |
| Gözetimsiz eğitim | **KAPALI** (`unattended_training_enabled=false`) → her gerçek eğitim tek-kullanımlık insan onayı ister (Kural 8) |
| Arka plan döngüleri | Web açılışında çalışır; `HEKTOR_BACKGROUND_LOOPS_ENABLED=false` ile kapatılır. **Bu makinede kapalı** (LLM ayarı LoRA'lı model — açmadan önce base'e döndür). |
| RAG | Hibrit (dense + BM25 + sezgisel rerank). **Sorgu çevirisi (TR→EN) CANLI** bu makinede (`.env`: `HEKTOR_RAG_QUERY_TRANSLATE=en`, çevirmen `qwen3:30b-a3b-instruct-2507-q4_K_M`; yedek `storage/env.bak-20260930-ragtranslate`) — kullanıcı kararı 2026-09-30; `.env.example`'da da açık, kod varsayılanı `off`. Çeviri hata verirse retrieval orijinal sorguyla sürer. ⚠ Sohbet modeli `hektor-v12-30b` + çevirmen ayrı 30B → önbellekte olmayan her sorguda GPU model takası olabilir (ÖLÇÜLMEDİ). Amaç filtresi kapalı. |
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

Ollama kapalıysa: `ollama serve` → `ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M` → `ollama pull nomic-embed-text`.

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

## Sıradaki adım

1. **v14 eğitimini izle** (`egitim-nobetcisi` ajanı / `.\scripts\start-train.ps1 -Status`). Bitince:
   `run_complete.json` (best checkpoint) → `lora-eval` (dejenere dedektörü artık iskelet-bilinçli)
   → `adapter_to_ollama.ps1 -Adapter hektor_lora_v14_30b -OllamaName hektor-v14-30b -AttnQuant q8_0`
   (köken anahtarı eşleşmezse durur) → LLM-30 2×2 **yeni run-id** ile, base `hektor-base-30b-q4a8`.
   v13 ile karşılaştır: ort. token, sayısal anahtar, alt madde, atıf (D), döngü. ADAY; terfi insan.
2. **Eğitim sürerken yapılacak eval işleri (Kademe 2 açık):** F4-5 2×2'ye canlı RAG istemli koşul
   (C′/D′ = `build_rag_prompt`; v14 bu biçimle eğitildi) · F4-7 v13 metriklerini üreten analiz
   betiği (repo'da yok; `abc_all`/`num_ok` yeniden üretilemiyor) · F4-8 n=30 soru başına raporla
   (temp 0'da seed43==44) · F4-9 merge KL kapısı tüm pozisyonlarda · F4-10 Modelfile parametre
   karşılaştırması · F4-11 dataset_version kanonik içerikten · F2-9 Ollama şablonu `SYS\n\n<|im_end|>`.
3. EMA doğrusallığı (S12) müfredatı **v14'e girmedi**: LLM-30 S12'yi doğrudan hedefler (geliştirme
   seti kirlenir); eklenecekse doğrulanmış kaynakla + S12 metrik dışı bırakılarak.
4. **2×2 rubrik puanlaması** (insan; `app.evals.llm30.RUBRIC`, kritik hatalar ayrı) — ham
   cevaplar `reports/evals/llm30/llm30_v13_2x2_20260930/raw.jsonl`.
5. RAG çevirisi canlı → sohbet modeli + çevirmen GPU takas gecikmesini ölç; insan etiket
   örneklemesi (`evals/rag_relevance/llm30_pooled_v1.jsonl`). 2×2'yi çevirili RAG ile tekrar koş.
6. **Canlı sohbet modeli** `hektor-v12-30b` (D1 şablon CR'si, v13'e benzer kısa/atıfsız davranış
   riski) → base `qwen3:30b-a3b-instruct-2507-q4_K_M`'e dönmek kullanıcı kararı.

Eğitim akışı (veri `data/`, `storage/`, `models/` git'te izlenmez):

```bash
uv run hektor ingest && uv run hektor synth-qa-bulk --target 1000
uv run python scripts/assemble_sft.py      # → data/lora_sft/lora_sft.jsonl (KANONİK)
uv run hektor lora-audit && uv run hektor pretrain-gate && uv run hektor lora-split
uv run hektor approval-approve <id>         # Kural 8
.\scripts\start-train.ps1                    # DETACHED; varsayılan moe30b_attn_local + bf16
```

`start-train.ps1` `pretrain-gate` + `lora-audit`'i zorunlu koşar (NO-GO → eğitim başlamaz;
bilinçli override `-SkipGate`). Eğitim sonrası: `lora-eval` → adapter **ADAY**; production
terfisi ayrı insan onayı ister. Servis yolu: `scripts/adapter_to_ollama.ps1` (merge → GGUF →
Ollama; 2×2 için aynı tarifle base: `-BaseRepo`).

---

## Son seans — 2026-10-01: v14 verisi (base öz-damıtma) + Kademe 2 + **v14 EĞİTİMİ KOŞUYOR**

**Teşhis (v13 verisi, ölçüm):** 1499 sentetik/disiplin satırında cevap medyanı ~250 kr, atıf 6;
canlı RAG istemi (`rag_answer.md` + `SOURCES / KAYNAKLAR … QUESTION / SORU:`) eğitimde HİÇ yoktu
→ LoRA "bağlam+soru → 2-4 cümle, atıfsız" öğrendi (2×2: ~5× kısa, D'de 0 atıf).

**v14 verisi (`hektor synth-distill`, yeni `app/training/self_distill.py`):** öğretmen = BASE
`qwen3:30b-a3b-instruct-2507-q4_K_M` (hektor-* reddedilir). Kipler: `rag` (canlı retrieval +
`build_rag_prompt` bayt-aynı istem) ve `plain` (genel istem + "bağlam verilmedi" notu). Sorular:
sentetik QA'nın bağımsız soruları + `--gen-questions` (865 parçadan 620 bağımsız soru, ⅓'ü a/b/c
alt maddeli). İki tur: r1 (`distill_qa.r1.jsonl`, kapılar sıkılaşmadan) + r2; birleştirme
`distill_qa*.jsonl`'i GÜNCEL kapılardan yeniden geçirir. Geçerli: **431 RAG + 61 RAG'sız**.
Birleştirme: sentetik QA ≤400 (zenginleştirilmiş önce), kalıplaşmış damıtma satırı inceltme
(%1.8; 10 satır). Kanonik set **1569** (train 1489 / valid 80): satır payı RAG-damıtma %27 ·
sentetik %25 · disiplin %25 · kart %19 · plain %4; cevap metni payı RAG-damıtma **%72**.
pretrain-gate GO · lora-audit 292/292 · sızıntı 0. Yeni profil `moe30b_attn_long`
(seq 6144, tüm havuz, eval/checkpoint her 96 örnek, `loss_weighting: token`).

**Kademe 2 (v14 öncesi; 4 bulucu, her bulgu 2 bağımsız şüpheci doğrulayıcı — F1'in 2. oyu kota
yüzünden yarıda kaldı, F1 iddiaları gerçek veride ölçülerek doğrulandı).** Onaylanıp düzeltilenler:
- Veri: RAG'sız damıtmada **87/90 uydurma kaynakça / "RAG bağlamı" iddiası** (Kural 7) → not +
  kapı · atıf kapısı tekrarları sayıyor, uydurma parça kimliğini ve çoklu kimlikli köşeliyi
  kabul ediyordu → strict çift + ≥2 farklı kaynak · Gate 5/7 damıtmada da · tekrar kapısı zorunlu
  iskelette yanlış pozitif · U+2028 satır bölme · aksansız/numaralı-kitap bağımlı sorular.
- Retrieval: ASCII Türkçe çevrilmiyordu (25/800) · çevirmen alt maddeli soruyu CEVAPLIYORDU ve
  cevap sorgu oluyordu · önbellek G/Ç hatası koşuyu düşürüyordu / önbelleği siliyordu · çevirmen
  ile öğretmen farklı num_ctx → **her RAG işinde 30B iki kez yeniden yükleniyordu** (toplu ön-çeviri).
- Eğitim: kayıp token ağırlıklı (karar: token kalsın) · profil sessizce 600 satır · checkpoint
  3-5 sa'te bir · start-train eski logu siliyordu · 45 dk canlılık eşiği · uzunluk kaybı sessizdi
  (artık >%1 ise eğitim başlamaz).
- Eval: `_is_degenerate` base'i %56-73 bayraklıyordu (v13 2×2 kalibrasyonu → A 6, C 6; adapter
  döngüleri korunur) · `adapter_to_ollama` eski birleşik/GGUF'u yalnız varlığa bakıp kullanıyordu
  (köken anahtarı) · llm30 sürdürme manifest denetimi · sızıntı: LLM-30 alt maddeleri + canlı
  `QUESTION / SORU:` biçimi · pretrain-gate şablon muafiyeti satır düzeyinde.
Commit'ler: 604f1f5, b7a683d, 24e20ac, 9cf1034 (main birleştirme), 1683792 + bu.

---

## Önceki seans — 2026-09-30 (5): v13 sonucu + GGUF + LLM-30 2×2 (v13 GERİLİYOR) + RAG deneyi + 4B temizliği

**v13 eğitimi TAMAMLANDI** (210/210 optimizer adımı, 30.09 11:44; kayıt `adapter_ee06d5a4dfa8`
**candidate**, terfi YOK). 07:43'te checkpoint-125'ten sürdürüldü (`recovery_attempts: 1`;
kesinti nedeni eski log üzerine yazıldığı için bilinmiyor). val_loss 0.733 → **0.5769**
(adım 175, en iyi) → 0.578; 125'ten sonra düz. Son adapter = checkpoint-175 (sha eşleşti,
`load_best` çalıştı). Train loss son 25 adım ort. 0.82 (dropout açık) — overfit işareti yok.

**GGUF (eşit tarif):** `hektor-v13-30b` (Q4_K_M + attn q8_0, sha `d8cf8964…`) ve aynı tarifle
**`hektor-base-30b-q4a8`** (`adapter_to_ollama.ps1 -BaseRepo`, sha `3e687b34…`) — 2×2'de base
ile LoRA'nın nicemlemesi/şablonu birebir aynı olsun diye. v13 birleştirme kapısı KL 0.0201
> 0.01 ile durdu (diğer ölçütler geçti); **kullanıcı kararıyla** bu koşu için `-MaxKL 0.025`
(`merge_info.json`: `kl_gate_override=true`). Varsayılan 0.01 değişmedi.

**LLM-30 2×2** (`app/evals/llm30_run.py`, run `llm30_v13_2x2_20260930`, 30 soru × A/B/C/D × 3
tekrar, temperature 0, num_predict 4096, num_ctx 16384): **TAMAMLANDI** (360/360). Protokol
harfleri: A base · B base+v13 · C base+RAG · D base+v13+RAG; base = `hektor-base-30b-q4a8`
(v13 ile AYNI GGUF tarifi + şablon). RAG bağlamı V0 (mevcut hat, çevirisiz) ile donduruldu.
Deterministik analiz (RUBRİK DEĞİL; `reports/evals/llm30/llm30_v13_2x2_20260930/`):

| | A base | B +v13 | C +RAG | D +v13+RAG |
|---|---|---|---|---|
| ort. çıktı token | 2261 | **495** | 2476 | **388** |
| sayısal anahtar (10 soru × 3, kaba regex) | **30/30** | 18/30 | 26/30 | 18/30 |
| a/b/c alt maddelerinin hepsi | 90/90 | 81/90 | 90/90 | **51/90** |
| atıflı cevap / uydurma atıf | — | — | 56/90 · 2/197 | **0/90** |
| kesilme (4096) / döngü | 1 / 0 | 4 / 3 (S13, S28×3) | 0 / 0 | 2 / 4 (S08, S28) |

**Sonuç (Kural 2; puanlama değil ölçüm):** v13 base'e göre **geriliyor** — cevaplar ~5× kısa,
sayısal anahtar kapsaması 30→18, RAG'da atıf talimatını tamamen yok sayıyor (D 0 atıf; v10'daki
bulgunun aynısı), alt maddeleri atlıyor, S28/S08'de döngüye giriyor. v13 **terfi edilmemeli**;
candidate kalır. Kök neden hipotezi (HANDOFF 09-30 (3) Öneri 3 ile tutarlı): eğitim verisi kısa,
atıfsız cevaplardan oluşuyor → LoRA uzunluğu/formatı bastırıyor.
**Kritik ortak hata:** S12'de 4 koşulun 12/12 cevabı "sabit alpha EMA doğrusal DEĞİLDİR" diyor
(yanlış — doğrusal zamanla-değişmez IIR filtre). Base modelin kavram hatası; LoRA/RAG düzeltmiyor →
eğitim verisine müfredat maddesi adayı (doğrulanmış kaynakla).
İlk koşu 108'de Ollama "token repeat limit" iptalinde durdu → koşucu artık iptali kısmi ham
cevapla `iptal:tekrar_limiti` kaydediyor (sürdürmede C S07 iptal etmedi). Bulgular:
- **Ollama bu kurulumda deterministik DEĞİL:** aynı model+seed+temp 0'da cevaplar 149. karakterde
  (RAG'lı ilk token'da) ayrışıyor. 2×2'de ölçüldü: tekrar 2 = tekrar 3 **30/30** her koşulda
  (önek önbelleği), tekrar 1 ≠ tekrar 2 **~30/30** → soru başına fiilen 2 bağımsız örnek var.
- `uzun_tekrar` bayrağı ilk sürümde YANLIŞ POZİTİFTİ (istenen alt-madde başlıkları 3× geçiyor);
  düzeltildi, `summary.json` bayrakları ham cevaptan yeniden hesaplar.
- Canlı `hektor-v12-30b` şablonu base'den FARKLI (D1: CR'ler; yeniden oluşturulmadı) →
  koşucu v12 ile başlamayı reddediyor.

**RAG deneyi (sorgu çevirisi × amaç filtresi)** — kök neden ölçüldü: Türkçe sorgu, ~%89
İngilizce korpusta tek Türkçe metin olan iki proje kılavuzuna çekiliyor (BM25 top-24'ün %82'si).
Yeni: `configs/rag/doc_purposes.yaml` + `HEKTOR_RAG_EXCLUDE_PURPOSES`, `HEKTOR_RAG_QUERY_TRANSLATE`
(off|en|bilingual; çeviri modeli ayrı, `hektor-*` reddedilir) — **ikisi de varsayılan KAPALI**.
Havuzlu kör etiket (407 çift, `evals/rag_relevance/llm30_pooled_v1.jsonl`, Claude etiketledi,
`human_verified=false`), `reports/evals/llm30/rag_exp_20260930T193022/score.json`:

| | P@6 | nDCG@6 | havuz-recall@6 | ilgilisiz soru | iç-doküman |
|---|---|---|---|---|---|
| V0 mevcut | 0.139 | 0.102 | 0.111 | 23/30 | 53/180 |
| V1 filtre | 0.167 | 0.115 | 0.138 | 21/30 | 0 |
| **V2 çeviri (en)** | **0.711** | **0.771** | **0.849** | **1/30** | 0 |
| V3 çeviri+filtre | 0.711 | 0.771 | 0.849 | 1/30 | 0 |
| V4 iki dil+filtre | 0.689 | 0.736 | 0.820 | 1/30 | 0 |

Sınırlar: tek hakem (Claude, kör ama insan değil); recall havuza göreli (gerçek recall'un üst
sınırı); aynı 30 geliştirme sorusu. Korpusta veri-doğrulama konuları (S01/S02/S05) zayıf.
Ek bulgular: 2 doküman OCR/kodlama çöpü parça üretiyor ("(4))" başlıklı ve başlıksız biri);
4/300 başlık yayıncı kalıbı; retrieval tekrarı iç-doküman sayısında 53↔57↔62 oynadı.
**CANLIYA ALINDI** (kullanıcı kararı, 2026-09-30): bu makinenin `.env`'inde
`HEKTOR_RAG_QUERY_TRANSLATE=en`; uçtan uca doğrulandı (S14 Kalman: 6/6 zaman serisi kaynağı).
Çeviri hatası (`TranslationError`/`LLMUnavailable`) retrieval'ı düşürmez. İnsan etiket
örneklemesi hâlâ önerilir. 2×2 bağlamı V0 ile donduruldu
(RAG'lı koşullar mevcut hattı ölçüyor).

---

## Son seans — 2026-09-30 (4): LoRA/RAG iyileştirme protokolü + LLM-30 benchmark (eğitim YOK)

Kullanıcının `Claude_Hektor_LoRA_RAG_Iyilestirme_Promptu.txt` + `Hektor_LLM_30_Soru.txt`
dosyaları repoya bağlandı → **`docs/PROTOKOL_LORA_RAG_IYILESTIRME.md`** (8 aşama → mevcut
modül eşlemesi + açık işler). ⚠ Protokolün 2×2 harfleri `hektor mix eval`'den farklı
(protokol B = base+LoRA = bizim `C_*`; protokol C = base+RAG = bizim `B_base_rag`).
- `evals/llm30/validation.jsonl` (30 soru, `EvalItem`) + sha256 manifest; `app/evals/llm30.py`
  (yükleyici, `RUBRIC`, 13 `CRITICAL_ERRORS` + soru eşlemesi, `numeric_keys()`).
  **Geliştirme setidir, final test DEĞİL**; `human_verified=false`.
- Sızıntı kapısı (`run_leakage_check` → `train --run`) llm30'u da tarıyor; mevcut
  train (1680) + valid (111) ↔ llm30 **0 isabet** → v13 sonrası eğitimi bloklamaz.
  `assert_not_eval_path` artık tüm `evals/`i kapsar.
- v13 koşusuna dokunulmadı (20:31'de python süreci yoktu — bitiş/sonuç bu seansta
  DENETLENMEDİ); llm30 2×2 koşusu, final test takımı, rubrik puanlama aracı ve
  Aşama 1 denetim raporu **yapılmadı** (v13 bitince, Ollama boşken).

---

## Son seans — 2026-09-30 (3): Kademe 2 + veri zenginleştirme + **v13 EĞİTİMİ KOŞUYOR**

**Koşu:** `hektor_lora_v13_30b` — başladı 02:14 (`start-train.ps1`, onay `apr_40a3e32f4aac`
kullanıcı talimatıyla, mix `trading_analysis_v1` kullanıcı seçimi). Qwen3-30B-A3B, profil
`moe30b_attn_local` (**GA 8, lr 2e-4, held-out eval her 25 adımda 64 valid, load_best**),
1676 mikro-adım = **210 optimizer adımı**, ~160 sn/adım → **~10–10.5 sa, bitiş ≈12:45**.
İlk adım doğrulandı. Eğitim sürerken Ollama/web LoRA sohbeti/eval AÇMA (RAM ~128 GB).
Bitince: `run_complete.json`'da `best_checkpoint`/`best_eval_loss`; `reports/training/
hektor_lora_v13_30b_loss.json`'da artık `val_loss` dolu.

**Veri (v13):** `hektor synth-enrich` (yeni) kısa sentetik cevapları aynı bağlamdan 2-4
cümleye genişletti: 726 adaydan **443** tüm kapılardan geçti (dil, kaynak-atfı, etiket,
tekrar, tavsiye, cümle-düzeyi grounding, token bütçesi); kalanlar orijinal. Asistan medyanı
214→**272** kr; <200 kr sentetik 726→283. Yedek: `storage/synthetic_qa.bak-20260930-020921.jsonl`.
Claude örneklem incelemesi (8 satır): 5 sadık, 3'te hafif desteksiz cümle (~¼) — insan
denetimi yapılmadı. + `survivorship` disiplin tuzağı (35 satır; veride hiç yoktu, v12
"look-ahead" diyordu). pretrain-gate **GO**, lora-audit 292/292.
⚠ İlk zenginleştirme koşusu **v12 adapter'ıyla** üretilmişti (başka oturum 01:05'te `.env`
HEKTOR_LLM_MODEL=hektor-v12-30b yaptı) → çıktı ATILDI; artık `--model` + hektor-* reddi.

**Kademe 2 (v13 öncesi, 4 bulucu; kullanım sınırı yüzünden 2-oylu ajan doğrulaması yerine
kritikler elle yeniden koşturularak doğrulandı):**
- A (eğitici): engelleyici yok. A1 CLI `--iterations` varsayılanı 0 + planı aşan ret; A7 dolu
  adapter klasörü ret; A5 web kayıp grafiği (`step`) + ilerleme `val_loss`. `start-train.ps1`
  artık planı `ensure_train_split`'ten alır (lora-split 1702 ↔ 1680 farkı → 22 örnek 2. kez).
- B (eval): **B1 kritik — `max_new_tokens` 220'de base cevaplarının 80/80'i kesiliyordu →
  v10/v11/v12 "accept" kararları GÜVENİLİR DEĞİL**; artık 1024 + `truncated` + >%20 kesikse
  accept yok. B3 (1e2d5af gevşemesi gerçek ihlali kaçırıyordu) + B2/B4 contains: düzeltildi;
  B5 garanti vetosu; B7 dejenere eşiği uzunlukla. AÇIK: B6 (terfi PEFT mi GGUF mi — eval
  servis edilen GGUF'u ölçmüyor), B8 (disiplin/kalite bayrakları ayrı), B9 (rag_model_run
  parametre kaydı), B10 (auto_pipeline fp32 30B).
- C (veri): C1-C5, C6 (sızıntı kapısı artık evals/*.jsonl 80 kalem), C9, C10 düzeltildi.
- D (merge/GGUF): D1 **canlı `hektor-v12-30b` şablonunda 47 CR** (Qwen'de \r\n ≠ \n →
  eğitimde görülmemiş istem; v12'nin TÜM Ollama ölçümleri bununla) — betik düzeltildi
  (LF + create sonrası CR denetimi); D2 attention q8_0; D3 no-op adapter kapısı; D4 .partial
  + köken; D5 ad çakışması -Force. **Canlı model YENİDEN OLUŞTURULMADI** (otomatik izin
  denetçisi reddetti) → kullanıcı: `tr -d '\r'` ile Modelfile'ı düzeltip
  `ollama create hektor-v12-30b -f models/gguf/Modelfile.hektor-v12-30b`.

**v13 sonrası sıra:** (1) eval (yeni B1 ayarlarıyla, ~daha uzun) · (2) B6 kararı: terfi
edilecek yapıt GGUF ise eval Ollama üzerinden de koşulmalı · (3) merge → GGUF (attn q8_0) →
Ollama (`adapter_to_ollama.ps1`, yeni ad `hektor-v13-30b`) · ADAY; terfi insan kararı.

**Kapı:** ruff + mypy 257 + pytest **2794 passed** (Windows). Commit'ler: 1f1faaf, 01def42 + bu.

---

## Son seans — 2026-09-30 (2): v11/v12 eval + `hektor-v12-30b` Ollama'da + RAG testi

**Eval** (`logs/lora-eval-v12-v11.ps1` → `logs/lora-eval-adapter.ps1`; CPU, bf16, 80 soru,
~1 sa 35 dk/adapter; base puanları iki koşuda BİREBİR aynı → ölçüm tekrarlanabilir):

| Set | n | base 30B | v11 (600) | **v12 (tüm havuz)** | v10 4B (ref.) |
|---|---|---|---|---|---|
| discipline_core | 16 | +0.19 (13) | +0.38 (10) reject* | **+0.69 (5) reject*** | +0.94 (1) |
| risk_management | 12 | −0.17 (14) | +0.00 (12) reject* | **+0.33 (8)** | +0.67 (4) |
| overfit_awareness | 12 | +0.25 (9) | +0.67 (4) | **+0.67 (4)** | +0.83 (2) |
| format_compliance | 12 | −1.08 (25) | −0.75 (21) | **−0.33 (16)** | −0.25 (15) |
| trader_persona | 16 | −3.19 (67) | −2.06 (49) reject** | **−1.25 (36)** | −1.56 (41) |
| rag_integration | 12 | −1.17 (26) | −1.00 (24) | **−0.92 (23)** | −0.83 (22) |
| ağırlıklı ort. | 80 | −0.93 | −0.50 | **−0.15** | — |

\* `degenerate` vetosu (gerçek: cümle döngüsü). \*\* v11 `guaranteed_profit` vetosu YANLIŞ
POZİTİF ("%5 kazanç garantisi olmaz" reddi). v12: 1 CJK sızıntısı ("期权"), ort. cevap ~250
krk. Kayıt: v11 `adapter_6bd64fb2669e`, v12 `adapter_d012a940f94d` — ikisi de **candidate**,
terfi YOK.

**Ollama:** `scripts/adapter_to_ollama.ps1` (yeni; C9 kapandı) + `scripts/merge_adapter.py`
(yeni) → `hektor-v12-30b` (Q4_K_M 18.6 GB, sha256 `d91a25d231bbc660…`, şablon
`qwen3:30b-a3b-instruct-2507-q4_K_M`'den). Birleştirme doğrulaması: |fark|max 0.42 / adapter
etkisi 13.75 (%3), KL 0.0016, top-10 9/10. İlk sürümün mutlak 0.25 eşiği yanlış alarm
verdi → ölçüt adapter etkisine göre oran + KL + top-10 yapıldı. GPU'da ~90 tok/s. Diskte
`models/merged/hektor_lora_v12_30b` (61 GB) + `…-bf16.gguf` (61 GB) duruyor — silinebilir.

**RAG testi** (`reports/evals/rag_test_v12_vs_base.json`, 6 soru, gerçek RAG hattı):
atıflı cevap base 6/6 · **v12 5/6** (v10 0/30 idi) · ort. uzunluk 2.800 vs **~1.000 krk**
(4 soruda 1–2 cümle) · format bölümleri 6/6 vs 2/6 · Kelly sorusunda retrieval boşken v12
atıfsız kendi bilgisiyle cevapladı (Kural 7). İlk karar: web RAG base kalır.

**Sonra (2026-09-30 ~01:05, kullanıcı kararı): v12 TAM ana model yapıldı.**
- `repeat_penalty` 1 → **1.1** (`adapter_to_ollama.ps1 -RepeatPenalty`, eski Modelfile
  `…rp1.bak`). Uygulama `repeat_penalty` göndermez → Modelfile geçerli. RAG testinde ETKİSİZ
  (`rag_test_v12_rp11.json`: atıflı 5/6, ort. ~900 krk, S1 cümle tekrarı yine %33) → kısa
  cevap/tekrar eğitim verisinden, örnekleme ayarından değil.
- `.env`: `HEKTOR_LLM_MODEL=hektor-v12-30b` (yedek `storage/env.bak-20260930-v12web`;
  geri dönüş: `qwen3:30b-a3b-instruct-2507-q4_K_M`). `HEKTOR_LLM_MODEL` TÜM LLM işlerini
  belirler (kart, sentetik QA, mastery, RAG) → **arka plan döngüleri KAPALI** tutuldu ki
  eğitim verisi v12 ile üretilmesin (kendini besleyen kalite kaybı). Döngüleri açmadan ya da
  `synth-qa`/`card` CLI koşmadan ÖNCE modeli base'e döndür (ya da ayrı chat-model ayarı ekle).
- Web uçtan uca ✅: `/api/ask` 6 kaynak, v12 Kelly sorusunda bu kez "kaynakta yok" diye
  çekimser kaldı. Not: arka plan döngüleri base'i kullanırken RAG testi 70–117 sn/soru
  sürdü (20 GB GPU'da iki 30B takası — 2026-09-27 olayıyla aynı).

---

## Son seans — 2026-09-30: v13 reçetesi — gradient accumulation + held-out eval (eğitim YOK)

**Neden (v12 ölçümü):** batch 1, birikim yok → kayıp ~100. adımdan sonra düz + gürültülü
(0.28↔1.46, grad_norm ≤7.6); `eval_strategy: no` → validation loss yok, en iyi checkpoint
seçilemiyordu. v12 eval'i (başka oturum): 5/6 accept, **discipline_core REJECT —
`degenerate` vetosu gerçek** (survivorship cevabı cümle döngüsüne giriyor).

**Değişiklik (`peft_lora_train.py`, profil `moe30b_attn_local`):**
- `gradient_accumulation_steps` (profil: 8) · `eval_every_examples` (200) ·
  `eval_max_examples` (64, `valid.jsonl`'den seed'li) · `load_best_model_at_end` (true).
  Profilde lr 1e-4 → **2e-4** (efektif batch 8).
- `iterations` üst katmanlarda (web/start-train/nöbetçi/`plan_iterations`) **mikro-adım**
  olarak KALIR; optimizer adımına çeviri yalnız `train()`'de (`optimizer_steps` = ceil,
  HF'nin kısmi-birikim sayımıyla aynı → 1678 örnek / 8 = 210 adım = 1 epoch).
  `run_plan.json` fiili optimizer hedefini + `micro_steps` yazar → kurtarma kıyası doğru.
- save/log aralığı örnek cinsinden sabit (25/5 örnek); load_best açıkken checkpoint = eval
  aralığı (HF şartı) → çökmede en çok ~200 örnek (~1 sa) kayıp.
- Eval tqdm çubuğu kapatıldı: web/nöbetçi ilerlemeyi log'daki SON `x/y [..<..]`
  satırından okuyor; eval çubuğu (`64/64`) "koşu bitti" sandırırdı.
- `run_complete.json` artık `best_checkpoint` + `best_eval_loss` yazar; loss grafiğinde
  `val_loss` (eskiden HF'nin ayrı eval girdisi yüzünden hep None) dolar.
- Doğrulama: rastgele küçük Qwen3 (4B tokenizer) ile GERÇEK `train()` uçtan uca: 40 mikro →
  10 optimizer adımı, 5 eval, best checkpoint kaydı ✅. 30B'de eval süresi **ÖLÇÜLMEDİ**
  (tahmin ~7 dk/eval, tam havuzda +~1 sa).

**Kapı (Windows):** ruff format/check ✅ · mypy 256 ✅ · pytest **2743 passed, 4 deselected**.

**Açık bulgular (öneri sırası devamı, henüz yapılmadı):**
- Öneri 3 veri: kısa cevaplar neredeyse tamamen **sentetik QA**'dan (1051 satır, medyan 166
  kr, 726'sı <200); disiplin ~230, diğer ~850. → sentetik ağırlığını düşür / açıklamalı yeniden
  üret (karar kullanıcıda; veri değiştirilmedi).
- Öneri 7 — **ÖLÇÜLDÜ, değişiklik gereksiz:** Ollama `hektor-v12-30b` (GGUF, temp 0.7, seed 42)
  discipline_core 16 soruda `repeat_penalty` 1.0 / 1.1 / `presence_penalty` 1.5 → dejenere
  **0/16 her üçünde**, CJK sızıntısı 0. Yani eval'deki `degenerate` vetosu HF/PEFT greedy
  çözümlemeye ait; Modelfile'a dokunulmadı. İçerik hatası: survivorship bias'ı
  "look-ahead bias" diye adlandırıyor (veri/eğitim konusu).
- Öneri 8 — **YAPILDI:** `_token_hit` (eval `contains:` bayrağı) çok-kelimeli ifadeleri
  koşulsuz bayraklıyordu ve yoksunluk ekini görmüyordu ("tek backtest yeter" ⊂ "yetersiz").
  Artık: yoksunluk eki (-sız/-suz) = zıt anlam; çok-kelimeli ifade de olumsuzlama
  penceresinden geçer (6 kelime + yanıltıcı/yanlış/hatalı); "değil mi?", "yanlış değil",
  "risk yok / sorun değil" olumsuzlama SAYILMAZ. Kayıtlı 86 bayrak yeniden hesaplandı →
  7'si düştü, 7'si de elle çürütme. Eski `test_multiword_token_stays_strict` bilinçli
  olarak ters çevrildi (veri gerekçesi docstring'de). Mevcut rapor JSON'ları yeniden
  yazılmadı; yeni eval'ler yeni kuralla hesaplanır.

---

## Son seans — 2026-09-28 (3): Qwen3-30B-A3B-Instruct-2507 LoRA hazırlığı (eğitim YOK)

Dal `claude/lora-30b-a3b-prep`. **Kullanıcı kararları:** base `Qwen/Qwen3-30B-A3B-Instruct-2507`
indirilsin · LoRA **yalnız attention** (router + uzmanlar donuk) · karışım
**`trading_analysis_v1`** (`wd_d310e37d9a`, 24 saat taze; bayatlarsa yeniden sor) ·
eğitim web arayüzüne bağlı olsun ve düşmesin → nöbetçi zamanlayıcıya + uyku engeli.

**Makine:** 128 GB RAM, Xeon w7-2575X 22 çekirdek, torch CPU, RTX 4000 Ada 20 GB (eğitimde
kullanılmıyor). 30B bf16 ≈ 61 GB → CPU'da **bf16 zorunlu** (fp32 ≈ 146 GB, sığmaz).

**Bulgular / düzeltmeler:**
1. **MoE hedef tuzağı (ölçüldü, meta cihazda).** transformers 5.16 `Qwen3MoeExperts` uzmanları
   birleşik 3B parametre tutar → `gate_proj/up_proj/down_proj` MODÜL değil. 7'li
   `TARGET_MODULES` PEFT'te **sessizce yalnız attention**'a düşerdi. Artık `train()` model
   yüklendikten sonra `unmatched_target_modules` ile kontrol eder; eşleşmeyen hedef → açık hata.
   Gerçek config: 48 katman · 128 uzman · top-8 · hidden 2048 · 30.53B; attention-only r=16 →
   13.37M eğitilebilir (%0.044).
2. **Profildeki `target_modules` hiç uygulanmıyordu** (YAML'da yazılı, `load_lora_profile`
   atlıyordu). Artık uygulanır + doğrulanır (yalnız TARGET_MODULES alt kümesi; lm_head/embed
   yasak). Yan etki: `small_smoke_test` artık gerçekten yalnız attention eğitir (YAML niyeti).
3. Yeni profil **`moe30b_attn_local`** (attention, r16/α32, lr 1e-4, NEFTune 5, maskeli,
   `gradient_checkpointing: true`, `max_examples: 600` — adım süresi ÖLÇÜLMEDİ).
4. **RAM ön-kontrolü** (`check_cpu_ram`): checkpoint boyutu yerel index'ten; yetmezse model
   yüklenmeden hata (nöbetçinin OOM'u sonsuz diriltmesi önlenir). `dry_run` çıktısında `ram_check`.
5. **Uyku engeli**: `trainer.train()` süresince `SetThreadExecutionState` (kalıcı ayar değil).
6. **Web**: `/api/training/run` artık `profile` + `max_examples` alır (profil onay yakılmadan
   doğrulanır); formda profil seçimi + örnek tavanı; A3B yazınca MoE profili seçilir;
   iterasyon 0 = plandan.
7. **Nöbetçi tuzağı (canlı):** v10 BİTMİŞ ama `train_status.json` diskte kalmıştı ve
   `train-recovery-check` "kurtarma yetkili" diyordu → nöbetçi açılsaydı bitmiş koşuyu her 5
   dk diriltmeye çalışırdı. Artık son `checkpoint-N ≥ iterations` ise **YETKİSİZ (tamamlanmış)**.
8. **`HektorTrainingWatchdog` Görev Zamanlayıcı'ya kaydedildi** (kullanıcı onayı; 5 dk,
   gizli pencere, `start-server.ps1`'deki tanımın aynısı). Önceden bu makinede YOKTU.

**Kapı (Windows):** ruff format/check ✅ · mypy 256 ✅ · pytest **2720 passed, 4 deselected**
(`-m "not ollama"`; ilk koşuda statik arayüzde sabit model adı testi yakaladı → metin düzeltildi).
Commit/push YOK (kullanıcı istemedi). Çalışan web sunucusu yeni `/api/training/run` alanları
için **yeniden başlatılmalı** (statik dosyalar hemen, Python ucu yeniden başlatmada).

### Kademe 2 (30B öncesi) — 3 finder + her gruba 2 bağımsız şüpheci doğrulayıcı

Alt sistemler: **A** eğitici · **B** başlatma/kurtarma/web · **C** eğitim-sonrası (eval/kayıt/
sohbet/merge). 33 bulgu; iki oyda da "eğitimden ÖNCE EVET" alanlar düzeltildi:

| ID | Sorun (gerçek veriyle ölçüldü) | Düzeltme |
|---|---|---|
| A1=B2 | 600 örnekte 2 satır maskelenemiyor → hedef 598'e kırpılıyor, son checkpoint 598, durum `iterations=600` → BAŞARILI koşu "çökmüş" sanılıp her ~15 dk 61 GB yükleyen sonsuz diriltme | trainer `run_plan.json` (fiili max_steps) + `run_complete.json` yazar; `recovery_allowed` bunları kullanır; `train` bitişte durum kaydına `finished_at` |
| A2=B3 | hedef uyuşmazlığı 61 GB yüklemeden SONRA; `train` hata → exit 0 → sonsuz döngü | `precheck_target_modules` (meta cihaz, yükleme öncesi; web ön-kontrolünde de); hata → `failed_at` + exit 7 |
| N1 | start-train kurtarmada `started_at`'i yeniliyordu → 72 sa sınırı hiç işlemiyordu | kurtarmada `started_at` korunur + `recovery_attempts` (≥3 → dirilme yok) |
| B1 | web "Durdur" edilen koşu ~10 dk sonra onaysız diriliyordu | `stop_requested_at` → kurtarma YETKİSİZ |
| C3(+C5) | web/nöbetçi koşusu kayıt defterine HİÇ girmiyordu | supervised koşu da CANDIDATE kaydedilir (auto_pipeline hariç: `skip_register`); fiili örnek sayısı |
| B4/B5/B6/B12 (ucuz) | aynı adlı eski checkpoint · plandan fazla iterasyon · RAM kontrolü onaydan sonra · RESUME env sızıntısı | ön-kontrolde red · launch reddeder, form varsayılanı 0 · web ön-kontrolünde RAM · launch `RESUME=0` |

**Eğitim-SONRASI açık (düzeltilmedi, eval/sohbetten ÖNCE yapılmalı):** C1 web LoRA sohbeti
eski modeli boşaltmadan yenisini yükler (2×61 GB) → eğitim sürerken web sohbeti KULLANMA ·
C2 `peft_llm_shim` fp32 + None'da merdiven işaretsiz atlanır (auto_pipeline; 30B'ye erişilmez)
· C6 `lora-eval` adapter/base uyumunu base üretiminden SONRA doğrular, `--base-model` sessizce
ezer → 30B'de `--base-model` VERME · C7 sohbette "base" = 4B · C8 eval'de RAM ön-kontrolü yok
· C9 merge/GGUF kodda yok, `docs/arsiv/4b_donemi/RAG_LORA_ENTEGRASYON.md` eski · C10/C11 küçük. Ayrıca A3
(CLI'da onay profil doğrulamasından önce), A5, A6, A8 (uyku engeli dönüşü), A9 (KL yorum
yönü), A10, B7–B11 düşük önem. DB'de bayat pending onaylar var (`apr_98e76bb03dfa`,
`apr_da7a846b5bdc`) — onaylanmamalı.

**Kapı (Windows, Kademe 2 düzeltmeleri sonrası):** ruff ✅ · mypy 256 ✅ · pytest **2731 passed**.

### Koşu: `hektor_lora_v11_30b` — BAŞLADI 2026-09-28 21:25 (web ucundan)

- İndirme: Xet katmanı iki kez son parçalarda takıldı (19:34, 20:49) → `HF_HUB_DISABLE_XET=1`
  ile parça parça tamamlandı; 16/16 parça safetensors başlığıyla doğrulandı (61.06 GB).
- `.env`: `HEKTOR_PEFT_BASE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`,
  `HEKTOR_BACKGROUND_LOOPS_ENABLED=false` (eğitim bitince **true** yap; yedek
  `storage/env.bak-20260928-30b`). Web yeniden başlatıldı (`storage/web.20260928-2056.log`).
  Not: `HEKTOR_API_TOKEN` bu makinede BOŞ (önceden de öyleydi) → web API kimliksiz.
- Onay `apr_9ffc3cfa562e` (kullanıcı talimatıyla), karar `wd_d310e37d9a` tüketildi.
- 600 örnek → 2 satır maskelenemedi → **598 adım** (Kademe-2 A1 tam öngörüldüğü gibi;
  `run_plan.json` max_steps=598 yazıldı). ~15–20 sn/adım → toplam ~3 saat.
- **BELLEK (ölçüldü, tahmin YANLIŞ):** `check_cpu_ram` ≈71 GB dedi; gerçek süreç özel
  belleği **121.5 GB** (sabit), sistem commit 152/154 GB. Sayfa dosyası sistem-yönetimli,
  8 → 27 GB kendiliğinden büyüdü; OOM olmadı. Kök neden araştırılmadı (aday: transformers
  5.x birleşik-uzman dönüşümü / CPU grouped_mm tamponları). → `_RAM_OVERHEAD_*` MoE için
  düzeltilmeli; bu koşu sırasında Ollama/web sohbeti/eval AÇMA.

**v11 SONUÇ (2026-09-29 00:36):** 598/598 adım, 3 sa 10 dk, `train_loss` 1.053, epoch 1.0.
`run_complete.json` yazıldı; `train-recovery-check` → "koşu TAMAMLANMIŞ" (A1 düzeltmesi canlıda
işledi). Kayıt: `adapter_6bd64fb2669e` **candidate** (30B base, r16, attention, 598 örnek —
C3 düzeltmesi işledi). Bellek sonda ~128 GB özel (başta 121.5; ~2.4 GB/saat artış). Eval YOK,
terfi YOK.

**v12 BAŞLADI (2026-09-29 08:10):** `hektor_lora_v12_30b`, tüm havuz 1682 örnek × 1 epoch,
onay `apr_ff727c199d36` (gece 00:40'ta oluşturuldu; araç güvenlik sınıflandırıcısı geçici
arızalandığı için sabaha kaldı), karar `wd_63b1219424` (`trading_analysis_v1`). Web'den.

**Kullanıcı kararı (2026-09-28 23:15):** "adımları max yapalım ya da bir sonraki eğitimde" →
v11 (600 örnek) bitirilir; **bir SONRAKİ 30B eğitimi TÜM havuzla** (1682 örnek × 1 epoch ≈
1680 adım, ~9–10 saat; web: örnek tavanı 1682, iterasyon 0, adapter `hektor_lora_v12_30b`).
Önce: bellek artışı (~2.4 GB/saat) incelenmeli; karışım ağırlığı YENİDEN sorulmalı.

**Sıradaki (sırayla):**
1. ~~İndirme~~ — tamam.
   Yarıda kalırsa: `.venv/Scripts/hf.exe download Qwen/Qwen3-30B-A3B-Instruct-2507` kaldığı yerden sürer.
2. ~~Kademe 2~~ — YAPILDI (yukarıda).
3. Ollama modelini boşalt (30B q4 GGUF RAM/VRAM tutar), arka plan döngülerini kapat.
4. **Kısa ölçüm koşusu** (Kural 8 onayıyla, ör. `max_examples` 20) → s/adım ölç, sonra
   `max_examples`'ı süreye göre seç. Web: Temel model `Qwen/Qwen3-30B-A3B-Instruct-2507`,
   profil `moe30b_attn_local`, iterasyon 0. CLI eşdeğeri:
   `.\scripts\start-train.ps1 -Adapter hektor_lora_v11_30b -BaseModel Qwen/Qwen3-30B-A3B-Instruct-2507 -Profile moe30b_attn_local -MaxExamples <N>`
5. Eğitim sonrası: eval HF/PEFT ile CPU'da 30B → yavaş; merge (bf16 ≈ 61 GB) →
   llama.cpp `convert_hf_to_gguf` (qwen3moe destekli) → Q4_K_M → Ollama yolu 4B'dekiyle aynı.
   ADAY; terfi ayrı insan kararı.

---

## Son seans — 2026-09-27 (3): neden `qwen3:30b` → `qwen3:30b-a3b-instruct-2507-q4_K_M` (ölçüldü)

**Kod değişmedi, eğitim yok.** Bu makinenin `.env`'i 10:47'de `logs/switch-to-instruct.ps1`
ile `HEKTOR_LLM_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M`'e geçmişti; gerekçesi yazılmamıştı.
Bu kayıt onu ölçümle belgeler. Kod varsayılanı (`qwen3:4b-instruct-2507-q4_K_M`) aynı.

**Kök neden — 2026-09-13'teki 4B tuzağının aynısı.** Ollama'nın `qwen3:30b` etiketi hibrit
değil, **Qwen3-30B-A3B-Thinking-2507**'dir (`/api/show`: `general.finetune=Thinking`, şablon
son `<think>`'i kapatmaz → `classify_think_support` = `forced`, düşünme kapatılamaz). Instruct
etiketi `general.finetune=Instruct` → `toggle`. İkisi de 30.5B MoE (~3B aktif), q4_K_M,
18.6 GB: **boyut/bellek farkı yok, yalnız ince-ayar farkı.**

**Ölçüm (2026-09-27 ~21:10, Ollama 0.34.4, RTX 4000 Ada 20 GB).** Payload
`LocalLLM._generate_ollama` ile birebir: think kararı, `num_predict` (varsayılan 1024; forced +
serbest metinde +1024), çağrı yerlerinin temperature/seed değerleri, gerçek `rag_answer` ve
`knowledge_card` system prompt'ları. **Tek sapma:** Thinking modeli CPU'da (`num_gpu=0`) koştu,
çünkü mastery kuyruğu instruct'ı GPU'da kullanıyordu ve iki 30B 20 GB'a sığmıyor (aşağıdaki
olay). Süreler cihazlar arası karşılaştırılamaz; **karşılaştırılabilir metrik üretilen token
sayısıdır** (aynı mimari + kuantizasyon → token başına iş aynı).

| Çağrı | Instruct (GPU): token · üretim | Thinking (CPU): token · üretim | Not |
|---|---|---|---|
| "Tek kelimeyle: 2+2" (serbest metin) | **2** · 0.02 sn → "4" | **1219** · 41.9 sn → "dört" | 4.5k krk atılan düşünme |
| EN→TR çeviri, "5-10%" (serbest) | **36** · 0.4 sn | **849** · 28.6 sn | ikisi de "%5-10"u korudu |
| RAG cevabı, `rag_answer` prompt'u (serbest) | 1024 · 11.6 sn, `length` — cevap kesik ama VAR (2964 krk) | 2048 · 93.4 sn, `length` — **cevap 0 krk** | app'te `LLMUnavailable` |
| Bilgi kartı, `format=json`, max 700 | 637 · 7.3 sn, geçerli JSON, `main_claim` dolu | 379 · 14.8 sn, geçerli JSON, `main_claim` dolu | JSON'da `think=false` gider |

Hız: instruct GPU 87–127 tok/s, Thinking CPU 22–30 tok/s. **Tahmin (ölçülmedi):** Thinking
GPU'da aynı hızda koşsa 2+2 ≈ 13 sn, çeviri ≈ 9 sn, RAG ≈ 23 sn — ve yine cevapsız.

**Sonuç.**
1. Serbest metin çağrılarında Thinking ~24–600× daha çok token üretiyor; fazlası atılan düşünme.
2. En kritiği: gerçek RAG prompt'unda düşünme 1024+1024 bütçenin tamamını yedi → **cevap yok.**
   Küçük bütçeli çağrılarda risk daha büyük: `comprehension_scorer` `max_tokens=120` → forced'da
   1144 token; 2+2'nin düşünmesi bile 1219 token tuttu (çıkarım — bu çağrı ayrıca ölçülmedi).
3. `format=json` çağrılarında (kart, synth-qa, l3/l4 sınav, formül, sentez) `think=false`
   gittiği için fark küçük; ikisi de geçerli kart üretti.
4. Vaka başına n=1, sabit seed → yön göstergesi, istatistik değil.

**Yan bulgular (açık).**
- Instruct'ta da RAG cevabı 1024 varsayılan bütçeye takıldı (9 bölümlü format uzun) → cevabın
  sonu kesiliyor. `rag_answerer` `max_tokens` vermiyor; bütçe mi format mı kısalsın — karar/iş.
- `num_gpu=0` verilse bile Ollama 0.34.4 Thinking modelinin ~0.79 GB'ını GPU'ya koydu (instruct
  etkilenmedi; toplam ~20.0/20.5 GB).
- Çeviri ikisinde de kusurlu: instruct "drawdown"→"çekiliş", "net of"→"…ile birlikte";
  Thinking "net of"→"hariç". Yüzde ikisinde korundu.

**Olay (16:02–17:20).** 10:17'de başlayan `mastery-queue` eski `.env` ile `qwen3:30b`'yi, yeni
`.env` ile kalkan süreçler instruct'ı çağırdı. İki 30B 20 GB'a sığmadığı için Ollama modelleri
sürekli değiştirdi, embedding modeli yüklenemedi → `Read timed out`. 17:39'da kuyruk yeniden
başlatıldı (`logs/mastery-resume.ps1`), artık yalnız instruct. **Aynı makinede iki 30B'yi aynı
anda çağıracak iş başlatma.**

**`qwen3:30b` silindi (kullanıcı kararı, 2026-09-27 ~21:25).** Önce doğrulandı: yüklü değildi,
`.env`/`configs`/`storage` içinde düz `qwen3:30b` referansı yoktu. Ollama'da yalnız
`qwen3:30b-a3b-instruct-2507-q4_K_M` + `nomic-embed-text` kaldı; ~17 GB disk boşaldı; mastery
kuyruğu kesintisiz sürdü.

**Açık:** `setup.ps1`/`setup.sh` seçenek [4] ve README tablosu hâlâ `qwen3:30b`'yi (Thinking)
kuruyor → yeni kurulum aynı tuzağa düşer; `qwen3:30b-a3b-instruct-2507-q4_K_M` olmalı.

---

## Son seans — 2026-09-27 (2): LoRA karışım profilleri + router + profil eval (eğitim YOK)

Dal: `claude/lora-mix-eval`. **Eğitim başlatılmadı, model indirilmedi, base/production'a
dokunulmadı.** Ayrıntı: **[docs/LORA_MIX_EVAL.md](docs/LORA_MIX_EVAL.md)**.

- Achilles "Adapter Mixing / Profile Routing / Evaluation" şartnamesi Hektor'a uyarlandı:
  `configs/lora/mix_profiles.yaml` (eğitim reçeteleri olan `lora_profiles.yaml`'dan AYRI),
  `app/lora/{domain_adapter_registry,profile_registry,profile_builder,profile_router,
  weight_decision,mix_cli}.py`, `app/memory/rag_version.py`, `app/evals/profile/` (eval
  runner + deterministik değerlendiriciler + regression gate + insan denetimi + karşılaştırma),
  `evals/profile_mix/` (validation 27 + golden_test 27, sha256 manifest).
- **Yeni kullanıcı kuralı — her LoRA eğitiminden ÖNCE karışım ağırlıkları sorulur.**
  `train --run` karar olmadan exit 5; web/detached `launch()` alt süreci hiç açmaz;
  `start-train.ps1` exit 5/6 için yönlendirme basar. Karar: `uv run hektor mix weights`
  (etkileşimli ya da `--profile`/`--weights`) veya `train --run --mix-profile X`.
  Karar tek kullanımlık. Ayrıca `train --run` artık eval sızıntısında exit 6.
- Gerçek PEFT `add_weighted_adapter` küçük rastgele Llama ile 4 yöntemde (svd/linear/ties/
  dare_ties) doğrulandı (torch 2.14 · transformers 5.16 · peft 0.20, ayrı geçici venv) —
  base ağırlıkları bayt-bayt aynı kaldı. Ana venv'de peft yok → o 4 test burada `skip`.
- Kapı: ruff format/check ✅ · mypy (255 dosya) ✅ · pytest **2409 passed, 8 skipped**.

**Açık işler (sıra önerisi):**
1. `evals/profile_mix/*.jsonl` seed setleri Claude yazdı → `human_verified=false`; insan
   doğrulaması + korpusa özgü RAG soruları (`expected_chunk_ids`) eklenmeli, sonra
   `write_manifest()` ile hash yenilenmeli.
2. Domain başına eğitim veri setleri (math/statistics/reasoning/trading/coding) yok —
   her adapter base'ten bağımsız eğitilecek; eğitimden ÖNCE Kademe 2 + ağırlık sorusu.
3. GGUF→Ollama servis modeli olmadan C/D sistemleri "koşulamadı" raporlanır.

---

## Bilinen açık işler

- **LoRA/RAG protokolü** (`docs/PROTOKOL_LORA_RAG_IYILESTIRME.md` §10): final test takımı
  (görülmemiş şablonlar), teşhis varyantları, rubrik puanlama aracı + kritik hata kapısının
  `lora-eval`'e bağlanması, Aşama 1 kanıtlı denetim raporu.
- **B6:** terfi edilecek yapıt GGUF ise eval Ollama üzerinden de koşulmalı (eval PEFT'i ölçüyor).
- **D1:** canlı `hektor-v12-30b` Modelfile şablonunda CR — yeniden oluşturulmadı.
- **Ollama determinizmi:** aynı seed + temperature 0'da cevaplar ayrışıyor (önek önbelleği /
  CPU-GPU bölünmesi) → karşılaştırmalar tekrarla yapılmalı.
- **Korpus:** 2 doküman OCR/kodlama çöpü parça üretiyor; 4/300 başlık yayıncı kalıbı;
  `papers.source` hepsinde `manual` (amaç ayrımı `configs/rag/doc_purposes.yaml`'da).
- **CI yeşil olmadan merge — dal koruması dışarıdan kaldırılıyor (AÇIK, 2026-09-15).** Kimin
  kaldırdığı: github.com/settings/security-log → `protected_branch.destroy`.
  Kontrol: `gh api repos/alimirbagirzade/hektor/branches/main --jq .protected`.
- **Ajan envanteri §5, sıra 2-6** (`reports/agent-inventory/envanter-2026-09-09.md`):
  manifest ↔ test drift testi, `grounding_verifier` mutasyon testi, kart+SFT hattını
  doğrulayıcıdan geçirme, K3 normları, eval setini büyütme.
- `docs/MIGRASYON_2.0.md` §"Kalan adaylar"; `docs/MIMARI_REFERANS.md` v1 dönemi (uyarılı).
- 4B dönemindeki açık işlerin tam hâli: `docs/arsiv/4b_donemi/HANDOFF_2026-09_4B_donemi.md` sonu.


## Önemli dosyalar

| Dosya | Ne |
|---|---|
| `CLAUDE.md` | Çalışma kuralları (bağlayıcı) |
| `docs/MIGRASYON_2.0.md` | v1 → 2.0 farkları |
| `docs/MIMARI_REFERANS.md` | Alt sistem alt sistem mimari referansı (v1 dönemi) |
| `automation_manifest.yaml` | Runtime ajanlarının tek bildirimsel kaynağı + zincir |
| `configs/lora/lora_profiles.yaml` | LoRA eğitim profilleri (`moe30b_attn_local` varsayılan; düşük RAM `discipline_safe_local`) |
| `docs/SCOPE_ISOLATION.md` | Sürücü motor ≠ insan yetkisi |
| `SECURITY.md` | Tehdit modeli + ağa açma checklist'i |
| `docs/PROTOKOL_LORA_RAG_IYILESTIRME.md` | LoRA/RAG iyileştirme protokolü + LLM-30 |
| `docs/arsiv/4b_donemi/HANDOFF_2026-09_4B_donemi.md` | 4B dönemi oturum kayıtları (v8–v10) |
