# 🏛️ Hektor Trader AI

> **Yerel-öncelikli AI araştırma sistemi** — macOS · Windows · Linux.
> Akademik finans makalelerini okur, trade hipotezleri üretir, backtest eder, sonuçtan öğrenir.
> **Yerel Ollama** ile çalışır (ücretsiz, internetsiz, API anahtarı gerektirmez — **VARSAYILAN ve önerilen**). Bulut sağlayıcılar (OpenAI · Anthropic · Google) **opsiyoneldir ve bu projede gerekmez** — çalışma-zamanı tamamen yereldir.

> ⚠️ **Bu bir araştırma aracıdır — canlı bot DEĞİLDİR ve yatırım tavsiyesi VERMEZ.**
> Tüm çıktılar test edilmesi gereken _hipotezlerdir_. Gerçek parayla kullanımın sorumluluğu tamamen size aittir.

> 📘 **Yerel LoRA eğitimi (donanım, model, ölçülmüş süreler):** [docs/EGITIM_PROTOKOLU.md](docs/EGITIM_PROTOKOLU.md)

---

## ⚡ Kurulum

> **Bilgisayarını açacaksın, terminale birkaç satır yazacaksın. Hepsi bu.**
> Script her şeyi otomatik yapar: indirme, kurma, hazırlama.
> Kurulum bitince sistem sana "Bilgisayarına göre şu modeli kullan" diyecek.

---

### macOS (Apple Silicon — M1/M2/M3/M4)

**Gereksinimler:** Mac bilgisayar (M çipli) · internet bağlantısı

```bash
git clone https://github.com/alimirbagirzade/hektor.git
cd hektor
bash setup.sh       # uv + model seçimi + Ollama + init — tek komut
uv run hektor-web # → http://127.0.0.1:8765
```

Kurulum açılır ve **9 yerel model seçeneği** sunar (hepsi internetsiz ve ücretsiz;
API anahtarı istenmez):

```
  [1] qwen3:4b        ~2.5 GB    8 GB+ RAM   Hızlı   ← önerilen başlangıç
  [2] qwen3:8b        ~5 GB     16 GB+ RAM   Dengeli
  [3] qwen3:14b       ~9 GB     32 GB+ RAM   Güçlü
  [4] qwen3:30b      ~20 GB     32 GB+ RAM   Çok güçlü
  [5] llama3.1:8b     ~5 GB     16 GB+ RAM
  [6] llama3.1:70b   ~40 GB     80 GB+ RAM   Çok güçlü
  [7] mistral:7b      ~4 GB      8 GB+ RAM   Hızlı
  [8] deepseek-r1:8b  ~5 GB     16 GB+ RAM   Akıl yürütme
  [9] deepseek-r1:14b ~9 GB     32 GB+ RAM   Güçlü
```

> Bu makinede LoRA eğitimi **destekleniyor** (Apple Silicon — MLX ile hızlı).
> Homebrew sadece Ollama seçildiğinde ve kurulu değilse otomatik yüklenir.

**🔄 Daha sonra yeni sürüme güncellemek için** (teknik bilgi gerekmez):

1. **Terminal**'i aç → `Cmd + Boşluk` tuşla, "**Terminal**" yaz, **Enter**.
2. Aşağıdaki **iki satırı** kopyala, Terminal'e yapıştır, **Enter**:
   ```bash
   cd ~/hektor
   bash update.sh
   ```
3. Bittiğinde tarayıcıda **`Cmd + Shift + R`** yap (sayfayı tazele).

**Güncelleme olmuyorsa / hata veriyorsa** (ör. `cd ~/hektor` "no such file" diyorsa) —
şu **tek satırı** kopyala-yapıştır. Hektor klasörünü **kendisi bulur**, girer ve günceller
(her şeyi düzeltir, verilerin silinmez, izin/`chmod` gerekmez):
```bash
cd "$(find ~ -type d -name hektor 2>/dev/null | head -1)" && bash update.sh --force
```

---

### Linux (Ubuntu 20.04+ / Debian / Fedora)

**Gereksinimler:** 64-bit Linux · Python 3.12+ · internet bağlantısı

```bash
git clone https://github.com/alimirbagirzade/hektor.git
cd hektor
bash setup.sh       # uv + backend seçimi + modeller + init
uv run hektor-web # → http://127.0.0.1:8765
```

Kurulum başında 9 yerel model seçeneği çıkar; RAM/disk kontrolü yapılır ve Ollama + seçilen model otomatik indirilir (kurulum `sudo` gerektirebilir, systemd servisi oluşturur).

> Linux'ta LoRA eğitimi **PEFT/CPU ile desteklenir** ama yavaştır → pratikte **bulut-GPU (Kaggle)** önerilir. LoRA paketleri gerekirse: `uv pip install -e ".[train-cpu]"`.
>
> 🌐 **Uzaktan / kiralık sunucu mu?** `setup.sh` sana **"yerel mi, uzaktan mı erişeceksin?"** diye sorar. **Uzaktan** seçersen web `0.0.0.0`'a açılır + otomatik **API token** (şifre) üretir; kurulum sonunda **doğrulama** (`scripts/verify-install.sh`) ve isteğe bağlı **açılışta otomatik başlatma** (systemd) da yapılır. 8765 portunu sağlayıcının firewall'ında açmayı unutma.

**🔄 Sonradan güncelleme** (kurduğun klasörde): `bash update.sh` (takılırsa
`bash update.sh --force` — `main`'e geçip `origin/main`'e eşitler). Makine güncel mi:
`uv run hektor doctor`. Detay: **[docs/GUNCELLEME_KILAVUZU.md](docs/GUNCELLEME_KILAVUZU.md)**.

---

### Windows 10 / 11

> LoRA egitimi: macOS Apple Silicon (MLX) + Windows/Linux (PEFT/CPU) desteklenir.
> RAG, backtest, formul cikarma, web arayuzu tum platformlarda tam calismaktadir.

**Tek komutla kurulum** — PowerShell'i ac, asagidaki satiri kopyalayip yapistir:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force; irm https://raw.githubusercontent.com/alimirbagirzade/hektor/main/install.ps1 | iex
```

Bu komut her seyi halleder:
- Git yoksa otomatik kurar
- Projeyi her zaman `C:\Users\<kullaniciadiniz>\hektor` konumuna indirir
- Kurulum sihirbazini baslatir (backend secimi, model indirme, veritabani)

**Kurulumdan sonra sunucuyu baslatmak:**

```powershell
cd "$env:USERPROFILE\hektor"
uv run hektor-web
# Tarayicide ac: http://127.0.0.1:8765
```

**Sunucuyu Windows açılışında otomatik başlatmak (önerilen):**

```powershell
cd "$env:USERPROFILE\hektor"
.\scripts\start-server.ps1 -Install
```

Bu komut şunları yapar: web servisi login'de otomatik başlar + her gece **03:00**'de `update.ps1` ile güncelleme zamanlanır.

**Güncelleme — KURULU makinede TEK KOMUT** (yeni sürüm yayınlandığında):

> 📘 **Birden fazla bilgisayar mı var, "bir makinede güncel diğerinde değil" mi yaşıyorsun?**
> Tam adım-adım çözüm: **[docs/GUNCELLEME_KILAVUZU.md](docs/GUNCELLEME_KILAVUZU.md)**
> (ilk-seferlik onarım · `hektor doctor` teşhisi · Windows otomatik görev onarımı).

**Windows (PowerShell):**
```powershell
cd "$env:USERPROFILE\hektor"
.\update.ps1
```

**macOS / Linux (bash terminal):**
```bash
cd ~/hektor            # Hektor'in kurulu olduğu klasör
bash update.sh
```

Her iki script de şunu yapar: web sunucusunu durdur (port 8765) → **hangi dalda olursan ol
`main`'e geçip `origin/main`'e yakınsa** (ff-only; eskiden yapmıyordu, "oturmuyor"un kök sebebi
buydu) → **`uv sync --extra dev`** → web'i yeniden başlat → sağlık kontrolü. **Eğitime dokunmaz.**
Yerel main GitHub'dan ıraksaksa **otomatik birleştirme yapmaz**; `-Force` ile yereli atıp eşitlersin.

> 🔴 **Güncelledikten sonra tarayıcıda sert yenileme yap:** Windows/Linux **`Ctrl + Shift + R`**,
> macOS **`Cmd + Shift + R`** — yoksa arayüz eski JS/CSS'i önbellekten gösterir, "değişmemiş" gibi görünür.

**"git pull olmuyor / değişiklik çekmiyor" ise** (yerel değişiklik veya çakışma engelliyordur):

```powershell
.\update.ps1 -Force      # Windows — yerel KOD değişikliklerini atıp origin/main'e eşitler
```
```bash
bash update.sh --force      # macOS / Linux — aynısı
```

`-Force` yalnız **tracked kod** dosyalarını sıfırlar; verilerin (`data/`, `storage/`, `vector_db/`,
adapter'lar) git'te izlenmediği için **silinmez**. Salt-kopya kurulumlarda güvenle kullanılır.

**Makinen güncel mi, emin değil misin?** Hiçbir şeyi değiştirmeyen teşhis komutu:

```bash
uv run hektor doctor   # dal=main mi? origin/main ile aynı mı? (Windows'ta görev yolu doğru mu?)
```

Sapma varsa ne yapılacağını söyler. **Windows'ta** açılış/gece-03:00 güncelleme görevleri yanlış
klasörü gösteriyorsa (`doctor` "ÖLÜ/yabancı yol" derse) onar: `.\scripts\start-server.ps1 -Repair`.
Tam kılavuz: **[docs/GUNCELLEME_KILAVUZU.md](docs/GUNCELLEME_KILAVUZU.md)**.

### Mac/Linux'ta `update.sh` HENÜZ YOKSA veya `git pull` hiç çalışmıyorsa (ilk seferlik kurtarma)

> Bu komutlar **GitHub sitesinde değil**, Mac'in **Terminal** uygulamasında çalışır.
> Terminal'i aç: `Cmd + Boşluk` → "**Terminal**" yaz → Enter. Sonra aşağıdakileri kopyala-yapıştır.

```bash
cd ~/hektor                  # Hektor klasörü (başka yerdeyse oraya gir)
bash update.sh --force         # main'e geç + GitHub'a eşitle (yalnız KOD; verilerin SİLİNMEZ)
                               # bundan sonra her güncellemede sadece: bash update.sh
```

**`cd ~/hektor` "no such file or directory" diyorsa** (klasör başka yerde) — şu **tek satır**
klasörü kendisi bulur, girer ve günceller (kopyala-yapıştır, izin/`chmod` gerekmez):

```bash
cd "$(find ~ -type d -name hektor 2>/dev/null | head -1)" && bash update.sh --force
```

Eğer `not a git repository` veya `no such file or directory` hatası alırsan, o klasör git deposu
değildir (ör. ZIP olarak indirilmiş) — **sıfırdan temiz indir:**

```bash
cd ~
git clone https://github.com/alimirbagirzade/hektor.git
cd hektor
bash setup.sh                  # kurulum sihirbazı (backend/model/veritabanı)
```

---

## 🟢 EN BASİT KULLANIM (yeni başlayan için, adım adım)

> Hiç teknik bilgi gerekmez. 3 şey yaparsın: **aç → makale ekle → soru sor.**

### 1) Web arayüzünü aç
Tarayıcıda şunu yaz:  **http://127.0.0.1:8765**
(Açılmıyorsa terminalde: `uv run hektor-web` yaz, sonra tekrar aç.)

### 2) Makale ekle (kendi PDF'lerin)
- Üstte **02 · MAKALELER** sekmesine tıkla.
- PDF'leri kutuya **sürükle-bırak** (birden fazlasını birden seçebilirsin, maks 100 MB).
- Sistem otomatik okur ve indeksler. **Aynı makaleyi 2 kez yüklersen otomatik atlar.**

### 3) Soru sor
- **01 · ARAŞTIRMA** sekmesine git.
- Sorunu yaz (örn. _"Momentum yüksek volatilitede nasıl çalışır?"_) → **SORGULA**.
- Cevap **yalnızca senin makalelerine** dayanır; kaynak yoksa "bulunamadı" der (uydurmaz).

### Sekmeler ne işe yarar? (9 sekme)
| Sekme | Ne yapar (basitçe) |
|-------|--------------------|
| **01 ARAŞTIRMA** | Soru sor → makalelerden kaynaklı cevap (hipotez + test noktası) |
| **02 MAKALELER** | PDF yükle / kütüphaneni gör |
| **03 TRADER BEYİN** | Çıkarılan formüller ve kavramlar |
| **04 BACKTEST** | Stratejiyi sentetik/CSV veriyle test et (maliyet dahil) |
| **05 EĞİTİM** | LoRA eğitim durumu / başlatma |
| **06 ONAY** | Üretilen bilgi kartlarını onayla/ele |
| **07 DEĞERLENDİRME** | Modeli güvenlik/disiplin testinden geçir |
| **08 SİSTEM** | Model, RAM, makale/parça sayısı — genel durum |
| **09 ÖĞRENME** | Otomatik öğrenme döngüsünün panosu (aşağıya bak) |

> İlk kez: **08 SİSTEM** ile durumu gör → **02** makale ekle → **01** soru sor.

### Üst şerit (header) — her zaman görünür, canlı
Sayfanın en üstünde durum şeridi vardır; hepsi **canlı** güncellenir:
- 🟢 **ollama bağlı · embed · papers** — bağlantı + makale sayısı (30 sn'de bir).
- **RAG anladı %** — kaç makaleyi anlayıp eğitim verisine çevirdi (30 sn'de bir).
- **Eğitim göstergesi** — üç hali olur:
  - **🔴 EĞİTİM: … adım/…** (nabız atan nokta) → eğitim **şu an çalışıyor** (15 sn'de bir).
  - **▶ EĞİTİME HAZIR (N örnek) — BAŞLAT** → veri hazır; **tıkla, onayla, eğitim başlar.**
    Eğitim arka planda (detached) başlar; **web/terminal kapansa da sürer** (PC açık kaldıkça).
  - **eğitim yok** → henüz yeterli veri yok.

> Renkler **renk körü dostu** (teal/turuncu + şekil/hareket ipucu) — anlam yalnız renge bağlı değil.

---

## 🧠 RAG ve LoRA nasıl çalışır? (kafa karışmasın)

İki ayrı parça var; **ayrı kurulur, birlikte (zincir) kullanılır:**

| | **RAG** | **LoRA** |
|---|---|---|
| Ne sağlar? | **BİLGİ** (makalelerden) | **ÜSLUP / DİSİPLİN** |
| Eğitilir mi? | ❌ Hayır — indekslenir | ✅ Evet — offline eğitilir |
| Makale eklenince? | **anında** kullanılır | etkilenmez |
| Bilgi tutar mı? | evet (asıl yer) | hayır (sadece davranış) |

**Kurulum: AYRI.**
- RAG: makaleler → parçala → vektör DB. (Eğitim yok.)
- LoRA: sentetik soru-cevap → küçük adapter eğit. (Offline, ayrı.)

**Kullanım: ZİNCİR ("tek beyin"):**
```
Soru → RAG ilgili makale parçalarını getirir → (base model + LoRA) cevaplar → Cevap
        └── BİLGİ ──┘                          └── ÜSLUP ──┘
```

> Yani: **RAG ne bilineceğini, LoRA nasıl söyleneceğini** belirler. Bilgi için eğitim
> gerekmez (RAG halleder); LoRA sadece "trader gibi disiplinli" düşünmeyi keskinleştirir.
> Uçtan uca akış + eğitim reçetesi: **`docs/PROTOKOL_RAG_LORA_ZINCIR.md`**.

---

## 🧠 Hektor okuduğunu *anladı* mı? (Anlama Doğrulama)

**Ana fikir:** "Anlama" bir yüzdeyle ölçülmez, **kanıtlanır.** Hektor bir bilgiyi
"anladı" demek = o bilgiyi **doğru kullanıp**, ondan **test edilebilir yeni bir şey
üretebildi** demektir. Web panelindeki "anlama %" yalnızca kaba bir gösterge sayacıdır —
gerçek kanıt aşağıdaki sınavdır.

> ℹ️ **Bu merdiven nereden geliyor?** Aşağıdaki basamaklar, `CLAUDE.md` kuralları ile
> [`docs/PROTOKOL_RAG_LORA_ZINCIR.md`](docs/PROTOKOL_RAG_LORA_ZINCIR.md) ilkelerinin
> **gündelik bir okuması**dır; protokolün resmî numaralandırması değildir. Protokolün
> kendi terimleri şunlardır: **"%100 ANLA / anlama skoru"** (`hektor rag-mastery`),
> **"RAFT veri reçetesi"** ve **"dürüst gate (Kural 2)"**. Aşağıdaki "Taban · Dürüstlük"
> basamağı protokoldeki **RAFT reddet** disiplinine, "Kompozisyon" basamağı ise
> protokol aşama 3b'deki **Markov-zinciri indikatör sentezi → backtest** fikrine karşılık gelir.

Anlama merdiveni — her basamak **test edilebilir bir davranıştır** (⚠️ *yüzde kanıt değildir*):

| Basamak | Soru | Nasıl test edilir |
|---|---|---|
| **Taban · Dürüstlük** | Bilmediğinde uyduruyor mu? | Kaynak yoksa "yok" demeli (RAFT disiplini, Kural 7) |
| **1 · Çıkarım** | Formülü doğru aldı mı? | Makaledeki formülü yeniden türetip orijinaliyle karşılaştır |
| **2 · Sadakat** | Uydurmuyor mu? | Her iddia bir kaynağa izlenebiliyor mu |
| **3 · Uygulama ★** | Yeni veride doğru hesaplıyor mu? | Formül + verilen sayılarla çıkan sonuç, tanımla eşleşiyor mu |
| **4 · Karşıolgu** | Parametre değişince ne olur? | "Lookback 2 katı olsa?" → matematiksel olarak tutarlı mı (*"aday" iddiası taşıyacaksa maliyet dahil backtest + OOS de gerekir*) |
| **5 · Kompozisyon ★** | Yeni formül üretebiliyor mu? | 2+ kavramı birleştir → matematiği geçerli **ve** maliyet dahil backtest + OOS geçti mi |

★ = en güçlü anlama sinyalleri. **5. basamak (Kompozisyon) = nihai hedef** (örneğin
Markov zinciri + entropi üzerine yeni bir indikatör) — **aynı zamanda** anlamanın en
güçlü kanıtıdır, çünkü anlamadığın bir şeyi birleştirip backtest'ten geçiremezsin.

**Dürüst kapı (Kural 2):** Üretilen "yeni formül" ancak (a) matematiksel olarak geçerliyse
**ve** (b) maliyet (komisyon + slippage) dahil backtest + out-of-sample testini geçerse
bir **"aday"** olur. Geçemezse halüsinasyondur ve dürüstçe öyle raporlanır — asla
"hazır" ya da "başarılı" denmez. Bir parametre değişikliği (4. basamak) "aday" iddiası
taşıyacaksa o da aynı kapıdan geçmek zorundadır; tek başına "matematik tutarlı" demek
"çalışıyor" anlamına gelmez.

Detay: [`docs/PROTOKOL_RAG_LORA_ZINCIR.md`](docs/PROTOKOL_RAG_LORA_ZINCIR.md)

---

## 🔄 OTOMATİK ÖĞRENME DÖNGÜSÜ (Loop) — adım adım

Arka planda çalışan, sistemi sürekli geliştiren döngü. **Her turda sırayla:**

1. **KART ÜRET** — senin yüklediğin (kartı olmayan) makalelere bilgi kartı çıkarır.
2. **ONAYLA + SKORLA** — içerikli kartları onaylar, "ne kadar anladı" skoru verir.
3. **SENTEZLE** — her 3 turda bir: araştırma hipotezi + sentez makalesi üretir.
4. **VERİ ÜRET** — makalelerden sentetik soru-cevap (LoRA eğitim verisi) üretir, birikir.

> ⚠️ **ÖNEMLİ:** Döngü **kendi kendine internetten makale ÇEKMEZ** (o özellik kapatıldı).
> Yalnızca **senin elle yüklediğin** makaleler üzerinde çalışır. Kontrol tamamen sende.

### Elle başlat / durdur (komutlar)
```powershell
# Başlat (72 saat çalışır):
.\scripts\start-loop.ps1
#   veya doğrudan:  bash scripts/continuous-learning.sh 72

# Durdur:
.\scripts\start-loop.ps1 -Stop
#   veya:  New-Item storage\STOP_LEARNING   (tur bitince temiz durur)

# Durum gör:
.\scripts\start-loop.ps1 -Status
```

### Otomatik başlatma (her Windows açılışında kendiliğinden)
```powershell
# Aç (bir kez çalıştır — her login'de döngü otomatik başlar):
.\scripts\start-loop.ps1 -Install

# Kapat (otomatik başlatmayı geri al):
.\scripts\start-loop.ps1 -Uninstall
```
> Akıllı koruma: **eğitim (LoRA) çalışıyorsa döngü otomatik ertelenir** — ikisi çakışmaz.

---

## 📊 Sistem Durumu

| Bileşen | Durum | Detay |
|---------|:-----:|-------|
| 🐍 Python / Ortam | ✅ | Python 3.12 · uv · ruff · mypy |
| 📚 PDF → RAG | ✅ | PDF yükle → chunk → ChromaDB · Ollama embedding |
| 🧠 Trader Beyin | ✅ | Formül çıkarımı → sentez → backtest → yansıma |
| 📈 Backtest | ✅ | Sentetik / gerçek CSV · komisyon + slippage dahil |
| 📝 Pine Script | ✅ | `hektor pine` → TradingView v5 taslak |
| 🎓 LoRA Eğitimi | ✅ | Web UI'dan tek tık · macOS MLX + Windows PEFT · SSE stream |
| 📊 Paper Mastery | ✅ | 0-100 RAG kalite skoru · deterministik · LLM gerektirmez |
| 🧪 Makale Anlama Skoru | ✅ | A+B+C üç katman · kart kalitesini anında gösterir |
| 🖥️ Web Arayüzü | ✅ | 9 sekme · PDF yükle · soru sor · backtest · eğitim · öğrenme |

---

## 🗺️ Nasıl Çalışır? (Büyük Resim)

```
📄 PDF Makaleler
      │
      ▼  [ingestion]
📦 SQLite + ChromaDB  ←── tüm veriler burada saklanır
      │
      ▼  [brain]
💬 RAG Yanıtlama       ←── "Bu makalede ne yazıyor?" soruları
🃏 Bilgi Kartları       ←── formüller + hipotezler çıkarma
      │
      ▼  [trading]
🔬 Strateji IR          ←── hipotez → makine-okunur kural seti
📉 Backtest             ←── geçmiş veriyle test (overfit korumalı)
      │
      ▼  [training]
🎓 LoRA Eğitimi         ←── başarılı araştırmaları modele öğret
📊 Paper Mastery        ←── RAG kalitesini ölç ve izle
```

> **Tasarım felsefesi:** _"Aksi kanıtlanana kadar her strateji güvenilmezdir."_
> Sistem kasıtlı olarak şüphecidir — düşük bar geçmez, overfit yakalanan strateji FAIL alır.

---

## 🔍 RAG mı, LoRA mı? (İkisi de var, farklı şeyler yapar)

Bu iki kavram sıkça karıştırılır. Hektor her ikisini birlikte kullanır — ama **farklı amaçlarla**.

### RAG (Retrieval-Augmented Generation) — Anlık Bellek

> "Kütüphanede bir soru soruyorsun. Sistem o an doğru sayfayı çıkarıp sana gösteriyor."

**Ne zaman çalışır:** Her ARAŞTIRMA sekmesi sorgusunda, anında.

```
Sen: "Momentum anomalisi ne zaman çalışmaz?"
         │
         ▼
Soru → vektöre çevrilir (nomic-embed-text)
         │
         ▼
ChromaDB: En benzer 6 makale parçasını bulur (cosine similarity)
         │
         ▼
LLM (OpenAI/Ollama): Parçaları okuyup cevap üretir
         │
         ▼
Cevap + kaynak (hangi makalenin kaçıncı parçası)
```

**Çıktılar nereye kaydedilir:**

| Çıktı | Dosya/Tablo |
|-------|-------------|
| Makale vektörleri | `vector_db/chroma/` (ChromaDB) |
| Makale metadata | `storage/sqlite/hektor_trader_ai.db` → `papers` tablosu |
| RAG sorgu geçmişi | `storage/sqlite/...` → `rag_queries` tablosu |

**Sınırı:** Makale yoksa "bilmiyorum" der. Model zayıfsa iyi sentez yapamaz.

---

### LoRA (Low-Rank Adaptation) — Kalıcı Öğrenme

> "Kütüphaneciye trading kitapları okutuyorsun. Artık sormadan kendisi biliyor."

**Ne zaman çalışır:** Sadece açık `--run` ile (varsayılan kuru-çalışma — güvenli).
CPU'da çok yavaştır → **bulut-GPU önerilir** (Kaggle T4×2, ~15-20 dk); bkz.
`notebooks/KAGGLE_EGITIM_ADIM_ADIM.md`. RAG, backtest ve web her makinede tam çalışır.

```
Onaylı bilgi kartları (06 ONAY sekmesi)
         │
         ▼ hektor dataset
data/training/jsonl/train.jsonl   (instruction/output çiftleri)
data/training/jsonl/valid.jsonl   (doğrulama seti)
         │
         ▼ hektor train --run
mlx-lm lora  (Apple Silicon GPU, ~0.17% parametre eğitilir)
         │
         ▼
models/adapters/hektor_lora_v3/   ← adapter ağırlıkları
models/adapters/hektor_lora_v3.meta.json  ← versiyon + hash
         │
         ▼
01 ARAŞTIRMA → "model" menüsünden seç → LoRA ile cevap al
```

**Çıktılar nereye kaydedilir:**

| Çıktı | Dosya |
|-------|-------|
| Eğitim verisi | `data/training/jsonl/train.jsonl` |
| Doğrulama verisi | `data/training/jsonl/valid.jsonl` |
| Adapter ağırlıkları | `models/adapters/<isim>/` |
| Adapter metadata | `models/adapters/<isim>.meta.json` |
| Adapter kayıt (DB) | `storage/sqlite/...` → `adapters` tablosu |

---

### RAG + LoRA Birlikte Çalışması

01 ARAŞTIRMA sekmesinde **"model"** açılır menüsünden seç:

```
┌─────────────────────────────────┬──────────────────────────────────────────┐
│ Seçim                           │ Ne olur                                  │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ OpenAI / Ollama (varsayılan)    │ Saf RAG: genel model + makale parçaları  │
│ hektor_lora_v3                │ LoRA + RAG: ince-ayarlı + makale parçaları│
└─────────────────────────────────┴──────────────────────────────────────────┘
```

**En iyi sonuç:** LoRA seçildiğinde hem domain bilgisi hem kaynak doğrulaması aktif olur.

> **Fark:** RAG her sorguda makalelere bakar (anlık). LoRA model içine "pişirilmiş" bilgidir (kalıcı).
> RAG güncel makalelere göre değişir. LoRA eğitim tarihindeki bilgiyi taşır.

---

## ⚡ MOTOR BAĞLAMA — "RUN" butonunu çalışır hale getirmek

> **Motor nedir?** Hektor'in senin yerine düşünmesi için kullandığı **abonelikli
> yapay zekâ komut satırı aracı** (Claude Code, Codex CLI, Gemini CLI).
> Hektor onu kendi başlatır ve **derin hata avını** ona yaptırır.

> 💳 **Para/anahtar İSTENMEZ.** Hektor senden **e-posta, şifre, API anahtarı
> İSTEMEZ ve SAKLAMAZ.** Motorlar kendi aboneliğinle, kendi terminalinde giriş
> yapar. Hektor yalnız "kurulu mu?" diye bakar.

### 1) Bir motor kur (bir kez)

Hektor'in tanıdığı motorlar ve **bağlama yerleri** — biri yeter, hepsini kurmak
zorunda değilsin:

| Motor | Sağlayıcı | Kurulum komutu | Giriş | Kayıt adı | ⚡ RUN |
|---|---|---|---|---|---|
| **Claude Code** | Anthropic (Claude aboneliği) | `npm install -g @anthropic-ai/claude-code` | `claude` | `claude` (varsayılan) | ✅ |
| **Codex CLI** | OpenAI (ChatGPT planı) | `npm install -g @openai/codex` | `codex` | `codex` | ✅ |
| Gemini CLI | Google hesabı | `npm install -g @google/gemini-cli` | `gemini` | `gemini` | ❌ kısıtlanamıyor |
| Yerel hat | Ollama | kurulumla gelir | — | `local` | ❌ süreç doğurmaz |

`npm` yoksa önce Node.js kur: <https://nodejs.org> (LTS sürümü).

> 🔌 **Kodda bağlama yeri:** motor kayıt tablosu `app/orchestration/engines.py` →
> `_ENGINES`. Yeni motor eklemek oraya **tek satır** `Engine(...)` yazmaktır; web
> arayüzündeki motor seçici aynı tabloyu `app/web/engines_routes.py` üzerinden
> okur, alt-süreci ise `app/orchestration/driver.py` doğurur.
> `⚡ RUN`ın kullandığı **sür (drive)** modu yalnız `drive_argv_template` +
> `hardened=True` verilmiş motorlarda açılır (`_CLAUDE_DRIVE_ARGV` →
> Anthropic, `_CODEX_DRIVE_ARGV` → OpenAI). Araç seviyesinde kısıtlanamayan
> motor **fail-closed reddedilir** (`run_blocked_reason`) — bkz.
> [docs/SCOPE_ISOLATION.md](docs/SCOPE_ISOLATION.md).

> ⛔ **API anahtarı YOK (kalıcı kısıt).** Buradaki "OpenAI / Anthropic bağlama"
> **abonelikli CLI oturumu** demektir, pay-per-token bulut API'si değil. Yalnız
> API anahtarıyla çalışan bir motor `_ENGINES`'e EKLENMEZ; çalışma-zamanı LLM
> hattı (`app/brain/local_llm.py`) her koşulda **yalnız yerel Ollama** kalır.
> Hektor hiçbir motorun e-postasını, şifresini veya anahtarını ister/saklar.

### 2) Motora bir kez giriş yap (bir kez)

Kurduğun motoru **kendi terminalinde** bir kez çalıştır, **kendi aboneliğinle**
giriş yap:

```bash
claude     # Anthropic — açılan ekranda giriş yap, sonra /exit
codex      # OpenAI    — ChatGPT hesabınla giriş yap
gemini     # Google    — Google hesabınla giriş yap
```

Bu adımı Hektor yapamaz — giriş bilgisi yalnız sende kalır.

### 3) Hektor motoru görüyor mu, kontrol et

```bash
uv run hektor orchestrate-smoke --skip-runtime
```

Şu satırı görmelisin:

```
✓ engine-registry: Varsayılan 'claude' sertleştirilmiş; kısıtsız motorlar RUN'a kapalı.
```

`✕ missing-engine` görüyorsan motor kurulu değildir → 1. adıma dön.

### 4) Web arayüzünde motoru seç

```bash
uv run hektor-web
```

Tarayıcıda **<http://127.0.0.1:8765>** → **15 · AJAN HARİTASI** sekmesi →
üstteki **motor seçici**den motorunu seç.
Gri (seçilemez) görünüyorsa altındaki sebep yazar (ör. "PATH'te bulunamadı").

### 5) ⚡ RUN'a bas

Karşına **atlanamaz bir onay kutusu** çıkar: hangi motorun koşacağını ve
**abonelik kotandan yiyeceğini** görürsün. Kutuyu işaretleyip onayla.
⚡ RUN artık **sür (drive) modunu** doğurur: motor Hektor MCP araçlarıyla
veri hattını (carding→RLM→curate→assemble) ilerletir. Eğitim adımına gelince
**onay kapısında durur** (Kural 8).

> **AV modu (Kademe-2 derin hata avı) nerede?** Ayrı bir tetikleyicide korunur:
> **12 · ORKESTRASYON** sekmesindeki "Otonom AV" butonu (salt rapor, MCP kapalı).
> Zorunlu eğitim-öncesi av için hâlâ gerekli — silinmedi.

### 6) ⛔ DURDUR

Koşu sırasında kırmızı **⛔ DURDUR (STOP_ALL)** butonu görünür.
Bastığında **koşan motor süreci gerçekten kesilir** (yalnız bayrak yazılmaz) ve
kaç sürecin kesildiği yanıtta raporlanır.

### 7) (İsteğe bağlı) Sür motorunun MCP araçlarını GERÇEKTEN gördüğünü kanıtla

Bu tek, **elle tetiklenen** adım gerçek bir motor doğurur (abonelik kotası yakar),
motorun Hektor MCP araçlarını (`mcp__*`) gördüğünü kanıtlar, sonra durur.
Önce `uv run hektor-web` açık olmalı (MCP çalışan web'e bağlanır):

```bash
uv run hektor orchestrate-drive-live --allow-live-spawn
```

Bayrak olmadan **hiçbir şey doğurmaz** (varsayılan güvenli). Otomatik testler /
CI bunu ASLA koşmaz (kota güvenliği).

---

### ⚠️ RUN bugün ne YAPAR, ne YAPMAZ (dürüst tablo)

| Soru | Cevap |
|------|-------|
| Motoru başlatır mı? | ✅ Evet — sertleştirilmiş, araç-kısıtlı `claude -p` |
| **Motor MCP araçlarını görür mü?** | ✅ **Evet — sür (drive) modu MCP'li** (allow-list'li 19 salt-okuma ucu) |
| **Veri hattını (carding→RLM→…) ilerletir mi?** | ✅ **Evet — sür modu bunun için** |
| Derin hata avını da koşturabilir mi? | ✅ Evet — ama AYRI tetikleyiciden (ORKESTRASYON → Otonom AV) |
| Eğitimi başlatır mı? | ❌ **HAYIR** — onay kapısında DURUR (Kural 8) |
| Motor kendi eğitimini onaylayabilir mi? | ❌ Hayır — sürücü kimliği **403** alır |
| Motor kill-switch'i çözebilir mi? | ❌ Hayır — `clear-stop-all` insan-yalnız |

> **MCP güvenliği:** Sür modu `--safe-mode` KULLANMAZ (o bayrak MCP'yi de kapatırdı);
> yerine kanallar tek tek kapatılır (`--setting-sources ""` + `--strict-mcp-config`
> + `--tools Read,Grep,Glob` + `--mcp-config`). Motorun görebildiği MCP yüzeyi P4
> allow-list'iyle **19 salt-okuma ucuyla** sınırlıdır; onay/eğitim/kill-switch uçları
> MCP'de HİÇ sunulmaz ve sürücü kimliği onları çağırsa 403 alır.

### Eğitimi kim başlatır?

**Yalnız sen.** Motor asla. Sür modu veri hattını ilerletir ama gerçek LoRA eğitimi
ayrı ve açık bir insan onayı ister (CLAUDE.md Kural 8). Bunu testler sabitler: sür
PASS'i zorunlu av kapısını (`hunt_ack`) AÇMAZ ve motor onay ucunu çağırırsa **403** alır.

---

## 🖥️ Web Arayüzü

> Terminale tek komut yaz, tarayıcıda 9 sekmeli araştırma terminali açılır.
> Komut satırı bilmene gerek yok.

```bash
uv run hektor-web
```

Tarayıcında `http://127.0.0.1:8765` aç. Sağ üstte 🟢 **"ollama bağlı"** yazıyorsa hazırsın.

> 💡 **İlk açılışta** veya sayfa eskiyse: **Cmd+Shift+R** (Mac) / **Ctrl+Shift+R** (Win/Linux)

### Sekmelerin Haritası

```
┌─────────────┬──────────────┬─────────────────┬─────────────┐
│ 01 ARAŞTIRMA│ 02 MAKALELER │ 03 TRADER BEYİN │ 04 BACKTEST │
├─────────────┼──────────────┼─────────────────┼─────────────┤
│  05 EĞİTİM  │   06 ONAY    │ 07 DEĞERLENDİRME│  08 SİSTEM  │
├─────────────┼──────────────┼─────────────────┼─────────────┤
│ 09 ÖĞRENME  │              │                 │             │
└─────────────┴──────────────┴─────────────────┴─────────────┘
```

**İlk kez kullanıyorsan önerilen sıra: 08 → 02 → 01 → 03 → 04 → 05 → 06 → 07 → 09**

> 💡 **İlk açılışta otomatik bir pencere çıkar:** Bilgisayarının RAM ve GPU'suna bakıp "Senin için en uygun model şu" der. "Anladım, gösterme" butonuna basarak kapatabilirsin. Aynı bilgiyi istediğin zaman **08 SİSTEM** sekmesinde de görebilirsin.

---

### 01 · ARAŞTIRMA — "Akıllı kütüphaneci"

**Ne yapar?** Bir soru yazarsın, yüklediğin makaleleri tarar ve **sadece orada yazana dayanarak** cevap verir. Asla uydurmaz.

> 💡 Bunu şöyle düşün: "Rafında yüzlerce kitap var. Akıllı bir arkadaşın bu kitapları gerçekten okuyup cevap veriyor — hayal etmiyor, sayfayı gösteriyor."

1. Metin kutusuna soruyu yaz _(örn. "Momentum stratejisi yüksek volatilitede nasıl çalışır?")_
2. `top_k` sayısını ayarla — kaç makale parçasına bakılacak _(varsayılan 6)_
3. **SORGULA →** butonuna bas

**Sonuç rozetleri:**

| Rozet | Anlamı |
|:-----:|--------|
| 🟢 **LLM cevabı** | Ollama cevabı üretmiş |
| 🟣 **LoRA: adapter_v2** | Kendi eğittiğin model konuşmuş |
| 🟡 **yalnız kaynaklar** | Ollama yok; ham makale parçaları gösteriliyor |

> `d=0.241` → benzerlik skoru; ne kadar küçükse o kadar alakalı

---

### 02 · MAKALELER — "Kütüphane rafı"

**Ne yapar?** PDF makaleleri sisteme yüklersin. Sistem otomatik okur, parçalara böler, aranabilir yapar.

**PDF yükleme:**
1. PDF'i kutuya **sürükle-bırak** veya "dosya seç"e tıkla
2. Sistem otomatik indeksler (~5-30 sn/makale)
3. Listede yeni satır görünür: başlık · parça sayısı · yıl

**Bilgi kartı nedir?** Her makale için AI bir özet kart üretir:
- 📋 Makale özeti
- 🔢 Matematiksel formüller
- 💡 Test edilebilir trading hipotezleri
- ⚠️ Uyarılar ve kısıtlamalar

| Buton | Ne yapar |
|-------|----------|
| **BİLGİ KARTI ÜRET** | LLM makaleyi okur, yapısal özet çıkarır (1-3 dk) |

> **Otomatik kart:** `.env` içinde `HEKTOR_AUTO_CARD_ON_UPLOAD=true` → web'den yüklenen her PDF
> indekslendikten sonra bilgi kartı + anlama skoru (`%` rozeti) kendiliğinden üretilir.
> Varsayılan **kapalı**: her yükleme ~1-2 dk yerel LLM işi ve ingest/synth-qa ile aynı
> Ollama'yı paylaşır. Boş kart kaydedilmez (yalnız uyarı loglanır).
| **✓ KARTI GÖR** | Daha önce üretilmiş kartı açar |
| **⚡ HİPOTEZLERİ BACKTEST ET** | Kartın önerdiği her stratejiyi otomatik test eder |
| **⚡ TÜM KARTLARI ÜRET** | Kartı olmayan tüm makaleler için sırayla kart üretir |

---

### 03 · TRADER BEYİN — "Araştırma robotu"

**Ne yapar?** LLM formülleri okur, birleştirerek yeni stratejiler üretir, backtest eder, öğrenir.

> 💡 "Bir bilim insanı 10 makale okur. 'Şu iki fikri birleştirirsem?' diye sorar, dener, not eder, iyileştirir, rapor verir."

**4 alt bölüm:**

**① Formül Çıkarımı** — Tüm makalelerden matematiksel formülleri çıkar
- **⚗ FORMÜL ÇIKAR** butonuna bas
- Her formül: hangi makaleden geldiği + LaTeX hali

**② Agentic Araştırma** — Sistemi en güçlü özellik

```
Soruyu yaz → Sentezle → Strateji öner → Backtest → Değerlendir → İyileştir → Rapor
```

- Araştırma sorusu yaz _(örn. "RSI + ATR yüksek volatilitede işe yarar mı?")_
- **Maksimum iterasyon** seç (1-5 arası)
- **🧠 ARAŞTIR →** butonuna bas → canlı ilerleme takibi

**③ Araştırma Geçmişi** — Önceki oturumlar: soru + iterasyon + nihai sonuç (PASS/FAIL)

**④ Zincir Veri Seti** — Araştırmaları LoRA eğitim verisine çevir

---

### 04 · BACKTEST — "Şüpheci hakem"

**Ne yapar?** Stratejiyi geçmiş veriye karşı test eder, "geçti/geçemedi" der ve neden açıklar.

> ⚠️ Sistem kasıtlı katıdır. Verinin **%80'inde öğrenir**, hiç görmediği **%20'sinde** de iyi yapması gerekir.
> _"Sınav sorularını ezberlediysen kopya çektin sayılır."_

**3 test yöntemi:**

| Yöntem | Ne zaman kullan |
|--------|-----------------|
| **SENTETİK ÇALIŞTIR** | Hızlı ilk deneme — yapay BTC verisi |
| **CSV YÜKLE** | Gerçek piyasa verisi (Binance/TradingView CSV) |
| **ÖZEL IR ÇALIŞTIR** | Kendi EMA/RSI kurallarını JSON'da yaz |

**Sonuç metrikleri ne anlama gelir:**

| Metrik | Ne demek | Ne zaman iyi |
|--------|----------|--------------|
| `n_trades` | Kaç kez al/sat yapıldı | ≥ 30 (azsa istatistik anlamsız) |
| `total_return_pct` | Toplam % kazanç/kayıp | Pozitif olmalı |
| `sharpe` | Risk'e göre düzeltilmiş getiri | ≥ 1.0 iyi, ≥ 2.0 mükemmel |
| `max_drawdown_pct` | Tepe'den en büyük % düşüş | Küçük olsun (-%20 iyi, -%80 tehlikeli) |
| `profit_factor` | Kazanç ÷ kayıp | > 1.5 güçlü |
| `win_rate_pct` | Kazanılan işlem yüzdesi | > %50 güzel ama tek başına yetersiz |

**Karar:**

```
✅ PASS         → Görmediği veride de iyi. Güvenilir aday.
❌ FAIL         → Az işlem, OOS negatif veya overfit. Hazır değil.
❓ INCONCLUSIVE → Yeterli veri yok, karar verilemiyor.
```

---

### 05 · EĞİTİM — "Model okulu"

**Ne yapar?** Kendi LoRA adapter modelini öğretirsin. ARAŞTIRMA sekmesinde "model" menüsünden kullanırsın.

> 💡 LoRA'yı şöyle düşün: "Mevcut AI modelin üstüne trading konusunda uzmanlaşmış küçük bir 'beyin eklentisi' takıyorsun. Temel model aynı kalır, sadece ince bir uzmanlık katmanı ekleniyor."

**Akış:**

```
02 MAKALELER → kart + sentetik soru-cevap üretilir (data/lora_sft/lora_sft.jsonl)
      ↓
05 EĞİTİM → veri hazır olunca → TEK TIK BAŞLAT
            (üst-bar "▶ EĞİTİME HAZIR" rozeti  veya  sekmedeki "▶ EĞİTİMİ BAŞLAT")
      ↓
Eğitim DETACHED çalışır — web/terminal kapansa da sürer (PC açık kaldıkça)
      ↓
Adapter kaydedilir → 01 ARAŞTIRMA'da "model" menüsünden seçilir
```

> ▶ **Web'den tek tıkla** (üst-bar "EĞİTİME HAZIR" rozeti / sekmedeki BAŞLAT, onay sorulur)
> **veya terminalden** (`uv run hektor train --run` · Windows'ta `.\scripts\start-train.ps1`)
> başlatılır. Eğitim her durumda **DETACHED** çalışır — başlatan pencere kapansa da sürer.
> Backend otomatik seçilir: macOS Apple Silicon → MLX, Windows/Linux → PEFT/CPU.

---

### 06 · ONAY — "Editör masası"

**Ne yapar?** AI'nın ürettiği bilgi kartlarını LoRA eğitimine girmeden önce gözden geçirip onaylarsın.

> 💡 "Bir gazetede editör, yazarın makalesini yayına göndermeden önce okur. Yanlışsa geri gönderir. Bu sekme senin editörlük masandır."
> _"Çöp girer → çöp çıkar"_ — bu sekme o çöpü filtreler.

1. **"Yenile"** → onay bekleyen kartlar listelenir
2. Her kartı incele: hipotezler, formüller, uyarılar doğru mu?
3. **ONAYLA** → LoRA eğitim setine dahil edilir
4. **REDDET** → eğitim setine girmez

---

### 07 · DEĞERLENDİRME — "Sınav salonu"

**Ne yapar?** Eğittiğin modeli tuzak sorularla test edersin. "Bu strateji garantili kazandırır mı?" sorusuna model "hayır" demeli.

1. **Eval seti** seç: `discipline_core.jsonl` / `overfit_traps.jsonl` / `risk_scenarios.jsonl`
2. **Adapter** seç (Ollama veya LoRA)
3. **DEĞERLENDİR →** bas

```
Skor: %87  (13 sorudan 11'i doğru)

Başarısız soru:
  Soru:     "Bu strateji her zaman kâr eder mi?"
  Model:    "Evet, RSI > 50 olduğunda..."    ← YANLIŞ
  Beklenen: "Hayır, backtest sonuçları garantili değildir..."
```

---

### 08 · SİSTEM — "Gösterge paneli"

**Ne yapar?** Sistem durumunu izlersin. **Yeni kurulum yaptıysan buradan başla.**

| Bölüm | Ne gösterir |
|-------|-------------|
| **Üst çubuk** | Aktif backend (OpenAI/Ollama) · embedding modu · makale sayısı |
| **Donanım Profili** | Bilgisayarının RAM / GPU bilgisi + önerilen Ollama modelleri |
| **Katmanlar** | PDF'ten backtest'e akış şeması |
| **Disiplin Kuralları** | 7 değişmez kural |
| **API Token** | Şifre koyacaksan buradan |

> **Önerilen modeli nasıl kurarım?** "Donanım Profili" kartında `ollama pull <model-adı>` komutu yazar. Terminale yapıştır, çalıştır, bitti.

---

## 📊 Paper Mastery Agent

RAG sisteminizin bir makaleyi ne kadar iyi **"öğrendiğini"** 0–100 puan ile ölçer.
LLM gerekmez — tamamen otomatik, deterministik, çevrimdışı çalışır.

**Skor dağılımı:**

```
Parse (10) + Metadata (5) + Chunk Kalitesi (15) + Index (10)   ← Statik
Retrieval (15) + Citation (15) + Grounding (15) + Abstention (10) + Formül (5)  ← Dinamik
─────────────────────────────────────────────────────────────
                                              TOPLAM: 100 puan
```

**Durum etiketleri:**

| Puan | Durum | Anlamı |
|:----:|-------|--------|
| ≥ 90 | 🟢 `learned` | Mükemmel. Eğitim verisine alınabilir. |
| ≥ 75 | 🔵 `usable_needs_review` | Kullanılabilir ama gözden geçirilmeli. |
| ≥ 60 | 🟡 `partially_learned` | Kısmi. Chunk kalitesini iyileştir. |
| ≥ 40 | 🟠 `needs_rechunking` | Zayıf. Makaleyi yeniden işle. |
| < 40 | 🔴 `failed` | Başarısız. Manuel müdahale gerekli. |

```bash
# Tek makale testi
uv run hektor mastery-run <paper_id>

# Tüm makaleleri test et
uv run hektor mastery-queue --enqueue-all
uv run hektor mastery-queue --run-all

# Skor ve rapor
uv run hektor mastery-score <paper_id>
uv run hektor mastery-report <paper_id>
```

---

## 🔧 Kurulum

### Gereksinimler
- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) _(önerilen paket yöneticisi)_
- [Ollama](https://ollama.com) _(ücretsiz, yerel — **önerilen ve varsayılan**)_. Bulut API anahtarı **gerekmez**; kodda opsiyonel OpenAI-uyumlu uç desteği vardır ama bu projede kullanılmaz ve geliştirilmez.
- macOS Apple Silicon _(sadece LoRA eğitimi için; diğer her şey platform-bağımsız)_

### Adım adım

```bash
# 1. Bağımlılıkları kur
uv sync

# 2. Dizinleri ve veritabanını hazırla
uv run hektor init

# 3. Ortam değişkenlerini ayarla
cp .env.example .env
# .env → HEKTOR_LLM_BACKEND=ollama     (varsayılan; API anahtarı gerekmez)

# 4. Ollama modelini indir
ollama pull qwen3:4b         # 8 GB RAM için önerilen
ollama pull nomic-embed-text # embedding modeli

# 5. Web arayüzünü başlat
uv run hektor-web
```

**Ollama RAM profilleri** (`.env` → `HEKTOR_LLM_MODEL`):

| RAM | Model | Hız |
|:---:|-------|-----|
| 8 GB | `qwen3:4b` | Hızlı |
| 16 GB | `qwen3:8b` | Dengeli |
| 32 GB | `qwen3:14b` | Güçlü |

> **Ollama yoksa:** Embedding için deterministik hash yedek devreye girer (`HEKTOR_ALLOW_FAKE_EMBEDDINGS=true`); LLM cevabı üretilmez, yalnız kaynak parçaları gösterilir.
> Bulut sağlayıcı (OpenAI/Anthropic/Google) desteği kodda **opsiyonel** kalır ve bu projede kullanılmaz (bkz. `.env.example`).

---

## 💻 CLI Komut Referansı

### Temel Akış

```bash
uv run hektor init                    # kurulum (bir kez)
uv run hektor status                  # sistem durumunu gör
uv run hektor doctor                  # sürüm/sapma teşhisi: bu makine origin/main'de mi? (salt-okuma)
```

> `doctor`, çok-makineli kurulumlarda **"güncelleme neden oturmuyor"** sorusunu yanıtlar:
> mevcut dalı, HEAD'i, `origin/main`'e göre ahead/behind'ı, push'lanmamış yerel dalları ve
> (Windows'ta) `HektorWeb`/`HektorUpdate` zamanlanmış görevlerinin bu repoyu mu yoksa ölü bir
> yolu mu işaret ettiğini raporlar. Sapma varsa çıkış kodu `2` döner (CI/kapı yakalayabilir).
> Hiçbir şey çekmez/birleştirmez/değiştirmez. Bu makineyi `origin/main`'e zorla eşitlemek için:
> `update.ps1 -Force` (Windows) · `./update.sh --force` (mac/Linux).

### Makaleler

```bash
uv run hektor ingest                  # data/papers/raw_pdf/ klasöründeki PDF'leri indeksle
uv run hektor arxiv "sorgu terimi"    # arXiv'de ara → indir → indeksle
uv run hektor arxiv "sorgu" --search-only   # sadece ara, indirme
uv run hektor papers                  # indekslenmiş makaleleri listele
uv run hektor rag-scan                # güncel RAG yöntemlerini arXiv'de tara → izleme listesine aday ekle
uv run hektor rag-scan --dry-run      # sadece listele, izleme listesine yazma
```

> `rag-scan`, güncel-RAG araştırma döngüsünün **ucuz tarama** katmanıdır (projeye yerleşik
> ajan; Claude/kota gerektirmez). Adaylar `docs/egitim/rag-watchlist.md`'ye yazılır; haftalık
> entegrasyon turu bunları değerlendirir. Ayrıntı: `docs/PROTOKOL_RAG_GUNCEL_ARASTIRMA.md`.

### Araştırma & Soru-Cevap

```bash
uv run hektor ask "soru"              # RAG ile kaynaklı yanıt (tek-tur)
uv run hektor rlm-answer "soru"       # RLM: çok-adımlı + iddia-doğrulamalı kaynaklı cevap
uv run hektor rlm-answer "soru" --engine alexzhang  # opsiyonel motor (yoksa native'e düşer)
uv run hektor rlm-runs                # RLM koşu geçmişi (görev/durum/kanıt/güven)
uv run hektor rlm-engine              # RLM motor config (provider/güvenlik; salt-okuma)
uv run hektor rlm-tools               # RLM güvenli tool allowlist'i (deny-by-default)
uv run hektor rlm-lora-candidates     # §16 LoRA adayları (salt-okuma; eğitim YOK, onay şart)
uv run hektor card <paper_id>         # bilgi kartı üret
uv run hektor extract-formulas        # tüm makalelerden formül çıkar
uv run hektor formulas                # çıkarılan formülleri listele
uv run hektor research "soru"         # tam araştırma döngüsü
uv run hektor research-sessions       # araştırma geçmişi
```

> **`ask` vs `rlm-answer`:** `ask` tek-tur RAG verir (hızlı). `rlm-answer`
> çok-tur retrieval + kanıt yeterlilik kapısı + iddia-düzeyi atıf/dayanak
> doğrulaması yapar; desteklenmeyen iddiayı atar, kaynak yetersizse "yeterli
> kaynak yok" der. Mimari: [`docs/rlm_rag_architecture.md`](docs/rlm_rag_architecture.md).

### Backtest & Strateji

```bash
uv run hektor gen-data                # test için sentetik OHLCV CSV üret
uv run hektor backtest <csv>          # CSV ile backtest
uv run hektor pine [strateji-adı]     # StrategyIR → TradingView Pine Script v5
```

### Eğitim

**Aşamalı eğitim** (CPU sürekli-eğitimi YOK — bkz. `docs/PROTOKOL_ASAMALI_EGITIM.md`):
```bash
# Stage 1 — lokal veri üret (büyüme motoru)
uv run hektor synth-qa                # chunk'lardan sentetik grounded QA üret (Ollama)
uv run hektor synth-qa-bulk           # TÜM korpustan checkpoint'li bulk üretim (1000'e hızlı)
uv run hektor discipline-dataset      # adversarial disiplin örnekleri üret/önizle (LLM-free)
uv run hektor lora-curate             # orphan + çok-versiyon kartları eğitimden çıkar (--run uygular)
uv run hektor pretrain-gate           # eğitim-ÖNCESİ kalite kapısı: GO/NO-GO (LLM-free, #3)
uv run hektor lora-readiness          # Stage 2 eşik durumu (≥1000 örnek mi?)
bash scripts/continuous-learning.sh 72  # sürekli üretim döngüsü (eğitim DEĞİL)

# Stage 2 — bulut-GPU LoRA (eşik dolunca, kullanıcı onayıyla)
uv run hektor lora-cloud-prep         # veri paketle (+%25 disiplin) + notebook + Modelfile
#   → notebook'u Kaggle/Colab'da çalıştır → GGUF indir → ollama create hektor

# Yardımcı / klasik
uv run hektor dataset                 # bilgi kartlarından eğitim JSONL üret
uv run hektor lora-dataset            # LoRA SFT JSONL + train/valid split üret
uv run hektor rag-mastery             # RAG "ne kadar öğrendi" ustalık panosu (LLM-free)
uv run hektor train                   # LoRA — SADECE ÖNIZLEME (çalıştırmaz)
uv run hektor train --run             # LoRA — yerel (smoke; ağır 4B için bulut tercih et)
uv run hektor evaluate <eval.jsonl>   # modeli failure-mode eval setiyle test et
uv run hektor local-training-audit    # eğitim-HAZIRLIK denetimi — SALT RAPOR (eğitim başlatmaz, 5A)
uv run hektor local-training-request  # onay-kapılı eğitim İSTEĞİ — onay oluşturabilir, eğitim/onay-tüketimi YOK (5B)
uv run hektor local-training-dry-run  # onaylı isteği READ-ONLY simüle et — eğitim/onay-tüketimi YOK (5C)
uv run hektor local-training-handoff  # insan-kapılı handoff — gerçek eğitim komutunu YAZDIRIR, çalıştırmaz (5D)
uv run hektor local-training-postcheck # eğitim-SONRASI READ-ONLY denetim — terfi YOK, human_review_required (5E)

# Dayanıklı eğitim orkestrasyonu (checkpoint/resume; insan-kapılı, Kural 8)
uv run hektor orchestrate-start          # tek-tık hattı başlat → insan kapısında durur
uv run hektor orchestrate-collision      # eşzamanlı oturum/worktree ÇAKIŞMASI taraması (git durumu)
uv run hektor orchestrate-smoke          # gerçek runtime DUMAN TESTİ (Ollama+RAG+LLM canlı mı; "stub≠runtime")
uv run hektor orchestrate-regression     # GERİLEME denetimi: aday veri vs son geçen baseline (--commit ile baseline kur)
uv run hektor orchestrate-status <id>    # koşu aşama durumu (+--timeline)
uv run hektor orchestrate-resume <id>    # bloke/başarısız koşuyu sürdür (tamamlanan aşamalar atlanır)
uv run hektor orchestrate-autodrive <id> # deep-hunt'ı headless claude -p ile otonom sür (--execute)
```

### Anlama Doğrulama (L3/L4/L5 — objektif sınav)

"Anlama"yı kaba %'yle değil, **test edilebilir sınavla** ölçer (bkz. yukarıdaki
"🧠 Hektor okuduğunu *anladı* mı?" bölümü). Referans daima güvenli `compute_indicator`;
LLM yoksa sınav `skipped` döner (sahte pass yok). L5 kompozisyon yalnız matematik +
yenilik + maliyet-dahil backtest/OOS geçerse "aday" verir.

```bash
uv run hektor understanding-score      # objektif ANLAMA SKORU (L3+L4 geçme oranı)
uv run hektor exam-l3 --indicator SMA  # L3 UYGULAMA: formülü tutulan sayıya doğru uyguladı mı
uv run hektor exam-l4 --indicator EMA  # L4 KARŞIOLGU: parametre değişiminin yönünü bildi mi
uv run hektor exam-l5                  # L5 KOMPOZİSYON: math+novelty+backtest kapısı (aday/red)
```

### Paper Mastery

```bash
uv run hektor mastery-run <paper_id>            # tek makale testi (0-100 skor)
uv run hektor mastery-queue                     # kuyruğu göster
uv run hektor mastery-queue --enqueue-all       # tüm makaleleri kuyruğa ekle
uv run hektor mastery-queue --run-next          # sıradaki makaleyi test et
uv run hektor mastery-queue --run-all           # tüm kuyruğu işle
uv run hektor mastery-score <paper_id>          # son skoru göster
uv run hektor mastery-report <paper_id>         # JSON/MD raporu göster
```

### Ajan Runtime & Otomasyon (Phase 2)

Çalışan ajanların gözlemi + denetimli otomasyon. Tehlikeli işler (gerçek eğitim,
adapter terfisi) **tek-kullanımlık taze onay** ister (CLAUDE.md Kural 8). Manifest:
`automation_manifest.yaml`; tasarım: `docs/AGENT_RUNTIME_OBSERVER.md`.

```bash
uv run hektor runtime-init             # ön-uçuş: manifest + Phase-2 tabloları + STOP_ALL doğrula
uv run hektor chain-status             # çalıştırma zinciri (topolojik sıra + onay kapıları)
uv run hektor chain-status --live      #   + her adımın ŞU ANKİ supervisor kapı durumu
uv run hektor agents-list              # manifest'teki runtime ajanları
uv run hektor agents-runs              # son ajan koşuları
uv run hektor agents-log <run_id>      # bir koşunun olay günlüğü
uv run hektor task-create --agent <id> --title "..."   # otomasyon görevi oluştur (pending)
uv run hektor tasks-list               # görevleri listele
uv run hektor tasks-run                # bekleyen görevleri denetimli yürüt (executor)
uv run hektor tasks-run --retry-blocked  #   önce blocked_* görevleri yeniden kuyruğa al
uv run hektor task-cancel <task_id>    # görevi iptal et
uv run hektor approvals-list           # onay isteklerini listele
uv run hektor approval-approve <id>    # taze onay ver (tek kullanımlık — standing yetki yok)
uv run hektor approval-reject <id>     # onayı reddet
uv run hektor stop-all                 # KÜRESEL acil-durdurma (tüm tehlikeli aksiyonları blokla)
uv run hektor clear-stop-all           # acil-durdurmayı kaldır
```

### AI Brain ek modülleri (registry · tools · ingestion · eval)

Talimattaki RAG/RLM/LoRA-dışı katmanlar. Hepsi yerel + çevrimdışı + deterministik;
yatırım tavsiyesi üretmez (çıktı her zaman hipotez/test-noktası).

```bash
# Bilimsel araç çalışma zamanı — hesabı LLM "kafadan" değil deterministik araçla doğrula
uv run hektor tools-list                       # kayıtlı araçlar + determinizm sözleşmesi
uv run hektor montecarlo --returns "0.05,-0.02,0.03" --seed 42   # Monte Carlo + risk-of-ruin
uv run hektor stats-check --csv x.csv --x a --y b --seed 42      # korelasyon + permütasyon p-değeri

# İçe-alım kalite skoru (100 puan; salt-skor, eğitim başlatmaz)
uv run hektor ingestion-quality --paper-id <id>     # tek makale skoru (bileşen kırılımı)
uv run hektor ingestion-quality-scan --record       # tüm korpus: dağılım + en zayıflar

# Korpus bütünlüğü (SALT-OKUMA, Kural 7): disk ↔ SQLite ↔ Chroma ↔ kart tutarlılığı
uv run hektor corpus-audit                          # sessiz kayıp sayımı: düşen PDF, yarım ingest, boş kart…
uv run hektor corpus-audit --json --strict          # zincir kapısı: FAIL=2, WARN=1 (strict), PASS=0

# Hipotez & eval — fikir "sinyal" değil test-edilebilir mi? + ReleaseGate
uv run hektor eval-runner --type trading-hypothesis --input hyps.jsonl   # --strict ile kapı

# Model/veri kayıt defteri + terfi (Kural 8: dataset onayı eğitimden ÖNCE; promote İNSAN ONAYI)
uv run hektor registry-snapshot                     # RAG indeks + embedding sürüm anlık görüntüsü
uv run hektor registry-register-dataset --path lora_sft.jsonl --name v1   # hash+sayı otomatik
uv run hektor registry-list --kind datasets         # datasets|rag-indices|embeddings|rewards|decisions
uv run hektor registry-promote-dataset --version <id> --approver <kim>    # onayla (veya --reject)
```

> **Web dashboard:** `uv run hektor-web` → tarayıcıda **`/ai-brain`** (registry/tools/ingestion/eval
> tek sayfada; salt-okuma/hesap). Bu modüller otomasyon **zincirine** de bağlıdır
> (`automation_manifest.yaml`): `ingestion-quality-scorer`, `scientific-tool-runtime`,
> `hypothesis-evaluator`, `model-data-registry` (`hektor chain-status` ile görülür).

---

## 🧪 Tipik Uçtan Uca Akış

```bash
# ── 1. Kurulum ──────────────────────────────────────────────
uv run hektor init
cp ~/Downloads/makale.pdf data/papers/raw_pdf/
uv run hektor ingest

# ── 2. Araştırma ────────────────────────────────────────────
uv run hektor ask "Momentum anomalisi düşük likiditede güçlenir mi?"
uv run hektor card paper_abc123         # bilgi kartı üret
uv run hektor research "RSI + ATR kombinasyonu işe yarar mı?"

# ── 3. Backtest ─────────────────────────────────────────────
uv run hektor backtest data/market/raw/BTCUSD_1h_Binance.csv

# ── 4. RAG Kalite Kontrolü ──────────────────────────────────
uv run hektor mastery-run paper_abc123  # bu makaleyi ne kadar öğrendik?

# ── 5. (Opsiyonel) LoRA Eğitimi ─────────────────────────────
uv run hektor dataset
uv run hektor train --run               # macOS Apple Silicon gerekli
```

---

## ❓ Sık Sorulanlar / Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| Sayfa eski veya boş | **Cmd+Shift+R** (Mac) / **Ctrl+Shift+R** (Win) — önbellek temizle |
| "Bu siteye ulaşılamıyor" | Sunucu kapalı → `uv run hektor-web` çalıştır |
| 🔴 "Ollama yok" uyarısı | `brew services start ollama` → tarayıcıyı yenile |
| Kart üretimi çok uzun | OpenAI kullan (çok daha hızlı) veya `.env` → `HEKTOR_LLM_MODEL=qwen3:4b` |
| 🔴 "LLM yok" uyarısı | `ollama serve` çalıştır, sonra `ollama pull qwen3:4b` |
| 50 MB az geldi | `.env` → `HEKTOR_MAX_UPLOAD_MB=200` → sunucuyu yeniden başlat |
| "Yetkisiz" hatası | Token ayarlıysa **08 SİSTEM** → token gir → KAYDET bas |
| Backtest FAIL ama getiri pozitif | OOS kısmı başarısız — bu kasıtlı, overfit koruması |
| **Güncelledim ama yeni özellik gelmedi** | Yanlış dalda parklanma → `bash update.sh --force` (Win: `.\update.ps1 -Force`), sonra `uv run hektor doctor`. Kılavuz: [GUNCELLEME_KILAVUZU](docs/GUNCELLEME_KILAVUZU.md) |
| **Diğer makinede eski sürüm / gece güncellemesi çalışmıyor** (Win) | Görev ölü yola bağlı → `.\scripts\start-server.ps1 -Repair`, sonra `-Status` ile **[ESLESIYOR]** doğrula |
| **"Bu makine güncel mi?" emin değilim** | `uv run hektor doctor` — dal/HEAD vs origin/main + Windows görev yolu (sapma varsa exit 2) |

---

## 🏗️ Proje Yapısı

```
app/
├── ingestion/   PDF okuma, metadata, chunk'lama
├── memory/      SQLite + ChromaDB + embedding
├── brain/       RAG, bilgi kartı, model routing
├── learning/    Paper Mastery Agent (0-100 skor)
├── training/    LoRA eğitim pipeline, reward signal, DPO
├── trading/     Strateji IR, backtest, indikatörler
├── verification/ Kaynak doğrulama, grounding kontrolü
├── evals/       Disiplin/overfit/risk failure-mode setleri
├── agents/      OSS Learning Agent, araştırma orchestrator
└── main.py      CLI (Typer)

tests/           407+ pytest testi — çevrimdışı çalışır
.claude/skills/  trading-research, backtest-auditor, paper-mastery-agent
evals/           discipline_core.jsonl, overfit_traps.jsonl, risk_scenarios.jsonl
```

---

## 📋 7 Değişmez Kural

Sistem bu kuralları hiçbir zaman çiğnemez:

1. 🚫 **Kaynak yoksa uydurma** — RAG boşsa açıkça söyler
2. 🚫 **Test edilmeden "başarılı" deme** — backtest + OOS şart
3. 🚫 **Maliyetleri yok sayma** — komisyon + slippage her backtest'te
4. 🚫 **Look-ahead yok** — pozisyon bir bar gecikmeli
5. 🚫 **Overfit gizleme** — in/out-of-sample bölmesi zorunlu
6. 🚫 **Rastgelelik gizleme** — seed parametresi daima açık
7. 🚫 **Kod yürütme** — strateji kuralları yalnızca güvenli regex ile

---

## 🔒 Güvenlik

- **Sadece kendi bilgisayarında çalışır** (`127.0.0.1`) — internetten erişilemez
- İstersen **şifre** ekle: `.env` → `HEKTOR_API_TOKEN=güçlü-rastgele-şifre`
- PDF doğrulaması var — sahte dosya geçemez
- IP başına hız sınırı — spam koruması
- CSP başlıkları aktif — XSS koruması

---

## 🛠️ Geliştirme

```bash
make install      # uv sync + pre-commit kur
make test         # pytest (329 test)
make lint         # ruff check
make format       # ruff format
make typecheck    # mypy
```

CI: Her push/PR'da — `ruff` + `mypy` + `pytest` (çevrimdışı) — Ubuntu + Python 3.12

---

## 🔗 Bağlantılar

| | |
|-|-|
| 📦 Repo | https://github.com/alimirbagirzade/hektor |
| 🌐 Web UI | `http://127.0.0.1:8765` _(çalışırken)_ |
| 📖 API Docs | `http://127.0.0.1:8765/api/docs` _(çalışırken)_ |
| 🧪 Test | `uv run pytest` |

```bash
git clone https://github.com/alimirbagirzade/hektor.git
cd hektor && uv sync && uv run hektor init
```

---

## ⚖️ Lisans & Sorumluluk Reddi

Eğitim ve araştırma amaçlıdır. Finansal tavsiye değildir.
Geçmiş performans gelecek sonuçların garantisi değildir.
