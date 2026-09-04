# 🔄 Güncelleme ve Çok-Makine Kullanım Kılavuzu

> **Kime?** Birden fazla bilgisayarda (ör. 2× Windows + 1× Mac) Achilles çalıştıran
> ve **"bir makinede güncel, diğerlerinde güncelleme tam oturmuyor"** sorununu yaşayan
> herkese. Teknik bilgi gerektirmez — komutları kopyala-yapıştır.

---

## 0) Altın kural: tek doğru kaynak GitHub `main`

Her bilgisayar kodu tek bir yerden alır: GitHub'daki **`main`** dalı.
- **Geliştirme bu makinede yapılır → `main`'e push edilir.**
- **Diğer makineler yalnızca `main`'i çeker** (`update`).
- RAG indeksi, eğitilmiş LoRA adapter'ları, veritabanı **her makinede sıfırdan üretilir**
  (git'e konmaz). Yani "güncelleme" = **kodu** eşitlemek; verini değil.

Bir makine `main` yerine başka bir dalda "parklanmış" kalırsa, güncelleme oturmaz.
Bunu kontrol eden tek komut: **`achilles doctor`** (aşağıda).

---

## 1) Makinem güncel mi? → `achilles doctor`

Herhangi bir makinede, Achilles klasöründe:

```bash
uv run achilles doctor
```

Şunu raporlar (hiçbir şeyi **değiştirmez**, tamamen güvenli):

| Satır | Ne demek | İstenen |
|-------|----------|---------|
| **Mevcut dal** | Bu kopya hangi dalda | `main` |
| **HEAD == origin/main** | GitHub ile birebir aynı mı | **EVET (yakınsamış)** |
| **behind / ahead** | Kaç commit geride / ileride | **0 / 0** |
| **Görev AchillesWeb / AchillesUpdate** (Windows) | Otomatik başlatma/güncelleme görevi doğru klasörü mü gösteriyor | **"bu repoyu işaret ediyor"** |
| **Push'lanmamış yerel dal** | GitHub'a gitmemiş iş var mı | (bilgi amaçlı) |

- Her şey yeşil/EVET ise: makine güncel. ✅
- **"HAYIR — SAPMA"** veya **"ÖLÜ/yabancı yol"** görürsen → **2. bölüme** geç.

> `doctor` çevrimdışı çalışır (interneti yoklamaz). `behind` sayısı şüpheliyse önce
> `git fetch origin main` yapıp tekrar bak.

---

## 2) Güncelleme — günlük kullanım

Yeni sürüm yayınlandığında, kurulu makinede **tek komut**:

**Windows (PowerShell):**
```powershell
cd "$env:USERPROFILE\achilles"
.\update.ps1
```

**macOS / Linux (Terminal):**
```bash
cd ~/achilles
./update.sh
```

Bu betikler artık **hangi dalda olursan ol `main`'e geçip** GitHub ile eşitler
(eskiden yapmıyordu — kök sorun buydu), bağımlılıkları kurar, web'i yeniden başlatır.

> 🔴 Bittikten sonra tarayıcıda **sert yenileme**: Windows/Linux `Ctrl + Shift + R`,
> macOS `Cmd + Shift + R`. Yoksa eski arayüz önbellekten gelir, "değişmemiş" gibi görünür.

---

## 3) İlk-seferlik onarım (makine zaten bozulduysa)

Geçmişte bir makinende üç şey ters gitmiş olabilir (hepsi sessizdi):

1. **Yanlış dalda parklanma** — kopya `main` yerine eski bir özellik dalında kaldı.
2. **Ölü otomatik-güncelleme görevi** (Windows) — gece 03:00 görevi var-olmayan bir
   klasörü çağırıp sessizce hiçbir şey yapmıyordu.
3. **Sessiz `git pull` başarısızlığı** — kullanıcı "güncelledim" sanıyor ama kod değişmedi.

Her bozuk makinede **tek sefer** şunu yap (verilerin **silinmez** — `data/`, `storage/`,
`vector_db/`, adapter'lar git'te izlenmez):

**Windows (PowerShell, Achilles klasöründe):**
```powershell
cd "$env:USERPROFILE\achilles"
.\update.ps1 -Force                  # main'e geç + GitHub'a zorla eşitle
.\scripts\start-server.ps1 -Repair   # ölü autostart/03:00 görevlerini BU klasöre yeniden bağla
uv run achilles doctor               # doğrula: dal=main, +0/-0, görevler "bu repoyu işaret ediyor"
```

**macOS / Linux (Terminal):**
```bash
cd ~/achilles
./update.sh --force
uv run achilles doctor
```

`-Force` / `--force` yalnız **izlenen kod dosyalarını** GitHub `main`'e sıfırlar;
verilerini ve eğitilmiş modellerini **silmez**.

> **Klasör başka yerde / `cd ~/achilles` "no such file" diyorsa** — şu tek satır klasörü
> kendisi bulur, girer ve onarır (Mac/Linux):
> ```bash
> cd "$(find ~ -type d -name achilles 2>/dev/null | head -1)" && ./update.sh --force
> ```

---

## 4) Windows: otomatik başlatma ve gece güncellemesi

Açılışta otomatik başlatma + her gece 03:00 otomatik güncelleme kurmak / onarmak:

```powershell
cd "$env:USERPROFILE\achilles"
.\scripts\start-server.ps1 -Install   # ilk kurulum (autostart + 03:00 güncelleme)
.\scripts\start-server.ps1 -Repair    # görevlerin yolu BU klasöre uymuyorsa onar
.\scripts\start-server.ps1 -Status    # görev yolu bu repoyu mu işaret ediyor?
```

`-Status` çıktısında **Web yolu** / **Upd yolu** satırları:
- **[ESLESIYOR]** → görev doğru klasörü çalıştırıyor. ✅
- **[FARKLI → bu repo degil; -Repair calistir]** → ölü/yabancı yol; `-Repair` çalıştır.

> **İki kopya tuzağı:** Achilles'i birden çok klasöre kurma (ör. `~\achilles` *ve*
> `~\Development\achilles`). Görevler birini, sen diğerini güncellersen "oturmuyor" yaşarsın.
> `install.ps1` artık ikinci kopyayı tespit edip uyarır; tek klasörde kal.

---

## 5) Sık sorunlar — hızlı tablo

| Belirti | Sebep | Çözüm |
|---------|-------|-------|
| "Güncelledim ama arayüz aynı" | Tarayıcı önbelleği | `Ctrl/Cmd + Shift + R` (sert yenileme) |
| "Güncelledim ama yeni özellik yok" | Yanlış dalda parklanma | `update --force` → `achilles doctor` |
| "Gece güncellemesi hiç çalışmıyor" (Win) | Görev ölü yola bağlı | `start-server.ps1 -Repair` |
| `doctor` "ÖLÜ/yabancı yol" diyor | Görev başka klasörü gösteriyor | `start-server.ps1 -Repair` |
| `git pull` "diverged/iraksak" hatası | Yerel main GitHub'dan ayrışmış | `update --force` (yereli atar) |
| İki ayrı achilles klasörü var | Mükerrer kurulum | Birini sil, kalanı `-Repair` ile bağla |

---

## 6) Geliştirme yapan makine için not

Bu makinede kod yazıp **`main`'e push** ettiğin sürece diğer makineler `update` ile alır.
Eğer iş bir özellik dalında kalıp **`main`'e merge edilmezse**, diğer makinelere asla ulaşmaz.
`achilles doctor` çıktısındaki **"Push'lanmamış yerel dal"** satırı, GitHub'a gitmemiş işi
hatırlatır — orada bir şey görüyorsan, o iş henüz hiçbir makineye ulaşmamış demektir.
