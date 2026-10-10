# Tasarım — Gece döngüsü (gün sonu otomasyonu) · Faz 1

_2026-10-10 · Durum: **uygulandı (Faz 1)**, testler çevrimdışı (sahte üretici + sentetik OHLCV).
Gerçek Ollama hakemi ve gerçek kullanıcı CSV'siyle henüz DENENMEDİ. Üstüne kurulduğu iş:
`docs/TASARIM_SUREKLI_DONGU.md` (#52)._

## 1 · İstek

"Otomasyonu ajantik ve tam otomatik yap: gün sonunda eğitim çıktısı (faydalı bilgi) API ya da
Claude/GPT ile kontrol edilsin, kaydedilsin ve devam edilsin; model sorulara gerçekçi cevap
versin; gün sonunda verdiğimiz CSV'ye dayanarak çalışan gösterge yazılabilsin."

## 2 · Kararlar ve sınırlar

| Konu | Karar |
|---|---|
| Gözetimsiz hakem | **Yerel Ollama** (`HEKTOR_NIGHTLY_JUDGE_MODEL`, boş → `llm_model`). Abonelikli bulut CLI'ı gözetimsiz/toplu çağırmak sağlayıcı şartlarına ve CLAUDE.md'ye ("toplu/otomatik bulut döngüsü yasak") aykırı; bulut K1/K2 (#52) tık başına kalır. Ollama Cloud etiketleri (`-cloud`) reddedilir — uzakta çalışırlar. |
| Hakemin yetkisi | **Tek yönlü:** şüpheli/belirsiz → karantina. "Tutarlı" hiçbir şeyi onaylamaz (çoğu kurulumda hakem = cevabı üreten model; kendini kayırır). |
| Karantinayı kim kaldırır | Yalnız gerekçeli insan (≥10 kr; web "Karantinayı kaldır" ya da `hektor karantina-kaldir`). Aynı hedef metin için kaldırılmış karantina geri konmaz; hedef değişirse yeniden yargılanır. |
| CSV piyasası | **Hepsi:** kripto vadeli, kripto spot, forex/CFD (altın dahil), BIST. Maliyet profili dosya adından ya da yan dosyadan; bulunamazsa **ihtiyatlı (yüksek maliyet) varsayımı**, raporda VARSAYIM diye yazılır. |
| CSV zaman dilimi | **Hepsi:** verinin medyan adımından çıkarılır (1m…1w); tam eşleşmezse dosya atlanır, sebep raporda. |
| Eğitim | **Faz 1: başlatılmaz.** Gece yalnız hazırlık raporu yazar; onay "Bekleyen kararlar"da insanda. |

**Faz 2 (açık, bu PR'da YOK):** gözetimsiz gece eğitimi. CLAUDE.md Kural 8 ("gerçek eğitim yalnız
açık `--run` + taze insan onayı") ve "her eğitimden önce Kademe 2" kuralının değişmesini ister;
terfi her durumda insanda kalır. Kullanıcı kararı + Kademe 2 + ayrı PR gerekir.

## 3 · Akış (`hektor gece`)

```
STOP_ALL / STOP_LEARNING?  → hiçbir adım koşmaz (rapor "stopped")
kilit (tek koşu; 6 sa ya da ölü pid → bayat)
 1 hakem   ağır iş sürüyorsa atla · Ollama yoksa atla (aday dokunulmaz)
           eligible/review adaylar (test sohbeti hariç) → yerel hakem (temperature 0, seed 42)
           şüpheli/belirsiz → quarantined   · gece başına en fazla N yeni yargı (kalan ertesi gece)
 2 csv     data/market/raw/*.csv (dosya+yan dosya özeti başına BİR kez; defter)
 3 egitim  easy_train.readiness() → yalnız rapor (started_training=false)
rapor      reports/nightly/<gün>/gece_*.{json,md} + storage/nightly/latest.json
           "Bekleyen kararlar" kutusu: karantina sayısı · OOS'ta tutarlı CSV adayları · 36 sa+ koşmadı
```

Bir adımın hatası diğerlerini durdurmaz (`partial`). Zamanlayıcı:
`scripts/install-nightly-task.ps1` (varsayılan 23:30, gizli pencere) → `scripts/nightly.ps1`.

## 4 · Karantina (veri modeli)

`learning_candidates.quarantine` (JSON, yeni kolon; eski veritabanına `ALTER TABLE` ile eklenir):
`{active, verdict, reason, judge_id, model, target_sha, at, lifted:{reason, at, target_sha}}`.
Durum önceliği: hariç > sızıntı > aile köprüsü > **karantina** > doğrulama kararı > aile denetimi.
Karantina yalnız `eligible`/`review` kararını bastırır (ret daha güçlü kalır). Eğitim seçimi
yalnız `eligible` okuduğundan karantinadaki aday hiçbir veri sürümüne girmez. Hakem metni
`storage/nightly/judge/lj_*.json`'da durur; eğitim verisine yazılmaz.

## 5 · CSV laboratuvarı (`app/trading/csv_lab.py`)

1. **Meta:** yan dosya `<ad>.meta.json` (`profile`, `tz`, `timeframe`, `market`, `costs`) > dosya
   adı > ayar (`HEKTOR_CSV_LAB_DEFAULT_TZ`). Ofsetsiz gün-içi damga + saat dilimi yok → atlanır
   (sessizce UTC varsayılmaz); günlük/haftalık veride UTC varsayılır ve yazılır.
2. **Veri denetimi:** `data_quality.load_checked_csv`.
3. **Dönemler:** sohbet stratejileriyle AYNI sabit protokol (%60 geliştirme · %20 doğrulama ·
   %20 final). **Final dönemine bu modül hiç dokunmaz.**
4. **Arama:** sabit ızgara, N=8 (yalnız uzun) ya da 16 (uzun+kısa): EMA trend (4), RSI dönüş (2),
   SMA kırılım + permütasyon entropisi filtresi (2). Kayıtlı göstergeler + güvenli kural dili
   (eval yok). Sinyal bar kapanışında, dolum sonraki açılışta; dört maliyet de düşülür.
5. **Seçim:** ≥30 işlemli VE geliştirmede maliyet sonrası Sharpe'ı pozitif denemeler; Deflated
   Sharpe (Bailey & López de Prado 2014) tüm denemelerin Sharpe dağılımıyla hesaplanır.
6. **Doğrulama:** ilk K (`HEKTOR_CSV_LAB_TOP_K`, 3) için önek-değişmezliği sızıntı kontrolü + OOS
   dönemi BİR kez; erişim `csvlab` ailesine yazılır (ikinci bakış etiketlenir).
7. **Etiket:** `aday_oos_tutarli` (DSR ≥ 0,95 · OOS Sharpe > 0 · OOS getiri > 0 · ≥30 işlem) ya
   da `aday_zayif` / `sizinti_supheli`. Hiçbiri "hazır" değildir.
8. **Çıktı:** `reports/csv_lab/<gün>/<ad>_<özet>.{json,md}` + Pine v5 **gösterge** taslağı
   (yalnız EMA/SMA/RSI/ATR; permütasyon entropisinin Pine karşılığı yok).

Maliyet profilleri (bp; kaba, temkinli perakende varsayımı — gerçek hesabınız için yan dosyada
`costs` ile ezin):

| Profil | Komisyon/dolum | Kayma/dolum | Makas (tam) | Fonlama/gün | Kısa |
|---|---|---|---|---|---|
| kripto_vadeli | 5 | 3 | 2 | 3 | var |
| kripto_spot | 10 | 3 | 2 | 0 (spot) | yok |
| forex_cfd | 0,5 | 1 | 3 | 1 | var |
| bist | 10 | 5 | 5 | 0 (spot hisse) | yok |
| ihtiyatli | 10 | 5 | 5 | 3 | var |

## 6 · Ayarlar

`HEKTOR_NIGHTLY_JUDGE_MODEL` · `HEKTOR_NIGHTLY_JUDGE_MAX` (40) · `HEKTOR_CSV_LAB_DEFAULT_TZ` ·
`HEKTOR_CSV_LAB_MAX_FILES` (5) · `HEKTOR_CSV_LAB_TOP_K` (3).

## 7 · Test

`tests/test_nightly_loop.py` (çevrimdışı): şüpheli/belirsiz → karantina, yeniden hesaplamada
kalır, eğitim seçimine girmez · tutarlı hiçbir şeyi değiştirmez · Ollama hatası/yokluğu adaya
dokunmaz · bulut etiketi reddedilir · yalnız gerekçeli insan kaldırır, hakem geri koymaz, hedef
değişince yeniden yargılar · gece sınırı · web kaldırma ucu `require_human` · gece kodu bulut/
eğitim başlatma modüllerini içe aktarmaz · profil/zaman dilimi tespiti · DSR deneme sayısıyla
düşer · Pine yalnız gösterge · uçtan uca CSV: final dokunulmaz, OOS erişimi kayıtlı, ikinci koşu
yeniden işlemez · saat dilimsiz gün-içi atlanır · yan dosya profili ezer · STOP_ALL no-op · adım
hatası izole · eşzamanlı koşu reddi + bayat kilit · eğitim adımı yalnız rapor.

**Sıradaki elle deneme:** gerçek Ollama ile `hektor gece -a hakem` (birkaç aday) ve gerçek bir
CSV ile `hektor csv-lab --dosya <ad>.csv`; raporları okuyup eşikleri birlikte ayarlamak.
