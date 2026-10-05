# Sohbetten strateji testi — Faz 2B protokolü

_Amaç: Hektor'un sohbette önerdiği stratejiyi, stop ve maliyetleriyle, gerçek veride test
edebilmek. Sonuç "kayıtlı hesap"tır; gelecekte başarı, yatırım tavsiyesi ya da modelin daha iyi
trader olduğu anlamına gelmez._

## Akış (00 · SOHBET → cevap altındaki **Strateji testi**)

1. **Taslak** — yerel Ollama modeli (ana sohbet modeli) cevabı JSON taslağa çevirir
   (`app/trading/strategy_draft.py`, seed 42, sıcaklık 0). Kayıt yapılmaz.
   - Şablon/örnek stratejiyle (`example_ir`) **sessiz ikame yoktur**; çıkarım başarısızsa hata.
   - Model değer uydurmaz: cevapta olmayan maliyet/stop alanı boş kalır.
   - Kural diline çevrilemeyen koşul (seans/saat filtresi, formasyon, haber, VEYA, kesişim…)
     **ÖNEMLİ desteklenmeyen kural** olur. Cevapta stop/hedef geçip taslakta yoksa da öyle.
2. **Okunur form** — piyasa, zaman dilimi, yön, giriş/çıkış kuralları, stop, hedef, pozisyon
   büyüklüğü, maliyetler, desteklenmeyen kurallar. Kullanıcı düzeltir ve **kaydeder**.
   Kayıt değişmezdir (içerik özetinden kimlik); düzenleme = aynı ailede yeni varyant.
3. **Basitleştirilmiş strateji** (yalnız açık seçimle) — çıkarılan kurallar listelenir, ayrı
   kimlik + `simplified_from`; asıl strateji test edilmiş **sayılmaz**.
4. **Veri** — `data/market/raw/*.csv` (yükleme düğmesi) + saat dilimi → denetim raporu.
5. **Test** — Geliştirme / Doğrulama (OOS) / Final (tek kullanım).

## Strateji tanımı (`app/trading/strategy_spec.py`)

| Alan | Desteklenen | Not |
|---|---|---|
| Yön | long, short | |
| Stop | yok · sabit % · ATR girişte sabit · ATR takip eden | Takip eden: bar kapanışında yalnız lehe, sonraki bardan geçerli |
| Hedef | yok · sabit % · ATR katı · R katı | R katı stop ister |
| Boyut | özsermaye katı (nominal) · işlem başına risk (stop'a göre) | Kaldıraç tavanı açık; risk stop ister |
| Maliyet | komisyon bp/dolum · kayma bp/dolum · makas bp (tam) · fonlama bp/gün | Hepsi zorunlu; sıfır yalnız ≥10 kr gerekçeyle, raporda görünür |

## Motor (`app/trading/event_engine.py`, `hektor-event-1.0.0`, metrik `m1`)

- Sinyal bar kapanışında (yalnız o bar ve öncesi), dolum **sonraki açılışta**.
- Stop/hedef: giriş dolumu + **sinyal barının** ATR'si.
- Giriş barında da stop geçerli; aynı barda stop ve hedef → **stop önce**.
- Gap: açılış stop'un ötesindeyse dolum **açılıştan**; hedefin ötesindeyse **hedeften**.
- Her dolumda fiyat aleyhe (kayma + makas/2), nominal üzerinden komisyon; açık nominalden bar
  süresince fonlama. Dönem sonunda açık pozisyon son kapanışta maliyetle kapanır.
- **Sızıntı:** `shift(1)` tek başına kanıt sayılmaz. Her koşuda veri seed'li noktalardan
  kesilir; öneklerde sinyal/ATR tam veriyle aynı değilse koşu reddedilir.

## Veri denetimi (`app/trading/data_quality.py`)

Zaman kolonu zorunlu · ofsetsiz zaman damgasında saat dilimi zorunlu (sessiz UTC yok) · sıra
düzeltilir ve kaydedilir · birebir yineleme düşürülür, çelişen yineleme reddedilir · geçersiz
OHLC / pozitif olmayan fiyat / eksik değer reddedilir · zaman dilimi uyuşmazlığı reddedilir ·
eksik bar doldurulmaz, sayısı kaydedilir. Ham dosya ve temiz veri özetleri rapora yazılır.

## Dönemler ve örneklem dışı disiplini (`app/trading/strategy_testing.py`)

- Temiz veri özeti başına **sabit** sınırlar: %60 geliştirme · %20 doğrulama · %20 final.
- Göstergeler önceki tüm veriyle ısınır; işlem ve getiri yalnız dönem içinde.
- Doğrulama dönemine her bakış kaydedilir. Aynı aileden başka varyant baktıysa dönem
  **"geliştirmede kullanıldı"** olur; aile deneme sayısı sonuçta görünür.
- Final dönemi aile başına **bir kez**; ikinci istek reddedilir (yeni veri gerekir).
- Sonuçta uyarılar: az işlem (<30), Sharpe > 4, şüpheli profit factor, işlem yok.

## Zaman alanları

Her koşu ayrı kaydeder: `data_start/data_end`, `period_start/period_end`,
`strategy_created_at`, `backtest_run_at`, `knowledge_available_at` (= koşu zamanı; fiyat
verisinin bitişi DEĞİL).

## Eğitim verisine bağlantı ve performans iddiaları (Bölüm 4)

- Öğrenme Havuzu → aday kartı → **Test koşusu bağla** (`link_run`): adaya koşunun zaman
  alanları yazılır (`time_meta`): veri başlangıç/bitişi, test dönemi, strateji oluşturma, koşu
  zamanı, tur zamanı ve **bilgi zamanı = max(tur, koşu)**. `as_of` bilgi zamanıdır; trading
  bölmesi buna göre zaman sıralıdır. Fiyat verisinin bitişi bilgi zamanı DEĞİLDİR.
- Bağlanan aday strateji **ailesine** (`fam_<strateji ailesi>`) girer: yakın varyantlar ve
  basitleştirilmiş sürümler aynı ailede kalır, train/eval'e bölünmez.
- `backtest` kontrolü (`app/trading/claim_check.py`): koşu bağlı değilse "yapılamadı" (insan
  onayı geçemez). Bağlıysa hepsi gerekir: parmak izi yeniden hesaplanıp tutmalı (strateji,
  temiz veri, dönem, maliyet, metrik/motor sürümü), motor güncel olmalı, metin dönemin
  başlangıç/bitiş tarihlerini içermeli, metrik sayıları yazıldıkları hassasiyette eşit olmalı,
  geliştirme sonucu "örneklem dışı", maliyetli hesap "maliyetsiz" diye sunulmamalı.
  Sonuç **"kayıtlı hesapla eşleşti"** ya da **"eşleşmedi" (çürütür)**; diğer ifadeler yine insan
  incelemesi bekler.

## Doğrulama durumu (2026-10-05)

- **Otomatik:** `tests/test_event_engine.py`, `tests/test_strategy_flow.py`.
- **Gerçek modelle (hektor-v12-30b) + tarayıcıda:** sohbet cevabından taslak (stop 2×ATR
  sabit, %1 risk, maliyetler boş, "yalnız Londra seansı" ÖNEMLİ desteklenmeyen) → asıl strateji
  testi engellendi → basitleştirilmiş strateji → veri denetimi → geliştirme + doğrulama koşusu.
- **Henüz doğrulanmadı:** gerçek piyasa verisi (makinede OHLCV CSV yok; canlı deneme sentetik
  CSV ile yapıldı).
