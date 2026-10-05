# Gerçek piyasa verisiyle backtest hesap denetimi

_Üretim: 2026-10-05T15:20:20.250064+00:00 · motor `hektor-event-1.0.0` · metrik `m1` · sonuç: **HESAP TUTARLI**_

> Başarı ölçütü kâr değil, hesapların ve veri akışının doğruluğudur. Sayılar **kayıtlı hesap**tır; strateji önerisi ya da yatırım tavsiyesi değildir. Final dönemi koşulmadı; doğrulama dönemi bu denetimde **geliştirme amaçlı** kullanıldı.

## Veri

- GERÇEK piyasa verisi — teknik doğrulama piyasası (kullanıcının nihai trading tercihi DEĞİLDİR). Sentetik değildir.
- Kaynak: Binance kamu veri arşivi (data.binance.vision) — borsanın yayımladığı spot kline dosyaları; hesap/anahtar/ödeme gerekmez — `https://data.binance.vision/data/spot/monthly/klines` (24 aylık dosya, her biri `.CHECKSUM` SHA256 ile doğrulandı)
- Sembol / tür / aralık: **BTCUSDT** · spot · 1h
- Saat dilimi: UTC (mum açılış zamanı, ISO-8601 +00:00)
- Dönem: 2023-01-01T00:00:00+00:00 → 2024-12-31T23:00:00+00:00 (17543 bar)
- İndirme zamanı: 2026-10-05T15:16:19.473486+00:00
- CSV SHA256: `0e050ec4e860248b8042d4679359e82059fcef276ae4939827c9275259610da9` (denetimde yeniden hesaplandı, eşleşti)
- Sınır: Kline verisinde alış/satış kotasyonu yok → makas (spread) ölçülmedi, varsayımdır. Spot piyasa → fonlama yok.

### Veri akışı tutarlılığı (ürün denetimi ↔ bağımsız okuma)

| Kontrol | Sonuç |
|---|---|
| rows_equal | ✅ |
| start_equal | ✅ |
| end_equal | ✅ |
| gap_count_equal | ✅ |

Bağımsız okuma: 17543 satır, artan sıralı=True, boşluk 1 (1 eksik bar; doldurulmadı): 2023-03-24T12:00:00+00:00 → 2023-03-24T14:00:00+00:00; geçersiz OHLC satırı 0; ofsetler ['+00:00'].
Dönem sınırları (sabit, temiz veri özeti başına): geliştirme → 2024-03-14T14:00:00+00:00 · doğrulama → 2024-08-07T19:00:00+00:00.

## Varsayımlar

- Giriş: kapanışta `ema_20 > ema_50` (long) / `ema_20 < ema_50` (short); dolum sonraki açılışta.
- Çıkış: kapanışta ters kesişim → sonraki açılış; stop 2×ATR(14) girişte sabit (sinyal barının ATR'si); hedef 3R; aynı barda stop+hedef → stop önce; gap → açılıştan.
- Pozisyon büyüklüğü: işlem başına özsermayenin %1 riski (stop mesafesine göre), kaldıraç ≤ 1.
- `commission_bps_per_side` = 10.0: Binance spot standart taker ücreti %0.10 = 10 bp/dolum (indirimsiz; hesap seviyesine göre değişir — varsayım)
- `slippage_bps_per_side` = 2.0: 2 bp/dolum aleyhe (1h BTCUSDT için varsayım; ölçülmedi)
- `spread_bps` = 2.0: 2 bp tam makas (kline verisinde kotasyon yok → ölçülmedi, varsayım)
- `funding_bps_per_day` = 0.0: 0 — spot, kaldıraçsız (gerekçe kayıtlı)

Kapsam: Yalnız geliştirme + doğrulama dönemleri; final dönemi koşulmadı. Bağımsız hesap kapsamı: EMA/ATR, atr_initial stop, R hedef, risk tabanlı boyut (takip eden stop ve ATR-katı hedef kapsam dışı).

Onay: Nihai strateji onayı denetim betiği tarafından otomatik verildi — kullanıcının strateji onayı DEĞİL; yalnız teknik doğrulama.

## Koşular

| Strateji | Dönem | OOS etiketi | İşlem (ürün/bağımsız) | Getiri ürün / bağımsız (%) | Azami göreli fark | Yapı aynı |
|---|---|---|---|---|---|---|
| denetim_ema20_50_long | gelistirme (2023-01-01 → 2024-03-14) | gelistirme | 288/288 | -6.6552 / -6.6552 | 5.65e-16 | ✅ |
| denetim_ema20_50_long | dogrulama (2024-03-14 → 2024-08-07) | ilk_bakis | 75/75 | -17.6443 / -17.6443 | 1.47e-15 | ✅ |
| denetim_ema20_50_short | gelistirme (2023-01-01 → 2024-03-14) | gelistirme | 227/227 | -48.6395 / -48.6395 | 2.69e-16 | ✅ |
| denetim_ema20_50_short | dogrulama (2024-03-14 → 2024-08-07) | ilk_bakis | 93/93 | 6.8301 / 6.8301 | 8.28e-16 | ✅ |

### denetim_ema20_50_long · gelistirme — 10 işlem karşılaştırıldı (toplam 288; >10 ise ilk, son ve aradan eşit aralıklı)

| # | Giriş | Çıkış | Neden | Giriş fiyatı | Çıkış fiyatı | Miktar | Maliyet | Net PnL | Δfiyat (g/ç) | Δmiktar | Δmaliyet | ΔPnL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2023-01-01T02:00 | 2023-01-02T21:00 | hedef | 16556.44 | 16772.57 | 0.00006040 | 0.00261697 | 0.01104114 | 0.0e+00/0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 32 | 2023-02-28T14:00 | 2023-02-28T21:00 | stop | 23410.19 | 23141.65 | 0.00004317 | 0.00261236 | -0.01360168 | 0.0e+00/0.0e+00 | -2.0e-20 | -8.7e-19 | 6.9e-18 |
| 64 | 2023-04-19T09:00 | 2023-04-19T10:00 | kural | 29177.22 | 29191.82 | 0.00002348 | 0.00178167 | -0.00102772 | 0.0e+00/0.0e+00 | 6.8e-21 | 6.5e-19 | -2.2e-19 |
| 96 | 2023-06-08T20:00 | 2023-06-09T22:00 | kural | 26547.82 | 26471.57 | 0.00002888 | 0.00199067 | -0.00373367 | 0.0e+00/0.0e+00 | 6.8e-21 | 8.7e-19 | -8.7e-19 |
| 128 | 2023-07-20T08:00 | 2023-07-20T14:00 | stop | 30211.84 | 29947.55 | 0.00003043 | 0.00237950 | -0.00987153 | 0.0e+00/-3.6e-12 | 3.4e-21 | 0.0e+00 | -1.1e-16 |
| 159 | 2023-08-27T18:00 | 2023-08-28T04:00 | kural | 26072.90 | 26005.18 | 0.00002981 | 0.00201790 | -0.00357079 | 0.0e+00/0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 191 | 2023-10-16T14:00 | 2023-10-20T08:00 | hedef | 28008.39 | 29723.72 | 0.00001260 | 0.00094535 | 0.02087914 | 0.0e+00/3.6e-12 | -3.4e-21 | -2.2e-19 | 4.2e-17 |
| 223 | 2023-12-13T21:00 | 2023-12-14T13:00 | stop | 42887.74 | 42171.24 | 0.00001209 | 0.00133703 | -0.00969199 | 0.0e+00/0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 255 | 2024-01-26T05:00 | 2024-01-26T12:00 | hedef | 40161.29 | 41424.82 | 0.00001799 | 0.00190793 | 0.02126161 | 0.0e+00/0.0e+00 | -1.0e-20 | -8.7e-19 | -1.4e-17 |
| 287 | 2024-03-06T09:00 | 2024-03-14T13:00 | donem_sonu | 66961.30 | 71863.24 | 0.00000345 | 0.00062343 | 0.01645376 | 0.0e+00/0.0e+00 | -8.5e-22 | -1.1e-19 | -3.5e-18 |

Çıkış sayaçları (ürün): stop 135 · gap-stop 0 · hedef 62 · kural 90 · dönem sonu 1 · aynı bar stop önce 0. Sızıntı (önek) kontrolü: geçti.

### denetim_ema20_50_long · dogrulama — 10 işlem karşılaştırıldı (toplam 75; >10 ise ilk, son ve aradan eşit aralıklı)

| # | Giriş | Çıkış | Neden | Giriş fiyatı | Çıkış fiyatı | Miktar | Maliyet | Net PnL | Δfiyat (g/ç) | Δmiktar | Δmaliyet | ΔPnL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2024-03-14T15:00 | 2024-03-14T16:00 | stop | 71711.50 | 70386.69 | 0.00000767 | 0.00141696 | -0.01125199 | 0.0e+00/0.0e+00 | -6.8e-21 | -1.5e-18 | 8.7e-18 |
| 8 | 2024-03-30T17:00 | 2024-03-30T20:00 | kural | 70199.22 | 69875.37 | 0.00001376 | 0.00250539 | -0.00638295 | 0.0e+00/0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 16 | 2024-04-09T15:00 | 2024-04-09T18:00 | kural | 69379.13 | 68803.34 | 0.00000729 | 0.00130972 | -0.00520548 | 0.0e+00/0.0e+00 | -8.5e-22 | -2.2e-19 | 0.0e+00 |
| 25 | 2024-05-10T02:00 | 2024-05-10T14:00 | stop | 62902.37 | 61934.39 | 0.00000967 | 0.00156987 | -0.01057116 | 0.0e+00/0.0e+00 | -3.4e-21 | -4.3e-19 | 3.5e-18 |
| 33 | 2024-05-20T05:00 | 2024-05-20T06:00 | stop | 67152.58 | 66467.60 | 0.00001352 | 0.00234769 | -0.01106359 | 0.0e+00/0.0e+00 | -1.7e-21 | 0.0e+00 | 1.7e-18 |
| 41 | 2024-05-27T20:00 | 2024-05-28T03:00 | stop | 69280.80 | 68390.59 | 0.00001035 | 0.00185274 | -0.01064071 | 0.0e+00/0.0e+00 | -5.1e-21 | -1.1e-18 | 5.2e-18 |
| 49 | 2024-06-05T16:00 | 2024-06-06T19:00 | stop | 71601.48 | 70664.87 | 0.00000972 | 0.00179675 | -0.01048134 | 0.0e+00/0.0e+00 | -8.5e-21 | -1.7e-18 | 8.7e-18 |
| 58 | 2024-06-20T11:00 | 2024-06-20T13:00 | stop | 66229.35 | 65538.19 | 0.00001233 | 0.00211130 | -0.01014284 | 0.0e+00/0.0e+00 | -8.5e-21 | -1.3e-18 | 6.9e-18 |
| 66 | 2024-07-09T06:00 | 2024-07-11T23:00 | kural | 57308.67 | 57321.38 | 0.00000635 | 0.00094658 | -0.00064740 | 0.0e+00/0.0e+00 | -9.3e-21 | -1.4e-18 | 9.8e-19 |
| 74 | 2024-08-07T04:00 | 2024-08-07T18:00 | stop | 56879.07 | 55183.14 | 0.00000496 | 0.00072219 | -0.00896282 | 0.0e+00/0.0e+00 | -5.1e-21 | -7.6e-19 | 8.7e-18 |

Çıkış sayaçları (ürün): stop 39 · gap-stop 0 · hedef 11 · kural 25 · dönem sonu 0 · aynı bar stop önce 0. Sızıntı (önek) kontrolü: geçti.

### denetim_ema20_50_short · gelistirme — 10 işlem karşılaştırıldı (toplam 227; >10 ise ilk, son ve aradan eşit aralıklı)

| # | Giriş | Çıkış | Neden | Giriş fiyatı | Çıkış fiyatı | Miktar | Maliyet | Net PnL | Δfiyat (g/ç) | Δmiktar | Δmaliyet | ΔPnL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2023-01-06T13:00 | 2023-01-06T13:00 | stop | 16725.63 | 16807.12 | 0.00005979 | 0.00260633 | -0.00687697 | 0.0e+00/0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 25 | 2023-03-07T16:00 | 2023-03-08T23:00 | hedef | 22315.28 | 21643.18 | 0.00004254 | 0.00243121 | 0.02672384 | 0.0e+00/0.0e+00 | 6.8e-21 | 4.3e-19 | 3.5e-18 |
| 50 | 2023-04-19T10:00 | 2023-04-21T17:00 | hedef | 29191.82 | 27796.78 | 0.00001883 | 0.00139480 | 0.02519131 | 0.0e+00/0.0e+00 | 3.4e-21 | 2.2e-19 | 3.5e-18 |
| 75 | 2023-06-05T16:00 | 2023-06-06T16:00 | stop | 25999.73 | 26321.08 | 0.00002789 | 0.00189719 | -0.01042280 | 0.0e+00/0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 100 | 2023-07-14T22:00 | 2023-07-20T08:00 | kural | 30232.93 | 30211.84 | 0.00001552 | 0.00121944 | -0.00061079 | 0.0e+00/0.0e+00 | 0.0e+00 | 2.2e-19 | -1.1e-19 |
| 126 | 2023-08-17T23:00 | 2023-08-23T17:00 | kural | 26793.26 | 26403.93 | 0.00000916 | 0.00063344 | 0.00307883 | 0.0e+00/0.0e+00 | 0.0e+00 | 1.1e-19 | -4.3e-19 |
| 151 | 2023-10-15T13:00 | 2023-10-15T13:00 | stop | 26830.09 | 26914.58 | 0.00002518 | 0.00175903 | -0.00348029 | 0.0e+00/0.0e+00 | 6.8e-21 | 6.5e-19 | -8.7e-19 |
| 176 | 2023-12-18T02:00 | 2023-12-18T14:00 | stop | 40969.72 | 41544.68 | 0.00001090 | 0.00116889 | -0.00716447 | 0.0e+00/0.0e+00 | 1.7e-21 | 2.2e-19 | -8.7e-19 |
| 201 | 2024-01-18T20:00 | 2024-01-19T18:00 | stop | 41010.02 | 41721.18 | 0.00000843 | 0.00090617 | -0.00668891 | 0.0e+00/0.0e+00 | -1.7e-21 | -2.2e-19 | 1.7e-18 |
| 226 | 2024-03-06T08:00 | 2024-03-06T09:00 | kural | 66656.46 | 66961.30 | 0.00000191 | 0.00033174 | -0.00083739 | 0.0e+00/0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |

Çıkış sayaçları (ürün): stop 108 · gap-stop 0 · hedef 32 · kural 87 · dönem sonu 0 · aynı bar stop önce 0. Sızıntı (önek) kontrolü: geçti.

### denetim_ema20_50_short · dogrulama — 10 işlem karşılaştırıldı (toplam 93; >10 ise ilk, son ve aradan eşit aralıklı)

| # | Giriş | Çıkış | Neden | Giriş fiyatı | Çıkış fiyatı | Miktar | Maliyet | Net PnL | Δfiyat (g/ç) | Δmiktar | Δmaliyet | ΔPnL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2024-03-14T19:00 | 2024-03-15T00:00 | stop | 70077.46 | 71782.69 | 0.00000594 | 0.00109531 | -0.01097041 | 0.0e+00/0.0e+00 | -1.7e-21 | -4.3e-19 | 3.5e-18 |
| 10 | 2024-03-24T00:00 | 2024-03-24T08:00 | stop | 63970.82 | 65223.57 | 0.00000819 | 0.00137531 | -0.01131629 | 0.0e+00/0.0e+00 | -6.8e-21 | -1.3e-18 | 8.7e-18 |
| 20 | 2024-04-12T17:00 | 2024-04-12T18:00 | hedef | 68819.89 | 65720.22 | 0.00000939 | 0.00164291 | 0.02785222 | 0.0e+00/0.0e+00 | -5.1e-21 | -8.7e-19 | -1.4e-17 |
| 31 | 2024-04-29T23:00 | 2024-04-30T02:00 | kural | 63870.84 | 63625.09 | 0.00001100 | 0.00182267 | 0.00130043 | 0.0e+00/0.0e+00 | -5.1e-21 | -1.3e-18 | -4.3e-19 |
| 41 | 2024-05-24T19:00 | 2024-05-25T06:00 | kural | 68949.74 | 68738.83 | 0.00000930 | 0.00166434 | 0.00068086 | 0.0e+00/0.0e+00 | -3.4e-21 | -6.5e-19 | 0.0e+00 |
| 51 | 2024-06-10T23:00 | 2024-06-11T04:00 | hedef | 69475.14 | 67878.39 | 0.00001309 | 0.00233691 | 0.01910004 | 0.0e+00/0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 61 | 2024-06-24T09:00 | 2024-06-24T09:00 | hedef | 62712.10 | 60767.11 | 0.00001419 | 0.00227836 | 0.02585328 | 0.0e+00/0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 72 | 2024-07-05T17:00 | 2024-07-06T16:00 | stop | 56431.22 | 58079.64 | 0.00000627 | 0.00093278 | -0.01104653 | 0.0e+00/0.0e+00 | -2.5e-21 | -2.2e-19 | 5.2e-18 |
| 82 | 2024-07-23T09:00 | 2024-07-25T02:00 | hedef | 66867.93 | 64116.03 | 0.00001062 | 0.00180827 | 0.02783265 | 0.0e+00/0.0e+00 | -1.7e-21 | -2.2e-19 | -3.5e-18 |
| 92 | 2024-08-06T15:00 | 2024-08-07T04:00 | kural | 56115.17 | 56879.07 | 0.00000516 | 0.00075819 | -0.00452611 | 0.0e+00/0.0e+00 | -8.5e-22 | 0.0e+00 | 8.7e-19 |

Çıkış sayaçları (ürün): stop 43 · gap-stop 0 · hedef 22 · kural 28 · dönem sonu 0 · aynı bar stop önce 0. Sızıntı (önek) kontrolü: geçti.

Not: Tabloda 10 işlem gösterilir; yapı (giriş/çıkış zamanı, neden) ve fiyat/PnL göreli farkı TÜM işlemlerde ayrıca taranır (`all_trades_checked`, `max_rel_diff`). Fiyat/miktar/PnL birimi özsermaye = 1.0 ölçeğindedir (miktar = nominal / fiyat). Gözlem: EMA'lar ilk bardan başlar (ısınma eşiği yok) — verinin ilk ~50 barındaki sinyaller olgunlaşmamış göstergeyle üretilir; bu bir tasarım sınırıdır, hata değil. Gerçek veride gap ve aynı-bar stop/hedef olayları az ya da hiç oluşmayabilir; bu uç durumlar `tests/test_independent_backtest_crosscheck.py` kontrollü bar testleriyle ayrıca doğrulanır.
