# Hektor araştırma paketi

2026-10-01 · Kullanıcı kapsamı: araştırma + kontrollü RAG işleme + veri hazırlığı;
yeni ağır eğitim için öneri sunulur. Paket mevcut eğitimler tamamlanmadan iş başlatmaz.

## Web'den tek tuş

`09 · ÖĞRENME` sekmesinin üstündeki **Eğitim sonrası araştırma paketi** kartı,
aynı yöneticinin durumunu, bekleme nedenlerini, son kontrol ve yeniden deneme
zamanını gösterir. Sekme açıkken 15 saniyede yenilenir; **Araştırma paketini yönet**
bağlantısı motor seçimi ve başlat/durdur kontrollerine götürür. İkinci döngü başlatmaz.
Eğitim bitse bile başlatma kilidi mevcutsa paket bekler; web kilidi temizlemez.

`15 · AJAN HARİTASI` → Motor seç → **Araştırma döngüsünü çalıştır**.
İsteğe bağlı ikinci motor bağımsız plan incelemesi yapar; aynı motor iki kez seçilemez.
Kurulu ve kısıtlanabilir Codex/Claude CLI'ları kendi abonelik oturumlarını kullanır.
CLI kurulu değilse seçim kapalıdır; oturum/kota hatası çalıştırma sırasında raporlanır.
Gemini'nin güvenli çalıştırma profili olmadığı için seçim kapalı kalır. API anahtarı
ve bulut SDK eklenmez; RAG/kart/veri üreticisi yerel Ollama'dır.

Çalıştır, `service.json` içine etkinliği kaydeder. Eğitim varsa **Bekliyor** gösterir;
CLI incelemesi bile başlamaz. Koşullar sağlanınca her turda önce motor(lar) planı
inceler, ardından tek yerel aşama çalışır. Tarayıcı sekmesi kapansa da web sunucusu
açık kaldığı sürece döngü sürer. Sunucu yeniden açılınca etkinlik kaydından devam
eder; bilgisayar/web kapalıyken iş yürütemez. Kontrol aralığı 60 saniyedir; aşamaların
aşağıdaki ayrı saat bütçeleri korunur. Tüm ajanların aynı anda çalışması amaçlanmaz.

Motorlar yalnız `proceed/stage/reason` kararı verir; çıktıları komut olarak
çalıştırılmaz. Yanlış/eksik karar veya ret yerel aşamayı engeller. Hata üç turda
tekrarlanırsa yönetici inceleme bekler; düzeltme sonrası Durdur → Çalıştır yönetici
hatasını sıfırlar. Aşama başına üç hata sınırı ayrıca geçerlidir; aşama kaydı
incelenmeden kendiliğinden sıfırlanmaz. Eğitim/onay/terfi/kod yazımı bu döngüde yoktur.
Yöntem incelemesi tam bir bilimsel doğrulama veya zorunlu eğitim öncesi Kademe 2 değildir.

**Araştırmayı durdur** yalnız paketi, kendi CLI sürecini ve işçisini durdurur.
Çalışan LoRA eğitimi etkilenmez. Başka STOP işaretleri kaldırılmaz. Mevcut
**EĞİTİMİ DEVREYE SOK** düğmesi ayrı tek-seferlik akışı ve onay diyaloğunu korur.

Web yöneticisi etkinken dış `--run` çağrısı engellenir; sohbet heartbeat'i yalnız
durum izler. Web yönetimi açılmamışsa mevcut saatlik heartbeat, kapılar uygunsa
tek aşama çağırabilir. Böylece iki zamanlayıcı aynı paketi birlikte süremez.
Web sunucusunu kod güncellemesinden sonra yeniden başlatmak gerekir; eski süreç
yeni Python uçlarını kendiliğinden yüklemez. Aktif eğitim sırasında yeniden başlatma
bu değişikliğin parçası olarak yapılmadı.

CLI yaklaşımı için resmi kaynaklar:
[Codex non-interactive mode](https://developers.openai.com/codex/noninteractive/),
[Claude Code programmatic use](https://code.claude.com/docs/en/headless).
Bu depoda mevcut kısıtlı motor şablonları kullanılır; kurulu Codex'in `exec --help`
çıktısında gerekli seçenekler doğrulandı. Canlı abonelik çağrısı eğitim sonrasına bırakıldı.

## CLI girişi

```powershell
uv run --no-sync hektor research-package           # salt-okunur plan / engeller
uv run --no-sync hektor research-package --run     # uygunsa yalnız bir aşama
uv run --no-sync hektor research-package --pause   # paket durdurma işareti
uv run --no-sync hektor research-package --resume  # yalnız bu işareti kaldır
```

Reçete: `configs/research_package.json`. `--run` eğitim başlatmaz; yalnız araştırma
paketinin bir aşamasını yürütür. Komutu saatlik çağıran tek zamanlayıcı yeterlidir.
Eski web arka plan anahtarı açılmaz; başka supervisor veya Auto-LoRA doğurulmaz.
Sohbet modeli ve `.env` değişmez. Üretici ve sorgu çevirmeni yalnız işçi sürecinde
izin verilen temel Ollama modeline sabitlenir. Sahte embedding yedeği kapalıdır.

## Eğitim sonrası başlama koşulu

- Reçetedeki `wait_for_adapters` (ilk paket: v14) tamamlanmış olmalı.
- Diskte `run_plan.json` bulunan diğer bütün adapter'lar ve `train_status.json`'ın
  gösterdiği koşu da tamamlanmış olmalı. İşaretin varlığı yetmez; hedef/fiilî adımlar
  eşleşmeli. Çöken, yarım veya bozuk kayıt otomatik tamamlanmış sayılmaz.
- Tamamlanmadan sonra 15 dakika beklenir. Bilinen canlı eğitim/eval/birleştirme
  süreçleri ve eğitim başlatma kilidi işi bekletir.
- En az 32 GB kullanılabilir RAM gerekir. STOP_ALL, STOP_LEARNING, STOP_RESEARCH
  işaretlerinden herhangi biri varken iş başlamaz.
- Gelecekte başlatılması planlanan fakat henüz diskte kaydı olmayan eğitimler
  kendiliğinden bilinemez: bekleme listesine eklenmelidir.

Her aşama başlamadan ve her iş biriminden önce kontrol tekrarlanır. Yönetici ayrıca
iki saniyede bir kontrol eder; yeni eğitim başlarsa yalnız kendi işçisini sonlandırır.
Bu, bütün Hektor süreçlerinin ortak hesaplama kilidi değildir: kontrol aralığında
kısa bir çakışma mümkün; Ollama'ya ulaşmış isteğin sunucuda anında iptali garanti değil.
Manuel sohbet/eval ile kaynak paylaşımı pilot sırasında ayrıca gözlenmeli.

## Aşamalar ve başlangıç bütçesi

| Aşama | En sık | Kapsam |
|---|---|---|
| Keşif | 24 saat | RAG, LoRA, RLM, matematik/istatistik; konu başına en çok 1 PDF |
| İçe alma | 6 saat | En çok 2 aday; dosya hash'i, PDF/metin kalitesi, kaynaklı alaka kapısı |
| Kart | 6 saat | En çok 2 makale; kartlar mevcut builder ile üretilir, otomatik onaylanmaz |
| Veri | 24 saat | En çok 6 yeni soru + 6 cevap girişimi; v14 öz-damıtma kapıları |
| Yöntem önerisi | 7 gün | En çok 2 yeni LoRA/RAG/RLM kaynağı; hipotez + deney önerisi |
| Rapor | 24 saat | Aşama sonuçları, inceleme bekleyenler, kesilmiş işler |

Aralıklar en erken tekrar zamanıdır; gerçek saat, çağırıcı ve iş sırasına bağlıdır.
Bir çağrıda yalnız bir aşama çalışır. Aşama süresi en çok 45 dakika. Üç ardışık aşama
hatasından sonra o aşama otomatik denenmez; diğer aşamalar ilerleyebilir. Makale ve
kart denemeleri ayrıca üçle sınırlıdır. Paket kilidi çökme sonrası kendiliğinden
silinmez: gerçekten işçi kalmadığı doğrulanarak kurtarılır. Boş sonuç ve ağ hatası
ayrılır; içerik üretmeyen tur kalite başarısı sayılmaz.

## Veri ve kalite sınırları

PDF'ler `data/research_package/inbox/` altında; kuyruğun kaydettiği hash uyuşmalıdır.
Dosya 40 KB–25 MB, metin en az 2000 karakter, sayfa başına en az 400 karakter ve
temiz metin skoru en az 6/10 olmalıdır. Bu bir ön elemedir; formül/tablo çıkarımının
eksiksiz olduğunu kanıtlamaz. Alaka değerlendirmesi makaleden doğrulanabilir birebir
alıntı ister. Alıntının varlığı, hipotezin bilimsel olarak doğru olduğunun kanıtı değildir.

Kabul edilen PDF mevcut idempotent `PaperIndexer` ile işlenir; korpus-geneli otomatik
sentez çağrılmaz. Kartlar inceleme bekler. Derin korpus bütünlüğü ve anlama skorlaması
bu ilk paketin ayrı geliştirme konusudur; mevcut ajanlarla ayrıca yürütülebilir.

Eğitim adayları `data/research_package/staging/distill_qa.jsonl` dosyasına yazılır.
Canlı `data/lora_sft/lora_sft.jsonl`, train/valid dosyaları ve v14 verisi değiştirilmez.
Soru üretimi kaynak parçalarına dayanır; değerlendirme soruları ve önceki damıtma
soruları dışlanır. Uzun/atıflı cevaplar mevcut `distill_one` kapılarını kullanır.
RAG'sız soru payı her turda en çok 1/6'dır (uygun soru bulunmasına bağlıdır).
Kabul edilen satırlar hâlâ ADAYDIR: veri birleştirme, yakın-kopya/sızıntı denetimi,
kalite kapıları, kaynak gruplu bölme ve eğitim için insan kararı ayrı aşamalardır.

Yöntem raporları yalnız kaynaklı aday öneridir. Haftalık çıktı bağımsız çok-ajanlı
derin doğrulama yerine geçmez; reçete/kod/üretim modeli kendiliğinden değiştirilmez.
Kaynaklarda önerilen yöntemlerin v14'ten üstün olduğu test edilmeden söylenmez.

## Ortak durum

- `storage/research_package/state.json`: son aşama/sonuç, deneme ve hata sayısı.
- `storage/research_package/service.json`: etkinlik, seçilen motorlar, son incelemeler,
  yönetici hatası ve yeniden deneme zamanı.
- `storage/research_package/papers.json`: aday → ret/içe alındı → kart/öneri durumu.
- `storage/research_package/questions_done.json`: soru başına sonuç ve üretici.
- `storage/research_package/<aşama>.log`: işçi çıktısı.
- `reports/research_package/status.json`: ortak rapor.
- `reports/research_package/methods-*.json`: kaynaklı yöntem adayları.

Yalnız ilk gerçek işin başlaması, anlamlı yeni aday/çıktı, hata veya insan müdahalesi
gerektiren durum bildirilir. Değişmeyen eğitim beklemesinde bildirim gönderilmez.

## Birlikte kararlaştırılacak sonraki geliştirmeler

1. **Kalite öncelikli alım:** korpus bütünlüğü + formül/tablo skoru + kanıtlı anlama
   değerlendirmesi. Yalnız makale sayısını artırmayı hedeflememeli.
2. **Hata odaklı kaynak arama:** v14'ün hata sınıflarını arama temalarına bağlamak;
   test sorularını eğitim verisine doğrudan taşımadan, ayrı doğrulama setiyle ölçmek.
3. **RAG sürümleme ve geri dönüş:** mevcut canlı indeksi büyütmek yerine aday indeks
   kurup sabit retrieval testinde karşılaştırmak. İlk pakette canlı indeks kullanılır.
4. **Gerçek derin yöntem incelemesi:** yöntem adayına bağımsız karşı inceleme,
   Hektor/GGUF uyumluluğu, maliyet ve tek-değişkenli deney planı. Otomatik kod yazımı
   yerine incelenebilir PR; eğitim ve üretim terfisi ayrı insan kararı.
5. **Ortak kaynak kilidi:** eğitim, Ollama, embedding, web ve eval arasında aynı
   kira/kilit protokolü; elle başlatılan işlerle de yarış penceresini kapatmak.
6. **Veri dengeleme:** uzun/kısa, kaynaklı/kaynaksız, çekimserlik ve disiplin dağılımını
   ölçmek; token ağırlıklarını otomatik değiştirmek yerine deney önerisi üretmek.

## Doğrulama ve etkinleştirme

Yeni paketin testleri gerçek Ollama, ağ veya eğitim başlatmadan çalışır. Canlı kabul
testi eğitim bittikten sonraki ilk sınırlı turlardır; ilk turdan önce canlı başarısı
iddia edilmez. Araştırma paketi kendisi işletim sistemi görevi kurmaz; bu sohbetin
zamanlayıcısı web yönetimi açılmadıysa `research-package --run` çağrısını ve değişiklik
bildirimini yönetir. Web yönetimi açıldığında yalnız izler. MCP'de yalnız durum GET
ucu bulunur; sürücü ajan kendi döngüsünü başlatamaz/durduramaz. MCP araç üretimi
in-process doğrulandı; değişikliklerin görünmesi için mevcut MCP oturumu yenilenmelidir.
