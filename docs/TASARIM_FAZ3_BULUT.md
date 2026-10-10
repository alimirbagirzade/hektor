# Faz 3 — bulut desteği (hibrit, varsayılan KAPALI)

_Durum 2026-10-06: dal `claude/faz3-bulut-ikinci-gorus` (main'e birleşmedi). Gerçek/ücretli bulut
çağrısı YAPILMADI; testler sahte sağlayıcıyla. Anahtar istemci kodunda / Git'te YOK._

## 0 · Ön koşul

Faz 1–2 main'de (#30–#33). Bu tasarım yerel hattı değiştirmez: cevap ve eğitim hattı yalnız
yerel Ollama'dır. Bulut yalnız **insanın tur başına açıkça başlattığı** kontroldür.

## 1 · Kullanım koşulları — 2026-10-06'da yeniden okundu

| Kaynak | Ne diyor (kısa) | Sonuç |
|---|---|---|
| [Claude Code — Legal and compliance][cc-legal] | Free/Pro/Max → Tüketici Şartları; Team/Enterprise/API → Ticari Şartlar. OAuth (abonelik) "ordinary use of Claude Code and other native Anthropic applications" içindir; ürün/servis geliştirenler (Agent SDK dahil) API anahtarı kullanmalı. Pro/Max limitleri "ordinary, individual usage" varsayar. Kullanıcının kendi aboneliğiyle **değiştirilmemiş** Claude Code ikilisine girmesi engellenmez. | Abonelikli `claude -p`'yi kendi aracından, kendi hesabınla, tur başına elle tetiklemek bu metnin dışında AÇIKÇA yasak değil; ama "uygulama entegrasyonu" için önerilen yol API anahtarıdır. Otomatik/toplu döngü kapsam dışı. |
| [Tüketici Şartları][consumer] (8 Ekim 2025) | Rakip ürün geliştirmek "including to develop or train any artificial intelligence or machine learning algorithms or models" yasak; API anahtarı dışında, açık izin yoksa "automated or non-human means" ile erişim yasak. | **B (eğitim) abonelikle YASAK.** A yalnız insan tıklamasıyla. |
| [Ticari Şartlar][commercial] (17 Haziran 2025) D.4 | "build a competing product or service, including to train competing AI models … except as expressly approved by Anthropic". Çıktı müşterinindir. | A (API anahtarıyla ikinci görüş) uygun. B: yerel trading-araştırma LoRA'sının "rakip model" sayılıp sayılmadığı hukuki yorum → **yazılı onay olmadan açılmaz**. |
| OpenAI Kullanım Şartları / Codex kimlik doğrulama | "Use Output to develop models that compete with OpenAI" ve "automatically or programmatically extract data or Output" yasak; Codex: otomasyon/CI için API anahtarı öneriliyor. Resmi sayfalar bu ortamdan okunamadı (HTTP 403) — maddeler arama sonuçlarıyla doğrulandı; uygulamadan önce **insan okumalı**. | B ChatGPT/Codex aboneliğiyle YASAK; A için API anahtarı yolu. |

**B için uygun yöntem (dolanma değil):** çıktıları eğitimde kullanmaya lisansı açıkça izin veren
**açık ağırlıklı** bir öğretmen model (ör. Apache-2.0/MIT lisanslı büyük bir model; yerelde ya da
barındırıcının şartları çıktı kullanımını kısıtlamayan bir sağlayıcıda). Lisans metni + tarih +
sağlayıcı şartı kayda bağlanır; insan onayı olmadan politika açılmaz.

## 2 · A · Buluttan ikinci görüş (UYGULANDI, varsayılan kapalı)

- **Tetik:** sohbet turunda "Buluttan ikinci görüş" → **önizleme ekranı** (tek ekranda: gönderilecek
  metnin TAMAMI + sha256, sağlayıcı ve şart notu, istek başına tahmini token, günlük kota
  kullanımı, maliyet/kota açıklaması, iptal davranışı) → "Gönder" (`require_human`). Gönderim
  önizlenen metnin özetiyle eşleşmezse reddedilir (gösterilmeyen bir şey gitmez).
- **Gönderilen:** soru, yerel cevap, yerel deterministik kontrol özetleri. **Gönderilmez:** makale
  tam metinleri, kaynak parçaları, sohbet geçmişinin geri kalanı, `.env`, `data/`, `storage/`.
- **Saklama (üç ayrı alan):** `yerel` (cevap + özeti, değişmez), `bulut` (öneri metni, sağlayıcı,
  model, süre), `bagimsiz_dogrulama` (bulut metnine yerel deterministik kontroller: hesap,
  Kural 1 dili; sonuç ayrı alan). Bulut cevabı **doğrulama sayılmaz**, öğrenme adayı OLMAZ.
- **İptal:** iş dosyası + alt süreç; "İptal" süreci sonlandırır, yarım cevap saklanmaz
  ("iptal edildi" kaydı kalır). Zaman aşımı aynı davranış.
- **Sağlayıcılar:** `fake` (test), `claude_code_cli` (resmi `claude -p --safe-mode`, kullanıcının
  kendi aboneliği; araçsız). İkili mutlak yola çözülür (çalışma dizinindeki taklitçiye düşmez),
  `.cmd/.bat` sarmalayıcı reddedilir (cmd.exe argümanı yeniden yorumlar → enjeksiyon), CLI **boş
  geçici dizinde** koşar (projenin CLAUDE.md'si vb. bağlam olarak gitmez). Aynı tur + aynı metin
  için bekleyen istek varsa ikinci gönderim yeni istek açmaz. API anahtarlı sağlayıcı EKLENMEDİ (proje kararı "API yok" + anahtar
  yönetimi kullanıcı kararı). Açmak için `.env`: `HEKTOR_CLOUD_SECOND_OPINION=1`,
  `HEKTOR_CLOUD_PROVIDER=claude_code_cli`, `HEKTOR_CLOUD_TERMS_ACK=<YYYY-MM-DD>` (şartları
  okuduğunuz tarih), `HEKTOR_CLOUD_DAILY_MAX=<n>`.

**Genişleme (2026-10-10):** aynı kurallarla bulut hakem — K1 (eğitim sonrası kör paket, parça
başına tık) ve K2 (öğrenme adayı, aday başına tık); ortak günlük kota. Ayrıntı:
`docs/TASARIM_SUREKLI_DONGU.md`.

## 3 · B · Öğretmen çıktısından eğitim adayı (UYGULANMADI — kapı)

Kod yalnız **politika kapısı** içerir: `cloud_teacher` kökenli satır eğitim verisine giremez;
sağlayıcı politikası (`app/cloud/policy.py`) Anthropic abonelik/API ve OpenAI için `izin yok`
gerekçesi + kaynak bağlantısı taşır. Açık ağırlıklı öğretmen için politika ancak lisans kaydı +
insan onayıyla açılabilir (bu görevde açılmadı). Öğretmen çıktısı açılsa bile: bağımsız doğrulama
(hesap/kaynak/backtest kaydı) + insan onayı + ayrı veri sürümü + karışımda üst pay + Kademe 2.

## 4 · Kalan kararlar (sizin)

1. A için sağlayıcı: (a) abonelikli Claude Code CLI, tur başına elle — mevcut kurulum, ek maliyet
   yok, kota abonelikten; (b) Anthropic API anahtarı (Ticari Şartlar) — kullanım başına ücret,
   anahtar `.env`'de, "API yok" proje kararının değişmesi gerekir.
2. Günlük üst sınır (öneri: 10 istek/gün) ve aylık bütçe (API seçilirse).
3. B: açık ağırlıklı öğretmen model + lisans + nerede koşacağı (yerel RAM/VRAM yetmeyebilir).

[cc-legal]: https://code.claude.com/docs/en/legal-and-compliance
[consumer]: https://www.anthropic.com/legal/consumer-terms
[commercial]: https://www.anthropic.com/legal/commercial-terms
