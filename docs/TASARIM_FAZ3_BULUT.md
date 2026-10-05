# Faz 3 — bulut desteği tasarımı (TASLAK, uygulanmadı)

_Durum 2026-10-05: yalnız tasarım. Bu belgede bulut çağrısı yapılmadı, proje talimatları
(CLAUDE.md "bulut sağlayıcı istemcisi kodda YOKTUR — kalıcı kısıt") DEĞİŞTİRİLMEDİ. Uygulamaya
geçmek kullanıcının açık kararıdır ve aşağıdaki talimat değişikliğini gerektirir._

## 0 · Ön koşul (sağlandı mı?)

Yerel Faz 2 açıkları kapanmadan buluta geçilmez. Bu turda kapatılanlar: strateji çevirisinin
görünür farkları + onay, gerçek veriyle motor denetimi, aday hattı web akışı, Kademe 2 reçete
bağlaması. **Açık kalan yerel iş:** gerçek bir aday için insan kör incelemesi + kabul kararıyla
ana model etkinleştirmesi (bkz. `docs/evidence/faz2_tamamlama_2026-10-05.md`).

## 1 · İki AYRI özellik

| | A · İkinci görüş | B · Eğitim için öğretmen çıktısı |
|---|---|---|
| Amaç | Yerel cevabı/strateji taslağını bulut modeline **eleştirtmek** (hata, eksik varsayım) | Bulut çıktısını yerel LoRA eğitim verisine **hedef** olarak koymak |
| Çıktının gittiği yer | Yalnız ekrandaki "ikinci görüş" kutusu + denetim günlüğü | `lora_sft.jsonl` (eğitim) |
| Doğruluk kanıtı mı? | **Hayır.** Bulut onayı/itirazı doğrulama sayılmaz; mevcut deterministik kontroller (hesap, kaynak, backtest kaydı) aynen geçerli | Hayır; ayrıca aşağıdaki hukuki engel |
| Varsayılan | Kapalı; tur başına açık tıklama | **Kapalı ve şu an UYGULANAMAZ** (madde 2) |

İkisi kodda ayrı modül, ayrı ayar, ayrı günlük olmalı; B'nin çıktısı A'nın kutusundan eğitime
"kopyalanamamalı" (eğitim verisine giden satırın kaynağı `cloud_teacher` etiketi taşır ve
sızıntı/kaynak kapısı bu etiketi reddeder — madde 2 çözülene kadar).

## 2 · Kullanım koşulları (güncel resmi kaynaklar, 2026-10-05'te okundu)

- **Claude Code** abonelikle (Free/Pro/Max) kullanıldığında **Tüketici Şartları**, Team/
  Enterprise/API ile **Ticari Şartlar** geçerlidir ([Legal and compliance][cc-legal]).
  Pro/Max limitleri "olağan, bireysel kullanım" varsayar; OAuth (abonelik) girişi Anthropic'in
  kendi uygulamaları içindir, ürün/servis geliştirenler API anahtarı kullanmalıdır.
- **Tüketici Şartları** (yürürlük 8 Ekim 2025) şunu yasaklar: Hizmetlerle rekabet eden ürün
  geliştirmek "including to develop or train any artificial intelligence or machine learning
  algorithms or models" ve API anahtarı dışında, açıkça izin verilmedikçe hizmete otomatik/
  insan dışı yollarla erişmek ([Consumer Terms][consumer]).
- **Ticari Şartlar** (yürürlük 17 Haziran 2025, D.4): "access the Services to build a competing
  product or service, including to train competing AI models … except as expressly approved by
  Anthropic" ([Commercial Terms][commercial]).
- **OpenAI** (Codex CLI ChatGPT planıyla): Kullanım Şartları, Çıktının OpenAI ile rekabet eden
  modeller geliştirmek için kullanılmasını yasaklar ([Terms of Use][openai]). _Not: resmi sayfa
  bu ortamdan otomatik okunamadı (HTTP 403); madde arama sonucuyla doğrulandı — uygulamadan önce
  sayfayı insan okumalı._

**Sonuç:** B (öğretmen çıktısıyla yerel model eğitimi) için izin **varsayılamaz**. Yerel bir
alan modelini bulut çıktısıyla eğitmek "rakip model eğitimi" yasağına girebilir; bu bir hukuki
yorum sorusudur ve ancak sağlayıcıdan yazılı izin / uygun ticari sözleşme ile açılabilir. A
(ikinci görüş) için abonelikli CLI'yı bir uygulama içinden otomatik çağırmak tüketici
şartlarındaki "otomatik erişim" ve "olağan bireysel kullanım" sınırlarına takılabilir; güvenli
yol **API anahtarı + Ticari Şartlar** ya da kullanıcının kendisinin elle çalıştırdığı CLI'dır.

## 3 · A · İkinci görüş — uygulanabilir taslak

- **Tetik:** yalnız insan tıklaması (`require_human`), tur başına; toplu/otomatik döngü yok.
- **Gönderilen içerik (en az):** soru + yerel cevap metni + (varsa) strateji taslağı JSON'u +
  deterministik kontrol sonuçları. **Gönderilmez:** makale tam metinleri/PDF, `data/`,
  `storage/`, sohbet geçmişinin geri kalanı, kimlik/anahtar, model ağırlıkları. Gönderilecek
  metin tıklamadan önce kullanıcıya AYNEN gösterilir ("şu gidecek").
- **Sağlayıcı:** kullanıcı seçer (Anthropic API / OpenAI API); anahtar `.env`'de, commit'lenmez.
  Abonelik CLI'sının uygulama içinden çağrılması yalnız sağlayıcı izni netleşirse.
- **Maliyet/kota:** istek başına tahmini token ve üst sınır (ör. 4k girdi / 1k çıktı) gösterilir;
  günlük harcama tavanı ayarı; tavan aşılırsa düğme kapalı + gerekçe.
- **İptal:** istek iptal edilebilir (zaman aşımı + durdur düğmesi); yarım cevap gösterilmez.
- **Kayıt:** `storage/cloud_second_opinion.jsonl` — zaman, sağlayıcı, model kimliği, gönderilen
  metnin özeti (sha256), token sayıları, maliyet tahmini, cevap. Bulut cevabı eğitim adayı
  OLMAZ, doğrulama kontrolü sayılmaz; ekranda "ikinci görüş — doğrulama değildir" etiketi.
- **Kural 1:** bulut cevabı da `correction_safety_reason` süzgecinden geçer (tavsiye dili).

## 4 · B · Öğretmen çıktısı — koşullu taslak

Yalnız sağlayıcının yazılı izni / uygun sözleşme belgelenirse. O zaman: ayrı onaylı veri
sürümü, satır başına kaynak etiketi + sağlayıcı/model/tarih, karışımda üst pay (ör. ≤ %10),
eval sızıntı kapısı, Kademe 2 kapsamına girer; aksi halde kod yolu açılmaz.

## 5 · Gerekecek talimat değişiklikleri (BU GÖREVDE YAPILMADI)

`CLAUDE.md` "Proje nedir" paragrafı şu an: _"LLM hattı yalnız yerel Ollama'dır; bulut sağlayıcı
istemcisi kodda YOKTUR ve eklenmez (kalıcı kısıt)."_ A uygulanacaksa önerilen metin:

```diff
-LLM hattı **yalnız yerel Ollama**'dır; bulut sağlayıcı istemcisi kodda YOKTUR ve
-eklenmez (kalıcı kısıt).
+Cevap/eğitim hattı **yalnız yerel Ollama**'dır. Tek istisna: insan tıklamasıyla, tur başına
+"ikinci görüş" (docs/TASARIM_FAZ3_BULUT.md §3) — çıktısı eğitime GİRMEZ, doğrulama sayılmaz,
+gönderilen içerik önceden gösterilir ve günlüğe yazılır. Bulut çıktısıyla eğitim yasaktır
+(sağlayıcı şartları; §2).
```

Ayrıca: "Mutlak kurallar"a _"Bulut cevabı kanıt değildir"_ maddesi; README'ye anahtar/maliyet
ayarları; `automation_manifest.yaml`'a dış ağ çağrısı kaydı; güvenlik testi (anahtar commit
taraması, gönderilen içerik beyaz listesi).

[cc-legal]: https://code.claude.com/docs/en/legal-and-compliance
[consumer]: https://www.anthropic.com/legal/consumer-terms
[commercial]: https://www.anthropic.com/legal/commercial-terms
[openai]: https://openai.com/policies/row-terms-of-use/
