# Sohbetten Öğrenme — Faz 1 protokolü

_Kapsam: tek sohbet ekranı, gerçek geçmiş aktarımı, model kimliği, geri bildirim, aday
doğrulama ve sürümlü eğitim verisi bağlantısı. **Eğitim başlatma, aday/mevcut model
karşılaştırması, etkinleştirme/geri dönüş ve seçilen stratejinin gerçek veride backtest'i
Faz 2'dir.** Bulut kontrolü Faz 3'tür; Faz 1'de hiçbir bulut çağrısı yoktur._

## Kullanıcı akışı

1. **00 · SOHBET** — soruyu yaz. Üst şerit: cevaplayan modelin etiketi + Ollama digest'i, ayar
   kaynağı (`HEKTOR_CHAT_MODEL` ya da boşsa `HEKTOR_LLM_MODEL`), köken kaydı (yoksa "köken kaydı
   yok"), kaynak durumu. Her cevapta: kaynaklar, kontroller, **modele aktarılan önceki turlar**
   ("göster" ile tam metin). Model çağrılmayan turda "model çağrılmadı" yazar.
2. Cevabın altında:
   | Düğme | Etki | Eğitim adayı? |
   |---|---|---|
   | Faydalı | Tura geri bildirim | Hayır |
   | Hatalı | Hata kuyruğu (seçili metin işaretlenir, not alınır) | Hayır; puanlanabilir test de değil |
   | Düzelt | Cevabın kopyası düzenlenir → hedef metin | **Evet** (turda tek aktif düzeltme) |
   | Öğrensin | Model cevabı hedef olur (tek tıklama) | **Evet** |
   | Eğitimden hariç tut | Tur + adayları eğitimden çıkar (geri alınabilir) | Adaylığı kaldırır |
3. **16 · ÖĞRENME HAVUZU** — sayaçlar (Toplandı / Doğrulandı / Eğitimde kullanıldı ayrı),
   inceleme kuyruğu, hata kuyruğu, reddedilenler, hariç/çatışma/sızıntı, eski Echo kayıtları
   (salt okunur). "Özet göster" → "Veri sürümü oluştur".
4. Eğitim verisine bağlama (Faz 1'de CLI, Faz 2'de arayüz):
   `uv run python scripts/assemble_sft.py --chat-dataset chat_vN` (kaldırmak: `--no-chat`).

## Geçmiş

- Konuşmanın **tamamı** `chat_turns`'te saklanır; her tur modele GİDEN istemi (sistem + kullanıcı)
  birebir tutar.
- Modele son `HEKTOR_CHAT_HISTORY_TURNS` (2) cevaplanmış tur, `HEKTOR_CHAT_HISTORY_CHAR_BUDGET`
  karakter bütçesiyle aktarılır; blok başlığı: "yalnız bağlamdır; doğrulanmış kaynak DEĞİLDİR".
- Takip sorusunun retrieval sorgusu = soru + önceki kullanıcı sorusu (deterministik).
- Geçmiş verilmeyen istem (`build_rag_prompt(history=None)`) öncekiyle **bayt-aynıdır**
  (öz-damıtma verisi etkilenmez).

## Doğrulama (LLM'siz, kapsamı görünür)

| Kontrol | Ne kanıtlar | Ne KANITLAMAZ |
|---|---|---|
| Hesap | `a op b = c` ifadesinin yeniden hesabı (`safe_eval`, eval/exec yok) | Açıklamanın geri kalanı |
| Kaynak | İfadenin turun KENDİ parçalarıyla sözcüksel örtüşmesi | Anlamsal doğruluk |
| Atıf kimliği | `[paper:chunk]` turda getirildi mi | İddianın desteklendiği |
| Güvenlik | Kural 1 (garanti/tavsiye/kesinlik, sır/PII) | — |
| Backtest | Faz 2'ye kadar **yapılamaz** | — |
| Kod testi | Test koşucusu yok → **yapılamaz** | — |

Karar: çürüten kontrol (yanlış hesap, getirilmeyen atıf kimliği, Kural 1) → **reddedildi**;
insan onayı bunu geçemez, önce metin düzeltilmeli. Tüm ifadeler kapsandıysa → **eğitime uygun
(otomatik kanıtlı)**. Kısmi kapsam → **inceleme bekliyor**; gerekçeli insan onayı (≥10 kr)
→ **eğitime uygun (insan onaylı)**, kısmi kapsam bilgisi korunur. Kontrol edilecek ifade
bulunamazsa → inceleme ("doğrulama başarısı değildir"). Trading performans iddiası backtest
olmadan onaylanamaz. Önceki model cevapları hiçbir kontrolde kanıt sayılmaz.

## Aileler, bölme, sızıntı

- Soru benzerliği (normalize + kelime 3-gram Jaccard ≥ `HEKTOR_LEARNING_FAMILY_JACCARD`) → aile.
- Train/eval ataması **aile düzeyinde ve kalıcı** (seed'li özet). Yeni örnek farklı bölmelerdeki
  aileleri birbirine bağlarsa → **sızıntı çatışması**; otomatik eğitime girmez.
- Aile içinde aynı soru+hedef → **yinelenen**; farklı yeniden-hesaplanmış sayısal sonuç →
  **çatışma**. Anlamsal çelişki otomatik tespit edilmez.
- Trading aileleri her sürümde **zaman sırasıyla** bölünür (en yeni pay eval); eğitim
  ailelerinin hepsi değerlendirme ailelerinden eskidir.
- Eğitim satırı (gerçek istem + hedef) korunan eval setleriyle (`leakage_eval_items`) taranır;
  geçmişte gizli soru varsa o tur da sızıntıdır. Sohbet içeriği RAG indeksine **yazılmaz**.

## Veri sürümleri

- `data/learning/chat_datasets/chat_vN/`: `train.jsonl` (SFT), `eval.jsonl` (**SFT değil**,
  eğitime girmez), `manifest.json` (üyeler, sınıflar, sayılar, token yöntemi, özetler).
  `version_store`'a `chat_sft` olarak kaydedilir. Aynı içerik → aynı sürüm. 50 aile eşiği
  yalnız eğitim hazırlığı içindir; daha az örnekle sürüm oluşturulabilir.
- Kanonik birleştirme yalnız `data/lora_sft/chat_selection.json` varsa sohbet **train**
  satırlarını ekler. Sınır: sohbet / (taban + sohbet) **satır VE eğitim hedef token payı ≤
  `HEKTOR_LEARNING_CHAT_MAX_SHARE` (0.10)**; eksiltme aile düzeyinde, seed'li sırayla; tekrarla
  çoğaltma yok. Token yöntemi raporlanır (PEFT base tokenizer yerelde yoksa "yaklaşık:
  karakter/4"). Taban boşsa sohbet satırı girmez.
- Anlık görüntüler **değiştirilmez**. Hariç tutma/ret/düzenleme sonrası seçili sürümde geçersiz
  kayıt varsa `pretrain-gate` ve web/CLI eğitim başlatma kapısı **durdurur**; yeni sürüm istenir.
  Hariç tutma önceden eğitilmiş bir modelden bilgiyi silmez.
- "Eğitimde kullanıldı": bağlama = **planlandı**; `train_status.json` `data_sha256` eşleşmesi →
  **sürüyor / tamamlandı (`run_complete.json`) / başarısız**. Gözlem panelde "↻ Yenile" ile.

## Kaynak koruması (sunucuda)

- Eğitim yoksa cevap verilir; cevap sonrası `/api/ps`'ten bellek ayak izi (RAM = toplam − VRAM,
  VRAM, bağlam uzunluğu, digest) kaydedilir.
- Eğitim sürüyor/başlatılıyorsa: model zaten yüklüyse izin; değilse **ölçülmüş** RAM kısmı +
  pay ≤ boş RAM VE VRAM kısmı ≤ boş VRAM olmalı. Toplam model boyutu ek RAM sayılmaz. Ölçüm
  yoksa ya da ölçülemiyorsa cevaplama kapalıdır; geçmiş ve inceleme açık kalır. Ölçüm eğitim
  dışında ilk cevapta ya da **Kaynak ölç** ile alınır (kalıcı kilit yok).
- Yarış: cevap üretilirken `storage/chat_leases/` kirası tutulur. Sohbet kirayı yazıp sonra
  eğitim kilidine bakar; `detached_launch` kilidi alıp sonra kiralara bakar (ön-kontrolde de).
- **Sınır:** `scripts/start-train.ps1` / elle `hektor train --run` yolu kira denetimi yapmaz;
  sohbet bu koşuları `train_status.json` pid'i ve log tazeliğiyle görür.

## Yetki

Aday oluşturma (Öğrensin/Düzelt), düzenleme, gerekçeli onay, veri sürümü oluşturma ve ölçüm
**yalnız insan** kapsamındadır (`require_human`); motor kendi eğitim verisini üretemez.
Sürücü kapsamı konuşmaya yazamaz. Yeni uçlar MCP allow-list'ine eklenmedi.

## Geri alma

- `uv run python scripts/chat_learning_rollback.py` (rapor) / `--apply` (yedek + YALNIZ yeni
  tabloları siler; Echo dahil diğer tablolara dokunmaz).
- `data/learning/chat_datasets/` ve `data/lora_sft/chat_selection.json` elle kaldırılır.

## Bilinen sınırlar (Faz 1)

- Kaynak desteği sözcüksel örtüşmedir; anlamsal doğrulama değildir.
- Gerçek bir Ollama cevabının uçtan uca akışı yalnız çevrimdışı sahte LLM ve yalıtılmış test
  sunucusuyla doğrulandı; gerçek 30B modelle canlı tur, "Kaynak ölç" ve eğitim sırasındaki
  davranış canlı ölçülmedi.
- Kod adayları hiç otomatik doğrulanamaz; trading performans iddiaları Faz 2'ye kadar onaylanamaz.
