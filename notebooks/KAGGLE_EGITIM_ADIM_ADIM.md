# Achilles LoRA Eğitimi — Adım Adım (ÜCRETSİZ, Kaggle T4×2)

> Bu kılavuz teknik bilgi varsaymaz. Sırayla, hiçbir adımı atlamadan yap.
> **Hugging Face hesabına GEREK YOK.** Sadece ücretsiz bir Kaggle hesabı yeter.
> Tahmini süre: hesap kurulumu ~5 dk + eğitim ~15-20 dk.

Elinde HAZIR olan 2 dosya (bu klasörde):
- `notebooks/achilles_lora_stage2.ipynb` ← eğitim defteri
- `data/lora_sft/lora_sft.jsonl` ← eğitim verisi (671 örnek, kalite kapısı: GO)

---

## ADIM 1 — Ücretsiz Kaggle hesabı aç (~2 dk)
1. Tarayıcıda aç: **https://www.kaggle.com**
2. Sağ üst **Register** > **Register with Google** > kendi Gmail'inle gir (alimirbagirzade@gmail.com).
3. Açılan formda kullanıcı adı seç, onayla.

## ADIM 2 — Telefon doğrulaması yap (GPU + İnternet için ZORUNLU) (~2 dk)
> Kaggle, ücretsiz GPU ve internet erişimini ancak telefon doğrulanınca açar.
1. Sağ üstte profil resmin > **Settings**.
2. Aşağı in, **Phone Verification** bölümü > telefon numaranı gir > gelen SMS kodunu yaz.
3. "Verified" yazısını gör.

## ADIM 3 — Eğitim verisini Kaggle'a yükle (Dataset olarak) (~2 dk)
1. Sol menü **Create** (veya **+**) > **New Dataset**.
2. Açılan pencereye bilgisayarından şu dosyayı **sürükle-bırak**:
   `C:\Users\sevinc\Development\achilles\data\lora_sft\lora_sft.jsonl`
3. Üstteki başlık (Title) kutusuna yaz: **achilles-lora-sft**
4. Sağ üst **Create** > yükleme bitsin (yeşil onay).

## ADIM 4 — Eğitim defterini yükle (~1 dk)
1. Sol menü **Create** > **New Notebook**.
2. Açılan notebook'ta üst menü **File** > **Import Notebook** > **Upload**.
3. Şu dosyayı seç: `C:\Users\sevinc\Development\achilles\notebooks\achilles_lora_stage2.ipynb`

## ADIM 5 — GPU + İnternet + Veriyi bağla (~1 dk)
Sağ panelde (görünmüyorsa sağ üstteki **⟩** okuyla aç):
1. **Settings > Accelerator** > **GPU T4 x2** seç.
2. **Settings > Internet** > **On** (açık) yap.
3. **Input** bölümü > **Add Input** > **Datasets** sekmesi > arama kutusuna `achilles-lora-sft` yaz > kendi yüklediğin dataset'e **+** ile ekle.

## ADIM 6 — Çalıştır (~15-20 dk, beklerken bir şey yapma)
1. Üst menü **Run** > **Run All** (veya ▶▶ "Run All" düğmesi).
2. Hücreler sırayla yeşil tik alır. En sonda "GGUF: X GB" ve indirme satırlarını görürsün.
   - İlk kez base model (~8 GB) inerken birkaç dk sürebilir — normaldir.

## ADIM 7 — Sonucu indir (~1 dk)
Eğitim bitince çıktı dosyalarını bilgisayarına indir:
- **Yol A (Colab değil, Kaggle):** sağ panel **Output** sekmesi > şu 2 dosyayı indir:
  - `achilles-Q4_K_M.gguf`  (~2.5 GB — model)
  - `Modelfile`             (küçük metin)
- İkisini de **aynı klasöre** koy, örn: `C:\Users\sevinc\Downloads\achilles-gguf\`

## ADIM 8 — Modeli yerelde kur (Ollama)
> Ollama kurulu olmalı (zaten var). Bu klasöre gir ve modeli oluştur.
PowerShell'de:
```powershell
cd C:\Users\sevinc\Downloads\achilles-gguf
ollama create achilles -f Modelfile
```
"success" görürsen model kuruldu.

## ADIM 9 — Eğitildi mi diye SINA (CLAUDE.md kural 2: kurmak ≠ çalışıyor)
PowerShell'de proje klasöründe:
```powershell
cd C:\Users\sevinc\Development\achilles
$env:ACHILLES_LLM_MODEL='achilles'
uv run achilles evaluate evals/discipline_core.jsonl
```
Bu, modelin disiplin kurallarına (garanti vaadi yok, maliyet, backtest şartı) uyup
uymadığını ölçer. Sonucu bana getir — **geçerse** koşullu terfi (base ile kıyas),
**geçmezse** REJECT + reçete revizyonu yaparız (v5'te olduğu gibi).

---

### Sık sorunlar
- **"GPU yok / kota" hatası:** Telefon doğrulaması (Adım 2) eksik. Tamamla.
- **"JSONL bulunamadi" hatası:** Adım 5.3'te dataset'i Input olarak eklemeyi atladın.
- **İndirme yavaş/koptu:** Kaggle Output sekmesinden tek tek tekrar indir.
- **Colab kullanmak istersen:** notebook Colab'i de destekler (Runtime > T4 GPU);
  ama oturum kopabilir, Kaggle daha güvenli.
