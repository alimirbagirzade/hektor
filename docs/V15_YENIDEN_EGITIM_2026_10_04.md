# v15 ayrı yerel revizyon deneyi — 4 Ekim 2026

Durum: eğitim tamamlandı; sayısal birleşim kapısı geçti, cevap kalitesi reddedildi. Üretim varsayılanı değiştirilmedi. v13/v14 dosyaları korunuyor.

## Önceki reddin nedeni ve onarım

Eski pilotun standart PEFT forward ile bf16 birleşimi arasında 117 token konumunda en yüksek KL 0.46863007545 ölçüldü; sabit 0.01 sınırı aşıldı. Bu ret cevap kalitesi ölçümü değildi. fp32 projection denemesi de geçmedi.

Deneysel merge-aware forward, frozen bf16 base ile fp32 LoRA delta toplamını bf16 ağırlığa çevirip tek linear işlem yapar. Cast geri türevi yaklaşık straight-through davranışıdır. Standart PEFT numeriğiyle aynı yöntem olduğu iddia edilmez. Dropout/NEFTune kapalıdır. Yeni adapter 384/384 tensörle yüklendi; üç probun 117 konumunda birleşim ve ayrı süreçte yeniden yükleme farkı/KL sıfırdı. Bu yalnız belirtilen prob ve recipe kapsamındadır.

## Gerçek eğitim ve doğrulamalar

- Base: Qwen3-30B-A3B-Instruct-2507, immutable yerel revizyon `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`; v14 kesin revizyonu bilinmediğinden ayrı deneydir.
- 73 train, 4 valid; 54 küçük sayısal kod örneği gerçekten çalıştırıldı, 19 kavramsal örnek. Bu 54 ayrı bağımsız algoritma değildir.
- r16, alpha32, q/k/v/o; LR5e-5, GA8, assistant-only example-weighted loss, seq6144, seed42.
- 10 optimizer adımı, 986.3 saniye CPU eğitimi. İlk warmup adımında LR0; dokuz adımda LR sıfırdan farklı. Son checkpoint önceden seçildi.
- Adapter SHA256: `209a8dd4e5e34cbce6d98d953c295881eaab527610bca343ae3327df92e4d074`.
- Son tam çevrimdışı suite: 2975 geçti, 4 deselected, 1 warning; ruff/format/mypy geçti.
- BF16 GGUF'taki 192 attention projection tensörü HF birleşimine byte eşit; Q4_K_M modelde bunlar Q8_0. Bu tüm model logits eşdeğerliği değildir.
- Quantization KL aracı istatistikten sonra heap-corruption çıkışıyla çöktü (`-1073740940`); başarılı test sayılmadı. Ham istatistik tanısaldır.

## Ortak koşullardaki ilk cevaplar

Base/v13/v14 için daha önce bu oturumda üretilmiş 21 cevap ve yeni v15 için 7 cevap birlikte incelendi. Temp0, seed42, ctx16384, predict4096, repeat penalty1.1, aynı system/template/soru hashleri; RAG kapalı. Eski PDF cevapları ortak ölçüm diye kullanılmadı. Etiketleri maskelenmiş iki tamamlanmış inceleme ve bir kısmi inceleme var; önceki bağlam yüzünden tam körlük garantisi yok.

Yeni aday kritik hatalar taşıyor: S07'de standart alt Bollinger bandını üstten büyük sayıyor; S09 kodu gerçek çalıştırmada skaler `.shift` ile hata veriyor; S20 ve S28 bilgi erişimi/gecikme çelişkileri var; S24 Gaussian return-only HMM varyans kapasitesini yanlış açıklıyor; S29 veri olmadan maliyet ve getirilerin gerçekten hesaplandığını söylüyor. S14 ana hesabı doğru. Otomatik biçim bayrağı olmaması içerik doğruluğu değildir.

Karar: **KABUL EDİLMEDİ**. Küçük geliştirme altkümesi tam benchmark değildir. 60 yeni soru, final kabul, tam 30 soru ve 2×2 RAG karşılaştırması bu aday için çalıştırılmadı. Önceden belirlenen kritik hata kapısı nedeniyle final kilitli kaldı. %85 genel başarı veya v14'e üstünlük iddiası yok.

## Yerel kanıtların konumu

`reports/v15/repair_r3/`: training_config/result/summary, code_oracle_evidence, merge_attempt, reload_verification, gguf_tensor_verification, final_code_suite.log/xml, code_checks/s09/execution.json, common/critical/raw.jsonl ve common_condition_verification.json.
Önceki ortak cevaplar: `reports/v15/common/critical/raw.jsonl`.

Veri, modeller, PDF içerikleri, özel final anahtarları ve büyük ham runtime dosyaları GitHub'a eklenmez; yerelde korunur. GitHub'da yöntem, testler, yeniden üretim scriptleri ve bu özet bulunur. Yeni eğitim adayları ayrı klasörler kullanmalıdır; eğitim başarısı kabul kararı değildir.
