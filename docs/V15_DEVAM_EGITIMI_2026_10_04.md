# v15 üç epoch deneyinin sonucu — 4 Ekim 2026

**v15 kabul edilmedi.** Eğitim ve native bf16 aktarım kapıları tamamlandı; kritik cevap hataları devam ediyor. Üretim varsayılanı değiştirilmedi.

Aynı kilitli 73 eğitim ve 4 validation örneğiyle ayrı bir adaptör, sıfırdan üç tam epoch eğitildi. Gerçek kapsam: **30 optimizer adımı ve 219 örnek işlenişi**. Config'teki iterations=240 yalnız 30 adımlık dönüşüm bütçesidir. Son checkpoint önceden sabitlendi. CPU Trainer süresi yaklaşık 50,5 dakika; dört validation örneğinin son kaybı 3,2253. Bu genel cevap doğruluğu ölçümü değildir.

Base revizyonu `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`; v14'ün kesin revizyonu bilinmediğinden ayrı deneydir. Deneysel merge-aware bf16 forward yaklaşık cast gradyanı kullanır. Recipe: r16, alpha32, q/k/v/o, LR5e-5, GA8, assistant-only example-weighted loss; dropout ve NEFTune kapalı.

384/384 LoRA tensörü yüklendi. Üç probun toplam 117 token konumunda native bf16 birleşim ve yeniden yükleme farkı/KL sıfırdı; 0,01 eşiği gevşetilmedi. GGUF'ta 192 attention tensörünün bf16 byte eşliği ve Q8_0 türü doğrulandı. Bu, tüm model logit eşdeğerliği değildir. Nicemleme KL aracı CPU'da da istatistikten sonra `-1073740940` ile çöktü; başarılı test sayılmadı.

Yedi kritik sorunun ilk cevapları sabit decoding, system, template ve soru koşullarında üretildi. Önceki aynı gün baseline 21 cevabı ve önceki v15'in yedi cevabıyla toplam 35 ham cevap karşılaştırıldı. Baseline'lar şimdi yeniden çalıştırılmış gibi sunulmadı.

Yeni aday S28'de zero shift kimliği, S29'da evrensel gap ve log-loss sınırı, S24'te filtering/smoothing ayrımında hata yaptı. Dört deterministik karşı örnek gerçekten çalıştırıldı. S07'de ters bant sırası doğru sınıflandırılıyor; S14'te Kalman ana hesabı doğru. Kritik kapı yine başarısızdır.

Kritik ret kapısı nedeniyle 30 yeni final sorusu, tüm 30 eski soru ve 2×2 RAG koşusu çalıştırılmadı. İnsan/kör tam benchmark puanı, %85 genel başarı veya v14'e istatistiksel üstünlük iddiası yok. v13/v14 dosyaları ve Ollama digestleri, eski pilot ve önceki v15 adaptörü korunuyor.

Yerel kanıtlar `reports/v15/continuation_r4/` altında: `README.md`, `decision.json`, `training_summary.json`, `common/critical/raw.jsonl`, `critical_counterexamples_execution.json`, `merge_attempt.json`, `reload_verification.json`, `gguf_tensor_verification.json`, `quant_kl_exit.json`, `preservation_final.json`. Ham veri, modeller ve özel final anahtarları GitHub'a eklenmez.

#29 CI düzeltmesi `661933e`, main'e `afa3edc` olarak birleşti. CI değişikliği yalnız dinamik metot bağlama için tip sınırıdır; eğitimde import edilmiş özgün kaynaklar yerelde saklandı. Yerel tam suite 2975 geçti; GitHub'da 2954 geçti, 12 atlandı, 4 deselected. Son 26 PEFT binding testi bu sayılara ayrıca toplanmaz.

Adaptör SHA256: `0f785092758e408cea60cf568d951cfb73b779d73a188d8338c897198e4d3fa2`.
