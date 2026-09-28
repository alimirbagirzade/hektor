# LoRA Karışım Profilleri + Profil Eval — Hektor

> Achilles için yazılmış "Adapter Mixing / Profile Routing / Evaluation" şartnamesinin
> Hektor'a uyarlanmış hâli. **Eğitim başlatmaz**, büyük model indirmez, base modeli
> değiştirmez (merge_and_unload YOK), production adapter'ı değiştirmez.

## 1. Ana fikir

- Tek base LLM (şu an `settings.llm_model` / `settings.peft_base_model`; 30B dahil) **değişmez**.
- **Bilgi = RAG.** LoRA bilgi depolamaz; davranış öğretir (matematik / istatistik / neden-sonuç /
  belirsizlik / trading metodolojisi / kod / hesap kontrolü / kaynak kullanımı / halüsinasyon azaltma).
- Beş **bağımsız** domain adapter'ı: `math`, `statistics`, `reasoning`, `trading`, `coding`.
  Her biri aynı base checkpoint'ten eğitilir; adapter üstüne adapter eğitimi registry'de reddedilir.
- Profiller bu adapter'ları **yeniden eğitmeden** ağırlıklı birleştirir. Katsayılar **semantik
  yüzde değildir** (PEFT ölçek katsayısı); gerçek performansı yalnız eval söyler.

## 2. Hektor'a uyarlama kararları

| Şartname | Hektor | Neden |
|---|---|---|
| `configs/lora_profiles.yaml` | `configs/lora/mix_profiles.yaml` | `configs/lora/lora_profiles.yaml` zaten **eğitim hiperparametreleri** için var; aynı ad iki kavramı karıştırırdı |
| `app/eval/` | `app/evals/profile/` | `app/evals/` zaten var (`eval_runner.py`, `retrieval_eval.py`); yan yana `app/eval` import karışıklığı yaratır |
| `data/eval/golden/` | `evals/profile_mix/` (+ `manifest.json` sha256) | `data/` commit edilmez (CLAUDE.md); golden **git'te** hash'le korunmalı |
| `python -m app.eval.X` | `python -m app.evals.profile.X` + `hektor mix …` | CLI kuralı: yeni komut `app/main.py` + README |
| Adapter registry | Yeni `app/lora/domain_adapter_registry.py` | Mevcut `app/lora/adapter_registry.py` (tek-adapter yaşam döngüsü) bozulmadı |
| — | **Her eğitimden önce ağırlık sorusu** | Kullanıcı isteği: `train --run` karar olmadan başlamaz (§6) |

## 3. Modüller

```
configs/lora/mix_profiles.yaml      profiller + merge yöntemleri + router eşlemesi
configs/eval/profile_eval.yaml      system prompt, üretim ayarı, eval_weights, regression gate
app/lora/mix_common.py              domain listesi, ağırlık ayrıştırma (regex, eval YOK), hash/parmak izi
app/lora/domain_adapter_registry.py domain adapter kaydı (tüm izlenebilirlik alanları)
app/lora/profile_registry.py        profil kaydı + durum makinesi
app/lora/profile_builder.py         uyumluluk denetimi + PEFT add_weighted_adapter (dry-run varsayılan)
app/lora/profile_router.py          v1 router: yalnız validated profiller, düşük güvende balanced
app/lora/weight_decision.py         eğitim-başı ağırlık kararı (tek kullanımlık)
app/lora/mix_cli.py                 `hektor mix …` alt komutları
app/memory/rag_version.py           RAG anlık görüntüsü + rag_version + RetrievalTrace
app/evals/profile/                  schema, dataset_loader, leakage, eval_runner, eval_registry,
                                    retrieval/answer/grounding/math/statistics/coding/reasoning/
                                    trading/abstention/consistency/regression eval, failure_diagnosis,
                                    judge, human_audit, score_aggregator, comparison, manifest, generators
evals/profile_mix/                  validation.jsonl · golden_test.jsonl · manifest.json
```

## 4. Registry'ler (`<root>/registry/lora/`, git'e girmez)

- **domain_adapters.jsonl** — adapter_id, adapter_version, base_model, base_model_revision,
  base_model_hash, tokenizer_hash, dataset_version, dataset_hash, training_config_hash, domain,
  r, lora_alpha, lora_dropout, target_modules, learning_rate, epochs, max_seq_length,
  training_seed, created_at, git_commit, eval_status, production_status (+ parent_adapter,
  serving_model). `base_model_hash` 30B ağırlığı baştan sona okumamak için config + ağırlık
  dosyası (ad, boyut) **parmak izidir**; tokenizer hash'i tam içeriktir.
- **profiles.jsonl** — profile_id (`<profil>_<yöntem>`), profile_version, base_model_version,
  rag_version, adapter_versions, adapter_weights, merge_method, merge_parameters, profile_hash,
  eval_dataset_version, eval_score, domain_scores, grounding_score, hallucination_score,
  regression_score, status. Durumlar: experimental → evaluating → validated/rejected →
  production → deprecated. **production = regression gate + `--approve`.** Aynı profile_id
  farklı içerikle yeniden kaydedilemez (eval geçmişi sessizce ezilmez).
- **router_decisions.jsonl** — query_id, detected_domains, selected_profile,
  router_confidence, fallback_profile, created_at.
- **weight_decisions.jsonl** — eğitim-başı ağırlık kararları (kim tüketti, ne zaman). En fazla
  bir açık karar: yeni kayıt ya da bir tüketim diğer açık kararları `superseded_at/_by` ile
  geçersiz kılar (silinmez); oku-yaz döngüsü `.lock` dosyasıyla korunur.
- Aynı profile_id + aynı profile_hash ile yeniden kayıt durumu, eval alanlarını, notları ve
  servis modelini korur (içerik değişmedi → eval kanıtı geçerli).
- Production'a karşı regression gate, iki koşunun manifestinde `eval_dataset_hash`,
  `rag_version`, `base_model_hash`, `eval_config_hash` eşleşmezse **kapalı** kalır.

## 5. Eval sistemi

- Her aday için **A** base · **B** base+RAG · **C** base+profil · **D** base+RAG+profil
  (`--matrix` ile tekil adapter satırları da). Aynı soru / system prompt / üretim parametreleri
  (temperature 0, seed 42) / max_tokens / RAG indeksi / retrieval ayarı / veri seti.
  Retrieval kalem başına **bir kez** yapılır → B ve D **aynı bağlamı** görür.
- RAG anlık görüntüsü koşu başında ve sonunda karşılaştırılır; değiştiyse koşu geçersiz.
- **Deterministik önce:** sayı+tolerans, exact match, unit test yürütme (izole alt süreç,
  statik güvensiz-kod elemesi, zaman aşımı), chunk_id ile retrieval/atıf. LLM judge yalnız
  reasoning/explanation/semantic/grounding için; yapılandırılmış alanlar saklanır, CoT saklanmaz,
  `can_promote=False`.
- **Metrikler ayrı:** domain accuracy'leri, retrieval recall@k/precision, context_relevance,
  answer/factual correctness, grounding, citation_accuracy, abstention_accuracy,
  hallucination_rate, consistency, latency, tokens/s, VRAM/RAM. Overall **yalnız rapor**.
- **Hata teşhisi:** retrieval / reranker / context_missing / context_noise /
  LoRA_reasoning_failure (base doğru, LoRA yanlış) / math / statistics / unsupported_claim /
  hallucination / citation / router / profile_merge_failure (tekil adapter doğru, profil yanlış)
  / base_model_failure.
- **Regression gate:** statistics & grounding düşemez, domain düşüşü > 0.03 engel,
  hallucination artamaz, abstention ≥ 0.80, domain başına ≥ 5 örnek, ölçülemeyen metrik = engel.
  Şartnamedeki örnek (Math 92→75, overall yükselmiş) testte **engellenir**.

## 6. Eğitim-başı ağırlık sorusu (kullanıcı kuralı)

`hektor train --run` şu sırayla ilerler: STOP_ALL → **ağırlık kararı** → load-doctor → taze onay
→ veri tazeleme → **sızıntı kapısı** → kararın tüketilmesi → eğitim.
Karar kaynakları: `--mix-profile X` / `--mix-weights "math=0.3,..."`, önceden `hektor mix weights`
ile kaydedilmiş taze (≤24 sa) karar (web butonu / `start-train.ps1` gibi etkileşimsiz yollar
için) ya da terminalde etkileşimli soru. Karar yoksa **exit 5**; sızıntı varsa **exit 6**.
Bayrakla verilen karar, eğitim bloklanırsa kayıt bırakmaz.

## 7. Sızıntı koruması

- exact / normalized (harf, noktalama, aksan) / contained (normalize eval metni ≥ 30 karakter
  daha uzun bir eğitim metninin içinde) / near-duplicate (kelime 3-gram Jaccard ≥ 0.8) /
  semantik (enjekte embedding kosinüsü ≥ 0.95, opsiyonel) / source-id örtüşmesi.
- `BAĞLAM: … SORU: <soru>` biçimli kullanıcı mesajının SORU bölümü ayrı metin olarak da
  denetlenir. `train.jsonl` yanında `valid.jsonl` varsa o da taranır.
- Golden `purpose="selection"` ile okunamaz; `--split golden_test` yalnız `--final` + profil
  validated/production iken. Manifest hash'i tutmazsa yükleme reddedilir.
- Statik test: `app/training`, `app/lora`, `scripts` golden/validation yolunu referans etmez.

## 8. İnsan denetimi

`hektor mix audit --run-id RUN --sample-size 100` → domain-dengeli, seed'li örnek.
`audit_blind.jsonl` otomatik karar/skor **içermez**; reviewer PASS/PARTIAL/FAIL + hata kategorisi
yazar. `--score` → uyum, Cohen kappa, **false PASS** (otomatik PASS, insan FAIL/PARTIAL) ayrı liste.

## 9. Dry-run akışı (LLM/GPU gerektirmez)

```bash
uv run hektor mix weights --show
uv run hektor mix build --profile balanced_v1 --merge-method svd     # adapter yoksa "blocked" planı
uv run hektor mix route "Sharpe oranının standart hatası nedir?"
uv run hektor mix eval --profile balanced_v1 --dry-run
uv run hektor mix leakage
```

## 10. Henüz ÇALIŞTIRILMAYANLAR

- Beş domain adapter'ının hiçbiri eğitilmedi (domain veri setleri de henüz yok).
- Gerçek profil kurulumu (`build --run`) ve GGUF→Ollama servis modeli yok → C/D sistemleri
  "koşulamadı" raporlanır (sessiz 0 değil).
- Gerçek A/B eval koşusu (Ollama) yapılmadı; golden/validation seed setleri Claude tarafından
  yazıldı, `human_verified=false` — **insan doğrulaması şart**. RAG'a özgü sorular için
  `expected_chunk_ids` korpusa göre eklenmeli (uydurulmadı).
- Weight search (kontrollü ağırlık araması) v2 işi; yalnız validation split'ine karşı yapılmalı.
