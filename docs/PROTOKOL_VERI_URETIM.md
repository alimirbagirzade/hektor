# Stage 1 — Veri Üretim Protokolü

> Amaç: Lokal CPU'da, eğitim yapmadan, makale chunk'larından **grounded sentetik
> QA** üreterek 15-50 örnekten **≥1000 çeşitli SFT örneğine** çıkmak. Bu, anlamlı
> bulut-GPU LoRA eğitiminin (Stage 2) ön koşuludur. Üst protokol:
> [PROTOKOL_ASAMALI_EGITIM.md](PROTOKOL_ASAMALI_EGITIM.md).

## Akış (gece döngüsü — `scripts/continuous-learning.sh`)
Her tur sırayla:
1. **ZENGİNLEŞTİR** — arxiv'den sıradaki konudan ~3 makale (dönüşümlü konu listesi:
   davranışsal finans, Knightian belirsizlik, olasılık felsefesi, rejim değişimi…).
2. **KAVRA** — kartsız makalelere bilgi kartı + içerikli onay + anlama skoru.
3. **SENTEZLE** — her 3 turda research hipotezi + sentez makalesi.
4. **VERİ-ÜRET** — `synth-qa` ile yeni makalelerden grounded QA üret (birikir).
   *(Eskiden burada CPU-LoRA eğitimi vardı; kaldırıldı — bkz. redesign.)*

Başlat: `bash scripts/continuous-learning.sh 72` (72 saat; `storage/STOP_LEARNING`
ile nazikçe durur). Web panosu: http://127.0.0.1:8765.

## Sentetik QA üretici (`synth-qa`)
```bash
uv run achilles synth-qa --per-chunk 5 --max-chunks 12 --max-papers 0 --seed 0
# --append (vars.): mevcut dosyaya birikir; --overwrite: sıfırdan
```
**Ne yapar:** her chunk'tan N grounded soru-cevap üretir.
- **Persona çeşitliliği:** kantitatif araştırmacı / risk yöneticisi / backtester /
  şüpheci denetçi (aynı pasajdan farklı bakışlar → tek-tip olmayan veri).
- **RAG-stili SFT:** pasaj BAĞLAM olarak kullanıcı mesajına gömülür → model
  bağlamı kullanmayı öğrenir (nihai RAG+LoRA hibridiyle uyumlu).
- **Grounding kapıları (uydurma yasak — CLAUDE.md kural 7):**
  1. **Sayı-altküme:** cevaptaki her metrik (oran/yüzde/≥2 haneli) pasajda olmalı —
     uydurulan "Sharpe 2.3" / "%47 getiri" reddedilir.
  2. **Bilgi-fakir pasaj reddi:** yeterli anchor yoksa örnek üretilmez.
  3. **Anchor örtüşmesi:** cevap pasajla en az bir teknik terim/sayı paylaşmalı.
- **Determinizm:** `--seed` örneklemeyi sabitler (CPU FP nedeniyle bit-bazında değil,
  *yaklaşık* tekrar-üretilebilir — kural 6 mekanizması mevcut).
- **Atomik yazma + içerik-temelli dedup:** timeout/çökme dosyayı bozmaz; turlar-arası
  near-duplicate elenir.

Çıktı: `data/lora_sft/synthetic_qa.jsonl` — her satır
`{"messages":[{"role":"system",…},{"role":"user",…},{"role":"assistant",…}]}`.

## İlerleme ve eşik takibi
```bash
uv run achilles lora-readiness   # sentetik örnek sayısı + ≥1000 eşik durumu
uv run achilles rag-mastery      # RAG kapsama/anlama/hazırlık panosu (LLM-free)
```

## Kalite ilkeleri (Stage 2'ye temiz veri geçsin)
- Düşük sıcaklık üretimi değil, **grounding kapıları** kaliteyi korur.
- Hedef karışım: ~%15-20 abstention/refusal örneği (kaynak yetersizse "bilmiyorum").
- Stage 2'den önce **mutlaka** `lora-audit` (Gate 0-7) + OOS bölme.

## Çıkış kriteri (GATE)
≥1000 örnek **VE** `lora-audit` geçti **VE** kullanıcı onayı → **Stage 2**
([PROTOKOL_BULUT_EGITIM.md](PROTOKOL_BULUT_EGITIM.md)).
