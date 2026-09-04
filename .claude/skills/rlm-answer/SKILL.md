# /rlm-answer — RLM Controller (çok-adımlı, kaynaklı, denetimli cevap)

Makale havuzu üzerinde bir soruyu **çok-adımlı retrieval + iddia-düzeyi doğrulama**
ile cevaplar. Tek-tur `ask`'tan farkı: kanıt yeterlilik kapısı, sorgu yeniden-
formülasyonu, taslağı iddialara bölüp her birini atıf/dayanak ile doğrulama,
desteklenmeyen iddiayı atma, ve yetersiz kaynakta **uydurmak yerine "yeterli kaynak
yok" deme**. RLM yeni bilgi deposu DEĞİLDİR — mevcut RAG + verifier'ları orkestre eden
bir kontrol katmanıdır (`app/rlm`).

## Ne zaman kullan

- Çok makaleli sentez / karşılaştırma / literatür sorusu (tek-tur RAG yüzeysel kalır).
- Cevabın **kaynaklı ve doğrulanmış** olması kritikse (her iddia chunk'a dayanmalı).
- Trading-içerikli soru (çıktı yalnız HİPOTEZ + zorunlu uyarı olmalı — kural 1).
- "Bu makalelerde bu soruya cevap var mı?" — yetersizse sistem çekimser kalmalı.

Tek bir makaleden hızlı, doğrulamasız özet yeterliyse `/rag-reliability-engineer`
veya `achilles ask` kullan; RLM daha ağırdır (çok-tur + doğrulama).

## Temel komutlar

```bash
# Çok-adımlı kaynaklı cevap (genel havuz)
uv run achilles rlm-answer "Bu makalelere göre volatilite rejimi momentum'u nasıl etkiler?"

# Belirli makale(ler)e sınırla
uv run achilles rlm-answer "Bu makalenin metodolojisi nedir?" --paper-ids paper_abc123

# Tur/chunk ayarı
uv run achilles rlm-answer "..." --rounds 3 --top-k 8

# Kayıtlı koşuları listele (durum/kanıt/güven)
uv run achilles rlm-runs

# §16 LoRA dataset ADAYLARI (salt-okuma; eğitim YOK, insan onayı şart)
uv run achilles rlm-lora-candidates [--export data/rlm_lora_candidates.jsonl]
```

API: `POST /api/rlm/answer` · `GET /api/rlm/runs` · `GET /api/rlm/runs/{run_id}`.

## LoRA aday seçimi (§16)

Yüksek-güvenli RLM koşuları ileride LoRA için aday olabilir — `app/rlm/lora_candidate.py`.
Eşik (§16): `final_confidence≥0.85 ∧ citation≥0.90 ∧ grounding≥0.90 ∧ unsupported=[]` ∧
status gerçek-cevap. **ADAY ≠ eğitim verisi:** `requires_human_approval=True`; onaysız
eğitim YOK (kural 8). Salt-okuma seçim + JSONL export; hiçbir eğitim başlatmaz.

## Akış (özet)

```
classify → plan → (retrieval ⇄ reformulation)* → kanıt kapısı → taslak (LLM, seed=42)
→ iddia çıkarımı → atıf/dayanak/çelişki doğrulama → güven → çekimser → yapısal cevap
→ run logları (rlm_runs/steps/evidence/verifications) + reports/rlm_runs/*.json
```

## Çıktı durumları (`status`)

| status | anlamı |
|--------|--------|
| `answered` | yüksek güven, desteklenen iddialar |
| `answered_with_limitation` | cevap var ama kanıt güveni tam değil (sınırlama belirtilir) |
| `abstained` | desteklenen iddia yok / kanıt yetersiz → **uydurma YOK** |
| `no_llm` | LLM çevrimdışı/zaman aşımı → yalnız retrieval sonuçları (iddia üretilmez) |
| `failed` | beklenmeyen hata (run 'failed' işaretlenir, 'running' asılı kalmaz) |

## Mutlak kural güvenceleri (CLAUDE.md)

- **Kural 1** — trading-içerikli her çıktıya (soru VEYA cevap) zorunlu uyarı bloğu
  eklenir (`_apply_trading_guard`, içerik-tabanlı, görev tipinden bağımsız). Canlı
  sinyal/yatırım tavsiyesi ASLA.
- **Kural 4** — desteklenmeyen iddia nihai cevaba KONMAZ.
- **Kural 6** — sınıflandırma/skorlama saf-kural; LLM çağrıları `seed=42`
  (Anthropic backend'inde `temperature=0.0`).
- **Kural 7** — iki kapılı güvence: kanıt skoru çok düşükse LLM hiç çağrılmaz;
  cevap sonrası düşük güvende çekimser kal.

## Yapılandırma (env `ACHILLES_RLM_*`)

`rlm_max_retrieval_rounds=3` · `rlm_min_evidence_to_retry=40` ·
`rlm_min_evidence_to_answer=60` · `rlm_min_evidence_to_skip_retry=80` ·
`rlm_draft_max_tokens=900` · `rlm_draft_timeout_s=600` · `rlm_seed=42` ·
`rlm_allow_live_trading_signal=false` (MUTLAK — asla true).

## Notlar

- Yavaş CPU'da (GPU yok) tek cevap birkaç dakika sürebilir; `rlm_draft_timeout_s`
  aşılırsa graceful `no_llm` döner (run asılı kalmaz). Asılı eski 'running' run'lar
  her yeni çağrıda reaper ile temizlenir.
- `rlm-answer` çıktısı non-TTY pipe'a yönlendirilirse boş görünebilir (rich); kanıt
  her zaman DB'de (`rlm_runs`) ve `reports/rlm_runs/<run_id>.json`'da.

## İlgili

Mimari: `docs/rlm_rag_architecture.md`. Skill'ler: `/paper-mastery-agent` (makale
öğrenildi mi?), `/rag-reliability-engineer`, `/scientific-rag-reasoning`. Zincirde:
`automation_manifest.yaml` → `rlm-controller` (paper-mastery-agent sonrası).
