# v1 → 2.0 Migrasyonu — ne çıkarıldı, ne onarıldı

_2026-09-03 · Kaynak depo: `alimirbagirzade/achilles` (v1) · Hedef: o tarihte
`alimirbagirzade/achilles2.0`, 2026-09-04'ten beri `alimirbagirzade/hektor`_

> **Not:** Bu belge v1 → 2.0 migrasyonunu anlatır. Proje 2026-09-04'te **Hektor** olarak
> yeniden adlandırıldı; aşağıdaki `HEKTOR_*` değişken adları ve `hektor` komutları o
> tarihte `ACHILLES_*` / `achilles` idi. Geriye dönük uyum: bkz. `HANDOFF.md`.

Amaç: **çalışan sistemi taşımak, bayat ve gereksiz olanı taşımamak.** Bu belge her
kaldırma ve onarım için gerekçeyi kaydeder; "neden yok?" sorusunun tek cevabıdır.

Kapı durumu (taşımadan önce, temiz ağaçta): `ruff format` ✓ · `ruff check` ✓ ·
`mypy` (214 dosya) ✓ · `pytest -m "not ollama"` → **1736 passed, 4 skipped, 0 failed**.

---

## 1. Kaldırılanlar

### 1.1 Ölü kod (hiçbir üretim yolu çağırmıyordu)

Kriter: hiçbir modül/CLI/route/script import etmiyor **ve** yalnız kendi testi var.

| Kaldırılan | Neden |
|---|---|
| `app/brain/query_expander.py` | Yalnız `multi_query_retriever` kullanıyordu (o da ölü) |
| `app/brain/multi_query_retriever.py` | Üretimde çağıran yok |
| `app/memory/hybrid_retriever.py` | Canlı hibrit yol `reranking_retriever._add_bm25_candidates` + `query_router.convex_fuse` |
| `app/evals/regression_runner.py` | Üretimde çağıran yok (orkestrasyon `regression.py` ayrı ve canlı) |
| `app/evals/answer_eval.py` | Yalnız `regression_runner` kullanıyordu |
| `app/evals/golden_generator.py` | Üretimde çağıran yok |
| `app/cli/` (boş paket) | `pyproject` konsol scripti `app.main:app`'e bakar |
| `strategies/{pine,mql5,python}/` | Hiçbir kod okumaz/yazmaz |
| `TrainingProgressResponse` şeması | Hiçbir route kullanmıyordu |
| 4 test dosyası | Yalnız yukarıdakileri test ediyorlardı |

### 1.2 Bulut LLM istemcileri

Projenin kalıcı kısıtı "pay-per-token API asla" olmasına rağmen kodda OpenAI / Anthropic /
Google istemcileri duruyordu ve kurulum sihirbazının **varsayılan seçeneği** bulut modeldi.

- `app/brain/local_llm.py` → yalnız Ollama (`_generate_openai/_anthropic/_google` ve
  `_*_ready` probe'ları silindi; `available()`/`active_backend()` sadeleşti).
- `Settings`'ten silinen alanlar: `llm_backend`, `openai_api_key/model/base_url`,
  `anthropic_api_key/model`, `google_api_key/model`.
- `pyproject` bağımlılıkları `anthropic` ve `google-genai` kaldırıldı.
- `setup.sh` / `setup.ps1` menüsü yalnız 9 yerel modele indi; API-key talimat blokları silindi.
- `.env.example` bulut bölümleri kaldırıldı.
- Regresyon kilidi: `tests/test_web_api.py::test_no_cloud_llm_client_in_codebase` — bulut SDK
  importu veya api-key ayarı geri gelirse test kırılır.

**Yan bulgu (gerçek bug, düzeltildi):** kurulum sihirbazında yerel model seçilince
`MODEL_ENV` varsayılanı `HEKTOR_OPENAI_MODEL`'de kalıyordu → `.env`'e
`HEKTOR_OPENAI_MODEL=qwen3:8b` yazılıp `HEKTOR_LLM_MODEL` hiç ayarlanmıyordu. Yalnız
varsayılan modeli (`qwen3:4b`) seçen kullanıcılar bunu fark etmiyordu.

### 1.3 Opsiyonel dış RLM motoru (alexzhang13/rlm)

Varsayılan kapalıydı, gerçek kullanımı API anahtarı + docker istiyordu ve hiç koşturulmadı.

- Silinen: `app/rlm/adapters/` (base, native, alexzhang, security), `app/rlm/answer_pipeline.py`,
  `app/rlm/engine_config.py`, `tests/test_rlm_engine_adapters.py`, `tests/test_rlm_engine_security.py`,
  `docs/rlm_alexzhang_integration.md`, `docs/rlm_security_model.md`, `docs/rlm_runtime_modes.md`,
  `.claude/skills/rlm-integration/`, `.claude/agents/rlm-integration-agent.md`, `rlm` extra.
- `ALLOWED_TOOL_NAMES` (deny-by-default araç allowlist'i) `app/rlm/tool_registry.py`'ye taşındı —
  artık tek gerçek-kaynak orası.
- `GET /api/rlm/config` korundu ama sadeleşti (motor/model/üretim modu/izinli araçlar/seed);
  `POST /api/rlm/test-adapter`, `hektor rlm-test-adapter` ve `rlm-answer --engine` kaldırıldı.
- **RLM sekmesi ve koşu tablosu korundu** (yalnız motor-seçim alt paneli sadeleşti).

### 1.4 Bayat oturum geçmişi ve raporlar

`docs/arsiv/` altına taşındı veya silindi: 1431 satırlık v1 HANDOFF'u, 3 eski bug-scan raporu,
3 oturum devir notu, migrasyon-anı belgeleri (`ITHAKA_SPEC.md`, `SADELESTIRME_RAPORU_v1.md`).
`reports/bug-scan/` artık `.gitignore`'da (dizin `.gitkeep` ile korunur) — haftalık tarama
raporları depoya birikmez.

---

## 2. Onarılan gerçek hatalar

| # | Hata | Etki | Düzeltme |
|---|---|---|---|
| 1 | **Testler gerçek `data/` ve `storage/` ağacına yazıyordu** | Her tam koşu `data/lora_sft/lora_sft.jsonl` + `train/valid.jsonl` üretiyor, sonraki koşuda data-gate GO verip orkestrasyon testini düşürüyordu (sıra-bağımlı flakiness). Geliştiricinin gerçek `lora_sft.jsonl`'ini 5 satırla ezme riski. | Tüm veri yolları `settings.root`'tan türer; `HEKTOR_ROOT_PATH` ile yönlendirilebilir. conftest kökü tmp'ye alır **ve** teardown'da gerçek ağaca sızıntı olursa paketi FAIL eder. |
| 2 | **Testler gerçek abonelik motoru doğurabiliyordu** | `TestClient` lifespan'i tetikliyor, unattended supervisor bir koşu açıp `codex` motorunu başlatmaya çalışıyordu (kota + Kural 8). Koşu sonrası `storage/unattended_supervisor_state.json` `{"status":"backoff","engine":"codex"}` kalıyordu. | `background_loops_enabled` ayarı; testlerde `false` → hiçbir döngü başlamaz. Yan etki: test süresi 295 s → 99 s. |
| 3 | **Durum dosyaları CWD'ye bağlıydı** (`Path("storage")/...`) | Süreç başka dizinden başlatılırsa (Windows servisi, detached eğitim) durum sessizce yanlış yere yazılır/okunurdu. 5 modül etkileniyordu. | Hepsi `settings.state_dir` kullanıyor. |
| 4 | **Eval setleri CWD'ye bağlıydı** (`auto_pipeline`: `Path("evals")`) | Detached koşuda yanlış dizine bakıp sessizce `EVAL_SKIPPED` üretiyordu → terfi kapısı sessizce kapanıyordu. | `settings.eval_sets_dir` (kaynak kökü). |
| 5 | `evaluator.evaluate` çıplak `assert oos is not None` | Çok kısa veride `AssertionError`; `python -O` altında `AttributeError`. | Yetersiz veri → `Verdict("inconclusive")` (Kural 2). |
| 6 | `scripts/train-loop.ps1` `--profile` geçmiyordu | Vanilya reçete (maskesiz, NEFTune'suz) = v5 regresyon tuzağı. Diğer üç başlatıcı profili geçiyordu. | `--profile discipline_safe_local` eklendi. |
| 7 | `canli.html` Google Fonts `<link>`'leri | Sunucu CSP'si (`font-src 'self'`) zaten blokluyordu; yalnız konsol hatası üretiyordu. | Kaldırıldı. |
| 8 | `to_pine` var olmayan alanları okuyordu (`getattr(ind,"smooth_k",3)`) | `IndicatorSpec`'te bu alanlar yok → daima varsayılan; sahte esneklik. | Sabitlere indirildi, yorumla açıklandı. |
| 9 | `pytest` `addopts`'ta `-q` | Elle `-q` eklenince `-qq` olup "N passed" özet satırını gizliyordu. | `addopts`'tan çıkarıldı. |

Ayrıca ölü parametreler temizlendi: `classify_curriculum(card_json, ...)` ve
`LoRAControlPlane.run_full(dry_run)` (ikisi de argümanı yok sayıyordu),
`agent_graph._GROUP["makale-arastirma"]` (manifest'te karşılığı yok).

---

## 3. Değişen varsayılanlar

| Ayar | v1 | 2.0 | Neden |
|---|---|---|---|
| `unattended_training_enabled` | `True` | **`False`** | Her gerçek eğitim ve terfi tek-kullanımlık insan onayı ister (Kural 8). Kullanıcı bilinçli açabilir. |
| Kurulum sihirbazı varsayılanı | `gpt-4o-mini` (bulut) | **`qwen3:4b` (yerel)** | "API asla" kuralı |
| `storage/*.json` git takibi | tek tek listeleniyordu | **hepsi ignore** | Yeni ajan state dosyaları listeye eklenmeyi unutunca sızıyordu |

Yeni ayarlar: `root_path` (veri kökü), `background_loops_enabled`.
Yeni türetilmiş yollar: `source_root`, `state_dir`, `eval_sets_dir`, `manifest_file`.

> **Kaynak / veri ayrımı:** `root` yalnız *veri*dir (`data/`, `reports/`, `storage/`,
> `models/`, `logs/`). Depoyla gelen okunur dosyalar (`automation_manifest.yaml`,
> `evals/*.jsonl`, `configs/`, `app/prompts/`, `.claude/`) `source_root` altındadır ve
> veri kökü değişse bile sabit kalır.

---

## 4. Taşınmayanlar (bilinçli)

- **Git geçmişi.** Bu depo tek "initial commit" ile başlar; v1 geçmişi eski depoda arşivdir.
  Eski PR numaraları (#136 vb.) yeni depoda karşılıksız kalırdı.
- **Veri ve modeller.** `data/`, `storage/`, `vector_db/`, `models/` git'te izlenmez; RAG indeksi
  ve adapter'lar her makinede sıfırdan üretilir (v1'in taşıma kararı korundu).

## 5. Kalan adaylar (karar bekliyor, kaldırılmadı)

| Aday | Durum |
|---|---|
| Phase-4 GitHub otomasyonu (`.github/workflows/claude-code-task.yml`, `docs/PHASE4*.md`, 2 yönetişim testi) | Hiç aktive edilmedi (`vars.ENABLE_CLAUDE_TASK` olmadan INERT) ve `ANTHROPIC_API_KEY` ister. Guard'lı ve testli olduğu için kaldırılmadı. |
| `app/training/dataset_builder.py` | İkinci veri hattı (SQLite `training_examples`); web uçları kanonik `sft_assembly` yoluna taşındı, yalnız `hektor dataset` kullanıyor. Müfredat pacing (%60/30/10) yalnız burada. |
| Bulut-GPU eğitim hattı (`cloud_notebook.py`, `lora-cloud-prep`, `PROTOKOL_BULUT_EGITIM.md`) | Yerel küçük-model eğitimine pivot edildi ama kod çalışıyor ve testli; 4B için tek pratik yol. |
| `docs/MIMARI_REFERANS.md` | v1 temizliğinden ÖNCE yazıldı; kaldırılan modülleri hâlâ anlatıyor. Dosya başında uyarı var. |
