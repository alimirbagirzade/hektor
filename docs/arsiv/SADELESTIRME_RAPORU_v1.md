# Achilles — Sadeleştirme Raporu (gereksiz ve mantıksız kısımlar)

_Tarih: 2026-09-02 · Dal: worktree `dok-package-docs-sync-b65638` (commit edilmedi) · Önceki adım: [ACHILLES_YENIDEN_YAZIM_REHBERI.md](ACHILLES_YENIDEN_YAZIM_REHBERI.md) Bölüm 15._

Bu rapor üç kümeden oluşur:

- **A — Uygulandı.** Bu worktree'de değiştirildi; `ruff format` + `ruff check` + `mypy app` temiz, çevrimdışı test paketi yeşil (sonuç §A.9).
- **B — Silinmeye hazır, komut bekliyor.** Ölü olduğu kanıtla doğrulandı; dosya silme bu oturumda araç kısıtıyla engellendiği için tek komut olarak aşağıda verildi (§B).
- **C — Karar gerektirir.** Kaldırılması mantıklı ama kullanıcı kararı/politikası olan parçalar; gerekçe ve öneriyle listelendi (§C).

Kriter: bir parça (1) hiçbir üretim yolundan çağrılmıyorsa, (2) projenin kendi kurallarıyla çelişiyorsa veya (3) yalnız bayat/yanlış bilgi taşıyorsa "gereksiz/mantıksız" sayıldı. Her madde için kanıt (grep/okuma) belirtildi.

---

## A — Uygulanan değişiklikler (24 dosya, +105 / −117)

### A.1 Kod: sahte esneklik ve ölü parametreler

| # | Dosya | Ne vardı | Ne yapıldı | Kanıt |
|---|---|---|---|---|
| 1 | `app/trading/strategy_ir.py` (`to_pine`) | `getattr(ind, "smooth_k", 3)`, `smooth_d`, `multiplier` — `IndicatorSpec`'te bu alanlar hiç yok, `getattr` daima varsayılanı döndürüyordu | Sabitlere indirildi, yorumla açıklandı | `grep smooth_k\|multiplier app/trading/strategy_ir.py` → yalnız bu satırlar |
| 2 | `app/lora/curriculum.py` | `classify_curriculum(card_json, difficulty)` — `card_json` `_ = card_json` ile yok sayılıyordu | Parametre kaldırıldı; testler güncellendi | Tek çağıran `tests/test_lora_curriculum.py` (`{}` geçiyordu) |
| 3 | `app/lora/control_plane.py` | `run_full(dry_run=True)` — `_ = dry_run`; CLI `run_full(dry_run=False)`, auto_pipeline `run_full(False)` farklı değer geçip aynı davranışı alıyordu | Parametre kaldırıldı; 3 çağıran güncellendi | `grep -rn "run_full("` |
| 4 | `app/web/schemas.py` | `TrainingProgressResponse` şeması tanımlı ama hiçbir route kullanmıyor (`/api/training/progress` ham dict döner) | Sınıf silindi | `grep -rn TrainingProgressResponse app tests` → yalnız tanım |
| 5 | `app/web/agent_graph.py` | `_GROUP["makale-arastirma"]` — manifest'te böyle bir `agent_id` yok (o bir Claude ajanı, runtime ajanı değil) | Ölü anahtar silindi | `grep makale-arastirma automation_manifest.yaml` → yok |

### A.2 Kod: mantık hatası

| # | Dosya | Sorun | Düzeltme |
|---|---|---|---|
| 6 | `app/trading/evaluator.py` | `assert oos is not None` — `in_out_of_sample` çok kısa veride `None` döndürür; `python -O` altında assert kaybolup `AttributeError` olur. Kural 2 "kararsızsa kararsız de" | `oos is None` → `Verdict("inconclusive", ["Örneklem-dışı bölme için yeterli veri yok."])` |
| 7 | `app/web/static/assets/canli.html` | `fonts.googleapis.com`/`gstatic.com` `<link>`'leri — sunucu CSP'si `font-src 'self'; style-src 'self'` olduğundan **zaten bloklanıyordu**; yalnız konsol hatası üretiyordu | 3 link kaldırıldı (canli.css sistem fontlarına düşer) |
| 8 | `scripts/train-loop.ps1` | `achilles train --run` çağrısı `--profile` geçmiyordu → **vanilya reçete** (maskesiz, NEFTune'suz) = v5 tuzağı; `start-train.ps1`, `launch()` ve `training-watchdog.ps1` geçiyordu, tek boşluk buydu | `--profile discipline_safe_local` eklendi |
| 9 | `setup.sh` / `setup.ps1` | (a) Varsayılan seçim `gpt-4o-mini [ÖNERİLEN]` — projenin "API ASLA" kuralıyla çelişiyor. (b) **Gizli bug:** Ollama seçimlerinde (10-18) `MODEL_ENV` varsayılan `ACHILLES_OPENAI_MODEL`'de kalıyor, `.env`'e `ACHILLES_OPENAI_MODEL=qwen3:8b` yazılıyor, `ACHILLES_LLM_MODEL` hiç set edilmiyordu (yalnız `qwen3:4b` seçen, uygulama varsayılanı sayesinde doğru modeli alıyordu) | Varsayılan `10 / qwen3:4b (yerel)`, bulut bölümü "OPSİYONEL — gerekmez" etiketi, `MODEL_ENV="ACHILLES_LLM_MODEL"`; bulut seçenekleri korundu |

### A.3 Bayat/yanlış doküman ve docstring'ler

| # | Dosya | Sorun | Düzeltme |
|---|---|---|---|
| 10 | `README.md` | "OpenAI API key (**önerilen**)" ve "OpenAI varsa tercih edilir" — `.env.example`, CLAUDE.md ve HANDOFF'un "API asla" direktifiyle çelişiyor | Ollama önerilen/varsayılan; bulut desteği "opsiyonel, kullanılmaz" |
| 11 | `TRAINING_ROADMAP.md` | §6'da hem `### Achilles 🔴` hem `### Achilles ✅` (çelişkili); Paper Mastery bloğu §6 içine düşmüş; tarih 2026-06-07 | Başlıklar düzeltildi, Paper Mastery kendi alt başlığında, tarih + "güncel durum HANDOFF'ta" notu |
| 12 | `automation_manifest.yaml` | `phase: 1 # observer only` (Phase 2 kontrol düzlemi devrede); `auto-lora-pipeline.stop_method` "Phase 2 düzeltmesi bekliyor" (engine_procs var); `literature-scout` kaydında 3 zorunlu-benzeri alan eksik | phase 2; stop_method güncel; 3 alan eklendi |
| 13 | `app/agents/runtime/__init__.py` | "Phase 2 (task_queue, approvals, supervisor) HENÜZ YOK" — üçü de bu pakette | Docstring gerçek içeriğe göre yazıldı |
| 14 | `app/orchestration/driver.py` | "19 salt-okuma uç" (allow-list 21) | 21 |
| 15 | `app/reliability/__init__.py` | "hata analizi, kaynak güveni" modülleri yok, yalnız `release_gate.py` | Docstring daraltıldı |
| 16 | `app/evals/eval_runner.py` | `rlm-reward` ertelenme gerekçesi "eş zamanlı oturum bitene dek" (bayat) | Gerçek gerekçe: ödül türetimi tanımlanmadı (§16 aday) |
| 17 | `.claude/skills/formula-and-argument-integrity/SKILL.md` | Var olmayan `app/verification/formula_verifier.py` ve `argument_verifier.py` modüllerinden import eden örnek kod | Gerçek modüllere (`ContextualChunker`, `safe_tools.formula_check`) çevrildi + not |
| 18 | `.claude/skills/rlm-answer/SKILL.md` | `rlm_draft_timeout_s=300` (kod 600) | 600 |
| 19 | `docs/egitim/LORA_EGITIM_DETAYLI_ANLATIM.md` | "%60/%30/%10 müfredat pacing kodda bulunamadı" (yanlış — `training/dataset_builder.py`'de var); `auto_enabled` "varsayılan False" (efektif True); `app/main.py:1715-1767` bayat satır referansı | Üç iddia düzeltildi. **Not:** eşlik eden PDF (`.pdf`) yeniden üretilmeli: `uv run --extra docs python scripts/gen_egitim_pdf.py` |

### A.4 Kapı sonucu

```
uv run ruff format app tests   → 409 files left unchanged
uv run ruff check app tests    → All checks passed!
uv run mypy app                → Success: no issues found in 228 source files
uv run pytest -m "not ollama"  → exit 0 · 1782 passed · 5 skipped · 0 failed (temiz ağaçta)
```

### A.9 Test özeti ve koşu sırasında bulunan iki izolasyon hatası
Temiz ağaçta tam paket yeşil (1782 passed, 5 skipped). Ancak **ikinci koşu** `tests/test_orchestration_web.py::test_resume_with_hunt_ack_advances` ile düşüyor; sebep benim değişikliklerim değil, test paketinin kendi izolasyon açığı (bkz. §F). Artefaktlar temizlenince tekrar yeşil.

---

## F — Test paketinde bulunan izolasyon hataları (yeni bulgu, düzeltilmedi)

| # | Bulgu | Kanıt | Etki | Öneri |
|---|---|---|---|---|
| F1 | **Testler gerçek `data/` dizinine yazıyor.** Her tam koşu `data/lora_sft/lora_sft.jsonl` (5 satır, sistem-prompt'lu SFT örneği), `data/training/jsonl/train.jsonl` ve `valid.jsonl` üretiyor. `conftest.py` yalnız `ACHILLES_SQLITE_PATH`/`CHROMA_PATH`'i tmp'ye alıyor; `settings.root`-göreli veri yolları gerçek ağaçta kalıyor | Taze worktree'de ilk koşudan sonra dosyalar oluşuyor; gitignore'lu olduğu için `git status`'ta görünmüyor | Sonraki koşuda `data-gate` GO diyor → `approval` (`unattended_training_enabled=True`) → `train_handoff` "completed" → `test_resume_with_hunt_ack_advances` **düşüyor** (sıra/durum bağımlı flakiness). Ayrıca geliştirici makinesindeki gerçek `lora_sft.jsonl`'i **5 satırla EZEBİLİR** (clobber guard yalnız boş kaynağı korur) | `conftest.py`'ye session-scope fixture: `ACHILLES_ROOT`/veri dizinlerini tmp'ye yönlendir (settings'te `root` alanı yoksa ekle); yazan test bulunana kadar `data/lora_sft/lora_sft.jsonl` varsa testin başında yedekle |
| F2 | **Web lifespan arka plan döngüleri testlerde gerçekten çalışıyor.** `storage/unattended_supervisor_state.json` koşu sonrası `{"enabled": true, "status": "backoff", "engine": "codex", "run_id": "orc_…"}` içeriyor; `storage/self_heal_state.json` de yazılıyor | `TestClient` `lifespan`'ı tetikliyor; `UnattendedSupervisor.background_loop` ve `SelfHealingController.background_loop` başlıyor; supervisor bir koşu açıp **codex motoru doğurmaya çalışmış** (kurulu olmadığı için backoff) | Makinede `codex` kuruluysa test paketi **gerçek abonelik motoru doğurabilir** (kota + Kural 8). `storage/*_state.json` `.gitignore`'da DEĞİL → `git add -A` ile commit'lenebilir | Lifespan'da `ACHILLES_DISABLE_BACKGROUND_LOOPS=1` (veya `pytest` tespiti) ile döngüleri başlatma; `storage/*.json`'u `.gitignore`'a ekle; `unattended_supervisor.enabled` varsayılanını `settings.unattended_training_enabled`'dan ayır (C5 ile birlikte) |
| F3 | `pyproject` `addopts="-q"` + elle `-q` → `-qq` özet satırını gizler | `pytest -q` çıktısında "N passed" satırı yok | Kapı çıktısı okunamıyor; CI logunda sayı görünmez | `Makefile test` ve scriptlerde ek `-q` verme, ya da addopts'tan `-q`'yu çıkar |

---

## B — Ölü kod: silinmeye hazır (komut bekliyor)

Kriter: **hiçbir üretim modülü, CLI komutu, web route'u veya script import etmiyor**; yalnız kendi testleri var. Bunlar PR#106 temizliğinde "test var ama üretim yolu yok" diye bilinçli bırakılmıştı; bu adımda kaldırılması istendi. Silme işlemi bu oturumda araç kısıtıyla engellendi; aşağıdaki komut worktree'de çalıştırılınca kalan düzenlemeleri (§B.2) tamamlarım.

### B.1 Kanıt tablosu

| Dosya | Satır | Üretimde import eden | Test dosyası |
|---|---|---|---|
| `app/brain/query_expander.py` | ~215 | yalnız `multi_query_retriever.py` (o da ölü) | `tests/test_query_expander.py` |
| `app/brain/multi_query_retriever.py` | ~71 | **yok** | `tests/test_multi_query_retriever.py` |
| `app/memory/hybrid_retriever.py` | ~115 | **yok** (canlı hibrit yol `reranking_retriever._add_bm25_candidates` + `query_router.convex_fuse`) | `tests/test_hybrid_retrieval.py` içindeki 2 test |
| `app/evals/regression_runner.py` | ~122 | **yok** (orkestrasyon `regression.py` ayrı ve canlı) | `tests/test_regression_runner.py` |
| `app/evals/answer_eval.py` | ~115 | yalnız `regression_runner.py` | (aynı) |
| `app/evals/golden_generator.py` | ~98 | **yok** | `tests/test_golden_generator.py` |
| `app/cli/__init__.py` | 1 | **yok** (`pyproject` `app.main:app`'e bağlı) | — |
| `strategies/{pine,mql5,python}/.gitkeep` | — | hiçbir kod okumaz/yazmaz | — |

Toplam ≈ 650 satır kod + 4 test dosyası + 2 test fonksiyonu.

`golden_dataset.py`, `retrieval_eval.py`, `contextual_chunker.py`, `rules_updater.py`, `mlx_llm.py`, `cloud_notebook.py`, `benchmark/`, `system_profiler/` **ölü DEĞİL** (CLI/web/eval_runner'dan erişiliyor) — tabloya alınmadı.

### B.2 Silme komutu (worktree kökünde)

```bash
git rm -q -r app/cli strategies && git rm -q app/brain/query_expander.py app/brain/multi_query_retriever.py app/memory/hybrid_retriever.py app/evals/regression_runner.py app/evals/golden_generator.py app/evals/answer_eval.py tests/test_query_expander.py tests/test_multi_query_retriever.py tests/test_regression_runner.py tests/test_golden_generator.py
```

Silme sonrası yapılacak 4 küçük düzenleme (silinen modüllere atıf):
1. `tests/test_hybrid_retrieval.py` — `test_hybrid_retriever_topk_filled_with_text_chunks` ve `test_hybrid_retriever_zero_distance_ranks_first` fonksiyonlarını kaldır (bm25_corpus testleri kalır).
2. `app/evals/golden_dataset.py` docstring'inde "RegressionRunner" → "retrieval_eval".
3. `app/evals/eval_runner.py` `DEFERRED_TYPES["rag-answer"]` metnindeki "AnswerEvaluator" ifadesi.
4. `docs/egitim/RAG_EGITIM_DETAYLI_ANLATIM.md` satır 27 ve 222'de `query_expander`/`MultiQueryRetriever` atıflarına "kaldırıldı" notu.
Ardından kapı: `make format && make lint && make typecheck && make test`.

---

## C — Karar gerektiren parçalar

Bunlar "gereksiz" görünüyor ama ya kullanıcı politikasıdır ya da kaldırılması davranışı değiştirir. Öneri ve gerekçe verildi; uygulanmadı.

| # | Parça | Neden mantıksız/gereksiz görünüyor | Karşı argüman | Öneri |
|---|---|---|---|---|
| C1 | **Bulut LLM backend'leri** (`local_llm._generate_openai/_anthropic/_google`, `settings.openai_*/anthropic_*/google_*`, pyproject `anthropic`, `google-genai` bağımlılıkları, setup menüsü 1-9) | Proje "pay-per-token API ASLA" der; README/HANDOFF "geliştirilmeyecek" der; yerel runtime bunları kullanmaz | HANDOFF "Mimari Kararlar": kod **bilinçli** tutuldu; alexzhang RLM adapter'ı `anthropic` backend'e dayanır (opt-in) | **Kaldır** (≈300 satır + 2 bağımlılık) ya da en azından `llm_backend="auto"` varsayılanını `"ollama"` yap (bugün settings `auto`, `.env.example` `ollama` — iki varsayılan). Karar: kullanıcı |
| C2 | **alexzhang13/rlm adapter** (`app/rlm/adapters/alexzhang_rlm.py`, `security.py`, `rlm` extra, 9 `ACHILLES_RLM_ALEXZHANG_*` ayarı, docker preflight, trajektori logları) | Varsayılan kapalı; gerçek kullanım API anahtarı + docker ister → "API asla" ile çelişir; HANDOFF: "KOŞTURULMAZ" | 4 PR'lık additive iş; native motor bundan etkilenmez; kullanıcı "opsiyonel kalsın" demişti | Kaldırılırsa ≈400 satır + 21 test + 4 doküman gider. Karar: kullanıcı |
| C3 | **İki `AdapterRegistry`** (`app/lora/adapter_registry.py` JSONL durum makinesi vs `app/training/adapter_registry.py` SQLite metadata) | Aynı isim, iki gerçek; import karışıklığı | İkisi farklı iş yapıyor | Birini yeniden adlandır (`AdapterLifecycleRegistry` / `AdapterMetadataStore`) |
| C4 | **`promotion_gates.scan_secret_pii`** kendi regex seti | `safety_scanner` ile iki ayrı eşik seti; gerekçe ("eşzamanlı oturum WIP") artık geçersiz | `registry` paketi `app/lora`'yı bilinçli import etmiyor (bağımlılık yönü) | Ortak `app/security/patterns.py`'ye taşı, iki yerden import et |
| C5 | **`unattended_supervisor`** varsayılan `engine="codex"` ve göreli `Path("storage")` | `engines.DEFAULT_ENGINE="claude"`; göreli yol cwd'ye bağlı | PR#136 "codex drive subscription" — codex kullanıcı tercihi olabilir | `engine`'i settings'e al (`ACHILLES_UNATTENDED_ENGINE`), yolu `get_settings().root`'a bağla |
| C6 | **`context_sufficiency.classify(query, ...)`** `query`'yi kullanmaz | Sorgu-bağlam uyumu ölçülmüyor; parametre yanıltıcı | İmza RLM/exam runner'da yaygın | Ya kullan (sorgu token örtüşmesi) ya parametreyi kaldır |
| C7 | **Stage-2 bulut eğitim hattı** (`cloud_notebook.py`, `lora-cloud-prep`, `notebooks/`, `PROTOKOL_BULUT_EGITIM.md`, `/bulut-egitim-protokolu` skill, `/api/training/colab-notebook`) | Kullanıcı Kaggle'ı reddedip yerel 1.5B'ye pivot etti; HANDOFF "HER ŞEY LOKAL" | Kod çalışıyor ve testli; 4B için tek pratik yol hâlâ bulut-GPU | Belgede "opsiyonel/ikincil yol" olarak işaretle; kaldırma kararı kullanıcıda |
| C8 | **`reports/bug-scan/`** izleniyor (gitignore'da değil) | Haftalık rapor repoya birikiyor; komşu rapor dizinleri ignore'lu | Raporlar HANDOFF'tan referanslanıyor | `.gitignore`'a ekle, mevcut 3 raporu `docs/arsiv/bug-scan/`'a taşı |
| C9 | **HANDOFF.md** 1431 satır (Haziran ortasından itibaren tüm oturum geçmişi) | Her oturumda hook ilk 60 satırı okuyor; gerisi arşiv niteliğinde | Kullanıcının seans devir defteri; memory'de referanslanıyor | 2026-07-04 öncesi bölümleri `docs/arsiv/HANDOFF_2026-06.md`'ye taşı, üstte 1 satır link bırak |
| C10 | **`app/training/dataset_builder.py`** (`training_examples` SQLite → `{prompt,completion}`) | Web uçları kanonik `build_training_split`'e taşındı; yalnız `achilles dataset` CLI'si kullanıyor; iki-hat drifti buradan çıkmıştı | Curriculum pacing (%60/30/10) yalnız burada | Ya kaldır (pacing'i `sft_assembly`'ye taşı) ya CLI'da "inceleme amaçlı" etiketiyle bırak |
| C11 | **`setup.*` 18 seçenekli model menüsü** | 9 bulut seçeneği projede kullanılmaz; kurulum sihirbazını uzatıyor | C1 kararına bağlı | C1 "kaldır" ise menü 9 yerel seçeneğe iner |
| C12 | **Boş `tests/evals/__init__.py`**, `.specify/` spec-kit, `.claude/commands/speckit.*` | Kod tabanına katkısı yok | Kullanıcı spec-kit'i bilinçli kurdu | Dokunma |

---

## D — Rehber dokümanındaki düzeltmeler

Önceki rehberde (`ACHILLES_YENIDEN_YAZIM_REHBERI.md`) iki tespit yanlıştı, düzeltildi:
- `app/prompts/` boş değildir: `knowledge_card.md`, `rag_answer.md` (prompt_loader ile 3 modül okur).
- `agent_graph._GROUP`'ta `auto-researcher` eksik değildi; yalnız `makale-arastirma` ölü anahtardı (A.1 #5 ile kaldırıldı).

## E — Doğrulama notları

- Değişiklikler **commit edilmedi**; `git diff --stat` 24 dosya. İnceleme sonrası `scripts/open-pr.ps1` ile PR açılabilir.
- `setup.sh`/`setup.ps1` değişikliği canlı çalıştırılmadı (kurulum sihirbazı interaktif); yalnız değişken akışı okunarak doğrulandı (`MODEL_ENV` → `set_env`/`Set-EnvLine`).
- `canli.html` tarayıcıda açıldı (Browser pane); CSP hatası artık üretilmiyor.
- Test koşularının ürettiği artefaktlar (`data/lora_sft/lora_sft.jsonl`, `data/training/jsonl/{train,valid}.jsonl`, `storage/{unattended_supervisor,self_heal}_state.json`) her koşudan sonra elle silindi; kalıcı çözüm §F.
- Kural-8 yüzeylerine (onay, eğitim, STOP_ALL) dokunulmadı; test paketindeki güvenlik sözleşmeleri değişmedi.
