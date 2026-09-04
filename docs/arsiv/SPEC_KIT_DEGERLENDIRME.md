<!-- spec-kit /analyze tarzı değerlendirme — 4 paralel anayasa-uyum denetiminin sentezi (2026-06-13). -->

# Achilles — Anayasa Uyum Değerlendirmesi (spec-kit)

> Kapsam: Anayasa Prensip I–VII + Teknoloji Standartları. Dört bağımsız denetim raporunun tek değerlendirmeye birleştirilmiş hali (spec-kit `/analyze` tarzı). Tüm bulgular `file:line` kanıtına dayanır; doğrulanamayan iddialar açıkça işaretlenmiştir (Prensip II gereği overclaim yok).

## Özet skor tablosu

| Prensip | Konu | Sonuç |
|---|---|---|
| **I** | Yatırım tavsiyesi değil — tavsiye/garanti dili engelli | ✅ UYUMLU |
| **II** | Test edilmeden onay yok — OOS zorunlu | ✅ UYUMLU |
| **III** | Backtest disiplini (shift(1) + maliyet + seed + verdict) | ✅ UYUMLU |
| **IV** | `eval`/`exec` yasak — güvenli regex parse | ✅ UYUMLU |
| **V** | Kaynak uydurma yok — boş retrieval'de belirt + FK bağı | ✅ UYUMLU |
| **VI** | Kontrollü eğitim (dry-run default + onay + COMPLETED) | ⚠️ KISMİ |
| **VII** | Teknoloji standartları (offline test, ruff/mypy, vektörizasyon) | ⚠️ KISMİ |

**Genel:** 5 tam uyumlu, 2 kısmi. **Sert ihlal (❌) yok.** İki kısmi durum somut ve düşük efor düzeltilebilir.

---

## Prensip bazında bulgular

### Prensip I — Yatırım tavsiyesi değil → ✅ UYUMLU
- `app/prompts/system_trader.md:3,6,9` — "Canlı işlem botu veya finansal danışman DEĞİLSİN", "test edilmeden ASLA 'çalışıyor' deme", "Kesin kazanç vaadi verme".
- `app/prompts/rag_answer.md:10,13` — trading çıktısı "hipotez (test edilmemiş)" etiketli; akademik bulgu doğrudan işlem kuralı gibi sunulmaz.
- `app/brain/rag_answerer.py:41-44` — çevrilemiyorsa "Bu bulgu doğrudan trading kuralına çevrilemez" yazmaya zorlanır; çıktı Hipotez + Test Planı + Riskler bölümlü.
- `app/lora/dataset_builder.py:14-19` — eğitim system prompt'u da belirsizliği doğru ifade etmeye zorlar; tavsiye dili eğitim setine yapısal olarak girmez.
- **Gözlem (ihlal değil):** Guard'lar prompt + yapı seviyesinde; çıktı metnini yasaklı ibareler için tarayan deterministik post-filter yok. İyileştirme adayı, ihlal değil.

### Prensip II — Test edilmeden onay yok → ✅ UYUMLU
- `app/trading/evaluator.py:1-9` — modül politikası: OOS testi olmadan hiçbir şey "başarılı" sayılmaz.
- `evaluator.py:29` — `evaluate()` her zaman `in_out_of_sample(df, ir)` çağırır; OOS atlanamaz.
- `evaluator.py:35-40` — hard-fail: OOS getiri pozitif değilse / az işlem / aşırı drawdown → `fail`.
- `evaluator.py:42-47` — soft uyarı veya OOS Sharpe < 0.5 → `inconclusive`, asla `pass` değil.
- `app/trading/overfit_checks.py:35-49` — az işlem, şüpheli profit factor, aşırı drawdown, gerçekçi olmayan Sharpe guardrail'leri.
- **Gözlem:** `evaluator.py:33` `assert oos is not None` — `python -O` altında elenir; `in_out_of_sample` her zaman dolu döndüğünden pratik risk yok.
- **Not (doğrulandı):** Denetim 1'in atıfta bulunduğu `strategy_evaluator.py` dosyası repoda **yok**; gerçek karşılığı `app/trading/evaluator.py`.

### Prensip III — Backtest disiplini → ✅ UYUMLU
- **shift(1) gecikme:** `app/trading/backtester.py:163` — `position.shift(1).fillna(0.0) * bar_ret`. Trade tespiti/getirileri de `position.shift(1)` üzerinden (`:131-132`). İndikatörler nedensel: ATR `prev_close = close.shift(1)` (`indicators.py:43`), EMA `ewm` (`:15`), rolling tabanlılar (`:19,53`).
- **Maliyet:** `backtester.py:166-168` — `turnover * (commission + slippage)`; tüm metrik/equity **net** getiri üzerinden (`:170-171`). Default maliyet sıfır değil: `CostSpec.commission=0.0005, slippage=0.0005` (`strategy_ir.py:38-39`).
- **Seed/determinizm:** Tek stokastik kaynak sentetik veri; `market_data_loader.py:56` `np.random.default_rng(seed)`. Çağrı yerleri seed geçirir (`web/server.py:1082`, `main.py:648`, `research/orchestrator.py:112`). Backtester'ın kendisi saf-deterministik (rastgelelik yok).
- **verdict != pass → aday:** `research/synthesis_paper.py:109-110` — PASS yoksa "HİPOTEZ (backtest geçmedi)" etiketi; `:117-119` zorunlu uyarı: "OOS doğrulama olmadan hiçbir strateji 'hazır' sayılmaz".

### Prensip IV — eval/exec yasak + güvenli parse → ✅ UYUMLU
- Tüm repoda `\b(eval|exec)\s*\(` regex grep → **sıfır eşleşme**.
- `app/trading/strategy_ir.py:17-19` — `_RULE_RE`: kurallar yalnız `^<col> <op> <col|sayı>$`; operatör whitelist, operand sadece identifier/sayı.
- `strategy_ir.py:52-60` — pydantic `field_validator` her kuralı regex'e zorlar; eşleşmezse `ValueError`.
- `strategy_ir.py:122-124,141-145` — Pine üretimi yalnız doğrulanmış üç parçayı string birleştirir; hiçbir değerlendirme yok.
- `app/web/server.py:1090-1095` — backtest endpoint `model_validate` ile doğrular ("kural çalıştırma YOK, yalnız regex parse").
- JSON parse güvenli: `knowledge_card_builder.py:45-58` sadece `json.loads`.

### Prensip V — Kaynak uydurma yok → ✅ UYUMLU
- **İki yanıt yolu da boş-kaynak kısa devresi yapar:**
  - Varsayılan: `app/brain/rag_answerer.py:87-99` — `if not chunks:` → LLM çağrılmaz, "Kaynak bulunamadı", `sources=[]`, `llm_used=False`.
  - Adapter (MLX): `app/web/server.py:274-283` — `if not chunks:` → erken return, `generate` hiç çağrılmaz.
- Prompt pekiştirmesi: `app/prompts/rag_answer.md:3-4`, `knowledge_card_builder.py:119-121` ("makalede olmayanı uydurma", parse başarısızsa boş-ama-geçerli kart).
- **FK bağı (gerçek):** `app/memory/sqlite_store.py:100` (`KnowledgeCard.paper_id` → `ForeignKey("papers.paper_id")`), `:212` (`Formula.paper_id`), `:73,90` (Chunk/Comprehension). `knowledge_card_builder.py:159` paper_id'yi LLM'den değil argümandan zorlar. LoRA: `dataset_builder.py:84-87,107-113` — yalnız `approved` + `lora_eligible` örnekler, metadata'da paper_id.
- **/api/ask adapter halüsinasyon bug'ı:** Düzeltme **gerçek ve commit'li** — ancak Denetim 2'nin verdiği hash `bbf4cec` bu çalışma dizininde **doğrulanamadı**. Live git geçmişinde kök-neden düzeltme commit'i `1530abb` ("fix(comprehension): 'anlama %' kök nedeni — kırık embedding import") olarak görünür. Mevcut kod (`server.py:274-283`) boş retrieval'de erken döner — **kod durumu uyumlu doğrulandı**; yalnız atıf hash'i farklıdır.

### Prensip VI — Kontrollü eğitim → ⚠️ KISMİ
- **dry-run default:** `app/main.py:250` `run: bool = Option(False)`; gerçek eğitim yalnız `if run:` (`:302-303`), aksi `dry_run` (`:311-312`). Modül: `peft_lora_train.py:405-410` `--run` yoksa `dry_run` + `sys.exit(0)`. ✅
- **Onay kapıları:** `adapter_registry.py:122-129` `promote(... user_approved)`: `if not user_approved: return False`. `auto_pipeline.py:192-202` eğitim yalnız `READY_TO_TRAIN`'de + web UI onayı; `background_loop` (`:414-428`) yalnız `check_and_prepare` çağırır, otomatik eğitim tetiklemez. `control_plane.py:105-128` "hiçbir koşulda ağır eğitim başlatmaz". ✅
- **COMPLETED + eval:** `training_manager.py:126-130` durum yalnız `returncode==0` ise `COMPLETED`. `auto_pipeline.py:263-269` eval yalnız `COMPLETED` ise; `:308-339` `EVAL_PASSED` eşik kontrolüne (`avg >= eval_pass_threshold`) bağlı. ✅
- **⚠️ Kısmi (gerilim):** `auto_pipeline.py:289-294` — eval seti yoksa sistem uyarı loglayıp **doğrudan `EVAL_PASSED` varsayıyor** ("eval seti yok, EVAL_PASSED varsayılıyor") ve adapter'ı kaydediyor. "Test edilmeden başarılı deme" ruhuyla gerilimde. Production terfi ayrı onay gerektirdiği için sert ihlal değil, ama düzeltilmeli.

### Prensip VII — Teknoloji standartları → ⚠️ KISMİ
- **Offline test:** ✅ `tests/conftest.py` `_isolate_storage` izole DB/chroma + `ACHILLES_ALLOW_FAKE_EMBEDDINGS=true`. Fake embed deterministik (`embedding_service.py:116` SHA-256). Ollama testleri `@pytest.mark.ollama` ile otomatik atlanır; `pyproject.toml:104` `addopts = "-m 'not ollama and not slow'"`. Temiz basetemp'te **405 passed** offline.
- **mypy:** ✅ `Success: no issues found in 135 source files`.
- **ruff:** ⚠️ **2 hata doğrulandı (live tree):**
  - `app/memory/sqlite_store.py:608` — `F821 Undefined name 'ComprehensionScore'` (annotation tanımsız ad; `# type: ignore[name-defined]` ile maskelenmiş, ama isim TYPE_CHECKING/import dışında).
  - `app/memory/embedding_service.py:55` — `E501` (Türkçe debug log satırı 100 karakteri aşıyor).
- **Vektörizasyon:** ⚠️ İndikatörler tam vektörize (`indicators.py`, döngü yok). Ama iki Python döngüsü var: `backtester.py:105` `_position_series` (stateful 0/1 holding) ve `risk_manager.py:279` `_extract_trade_returns`. Fonksiyonel olarak doğru, look-ahead yok; ama "vektörize, döngü değil" standardına aykırı (stateful zincir nedeniyle vektörizasyon zor).
- **`from __future__ import annotations`:** ✅ 135 dosyanın 115'inde; eksik 20'nin tamamı trivial `__init__.py` (gerçek modüllerde %100).
- **pydantic v2 / SQLAlchemy 2.0 / py3.12:** ✅ `pyproject.toml` — `pydantic>=2.7`, `sqlalchemy>=2.0`, `requires-python>=3.12`. Türkçe metin/log/docstring uyumlu.

---

## Tespit edilen boşluklar/riskler (öncelikli)

1. **[Prensip VII — ruff F821] `sqlite_store.py:608`** — `ComprehensionScore` adı tanımsız; `# type: ignore` maskeliyor ama ruff yakalıyor. CLAUDE.md "ruff temiz" sözleşmesini ihlal eder. **Öncelik: Yüksek (somut ihlal).**
2. **[Prensip VI — eval bypass] `auto_pipeline.py:289-294`** — eval seti yokken `EVAL_PASSED` varsaymak Prensip II/VI ruhuyla çelişir. Terfi ayrı onaya bağlı olsa da yanlış "geçti" damgası riski. **Öncelik: Yüksek (mantıksal boşluk).**
3. **[Prensip VII — ruff E501] `embedding_service.py:55`** — uzun satır. **Öncelik: Orta (kozmetik ama sözleşme ihlali).**
4. **[Prensip VII — vektörizasyon] `backtester.py:105`, `risk_manager.py:279`** — stateful `for` döngüleri standart dışı; büyük seride yavaş. **Öncelik: Düşük (fonksiyonel bug değil).**
5. **[Prensip I — derinlik savunması]** Yasaklı-ibare çıktı post-filter'ı yok. **Öncelik: Düşük (mevcut prompt-seviyesi guard yeterli).**
6. **[İz/dokümantasyon]** Denetim 2'nin `/api/ask` düzeltmesi için verdiği commit hash `bbf4cec` bu repoda doğrulanamadı (live karşılığı `1530abb`). Kod durumu uyumlu, ama denetim notlarındaki hash güncellenmeli. **Öncelik: Düşük (izlenebilirlik).**

---

## Önerilen aksiyonlar (spec-kit /tasks tarzı, sıralı)

- [x] **T1 — ruff F821 DÜZELTİLDİ** ✅ · `sqlite_store.py` · `ComprehensionScore` `if TYPE_CHECKING:` bloğuna alındı, `# type: ignore[name-defined]` kaldırıldı. *(Prensip VII)*
- [x] **T2 — eval-skip durumu EKLENDİ** ✅ · `auto_pipeline.py` · Eval seti yokken `EVAL_PASSED` yerine yeni `EVAL_SKIPPED`; `promote_to_production` yalnız `EVAL_PASSED`'i terfi ettirdiğinden terfi bloklanır + registry status `SMOKE_PASSED` (yanlış "eval geçti" damgası yok). *(Prensip VI/II)*
- [x] **T3 — ruff E501 DÜZELTİLDİ** ✅ · `embedding_service.py:55` debug log bölündü. *(Prensip VII)*
- [x] **T4 — DOĞRULANDI** ✅ · ruff **0 hata**, mypy temiz, ilgili testler geçti.
- [ ] **T5 — vektörizasyon (opsiyonel)** · `backtester.py:_position_series`, `risk_manager.py:_extract_trade_returns` · numpy kümülatif maske/`numba` ile vektörize **veya** stateful gerekçeyi docstring'e ekle. *(Prensip VII — standart uyumu)*
- [ ] **T6 — çıktı ibare-tarayıcı (opsiyonel)** · RAG/sentez çıktısında "garanti/kesin kazanç" gibi yasaklı ibareler için deterministik post-filter. *(Prensip I — derinlik savunması)*
- [ ] **T7 — denetim izi (opsiyonel)** · `/api/ask` düzeltme atfını doğrulanmış commit (`1530abb`) ile güncelle; `bbf4cec` referansını düzelt. *(İzlenebilirlik)*
