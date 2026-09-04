# CLAUDE.md — Achilles 2.0 çalışma kuralları

Bu dosya, bu repoda çalışan Claude (Claude Code) için bağlayıcı yönergeleri içerir.

## Proje nedir
Yerel-öncelikli AI trading **araştırma** sistemi: PDF literatür → RAG/bilgi
kartı → (opsiyonel LoRA) → disiplinli backtest. **Canlı bot değil, tavsiye değil.**

LLM hattı **yalnız yerel Ollama**'dır; bulut sağlayıcı istemcisi kodda YOKTUR ve
eklenmez (kalıcı kısıt). Geliştirme yardımı aylık abonelikli CLI araçlarıyla yapılır.

## Mutlak kurallar (asla ihlal etme)
1. **Yatırım tavsiyesi üretme.** Çıktılar her zaman _hipotez_ + _test noktası_.
2. **Test edilmeden "başarılı/çalışıyor" deme.** backtest + out-of-sample şart.
3. **Maliyetleri yok sayma** (komisyon + slippage).
4. **Look-ahead bias yasak** — pozisyon `shift(1)` ile gecikmeli.
5. **`eval`/`exec` yok** — strateji kuralları yalnızca güvenli regex ile parse.
6. **Determinizm** — rastgelelik daima `seed` parametresiyle.
7. **Kaynak uydurma** — retrieval boşsa açıkça belirt.
8. **Otomatik ağır eğitim yok** — `train` varsayılan dry-run; gerçek eğitim
   yalnızca açık `--run` ile.

## Kod stili
- Python ≥ 3.12, `from __future__ import annotations`
- pydantic v2 modelleri; SQLAlchemy 2.0 tipli API
- ruff (line-length 100, target py312), mypy (pydantic plugin)
- Saf pandas/numpy indikatörler (vektörize, döngü değil)
- Kullanıcıya dönük metinler/log/docstring **Türkçe**

## Doğrulama (değişiklik sonrası zorunlu)
```bash
make format && make lint && make typecheck && make test
```
Testler **çevrimdışı** çalışmalı (fake embedding + sentetik veri). Ollama/MLX
gerektiren testler `@pytest.mark.ollama` / `@pytest.mark.slow` ile işaretli.

## 🔁 Bug-avı kadansı (kademeli)

Bug yoğunluğu takvimle değil **kod değişimiyle (churn)** artar; bu yüzden maliyet
seviyesine göre kademeli tarama:

| Kademe | Ne | Tetikleyici | Otonomi |
|--------|-----|-------------|---------|
| **0 — Kapı** | `make format && lint && typecheck && test` (+ pre-commit/CI) | **Her commit** | Otomatik |
| **1 — Hafif tarama** | Tek `claude -p` rapor-only tarama (son diff + çekirdek) | **Haftalık** (yerel Task Scheduler: `scripts/weekly-bug-scan.ps1`) | **Rapor-only** (kod değiştirmez, push etmez) |
| **2 — Derin adversarial av** | Çok-ajan workflow (finder + 2-oylu adversarial doğrulama) | **Ayda 1 + her LoRA eğitiminden ÖNCE (zorunlu)** veya ~25-30 commit'te | **Denetimli** (fix+push insan gözetiminde) |

- **Kademe 2 her eğitimden önce zorunlu** — projenin tüm amacı backtest/eval'e güvenmek;
  v5 regresyonu tam bu yüzden olmuştu. Eğitime başlamadan derin av çalıştır.
- Kademe 1 raporu: `reports/bug-scan/scan-<tarih>.md` + HANDOFF özeti. Bulgular bir sonraki
  **denetimli** seansta düzeltilir (otomatik fix YOK — yanlış fix'i gözetimsiz main'e basma).
- Derin av deseni: alt-sistem başına paralel finder → her bulgu adversarial doğrulama
  (şüpheci, varsayılan çürütülmüş) → yalnız onaylananları düzelt → Kademe 0 kapısı → commit+push.

## Mimari sözleşmeleri
- `paper_id` içerik hash'inden türer → ingestion **idempotent**.
- Strateji yaşam döngüsü: `hipotez → StrategyIR → backtest → evaluate → verdict`.
  `verdict != pass` ise çıktı "aday"dır, "hazır" değildir.
- Yeni indikatör → `app/trading/indicators.py` registry'sine ekle + test yaz.
- Yeni CLI komutu → `app/main.py` + README tablosu güncelle.

## ⚡ Yeni seans başlangıcı

1. **`HANDOFF.md`'yi oku** — durum, sıradaki adım ve açık işler oradadır.
2. **Sistem durumunu kontrol et:**
   ```bash
   uv run achilles status     # Ollama + korpus + model
   uv run achilles doctor     # bu makine origin/main'de mi (salt-okuma)
   ```
3. **Kapıyı çalıştır** (değişiklikten önce ve sonra): `make ci`.

Bu depo `alimirbagirzade/achilles2.0`'dır; v1'den ne taşınmadığı `docs/MIGRASYON_2.0.md`'de.

### Proje skill'leri (`.claude/skills/`)

| Skill | Ne zaman |
|-------|----------|
| `/trading-research` | Araştırma döngüsü: formül çıkar → sentez → backtest |
| `/rlm-answer` | Kaynaklı + doğrulanmış cevap (çok-tur retrieval → iddia doğrula → çekimser) |
| `/backtest-auditor` | Backtest sonucu denetimi: look-ahead + OOS + overfit |
| `/codegen-review` | Yeni indikatör/strateji kodu: ruff + mypy + test |
| `/lora-training-control-plane` | LoRA hattı: audit → gate → eval → registry |
| `/veri-uretim-protokolu` | Stage 1 sentetik veri üretimi |
| `/paper-mastery-agent` | Makale ustalık ölçümü (0-100) |
| `/scientific-tool-runtime` | Hesabı deterministik araçla doğrula (seed zorunlu) |
| `/hypothesis-evaluator` | Fikir "sinyal" değil test-edilebilir hipotez mi |
| `/model-data-registry` | Sürüm kaydı + terfi kapısı (Kural 8) |

## Yapma
- Gizli anahtar/credential commit etme (`.env` ignore'da).
- `data/`, `models/`, `vector_db/`, `storage/` çıktısını commit etme (.gitkeep hariç).
- Stratejiyi backtest+denetimden geçirmeden "kullanıma hazır" sunma.
