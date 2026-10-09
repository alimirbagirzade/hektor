# CLAUDE.md — Hektor çalışma kuralları

Bu dosya, bu repoda çalışan Claude (Claude Code) için bağlayıcı yönergeleri içerir.

## Proje nedir
Yerel-öncelikli AI trading **araştırma** sistemi: PDF literatür → RAG/bilgi
kartı → (opsiyonel LoRA) → disiplinli backtest. **Canlı bot değil, tavsiye değil.**

Cevap ve eğitim hattı **yalnız yerel Ollama**'dır; API anahtarlı bulut istemcisi kodda
YOKTUR. Tek istisna (Faz 3, `app/cloud/`, varsayılan KAPALI): insanın tur başına açıkça
başlattığı **ikinci görüş** — gönderilecek metin önceden aynen gösterilir, çıktısı doğrulama
sayılmaz, öğrenme adayı olmaz, eğitime GİRMEZ (bulut kökenli satır = eğitim kapısında NO-GO;
`docs/TASARIM_FAZ3_BULUT.md`). Toplu/otomatik bulut döngüsü ve bulut çıktısıyla eğitim yasak.
Geliştirme yardımı aylık abonelikli CLI araçlarıyla yapılır.

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
  v5 regresyonu tam bu yüzden olmuştu. Eğitime başlamadan derin av çalıştır. Kod bunu
  ZORLAR: her eğitim yolu (web, Auto-LoRA, kolay akış, `start-train.ps1`, `hektor train --run`,
  `pretrain-gate`) temiz ağaç + bu kod özeti ve güncel eğitim verisi için kapanmış kayıt ister
  (`uv run hektor kademe2-kayit --findings f.json --data-sha <sha> --evidence "..."`).
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
- LoRA/RAG iyileştirme işi → `docs/PROTOKOL_LORA_RAG_IYILESTIRME.md` (2×2 base/LoRA × RAG,
  LLM-30 geliştirme seti `evals/llm30/`, rubrik + kritik hata kapısı). Eval soruları/anahtarları
  eğitime ve retrieval indeksine GİRMEZ.

## ⚡ Yeni seans başlangıcı

1. **`HANDOFF.md`'yi oku** — durum, sıradaki adım ve açık işler oradadır.
2. **Sistem durumunu kontrol et:**
   ```bash
   uv run hektor status     # Ollama + korpus + model
   uv run hektor doctor     # bu makine origin/main'de mi (salt-okuma)
   ```
3. **Kapıyı çalıştır** (değişiklikten önce ve sonra): `make ci`.

Bu depo `alimirbagirzade/hektor`'dır; v1'den ne taşınmadığı `docs/MIGRASYON_2.0.md`'de.

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
- **Bu makine kullanıcının masaüstüdür — stres/yük testi sınırlı.** 2026-10-06'da paralel
  stres (8 paralel pytest × onlarca tur + 48 süreçli CPU yakıcı) yüzlerce konsol penceresi
  açtı, masaüstü kilitlendi, yeniden başlatma gerekti. Bu yüzden:
  - CPU yakıcı / yapay yük süreci (`multiprocessing` busy-loop vb.) **çalıştırma**.
  - Paralel test kopyası en fazla **4**, toplam stres süresi **≤ 10 dk**; daha fazlası
    gerekiyorsa önce kullanıcıya sor. Gerçek alt süreç doğuran testlerde tur sayısını
    düşük tut ve bittiğinde artık süreç kalmadığını (`Get-CimInstance Win32_Process`) doğrula.
  - Yeni `subprocess` çağrısı → `creationflags=NO_WINDOW`; arka plan süreci →
    `procutil.DETACHED_HIDDEN` (`DETACHED_PROCESS` YASAK). `tests/test_no_console_windows.py`
    bunu zorlar.
