# Achilles Trader AI Constitution

> Yerel-öncelikli AI trading **araştırma** sistemi: PDF literatür → RAG/bilgi kartı
> → (opsiyonel LoRA) → disiplinli backtest. **Canlı bot değil, tavsiye değil.**
> Bu anayasa `CLAUDE.md`'deki mutlak kuralları spec-driven prensiplere kodlar.

## Core Principles

### I. Yatırım Tavsiyesi Değil — Hipotez + Test Noktası
Sistem ASLA yatırım tavsiyesi üretmez. Her çıktı bir _hipotez_ ve onu doğrulayacak
_test noktası_ olarak sunulur. "Al/sat", "kesin kazanç", "garanti" gibi ifadeler
yasaktır. Kullanıcıya dönük her sonuç araştırma çıktısıdır, eylem talimatı değil.

### II. Test Edilmeden Onaylama Yok (NON-NEGOTIABLE)
Bir strateji/indikatör/adapter, backtest **ve** örneklem-dışı (out-of-sample)
doğrulamadan geçmeden "başarılı/çalışıyor/hazır" sayılamaz. `verdict != pass` ise
çıktı **"aday"**dır, "hazır" değildir. Eğitim için: eğitim COMPLETED + eval geçmeden
"model eğitildi/iyileşti" denmez.

### III. Backtest Disiplini — Maliyet, Look-ahead, Determinizm
Her backtest: (a) komisyon **+** slippage maliyetlerini dahil eder; (b) look-ahead
bias YASAK — pozisyonlar `shift(1)` ile gecikmelidir; (c) rastgelelik daima `seed`
parametresiyle belirleyicidir. Bu üçü olmadan üretilen sonuç geçersizdir.

### IV. Güvenli Yürütme — `eval`/`exec` Yasak
Strateji kuralları ve model çıktıları yalnızca güvenli regex/JSON ile ayrıştırılır.
`eval`, `exec` veya dinamik kod yürütme hiçbir koşulda kullanılmaz.

### V. Kaynak Bütünlüğü — Uydurma Yok
RAG cevabı yalnızca getirilen gerçek kaynaklara dayanır. Retrieval boşsa sistem
açıkça "kaynak bulunamadı" der; kaynak/formül/sonuç UYDURMAZ. Her bilgi kartı ve
formül gerçek bir makaleye (`paper_id`, foreign key) bağlıdır.

### VI. Kontrollü Eğitim — Varsayılan Dry-Run
Ağır eğitim asla otomatik başlamaz. `train` varsayılan olarak dry-run'dır; gerçek
eğitim yalnızca açık `--run` ile yapılır. Production adapter terfisi ve smoke test
(200+ örnek) kullanıcı onayı gerektirir (`lora-training-control-plane`).

### VII. Yerel-Öncelikli & Belirleyici
Sistem yerel Ollama/MLX ile çalışır; bulut/API zorunluluğu yoktur. Testler
**çevrimdışı** çalışır (fake embedding + sentetik veri); Ollama/MLX gerektirenler
`@pytest.mark.ollama` / `@pytest.mark.slow` ile işaretlidir. Belirleyicilik (seed)
her stokastik adımda zorunludur.

## Teknoloji & Kod Standartları

- Python ≥ 3.12, `from __future__ import annotations`.
- pydantic v2 modelleri; SQLAlchemy 2.0 tipli API.
- ruff (line-length 100, target py312) + mypy (pydantic plugin) — **temiz** olmalı.
- İndikatörler saf pandas/numpy ile **vektörize** (döngü değil).
- Kullanıcıya dönük metin/log/docstring **Türkçe**.
- Gizli anahtar/credential commit edilmez; `data/`, `models/`, `vector_db/`,
  `storage/` çıktısı commit edilmez (`.gitkeep` hariç).

## Geliştirme Akışı & Kalite Kapıları

Her değişiklik sonrası ZORUNLU doğrulama:
```bash
make format && make lint && make typecheck && make test
```
Mimari sözleşmeler:
- `paper_id` içerik hash'inden türer → ingestion **idempotent**.
- Strateji yaşam döngüsü: `hipotez → StrategyIR → backtest → evaluate → verdict`.
- Yeni indikatör → `app/trading/indicators.py` registry + test.
- Yeni CLI komutu → `app/main.py` + README tablosu.
- Yeni web endpoint → MCP senkronu (`scripts/sync-mcp.sh`, [docs/PROTOKOL_MCP.md](../../docs/PROTOKOL_MCP.md)).

## Governance

Bu anayasa diğer tüm pratiklerin üzerindedir. Bir prensiple çelişen değişiklik
gerekçelendirilmeli ve anayasa güncellenmelidir (sürüm artışı + tarih). Runtime
geliştirme yönergesi: `CLAUDE.md`. Spec-driven akış: `/speckit.specify` →
`/speckit.plan` → `/speckit.tasks` → `/speckit.analyze` → `/speckit.implement`.

**Version**: 1.0.0 | **Ratified**: 2026-06-13 | **Last Amended**: 2026-06-13
