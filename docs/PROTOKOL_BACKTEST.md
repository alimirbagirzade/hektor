# BACKTEST PROTOKOLÜ

> Achilles'te bir strateji hipotezinin **disiplinli** test edilme süreci.
> İlke (CLAUDE.md): maliyet zorunlu, look-ahead yasak, out-of-sample şart,
> determinizm seed ile. `verdict != pass` → çıktı **"aday"**, "hazır" değil.

## 0. Disiplin sözleşmesi (kodla eşleşen)

| Kural | Uygulama (kod) |
|---|---|
| **Look-ahead yasak** | pozisyon `position.shift(1)` ile gecikmeli (`app/trading/backtester.py`) |
| **Maliyet** | `cost = turnover × (commission + slippage)`, varsayılan `0.0005/0.0005` (`CostSpec`) |
| **Güvenli parse** | strateji kuralları yalnız `_RULE_RE` regex ile — `eval`/`exec` YOK |
| **Determinizm** | `generate_synthetic_ohlcv(seed=…)` — aynı seed = aynı sonuç |

## 1. Backtest nasıl çalıştırılır

```bash
uv run achilles gen-data                 # sentetik OHLCV üret
uv run achilles backtest <csv> [--strategy-json strateji.json]
```
- **`POST /api/backtest`** — sentetik veri (`n_bars ∈ [200, 20000]`, `seed ∈ [0, 1e7]`).
- **`POST /api/backtest/csv`** — gerçek CSV (≥50 bar; kolonlar: time, open, high, low, close[, volume]).

## 2. Verdict kriterleri

`evaluate(df, ir, min_trades=30)` karar ağacı **OOS metrikleri** üzerinden:
- **`fail`** (hard): `n_trades < 30` **veya** `max_drawdown < -50%` **veya** `return ≤ 0`.
- **`inconclusive`**: uyarı var veya `sharpe < 0.5`.
- **`pass`**: yukarıdakilerin hiçbiri yok.

> Örnek: `bt_00035294a4` (ema_rsi_trend_filter_v1) → `fail` (Sharpe negatif, az işlem).
> Bu bir **aday değil**, reddedilmiş hipotez olarak raporlanır.

## 3. Zorunlu denetimler (backtest-auditor kapıları — `app/trading/overfit_checks.py`)
- **Look-ahead taraması** — `shift` + `sharpe > 4` ⇒ şüphe bayrağı.
- **Out-of-sample** — `in_out_of_sample` split=0.7; IS≫OOS = klasik overfit işareti.
- **Statik overfit** — `static_checks` (4 uyarı kalıbı).
- **Maliyet duyarlılığı** — komisyon/slippage 2×–5× ile Sharpe değişimi.
- `/backtest-auditor` skill'i bu kapıları otomatik uygular.

## 4. Sonuçlar nerede saklanır (İKİ AYRI YOL — önemli)

| Yol | Saklama | Görünürlük |
|---|---|---|
| Tekil/web backtest | `persist_backtest` → **`backtests`** tablosu + `reports/backtests/*.json` | `GET /api/backtests` |
| Araştırma döngüsü (orchestrator) | `run_backtest` çağırır, `persist_backtest` çağırmaz → **`research_sessions`** | Araştırma sekmesi |

> **Not:** Döngüdeki araştırma backtest'leri `research_sessions`'a yazılır, `backtests`
> tablosuna DEĞİL — Backtest sekmesinde az görünmesinin sebebi bu. İyileştirme:
> orchestrator'a `persist_backtest` eklenebilir.

## 5. Sürekli öğrenme döngüsünde
`scripts/continuous-learning.sh` her **3 turda** (`round % 3 == 1`) `achilles research`
tetikler → hipotez → backtest → yansıma → sentez makalesi → LoRA dataset. Eğitimle sıralı.

## 6. Uygula (şimdi)
```bash
uv run achilles gen-data && uv run achilles backtest data/market/raw/synthetic.csv
curl -s http://127.0.0.1:8765/api/backtests
```
