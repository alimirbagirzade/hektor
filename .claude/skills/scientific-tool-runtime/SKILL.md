# /scientific-tool-runtime — Bilimsel Araç Çalışma Zamanı (deterministik doğrulama)

Matematik / istatistik / olasılık / risk iddialarını **LLM'in kafasından değil**, saf-numpy
deterministik araçlarla doğrular (talimat §8, Modül 5). Amaç: ikna edici-ama-yanlış formül/
olasılık/backtest yorumunu önlemek. Her çıktı **hipotez/test-noktası**dır — tavsiye değil.

## Ne zaman kullan
- Bir olasılık/risk-of-ruin/beklenen-değer iddiası kesinleşmeden önce.
- "Bu korelasyon anlamlı mı?" (p-değeri permütasyon testiyle, t-testi değil).
- Bir backtest/risk metriği gerçekçi mi (Sharpe>5 → look-ahead şüphesi).
- Strateji hipotezini sayısal sınamadan önce kaba risk profili.

LLM'e açık bir hesap sorusu sorma; **araca** sor. Hesap kritikse bu skill'i kullan.

## Araçlar (CLI)
```bash
# Kayıtlı araçları + determinizm sözleşmesini listele
uv run achilles tools-list

# Monte Carlo equity simülasyonu + risk-of-ruin (seed ZORUNLU — Kural 6)
uv run achilles montecarlo --returns "0.05,-0.02,0.03,-0.04,0.06" --seed 42 --n 1000 --json
uv run achilles montecarlo --csv data/market/trades.csv --seed 42 --ruin 0.5

# İstatistik: iki kolon → Pearson/Spearman + permütasyon p-değeri; tek → betimsel
uv run achilles stats-check --csv data/x.csv --x ret_a --y ret_b --seed 42 --json
```

Programatik:
```python
from app.tools import monte_carlo_equity, correlation_report, list_tools, validate_params
res = monte_carlo_equity([0.05, -0.02, 0.03], seed=42, n_paths=1000)   # res.ruin_probability, res.var_95_pct
rep = correlation_report(xs, ys, seed=42)                              # rep.p_value (permütasyon), rep.warnings
```

## Çıktının anlamı
| Alan | Anlam |
|------|-------|
| `ruin_probability` | yol boyunca sermayenin ≤%(ruin_fraction) seviyesine düşme olasılığı |
| `var_95_pct` / `expected_shortfall_pct` | 5. yüzdelik kayıp % / en kötü %5'in ortalama kayıp %'si |
| `p_value` (stats) | permütasyon testi (iki yönlü, seed'li) — `< 0.05` istatistiksel; **nedensellik DEĞİL** |
| `warnings` | örneklem küçük (n<30) / yüksek korelasyon nedensellik değil / vb. |

## Mutlak kurallar
- **Seed zorunlu** (Kural 6) — `validate_params` seed'siz aracı eksik bildirir.
- **Tavsiye yok** (Kural 1) — Monte Carlo çıktısını "al/sat" sinyaline çevirme.
- **eval/exec yok** (Kural 5) — araçlar saf numpy; rastgele kod yok.
- `result_verifier` (`verify_backtest_metrics`/`verify_kelly`) gerçekçi-olmayan değeri
  (Sharpe>5, Kelly>1, inf/nan) işaretler → iddiayı kesinleştirmeden önce kontrol et.

## Denetim izi
Her CLI koşusu `sqlite: tool_runs` (+ `tool_artifacts`) tablosuna loglanır (seed + özet).
`store.list_tool_runs()` / `store.get_tool_run(run_id)` ile geçmiş çağrılar incelenebilir.
