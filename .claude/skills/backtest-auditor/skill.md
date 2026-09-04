# /backtest-auditor — Backtest Denetim Skili

Bir backtest sonucunu güvenilirlik açısından sistematik olarak denetler.

## Ne zaman kullan

- Bir `StrategyIR` backtest'ten sonra sonucu değerlendirmeden önce
- `verdict == pass` gelmeden önce ek doğrulama yapmak istediğinde
- Overfit şüphesi olduğunda

## Denetim kontrol listesi

### 1. Look-ahead bias kontrolü
```python
# strategy_ir.py'daki her sinyal:
# signal = (condition).shift(1)  # pozisyon bir bar gecikmeli mi?
grep -n "shift" app/trading/backtester.py
grep -n "shift" app/trading/strategy_ir.py
```

### 2. Out-of-sample (OOS) split kontrolü
```bash
# evaluator.py varsayılan: 80/20 split
# Sonuçlarda IS Sharpe > OOS Sharpe ise overfit
uv run achilles backtest <csv> --verbose
```

### 3. Overfit kontrolleri
```python
# app/trading/overfit_checks.py
# - Parametre sayısı vs işlem sayısı oranı
# - İn-sample vs out-of-sample performans farkı
# - Minimum işlem eşiği (10 işlem < → inconclusive)
```

### 4. Maliyet doğrulama
```bash
# Strateji her zaman komisyon + slippage içermeli
grep "commission" app/trading/backtester.py
grep "slippage" app/trading/backtester.py
```

### 5. SQLite kaydı kontrol
```bash
uv run python -c "
from app.memory.sqlite_store import SqliteStore
s = SqliteStore()
print(s.list_backtests())
"
```

## Geçer/Kalır kriterleri

| Kriter | Geçer | Kalır |
|--------|-------|-------|
| OOS Sharpe | > 0.5 | ≤ 0 |
| İşlem sayısı | ≥ 10 | < 10 |
| IS-OOS farkı | < 0.5 | ≥ 0.5 |
| Max drawdown | > -50% | ≤ -70% |

## Önemli dosyalar
- `app/trading/evaluator.py` — OOS verdict mantığı
- `app/trading/overfit_checks.py` — static overfit testleri
- `app/trading/backtester.py` — backtest motoru
