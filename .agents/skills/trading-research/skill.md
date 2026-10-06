# /trading-research — Araştırma Döngüsü Skili

Bu skill, Hektor'in tam araştırma döngüsünü adım adım yürütür:
**Formül Çıkarımı → Kavram Grafiği → Sentez → Backtest → Yansıma → İyileştirme**

## Ne zaman kullan

- Yeni bir trading hipotezi test etmek istediğinde
- Makalelerden formülleri çıkarıp birleştirip yeni indikatör önermek istediğinde
- Trader Beyin araştırma döngüsünü çalıştırmak istediğinde

## Adımlar

### 1. Formül çıkar (Ollama gerekir)
```bash
uv run hektor extract-formulas
uv run hektor formulas          # çıkarılanları listele
```

### 2. Araştırma döngüsü
```bash
uv run hektor research "<soru>"
# Örnek:
uv run hektor research "Momentum göstergeleri yüksek volatilitede nasıl filtrelenir?"
```
Bu komut:
- Tüm formülleri alır → kavram grafini günceller
- `synthesis_engine.py` ile yeni indikatör önerir
- `backtester.py` ile sentetik veri üzerinde test eder
- `reflection_agent.py` ile sonucu yansıtır → iyileştirilmiş IR üretir

### 3. Gerçek veriyle backtest
```bash
# BTCUSD 1H Binance verisi
uv run hektor backtest data/market/raw/BTCUSD_1h_Binance.csv

# Diğer borsalar
uv run hektor backtest data/market/raw/BTCUSD_1h_Coinbase.csv
uv run hektor backtest data/market/raw/BTCUSD_1h_OKX.csv
```

### 4. LoRA eğitim verisi oluştur
```bash
uv run hektor chain-dataset     # araştırma zincirleri → JSONL
uv run hektor dataset           # toplam dataset özeti
uv run hektor train --run       # gerçek LoRA eğitimi (MLX)
```

## Başarı kriterleri
- `verdict == pass`: OOS Sharpe > 0, yeterli işlem sayısı, overfit yok
- `verdict != pass`: çıktı "aday"dır, "hazır" değildir — iterasyona devam
- Her döngü LoRA eğitim verisine katkıda bulunur

## Önemli dosyalar
- `app/research/orchestrator.py` — tam döngü mantığı
- `app/research/synthesis_engine.py` — formül birleştirme + öneri
- `app/research/reflection_agent.py` — backtest → iyileştirme
- `app/trading/backtester.py` — look-ahead-safe backtest
