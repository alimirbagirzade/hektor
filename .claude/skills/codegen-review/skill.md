# /codegen-review — Kod Üretimi Kalite Denetimi

Yeni yazılan trading/araştırma kodu için kalite kontrolü yapar.

## Ne zaman kullan

- Yeni bir indikatör `app/trading/indicators.py`'a eklendiğinde
- Yeni bir strateji şablonu `strategy_generator.py`'a eklendiğinde
- Herhangi bir `app/` modülü değiştirildiğinde commit öncesi

## Denetim adımları

### 1. Kalite kapıları çalıştır
```bash
make format   # ruff format (line-length 100)
make lint     # ruff check (0 ihlal zorunlu)
make typecheck  # mypy strict (0 hata zorunlu)
make test     # tüm offline testler geçmeli
```

### 2. Yeni indikatör kontrol listesi
Yeni indikatör eklenirse şunlar zorunlu:
```python
# app/trading/indicators.py
# - registry sözlüğüne ekle
# - vektörize pandas/numpy kullan (döngü değil)
# - NaN başlangıç değerlerini doğru handle et

# tests/test_indicators.py
# - en az 3 test: normal input, edge case, NaN handling
```

### 3. Güvenlik kontrolleri
```python
# Yasak:
eval(...)  # kesinlikle yasak
exec(...)  # kesinlikle yasak

# Strateji kuralları yalnızca:
import re
re.match(pattern, rule)  # güvenli parse
```

### 4. Deterministik kontrol
```python
# Rastgelelik her zaman seed ile:
np.random.seed(seed)
random.seed(seed)
```

### 5. Belge/docstring kontrolü
```python
# Docstring Türkçe olmalı (kullanıcıya dönük metinler)
# Log mesajları Türkçe:
logger.info("İndikatör hesaplandı: %s", name)
```

## Önemli dosyalar
- `app/trading/indicators.py` — indikatör registry
- `app/trading/strategy_ir.py` — güvenli strateji parse
- `tests/test_indicators.py` — indikatör testleri
- `Makefile` — `make ci` tüm kontrolleri çalıştırır
