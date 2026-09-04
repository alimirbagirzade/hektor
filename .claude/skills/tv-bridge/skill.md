# tv-bridge — TradingView MCP Köprüsü

Achilles'in ürettiği Pine Script'i canlı TradingView'a gönder, strateji testi sonuçlarını al.

## Ne zaman kullan

- `achilles pine` çalıştırıldıktan sonra kodu TV'ye göndermek istediğinde
- Araştırma döngüsü → `PASS` → TV'de canlı doğrulama yapmak istediğinde
- Achilles backtest sonuçlarını TradingView Strategy Tester ile karşılaştırmak istediğinde

## Ön koşullar

```bash
# TradingView Desktop kurulu olmalı
# Bu skill çalışmadan önce:
uv run achilles-web   # Achilles API aktif olmalı (8765)
```

## Adım 1 — TradingView'ı başlat

```tool
mcp__tradingview__tv_launch
```

Başarılıysa: `{ "success": true, "pid": ... }`. Başarısızsa tekrar dene.

## Adım 2 — Sağlık kontrolü

```tool
mcp__tradingview__tv_health_check
```

`api_available: true` görene kadar birkaç saniye bekle ve tekrar dene.
`chart_symbol` değeri gerçek bir sembol ise (örn. "BTCUSD") hazırsın.

## Adım 3 — Pine Script al

Seçenek A — Backtest ID'ye göre (web UI'dan veya `/api/backtests`'ten al):
```bash
curl -s http://localhost:8765/api/backtest/<BACKTEST_ID>/pine | python3 -c "import sys,json; print(json.load(sys.stdin)['pine_code'])"
```

Seçenek B — Strateji adına göre:
```bash
uv run achilles pine <strateji_adı>
```

Seçenek C — Son PASS backtest (otomatik):
```bash
curl -s "http://localhost:8765/api/backtests?limit=50" | python3 -c "
import sys, json
recs = json.load(sys.stdin)['records']
passed = [r for r in recs if r.get('verdict') == 'pass']
if passed: print(passed[0]['backtest_id'])
else: print('PASS_YOK')
"
```

## Adım 4 — Grafiği hazırla

İstenen sembole geç (backtest ile aynı olmalı):
```tool
mcp__tradingview__chart_set_symbol
symbol: "BTCUSD"   # veya backtest'teki market değeri
```

Zaman dilimine geç:
```tool
mcp__tradingview__chart_set_timeframe
timeframe: "60"   # 15m→15, 1H→60, 4H→240, 1D→D
```

## Adım 5 — Pine Script'i yükle

Yeni script aç:
```tool
mcp__tradingview__pine_new
type: "strategy"
```

Kodu ekrana yaz:
```tool
mcp__tradingview__pine_set_source
source: "<PINE_CODE_BURAYA>"
```

Derle:
```tool
mcp__tradingview__pine_smart_compile
```

Hata varsa `pine_get_errors` ile oku, gider, tekrar derle.

## Adım 6 — Strateji Tester sonuçlarını al

```tool
mcp__tradingview__data_get_strategy_results
```

Dönen değerler: `net_profit`, `total_trades`, `win_rate`, `profit_factor`,
`max_drawdown`, `sharpe_ratio`, `sortino_ratio`

## Adım 7 — Sonuçları karşılaştır

Achilles backtest ile TradingView sonuçlarını yan yana göster:

| Metrik         | Achilles | TradingView |
|----------------|----------|-------------|
| Toplam getiri  | ?%       | ?%          |
| Sharpe         | ?        | ?           |
| Max drawdown   | ?%       | ?%          |
| Kazanma oranı  | ?%       | ?%          |
| İşlem sayısı   | ?        | ?           |

Büyük farklar varsa (>%10): entry/exit logic ve komisyon ayarlarını karşılaştır.

## Adım 8 — (Opsiyonel) Ekran görüntüsü

```tool
mcp__tradingview__capture_screenshot
region: "strategy_tester"
```

## Sık karşılaşılan sorunlar

| Sorun | Çözüm |
|-------|-------|
| `api_available: false` | TV henüz yükleniyor, birkaç sn bekle ve health_check tekrarla |
| Pine derleme hatası `undeclared identifier` | İndikatör adı yanlış; `ta.ema` → `ta.ema(close, period)` kontrol et |
| Strategy Tester boş | Grafik yeterli bar yüklenmemiş; `chart_scroll_to_date` ile eski tarihe git |
| Sonuçlar Achilles'ten çok farklı | Commission type/value uyuşmuyor; Pine'daki `commission_value` % mi? |

## Tam örnek (BTCUSD 1H)

```bash
# 1. Pine kodu al
curl -s "http://localhost:8765/api/backtest/bt_cd00c55600/pine" > /tmp/btc_pine.json
PINE=$(python3 -c "import json; print(json.load(open('/tmp/btc_pine.json'))['pine_code'])")

# 2. TV'ye gönder (Claude Code MCP araçlarıyla)
# → chart_set_symbol BINANCE:BTCUSDT
# → chart_set_timeframe 60
# → pine_new strategy
# → pine_set_source $PINE
# → pine_smart_compile
# → data_get_strategy_results
```
