<!-- Achilles PR şablonu — otomatik PR akışı (scripts/open-pr.sh) ile uyumlu. -->

## Ne değişti?
<!-- 1-3 cümle: bu PR neyi/neden değiştiriyor -->

## Doğrulama (CLAUDE.md Kademe-0 kapısı)
- [ ] `make format && make lint && make typecheck && make test` — hepsi yeşil
- [ ] Yeni indikatör/strateji varsa: registry'ye eklendi + test yazıldı
- [ ] Backtest/eval içeriyorsa: **look-ahead yok** (`shift(1)`), **maliyet dahil** (komisyon+slippage), **OOS** geçti

## Tür
- [ ] Düzeltme (fix)
- [ ] Özellik (feat)
- [ ] Doküman / araç (docs/chore)

## Notlar
<!-- Risk, takip işleri, ilgili bağlam (varsa) -->

> CI (`lint · types · tests (offline)`) yeşil olunca bu PR otomatik squash-merge olur.
