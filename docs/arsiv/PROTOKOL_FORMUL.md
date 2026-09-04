# FORMÜL OLUŞTURMA PROTOKOLÜ

> Amaç: makalelerden matematiksel formülleri çıkarıp kavram grafiğine bağlamak;
> bu birikimden **var olan bilgi üzerine yeni indikatör/strateji hipotezleri** üretmek.
> İlke: kaynak uydurma yok — her formül gerçek bir makaleye (`paper_id`) bağlıdır.

## 1. Çıkarım (LLM tabanlı)

`FormulaExtractor.extract_from_paper` — chunk başına LLM sorgusu (ilk ~3000 karakter,
`fmt="json"`, `_EXTRACT_PROMPT`; "formül yoksa `[]`"), `_parse_json_list` ile güvenli parse.
Ollama kapalıysa **`_rule_based_extract`** yedeği (`_KNOWN_INDICATORS` — ATR, RSI, EMA… `latex/plain=None`).

```bash
uv run achilles extract-formulas [paper_id] [--force]   # CLI
# veya  POST /api/research/extract   (web Trader Beyin → Formül Çıkar)
```

## 2. Şema + kavram grafiği

**`Formula` tablosu** (`sqlite_store.py`): `formula_id, paper_id (FK), name, latex,
plain, description, variables_json, category, created_at`.
Kategoriler: `momentum / trend / volatility / volume / risk / statistical`.

**`ConceptLink` tablosu** — yönlü kenarlar, yalnız `_VALID_RELATIONS`:
`extends / measures / limits / combines / opposite_of / requires`.
`ConceptGraph.build_from_papers()` formüller+kavramlar arası bağları kurar.

## 3. Yeni indikatör sentezi (asıl hedef)

Çıkarılan formüller + kavram grafiği → `cross_paper_synthesizer` / `synthesis_engine`
**çapraz makale sentezi** yapar → daha önce denenmemiş indikatör/algoritma önerir →
otomatik backtest (bkz. [PROTOKOL_BACKTEST.md](PROTOKOL_BACKTEST.md)) → sentez makalesi.

Tetik: `uv run achilles research "..."` veya web Araştırma sekmesi.

## 4. Bütünlük kuralları
- **Kaynak zorunlu** — `paper_id` foreign key; kaynaksız formül kaydedilmez.
- **Güvenli parse** — yalnız JSON; `eval`/`exec` yok.
- **Determinizm** — aynı makale + aynı model = aynı çıkarım (sıcaklık düşük, `fmt=json`).
- Çıkarım yoksa açıkça `[]` — uydurma yapılmaz.

## 5. Doğrulama
```bash
curl -s http://127.0.0.1:8765/api/research/formulas   # çıkarılan formüller
curl -s http://127.0.0.1:8765/api/research/graph       # kavram bağlantıları
```
Web: **Trader Beyin** sekmesi → "1 · Formül Çıkarımı" listesi + kavram grafiği.

## 6. Sürekli öğrenme döngüsünde
Her yeni makale ingest edildiğinde kart üretimiyle birlikte formül çıkarımı
çalıştırılabilir; çıkan formüller bir sonraki araştırma turunun sentez havuzunu büyütür.
