# /ingestion-quality-scorer — İçe-alım kalite skoru (100 puan, compute-on-demand)

Bir makalenin parse/section/formül/tablo çıkarım kalitesini 100-puanlık rubrikle ölçer (talimat
§4, Modül 1). Kötü parse doğrudan yanlış bilgi üretir; bu kapı düşük-kaliteli makaleyi RAG/eğitim
öncesi işaretler. **PaperIndexer'ın sıcak yolu DEĞİŞMEZ** — skor istek üzerine hesaplanır.

## Ne zaman kullan
- Yeni makaleler ingest edildikten sonra "RAG'a hazır mı?" kontrolü.
- Bir makale RAG'da kötü cevap veriyorsa kök-neden (parse mi bozuk?).
- LoRA/mastery öncesi havuz kalite taraması.

## Komut
```bash
# Tek makaleyi skorla (bileşen kırılımıyla)
uv run achilles ingestion-quality --paper-id paper_abc123 --json

# Skoru KALICI yap (paper_ingestion_runs + papers.quality_score/ingest_status)
uv run achilles ingestion-quality --paper-id paper_abc123 --record
```

Programatik:
```python
from app.ingestion.quality_scorer import score_paper
from app.memory.sqlite_store import SqliteStore
res = score_paper(SqliteStore(), "paper_abc123")   # res.total, res.status, res.components, res.notes
```

## Rubrik (100 puan) ve durum eşikleri
| Bileşen | Puan | Kaynak |
|---------|------|--------|
| parse / ocr | 15 / 10 | karakter yoğunluğu (char/sayfa) |
| metadata | 10 | title + authors + year |
| section | 15 | tanınan bölüm sayısı (abstract/intro/methods/…) |
| formula / table / figure | 15 / 15 / 10 | Formula tablosu / has_table bayrağı / caption regex |
| cleantext | 10 | kontrol-karakteri / encoding-hatası (salt-regex) |

| Durum | Eşik | Anlam |
|-------|------|-------|
| `ready_for_rag` | ≥90 | temiz, kullanıma hazır |
| `usable` | 70-89 | kullanılabilir, küçük eksik |
| `slow_but_usable` | 50-69 | bölüm/formül zayıf |
| `unstable` / `failed` | 40-49 / <40 | yeniden-parse / human-review adayı |

## Kurallar
- **Salt-skor** — yeniden-ingest/eğitim başlatmaz.
- **NULL skor eski makaleyi engellemez** (retrieval gating YOK).
- **Sezgisel** — parse başarısızsa (chunk yok) çıkarım bileşenleri 0; formül/tablo yokluğu parse
  iyiyse nötr puanlanır (eksik çıkarımı haksızca cezalandırmaz).
- Bilinmeyen paper_id → `ValueError`.

Düşük skorda yorumla: hangi bileşen düşük + olası sebep (taranmış PDF → düşük ocr; bölüm yok →
parse bölüm-algısı zayıf). RAG'a hazır değilse açıkça söyle.
