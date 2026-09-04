# /paper-mastery — Paper Mastery Agent

RAG sisteminizin bir makaleyi ne kadar iyi "öğrendiğini" deterministik olarak ölçer.
100 üzerinden 9 bileşenli skor üretir, durum makinesini günceller, JSON + Markdown rapor yazar.

## Ne Zaman Kullan

- Yeni PDF eklendikten sonra RAG kalitesini doğrulamak istiyorsun
- Bir makalenin "learned" / "needs_rechunking" durumunu anlamak istiyorsun
- Tüm makaleleri toplu değerlendirmek istiyorsun

## Temel Komutlar

```bash
# Tek makale testi
uv run achilles mastery-run <paper_id>

# Kuyruğu göster
uv run achilles mastery-queue

# Tüm makaleleri kuyruğa ekle
uv run achilles mastery-queue --enqueue-all

# Sıradaki makaleyi test et
uv run achilles mastery-queue --run-next

# Tüm kuyruğu işle (maks 50)
uv run achilles mastery-queue --run-all --limit 50

# Son skoru göster
uv run achilles mastery-score <paper_id>

# Raporu göster
uv run achilles mastery-report <paper_id>
```

## Skor Formülü (0–100)

| Bileşen             | Maks | Kaynak             |
|---------------------|------|--------------------|
| Parse               | 10   | SQLite (n_chars, n_pages, hash) |
| Metadata            | 5    | SQLite (title, year, authors)   |
| Chunk Kalitesi      | 15   | SQLite (chunk sayısı, kısa/uzun oran) |
| Index               | 10   | SQLite (embedded chunk sayısı)  |
| Retrieval           | 15   | RAG exam (context_sufficient)   |
| Citation            | 15   | CitationVerifier                |
| Grounding           | 15   | GroundingVerifier               |
| Abstention          | 10   | Trick sorulara "bilmiyorum" demek |
| Formül/Argüman      | 5    | Ticaret hipotezi soruları       |

## Durum Makinas

```
≥ 90 → learned
≥ 75 → usable_needs_review
≥ 60 → partially_learned
≥ 40 → needs_rechunking
< 40 → failed
```

## Raporlar

`reports/papers/mastery/<paper_id>_mastery_report.json`
`reports/papers/mastery/<paper_id>_mastery_report.md`
