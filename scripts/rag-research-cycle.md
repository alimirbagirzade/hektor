# Achilles RAG Güncel-Araştırma Turu — Otonom Tur Talimatı

> Bu dosya, headless Claude Code'a (veya `/loop`'a) verilen **tek bir araştırma turunun**
> talimatıdır. Amaç: güncel RAG literatürünü tarayıp işe yarayanları Achilles'e entegre
> etmek, eğitim dokümanını sürümleyip güncellemek ve push etmek. `scripts/rag-research-loop.ps1`
> bunu ~6 saatte bir headless çalıştırır. Manuel: `claude -p "$(Get-Content -Raw scripts/rag-research-cycle.md)"`.

## Bağlayıcı kurallar (CLAUDE.md)
- Yatırım tavsiyesi üretme; çıktılar hipotez + test noktası.
- **Test edilmeden "çalışıyor" deme** — sadece "kodda eklendi / önerildi" de.
- `eval`/`exec` YOK; determinizm (seed); saf pandas/numpy; offline çalışmalı.
- Ağır bağımlılık eklemekten kaçın; mevcut opt-in desenine uy (varsayılan davranışı bozma).
- Kullanıcıya dönük metin/log/docstring Türkçe.

## Eşik (önce bunu kontrol et)
Bu pahalı turdur (kod+sürüm+test+push). `docs/egitim/rag-watchlist.md`'yi oku: **≥1 "güçlü
aday"** (ucuz tarama turlarının işaretlediği) yoksa, hızlı bir web doğrulaması yap; yine de
nitelikli aday çıkmazsa **tam turu KOŞMA** — watchlist'e "entegrasyona değer aday yok" notu
düş ve çık (no-op; uydurma yapma, Kural 7). Güçlü aday varsa aşağıdaki adımlarla devam et.

## Tur adımları (sırayla)
1. **Durumu oku:** `docs/egitim/rag-watchlist.md` (entegre edilecek adaylar) +
   `docs/egitim/RAG_EGITIM_DETAYLI_ANLATIM.md` ("Güncel Araştırma Entegrasyonu (Sürüm Günlüğü)" —
   son tur ne yaptı, neler ertelendi) + `HANDOFF.md` ilgili kısım. Entegrasyona watchlist'teki
   güçlü adaylardan başla; gerekirse aşağıdaki taze tarama ile tamamla.
2. **Tara (web):** Son ~6 ay RAG/retrieval/eğitim gelişmeleri. Konu örnekleri: reranking &
   late-interaction, chunking (late/contextual/semantic), corrective/self/adaptive RAG, query
   dönüşümü (HyDE/step-back/RRF), GraphRAG ailesi, RAG değerlendirme (RAGAS/groundedness),
   RAFT/embedding ince-ayar, bilimsel/finansal uzun-doküman & formül retrieval. arXiv/resmi
   blog/benchmark tercih et; blog-hype'a güvenme. Önceki turlarda "ertele" denenleri tekrar değerlendir.
3. **Eşle + adversarial doğrula:** Her tekniği Achilles koduna eşle (gerçekten var mı? Read/Grep
   ile kontrol et, varsayma). is_real (hype mi?), değer, offline mı, entegrasyon riski.
4. **Entegre et (seçici):** SADECE gerçekten eksik VEYA belirgin iyileştirme olan, **offline-uyumlu,
   düşük-riskli, deterministik, mevcut mimariye temiz oturan** 1-3 tekniği uygula. Yeni opt-in
   ayar → `settings.py` (varsayılan kapalı). Her yeni davranışa **test yaz**. Riskli/LLM-yoğun/
   ağır olanları ertele (gerekçeyle belgele).
5. **Dokümanı güncelle:** sürüm numarasını ARTIR (Sürüm Geçmişi satırı) + "Güncel Araştırma
   Entegrasyonu" bölümüne **yeni tarihli tur** ekle (en yeni üstte): taranan teknikler tablosu +
   adopt/belgele/ertele gerekçesi + kaynak URL'leri. İlgili Aşama/Sözlük/Dosya-Haritası/Test-Kapsamı
   bölümlerini de güncelle. Ayrıca `docs/egitim/rag-watchlist.md`'de entegre edilen adayları
   `entegre` olarak işaretle (durumları güncel tut).
6. **PDF üret:** `uv run python scripts/gen_egitim_pdf.py` (repo + Desktop'a kopyalar). LORA PDF
   içeriği değişmediyse `git checkout -- docs/egitim/LORA_EGITIM_DETAYLI_ANLATIM.pdf` ile geri al.
7. **Doğrula (zorunlu):** `uv run ruff format .` + `uv run ruff check .` + `uv run mypy app` +
   `uv run pytest -q --basetemp=.pytest_tmp`. Hepsi yeşil olmadan commit etme.
8. **Commit + push:** SADECE kendi dosyalarını `git add <yol>` ile ekle (asla `git add -A/.`).
   Çalışma ağacındaki başka WIP'i karıştırma (gerekirse `git diff --cached` ile doğrula).
   `git fetch origin && git rebase origin/main` (eşzamanlı makine) sonra `git push origin main`.
   Commit mesajı sonu: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
9. **Bir şey bulunmazsa:** Yeni/uygulanabilir bir şey çıkmazsa kod değişikliği ZORLAMA; turu
   "bu turda entegre edilecek yeni teknik yok; şunlar tarandı/teyit edildi" notuyla dokümana
   yaz, sürümü yine de küçük artır ve push et. Uydurma yapma (Kural 7).

## Oturum limiti
Alt-ajan/Workflow oturum limitine takılırsa web aramayı **ana döngüde** kendin yap (limit
genelde yalnız alt-ajan başlatmayı etkiler). Yine de takılırsan turu kısalt ve bir sonraki
pencerede devam et.
