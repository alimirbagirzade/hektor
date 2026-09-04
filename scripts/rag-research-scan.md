# Achilles RAG — Ucuz Tarama (Triage) Turu

> Bu, **hafif** bir turdur: güncel RAG literatürünü tarar, yeni adayları
> `docs/egitim/rag-watchlist.md`'ye işler. **Kod yok, sürüm yok, PDF yok, test yok.**
> Yalnız watchlist push edilir. Pahalı entegrasyon `scripts/rag-research-cycle.md`'dedir.

## Adımlar
1. `docs/egitim/rag-watchlist.md`'yi oku (zaten ne var, ne ertelendi).
2. **Web'i tara** (ana döngüde kendin; alt-ajan/Workflow oturum limitine takılabilir):
   son ~1 hafta RAG/retrieval/eğitim gelişmeleri. arXiv/resmi blog/benchmark tercih et.
3. **Triage:** her yeni teknik için kısa adversarial bak — gerçek mi (hype değil),
   offline-uyumlu mu, Achilles'te zaten var mı, kabaca değer/risk.
4. **Watchlist'i güncelle:** yeni satır ekle (durum=`aday`) veya mevcut satırı güncelle
   (tekrar ekleme). Net şekilde güçlü görünenleri "güçlü aday" diye işaretle (örn. notta `**güçlü**`).
5. **Yeni bir şey yoksa:** dosyayı DEĞİŞTİRME, "tarama yapıldı, yeni aday yok" de ve çık
   (ucuz no-op — uydurma yapma, Kural 7).
6. **Push (yalnız watchlist):** değişiklik varsa `git add docs/egitim/rag-watchlist.md`
   → `git fetch origin && git rebase origin/main` → commit (`docs(rag-watchlist): tarama …`)
   → `git push origin main`. Başka dosyaya DOKUNMA.

## Eşik (entegrasyona ne zaman geçilir?)
Watchlist'te **≥1 "güçlü aday"** birikince entegrasyon turu (`rag-research-cycle.md`)
mantıklı olur. Tarama turu kendi başına entegrasyon YAPMAZ.
