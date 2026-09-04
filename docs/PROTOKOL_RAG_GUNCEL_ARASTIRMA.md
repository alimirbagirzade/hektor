# PROTOKOL — Güncel RAG Araştırma & Entegrasyon Döngüsü

_Sürüm: 1.0 · 2026-06-17_

Bu protokol, "güncel RAG yeniliklerini periyodik tarayıp işe yarayanları projeye entegre
etme" işinin **nasıl, hangi sıklıkta ve hangi ajanlarla** yürütüldüğünü tanımlar. CLAUDE.md
kurallarına tabidir (özellikle Kural 2: test edilmeden "çalışıyor" deme; Kural 7: uydurma yok).

---

## 1. Neden iki katman? (mimari karar)

İş iki farklı yetenek gerektirir:

| Katman | İş | Gereken yetenek | Nerede koşar |
|---|---|---|---|
| **Tarama (triage)** | arXiv'de RAG yöntemlerini ara → adayları izleme listesine yaz | arama + heuristik | **Projeye yerleşik** (`achilles rag-scan`) — Claude/kota YOK |
| **Entegrasyon** | Adayı değerlendir → kod yaz → doküman sürümle → test → push | **kodlama ajanı** | Claude Code headless (`claude -p`) |

**Önemli dürüstlük notu:** Entegrasyon (yeni modül yazma, dokümanı sürümleme, commit/push)
projenin kendi Python çalışma-zamanıyla yapılamaz; bir kodlama ajanı gerektirir. Bu yüzden
"tamamen projeye gömülü tek ajan" mümkün değildir — tarama gömülüdür, entegrasyon kodlama
ajanıyladır. Bu ayrım maliyeti (Claude kotası) yalnız gerçekten gerektiği yere (entegrasyon)
sınırlar.

---

## 2. Sıklık (KARAR)

6 saatte bir **tam tur FAZLA SIK**: entegre edilmeye değer bir gelişme o hızda çıkmaz →
ya anlamsız no-op'lar ya da "boş geçmeme" baskısıyla marjinal/riskli entegrasyon (Kural 2/7
ihlali). Ayrıca kota tükenir ve eşzamanlı gece-loop'uyla push çekişmesi olur.

**Benimsenen ritim:**

| Tur | Sıklık | Ne yapar | Maliyet |
|---|---|---|---|
| **Tarama** | **günlük (24 saat)** | yeni adayları watchlist'e ekler; push | ucuz (kota yok) |
| **Entegrasyon** | **haftalık (168 saat)** | watchlist'te ≥1 **güçlü aday** varsa tam tur; yoksa no-op | Claude kotası |

**Eşik kuralı:** Entegrasyon turu kendiliğinden entegre etmez; `docs/egitim/rag-watchlist.md`'de
adversarial filtreyi geçen (gerçek + offline + düşük-risk + Achilles'te eksik) **≥1 güçlü aday**
yoksa kod değişikliği zorlamaz — turu no-op olarak kapatır.

---

## 3. Bileşenler

| Dosya | Rol |
|---|---|
| `app/research/rag_trend_scanner.py` | Tarama ajanı çekirdeği (arXiv + heuristik skor; offline-test edilebilir) |
| `achilles rag-scan` (CLI, `app/main.py`) | Tarama ajanı giriş noktası (`--max-per-query`, `--min-score`, `--dry-run`) |
| `docs/egitim/rag-watchlist.md` | Aday biriktirme tablosu (durum: aday/entegre/ertelendi/red) + otomatik tarama bölümü |
| `scripts/rag-research-cycle.md` | Entegrasyon turu talimatı (kodlama ajanına verilir; watchlist'ten başlar + eşik kapısı) |
| `scripts/rag-research-scan.md` | (Opsiyonel) daha zengin, Claude-tabanlı manuel tarama talimatı |
| `scripts/rag-research-loop.ps1` | Zamanlayıcı sarmalayıcı: `-Mode Scan` → `achilles rag-scan`; `-Mode Integrate` → `claude -p` |
| `docs/egitim/RAG_EGITIM_DETAYLI_ANLATIM.md` | Her entegrasyon turunda SÜRÜMLENİR + "Güncel Araştırma Entegrasyonu (Sürüm Günlüğü)" |

---

## 4. Aktivasyon

Sistem görevleri **kullanıcı tarafından** kurulur (kalıcı görev = geri-dönüşü zor):

```powershell
# Önerilen: günlük tarama + haftalık entegrasyon
.\scripts\rag-research-loop.ps1 -Mode Scan -Register        # her 24 saat
.\scripts\rag-research-loop.ps1 -Mode Integrate -Register   # her 168 saat (haftalık)

# Tek seferlik elle:
.\scripts\rag-research-loop.ps1 -Mode Scan
uv run achilles rag-scan --dry-run        # yalnız listele, yazma

# Kaldırma:
.\scripts\rag-research-loop.ps1 -Mode Scan -Unregister
.\scripts\rag-research-loop.ps1 -Mode Integrate -Unregister
```

> Tam gözetimsiz entegrasyon (pytest + git push sorulmadan) için
> `-Mode Integrate -PermissionMode bypassPermissions` — güvenlik etkisini bilerek seç.

---

## 5. Guardrail'ler ve notlar

- **Sadece kendi dosyalarını commit et** (`git add <yol>`, asla `-A/.`); eşzamanlı gece-loop
  WIP'ini karıştırma. Push öncesi `git fetch + rebase origin/main`.
- **Doğrulama zorunlu** (entegrasyon turu): `ruff format` + `ruff check` + `mypy app` +
  `pytest -q` yeşil olmadan commit yok.
- **Kota/eşzamanlılık:** Workflow alt-ajanları oturum limitine takılabilir → web aramayı ana
  döngüde yap. pytest'i ayrı `--basetemp` ile koş (gece-loop'u `.pytest_tmp`'i kilitleyebilir).
- **Sürümleme:** her entegrasyon turunda doküman sürümü ARTAR; entegre edilen adaylar
  watchlist'te `entegre` işaretlenir.
