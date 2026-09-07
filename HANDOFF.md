# HANDOFF — Hektor

_Depo: https://github.com/alimirbagirzade/hektor · Son güncelleme: 2026-09-07 (Gate 7 yanlış pozitifi + eğitim hattı kapıları)_

Yerel-öncelikli AI **trading araştırma** sistemi (Windows · macOS Apple Silicon · Linux).
**Canlı bot değil, yatırım tavsiyesi değil.**

---

## Bu depo nedir?

`alimirbagirzade/achilles` (v1) deposunun **temizlenmiş ve onarılmış** hâlidir. v1'in tüm
çalışan sistemi taşındı; ölü kod, bulut-API kalıntıları ve bayat oturum geçmişi taşınmadı.
Ne çıkarıldığı ve neden: **[docs/MIGRASYON_2.0.md](docs/MIGRASYON_2.0.md)**.

v1 geçmişi arşiv olarak eski depoda durur; bu depo tek "initial commit" ile başlar.

**2026-09-04 — proje `achilles2.0` → `hektor` olarak yeniden adlandırıldı.** CLI `hektor`
/ `hektor-web`, ortam öneki `HEKTOR_`. Mevcut kurulumlar bozulmasın diye iki geriye dönük
uyum kancası korunur (bkz. `app/config/settings.py`, `tests/test_legacy_env_migration.py`):

- Eski `ACHILLES_*` ortam değişkenleri ve `.env` satırları hâlâ okunur (uyarı loglar);
  açık `HEKTOR_*` ayarı her zaman kazanır. Bu destek **geçicidir**.
- Yalnız eski `storage/sqlite/achilles_trader_ai.db` varsa ona düşülür — korpus/kart
  geçmişi öksüz kalmaz. Dosyayı (WAL/SHM ile birlikte) yeniden adlandırmak yeterlidir.

Bilinçli olarak **değişmeyen** dış sözleşme: `.achpkg` uzantısı, JSON'daki
`achilles_package_version` anahtarı ve `source: achilles_research` değeri — bunları
Entropia tarafı okur, kırılmasınlar diye korundu.

---

## Durum

| Alan | Durum |
|---|---|
| Kapı (`make ci`) | ✅ ruff format + ruff check + mypy (217 dosya) + pytest **1843 passed, 1 skipped** (2026-09-07, `2b0c263`) |
| LLM | Yalnız yerel Ollama (`qwen3:4b` varsayılan). Bulut API istemcisi YOK. |
| Gözetimsiz eğitim | **KAPALI** (`unattended_training_enabled=false`) → her gerçek eğitim tek-kullanımlık insan onayı ister (Kural 8) |
| Arka plan döngüleri | Web açılışında çalışır; `HEKTOR_BACKGROUND_LOOPS_ENABLED=false` ile kapatılır (testlerde kapalı). **Bu makinede `.env` şu an `false`** — 2026-09-06 sunucu yeniden başlatmasında döngüler kapalı açıldı; açmak bilinçli karar ister |
| Bilgi kartı tanımı | "Kartı var" = canlı (`rejected` değil) **ve içerikli** (`card_has_content`: title veya main_claim alfanümerik). Boş kart = kart yok → makale yeniden kartlanabilir (`has_knowledge_card` / `get_latest_knowledge_card`) |
| Test izolasyonu | Testler gerçek `data/` · `storage/` ağacına **yazamaz**; ihlal ederse paket FAIL verir |

---

## Yeni seansta ilk 5 dakika

```bash
uv sync --extra dev            # bağımlılıklar (pytest/ruff/mypy 'dev' extra'sındadır)
uv run hektor status         # Ollama + korpus + model durumu
uv run hektor doctor         # bu makine origin/main'de mi (salt-okuma teşhis)
make ci                        # format + lint + typecheck + test
uv run hektor-web            # http://127.0.0.1:8765
```

Ollama kapalıysa: `ollama serve` → `ollama pull qwen3:4b` → `ollama pull nomic-embed-text`.

---

## Sekiz mutlak kural (CLAUDE.md)

1. Yatırım tavsiyesi üretme — çıktı daima hipotez + test noktası.
2. Test edilmeden "başarılı" deme — backtest + out-of-sample şart.
3. Maliyetleri yok sayma — komisyon + slippage her backtest'te.
4. Look-ahead yasak — pozisyon `shift(1)` ile gecikmeli.
5. `eval`/`exec` yok — strateji kuralları yalnız güvenli regex ile.
6. Determinizm — rastgelelik daima `seed` ile.
7. Kaynak uydurma — retrieval boşsa açıkça söyle.
8. Otomatik ağır eğitim yok — `train` varsayılan dry-run; gerçek eğitim `--run` + taze insan onayı.

---

## Sıradaki adım — LoRA eğitimi (insan onayı bekliyor)

Veri hattı v1'de kapanmıştı; bu depoda **veri taşınmadı** (`data/`, `storage/`, `models/`,
`vector_db/` git'te izlenmez). Yeni makinede sıfırdan üretilir:

```bash
uv run hektor ingest                 # PDF'leri data/papers/raw_pdf/ altına koy, sonra indeksle
uv run hektor synth-qa-bulk --target 1000
uv run hektor lora-curate --run
uv run python scripts/assemble_sft.py  # → data/lora_sft/lora_sft.jsonl (KANONİK)
uv run hektor lora-audit             # Gate 0-7 (--run ile 0-8)
uv run hektor pretrain-gate          # GO / NO-GO
uv run hektor lora-split
# Kural 8 kapısı:
uv run hektor approval-approve <id>
.\scripts\start-train.ps1 -Profile discipline_safe_local   # DETACHED
```

Eğitim sonrası: `lora-eval` (min_n≥5, degenerasyon + boş-cevap vetolu) → adapter **ADAY**;
production terfisi ayrı insan onayı ister.

---

## Son seans — 2026-09-07: Eğitim hattı kapıları + Gate 7 yanlış pozitifi (kapandı)

**Gate 7 (BLOCKER) meşru veri setini kilitliyordu.** Gece üretimiyle onaylı kart 14 → 161
olunca `lora-audit` Gate 7 (safety) BAŞARISIZ verdi. Sebep sır DEĞİL, dedektör hatasıydı:
`_API_KEY_CANDIDATE` aday regex'i `/` ve `-` içerdiğinden bir GitHub bağlantısının host+yolu
TEK token olarak eşleşti (`com/AThreeH1/Global-Permutation-Entropy`: 3 karakter sınıfı,
4.41 entropi > 3.5 eşiği). Kartta (`card_ddb93d79f6b9`, paper_044fec06f4ff) hiçbir kimlik
bilgisi yok — kart okundu ve doğrulandı.

**Düzeltme (`2b0c263`)** veriye değil dedektöre: genel entropi sezgisi artık URL'in
şema+host+YOL bölümünde uygulanmaz (`_url_path_spans`). Yanlış-negatif korunur:
bilinen sır ön-ekleri (`ghp_`/`AKIA`/`sk-`/`xox…`) TÜM metinde — URL yolu dahil — önce aranır,
URL'in query/fragment bölümü maskelenmez (`?api_key=<sır>` hâlâ yakalanır). 3 regresyon testi;
fikstür dizgeleri parça parça kurulur (gitleaks pre-commit kancası bir kez tetiklendi —
kanca ATLANMADI, fikstür düzeltildi).

**Eğitim hattı durumu (2026-09-07 08:00):**

| Kapı | Durum |
|---|---|
| Stage 1 eşiği | 867/1000 (synth 708 + kart 159) — `synth-qa-bulk` canlı üretiyor |
| `pretrain-gate` | **GO** (blocker 0; uyarı: 57 maliyet-token'sız cevap, disiplin 289/528) |
| `lora-audit` Gate 0-7 | **GEÇTİ** (161/161 onaylandı, 20 inceleme işaretli) |
| `lora-curate` | 159 kanonik kart (orphan 0, çok-versiyon 0) |
| `lora-split` | train/valid ayrımı hazır |
| Kademe-2 derin av | **ÇALIŞIYOR** — bitmeden `hunt_ack` YOK |
| Kural 8 taze insan onayı | **BEKLİYOR** — ajan tüketmez |

**Kural 8 sınırı korundu:** gerçek eğitim başlatılmadı, `approval-approve` / `train --run` /
`/api/training/run` ÇAĞRILMADI. Orkestrasyon `deep-hunt` kapısında bloke
(`orc_f8fd6d720df34c24`, `orc_0de294252b6341e2`).

---

## Son seans — 2026-09-06: Boş bilgi kartı arayüz hatası (kapandı)

**Belirti.** Kütüphane'de bazı makaleler "✓ KARTI GÖR" gösteriyor, kart "(başlıksız)" açılıyor
ve "BİLGİ KARTI ÜRET" düğmesi kaybolduğu için makale bir daha kartlanamıyordu.

**Kök sebep.** Eski builder'ın (6dd6214 öncesi) yazdığı 7 **boş `pending`** kart
`has_knowledge_card` tarafından "kart var" sayılıyordu; `get_latest_knowledge_card` de en yeni
kartı içerik bakmadan döndürüyordu. Arayüz kodunda hata yoktu.

**Düzeltme (`75652df`).**
- `app/memory/sqlite_store.py`: iki erişimci de reddedilmiş VE içeriksiz kartı atlar; en yeni
  kart boş olsa bile daha eski içerikli canlı kart döner. Tek tanım: `card_has_content`.
- `app/web/static/assets/app.js` `renderCard`: içeriksiz kart açıkça "Bilgi kartı içeriksiz"
  + "↻ YENİDEN ÜRET" düğmesi (artık "(başlıksız)" yok).
- `tests/test_has_knowledge_card_rejected.py`: boş pending kart sayılmaz; içerikli kart tercih edilir.

**Canlı doğrulama.** Sunucu main'den yeniden başlatıldı (ayrık `uv run hektor-web`, çıktı
`logs/hektor-web.log` / `logs/hektor-web.err.log`). Boş kartlı makalede `GET /api/card/<id>` 404;
yedi makale yeniden "BİLGİ KARTI ÜRET" gösteriyor. Aynı gün başka oturumun `hektor read-all`
koşuları 5 yeni içerikli kart üretti → kartlı makale 10/159.

**Temizlik.** Aynı gün 7 boş kart `rejected` yapıldı (bu seans dışından). Ardından kullanıcı kararıyla
veritabanındaki **28 `rejected` satırın tamamı silindi** (hepsi içeriksizdi; `knowledge_cards`'a FK veren
tablo yok). Silme öncesi tutarlı yedek: `storage/sqlite/backups/hektor_trader_ai.pre-rejected-delete-20260906-225927.db`
(42 kart). Kalan: 11 approved + 3 pending, hepsi içerikli. Aynı kararla `reports/papers/` altındaki
eski builder kalıntısı **31 içeriksiz `*_card.json`** de silindi (önce zip arşivi:
`storage/sqlite/backups/reports_papers_empty_cards-<zaman>.zip`); 17 içerikli dosya kaldı.

---

## Bilinen açık işler


- `docs/MIGRASYON_2.0.md` §"Kalan adaylar" — Phase-4 GitHub otomasyonu (hiç aktive edilmedi),
  `training/dataset_builder.py` ikinci veri hattı, bulut-GPU protokol dokümanları.
- `docs/MIMARI_REFERANS.md` v1 temizliğinden ÖNCE yazıldı; kaldırılan modülleri hâlâ anlatır
  (dosya başında uyarı vardır).

## Önemli dosyalar

| Dosya | Ne |
|---|---|
| `CLAUDE.md` | Çalışma kuralları (bağlayıcı) |
| `docs/MIGRASYON_2.0.md` | v1 → 2.0 farkları |
| `docs/MIMARI_REFERANS.md` | Alt sistem alt sistem mimari referansı (v1 dönemi) |
| `automation_manifest.yaml` | Runtime ajanlarının tek bildirimsel kaynağı + zincir |
| `configs/lora/lora_profiles.yaml` | LoRA eğitim profilleri (`discipline_safe_local` varsayılan) |
| `docs/SCOPE_ISOLATION.md` | Sürücü motor ≠ insan yetkisi |
| `SECURITY.md` | Tehdit modeli + ağa açma checklist'i |
