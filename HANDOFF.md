# HANDOFF — Hektor

_Depo: https://github.com/alimirbagirzade/hektor · Son güncelleme: 2026-09-08 (RAG retrieval config'i ölçümle düzeltildi)_

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
| Kapı (`make ci`) | ⚠️ 2026-09-08: ruff + mypy (217 dosya) ✅, pytest **2039 passed, 1 failed**, 1 skipped. Tek hata `tests/test_peft_lora_recipe.py::test_build_training_kwargs_defaults` — kurulu **transformers 5.16.1** `warmup_ratio`'yu artık kabul etmiyor; `peft_lora_train.py` uyum katmanı doğru davranıyor, TEST bayat. Ayrı görevde düzeltiliyor |
| LLM | Yalnız yerel Ollama (`qwen3:4b` varsayılan). Bulut API istemcisi YOK. |
| Gözetimsiz eğitim | **KAPALI** (`unattended_training_enabled=false`) → her gerçek eğitim tek-kullanımlık insan onayı ister (Kural 8) |
| Arka plan döngüleri | Web açılışında çalışır; `HEKTOR_BACKGROUND_LOOPS_ENABLED=false` ile kapatılır (testlerde kapalı). **Bu makinede `.env` şu an `false`** — 2026-09-06 sunucu yeniden başlatmasında döngüler kapalı açıldı; açmak bilinçli karar ister |
| RAG retrieval | **Router AÇIK** (`.env`: `HEKTOR_RAG_ROUTER=true`, `ROUTER_ALPHA=0.3`, `RERANK=true`). 2026-09-08 ölçümü: keyword recall@1 66→84, semantik gecikme 1074→212 ms. Gerekçe + tablolar: `reports/rag_retrieval_ab_findings.md` §5. Cross-encoder/FlashRank/RRF ölçümle elenmiş → KAPALI |
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

## Yerel eğitim denemesi — 2026-09-07: onay yarışı + bellek darboğazı

**Yerel eğitim yolu ÇALIŞIR durumda ve hiçbir eksiği yok.** Ölçüldü: `train` dry-run
`missing_packages: []`; torch 2.12+cpu / peft 0.19.1 / transformers 5.9.0 kurulu;
`Qwen3-4B-Instruct-2507` (7.6 GB) ve `Qwen2.5-1.5B-Instruct` (2.9 GB, ChatML şablonlu)
HF önbelleğinde indirilmiş; veri bölünmüş (train=1447, valid=76). Disk 158 GB boş.

**Yerel eğitim ABONELİK KULLANMAZ.** Abonelik ajanların kod/araştırma işi içindir;
LoRA eğitimi yalnız yerel CPU'da Python hesabıdır — API çağrısı ve ücret yoktur.

**Bellek gerçeği (bu makine):** 32 GB toplam. 4B model bf16'da ~8 GB ağırlık + aktivasyon
ile pratikte **~18-20 GB**'a çıkıyor (ölçüldü) ve Ollama'nın llama-server'ı ayrıca 4-7 GB
tutuyor → boş RAM ~0.8-1 GB'a iniyor. Yerel koşu için **Qwen2.5-1.5B-Instruct** seçilmeli
(profil notu da ≤1.5B diyor). `scripts/start-train.ps1 -BaseModel ...` bunun içindir ve
seçim `train_status.json`'a yazılıp watchdog tarafından geri okunur (`cf7e893`).

**AÇIK BULGU — onay yarışı (Kural 8 hijyeni).** İki eşzamanlı `train --run` çağrısında
`consume_fresh_approval` CAS'i doğru çalışıyor (yalnız biri tüketir) ama **kaybeden taraf
YENİ bir pending onay üretip kuyrukta bırakıyor**. Gözlendi: kullanıcının verdiği onay
(`apr_7f612c303431`) başka bir oturumun `hektor_smoke_olcum` koşusu tarafından tüketildi;
bizim koşumuz yetki alamayıp `apr_042388be85de`'yi üretti ve durdu. Kuyrukta bu yüzden
kullanılmayan pending istekler birikiyor (şu an birkaç adet).
Öneri: yetki alamayan çağrı, aynı agent+action için **zaten bekleyen** bir istek varsa
yenisini üretmesin (idempotent istek); ya da kuyruk temizliği için `approval-prune`.

**Sıradaki adım:** `hektor_smoke_olcum` koşusu bitip RAM boşalınca, bekleyen onay
onaylanıp gerçek koşu başlatılacak:
`hektor train --run --backend peft --adapter-name hektor_lora_v6_local --iterations 300
--profile discipline_safe_local` (HEKTOR_PEFT_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct).
Reçete doğrulandı: 300 örnek = tam 1 epoch, r=16, lr=1e-4, NEFTune 5, assistant_only_loss
(v5 ezber-regresyonunun panzehiri), seed 42.

---

## Kademe-2 derin av — 2026-09-07: FAIL → 12 bulgu düzeltildi (`ea03b04`)

Eğitim öncesi **zorunlu** Kademe-2 avı çalıştırıldı (8 alt-sistem paralel bulucu →
HIGH/BLOCKER için 3 bağımsız şüpheci oy). **Verdict FAIL:** 45 ham bulgudan 32'si
onaylandı, 10'u bloklayan. Hepsi kodda ayrı ayrı doğrulanıp düzeltildi, her biri kendi
regresyon testiyle kilitlendi. Kapı: pytest **1999 passed**.

| Ciddiyet | Dosya | Neydi |
|---|---|---|
| BLOCKER | `market_data_loader` | CSV zaman sırasına göre **sıralanmıyordu** → ters sıralı dosya ters yönde backtest, fiili look-ahead (Kural 4), OOS dilimi en eski veri |
| HIGH | `peft_lora_train` | Checkpoint'ten koşulsuz devam; eski adım ≥ hedef ise **0 adım eğitip `ok=True`** (Kural 2) |
| HIGH | `detached_launch` | "1 epoch" fiilen **~4.8 epoch** (profil 300'e kırpıyor); satır-düzeyi bölme **18 makaleyi** train+valid'e dağıtıyordu |
| HIGH | `evaluate_model` | Garanti-vaadi deseni 17 varyantın 13'ünü kaçırıyordu |
| HIGH | `delegates` | `approval` aşaması **tek-kullanımlık onayı tüketiyordu** + her resume'da yeni pending (4 birikmişti) |
| HIGH | `engines` | codex av motoru `hardened=True` iddiasına rağmen **çıplak argv** (av, sürüşten gevşek) |
| HIGH | `knowledge_card_builder` | Tip sapmasında **içerikli kart kaydedilmeden çöküyordu** (8 sapma ölçüldü) |
| HIGH | `rag_learning_loop` + `paper_reader` | Başarısız deneme "üretildi" sayılıyor → bütçe tükeniyor, diğer makaleler **açlığa** düşüyordu |
| HIGH | `rlm/lora_candidate` | §16 atıf kapısı **atıfsız** koşularda boş yere sağlanıyordu (Kural 7) |
| HIGH | `confidence_scorer` | Aynı kök: atıfsız cevap ağırlıklı ortalamada **bedava 0.30** puan alıyordu |
| MEDIUM | `rlm_controller` | Zorunlu trading uyarısının idempotens kontrolü serbest cümleye bakıyordu → uyarının kalan 3 satırı eklenmiyordu (Kural 1) |
| — | `start-train.ps1` + `training-watchdog.ps1` | Resume varsayılanı kapanınca **çökme-kurtarma kırıldı**; `-Resume` switch'i eklendi |

**Yanlış-pozitif disiplini:** garanti deseni genişletilirken gerçek veri setindeki 7 meşru
akademik "garanti" (konformal tahmin aralığı, drawdown olasılık sınırı, FDP sınırı) elle
doğrulandı — hepsi hâlâ GO alıyor. Aynı hafta Gate 7'de bir GitHub URL'i API anahtarı
sanılıp 161 kartlık veri setini kilitlemişti; o sınıf hata tekrarlanmadı.

**Not:** Bu avdan sonra düzeltmelerin kendisi ayrıca şüpheci denetimden geçirilmelidir
(düzeltme yeni hata üretmiş olabilir). Eğitimden önce av YENİDEN koşturulmalı.

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

## Son seans — 2026-09-08: RAG retrieval config'i ölçümle düzeltildi (router ENABLE, α 0.7→0.3)

Rapor (`reports/rag_retrieval_ab_findings.md`) iki karar veriyordu ama **hiçbiri
uygulanmamıştı**; uygulamadan önce yeniden ölçtüm — ikisi de kısmen yanlış çıktı.

- **Korpus arada yeniden kurulmuş:** 161 makale / **13.103 chunk** (rapor: 153 / ~94k).
  Eski mutlak sayılar kıyaslanamaz → her config baştan ölçüldü.
- **Semantik metrik DOYDU:** beş config de recall@1 %100 / MRR 1.0. Yani §3'ün
  `HEKTOR_RAG_HYBRID=false` kararının dayanağı (68.6 > 64.3) artık ölçülemiyor.
- **Keyword rejimi karar verdi:** canlı `hybrid+rerank` 66.0/MRR 0.7475; router VARSAYILAN
  α=0.7 ile 46.0/0.5799 (yani §4b'nin "router'ı aç" kararı varsayılan α ile KALİTE KAYBI
  olurdu — rapor router'ı yalnız dense ile kıyasladığı için görememişti).
- **Kök neden füzyon ağırlığı:** α dense ağırlığıdır; keyword rejiminde dense en zayıf
  taraftır. α süpürüldü (0.7/0.5/0.3/0.15/0.0) → **α=0.3'te 84.0/0.8733**, gecikme aynı.
- **Bayrak çakışması ölçümle çözüldü:** `rag_rerank` ANA KAPI (false → router hiç çalışmaz),
  `rag_hybrid` router altında ÖLÜ (`router hybrid=false` ve `hybrid=true` sonuçları birebir aynı).

**Karar (.env, kod varsayılanı değişmedi, geri alınabilir; yedek `.env.bak-20260908-rag`):**
`HEKTOR_RAG_ROUTER=true` · `HEKTOR_RAG_ROUTER_ALPHA=0.3` · `HEKTOR_RAG_RERANK=true`.
`HEKTOR_RAG_HYBRID` bilerek yazılmadı: router kapatılırsa dense-only'e (keyword recall@1
42.0 — en kötü) düşmesin.

**Ölçüm hijyeni (bu makinenin bilinen tuzağı):** ölçümler otomatik "temiz ortam" kapısının
arkasında koştu — ağır süreç yok (lora-eval/pytest/mypy/hektor CLI), Ollama'da üretim modeli
yok, toplam CPU < %30 (üst üste 2 yoklama). `HEKTOR_ALLOW_FAKE_EMBEDDINGS=false` + her koşuda
`# embedder mode:` basılır; ikisi de `ollama` raporladı (sahte embedding ile "sonuç" üretme
riski kapatıldı).

## Bilinen açık işler


- `docs/MIGRASYON_2.0.md` §"Kalan adaylar" — Phase-4 GitHub otomasyonu (hiç aktive edilmedi),
  `training/dataset_builder.py` ikinci veri hattı, bulut-GPU protokol dokümanları.
- **Daha zor semantik golden-set gerek** — kart-türevi self-retrieval metriği doydu (tüm config'ler %100), retrieval kalitesini semantik rejimde artık ayırt edemiyor.
- `rag_contextual_embed` HÂLÂ ÖLÇÜLMEDİ — önce `hektor reindex-contextual` ister (tüm korpusu Ollama ile yeniden embed eder, AĞIR); eğitim hattı boştayken yapılmalı.
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
