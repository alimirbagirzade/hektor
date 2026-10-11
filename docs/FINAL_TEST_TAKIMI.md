# Final test takımı + rubrik kapısı (TASLAK — soru içeriği kullanıcıdan bekleniyor)

_Durum 2026-10-11 · karar: denetim `reports/audit/denetim-2026-10-11.md` §3 E1/E2 · protokol
`docs/PROTOKOL_LORA_RAG_IYILESTIRME.md` §10_

**Yeni LoRA eğitimi ölçüm hazır olana kadar DURDURULDU.** Terfi kararı: kilitli final takımı +
rubrik/kritik hata kapısı → son adım kör **insan** incelemesi. Bu doküman biçimi ve süreci
tanımlar; **soru metni ve anahtarlar kullanıcıdan gelir** (Claude soru yazmaz).

## 1. Neden ayrı bir final takımı

| | LLM-30 (`evals/llm30/`) | Final takımı (`evals/final/`) |
|---|---|---|
| Rol | geliştirme / teşhis — okunur | terfi kararı — görülmez |
| Seçimde kullanım | evet | **yasak** (`purpose="selection"` → hata) |
| Depo | açık depoda | **yerel** (`.gitignore`; v15 deseni) |
| Kullanım | sınırsız | **tek** final kullanımı; ikincisi geliştirme sayılır (erişim günlüğü) |
| Puan | rubrik (elle) | rubrik (kör insan) + deterministik sayısal anahtar |

## 2. Kategoriler (öneri — onay bekliyor)

Toplam ≥ **60** soru; her kategori ≥ 20 soru ve ≥ 5 aile ("aile" = aynı şablondan türeyen
sorular; bootstrap birimi — aynı ailenin çok sorusu tek kanıt sayılır).

| Kategori | Alt kategori | Önerilen adet | Ölçtüğü | Tipik kritik kod |
|---|---|---|---|---|
| **A. bicim_disiplin** | `kural1_dili` | 6 | tavsiye/kesinlik isteyen soruya hipotez + test noktası | `kural1_tavsiye_dili` |
| | `maliyet_oos_uyarisi` | 5 | maliyetsiz / OOS'suz "başarı" iddiasına doğru tepki | `maliyet_yok_sayildi`, `oos_olmadan_basari` |
| | `kaynakli_uzun_cevap` | 5 | dondurulmuş kanıtla uzun, atıflı cevap (kanıt zorunlu) | `uydurma_kaynak` |
| | `cekimserlik` | 4 | kaynak/veri yetersizken neyin eksik olduğunu söyleme | `rastgele_deger`, `uydurma_kaynak` |
| **B. hesap** | `sharpe` | 5 | Sharpe (rf, yıllıklandırma, ddof açık) | `anahtara_aykiri_hesap` |
| | `drawdown` | 5 | en büyük düşüş (özsermaye eğrisinden) | `anahtara_aykiri_hesap` |
| | `net_getiri_maliyet` | 6 | komisyon + slippage sonrası net getiri | `anahtara_aykiri_hesap`, `maliyet_yok_sayildi` |
| | `diger_metrik` | 4 | Sortino / yıllık oynaklık / CAGR / profit factor / beklenti | `anahtara_aykiri_hesap` |
| **C. alan_bilgisi** | `backtest_metodoloji` | 4 | IS/OOS, walk-forward, aşırı uyum, final testte seçim | `final_testte_secim`, `look_ahead` |
| | `risk_yonetimi` | 4 | pozisyon boyutu, stop, risk-of-ruin | `kural1_tavsiye_dili` |
| | `piyasa_yapisi_maliyet` | 4 | spread, slippage, likidite, işlem sıklığı | `maliyet_yok_sayildi` |
| | `istatistik_zaman_serisi` | 4 | durağanlık, otokorelasyon, rejim, Markov | `markov_bagimsizlik_homojenlik` |
| | `strateji_indikator` | 4 | indikatör tanımları, sinyal gecikmesi, shift(1) | `yanlis_atr`, `shift1_her_sizintiyi_cozer` |

Kritik kod kataloğu = LLM-30 `CRITICAL_ERRORS` + final eki (`uv run hektor final-set format`).

## 3. Dosya biçimi (`evals/final/`, yerel)

`questions.jsonl` — satır başına bir soru:

```json
{"id": "final-hesap-01", "category": "hesap", "subcategory": "sharpe",
 "family": "sharpe-gunluk", "question": "<KULLANICI YAZAR — girdiler metinde açık; 'Son satıra yalnız sonucu yaz (2 ondalık)'>",
 "evidence": [], "written_by": "<ad>", "source_provenance": "<kaynak/tarih>", "human_verified": true}
```

`keys.jsonl` — satır başına bir anahtar (aynı `id`):

```json
{"id": "final-hesap-01", "rubric_points": ["<doğru cevapta olması gereken madde>"],
 "critical_checks": ["anahtara_aykiri_hesap"],
 "numeric": {"calc": "sharpe", "inputs": {"returns": [0.01, -0.005, 0.02],
   "rf_per_period": 0.0, "periods_per_year": 252, "ddof": 1},
   "tolerance": 0.01, "unit": "yıllık Sharpe", "human_value": null, "sign_insensitive": false},
 "reference_answer": "", "verified_by": "<ad>"}
```

Kurallar (kod zorlar — `app/evals/final_set.py`):

- **Sayısal anahtar elle yazılmaz**: `calc` + `inputs` verilir, değer `compute_numeric` ile
  hesaplanır. Her hesap türünün girdileri TAM listelenir (eksik/fazla = hata; sessiz varsayılan
  yok). `human_value` verilirse hesapla tolerans içinde eşleşmeli — eşleşmezse kilit yok.
- Sözleşmeler (soru metni bunlarla uyumlu yazılmalı): Sharpe `mean(r−rf)/std(r−rf,ddof)·√P`;
  drawdown pozitif yüzde puanı; net getiri `brüt − 2·(komisyon+slippage)` işlem başına,
  `compounding` ile bileşik/toplam; tümü `uv run hektor final-set format` çıktısında.
- `hesap` sorusu sayısal anahtarsız kilitlenmez; `kaynakli_uzun_cevap` dondurulmuş `evidence`
  olmadan kilitlenmez; her soru `human_verified=true`, her anahtar `verified_by` ister.
- Final soruları LLM-30 / profil / adapter eval setleriyle sızıntı kapısıyla AYNI ölçütle
  (tam, normalize, içerme, yakın kopya) çakışamaz.

## 4. Süreç

```bash
uv run hektor final-set format     # kategoriler, hesap sözleşmeleri, kritik kodlar, şablon
# (kullanıcı) evals/final/questions.jsonl + keys.jsonl yazar
uv run hektor final-set validate   # ≥60, kategori/aile, anahtar, LLM-30'dan ayrılık
uv run hektor final-set lock       # manifest (sha256) bir kez yazılır; sonrası değişiklik = HATA
uv run hektor mix leakage          # rapor "final takımı: kilitli (n=…)" göstermeli
uv run hektor final-set export     # → reports/comparisons/final_sets/final_<sha>.jsonl (yerel)
uv run hektor compare-run --set <export> --role final --active A --candidate C --base B --adapter <ad>
# web → Öğrenme Havuzu → Kör inceleme: rubrik boyutları + kritik hata kodları (insan)
```

## 5. Kapı kuralları (kod)

- **Tek kritik hata = NO-GO.** Final ölçütü (`final_rubrik_v1`, `critical_policy:
  any_candidate`): adayın herhangi bir cevabında, herhangi bir tekrarda kritik hata →
  `kritik_ret` — aktifte de aynı hata olsa bile. Eski kilitli ölçütlerde aday-yalnız kuralı
  korunur, ama kritik hatalı aday hiçbir ölçütte `kabul` alamaz (en fazla `yetersiz_kanit`).
- Otomatik kritik: Kural 1 dili (cevapta) ve sayısal anahtara aykırı / sonuçsuz hesap cevabı.
- Rubrik: `dogruluk 0–4 · alt_madde 0–2 · tutarlilik 0–2 · dogrulama 0–2 · kaynak_destegi
  0–2` (yalnız kanıtlı soruda); soru puanı = 4 × toplam / azami.
- **Tekrar:** Ollama temp 0'da deterministik değil → final ölçütü soru başına **3 tekrar**
  (tohum 42/43/44); tekrarlar soru içinde ortalanır (bağımsız örnek sayılmaz). Sonuç her
  zaman `measurement` bloğunda n (soru), aile, tekrar, cevap sayısı ve tekrarlar arası
  uyuşmazlık sayısıyla raporlanır.
- **Servis edilen yapıt (B6):** `compare-run` zaten Ollama GGUF etiketlerini ölçer.
  `hektor lora-eval <ad> --served-tag <aday> --served-base <base> --repeats 3` aynı eval setini
  Ollama yapıtında ölçer; PEFT yolu raporda "servis edilen GGUF ölçülmedi" notu taşır ve
  kapısı en iyi ihtimalle `yetersiz_kanit` olur.
- `hektor rubric-gate <llm30 raw.jsonl> --candidate B --reference A`: puanlanmış LLM-30 2×2
  koşusundan aynı kapı (soru başına ≥2 tekrar ister; `--seeds 42 43 44`).
- Kapı asla "GO" demez: geçen aday `insan_onayi_bekliyor` / `kabul` (etkinleştirme 2A'da ayrı
  insan eylemi).

## 6. Kullanıcıdan beklenen

1. Kategori/alt kategori/adet önerisinin onayı ya da düzeltmesi (§2).
2. 60+ sorunun metni + anahtarları (§3 biçiminde) — Claude yazmaz; istenirse yalnız biçim
   denetimi ve hesap anahtarlarının `compute_numeric` ile doğrulanmasında yardım eder.
3. `validate` temizse `lock`. Kilitten sonra soru/anahtar değişikliği yeni sürüm demektir.
