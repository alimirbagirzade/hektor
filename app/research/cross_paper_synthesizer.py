"""Çapraz-makale formül sentezi.

Farklı makalelerden çıkarılan formülleri kavram kategorilerine göre gruplar,
matematiksel olarak anlamlı kombinasyonlar üretir ve bunları LoRA eğitim
verisi olarak ``training_examples`` tablosuna yazar.

Çalışma prensibi:
  1. DB'deki tüm formülleri ``category`` alanına göre grupla
  2. Her kategori çifti / üçlüsü için (farklı makalelerden formüller seçerek)
     bir sentez eğitim örneği üret
  3. LLM varsa zengin matematiksel sentez; yoksa şablon tabanlı yedek
  4. Zaten var olan sentez örneklerini tekrar üretme (idempotent)
"""

from __future__ import annotations

import hashlib
import logging
from itertools import combinations
from typing import Any

from app.brain.local_llm import LLMUnavailable, LocalLLM
from app.memory.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
Sen bir quant araştırma uzmanısın. Aşağıdaki FARKLI akademik makalelerden alınan \
matematiksel formülleri ve kavramları inceleyerek ÖZGÜN bir trading indikatörü \
veya strateji bileşeni tasarla.

KAYNAK FORMÜLLER (farklı makalelerden):
{formula_block}

GÖREV — şunların tamamını içer:
1. **İndikatör Adı** — kısa, açıklayıcı
2. **Birleşik Formül** — plain math veya LaTeX, tüm değişkenler tanımlı
3. **Matematiksel Gerekçe** — bileşenler neden birbirine uyar?
4. **Trading Uygulaması** — sinyal üretimi, filtre, pozisyon boyutlandırma
5. **Sınırlar** — hangi piyasa koşullarında zayıflar?

Yalnızca matematiksel olarak tutarlı ve gerçekten uygulanabilir öner.
Yanıtı Türkçe, detaylı ver.
"""

# ---------------------------------------------------------------------------
# Şablon tabanlı yedek (LLM olmadan)
# ---------------------------------------------------------------------------

_FALLBACK_TEMPLATES: dict[frozenset[str], dict[str, str]] = {
    frozenset({"momentum", "entropy"}): {
        "name": "Belirsizlik-Ağırlıklı Momentum (UAM)",
        "formula": (
            "UAM = Momentum_score × (1 − H_norm)\n"
            "H_norm = H(getiri_dağılımı) / log₂(N)\n"
            "Momentum_score = (P_t − P_{t−n}) / P_{t−n}"
        ),
        "rationale": (
            "Shannon entropisi (H) yükseldikçe fiyat hareketleri öngörülemez hale gelir "
            "ve momentum sinyalleri güvenilirliğini yitirir. H_norm ile ağırlıklandırınca "
            "belirsiz dönemlerde sinyal doğal olarak sıfıra yaklaşır."
        ),
        "application": (
            "H_norm > 0.75 → sinyali filtrele\n"
            "H_norm < 0.30 → tam güç momentum sinyali\n"
            "Pozisyon büyüklüğü: f_kelly × (1 − H_norm)"
        ),
        "limits": "Trend kırılma anlarında H geçici olarak düşer; sinyal hâlâ yanlış olabilir.",
    },
    frozenset({"momentum", "volatility"}): {
        "name": "Volatilite-Normalize Momentum (VNM)",
        "formula": (
            "VNM = ROC_n / ATR_normalized\n"
            "ATR_normalized = ATR_14 / P_t\n"
            "ROC_n = (P_t − P_{t−n}) / P_{t−n}"
        ),
        "rationale": (
            "Ham momentum sinyalleri yüksek volatilite dönemlerinde gürültü içerir. "
            "ATR ile normalize etmek farklı volatilite rejimlerinde karşılaştırılabilir "
            "bir sinyal üretir ve aşırı kaldıraçtan korur."
        ),
        "application": (
            "VNM > +2σ → alım\nVNM < −2σ → satım\n"
            "ATR_normalized > 0.03 → pozisyon büyüklüğünü %50 azalt"
        ),
        "limits": "Ani volatilite şoklarında ATR gecikmeli tepki verir (~14 bar gecikme).",
    },
    frozenset({"trend", "entropy"}): {
        "name": "Rejim-Duyarlı Trend Filtresi (RSTF)",
        "formula": (
            "RSTF = EMA_trend × I(H_markov < θ)\n"
            "H_markov = −Σ P(s_t|s_{t−1}) × log P(s_t|s_{t−1})\n"
            "θ = 0.60  (kalibrasyon parametresi)"
        ),
        "rationale": (
            "Markov geçiş entropisi yüksekken piyasa rejim geçiş dönemindedir; "
            "bu dönemlerde EMA trend sinyali bastırılır ve yanlış sinyal riski azaltılır."
        ),
        "application": (
            "H_markov < 0.40 → trend sinyale gir\n"
            "H_markov > 0.60 → bekle / flat kal\n"
            "Geçiş bölgesi 0.40−0.60 → pozisyonu yarıya indir"
        ),
        "limits": "θ parametresi piyasa ve zaman dilimine göre kalibre edilmeli.",
    },
    frozenset({"risk", "momentum"}): {
        "name": "Kelly-Momentum Dinamik Pozisyon (KMDP)",
        "formula": (
            "f_adj = f_kelly × φ(Momentum)\n"
            "f_kelly = (b·p − q) / b\n"
            "φ(M) = 1 / (1 + exp(−k·M))  # sigmoid, k=3 varsayılan"
        ),
        "rationale": (
            "Kelly fraksiyonu teorik optimum pozisyonu verir; momentum gücü sinyal güvenini "
            "yansıtır. sigmoid(M) ile ikisini birleştirmek hem istatistiksel hem teknik "
            "analiz temelli pozisyon büyüklüğü üretir."
        ),
        "application": (
            "Güçlü momentum (M > +1σ) → f_adj ≈ f_kelly\n"
            "Zayıf/negatif momentum → f_adj → 0\n"
            "Her zaman f_adj ≤ f_kelly (Kelly sınırını aşma)"
        ),
        "limits": "Kelly'nin kazanma olasılığı (p) tahmininin doğruluğuna bağlı.",
    },
    frozenset({"regime", "risk"}): {
        "name": "HMM Rejim-Farkındal Risk Ölçeği (HRRS)",
        "formula": (
            "σ_adj = σ_base × P(rejim=bear | gözlem_t)\n"
            "P(rejim | gözlem) → HMM forward algoritması\n"
            "Stop = Entry × (1 − k·σ_adj)  # k=2 varsayılan"
        ),
        "rationale": (
            "Gizli Markov modeli gözlemlerden rejim olasılığını hesaplar. Ayı rejimi "
            "olasılığı arttıkça pozisyon riski orantılı olarak azaltılır; boğa dönemlerinde "
            "tam risk kullanılır."
        ),
        "application": (
            "P(bear) > 0.70 → riski %30'a indir\n"
            "P(bear) < 0.30 → tam risk al\n"
            "Geçiş 0.30−0.70 → lineer interpolasyon"
        ),
        "limits": "HMM eğitimi için en az 2 yıllık veri gerekli.",
    },
    frozenset({"entropy", "risk"}): {
        "name": "Belirsizlik-Düzeltmeli Kelly (UAK)",
        "formula": (
            "f_ua = f_kelly × (1 − H_norm)²\nf_kelly = (b·p − q) / b\nH_norm = H_shannon / log₂(N)"
        ),
        "rationale": (
            "Kelly kriteri beklenen büyümeyi maksimize eder ancak piyasa belirsizliğini "
            "görmez. H_norm² ile ikinci dereceden cezalama, yüksek belirsizlikte pozisyonu "
            "agresif küçülterek aşırı kaldıraçtan korur."
        ),
        "application": (
            "H_norm = 0.0 → f_ua = f_kelly (tam Kelly)\n"
            "H_norm = 0.5 → f_ua = 0.25 × f_kelly\n"
            "H_norm > 0.8 → f_ua ≈ 0 (piyasadan çık)"
        ),
        "limits": "Entropi hesabı için yeterli getiri gözlemi gerekir (min 30 bar).",
    },
    frozenset({"trend", "volatility"}): {
        "name": "Volatilite-Bant Trend Takibi (VBTT)",
        "formula": (
            "Üst = EMA_n + k·ATR_m\n"
            "Alt = EMA_n − k·ATR_m\n"
            "Sinyal = sign(P − EMA_n) × I(|P − EMA_n| > k·ATR_m)"
        ),
        "rationale": (
            "Trend yönü EMA ile, anlamlılık eşiği ATR ile belirlenir. Yalnızca fiyat "
            "ATR bandını kırdığında sinyal üretilmesi gürültüyü filtreler."
        ),
        "application": (
            "Kırılma yönünde gir, ATR × 1.5 stop-loss\n"
            "k = 1.0 agresif, k = 2.0 konservatif\n"
            "m = 14 (standart ATR periyodu)"
        ),
        "limits": "Yatay (ranging) piyasalarda çok sayıda yanlış kırılma üretir.",
    },
    frozenset({"momentum", "regime"}): {
        "name": "Rejim-Şartlı RSI (RCRSI)",
        "formula": (
            "RCRSI = RSI × P(trend_rejimi | gözlem_t)\n"
            "RSI = 100 − 100 / (1 + RS)\n"
            "P(trend_rejimi) → 2-durumlu HMM"
        ),
        "rationale": (
            "RSI tüm rejimlerde aynı eşiği kullanır; mean-reversion rejiminde RSI'ın "
            "30/70 seviyeleri çok daha anlamlıdır, trend rejiminde ise geç kalır. "
            "HMM ile ağırlıklandırma her rejime uygun bağlam sağlar."
        ),
        "application": (
            "P(trend) > 0.70 → RSI sinyalini bastır, trend takip et\n"
            "P(MR) > 0.70 → klasik RSI 30/70 kullan\n"
            "RCRSI < 25 → güçlü alım; > 75 → güçlü satım"
        ),
        "limits": "HMM'in doğru rejimi tespit etmesi 5−10 bar gecikme içerir.",
    },
}

# Kategori normalizasyon haritası
_CATEGORY_ALIASES: dict[str, str] = {
    "vol": "volatility",
    "trend_following": "trend",
    "mean_reversion": "momentum",
    "position_sizing": "risk",
    "hidden_markov": "regime",
    "markov": "regime",
    "information": "entropy",
    "uncertainty": "entropy",
    "kelly": "risk",
    "drawdown": "risk",
}


def _normalize_category(cat: str | None) -> str:
    if not cat:
        return "general"
    low = cat.lower().strip()
    return _CATEGORY_ALIASES.get(low, low)


def _formula_block(formulas: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for f in formulas:
        pid = (f.get("paper_id") or "?")[:12]
        name = f.get("name") or "?"
        cat = _normalize_category(f.get("category"))
        latex = f.get("latex") or ""
        plain = f.get("plain") or ""
        desc = f.get("description") or ""
        math_repr = latex or plain or "(formül gösterimi yok)"
        lines.append(
            f"• [{cat}] **{name}** (makale: {pid})\n  Formül: {math_repr}\n  Açıklama: {desc}"
        )
    return "\n\n".join(lines)


def _example_id(formula_ids: list[str]) -> str:
    key = "|".join(sorted(formula_ids))
    return "syn_" + hashlib.sha256(key.encode()).hexdigest()[:24]


def _synthesis_seed(block: str) -> int:
    """Sentez girdisinden deterministik LLM seed türet (CLAUDE.md kural 6).

    Aynı formül bloğu → aynı seed → aynı sentez (REPRODÜKLENEBİLİR); farklı blok →
    farklı seed → girdiye göre doğal çeşitlilik. Böylece seedsiz çağrının
    nondeterminizmi (kural 6 ihlali) kapanır AMA sabit-seed'in 'her girdi için hep
    aynı çıktı' sorununa düşülmez. Süreçler-arası KARARLI sha256 kullanılır (Python'un
    salt'lı ``hash()``'i değil); 32-bit aralık Ollama/OpenAI seed için yeterli.
    """
    return int(hashlib.sha256(block.encode("utf-8")).hexdigest()[:8], 16)


def _select_cross_paper(
    flist_a: list[dict[str, Any]],
    flist_b: list[dict[str, Any]],
    max_per_cat: int = 3,
) -> list[dict[str, Any]]:
    """A ve B kategorisinden farklı makale kaynaklı formüller seç."""
    seen_a: set[str] = set()
    selected_a: list[dict[str, Any]] = []
    for f in flist_a:
        pid = f.get("paper_id", "")
        if pid not in seen_a and len(selected_a) < max_per_cat:
            selected_a.append(f)
            seen_a.add(pid)

    seen_b: set[str] = set()
    selected_b: list[dict[str, Any]] = []
    for f in flist_b:
        pid = f.get("paper_id", "")
        if pid not in seen_b and len(selected_b) < max_per_cat:
            selected_b.append(f)
            seen_b.add(pid)

    if not selected_a or not selected_b:
        return []
    combined = selected_a + selected_b
    # "cross-paper" sözleşmesi: en az 2 FARKLI makale olmalı. Bir makale hem cat_a hem
    # cat_b'de formüle sahipse seçim tek makaleye çökebilir → yanıltıcı sentez. Reddet.
    if len({f.get("paper_id", "") for f in combined}) < 2:
        return []
    return combined


class CrossPaperSynthesizer:
    """Farklı makalelerden alınan formülleri sentezleyerek LoRA eğitim verisi üretir."""

    def __init__(
        self,
        store: SqliteStore | None = None,
        llm: LocalLLM | None = None,
    ) -> None:
        self.store = store or SqliteStore()
        self.llm = llm or LocalLLM()

    def synthesize_all(self, *, force: bool = False) -> int:
        """Tüm cross-paper sentez örneklerini üret. Döner: yeni örnek sayısı."""
        formulas = self.store.list_formulas()
        if not formulas:
            logger.info("Sentez: formül yok, atlanıyor.")
            return 0

        # Kategori bazında grupla
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for f in formulas:
            cat = _normalize_category(f.get("category"))
            by_cat.setdefault(cat, []).append(f)

        valid_cats = [c for c, fs in by_cat.items() if fs and c != "general"]
        if len(valid_cats) < 2:
            logger.info("Sentez: yeterli kategori çeşidi yok (%d).", len(valid_cats))
            return 0

        total = 0

        # İkili kombinasyonlar
        for cat_a, cat_b in combinations(valid_cats, 2):
            selected = _select_cross_paper(by_cat[cat_a], by_cat[cat_b])
            if not selected:
                continue
            ex_id = _example_id([f["formula_id"] for f in selected])
            if not force and self._exists(ex_id):
                continue
            example = self._build(selected, frozenset({cat_a, cat_b}))
            if example:
                self._save(ex_id, example)
                total += 1
                logger.info("Sentez: [%s × %s] → %s", cat_a, cat_b, ex_id)

        # Üçlü kombinasyon: en çok formülü olan ilk 3 kategori
        top3 = sorted(valid_cats, key=lambda c: -len(by_cat[c]))[:3]
        if len(top3) == 3:
            triple = [by_cat[c][0] for c in top3]
            # "cross-paper" sözleşmesi üçlüde de geçerli: ikili yol _select_cross_paper ile
            # ≥2 FARKLI makaleyi zorlar; üçlü körlemesine her kategoriden ilk formülü alır.
            # Tek makale 3 kategoride formüle sahipse üçlü tek makaleye çöker → 'farklı
            # akademik makalelerden' iddiası YANLIŞ olur (uydurma provenans, Kural 7) ve
            # eğitim verisine sızar. En az 2 farklı paper_id şartı koy (yoksa atla).
            if len({f.get("paper_id", "") for f in triple}) >= 2:
                ex_id = _example_id([f["formula_id"] for f in triple])
                if force or not self._exists(ex_id):
                    example = self._build(triple, frozenset(top3))
                    if example:
                        self._save(ex_id, example)
                        total += 1
                        logger.info("Sentez: [%s] → %s", "×".join(top3), ex_id)

        logger.info("CrossPaperSynthesizer: %d yeni örnek.", total)
        return total

    def _exists(self, ex_id: str) -> bool:
        from app.memory.sqlite_store import TrainingExample

        with self.store.session() as s:
            return s.get(TrainingExample, ex_id) is not None

    def _build(
        self,
        formulas: list[dict[str, Any]],
        categories: frozenset[str],
    ) -> dict[str, str] | None:
        block = _formula_block(formulas)
        instruction = (
            "Farklı akademik makalelerden alınan aşağıdaki matematiksel formülleri "
            "ve kavramları birleştirerek özgün bir trading indikatörü veya strateji "
            "bileşeni tasarla. Matematiksel olarak tutarlı ve gerçekten uygulanabilir ol."
        )

        # LLM dene
        try:
            raw = self.llm.generate(
                _SYNTHESIS_PROMPT.format(formula_block=block),
                max_tokens=1024,
                timeout=120,
                seed=_synthesis_seed(block),
            )
            if raw and len(raw.strip()) > 150:
                return {"instruction": instruction, "input": block, "output": raw.strip()}
        except LLMUnavailable:
            logger.debug("LLM yok — şablon kullanılıyor.")
        except Exception as exc:
            logger.debug("LLM sentez hatası: %s", exc)

        # Şablon yedek
        template = _FALLBACK_TEMPLATES.get(categories)
        if template is None:
            for key, tmpl in _FALLBACK_TEMPLATES.items():
                if key & categories:
                    template = tmpl
                    break
        if template is None:
            return None

        output = (
            f"## {template['name']}\n\n"
            f"**Formül:**\n```\n{template['formula']}\n```\n\n"
            f"**Matematiksel Gerekçe:** {template['rationale']}\n\n"
            f"**Trading Uygulaması:**\n{template['application']}\n\n"
            f"**Sınırlar:** {template['limits']}"
        )
        return {"instruction": instruction, "input": block, "output": output}

    def _save(self, ex_id: str, example: dict[str, str]) -> None:
        from app.memory.sqlite_store import TrainingExample

        # merge (upsert): force=True yeniden ürettiğinde içerik SESSIZCE atılmasın
        # (eski "yalnız yoksa ekle" mantığı force regenerasyonu boşa harcıyordu).
        with self.store.session() as s:
            s.merge(
                TrainingExample(
                    example_id=ex_id,
                    source_paper_id=None,
                    example_type="cross_paper_synthesis",
                    instruction=example["instruction"],
                    input_text=example["input"],
                    output_text=example["output"],
                )
            )
            s.commit()
