"""LLM-30 geliştirme benchmark'ı — ``Hektor_LLM_30_Soru.txt`` v1.0 (2026-09-30).

Protokol: ``docs/PROTOKOL_LORA_RAG_IYILESTIRME.md``. Sorular ``evals/llm30/validation.jsonl``
altında, profil eval şemasıyla (``EvalItem``) ve sha256 manifestiyle tutulur; yükleme
``app.evals.profile.dataset_loader.load_split`` üzerinden yapılır (hash değişirse RED).

Bu set **geliştirme** setidir (split = validation): teşhis ve aday seçimi için okunabilir,
bu yüzden FİNAL test yerine GEÇMEZ. Final test ayrı, görülmemiş şablonlarla yazılmalıdır.

Puanlama anahtar kelimeyle YAPILMAZ: çok maddeli açık uçlu cevaplar rubrik (``RUBRIC``)
ile insan/hakem tarafından puanlanır; kritik hata (``CRITICAL_ERRORS``) ortalamaya
gömülmez, tek başına terfiyi engeller. Küçük sayısal alt maddelerin anahtarı
``numeric_keys`` ile deterministik hesaplanır (elle yazılmış sabit değil).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from app.evals.profile.dataset_loader import load_split
from app.evals.profile.schema import EvalItem, Split
from app.lora.mix_common import repo_root

SOURCE = "Hektor_LLM_30_Soru.txt v1.0 (2026-09-30)"
QUESTION_IDS: tuple[str, ...] = tuple(f"llm30-s{n:02d}" for n in range(1, 31))

#: Soru dosyasının "ORTAK CEVAP TALİMATI" — 2×2'nin DÖRT koşulunda da aynı sistem istemi.
SYSTEM_PROMPT = (
    "Her alt maddeyi ayrı cevapla. İstenen yerde formül, birim, hesap veya kısa "
    "çalıştırılabilir kod ver. Gerekli varsayımı belirt. Yapmadığın hesaplamayı veya "
    "çalıştırmadığın kodu test edilmiş gibi sunma. Sayısal örneğin açıklamanla uyumunu kontrol "
    "et. Verilmeyen piyasa, komisyon ve performans değerlerini varsayım olarak etiketle.\n"
    "RAG açıkken kaynağın desteklediği iddiaya kaynak kimliği ekle. Kaynakta bulunmayan genel "
    "bilgiyi açıkça ayır. Kaynak gerektiren görevde yetersiz kaynak varsa neyin eksik olduğunu "
    "söyle. Her cevaba gereksiz backtest veya yatırım uyarısı ekleme.\n"
    "Kısa varsayılan format: doğrudan cevap / hesap veya örnek / varsayım ve sınırlama / "
    "doğrulama."
)


def user_prompt(question: str, context: str | None) -> str:
    """RAG kapalı: yalnız soru. RAG açık: dondurulmuş bağlam + soru (C ve D'de byte-aynı)."""
    if context is None:
        return question
    return f"KAYNAKLAR:\n{context}\n\nSORU:\n{question}"


_SHORT_LOOP = re.compile(r"(.{2,24}?)\1{5,}", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
_LETTER = re.compile(r"[^\W\d_]")
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def answer_flags(
    answer: str, *, done_reason: str, prompt_tokens: int, output_tokens: int, num_ctx: int
) -> list[str]:
    """Ham cevaba DOKUNMADAN sorun bayrakları (protokol Aşama 2: ayrı işaretle).

    - ``kesildi_token_siniri``: Ollama ``done_reason == "length"`` (num_predict doldu)
    - ``baglam_siniri``: istem + çıktı ≥ num_ctx (bağlam penceresi taştı/doldu)
    - ``kisa_tekrar``: 2-24 karakterlik birim art arda ≥6 kez
    - ``uzun_tekrar``: ≥20 karakterlik aynı cümle/satır ≥3 kez
    - ``bos_cevap``, ``cjk_sizinti``
    """
    flags: list[str] = []
    if done_reason == "length":
        flags.append("kesildi_token_siniri")
    if prompt_tokens + output_tokens >= num_ctx:
        flags.append("baglam_siniri")
    if not answer.strip():
        flags.append("bos_cevap")
    # Harf içermeyen birimler (markdown tablo çizgisi "---", "====", sıfır dizisi) sayılmaz.
    if any(_LETTER.search(m.group(1)) for m in _SHORT_LOOP.finditer(answer)):
        flags.append("kisa_tekrar")
    seen: dict[str, int] = {}
    for s in _SENT_SPLIT.split(answer):
        s = s.strip()
        if len(s) >= 20:
            seen[s] = seen.get(s, 0) + 1
    if any(n >= 3 for n in seen.values()):
        flags.append("uzun_tekrar")
    if _CJK.search(answer):
        flags.append("cjk_sizinti")
    return flags


#: Teknik puan boyutları → azami puan. ``kaynak_destegi`` yalnız RAG görevlerinde raporlanır;
#: kaynak gerektirmeyen koşulda sıfır kaynak cezası VERİLMEZ.
RUBRIC: dict[str, int] = {
    "dogruluk": 4,
    "alt_madde": 2,
    "tutarlilik": 2,
    "dogrulama": 2,
    "kaynak_destegi": 2,
}

#: Protokol Aşama 3 kritik hata listesi. Biri bile varsa aday terfi EDEMEZ.
CRITICAL_ERRORS: dict[str, str] = {
    "dusus_mumu_gecersiz": "Geçerli düşüş mumunu (close < open) veri hatası saymak",
    "ema_alpha_ters": "EMA'da alpha küçüldükçe tepkinin arttığını söylemek",
    "tz_lokalizasyon": "Saat dilimsiz zamanı yanlış lokalize etmek / UTC varsaymak",
    "ceyrek_bar_birlestirme": "Dört 15 dk barı yanlış kuralla (ör. close=first) birleştirmek",
    "split_temettu_getirisi": "Split/temettüyü getiri hesabında yanlış ele almak",
    "yanlis_atr": "True range / ATR tanımını yanlış vermek",
    "density_olasilik": "Histogram density çıktısını olasılık kütlesi sanmak",
    "indeks_hizalama": "Series indeks hizalamasıyla sahte geçiş matrisi üretmek",
    "global_normalizasyon": "Geçiş matrisini satır yerine global normalize etmek",
    "markov_bagimsizlik_homojenlik": (
        "Markov özelliğini bağımsızlık/zaman homojenliği ile karıştırmak"
    ),
    "final_testte_secim": "Final test verisinde parametre/model seçmek",
    "shift1_her_sizintiyi_cozer": "shift(1)'in tüm sızıntıları çözdüğünü söylemek",
    "rastgele_deger": "Hesap/test sonucu yerine uydurma/rastgele değer vermek",
}

#: Soru → o cevapta özellikle denetlenecek kritik hatalar (rubrik notu).
CRITICAL_BY_QUESTION: dict[str, tuple[str, ...]] = {
    "llm30-s02": ("dusus_mumu_gecersiz",),
    "llm30-s03": ("tz_lokalizasyon",),
    "llm30-s06": ("ceyrek_bar_birlestirme", "rastgele_deger"),
    "llm30-s08": ("shift1_her_sizintiyi_cozer",),
    "llm30-s09": ("yanlis_atr",),
    "llm30-s10": ("split_temettu_getirisi", "rastgele_deger"),
    "llm30-s12": ("ema_alpha_ters", "rastgele_deger"),
    "llm30-s13": ("final_testte_secim", "rastgele_deger"),
    "llm30-s14": ("rastgele_deger",),
    "llm30-s19": ("density_olasilik", "rastgele_deger"),
    "llm30-s20": ("shift1_her_sizintiyi_cozer",),
    "llm30-s21": ("markov_bagimsizlik_homojenlik",),
    "llm30-s22": ("indeks_hizalama", "global_normalizasyon", "rastgele_deger"),
    "llm30-s23": ("markov_bagimsizlik_homojenlik",),
    "llm30-s24": ("final_testte_secim",),
    "llm30-s25": ("rastgele_deger",),
    "llm30-s26": ("rastgele_deger",),
    "llm30-s27": ("final_testte_secim",),
    "llm30-s28": ("shift1_her_sizintiyi_cozer",),
    "llm30-s29": ("final_testte_secim",),
    "llm30-s30": ("final_testte_secim",),
}


def llm30_root() -> Path:
    return repo_root() / "evals" / "llm30"


def load_llm30(*, purpose: str, root: Path | None = None) -> list[EvalItem]:
    """30 geliştirme sorusunu hash doğrulamalı yükle (split = validation)."""
    return load_split(Split.VALIDATION, purpose=purpose, root=root or llm30_root())


def _transition_matrix(states: list[int], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Ardışık geçiş sayıları + SATIR-normalize matris (çıkışı olmayan satır NaN).

    Numpy dizisi üzerinde konumsal eşleşme: ``s[:-1]`` → ``s[1:]`` (indeks hizalaması yok).
    """
    s = np.asarray(states, dtype=int)
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (s[:-1], s[1:]), 1)
    row = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row > 0, counts / np.where(row == 0, 1, row), np.nan)
    return counts, probs


def numeric_keys() -> dict[str, object]:
    """Sayısal alt maddelerin deterministik anahtarları (float toleransı: 1e-9).

    Sözleşmeler: yüzdeler yüzde puanı (1.0 = %1), zamanlar ISO-8601, EMA
    ``yeni = alpha*fiyat + (1-alpha)*önceki``, entropi log2 (bit), geçiş matrisi durum
    sırası 0,1,2 ve satır normalize.
    """
    utc = (
        pd.Timestamp("2026-01-15 10:00:00")
        .tz_localize("Europe/Istanbul")
        .tz_convert("UTC")
        .isoformat()
    )

    bars = pd.DataFrame(
        {
            "open": [100, 104, 106, 102],
            "high": [105, 108, 107, 110],
            "low": [99, 102, 101, 100],
            "close": [104, 106, 102, 109],
            "volume": [10, 20, 30, 40],
        },
        index=pd.date_range("2026-01-15 10:00", periods=4, freq="15min", tz="UTC"),
    )
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    hourly = bars.resample("1h", label="left", closed="left").agg(agg).iloc[0]

    def ema(prev: float, price: float, alpha: float) -> float:
        return alpha * price + (1 - alpha) * prev

    prior_mean, prior_var, obs, r = 100.0, 4.0, 106.0, 2.0
    gain = prior_var / (prior_var + r)

    def entropy_bits(p: list[float]) -> float:
        return 0.0 - sum(x * math.log2(x) for x in p if x > 0)  # 0·log0 := 0; -0.0 değil

    counts, probs = _transition_matrix([0, 1, 0, 2, 1, 0], 3)

    p_up, win, loss, cost = 0.60, 1.0, 2.0, 0.10
    gross = p_up * win - (1 - p_up) * loss

    return {
        "s03_utc": utc,
        "s06_hourly_ohlcv": [float(hourly[c]) for c in ("open", "high", "low", "close", "volume")],
        "s10_split_toplam_deger": (10 * 100.0, 20 * 50.0),
        "s10_split_duzeltilmemis_getiri_pct": (50.0 / 100.0 - 1) * 100,
        "s10_temettu_toplam_getiri_pct": ((98.0 + 2.0) / 100.0 - 1) * 100,
        "s12_ema_alpha_0_1": ema(100.0, 110.0, 0.1),
        "s12_ema_alpha_0_5": ema(100.0, 110.0, 0.5),
        "s13_z": (104.0 - 100.0) / 2.0,
        "s14_kalman_kazanci": gain,
        "s14_guncel_durum": prior_mean + gain * (obs - prior_mean),
        "s14_guncel_varyans": (1 - gain) * prior_var,
        "s19_entropi_esit": entropy_bits([0.5, 0.5]),
        "s19_entropi_deterministik": entropy_bits([1.0, 0.0]),
        "s22_gecis_sayilari": counts.tolist(),
        "s22_satir_normalize": probs.tolist(),
        "s25_brut_beklenen_pct": gross,
        "s25_net_beklenen_pct": gross - cost,
        "s25_basabas_olasilik": (loss + cost) / (win + loss),
        "s26_serbest_parametre_k3": 3 * (3 - 1),
        "s26_serbest_parametre_k10": 10 * (10 - 1),
    }
