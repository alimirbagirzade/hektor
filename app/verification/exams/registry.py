"""Sınav registry'si — her gösterge için referans + sınav parametreleri.

Her ``ExamSpec``:
  - ``definition``: modele gösterilecek KESİN convention (belirsizlik olmasın diye;
    sınav "hangi formül" tahminini değil, UYGULAMAYI ölçer),
  - ``reference``: ``compute_indicator`` üzerinden güvenli referans seri,
  - ``rtol``/``atol``: kayan-nokta toleransı (np.allclose),
  - ``monotonic``: L4 karşıolgu için "parametre artarsa beklenen yön" kuralı.

Sadece close-only registry indikatörleri (SMA/EMA/RSI) ilk turda; ATR/Bollinger
OHLC/çok-çıktı gerektirir, sonraki turda eklenir.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.trading.indicators import compute_indicator

__all__ = ["ExamSpec", "get_spec", "list_specs"]


def _rsi_exam_reference(df: pd.DataFrame, period: int) -> pd.Series:
    """RSI referansı — üretim ``fillna(50)`` konvansiyonu OLMADAN (tanımsız bölge NaN).

    Üretim RSI'si warmup'ı (bar 0 + ilk düşüş öncesi) 50 ile doldurur; bu artefakt
    TANIMDA yok, dolayısıyla doğru hesaplayan modeli bile L3'te haksızca fail
    ettiriyordu. NaN bırakılınca ``defined`` maskesi bu bitişik öneki dışlar ve
    sınav yalnız tanımdan TÜRETİLEBİLİR barları puanlar.
    """
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # Kayıpsız pencere RSI=100 (tanımdan türetilebilir limit, RS→∞); kalan NaN = gerçek warmup.
    return out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)


def _entropy_exam_reference(df: pd.DataFrame, period: int) -> pd.Series:
    """ENTROPY referansı — bar 0'daki NaN-kaynaklı SAHTE 'düşüş' artefaktı OLMADAN.

    Üretim entropy()'si close.diff()[0]=NaN'i (delta>0)→0.0 (düşüş) sayar; ilk tanımlı
    pencere yalnız period-1 GERÇEK fark + 1 sahte düşüş içerir → tanımı doğru uygulayan
    modeli L3'te haksızca fail ettirir (RSI fillna(50) ile aynı warmup-artefakt sınıfı).
    Burada bar 0 tanımsız (NaN) bırakılır → pencere period GERÇEK fark ister.
    """
    close = df["close"]
    delta = close.diff()
    up = (delta > 0).astype(float)
    up[delta.isna()] = float("nan")  # sahte 'düşüş' yerine tanımsız
    p = up.rolling(period).mean()  # min_periods=period → period GERÇEK fark gerekir
    q = 1.0 - p
    out = pd.Series(0.0, index=p.index, dtype=float)
    mask = (p > 0.0) & (p < 1.0)
    out[mask] = -(p[mask] * np.log2(p[mask]) + q[mask] * np.log2(q[mask]))
    out[p.isna()] = float("nan")
    return out


@dataclass(frozen=True)
class ExamSpec:
    name: str
    definition: str
    default_period: int = 14
    rtol: float = 1e-3
    atol: float = 1e-4
    needs_ohlc: bool = False
    # L4 için: parametre adı -> beklenen monoton yön (insan-okur + işaret testi notu)
    monotonic: dict[str, str] = field(default_factory=dict)
    # L3 için gereken minimum nokta sayısı (ör. permütasyon entropi geniş pencere ister).
    min_points: int = 0
    # Sınava özel referans (üretim konvansiyonlarından arındırılmış); None → compute_indicator.
    ref_override: Callable[[pd.DataFrame, int], pd.Series] | None = None

    def reference(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Referans seri — sınav-özel override varsa onu, yoksa compute_indicator'ı kullan."""
        if self.ref_override is not None:
            return self.ref_override(df, period)
        return compute_indicator(self.name, df, period)


_SPECS: dict[str, ExamSpec] = {
    "SMA": ExamSpec(
        name="SMA",
        definition=(
            "SMA(p): basit hareketli ortalama. Her bar için, o bar dahil son p kapanış "
            "değerinin aritmetik ortalaması = (x[t-p+1] + ... + x[t]) / p."
        ),
        default_period=3,
        rtol=1e-6,
        atol=1e-6,
        monotonic={"period": "periyot artarsa seri daha pürüzsüz (daha az oynak) olur"},
    ),
    "EMA": ExamSpec(
        name="EMA",
        definition=(
            "EMA(p): üssel hareketli ortalama, alpha = 2/(p+1), adjust=False. "
            "İlk değer EMA[0] = x[0]; sonraki her bar EMA[t] = alpha*x[t] + (1-alpha)*EMA[t-1]."
        ),
        default_period=3,
        rtol=1e-6,
        atol=1e-6,
        monotonic={"period": "periyot artarsa daha pürüzsüz / daha çok gecikme"},
    ),
    "RSI": ExamSpec(
        name="RSI",
        definition=(
            "RSI(p) = 100 - 100/(1+RS). RS = Wilder ortalama kazanç / ortalama kayıp; "
            "kazanç/kayıp serileri ardışık close farklarından, ewm(alpha=1/p, adjust=False) ile."
        ),
        default_period=14,
        rtol=1e-2,
        atol=1e-2,
        monotonic={"period": "periyot artarsa daha az uç değer (daha pürüzsüz)"},
        ref_override=_rsi_exam_reference,
    ),
    "ENTROPY": ExamSpec(
        name="ENTROPY",
        definition=(
            "ENTROPY(p): yönsel ikili Shannon entropisi. Her bar için son p kapanış FARKINA bak; "
            "yukarı-hareket (fark>0) oranı q ise H = -(q·log2(q) + (1-q)·log2(1-q)); q=0 veya 1 "
            "iken H=0, q=0.5 iken H=1. Aralık [0,1]."
        ),
        default_period=4,
        rtol=1e-3,
        atol=1e-3,
        monotonic={"period": "periyot artarsa daha pürüzsüz (daha az oynak)"},
        ref_override=_entropy_exam_reference,
    ),
    "PERMENTROPY": ExamSpec(
        name="PERMENTROPY",
        definition=(
            "PERMENTROPY(p): Bandt-Pompe permütasyon entropisi (gömme boyutu order=3), "
            "[0,1] normalize. Son p bar üzerinde, ardışık 3'lü kapanış pencerelerinin "
            "SIRALAMA (ordinal) desenleri sayılır; desen dağılımının Shannon entropisi "
            "log2(3!)=log2(6) ile bölünür. Monoton seri → tek desen → 0; tüm desenler "
            "eşit olasılıkta → 1. Yalnız geçmiş pencere (look-ahead yok)."
        ),
        default_period=8,
        rtol=1e-3,
        atol=1e-3,
        monotonic={"period": "periyot artarsa daha pürüzsüz (daha kararlı tahmin)"},
        # order=3 + period=8 → ilk tanımlı bar ~10. konumda; L3 yeterli pencere ister.
        min_points=16,
    ),
    "FORBIDDEN": ExamSpec(
        name="FORBIDDEN",
        definition=(
            "FORBIDDEN(p): yasak ordinal desen oranı (gömme order=3). Son p bar üzerinde "
            "ardışık 3'lü kapanış pencerelerinin SIRALAMA (ordinal) desenleri sayılır; olası "
            "3!=6 desenden pencerede HİÇ görülmeyenlerin oranı = 1 - (görülen farklı desen)/6. "
            "Aralık [0,1]; monoton seri → tek desen → 5/6≈0.833; tüm desenler görülürse → 0. "
            "Yalnız geçmiş pencere (look-ahead yok)."
        ),
        default_period=8,
        rtol=1e-3,
        atol=1e-3,
        monotonic={"period": "periyot artarsa daha çok desen görülür → yasak oranı azalır"},
        min_points=16,
    ),
    "COMPLEXITY": ExamSpec(
        name="COMPLEXITY",
        definition=(
            "COMPLEXITY(p): Jensen-Shannon istatistiksel karmaşıklık (MPR), gömme order=3, "
            "[0,1]. Son p bar üzerinde 3'lü ordinal desen dağılımı P; C = Q0·JS(P,U)·H_norm(P) "
            "— H_norm normalize permütasyon entropisi, U düzgün dağılım (1/6), JS Jensen-Shannon "
            "ıraksaması, Q0 max-1 normalizasyonu. Tam düzen (H=0) ve tam rastgelelik (JS=0) → 0; "
            "yapılı rejimde tepe. Yalnız geçmiş pencere (look-ahead yok)."
        ),
        default_period=12,
        rtol=1e-3,
        atol=1e-3,
        # değer-periyot ilişkisi tek-yön monoton DEĞİL → L4 monotonik iddiası yok (Kural 2).
        min_points=20,
    ),
}


def get_spec(name: str) -> ExamSpec:
    key = name.upper()
    if key not in _SPECS:
        raise KeyError(f"Sınav tanımı yok: {name!r} (mevcut: {sorted(_SPECS)})")
    return _SPECS[key]


def list_specs() -> list[ExamSpec]:
    return list(_SPECS.values())
