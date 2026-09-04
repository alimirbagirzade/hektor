"""Curriculum seviyelendirme — zorluk derecesine göre eğitim aşaması.

Müfredat, modelin önce temel kavramları sonra ileri sentezi öğrenmesini
sağlar. Her kart 0.0-1.0 aralığında bir `difficulty` taşır; bu modül onu
beş ayrık seviyeye eşler.
"""

from __future__ import annotations

from enum import StrEnum

# Her seviyenin kapsadığı zorluk aralığı [alt, üst).
# En üst seviye 1.0'ı da kapsar (kapalı aralık).
LEVEL_BOUNDS: dict[str, tuple[float, float]] = {
    "LEVEL_0": (0.0, 0.2),
    "LEVEL_1": (0.2, 0.4),
    "LEVEL_2": (0.4, 0.6),
    "LEVEL_3": (0.6, 0.8),
    "LEVEL_4": (0.8, 1.0),
}

DIFFICULTY_MIN = 0.0
DIFFICULTY_MAX = 1.0


class CurriculumLevel(StrEnum):
    """Eğitim müfredatı seviyeleri (kolaydan zora)."""

    LEVEL_0 = "LEVEL_0"  # 0.0-0.2 — temel kavram ("bu nedir?")
    LEVEL_1 = "LEVEL_1"  # 0.2-0.4 — tanım + formül
    LEVEL_2 = "LEVEL_2"  # 0.4-0.6 — uygulama + yorum
    LEVEL_3 = "LEVEL_3"  # 0.6-0.8 — kombinasyon + strateji
    LEVEL_4 = "LEVEL_4"  # 0.8-1.0 — araştırma + sentez

    @property
    def bounds(self) -> tuple[float, float]:
        """Bu seviyenin (alt, üst) zorluk sınırlarını döndür."""
        return LEVEL_BOUNDS[self.value]


def classify_curriculum(difficulty: float) -> CurriculumLevel:
    """Zorluk derecesine göre müfredat seviyesini belirle (deterministik).

    Sınır dışı değerler en yakın geçerli seviyeye sabitlenir.
    """
    clamped = max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, difficulty))

    if clamped < 0.2:
        return CurriculumLevel.LEVEL_0
    if clamped < 0.4:
        return CurriculumLevel.LEVEL_1
    if clamped < 0.6:
        return CurriculumLevel.LEVEL_2
    if clamped < 0.8:
        return CurriculumLevel.LEVEL_3
    return CurriculumLevel.LEVEL_4


def is_curriculum_valid(level: CurriculumLevel, difficulty: float) -> bool:
    """Verilen zorluğun seviye aralığına ve genel [0,1] sınırına uyup uymadığını kontrol et."""
    if not (DIFFICULTY_MIN <= difficulty <= DIFFICULTY_MAX):
        return False
    low, high = level.bounds
    # En üst seviye üst sınırı da kapsar; diğerleri yarı-açık aralık.
    if level is CurriculumLevel.LEVEL_4:
        return low <= difficulty <= high
    return low <= difficulty < high
