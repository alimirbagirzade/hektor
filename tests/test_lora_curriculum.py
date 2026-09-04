"""Curriculum seviyelendirme testleri."""

from __future__ import annotations

import pytest

from app.lora.curriculum import (
    CurriculumLevel,
    classify_curriculum,
    is_curriculum_valid,
)


@pytest.mark.parametrize(
    ("difficulty", "expected"),
    [
        (0.0, CurriculumLevel.LEVEL_0),
        (0.1, CurriculumLevel.LEVEL_0),
        (0.2, CurriculumLevel.LEVEL_1),
        (0.35, CurriculumLevel.LEVEL_1),
        (0.4, CurriculumLevel.LEVEL_2),
        (0.55, CurriculumLevel.LEVEL_2),
        (0.6, CurriculumLevel.LEVEL_3),
        (0.79, CurriculumLevel.LEVEL_3),
        (0.8, CurriculumLevel.LEVEL_4),
        (1.0, CurriculumLevel.LEVEL_4),
    ],
)
def test_classify_curriculum_maps_difficulty_to_level(
    difficulty: float, expected: CurriculumLevel
) -> None:
    """Her zorluk değeri doğru seviyeye eşlenmeli."""
    assert classify_curriculum(difficulty) is expected


def test_classify_curriculum_clamps_below_zero() -> None:
    """Negatif zorluk en alt seviyeye sabitlenmeli."""
    assert classify_curriculum(-0.5) is CurriculumLevel.LEVEL_0


def test_classify_curriculum_clamps_above_one() -> None:
    """1.0 üstü zorluk en üst seviyeye sabitlenmeli."""
    assert classify_curriculum(1.5) is CurriculumLevel.LEVEL_4


def test_is_curriculum_valid_accepts_in_range() -> None:
    """Seviye aralığındaki zorluk geçerli olmalı."""
    assert is_curriculum_valid(CurriculumLevel.LEVEL_2, 0.5) is True


def test_is_curriculum_valid_rejects_out_of_global_bounds() -> None:
    """0-1 dışındaki zorluk her seviye için geçersiz olmalı."""
    assert is_curriculum_valid(CurriculumLevel.LEVEL_4, 1.5) is False
    assert is_curriculum_valid(CurriculumLevel.LEVEL_0, -0.1) is False


def test_is_curriculum_valid_rejects_wrong_level() -> None:
    """Zorluk başka bir seviyenin aralığındaysa geçersiz olmalı."""
    assert is_curriculum_valid(CurriculumLevel.LEVEL_0, 0.5) is False


def test_level_4_includes_upper_bound() -> None:
    """En üst seviye 1.0'ı kapsamalı (kapalı aralık)."""
    assert is_curriculum_valid(CurriculumLevel.LEVEL_4, 1.0) is True


def test_bounds_property_returns_tuple() -> None:
    """bounds property doğru (alt, üst) ikilisini döndürmeli."""
    assert CurriculumLevel.LEVEL_1.bounds == (0.2, 0.4)
