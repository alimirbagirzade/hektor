"""Reciprocal Rank Fusion (RRF) birim testleri — determinizm, ağırlık, k, kenar durumlar."""

from __future__ import annotations

import pytest

from app.memory.rank_fusion import (
    DEFAULT_RRF_K,
    fuse_ranked,
    reciprocal_rank_fusion,
)


def test_single_list_preserves_order() -> None:
    """Tek liste → giriş sırası korunur."""
    assert fuse_ranked([["a", "b", "c"]]) == ["a", "b", "c"]


def test_agreement_across_lists_wins() -> None:
    """İki listede de üst sırada olan öğe en yükseğe çıkar."""
    out = fuse_ranked([["a", "b", "c"], ["a", "c", "b"]])
    assert out[0] == "a"  # her iki listede de 1. sıra → en yüksek RRF


def test_consensus_beats_single_top() -> None:
    """İki listede de görülen öğe, tek listede tepede olan öğeyi geçer (RRF temel özelliği)."""
    # "both" her iki listede (L1 rank1 + L2 rank0); "single" yalnız L1 rank0.
    # 1/62 + 1/61 > 1/61 → uzlaşan "both" kazanır.
    out = fuse_ranked([["single", "both"], ["both"]])
    assert out[0] == "both"


def test_scores_match_formula() -> None:
    """Skorlar w/(k+rank) formülüne uymalı (rank 1-tabanlı)."""
    scores = reciprocal_rank_fusion([["a", "b"]], k=10)
    assert scores["a"] == pytest.approx(1.0 / 11)
    assert scores["b"] == pytest.approx(1.0 / 12)


def test_weights_applied() -> None:
    """Ağırlık, ilgili listenin katkısını ölçekler."""
    # a yalnız ağırlıklı listede; b yalnız düşük ağırlıklıda → a kazanır.
    out = fuse_ranked([["a"], ["b"]], weights=[5.0, 1.0])
    assert out[0] == "a"


def test_deterministic_tie_break_by_id() -> None:
    """Eşit RRF skorunda id alfabetik sırayla kararlılaşır (deterministik)."""
    out = fuse_ranked([["b", "a"], ["a", "b"]])
    # a ve b simetrik (toplam skor eşit) → alfabetik: a önce.
    assert out == ["a", "b"]


def test_empty_input() -> None:
    """Boş giriş → boş çıktı."""
    assert fuse_ranked([]) == []
    assert fuse_ranked([[], []]) == []
    assert reciprocal_rank_fusion([]) == {}


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def test_weights_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_default_k_constant() -> None:
    assert DEFAULT_RRF_K == 60
