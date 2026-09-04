"""Dataset bölücü — train/valid/test ayrımı ve sızıntı denetimi.

Gate 8 için kullanılır. Aynı `source_id` (kaynak makale) birden fazla
bölmede yer alamaz; aksi halde test sızıntısı (data leakage) oluşur.
Bölme deterministiktir (seed ile).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SEED = 42


@dataclass
class DatasetSplit:
    """Üçe ayrılmış veri seti."""

    train: list[Any] = field(default_factory=list)
    valid: list[Any] = field(default_factory=list)
    test: list[Any] = field(default_factory=list)


def _source_id(example: Any) -> str:
    """Bir örnekten source_id çıkar (LoRAExample.metadata veya dict)."""
    metadata = getattr(example, "metadata", None)
    if isinstance(metadata, dict):
        return str(metadata.get("source_id") or metadata.get("paper_id") or "")
    if isinstance(example, dict):
        meta = example.get("metadata", {})
        if isinstance(meta, dict) and (meta.get("source_id") or meta.get("paper_id")):
            return str(meta.get("source_id") or meta.get("paper_id"))
        return str(example.get("source_id") or example.get("paper_id") or "")
    return ""


def split_dataset(
    examples: list[Any],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> DatasetSplit:
    """Örnekleri train/valid/test olarak böl — kaynak bütünlüğünü koru.

    Aynı source_id'ye sahip tüm örnekler aynı bölmeye gider; böylece bir
    makaledeki örnekler hem train hem test'te bulunmaz. Bölme, kaynak
    grupları seviyesinde seed ile karıştırılarak yapılır.
    """
    if not examples:
        return DatasetSplit()

    groups: dict[str, list[Any]] = defaultdict(list)
    for ex in examples:
        groups[_source_id(ex)].append(ex)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = max(1, round(n_groups * train_ratio))
    n_valid = round(n_groups * valid_ratio)
    # Test en az bir grup alabilsin diye train/valid'i gerektiğinde kıs.
    if n_train + n_valid >= n_groups and n_groups >= 3:
        n_valid = max(1, n_valid)
        n_train = n_groups - n_valid - 1

    train_keys = group_keys[:n_train]
    valid_keys = group_keys[n_train : n_train + n_valid]
    test_keys = group_keys[n_train + n_valid :]

    split = DatasetSplit()
    for key in train_keys:
        split.train.extend(groups[key])
    for key in valid_keys:
        split.valid.extend(groups[key])
    for key in test_keys:
        split.test.extend(groups[key])
    return split


def check_leakage(split: DatasetSplit) -> list[str]:
    """Bölmeler arasında ortak source_id var mı kontrol et.

    Döndürülen liste her ihlali açıklar; boşsa sızıntı yoktur.
    """
    train_sources = {_source_id(ex) for ex in split.train}
    valid_sources = {_source_id(ex) for ex in split.valid}
    test_sources = {_source_id(ex) for ex in split.test}

    issues: list[str] = []
    for name_a, set_a, name_b, set_b in (
        ("train", train_sources, "valid", valid_sources),
        ("train", train_sources, "test", test_sources),
        ("valid", valid_sources, "test", test_sources),
    ):
        overlap = (set_a & set_b) - {""}
        for source in sorted(overlap):
            issues.append(f"sızıntı: '{source}' hem {name_a} hem {name_b} içinde")
    return issues
