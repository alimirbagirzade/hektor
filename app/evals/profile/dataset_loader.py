"""Profil eval veri seti yükleyici — hash korumalı validation / golden_test.

Veri ``evals/profile_mix/<split>.jsonl`` altında (git'te izlenir; ``data/`` commit edilmez
kuralı nedeniyle ``data/eval/golden`` yerine). ``manifest.json`` her split'in sha256'sını
tutar; dosya değişirse yükleme REDDEDİLİR (sessiz golden değişikliği = geçersiz karşılaştırma).

Golden koruması:
- ``golden_test`` yalnız ``purpose="final_comparison"`` ile yüklenir; ağırlık/profil
  seçimi (``purpose="selection"``) golden'ı OKUYAMAZ → eval setine overfit engellenir.
- Eğitim tarafı bu modülü kullanmaz; ``assert_not_eval_path`` eğitim veri yollarında
  eval dizinini reddeder (bkz. ``train --run`` sızıntı kapısı).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.evals.profile.schema import EvalItem, Split
from app.lora.mix_common import hash_file, repo_root

PURPOSES: frozenset[str] = frozenset({"selection", "final_comparison", "audit", "leakage_check"})


class GoldenAccessError(PermissionError):
    """Golden test seti izin verilmeyen amaçla istendi."""


class DatasetIntegrityError(ValueError):
    """Split dosyası manifest hash'iyle uyuşmuyor ya da şema dışı."""


def eval_root() -> Path:
    return repo_root() / "evals" / "profile_mix"


def split_path(split: Split, root: Path | None = None) -> Path:
    return (root or eval_root()) / f"{split.value}.jsonl"


def load_manifest(root: Path | None = None) -> dict[str, dict[str, str]]:
    p = (root or eval_root()) / "manifest.json"
    if not p.exists():
        raise DatasetIntegrityError(f"manifest yok: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return dict(data.get("splits") or {})


def dataset_version(split: Split, root: Path | None = None) -> str:
    """'<split>@<sha256[:12]>' — eval_dataset_version olarak kaydedilir."""
    return f"{split.value}@{hash_file(split_path(split, root))[:12]}"


def write_manifest(root: Path | None = None, *, version: str = "1") -> Path:
    """Split hash'lerini yeniden yaz (yalnız bilinçli veri güncellemesinde)."""
    base = root or eval_root()
    splits = {}
    for s in Split:
        p = split_path(s, base)
        if p.exists():
            splits[s.value] = {"file": p.name, "sha256": hash_file(p)}
    out = base / "manifest.json"
    out.write_text(
        json.dumps({"version": version, "splits": splits}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def load_split(
    split: Split, *, purpose: str, root: Path | None = None, verify_hash: bool = True
) -> list[EvalItem]:
    if purpose not in PURPOSES:
        raise ValueError(f"bilinmeyen amaç: {purpose} (geçerli: {sorted(PURPOSES)})")
    if split is Split.GOLDEN_TEST and purpose == "selection":
        raise GoldenAccessError(
            "golden_test profil/ağırlık SEÇİMİNDE kullanılamaz — validation split'ini kullan; "
            "golden yalnız profil seçildikten sonra nihai karşılaştırma içindir"
        )
    path = split_path(split, root)
    if not path.exists():
        raise DatasetIntegrityError(f"split dosyası yok: {path}")
    if verify_hash:
        manifest = load_manifest(root)
        expected = (manifest.get(split.value) or {}).get("sha256")
        if not expected:
            raise DatasetIntegrityError(f"manifest'te {split.value} hash'i yok")
        actual = hash_file(path)
        if actual != expected:
            raise DatasetIntegrityError(
                f"{path.name} manifest hash'iyle uyuşmuyor ({actual[:12]} ≠ {expected[:12]}); "
                "golden/validation değişikliği bilinçliyse manifest'i yeniden yaz"
            )
    items: list[EvalItem] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = EvalItem.model_validate_json(line)
        except ValueError as exc:
            raise DatasetIntegrityError(f"{path.name}:{n} şema hatası: {exc}") from exc
        if item.id in seen:
            raise DatasetIntegrityError(f"{path.name}:{n} yinelenen id: {item.id}")
        seen.add(item.id)
        items.append(item)
    return items


def assert_not_eval_path(path: Path) -> None:
    """Eğitim veri yolu eval dizininin içindeyse reddet (golden eğitime giremez)."""
    try:
        path.resolve().relative_to(eval_root().resolve())
    except ValueError:
        return
    raise GoldenAccessError(f"eval verisi eğitim girdisi olarak kullanılamaz: {path}")
