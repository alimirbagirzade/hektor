"""model_identity.py — sohbet cevabını veren modelin kimliği + bellek ayak izi ölçümü.

Kimlik üç parçadır ve hiçbiri tahmin edilmez:
- ``tag``: Ollama etiketi (``Settings.effective_chat_model``; istek başında sabitlenir).
- ``digest``: Ollama'nın bildirdiği içerik özeti (``/api/tags`` cevap öncesi, ``/api/ps``
  cevap sonrası). Özet okunamazsa ``""`` ve açık bir not döner.
- ``origin``: ``scripts/adapter_to_ollama.ps1``'in yazdığı ``models/gguf/Modelfile.<tag>``
  başlığından adapter adı. Dosya yoksa ``None`` → arayüz "köken kaydı yok" yazar. Bu dosyanın
  Ollama'daki özetle eşleştiği DOĞRULANAMAZ; bu yüzden ``verified=False`` taşınır.

Ayak izi: cevap sonrası ``/api/ps`` satırından (toplam boyut, VRAM'deki kısım, bağlam uzunluğu)
``storage/chat_model_footprint.json``'a yazılır. Kaynak koruması bu ÖLÇÜMÜ kullanır.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings

_GB = 1024**3
_MODELFILE_HEAD = re.compile(r"^#\s*(\S+)\s+-\s+(\S+)\s+(.*?)\s+GGUF\s+(\S+?);")


def _get_json(url: str, transport: httpx.BaseTransport | None, timeout: float = 5.0) -> Any:
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            r = client.get(url)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def ollama_tags(transport: httpx.BaseTransport | None = None) -> list[dict[str, Any]] | None:
    data = _get_json(f"{get_settings().ollama_host.rstrip('/')}/api/tags", transport)
    if not isinstance(data, dict):
        return None
    models = data.get("models")
    return list(models) if isinstance(models, list) else []


def ollama_ps(transport: httpx.BaseTransport | None = None) -> list[dict[str, Any]] | None:
    data = _get_json(f"{get_settings().ollama_host.rstrip('/')}/api/ps", transport)
    if not isinstance(data, dict):
        return None
    models = data.get("models")
    return list(models) if isinstance(models, list) else []


def match_entry(entries: list[dict[str, Any]] | None, tag: str) -> dict[str, Any] | None:
    """Etiket eşleşmesi (``x`` ≡ ``x:latest``)."""
    if not entries:
        return None
    want = {tag, f"{tag}:latest"} if ":" not in tag else {tag}
    for e in entries:
        if str(e.get("name") or e.get("model") or "") in want:
            return e
    return None


def model_origin(tag: str, root: Path | None = None) -> dict[str, Any] | None:
    """``models/gguf/Modelfile.<tag>`` başlığından köken (yoksa None — tahmin yok)."""
    base = (root or get_settings().root) / "models" / "gguf"
    name = tag[: -len(":latest")] if tag.endswith(":latest") else tag
    path = base / f"Modelfile.{name}"
    try:
        head = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    m = _MODELFILE_HEAD.match(head)
    if not m:
        return None
    return {
        "adapter": m.group(2),
        "kind": m.group(3),
        "quant": m.group(4),
        "record": f"models/gguf/Modelfile.{name}",
        "verified": False,
        "note": "Köken kaydı Modelfile başlığından okundu; Ollama digest'iyle eşleşmesi "
        "doğrulanmadı.",
    }


def describe_chat_model(
    transport: httpx.BaseTransport | None = None, slot: str = "main"
) -> dict[str, Any]:
    """Arayüz üst şeridi için yuvanın (ana/deneme) model kimliği (yazma yok)."""
    from app.feedback.model_activation import describe_slot

    s = get_settings()
    info = describe_slot(slot)
    tag = str(info.get("tag") or "")
    tags = ollama_tags(transport)
    entry = match_entry(tags, tag) if tag else None
    if not info.get("virtual", True):
        setting_source = "etkinleştirme kaydı"
    else:
        setting_source = (
            "HEKTOR_CHAT_MODEL" if s.chat_model.strip() else "HEKTOR_LLM_MODEL (chat_model boş)"
        )
    return {
        "tag": tag,
        "slot": slot,
        "slot_info": info,
        "setting_source": setting_source,
        "ollama_reachable": tags is not None,
        "installed": None if tags is None else entry is not None,
        "digest": str((entry or {}).get("digest") or ""),
        "origin": model_origin(tag) if tag else None,
        "footprint": read_footprint(tag) if tag else None,
    }


# ── ayak izi ölçümü ──────────────────────────────────────────────────────────


def footprint_path(root: Path | None = None) -> Path:
    return (root or get_settings().root) / "storage" / "chat_model_footprint.json"


def _read_all(root: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads(footprint_path(root).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_footprint(tag: str, root: Path | None = None) -> dict[str, Any] | None:
    fp = _read_all(root).get(tag)
    return fp if isinstance(fp, dict) else None


def footprint_from_ps(entry: dict[str, Any]) -> dict[str, Any]:
    size = float(entry.get("size") or 0)
    vram = float(entry.get("size_vram") or 0)
    ctx = entry.get("context_length")
    return {
        "digest": str(entry.get("digest") or ""),
        "size_gb": round(size / _GB, 2),
        "vram_gb": round(vram / _GB, 2),
        "ram_gb": round(max(0.0, size - vram) / _GB, 2),
        "context_length": int(ctx) if isinstance(ctx, int | float) and ctx else None,
    }


def record_footprint(
    tag: str, entry: dict[str, Any], *, measured_at: str, root: Path | None = None
) -> dict[str, Any]:
    """``/api/ps`` satırından ölçümü kaydet (etiket başına son ölçüm)."""
    fp = {**footprint_from_ps(entry), "measured_at": measured_at}
    data = _read_all(root)
    data[tag] = fp
    path = footprint_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return fp
