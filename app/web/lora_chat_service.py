"""Web için tembel-yüklemeli PEFT adapter sohbet servisi.

Eğitilen LoRA adapter'ı (veya base) transformers/PEFT ile YEREL yükler, chat-template
uygular ve cevap üretir — Ollama gerektirmez. Model bir kez yüklenir, bellekte tutulur
(yeniden yükleme maliyeti CPU'da yüksek). Üretim greedy/deterministtir (Kural 6).

`adapter_eval._load_model`/`_generate`/`_resolve_base_model` tekrar kullanılır (eğitim
doğrulamasıyla AYNI yükleme/üretim yolu → tutarlılık). Üretim ağırdır (CPU'da dakikalar);
endpoint senkron `def` olmalı ki FastAPI'nin threadpool'unda koşsun, event loop'u bloklamasın.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

log = logging.getLogger(__name__)

# Tek-örnek model önbelleği: aynı (base, adapter) tekrar istenirse yeniden yüklenmez.
_CACHE: dict[str, Any] = {"key": None, "tok": None, "model": None}
_LOCK = Lock()  # üretimi serileştir (tek kullanıcı; eşzamanlı model erişimini önle)


def list_adapters() -> list[str]:
    """models/adapters altındaki TAM adapter'ları (config + ağırlık var) listele."""
    from app.config import get_settings

    d = get_settings().adapters_dir
    out: list[str] = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*")):
        if (p / "adapter_config.json").exists() and (p / "adapter_model.safetensors").exists():
            out.append(p.name)
    return out


def chat(question: str, adapter: str | None, *, max_tokens: int = 256) -> dict:
    """Adapter (veya base) ile cevap üret. adapter=None/"" → yalnız base model.

    Dönüş: {answer, adapter, base_model}. Adapter yoksa FileNotFoundError.
    """
    from app.config import get_settings
    from app.training.adapter_eval import _generate, _load_model, _resolve_base_model

    s = get_settings()
    adapter_dir: str | None = None
    if adapter:
        ap = s.adapters_dir / adapter
        if not (ap / "adapter_config.json").exists():
            raise FileNotFoundError(f"Adapter bulunamadı: {adapter}")
        adapter_dir = str(ap)

    # Base önceliği: adapter'ın kendi config'i → settings (küçük-model 4B'ye yüklenmesin).
    base = (_resolve_base_model(adapter_dir) if adapter_dir else None) or s.peft_base_model
    key = f"{base}|{adapter_dir or ''}"

    with _LOCK:
        if _CACHE["key"] != key:
            log.info("lora-chat: model yükleniyor (key=%s) — ilk istek yavaş.", key)
            tok, model = _load_model(base, adapter_dir)
            _CACHE.update(key=key, tok=tok, model=model)
        answer = _generate(_CACHE["tok"], _CACHE["model"], question, max_new_tokens=max_tokens)

    return {"answer": answer, "adapter": adapter or "(base)", "base_model": base}
