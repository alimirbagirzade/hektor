"""RLM tool allowlist yaptırımı (talimat §12).

RLM motoruna serbest Python/exec verilmez. Yalnız allowlist'teki adlarda kayıtlı GÜVENLİ
wrapper'lar çağrılabilir. Allowlist DIŞINDA bir ad kaydedilemez ve çağrılamaz
(`ToolNotAllowed`). Her çağrı try/except ile sarılır → çağıran asla ham exception ile
çökmesin (structured sonuç döner).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# İzinli tool adları — RLM bunların DIŞINDA hiçbir şey çağıramaz (deny-by-default).
# Tek gerçek-kaynak burasıdır; başka modül kendi listesini tanımlamaz.
ALLOWED_TOOL_NAMES: tuple[str, ...] = (
    "rag_search",
    "get_paper_chunks",
    "get_paper_metadata",
    "citation_check",
    "grounding_check",
    "contradiction_check",
    "formula_check",
    "calculator",
)


class ToolNotAllowed(RuntimeError):
    """Allowlist dışında bir tool kaydı/çağrısı denendi."""


class SafeToolRegistry:
    """Yalnız allowlist'teki güvenli tool wrapper'larını tutan ve çağıran kayıt defteri."""

    def __init__(self, allowed: tuple[str, ...] = ALLOWED_TOOL_NAMES) -> None:
        self._allowed = set(allowed)
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Bir wrapper kaydet. Ad allowlist'te DEĞİLSE reddet (sahte-tool sızması yok)."""
        if name not in self._allowed:
            raise ToolNotAllowed(f"Tool allowlist dışında, kaydedilemez: {name!r}")
        self._tools[name] = fn

    def available(self) -> list[str]:
        return sorted(self._tools)

    def is_allowed(self, name: str) -> bool:
        return name in self._allowed

    def call(self, name: str, /, **kwargs: Any) -> dict[str, Any]:
        """Kayıtlı bir tool'u çağır. Bilinmeyen/izinsiz ad → ToolNotAllowed.
        Wrapper exception atarsa structured hata döner (ham exception sızmaz)."""
        if name not in self._allowed:
            raise ToolNotAllowed(f"İzin verilmeyen tool çağrısı: {name!r}")
        fn = self._tools.get(name)
        if fn is None:
            raise ToolNotAllowed(
                f"Tool allowlist'te ama bu kayıt defterinde kayıtlı değil: {name!r}"
            )
        try:
            return {"ok": True, "result": fn(**kwargs)}
        except Exception as exc:  # ham exception sızdırma → structured hata
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def runtime_info() -> dict[str, Any]:
    """RLM çalışma-zamanının GÜVENLİ (sır içermeyen) özeti — CLI/API görünümü.

    Motor tektir ve yereldir: `RlmController` + Ollama. Dış motor/sağlayıcı seçimi
    yoktur; bu yüzden burada anahtar, token veya uzak uç bilgisi de yoktur.
    """
    from app.config import get_settings

    s = get_settings()
    return {
        "engine": "native",  # RlmController (yerel); başka motor yok
        "llm_backend": "ollama",
        "llm_model": s.llm_model,
        "production_mode": s.rlm_production_mode,
        "allow_local_exec": False,  # kod yürütme YOK (Kural 5) — yapılandırılabilir değil
        "allowed_tools": list(ALLOWED_TOOL_NAMES),
        "seed": s.rlm_seed,
    }
