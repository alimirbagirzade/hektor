"""Executor handler kaydı — hangi ``agent_id``'ler ``tasks-run`` ile çalıştırılabilir.

GÜVENLİK (CLAUDE.md Kural 8): yalnız **salt-okuma / rapor üreten** ajanlar buraya
kaydedilir. BİLİNÇLİ OLARAK KAYDEDİLMEYENLER:
  - ``auto-lora-pipeline`` (gerçek LoRA eğitimi / adapter terfisi) — tehlikeli;
    kendi supervisor + tek-kullanımlık taze-onay kapısından geçer, kuyruktan
    otomatik tetiklenmemeli.
  - ``arxiv-fetcher`` / ``rag-learning-loop`` — ağ + yazma yapan otonom ajanlar;
    kendi background_loop runner'ları ve enable bayrakları var.
Yeni handler eklemek = bilinçli, denetimli bir karar (yalnız yan etkisiz/okunur işler).

``register_default_handlers()`` idempotenttir ve mevcut (ör. test/kullanıcı) kayıtları
EZMEZ. ``tasks-run`` CLI komutu çağırmadan önce bunu çalıştırır.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.agents.runtime.schemas import AutomationTask


def _model_advisor_handler(task: AutomationTask) -> dict[str, Any]:
    """model-advisor: donanım profiline göre model önerir. SALT-OKUMA (collect + recommend)."""
    from app.agents.model_advisor.advisor import recommend
    from app.agents.system_profiler.profiler import collect

    params = task.params or {}
    profile = collect()
    result = recommend(
        profile,
        task=str(params.get("task", "general")),
        top_k=int(params.get("top_k", 3)),
    )
    return {
        "ok": True,
        "recommended": [r.model_id for r in result.recommended],
        "rejected": len(result.rejected),
        "system": result.system_summary,
    }


# agent_id → handler. YALNIZ güvenli, salt-okuma ajanlar (yukarıdaki güvenlik notu).
_DEFAULT_HANDLERS: dict[str, Any] = {
    "model-advisor": _model_advisor_handler,
}


def register_default_handlers() -> list[str]:
    """Güvenli varsayılan handler'ları executor'a kaydet (idempotent; mevcutları ezmez).

    Yeni kaydedilen ``agent_id``'lerin listesini döndürür (zaten kayıtlı olanlar atlanır).
    """
    from app.agents.runtime import executor

    already = set(executor.registered_agents())
    newly: list[str] = []
    for agent_id, fn in _DEFAULT_HANDLERS.items():
        if agent_id not in already:
            executor.register_handler(agent_id, fn)
            newly.append(agent_id)
    return newly
