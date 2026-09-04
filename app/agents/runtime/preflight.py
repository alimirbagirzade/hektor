"""Runtime ön-uçuş (Phase 2.5) — taze makine için ajan-runtime hazırlık doğrulaması.

``achilles init`` zaten ``SqliteStore`` kurar ve TÜM tabloları (agent_runs /
agent_events / automation_tasks / approval_requests) ``Base.metadata.create_all``
ile idempotent oluşturur. Bu modül bunu DOĞRULAR: manifest geçerli mi, runtime
tabloları sorgulanabilir mi, STOP_ALL aktif mi.

Amaç: taze/başka makinede ilk agent komutundan ÖNCE "tablo yok / manifest bozuk"
sürprizini yakalamak. Hiçbir tehlikeli iş yapmaz; idempotent (tablolar yoksa
``SqliteStore`` örneği zaten oluşturur) + salt-doğrulama.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.memory.sqlite_store import SqliteStore


def runtime_preflight(store: SqliteStore | None = None) -> dict[str, Any]:
    """Ajan-runtime hazır mı? Sonuç sözlüğü döner (asla fırlatmaz).

    Dönüş: ``{ok, agents, dangerous, approval_required, tables, stop_all, errors}``.
    ``ok=False`` → çağıran (CLI) sıfır-dışı çıkışla taze-makine kurulumunu durdurabilir.
    """
    errors: list[str] = []

    # 1) SqliteStore örneği → şema (Base.metadata.create_all, idempotent).
    if store is None:
        from app.memory.sqlite_store import SqliteStore as _S

        store = _S(check_same_thread=False)

    # 2) Tabloları sorgula (varlık + erişilebilirlik kanıtı). Hepsi aynı create_all'dan
    #    gelir; list_ metodu olanları doğrulamak şemanın kurulu olduğunu kanıtlar.
    tables: dict[str, bool] = {}
    checks: dict[str, Any] = {
        "agent_runs": lambda: store.list_agent_runs(limit=1),
        "automation_tasks": lambda: store.list_automation_tasks(limit=1),
        "approval_requests": lambda: store.list_approval_requests(limit=1),
    }
    for name, fn in checks.items():
        try:
            fn()
            tables[name] = True
        except Exception as exc:
            tables[name] = False
            errors.append(f"tablo erişilemiyor: {name} ({exc!r})")

    # 3) Manifest geçerli mi + ajan sayıları.
    agents_count = dangerous_count = approval_count = 0
    try:
        from app.agents.runtime.registry import (
            agents_requiring_approval,
            dangerous_agents,
            list_agents,
        )

        agents_count = len(list_agents())
        dangerous_count = len(dangerous_agents())
        approval_count = len(agents_requiring_approval())
        if agents_count == 0:
            errors.append("manifest boş (ajan yok)")
    except Exception as exc:
        errors.append(f"manifest geçersiz: {exc!r}")

    # 4) STOP_ALL kill-switch durumu (bilgi amaçlı; hata değil).
    stop_all = False
    try:
        from app.agents.runtime import supervisor

        stop_all = supervisor.is_stop_all_active()
    except Exception:
        stop_all = False

    return {
        "ok": not errors,
        "agents": agents_count,
        "dangerous": dangerous_count,
        "approval_required": approval_count,
        "tables": tables,
        "stop_all": stop_all,
        "errors": errors,
    }
