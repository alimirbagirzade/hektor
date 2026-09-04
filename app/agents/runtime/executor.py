"""Task executor (Phase 2.5) — hibrit ince yürütücü.

``task_queue``'daki PENDING görevleri DRENAJ eder: her görevi
``supervisor.run_with_supervision`` ile çalıştırır (STOP_ALL + tek-kullanımlık
taze-onay kapıları ORADA uygulanır), sonucu ``complete`` / ``fail`` / ``blocked``
olarak işler.

Bu bir DAG / cron motoru DEĞİLDİR — yalnız "kuyruk → supervisor → handler"
köprüsüdür (tasarım kararı: tam DAG aşırı mühendislik; zincir sırası manifest
reads/writes'tan deterministik türetilir). Tehlikeli zincir (gerçek eğitim,
adapter terfisi) zaten supervisor + taze-onay kapısından geçer; **bu modül o
kapıyı ZAYIFLATMAZ** (CLAUDE.md Kural 8).

GÜVENLİK: yalnız AÇIKÇA KAYITLI handler'ı olan ``agent_id`` çalıştırılır.
Bilinmeyen ``agent_id`` sessizce çalışmaz → ``fail_task("handler yok")``. Böylece
kuyruğa düşen rastgele bir ``agent_id`` otomatik kod çalıştıramaz.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.agents.runtime import supervisor, task_queue
from app.agents.runtime.schemas import AutomationTask, TaskStatus

if TYPE_CHECKING:
    from pathlib import Path

    from app.memory.sqlite_store import SqliteStore

log = logging.getLogger(__name__)

# Görev handler imzası: (task) -> sonuç. Sonuç ``{"ok": False}`` ise başarısız sayılır.
TaskHandler = Callable[[AutomationTask], Any]

# agent_id → handler. Bilinçli olarak BOŞ başlar: yalnız açıkça kayıtlı, güvenli
# işler executor üzerinden çalışır (allow-list, deny-list değil).
_HANDLERS: dict[str, TaskHandler] = {}

_BLOCKED = {TaskStatus.blocked_approval.value, TaskStatus.blocked_stop_all.value}


def register_handler(agent_id: str, handler: TaskHandler, *, replace: bool = False) -> None:
    """Bir ``agent_id`` için görev handler'ı kaydet (allow-list'e ekle)."""
    if agent_id in _HANDLERS and not replace:
        raise ValueError(f"Handler zaten kayıtlı: {agent_id} (değiştirmek için replace=True)")
    _HANDLERS[agent_id] = handler


def unregister_handler(agent_id: str) -> None:
    """Bir handler'ı kaldır (yoksa no-op)."""
    _HANDLERS.pop(agent_id, None)


def registered_agents() -> list[str]:
    """Executor üzerinden çalıştırılabilen kayıtlı ``agent_id``'ler."""
    return sorted(_HANDLERS)


def _action_for(task: AutomationTask) -> str:
    """Görev için supervisor/onay aksiyon adını türet (params.action veya 'run')."""
    if isinstance(task.params, dict):
        a = task.params.get("action")
        if isinstance(a, str) and a:
            return a
    return "run"


def run_task(
    task: AutomationTask, store: SqliteStore | None = None, root: str | Path | None = None
) -> dict[str, Any]:
    """Tek bir görevi yürüt: claim → supervise(handler) → complete / fail / blocked.

    Asla fırlatmaz; her zaman sonucu özetleyen bir sözlük döner:
      ``{"ok": True, ...}`` | ``{"ok": False, "blocked": True, ...}`` | ``{"ok": False, ...}``

    Bu sarmalayıcı "asla fırlatmaz" sözleşmesini GARANTİ eder: claim/fail/complete
    sırasında beklenmedik bir DB hatası olsa bile görev (mümkünse) failed işaretlenir
    ve hata sözlüğü döner — görev 'claimed' durumda öksüz kalmaz.
    """
    try:
        return _run_task(task, store=store, root=root)
    except Exception as exc:  # beklenmedik (claim/fail/complete DB) hatası → öksüz bırakma
        log.error("run_task kritik hata: %s", task.task_id, exc_info=True)
        with contextlib.suppress(Exception):
            task_queue.fail_task(task.task_id, f"sistem hatası: {exc!r}", store=store)
        return {"ok": False, "task_id": task.task_id, "reason": f"sistem hatası: {exc!r}"}


def _run_task(
    task: AutomationTask, store: SqliteStore | None = None, root: str | Path | None = None
) -> dict[str, Any]:
    """``run_task`` iç gövdesi. Beklenen yollarda fırlatmaz; beklenmedik DB hataları
    dış sarmalayıcı (``run_task``) tarafından yakalanır."""
    handler = _HANDLERS.get(task.agent_id)
    if handler is None:
        # Bilinmeyen agent_id ASLA sessizce çalışmaz (güvenlik allow-list'i).
        task_queue.fail_task(task.task_id, f"handler yok: {task.agent_id}", store=store)
        return {
            "ok": False,
            "task_id": task.task_id,
            "agent_id": task.agent_id,
            "reason": "handler yok",
        }

    # ATOMİK claim (CAS): yalnız pending'i bu çağrı claim ederse devam; aksi halde
    # (yok / pending değil / eşzamanlı başka işçi aldı) None → çift-çalıştırma önlenir.
    claimed = task_queue.try_claim_task(task.task_id, store=store)
    if claimed is None:
        return {
            "ok": False,
            "task_id": task.task_id,
            "reason": "claim edilemedi (yok / pending değil / başka işçi aldı)",
        }

    action = _action_for(task)

    def _call() -> Any:
        return handler(task)

    try:
        result = supervisor.run_with_supervision(
            task.agent_id,
            _call,
            action=action,
            task_id=task.task_id,
            params=task.params,
            store=store,
            root=root,
        )
    except Exception as exc:  # handler patladı → görevi başarısız işaretle, döngü sürer
        task_queue.fail_task(task.task_id, f"handler hata: {exc!r}", store=store)
        log.warning("Görev handler hatası: %s", task.task_id, exc_info=True)
        return {"ok": False, "task_id": task.task_id, "reason": repr(exc)}

    # supervisor engelledi (STOP_ALL / taze onay yok) → görev zaten blocked işaretlendi.
    if isinstance(result, dict) and result.get("blocked"):
        return {
            "ok": False,
            "task_id": task.task_id,
            "blocked": True,
            "blocked_by": result.get("blocked_by"),
            "reason": result.get("reason"),
        }

    # handler kendi içinde başarısızlığı ``{"ok": False}`` ile bildirdiyse failed say.
    if isinstance(result, dict) and result.get("ok") is False:
        reason = result.get("reason") or result.get("error") or "ok=False"
        task_queue.fail_task(task.task_id, str(reason), store=store)
        return {"ok": False, "task_id": task.task_id, "reason": str(reason)}

    task_queue.complete_task(task.task_id, summary=action, store=store)
    return {"ok": True, "task_id": task.task_id, "agent_id": task.agent_id}


def run_pending(
    limit: int = 10,
    retry_blocked: bool = False,
    store: SqliteStore | None = None,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Bekleyen görevleri sırayla yürüt. Her görev izole: biri patlarsa diğerleri sürer.

    ``retry_blocked=True`` ise önce ``blocked_*`` görevler yeniden kuyruğa alınır
    (ör. onay verildikten sonra yeniden denemek için).
    """
    if retry_blocked:
        for status in (TaskStatus.blocked_approval.value, TaskStatus.blocked_stop_all.value):
            for t in task_queue.list_tasks(status=status, limit=limit, store=store):
                task_queue.requeue_task(t.task_id, store=store)

    tasks = task_queue.list_tasks(status=TaskStatus.pending.value, limit=limit, store=store)
    results: list[dict[str, Any]] = []
    for t in tasks:
        try:
            results.append(run_task(t, store=store, root=root))
        except Exception as exc:  # savunmacı: run_task zaten yutuyor, yine de döngüyü koru
            log.warning("run_task beklenmedik hata: %s", t.task_id, exc_info=True)
            results.append({"ok": False, "task_id": t.task_id, "reason": repr(exc)})
    return results
