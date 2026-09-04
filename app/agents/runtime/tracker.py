"""Agent run tracker — koşu yaşam döngüsünü + olaylarını SQLite + JSONL'e yazar.

PHASE 1 = YALNIZ GÖZLEM. Bu modül ajan davranışını DEĞİŞTİRMEZ ve İSTİSNA
FIRLATMAZ: tracking sırasında bir hata olursa sessizce yutulur (debug log) ve
sarılan ajan normal akışına devam eder. Kural: gözlemci asla üretimi bozmaz.

Her koşu için:
  * SQLite ``agent_runs`` + ``agent_events`` tablolarına yazılır
  * ayrıca insan-okunur JSONL: ``reports/agent_runs/<run_id>.jsonl``

``run_id`` deterministik değildir ama benzersiz + okunur:
``arun_YYYYMMDD_HHMMSS_<8hex>``.

API:
  * ``RunTracker`` — start_run / log_event / finish_run (+ güvenli safe_* sarmalları)
  * ``get_tracker`` / ``set_tracker`` — global tekil (test enjeksiyonu için)
  * ``track_agent_run(...)`` — context manager (elle sarmalama)
  * ``tracked(agent_id, ...)`` — async-farkında dekoratör (hafif sarmalama)
  * ``log_step(...)`` — mevcut (context) koşuya bir ara-adım olayı ekle
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import functools
import inspect
import json
import logging
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from app.agents.runtime.schemas import AgentEvent, AgentEventKind, AgentRun, AgentRunStatus
from app.config import get_settings

if TYPE_CHECKING:
    from app.memory.sqlite_store import SqliteStore

log = logging.getLogger(__name__)

# agent_events retention (kullanıcı kararı): 30 gün VEYA en çok 50.000 son olay.
_EVENT_RETENTION_MAX = 50_000
_EVENT_RETENTION_DAYS = 30

# Mevcut koşu id'si (iç içe/asenkron bağlamlarda doğru taşınır).
_current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "achilles_current_agent_run", default=None
)


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _new_run_id() -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    return f"arun_{stamp}_{uuid.uuid4().hex[:8]}"


def _summarize(result: Any) -> dict[str, Any] | None:
    """Dönüş değerini küçük, JSON'lanabilir bir özete indir."""
    if result is None:
        return None
    if isinstance(result, dict):
        return {k: result[k] for k in list(result)[:12]}
    if isinstance(result, (list, tuple)):
        return {"result_type": type(result).__name__, "count": len(result)}
    return {"result_type": type(result).__name__}


def _status_for(result: Any) -> tuple[AgentRunStatus, str | None]:
    """Dönüş değerinden koşu durumunu türet.

    Bazı ajanlar (rag-learning-loop, auto-lora-pipeline) istisnayı YUTUP
    ``{"ok": False, "reason": ...}`` döner; bunu ``failed`` say.
    """
    if isinstance(result, dict) and result.get("ok") is False:
        reason = result.get("reason") or result.get("error")
        return AgentRunStatus.failed, (str(reason) if reason else "ok=False")
    return AgentRunStatus.completed, None


class RunTracker:
    """Ajan koşularını SQLite + JSONL'e yazan ince izleyici."""

    def __init__(
        self, store: SqliteStore | None = None, jsonl_dir: str | Path | None = None
    ) -> None:
        self._store = store
        self._jsonl_dir = Path(jsonl_dir) if jsonl_dir else None

    @property
    def store(self) -> SqliteStore:
        if self._store is None:
            from app.memory.sqlite_store import SqliteStore

            # check_same_thread=False: çek/araştırma ajanları asyncio.to_thread
            # içinde de yazabilir → tek tracker'ı farklı thread'lerden kullanmak güvenli.
            self._store = SqliteStore(check_same_thread=False)
        return self._store

    @property
    def jsonl_dir(self) -> Path:
        if self._jsonl_dir is None:
            self._jsonl_dir = get_settings().agent_runs_dir
        return self._jsonl_dir

    def _append_jsonl(self, run_id: str, record: dict[str, Any]) -> None:
        try:
            d = self.jsonl_dir
            d.mkdir(parents=True, exist_ok=True)
            with (d / f"{run_id}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            log.debug("agent run JSONL yazılamadı: %s", run_id, exc_info=True)

    # --- yaşam döngüsü --------------------------------------------------
    def start_run(
        self,
        agent_id: str,
        task_id: str | None = None,
        trigger_type: str = "manual",
        trigger_payload: dict[str, Any] | None = None,
    ) -> str:
        run_id = _new_run_id()
        run = AgentRun(
            run_id=run_id,
            agent_id=agent_id,
            task_id=task_id,
            status=AgentRunStatus.running,
            trigger_type=trigger_type,
            trigger_payload=trigger_payload,
            started_at=_utcnow(),
        )
        self.store.create_agent_run(
            run_id=run_id,
            agent_id=agent_id,
            task_id=task_id,
            status=run.status.value,
            trigger_type=trigger_type,
            trigger_payload=trigger_payload,
            started_at=run.started_at,
        )
        self._append_jsonl(run_id, {"_record": "run", **run.model_dump(mode="json")})
        self.log_event(run_id, AgentEventKind.start, f"{agent_id} başladı ({trigger_type})")
        return run_id

    def log_event(
        self,
        run_id: str | None,
        kind: AgentEventKind | str,
        message: str | None = None,
        level: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not run_id:
            return
        kind_val = kind.value if isinstance(kind, AgentEventKind) else str(kind)
        ev = AgentEvent(
            event_id=f"aev_{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            ts=_utcnow(),
            kind=AgentEventKind(kind_val),
            level=level,
            message=message,
            payload=payload,
        )
        self.store.add_agent_event(
            event_id=ev.event_id,
            run_id=run_id,
            ts=ev.ts,
            kind=kind_val,
            level=level,
            message=message,
            payload=payload,
        )
        self._append_jsonl(run_id, {"_record": "event", **ev.model_dump(mode="json")})

    def finish_run(
        self,
        run_id: str | None,
        status: AgentRunStatus | str = AgentRunStatus.completed,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
        outputs: list[Any] | None = None,
    ) -> None:
        if not run_id:
            return
        status_val = status.value if isinstance(status, AgentRunStatus) else str(status)
        finished = _utcnow()
        self.log_event(
            run_id,
            AgentEventKind.finish,
            f"bitti: {status_val}",
            level=("error" if status_val == "failed" else "info"),
        )
        self.store.finish_agent_run(
            run_id=run_id,
            status=status_val,
            finished_at=finished,
            error=error,
            summary=summary,
            outputs=outputs,
        )
        self._append_jsonl(
            run_id,
            {
                "_record": "run_final",
                "run_id": run_id,
                "status": status_val,
                "finished_at": finished,
                "error": error,
                "summary": summary,
                "outputs": outputs,
            },
        )
        try:
            self.store.prune_agent_events(
                max_events=_EVENT_RETENTION_MAX, max_age_days=_EVENT_RETENTION_DAYS
            )
        except Exception:
            log.debug("agent_events budama hatası (yok sayıldı)", exc_info=True)

    # --- güvenli sarmallar (asla fırlatmaz) -----------------------------
    def safe_start(
        self,
        agent_id: str,
        task_id: str | None = None,
        trigger_type: str = "manual",
        trigger_payload: dict[str, Any] | None = None,
    ) -> str | None:
        try:
            return self.start_run(agent_id, task_id, trigger_type, trigger_payload)
        except Exception:
            log.debug("tracker.start_run hatası (yok sayıldı)", exc_info=True)
            return None

    def safe_finish(
        self,
        run_id: str | None,
        status: AgentRunStatus | str = AgentRunStatus.completed,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
        outputs: list[Any] | None = None,
    ) -> None:
        try:
            self.finish_run(run_id, status, summary, error, outputs)
        except Exception:
            log.debug("tracker.finish_run hatası (yok sayıldı)", exc_info=True)


_tracker: RunTracker | None = None


def get_tracker() -> RunTracker:
    """Genel tekil tracker (lazy)."""
    global _tracker
    if _tracker is None:
        _tracker = RunTracker()
    return _tracker


def set_tracker(tracker: RunTracker | None) -> None:
    """Genel tracker'ı değiştir/temizle (test enjeksiyonu için)."""
    global _tracker
    _tracker = tracker


def log_step(message: str, level: str = "info", payload: dict[str, Any] | None = None) -> None:
    """Mevcut (context) koşuya bir ``step`` olayı ekle. Koşu yoksa no-op; asla fırlatmaz."""
    run_id = _current_run.get()
    if not run_id:
        return
    try:
        get_tracker().log_event(run_id, AgentEventKind.step, message, level=level, payload=payload)
    except Exception:
        log.debug("log_step hatası (yok sayıldı)", exc_info=True)


class _RunHandle:
    """``track_agent_run`` içinde kullanıcıya verilen tutamaç."""

    def __init__(self, run_id: str | None) -> None:
        self.run_id = run_id

    def log_event(
        self,
        kind: AgentEventKind | str,
        message: str | None = None,
        level: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.run_id:
            return
        try:
            get_tracker().log_event(self.run_id, kind, message, level=level, payload=payload)
        except Exception:
            log.debug("RunHandle.log_event hatası (yok sayıldı)", exc_info=True)


@contextmanager
def track_agent_run(
    agent_id: str,
    task_id: str | None = None,
    trigger_type: str = "manual",
    trigger_payload: dict[str, Any] | None = None,
) -> Iterator[_RunHandle]:
    """Bir kod bloğunu ajan koşusu olarak izle (elle sarmalama).

    İçeride istisna olursa koşu ``failed`` işaretlenir ve istisna yeniden fırlatılır.
    Tracking hataları asla bloğun kendisini bozmaz.
    """
    tr = get_tracker()
    run_id = tr.safe_start(
        agent_id, task_id=task_id, trigger_type=trigger_type, trigger_payload=trigger_payload
    )
    token = _current_run.set(run_id)
    try:
        yield _RunHandle(run_id)
    except Exception as exc:
        tr.safe_finish(run_id, status=AgentRunStatus.failed, error=repr(exc))
        raise
    else:
        tr.safe_finish(run_id, status=AgentRunStatus.completed)
    finally:
        _current_run.reset(token)


def log_system_event(
    message: str,
    agent_id: str = "system",
    level: str = "info",
    action: str | None = None,
    payload: dict[str, Any] | None = None,
    run_id: str = "system",
) -> None:
    """Bir koşuya bağlı OLMAYAN sistem olayı (STOP_ALL, approval, task) → run_id='system'.

    Genel olay akışında (``/api/events``) görünür. Asla fırlatmaz.
    """
    body: dict[str, Any] = {"agent_id": agent_id}
    if action:
        body["action"] = action
    if payload:
        body.update(payload)
    try:
        get_tracker().log_event(run_id, AgentEventKind.info, message, level=level, payload=body)
    except Exception:
        log.debug("log_system_event hatası (yok sayıldı)", exc_info=True)


def cancel_stale_running_agent_runs(
    reason: str = "startup_sweep",
    stale_after_hours: int = 6,
    store: SqliteStore | None = None,
) -> list[str]:
    """Crash sonrası 'running' kalan ESKİ koşuları 'cancelled' yap (startup sweep).

    ``stale_after_hours``'tan eski başlamış running koşular süpürülür (canlı eşzamanlı
    koşuları yanlışlıkla iptal etmemek için). Süpürülen run_id'leri döndürür. Asla fırlatmaz.
    """
    try:
        st = store if store is not None else get_tracker().store
        cutoff = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=stale_after_hours)).isoformat()
        run_ids = st.mark_running_agent_runs_cancelled(reason, cutoff_iso=cutoff)
        for rid in run_ids:
            with suppress(Exception):
                get_tracker().log_event(
                    rid, AgentEventKind.finish, f"{reason}: iptal (bayat running)", level="warning"
                )
        if run_ids:
            log.info("Startup sweep: %d bayat 'running' koşu iptal edildi", len(run_ids))
        return run_ids
    except Exception:
        log.debug("cancel_stale_running_agent_runs hatası (yok sayıldı)", exc_info=True)
        return []


F = TypeVar("F", bound=Callable[..., Any])


def tracked(agent_id: str, trigger_type: str = "manual") -> Callable[[F], F]:
    """Bir fonksiyonu/metodu ajan koşusu olarak izleyen async-farkında dekoratör.

    Davranışı DEĞİŞTİRMEZ: dönüş değeri aynen geçer; istisna aynen yeniden fırlatılır
    (ama önce koşu ``failed`` kaydedilir). ``{"ok": False}`` dönüşü ``failed`` sayılır.
    Tracking hataları sessizce yutulur.
    """

    def deco(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrap(*args: Any, **kwargs: Any) -> Any:
                tr = get_tracker()
                run_id = tr.safe_start(agent_id, trigger_type=trigger_type)
                token = _current_run.set(run_id)
                try:
                    result = await fn(*args, **kwargs)
                except asyncio.CancelledError:
                    # CancelledError BaseException'dır (Exception DEĞİL) → ayrıca yakala ki
                    # sunucu kapanışında iptal edilen arka plan turu sonsuza dek 'running'
                    # kalmasın. İptal AYNEN yeniden fırlatılır (gözlemci davranışı bozmaz).
                    tr.safe_finish(run_id, status=AgentRunStatus.cancelled, error="cancelled")
                    raise
                except Exception as exc:
                    tr.safe_finish(run_id, status=AgentRunStatus.failed, error=repr(exc))
                    raise
                else:
                    status, err = _status_for(result)
                    tr.safe_finish(run_id, status=status, error=err, summary=_summarize(result))
                    return result
                finally:
                    _current_run.reset(token)

            return cast(F, awrap)

        @functools.wraps(fn)
        def wrap(*args: Any, **kwargs: Any) -> Any:
            tr = get_tracker()
            run_id = tr.safe_start(agent_id, trigger_type=trigger_type)
            token = _current_run.set(run_id)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                tr.safe_finish(run_id, status=AgentRunStatus.failed, error=repr(exc))
                raise
            else:
                status, err = _status_for(result)
                tr.safe_finish(run_id, status=status, error=err, summary=_summarize(result))
                return result
            finally:
                _current_run.reset(token)

        return cast(F, wrap)

    return deco
