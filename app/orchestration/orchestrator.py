"""orchestrator.py — dayanıklı eğitim orkestrasyonu (saf durum makinesi).

Eğitim iç-bağımlılığı YOK: aşamaları enjekte edilen *delege* fonksiyonlarıyla yürütür
(`delegates.py` gerçek bağlamayı sağlar; testler sahte enjekte eder → %100 çevrimdışı).

Sözleşme:
  - `step()` tam olarak BİR aşama ilerletir; durumu + checkpoint çıktısını DB'ye yazar.
  - `run_until_blocked()` aşamaları zincirler; blocked/failed/completed olunca durur.
  - Resume: tamamlanan aşamalar atlanır; blocked/failed aşama yeniden denenir (idempotent
    delege beklenir) → session-limit'e dayanıklı.
  - `recover_stale()`: kalp atışı durmuş 'running' aşamayı failed yapar (panic recovery).
  - Gerçek eğitim ASLA gözetimsiz başlamaz: `approval`/`train` delegesi onay sınırında durur.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.orchestration.pipeline import (
    HALTING_STAGE_STATUSES,
    PIPELINE_BY_NAME,
    TERMINAL_STAGE_STATUSES,
    RunStatus,
    StageStatus,
)
from app.orchestration.store import OrchestrationStore, iso_minutes_ago, utcnow

log = logging.getLogger(__name__)

_TERMINAL_STAGE_VALUES: frozenset[str] = frozenset(s.value for s in TERMINAL_STAGE_STATUSES)
_FINAL_RUN_VALUES: frozenset[str] = frozenset(
    {RunStatus.completed.value, RunStatus.cancelled.value}
)
_STOP_RUN_VALUES: frozenset[str] = frozenset(
    {
        RunStatus.blocked.value,
        RunStatus.failed.value,
        RunStatus.completed.value,
        RunStatus.cancelled.value,
    }
)


@dataclass
class StageResult:
    """Bir aşama delegesinin dönüşü."""

    status: StageStatus
    message: str = ""
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Delege'ye geçirilen bağlam (koşu + parametreler + depo erişimi)."""

    run_id: str
    stage: str
    run: dict[str, Any]
    params: dict[str, Any]
    store: OrchestrationStore


StageDelegate = Callable[[RunContext], StageResult]


def _noop_delegate(ctx: RunContext) -> StageResult:
    return StageResult(StageStatus.skipped, f"{ctx.stage}: delege tanımlı değil (skip).")


def _level_for(status: StageStatus) -> str:
    if status == StageStatus.failed:
        return "error"
    if status == StageStatus.blocked:
        return "warning"
    return "info"


class TrainingOrchestrator:
    """Eğitim hattını dayanıklı biçimde yürüten koordinatör."""

    def __init__(
        self,
        store: OrchestrationStore | None = None,
        delegates: Mapping[str, StageDelegate] | None = None,
    ) -> None:
        self.store = store or OrchestrationStore()
        if delegates is None:
            from app.orchestration.delegates import default_delegates

            delegates = default_delegates()
        self._delegates: dict[str, StageDelegate] = dict(delegates)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(
        self,
        *,
        model: str,
        profile: str,
        adapter_name: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        run_id = self.store.create_run(
            model=model, profile=profile, adapter_name=adapter_name, params=params or {}
        )
        log.info("Orkestrasyon koşusu başlatıldı: %s (model=%s)", run_id, model)
        return run_id

    def _next_actionable(self, run_id: str) -> dict[str, Any] | None:
        """İlk terminal-olmayan (pending/running/blocked/failed) aşama; yoksa None."""
        for st in self.store.get_stages(run_id):
            if st["status"] in _TERMINAL_STAGE_VALUES:
                continue
            return st
        return None

    def step(self, run_id: str) -> dict[str, Any]:
        """Tam olarak bir aşama ilerlet. Koşu/aşama anlık görüntüsünü döner."""
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"Bilinmeyen koşu: {run_id}")
        if run["status"] in _FINAL_RUN_VALUES:
            return self._snapshot(run_id, note="run terminal")

        target = self._next_actionable(run_id)
        if target is None:
            self.store.update_run(
                run_id, status=RunStatus.completed.value, current_stage="", error=""
            )
            self.store.add_event(run_id, "", "finish", "Tüm aşamalar tamamlandı")
            return self._snapshot(run_id)

        name = target["name"]
        sdef = PIPELINE_BY_NAME.get(name)
        title = sdef.title if sdef else name

        if not self.store.claim_stage_running(run_id, name):
            # Aşamayı başka bir eşzamanlı step() (FastAPI threadpool) zaten 'running'e
            # almış → çift-delege etme; no-op anlık görüntü dön (atomik CAS-claim ile yarış
            # kapatıldı). run_until_blocked döngüsü bunu max_steps sınırında doğal bırakır.
            return self._snapshot(run_id, note="stage already claimed")
        self.store.update_run(run_id, status=RunStatus.running.value, current_stage=name, error="")
        self.store.add_event(run_id, name, "info", f"{title} başladı")

        delegate = self._delegates.get(name, _noop_delegate)
        ctx = RunContext(
            run_id=run_id,
            stage=name,
            run=run,
            params=dict(run.get("params", {})),
            store=self.store,
        )
        try:
            result = delegate(ctx)
        except Exception as exc:  # delege patlarsa koşuyu failed yap (asla sessiz yutma)
            log.exception("Aşama delegesi hata verdi: %s", name)
            self.store.update_stage(
                run_id,
                name,
                status=StageStatus.failed.value,
                finished_at=utcnow(),
                message=f"hata: {exc}",
            )
            self.store.update_run(run_id, status=RunStatus.failed.value, error=f"{name}: {exc}")
            self.store.add_event(run_id, name, "error", f"{title} hata: {exc}")
            return self._snapshot(run_id)

        # Delege sürerken koşu eşzamanlı cancel/finalize edilmiş olabilir (threadpool yarışı).
        # Sonucu yazıp iptali/tamamlanmayı clobber etme; kapılan aşamayı temizleyip çık.
        run_now = self.store.get_run(run_id)
        if run_now is not None and run_now["status"] in _FINAL_RUN_VALUES:
            self.store.update_stage(
                run_id,
                name,
                status=StageStatus.skipped.value,
                finished_at=utcnow(),
                message="koşu sonlandı (iptal/tamamlandı) — aşama atlandı",
            )
            return self._snapshot(run_id, note="run finalized during stage")

        self.store.update_stage(
            run_id,
            name,
            status=result.status.value,
            message=result.message,
            output=result.output,
            finished_at=utcnow(),
        )
        self.store.add_event(
            run_id,
            name,
            _level_for(result.status),
            f"{title}: {result.message}" if result.message else f"{title} → {result.status.value}",
        )

        if result.status in HALTING_STAGE_STATUSES:
            if result.status == StageStatus.failed:
                self.store.update_run(run_id, status=RunStatus.failed.value, error=result.message)
            else:  # blocked
                self.store.update_run(run_id, status=RunStatus.blocked.value, error="")
        else:  # completed / skipped → bir sonrakine bak
            nxt = self._next_actionable(run_id)
            if nxt is None:
                self.store.update_run(
                    run_id, status=RunStatus.completed.value, current_stage="", error=""
                )
                self.store.add_event(run_id, "", "finish", "Tüm aşamalar tamamlandı")
            else:
                self.store.update_run(
                    run_id, status=RunStatus.running.value, current_stage=nxt["name"], error=""
                )

        return self._snapshot(run_id)

    def run_until_blocked(self, run_id: str, max_steps: int = 50) -> dict[str, Any]:
        """blocked/failed/completed/cancelled olana dek aşamaları zincirle."""
        snap = self._snapshot(run_id)
        for _ in range(max(1, max_steps)):
            snap = self.step(run_id)
            run = snap.get("run") or {}
            if run.get("status") in _STOP_RUN_VALUES:
                return snap
        # Döngü tükendi ama koşu hâlâ terminal değil → sessizce 'running' bırakma; uyar.
        # (Koşu resume edilebilir; ama çağıran fark etsin diye log + note.)
        log.warning(
            "run_until_blocked: max_steps (%d) aşıldı, koşu hâlâ ilerliyor: %s", max_steps, run_id
        )
        return self._snapshot(run_id, note="max_steps reached")

    def recover_stale(self, timeout_min: float = 30.0) -> list[dict[str, str]]:
        """Kalp atışı timeout_min'den eski 'running' aşamaları failed yap (panic recovery)."""
        cutoff = iso_minutes_ago(timeout_min)
        stale = self.store.find_stale_running_stages(cutoff)
        recovered: list[dict[str, str]] = []
        for run_id, name in stale:
            self.store.update_stage(
                run_id,
                name,
                status=StageStatus.failed.value,
                finished_at=utcnow(),
                message="panic-recovery: kalp atışı durdu (stale)",
            )
            self.store.update_run(
                run_id, status=RunStatus.failed.value, error=f"{name}: stale (yanıt yok)"
            )
            self.store.add_event(
                run_id,
                name,
                "error",
                "Panic recovery: stale 'running' → failed (resume ile sürdürülebilir)",
            )
            recovered.append({"run_id": run_id, "stage": name})
        return recovered

    def cancel(self, run_id: str, reason: str = "") -> dict[str, Any]:
        # Asılı kalmış 'running' aşamayı terminal duruma (skipped) çek → recover_stale onu
        # 'failed'a çevirip iptali clobber edemesin + zombi 'running' aşama kalmasın.
        note = f"koşu iptal edildi: {reason}" if reason else "koşu iptal edildi"
        for st in self.store.get_stages(run_id):
            if st["status"] == StageStatus.running.value:
                self.store.update_stage(
                    run_id,
                    st["name"],
                    status=StageStatus.skipped.value,
                    finished_at=utcnow(),
                    message=note,
                )
        self.store.update_run(run_id, status=RunStatus.cancelled.value, error=reason)
        self.store.add_event(run_id, "", "warning", f"Koşu iptal edildi: {reason}")
        return self._snapshot(run_id)

    # ── views ────────────────────────────────────────────────────────────────

    def status(self, run_id: str) -> dict[str, Any]:
        return self._snapshot(run_id)

    def timeline(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return self.store.get_events(run_id, limit=limit)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.list_runs(limit=limit)

    def _snapshot(self, run_id: str, note: str = "") -> dict[str, Any]:
        run = self.store.get_run(run_id)
        stages = self.store.get_stages(run_id)
        return {"run": run, "stages": stages, "note": note}
