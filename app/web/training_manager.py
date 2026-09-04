"""LoRA eğitim süreci yöneticisi — web UI için singleton.

Tek bir eğitimi yönetir: başlat / durdur / anlık durum.
SSE ile frontend'e canlı satır akışı sağlar.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import subprocess
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class TrainState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TrainProgress:
    state: TrainState = TrainState.IDLE
    current_iter: int = 0
    total_iters: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    pct: float = 0.0
    adapter_name: str = ""
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    log_lines: list[str] = field(default_factory=list)
    loss_curve: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "current_iter": self.current_iter,
            "total_iters": self.total_iters,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "pct": round(self.pct, 1),
            "adapter_name": self.adapter_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "log_lines": self.log_lines[-200:],
        }


_ITER_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")
_VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")
_START_RE = re.compile(r"Starting training.*iters:\s*(\d+)")
_SAVED_RE = re.compile(r"Saved.*adapters\.safetensors")

# HuggingFace Trainer (PEFT/Windows/Linux) format
_PEFT_LOSS_RE = re.compile(r"'loss':\s*([\d.]+)")
_PEFT_EVAL_RE = re.compile(r"'eval_loss':\s*([\d.]+)")
_PEFT_EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
_PEFT_STEP_RE = re.compile(r"\[(\d+)/(\d+)")
_PEFT_SAVED_RE = re.compile(r"Saving model checkpoint|adapter.*saved", re.IGNORECASE)


class TrainingManager:
    """Uygulama genelinde tek örnek."""

    def __init__(self) -> None:
        self._progress = TrainProgress()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._subscribers: list[asyncio.Queue] = []

    @property
    def progress(self) -> TrainProgress:
        return self._progress

    def start(self, command: list[str], adapter_name: str, total_iters: int) -> bool:
        with self._lock:
            if self._progress.state == TrainState.RUNNING:
                return False
            self._progress = TrainProgress(
                state=TrainState.RUNNING,
                total_iters=total_iters,
                adapter_name=adapter_name,
                started_at=datetime.datetime.now().isoformat(timespec="seconds"),
            )

        self._proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_output, daemon=True).start()
        return True

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._progress.state = TrainState.STOPPED
        self._progress.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        self._broadcast({"type": "stopped"})

    def _read_output(self) -> None:
        proc = self._proc
        # assert yerine açık guard: -O (optimize) modunda assert kaybolur, None deref'i
        # önlemek için erken dön (yaris/durma anında proc None olabilir).
        if proc is None or proc.stdout is None:
            return
        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                self._parse_line(line)
                self._progress.log_lines.append(line)
                self._broadcast({"type": "log", "line": line, **self._progress.to_dict()})
        finally:
            proc.wait()
            self._progress.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
            if self._progress.state == TrainState.RUNNING:
                ok = proc.returncode == 0
                self._progress.state = TrainState.COMPLETED if ok else TrainState.FAILED
                if not ok:
                    self._progress.error = f"Exit code: {proc.returncode}"
            self._broadcast({"type": "done", **self._progress.to_dict()})
            self._persist_loss_curve()

    def _parse_line(self, line: str) -> None:
        m = _START_RE.search(line)
        if m:
            self._progress.total_iters = int(m.group(1))

        m = _ITER_RE.search(line)
        if m:
            it = int(m.group(1))
            self._progress.current_iter = it
            self._progress.train_loss = float(m.group(2))
            total = self._progress.total_iters or 1
            self._progress.pct = min(it / total * 100, 100.0)
            self._progress.loss_curve.append(
                {
                    "iter": it,
                    "train_loss": self._progress.train_loss,
                    "val_loss": self._progress.val_loss,
                }
            )

        m = _VAL_RE.search(line)
        if m:
            self._progress.val_loss = float(m.group(2))
            if self._progress.loss_curve:
                self._progress.loss_curve[-1]["val_loss"] = self._progress.val_loss

        if _SAVED_RE.search(line):
            self._progress.pct = 100.0

        m = _PEFT_LOSS_RE.search(line)
        if m:
            self._progress.train_loss = float(m.group(1))
            self._progress.loss_curve.append(
                {
                    "iter": self._progress.current_iter,
                    "train_loss": self._progress.train_loss,
                    "val_loss": self._progress.val_loss,
                }
            )

        m = _PEFT_EVAL_RE.search(line)
        if m:
            self._progress.val_loss = float(m.group(1))
            if self._progress.loss_curve:
                self._progress.loss_curve[-1]["val_loss"] = self._progress.val_loss

        m = _PEFT_STEP_RE.search(line)
        if m and self._progress.total_iters:
            step = int(m.group(1))
            self._progress.current_iter = step
            self._progress.pct = min(step / self._progress.total_iters * 100, 100.0)

        if _PEFT_SAVED_RE.search(line):
            self._progress.pct = 100.0

    def _persist_loss_curve(self) -> None:
        if not self._progress.loss_curve or not self._progress.adapter_name:
            return
        try:
            out_dir = Path("reports") / "training"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{self._progress.adapter_name}_loss.json"
            out.write_text(
                json.dumps(
                    {
                        "adapter_name": self._progress.adapter_name,
                        "started_at": self._progress.started_at,
                        "finished_at": self._progress.finished_at,
                        "total_iters": self._progress.total_iters,
                        "curve": self._progress.loss_curve,
                    },
                    indent=2,
                )
            )
        except Exception:
            pass

    def _broadcast(self, msg: dict) -> None:
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        import contextlib

        for q in dead:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(q)

    async def subscribe(self) -> AsyncIterator[dict]:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.append(q)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                except TimeoutError:
                    yield {"type": "ping", **self._progress.to_dict()}
                    continue
                yield msg
                if msg.get("type") in ("done", "stopped"):
                    break
        finally:
            import contextlib

            with contextlib.suppress(ValueError):
                self._subscribers.remove(q)


_manager: TrainingManager | None = None


def get_training_manager() -> TrainingManager:
    global _manager
    if _manager is None:
        _manager = TrainingManager()
    return _manager
