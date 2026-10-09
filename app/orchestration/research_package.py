"""Eğitim sonrası araştırma paketi: kapalı varsayılan, seri ve sınırlı işler.

Komut varsayılan olarak salt-okunur plan verir. ``--run`` yalnız bir vadesi gelmiş
aşamayı çalıştırır; eğitim/terfi/git işlemi içermez. Zamanlayıcı ayrı çağırandır.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.procutil import NO_WINDOW

Stage = Literal["discovery", "ingestion", "cards", "data", "methods", "report"]
STAGES: tuple[Stage, ...] = ("discovery", "ingestion", "cards", "data", "methods", "report")
BASE_MODELS = frozenset({"qwen3:30b-a3b-instruct-2507-q4_K_M", "qwen3:4b-instruct-2507-q4_K_M"})


class PackageConfig(BaseModel):
    """Paketin tek reçetesi; bilinmeyen alanlar sessizce kabul edilmez."""

    model_config = ConfigDict(extra="forbid")
    teacher_model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M"
    wait_for_adapters: list[str] = Field(min_length=1)
    quiet_minutes: int = Field(default=15, ge=1, le=1440)
    min_available_ram_gb: int = Field(default=32, ge=4, le=1024)
    stage_timeout_minutes: int = Field(default=45, ge=1, le=180)
    seed: int = Field(default=42, ge=0)
    papers_per_cycle: int = Field(default=2, ge=1, le=5)
    questions_per_cycle: int = Field(default=6, ge=1, le=20)
    max_attempts: int = Field(default=3, ge=1, le=5)
    interval_hours: dict[Stage, int]

    @field_validator("teacher_model")
    @classmethod
    def base_only(cls, value: str) -> str:
        if value not in BASE_MODELS:
            raise ValueError("Üretici yalnız açıkça izin verilen BASE Ollama modeli olabilir.")
        return value

    @field_validator("wait_for_adapters")
    @classmethod
    def safe_adapters(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) for value in values):
            raise ValueError("Geçersiz adapter adı.")
        return values

    @field_validator("interval_hours")
    @classmethod
    def intervals(cls, values: dict[Stage, int]) -> dict[Stage, int]:
        if set(values) != set(STAGES) or any(not 1 <= v <= 720 for v in values.values()):
            raise ValueError("Altı aşamanın her biri için 1–720 saat aralığı gerekli.")
        return values


def read_json(path: Path, default: Any = None) -> Any:
    """Yokluk ile bozuk dosyayı ayır: bozuk kayıt hatası çağırana yükselir."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    """Kayıtları atomik değiştir; yarım JSON bırakma."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def package_dir(root: Path) -> Path:
    return root / "storage" / "research_package"


def completed_adapter(path: Path) -> bool:
    """Yalnız işaretin varlığı değil, tamamlanan adım ve plan da doğrulanır."""
    plan = read_json(path / "run_plan.json", {})
    done = read_json(path / "run_complete.json", {})
    target = int(plan.get("max_steps", 0))
    return bool(
        target > 0
        and done.get("finished_at")
        and int(done.get("max_steps", -1)) == target
        and int(done.get("global_step", -1)) >= target
    )


def process_is_heavy(args: list[str]) -> bool:
    """Eğitim, birleştirme ve eval süreçlerini komut sözcükleriyle tanı."""
    names = {Path(arg).name.lower() for arg in args}
    if names & {"peft_lora_train.py", "merge_adapter.py", "llm30_run.py"}:
        return True
    hektor = bool(names & {"hektor", "hektor.exe", "app.main"})
    return hektor and bool(
        names & {"lora-eval", "llm30-run"} or ("train" in names and "--run" in names)
    )


def blockers(root: Path, cfg: PackageConfig, *, check_processes: bool = True) -> list[str]:
    """Tüm kontroller salt-okunur; belirsiz eğitim durumu işi bekletir."""
    reasons: list[str] = []
    for marker in ("STOP_ALL", "STOP_LEARNING", "STOP_RESEARCH"):
        if (root / "storage" / marker).exists():
            reasons.append(f"Durdurma işareti: {marker}")
    if (root / "storage" / ".training_launching").exists():
        reasons.append("Eğitim başlatma kilidi var.")
    adapters = root / "models" / "adapters"
    names = set(cfg.wait_for_adapters)
    names.update(p.parent.name for p in adapters.glob("*/run_plan.json"))
    try:
        status = read_json(root / "storage" / "train_status.json", {})
        if status.get("adapter"):
            name = str(status["adapter"])
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
                raise ValueError("Eğitim durumunda geçersiz adapter adı")
            names.add(name)
        for name in sorted(names):
            path = adapters / name
            if not completed_adapter(path):
                reasons.append(f"Eğitim tamamlanması doğrulanmadı: {name}")
            elif time.time() - (path / "run_complete.json").stat().st_mtime < (
                cfg.quiet_minutes * 60
            ):
                reasons.append(f"Eğitim sonrası {cfg.quiet_minutes} dk bekleme: {name}")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        reasons.append(f"Eğitim kayıtları okunamadı: {exc}")
    if check_processes:
        for proc in psutil.process_iter(["pid", "cmdline", "name"]):
            try:
                if proc.pid != os.getpid() and process_is_heavy(proc.info["cmdline"] or []):
                    reasons.append(f"Ağır süreç çalışıyor: PID {proc.pid}")
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                reasons.append("Bir sürecin eğitim durumu doğrulanamadı.")
        if psutil.virtual_memory().available < cfg.min_available_ram_gb * 1024**3:
            reasons.append(f"Kullanılabilir RAM {cfg.min_available_ram_gb} GB altında.")
    return reasons


def next_stage(cfg: PackageConfig, state: dict[str, Any], now: float) -> Stage | None:
    """En uzun süredir bekleyen işi seç; hata ve boş turlar da bütçe tüketir."""
    due: list[tuple[float, int, Stage]] = []
    for i, stage in enumerate(STAGES):
        row = state.get(stage, {})
        if int(row.get("failures", 0)) >= cfg.max_attempts:
            continue
        last = float(row.get("last_attempt", 0))
        if now - last >= cfg.interval_hours[stage] * 3600:
            due.append((last, i, stage))
    return min(due)[2] if due else None


@contextmanager
def package_lock(root: Path) -> Iterator[None]:
    """Aynı paketin iki işini atomik kilitle engelle; çökmede elle inceleme gerekir."""
    path = package_dir(root) / "running.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def worker_env(cfg: PackageConfig) -> dict[str, str]:
    """Sohbet/.env ayarlarına dokunmadan yalnız alt sürecin üreticisini sabitle."""
    return {
        **os.environ,
        "HEKTOR_LLM_MODEL": cfg.teacher_model,
        "HEKTOR_BACKGROUND_LOOPS_ENABLED": "false",
        "HEKTOR_UNATTENDED_TRAINING_ENABLED": "false",
        "HEKTOR_ALLOW_FAKE_EMBEDDINGS": "false",
        "HEKTOR_RAG_TRANSLATE_MODEL": cfg.teacher_model,
        "PYTHONIOENCODING": "utf-8",
    }


def run_stage(
    root: Path,
    config_path: Path,
    cfg: PackageConfig,
    stage: Stage,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Sabit işçiyi çalıştır; eğitim başlarsa veya süre dolarsa yalnız işçiyi durdur."""
    output = package_dir(root) / "last_result.json"
    output.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "app.orchestration.research_worker",
        stage,
        str(config_path.resolve()),
    ]
    started = time.monotonic()
    with (package_dir(root) / f"{stage}.log").open("a", encoding="utf-8") as log:
        log.write(f"\n--- {dt.datetime.now(dt.UTC).isoformat()} ---\n")
        log.flush()
        child = subprocess.Popen(
            cmd,
            cwd=root,
            env=worker_env(cfg),
            stdout=log,
            stderr=log,
            creationflags=NO_WINDOW,
        )
        try:
            while child.poll() is None:
                if should_stop and should_stop():
                    raise RuntimeError("Paket yöneticisi durduruldu.")
                if time.monotonic() - started > cfg.stage_timeout_minutes * 60:
                    raise RuntimeError("Aşama süre bütçesini aştı.")
                reasons = blockers(root, cfg)
                if reasons:
                    raise RuntimeError("Aşama duraklatıldı: " + "; ".join(reasons))
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"Aşama hata kodu: {child.returncode}; bkz. {stage}.log")
            result = read_json(output)
            if not isinstance(result, dict) or result.get("stage") != stage:
                raise RuntimeError("Aşama sonuç kaydı eksik veya uyuşmuyor.")
            return result
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)


def tick(
    root: Path,
    config_path: Path,
    *,
    execute: bool = False,
    from_service: bool = False,
    review: Callable[[Stage, dict[str, Any]], dict[str, Any]] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Plan veya tek aşama. Canlı eğitim verisi ve model terfisi bu paketin dışında."""
    cfg = PackageConfig.model_validate(read_json(config_path))
    reasons = blockers(root, cfg)
    service = read_json(package_dir(root) / "service.json", {})
    if execute and service.get("enabled") and not from_service:
        reasons.append("Paket web yöneticisine bağlı; ikinci zamanlayıcı çalıştırılamaz.")
    state_path = package_dir(root) / "state.json"
    state = read_json(state_path, {})
    stage = next_stage(cfg, state, time.time())
    if (package_dir(root) / "running.lock").exists():
        reasons.append("Paket kilidi var; çalışan iş veya kesilmiş tur incelenmeli.")
    result: dict[str, Any] = {
        "mode": "run" if execute else "dry-run",
        "blocked": reasons,
        "next_stage": stage,
        "teacher": cfg.teacher_model,
        "state": state,
    }
    if not execute or reasons or stage is None:
        return result
    with package_lock(root):
        # Kilit öncesi iki çağıranın aynı eski planı seçmesini engelle.
        state = read_json(state_path, {})
        stage = next_stage(cfg, state, time.time())
        reasons = blockers(root, cfg)
        if reasons or stage is None:
            return {**result, "blocked": reasons, "next_stage": stage}
        old = state.get(stage, {})
        state[stage] = {**old, "last_attempt": time.time(), "status": "running"}
        write_json(state_path, state)
        try:
            if review:
                state[stage]["review"] = review(stage, result)
                write_json(state_path, state)
            reasons = blockers(root, cfg)
            if reasons or (should_stop and should_stop()):
                raise RuntimeError("İş öncesi kapı: " + "; ".join(reasons or ["Durduruldu"]))
            if should_stop is None:
                outcome = run_stage(root, config_path, cfg, stage)
            else:
                outcome = run_stage(root, config_path, cfg, stage, should_stop=should_stop)
            partial = outcome.get("errors", 0) or outcome.get("not_created", 0)
            state[stage].update(
                status="partial" if partial else "completed", failures=0, result=outcome
            )
        except Exception as exc:
            outcome = {"error": str(exc)}
            state[stage].update(status="failed", failures=int(old.get("failures", 0)) + 1)
            state[stage]["result"] = outcome
        write_json(state_path, state)
        return {**result, "executed": stage, "outcome": outcome, "state": state}


def control(root: Path, action: Literal["pause", "resume"]) -> dict[str, str]:
    """Paket durdurma işaretini yönet; diğer STOP işaretlerini kaldırma."""
    marker = root / "storage" / "STOP_RESEARCH"
    if action == "pause":
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(dt.datetime.now(dt.UTC).isoformat(), encoding="utf-8")
    else:
        marker.unlink(missing_ok=True)
    return {"action": action, "marker": str(marker)}
