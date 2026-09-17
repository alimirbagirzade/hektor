"""Eğitim nöbeti — koşan eğitimin SAĞLIĞI + kurtarma YETKİSİ (salt-okuma, LLM'siz).

2026-09-16 gecesinin iki olayı bu modülün varlık sebebidir:

1. **Onaysız dirilme.** Web'den başlatılan bir eğitim 0. adımda öldü; `train_status.json`
   diskte kaldı. `training-watchdog.ps1` dosyayı görüp koşuyu 13 dakika sonra yeniden
   başlattı ve 5,5 saat boyunca ESKİ veriyle, kimsenin onaylamadığı bir eğitim koştu.
   Durum dosyası fiilen "kalıcı yetki" gibi davranıyordu (Kural 8 ihlali).
2. **Fark edilmeyen donma.** Başka bir koşu RAM açmak için askıya alındı
   (``NtSuspendProcess``) ve 5,5 saat 21/600 adımda dondu; süreç "canlı" göründüğü için
   kimse fark etmedi.

Buradaki fonksiyonlar SAF'tır (zaman, süreç ve dosya bilgisi dışarıdan verilir) → testte
gerçek süreç/onay gerekmez. Eğitim BAŞLATMAZ, DURDURMAZ, hiçbir şey yazmaz (Kural 8).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Durum dosyası bundan eskiyse "bayat" sayılır: o kadar süre önce başlamış bir koşuyu
# diriltmek, aradaki veri/kod değişikliklerini görmezden gelmek demektir.
MAX_STATUS_AGE_HOURS = 72
# Tüketilen onayın koşu başlangıcına yakınlığı: onay tüketimi ile sürecin doğması arasında
# model yükleme + bölme kadar fark olur (dakikalar). Bu pencere dışındaki bir onay BAŞKA
# bir koşuya aittir (2026-09-08'de verilen onayın 2026-09-15'te tüketilmesi gibi).
# K8-b (2026-09-17): `train_status.json` artık `approval_id`'yi DOĞRUDAN taşıyor
# (bkz. `detached_launch._status_payload`, `start-train.ps1`) — bu pencere ARTIK YALNIZ
# o alan YOKSA (eski/harici durum dosyaları) yedek olarak kullanılır; kimlik varsa TAHMİN
# değil doğrudan eşleşme geçerlidir.
APPROVAL_WINDOW_MINUTES = 20
# Eğitim canlıyken log bu kadar süre ilerlemiyorsa: askıya alınmış / donmuş olabilir.
# CPU'da tek adım dakikalar sürer; eşik cömert tutuldu ki yavaş adım alarm üretmesin.
LOG_STALL_MINUTES = 45
# Veri, koşu başladıktan bu kadar sonra değiştiyse eğitilen set artık diskteki set değildir.
DATA_DRIFT_GRACE_MINUTES = 5


def _parse_iso(value: Any) -> dt.datetime | None:
    """ISO zaman damgasını UTC-farkındalıklı çevir (çevrilemezse None)."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.UTC) if parsed.tzinfo is None else parsed


def find_run_approval(
    approvals: list[dict[str, Any]],
    started_at: dt.datetime,
    *,
    action: str = "train_run",
    window_minutes: int = APPROVAL_WINDOW_MINUTES,
    approval_id: str = "",
) -> dict[str, Any] | None:
    """Bu koşuya ait TÜKETİLMİŞ insan onayını bul (yoksa None).

    K8-b: ``approval_id`` verilmişse (artık ``train_status.json``'da yazılı — bkz.
    ``detached_launch._status_payload``, ``start-train.ps1``) ÖNCE tam KİMLİK eşleşmesi
    denenir; bulunamaz/onaylı-tüketilmiş değilse ZAMAN PENCERESİNE DÜŞÜLMEZ (bir kimlik
    verilip de uyuşmuyorsa bu şüphelidir — daha zayıf bir sezgiye geri dönmek yanlış
    güven verir). Zaman penceresi yalnız bu alan hiç YOKSA (eski/harici durum dosyaları,
    ör. `mac-loop.sh`'nin yazdığı sade dosya) devreye girer — eşleşme ölçütü: aksiyon
    ``train_run``, durum ``approved`` ve ``consumed_at`` koşunun başlangıcına
    ``window_minutes`` içinde.
    """
    if approval_id:
        for row in approvals:
            if row.get("approval_id") != approval_id:
                continue
            if row.get("action") != action or row.get("status") != "approved":
                return None
            if _parse_iso(row.get("consumed_at")) is None:
                return None
            return row
        return None

    window = dt.timedelta(minutes=window_minutes)
    for row in approvals:
        if row.get("action") != action or row.get("status") != "approved":
            continue
        consumed = _parse_iso(row.get("consumed_at"))
        if consumed is None:
            continue
        if abs(consumed - started_at) <= window:
            return row
    return None


@dataclass
class RecoveryVerdict:
    """Nöbetçi bu koşuyu diriltebilir mi?"""

    allowed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason, "details": self.details}


def recovery_allowed(
    status: dict[str, Any],
    approvals: list[dict[str, Any]],
    *,
    now: dt.datetime,
    data_mtime: dt.datetime | None = None,
) -> RecoveryVerdict:
    """Çöken bir eğitimi yeniden başlatmak YETKİLİ mi? FAIL-CLOSED.

    Nöbetçi kurtarması Kural 8'den muaftır ("onay zaten tüketilmişti") — ama bu muafiyet
    yalnız o cümle DOĞRUYSA geçerlidir. Burada doğrulanır; doğrulanamıyorsa dirilme YOK.
    """
    if not status:
        return RecoveryVerdict(False, "durum dosyası yok/boş — diriltilecek koşu kaydı yok")

    started_at = _parse_iso(status.get("started_at"))
    if started_at is None:
        return RecoveryVerdict(
            False, "durum dosyasında okunabilir 'started_at' yok — koşu onaya bağlanamıyor"
        )

    age_h = (now - started_at).total_seconds() / 3600.0
    if age_h > MAX_STATUS_AGE_HOURS:
        return RecoveryVerdict(
            False,
            f"durum dosyası bayat ({age_h:.0f} saat > {MAX_STATUS_AGE_HOURS}) — "
            "yeni eğitim yeni onay ister",
            {"age_hours": round(age_h, 1)},
        )

    approval = find_run_approval(
        approvals, started_at, approval_id=str(status.get("approval_id") or "")
    )
    if approval is None:
        return RecoveryVerdict(
            False,
            "bu koşuya bağlı TÜKETİLMİŞ insan onayı bulunamadı (Kural 8) — "
            "durum dosyası kalıcı yetki değildir",
            {"started_at": started_at.isoformat()},
        )

    if data_mtime is not None:
        drift = data_mtime - started_at
        if drift > dt.timedelta(minutes=DATA_DRIFT_GRACE_MINUTES):
            return RecoveryVerdict(
                False,
                "eğitim verisi koşu başladıktan SONRA değişti — diriltilen koşu onaylanan "
                "veriyi eğitmez",
                {"data_mtime": data_mtime.isoformat(), "started_at": started_at.isoformat()},
            )

    return RecoveryVerdict(
        True,
        "onaylı koşu — kurtarma yetkili",
        {"approval_id": approval.get("approval_id", ""), "started_at": started_at.isoformat()},
    )


@dataclass
class TrainingDiagnosis:
    """Koşan (ya da koşmayan) eğitimin sağlık raporu."""

    verdict: str  # OK | DIKKAT | BOSTA
    problems: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "problems": self.problems, "info": self.info}


def diagnose(
    *,
    status: dict[str, Any],
    running: bool,
    now: dt.datetime,
    log_mtime: dt.datetime | None = None,
    cpu_percent: float | None = None,
    data_mtime: dt.datetime | None = None,
    approvals: list[dict[str, Any]] | None = None,
) -> TrainingDiagnosis:
    """Eğitim sağlığını değerlendir (saf). Hiçbir şey başlatmaz/durdurmaz."""
    problems: list[str] = []
    info: dict[str, Any] = {
        "running": running,
        "adapter": status.get("adapter", ""),
        "pid": status.get("pid", 0),
        "started_at": status.get("started_at", ""),
    }
    started_at = _parse_iso(status.get("started_at"))

    if running:
        if log_mtime is not None:
            stall_min = (now - log_mtime).total_seconds() / 60.0
            info["log_stall_minutes"] = round(stall_min, 1)
            if stall_min > LOG_STALL_MINUTES:
                problems.append(
                    f"eğitim canlı ama log {stall_min:.0f} dakikadır ilerlemiyor — "
                    "askıya alınmış (NtSuspendProcess) ya da donmuş olabilir"
                )
        if cpu_percent is not None:
            info["cpu_percent"] = round(cpu_percent, 1)
            if cpu_percent < 1.0:
                problems.append(
                    f"eğitim süreci canlı ama CPU kullanımı ~%{cpu_percent:.1f} — "
                    "hesap yapmıyor (askıda/bloke)"
                )
        if not status:
            # Durum kaydı yoksa koşuyu onaya BAĞLAYAMAYIZ: Kural 8 kontrolü sessizce
            # atlanır ve yetkisiz bir koşu "sağlıklı" görünür. Sessiz atlama yerine
            # açıkça bildir (Kural 2/7: doğrulanmayan şeye "tamam" denmez).
            problems.append(
                "eğitim koşuyor ama durum kaydı (train_status.json) YOK — bu koşu insan "
                "onayına bağlanamıyor ve reçetesi doğrulanamıyor"
            )
        if (
            approvals is not None
            and started_at is not None
            and find_run_approval(
                approvals, started_at, approval_id=str(status.get("approval_id") or "")
            )
            is None
        ):
            problems.append(
                "koşan eğitime bağlı TÜKETİLMİŞ insan onayı yok (Kural 8) — "
                "bu koşu yetkisiz başlatılmış olabilir"
            )
        if (
            data_mtime is not None
            and started_at is not None
            and data_mtime - started_at > dt.timedelta(minutes=DATA_DRIFT_GRACE_MINUTES)
        ):
            problems.append(
                "eğitim verisi koşu başladıktan sonra değişti — eğitilen set diskteki "
                "set değil (sonucu yorumlarken yanıltır)"
            )
    elif status:
        problems.append(
            "durum dosyası var ama eğitim süreci YOK — ölü koşu kaydı; nöbetçi bunu "
            "diriltmeye çalışır (kurtarma kontrolü: hektor train-recovery-check)"
        )

    if not running and not status:
        return TrainingDiagnosis("BOSTA", [], info)
    return TrainingDiagnosis("DIKKAT" if problems else "OK", problems, info)


# --------------------------------------------------------------------------
# I/O sarmalayıcı — gerçek durum/süreç/dosya bilgisini toplar (salt-okuma)
# --------------------------------------------------------------------------
def _file_mtime(path: Path) -> dt.datetime | None:
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)
    except OSError:
        return None


def _newest_train_log(root: Path) -> Path | None:
    """`logs/` altındaki en yeni eğitim log dosyası (train-*.log)."""
    logs = root / "logs"
    if not logs.is_dir():
        return None
    candidates = [p for p in logs.glob("train-*.log") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _trainer_processes() -> list[dict[str, Any]]:
    """Koşan eğitim süreçleri (cmdline'da `train --run` ya da `peft_lora_train`)."""
    try:
        import psutil
    except Exception:
        return []
    found: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
        except Exception:
            continue
        if not cmd:
            continue
        if ("train" in cmd and "--run" in cmd) or "peft_lora_train" in cmd:
            found.append({"pid": proc.info.get("pid", 0), "cmdline": cmd[:200], "proc": proc})
    return found


def _busiest_cpu_percent(procs: list[dict[str, Any]], interval: float = 1.0) -> float | None:
    """Eğitim süreçlerinin EN YÜKSEK CPU yüzdesi (askıda olan ~0 döner)."""
    if not procs:
        return None
    values: list[float] = []
    for entry in procs:
        proc = entry.get("proc")
        if proc is None:
            continue
        try:
            proc.cpu_percent(None)
        except Exception:
            continue
    try:
        import time

        time.sleep(interval)
    except Exception:  # pragma: no cover - uyku kesilirse ölçüm yine denenir
        pass
    for entry in procs:
        proc = entry.get("proc")
        if proc is None:
            continue
        try:
            values.append(float(proc.cpu_percent(None)))
        except Exception:
            continue
    return max(values) if values else None


def collect_diagnosis(
    root: Path, approvals: list[dict[str, Any]] | None = None
) -> TrainingDiagnosis:
    """Gerçek makineden veri toplayıp :func:`diagnose` çağır (salt-okuma)."""
    from app.training.detached_launch import read_detached_training_status

    status = read_detached_training_status(root)
    procs = _trainer_processes()
    running = bool(procs)
    log_path = _newest_train_log(root)
    diagnosis = diagnose(
        status=status,
        running=running,
        now=dt.datetime.now(dt.UTC),
        log_mtime=_file_mtime(log_path) if log_path else None,
        cpu_percent=_busiest_cpu_percent(procs) if running else None,
        data_mtime=_file_mtime(root / "data" / "training" / "jsonl" / "train.jsonl"),
        approvals=approvals,
    )
    diagnosis.info["log_file"] = str(log_path) if log_path else ""
    diagnosis.info["trainer_pids"] = [p["pid"] for p in procs]
    return diagnosis
