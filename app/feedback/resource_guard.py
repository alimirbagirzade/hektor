"""resource_guard.py — eğitim sürerken sohbet modelinin cevap verip veremeyeceği (sunucu tarafı).

Karar YALNIZ ölçüme dayanır, varsayılan değer uydurulmaz:

- Eğitim yoksa (ve başlatılmıyorsa) → izin. Ölçüm gerekmez; cevap sonrası ayak izi ölçülür.
- Eğitim sürüyorsa / başlatılıyorsa:
  * sohbet modeli Ollama'da ZATEN yüklüyse → izin (ek yükleme yok);
  * değilse son ölçümün RAM kısmı (toplam − VRAM) + güvenlik payı ≤ boş RAM VE VRAM kısmı ≤
    boş VRAM olmalı. Toplam model boyutu doğrudan ek RAM ihtiyacı SAYILMAZ.
  * ölçüm yoksa / boş RAM veya VRAM okunamıyorsa → cevaplama kapalı (gerekçesiyle). Ölçüm,
    eğitim DIŞINDA ilk cevapta ya da "Kaynak ölç" ile alınır → kalıcı kilit yok.

Yarış: cevap üretimi sırasında ``storage/chat_leases/`` altında bir KİRA dosyası tutulur. Sohbet
önce kirayı yazar SONRA ortak ağır iş kilidine (``app.training.resource_lock``) bakar; eğitim
(web, ``start-train.ps1``, doğrudan ``hektor train --run``), model dönüşümü ve karşılaştırma önce
o kilidi alır SONRA kiralara bakar. İki taraf da bayrağını yazdıktan sonra diğerini okuduğundan
ikisi birden ilerleyemez (en kötü ihtimalle ikisi de geri çekilir).

Dönüşüm / karşılaştırma sürerken sohbet cevabı HİÇ üretilmez (bellek + karşılaştırmanın servis
koşulları bozulmasın). Kira, sahibinin pid'ini taşır: süreç çökerse kira hemen bayat sayılır.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings


@dataclass
class GuardDecision:
    allowed: bool
    reason: str
    training: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── eğitim etkinliği ─────────────────────────────────────────────────────────


def training_activity(root: Path | None = None) -> dict[str, Any]:
    """Eğitim sürüyor mu / başlatılıyor mu (salt-okuma)."""
    from app.training import detached_launch as dl
    from app.training import resource_lock

    r = root or get_settings().root
    lock = resource_lock.status(r)
    info = lock["info"] or {}
    kind = str(info.get("kind") or "") if lock["held"] else ""
    starting = kind == "training" and info.get("state") in ("launching", "writing")
    active = kind == "training" and not starting
    source = "kilit" if active else ""
    try:
        from app.web.training_manager import get_training_manager

        state = getattr(getattr(get_training_manager().progress, "state", None), "value", "")
        if state == "running":
            active, source = True, "web"
    except Exception:
        pass
    if not active and not starting and dl.is_detached_training_running(r):
        active, source = True, "detached"
    heavy = kind if kind in ("conversion", "comparison") else ""
    return {
        "active": active,
        "starting": starting,
        "source": source,
        "heavy": heavy,
        "heavy_detail": resource_lock.describe(info) if kind else "",
    }


# ── kira (lease) ─────────────────────────────────────────────────────────────


def _lease_dir(root: Path) -> Path:
    return root / "storage" / "chat_leases"


def active_chat_leases(root: Path | None = None, ttl_s: float | None = None) -> list[dict]:
    r = root or get_settings().root
    ttl = float(ttl_s if ttl_s is not None else get_settings().chat_lease_ttl_s)
    out: list[dict] = []
    d = _lease_dir(r)
    if not d.is_dir():
        return out
    from app.training.resource_lock import owner_alive

    now = time.time()
    for p in sorted(d.glob("lease-*.json")):
        try:
            if now - p.stat().st_mtime > ttl:
                continue  # bayat kira yok sayılır (çöken üretim kalıcı engel olmasın)
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("pid") and not owner_alive(data):
            continue  # sahibi çökmüş: TTL'i beklemeden bayat
        out.append(data)
    return out


def chat_lease_blocker(root: Path | None = None) -> str | None:
    """Eğitim başlatma yolunun kullandığı kontrol: sohbet cevabı üretiliyorsa gerekçe."""
    leases = active_chat_leases(root)
    if not leases:
        return None
    return (
        f"Sohbet cevabı üretiliyor ({len(leases)} etkin kira) — eğitim bellek yarışmasın diye "
        "başlatılmadı; cevap bitince tekrar deneyin."
    )


def acquire_chat_lease(token: str, root: Path | None = None) -> tuple[Path | None, str]:
    """Kirayı yaz, SONRA eğitim durumuna bak. (kira_yolu | None, gerekçe)."""
    r = root or get_settings().root
    d = _lease_dir(r)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"lease-{token}.json"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None, "Bu istek zaten işleniyor."
    from app.training.resource_lock import process_create_time

    pid = os.getpid()
    rec = {"token": token, "pid": pid, "pid_create_time": process_create_time(pid)}
    try:
        os.write(fd, json.dumps({**rec, "at": time.time()}).encode())
    finally:
        os.close(fd)
    act = training_activity(r)
    if act.get("heavy"):
        release_chat_lease(path)
        return None, f"{act['heavy_detail']} — cevap üretimi bu an başlatılmadı."
    if act["starting"] or act["active"]:
        release_chat_lease(path)
        return None, "Eğitim başlatılıyor/sürüyor — cevap üretimi bu an başlatılmadı."
    return path, ""


def release_chat_lease(path: Path | None) -> None:
    if path is not None:
        with contextlib.suppress(OSError):
            path.unlink()


# ── karar ────────────────────────────────────────────────────────────────────


def evaluate(
    tag: str,
    *,
    activity: dict[str, Any],
    ps_entries: list[dict[str, Any]] | None,
    ram_available_gb: float | None,
    gpu_mem: tuple[float, float] | None,
    footprint: dict[str, Any] | None,
    margin_gb: float,
    unified_memory: bool = False,
) -> GuardDecision:
    """Saf karar fonksiyonu (tüm ölçümler dışarıdan) → test edilebilir."""
    from app.feedback.model_identity import footprint_from_ps, match_entry

    details: dict[str, Any] = {
        "model": tag,
        "ram_available_gb": ram_available_gb,
        "gpu_free_vram_gb": round(gpu_mem[1] - gpu_mem[0], 2) if gpu_mem else None,
        "footprint": footprint,
        "loaded_models": [
            {"name": e.get("name"), **footprint_from_ps(e)} for e in (ps_entries or [])
        ],
    }
    if activity.get("heavy"):
        return GuardDecision(
            False,
            f"{activity.get('heavy_detail') or 'Ağır iş sürüyor'} — bellek ve karşılaştırma "
            "koşulları bozulmasın diye cevaplama kapalı. Geçmiş ve inceleme açık.",
            activity,
            details,
        )
    if not activity.get("active") and not activity.get("starting"):
        return GuardDecision(True, "Eğitim yok — cevap verilebilir.", activity, details)
    if activity.get("starting") and not activity.get("active"):
        return GuardDecision(
            False,
            "Eğitim başlatılıyor — model cevaplaması geçici olarak kapalı.",
            activity,
            details,
        )
    if ps_entries is None:
        return GuardDecision(
            False,
            "Eğitim sürüyor ve Ollama'ya ulaşılamadı — bellek durumu ölçülemedi, cevaplama kapalı.",
            activity,
            details,
        )
    loaded = match_entry(ps_entries, tag)
    if loaded is not None:
        fp = footprint_from_ps(loaded)
        return GuardDecision(
            True,
            f"Eğitim sürüyor ama sohbet modeli zaten bellekte (RAM {fp['ram_gb']} GB, VRAM "
            f"{fp['vram_gb']} GB) — ek yükleme gerekmiyor.",
            activity,
            details,
        )
    if not footprint:
        return GuardDecision(
            False,
            "Eğitim sürüyor ve bu modelin bellek ölçümü yok — ölçüm eğitim dışında ilk cevapta "
            "ya da 'Kaynak ölç' ile alınır. Şimdilik cevaplama kapalı.",
            activity,
            details,
        )
    need_ram = float(footprint.get("ram_gb") or 0.0)
    need_vram = float(footprint.get("vram_gb") or 0.0)
    if need_vram > 0 and gpu_mem is None:
        if unified_memory:
            need_ram += need_vram  # Apple Silicon: VRAM = paylaşılan RAM
            need_vram = 0.0
        else:
            return GuardDecision(
                False,
                f"Model ölçümde {need_vram} GB VRAM kullanıyordu ama boş VRAM okunamadı "
                "(nvidia-smi yok) — cevaplama kapalı.",
                activity,
                details,
            )
    if ram_available_gb is None:
        return GuardDecision(False, "Boş RAM okunamadı — cevaplama kapalı.", activity, details)
    if need_vram > 0 and gpu_mem is not None:
        free_vram = gpu_mem[1] - gpu_mem[0]
        if need_vram > free_vram:
            return GuardDecision(
                False,
                f"Eğitim sürüyor: model {need_vram} GB VRAM istiyor, boş VRAM {free_vram:.1f} GB.",
                activity,
                details,
            )
    if need_ram + margin_gb > ram_available_gb:
        return GuardDecision(
            False,
            f"Eğitim sürüyor: model ~{need_ram} GB RAM (+{margin_gb} GB pay) istiyor, boş RAM "
            f"{ram_available_gb} GB — cevaplama kapalı. Geçmiş ve inceleme açık.",
            activity,
            details,
        )
    return GuardDecision(
        True,
        f"Eğitim sürüyor ama ölçülen ihtiyaç (RAM ~{need_ram} GB +{margin_gb} pay, VRAM "
        f"{need_vram} GB) boş belleğe sığıyor.",
        activity,
        details,
    )


def check_chat_resources(
    transport: Any = None, *, tag: str | None = None, slot: str = "main"
) -> GuardDecision:
    """Canlı ölçümleri topla ve karar ver (yazma yok). ``tag`` yoksa yuvadan çözülür."""
    from app.agents.system_profiler.profiler import _memory_info
    from app.feedback.model_identity import ollama_ps, read_footprint
    from app.training.train_load_doctor import _nvidia_smi_memory_gb

    s = get_settings()
    if tag is None:
        from app.feedback.model_activation import ActivationError, resolve_chat_tag

        try:
            tag = resolve_chat_tag(slot)
        except ActivationError as exc:  # ör. E-8a: ayardaki ana model pilot
            return GuardDecision(False, str(exc), training_activity(), {})
        if not tag:
            return GuardDecision(False, "Bu yuvada etkin model yok.", training_activity(), {})
    activity = training_activity()
    if not activity["active"] and not activity["starting"] and not activity.get("heavy"):
        return evaluate(
            tag,
            activity=activity,
            ps_entries=[],
            ram_available_gb=None,
            gpu_mem=None,
            footprint=read_footprint(tag),
            margin_gb=s.chat_ram_margin_gb,
        )
    from app.feedback.model_identity import match_entry, ollama_tags

    footprint = read_footprint(tag)
    # Etiket yeniden oluşturulduysa (farklı digest) eski ölçüm bu modele ait değildir.
    current = match_entry(ollama_tags(transport), tag)
    fp_digest = (footprint or {}).get("digest")
    cur_digest = (current or {}).get("digest")
    if fp_digest and cur_digest and fp_digest != cur_digest:
        footprint = None
    mem = _memory_info()
    return evaluate(
        tag,
        activity=activity,
        ps_entries=ollama_ps(transport),
        ram_available_gb=mem.ram_available_gb or None,
        gpu_mem=_nvidia_smi_memory_gb(),
        footprint=footprint,
        margin_gb=s.chat_ram_margin_gb,
        unified_memory=platform.system() == "Darwin",
    )
