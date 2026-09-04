"""Orkestrasyon web uçları — dayanıklı eğitim hattı (tek-tık + canlı izleme).

`server.py`'ye TEK satır (`include_router`) ile bağlanır → sıcak dosya minimal dokunulur.
Uçlar `TrainingOrchestrator`'ı sürer: salt-okuma aşamaları gözetimsiz yürür; insan
kapılarında (deep-hunt, approval) DURUR. Gerçek eğitim ASLA gözetimsiz başlamaz (Kural 8) —
tehlikeli train/eval/registry aşamaları varsayılan HANDOFF'tur.

Kimlik doğrulama: `require_auth` (token boşsa lokal-açık, mevcut davranışla aynı).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.web.security import require_auth, require_human

router = APIRouter(
    prefix="/api/orchestration", tags=["orchestration"], dependencies=[Depends(require_auth)]
)


class OrchestrationStartRequest(BaseModel):
    model: str = Field(default="", max_length=128)  # boş → ayardan (peft_base_model)
    # profile de adapter_name ile aynı disiplinde kalıplı: dry-run komut dizesine gömülür,
    # serbest-metin (boşluk/shell metakarakteri) yaniltici/kopyalaninca zararli olmasin.
    profile: str = Field(default="discipline_safe_local", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # adapter_name detached_launch._ADAPTER_RE ile aynı kalıp (yol-geçişi savunması, erken ret)
    adapter_name: str = Field(default="hektor_lora", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    iters: int = Field(default=300, ge=1, le=100000)  # üst sınır: anlamsız dev değerleri ele
    hunt_ack: bool = False  # Kademe-2 derin av tamamlandı mı (ZORUNLU gate)
    auto_run: bool = True  # başlattıktan sonra blocked olana dek ilerlet (tek-tık)


class OrchestrationResumeRequest(BaseModel):
    hunt_ack: bool = False


class OrchestrationRecoverRequest(BaseModel):
    timeout_min: float = Field(default=30.0, ge=1.0, le=100000.0)


def _orchestrator() -> Any:
    from app.orchestration.orchestrator import TrainingOrchestrator

    return TrainingOrchestrator()


@router.post("/start")
def orchestration_start(req: OrchestrationStartRequest, request: Request) -> dict[str, Any]:
    """Dayanıklı eğitim orkestrasyonunu BAŞLAT (tek-tık).

    `auto_run=true` (varsayılan): salt-okuma aşamalarını yürütür, insan kapısında durur.
    Gerçek eğitim için `hunt_ack=true` + ayrı taze onay (Kural 8) gerekir.

    `hunt_ack=true` bir İNSAN yetki beyanıdır ("Kademe-2 derin avı yaptım") ve
    doğrulanmadan kabul edilir → sürücü scope'una KAPALI. Aksi halde motor, zorunlu
    eğitim-öncesi avı basitçe `{"hunt_ack": true}` göndererek atlayabilirdi (v5
    regresyonunun kök nedeni tam olarak buydu). Başlatmanın kendisi serbesttir.
    """
    from app.config import get_settings

    if req.hunt_ack:
        require_human(request)

    orch = _orchestrator()
    model = req.model or getattr(get_settings(), "peft_base_model", "")
    run_id = orch.start(
        model=model,
        profile=req.profile,
        adapter_name=req.adapter_name,
        params={"iters": req.iters, "hunt_ack": req.hunt_ack},
    )
    snap = orch.run_until_blocked(run_id) if req.auto_run else orch.status(run_id)
    return {"run_id": run_id, **snap}


@router.get("/status/{run_id}")
def orchestration_status(run_id: str) -> dict[str, Any]:
    """Koşunun aşama durumunu döndür (dashboard polling)."""
    orch = _orchestrator()
    snap = orch.status(run_id)
    if snap.get("run") is None:
        raise HTTPException(status_code=404, detail=f"Koşu bulunamadı: {run_id}")
    from app.orchestration import engine_procs

    snap["driver_running"] = engine_procs.is_run_live(run_id)
    return snap


@router.get("/timeline/{run_id}")
def orchestration_timeline(run_id: str, limit: int = 200) -> dict[str, Any]:
    """Koşunun olay zaman çizelgesini döndür."""
    orch = _orchestrator()
    if orch.store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Koşu bulunamadı: {run_id}")
    return {"run_id": run_id, "events": orch.timeline(run_id, limit=limit)}


@router.post("/resume/{run_id}")
def orchestration_resume(
    run_id: str, req: OrchestrationResumeRequest, request: Request
) -> dict[str, Any]:
    """Bloke/başarısız koşuyu sürdür — tamamlanan aşamalar atlanır (checkpoint).

    `hunt_ack=true` insan yetki beyanıdır → sürücüye kapalı (bkz. `orchestration_start`).
    """
    import json

    if req.hunt_ack:
        require_human(request)

    orch = _orchestrator()
    run = orch.store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Koşu bulunamadı: {run_id}")
    if req.hunt_ack:
        params = dict(run.get("params") or {})
        params["hunt_ack"] = True
        orch.store.update_run(run_id, params_json=json.dumps(params, ensure_ascii=False))
    return orch.run_until_blocked(run_id)


@router.get("/runs")
def orchestration_runs(limit: int = 20) -> dict[str, Any]:
    """Son orkestrasyon koşularını listele."""
    from app.orchestration import engine_procs

    orch = _orchestrator()
    runs = orch.list_runs(limit=limit)
    for run in runs:
        run["driver_running"] = engine_procs.is_run_live(str(run.get("run_id", "")))
    return {"runs": runs}


@router.post("/recover")
def orchestration_recover(req: OrchestrationRecoverRequest) -> dict[str, Any]:
    """Panic recovery — kalp atışı durmuş 'running' aşamaları failed'a çevir."""
    orch = _orchestrator()
    return {"recovered": orch.recover_stale(timeout_min=req.timeout_min)}


class OrchestrationAutodriveRequest(BaseModel):
    execute: bool = False  # True → gerçek motor spawn (abonelik); False → dry-run komut
    # Boş → kayıt tablosundaki varsayılan. Kalıp erken-ret içindir; asıl doğrulama
    # `engines.get_engine` (bilinmeyen ad → ValueError) + `run_blocked_reason`.
    engine: str = Field(default="", pattern=r"^[A-Za-z0-9_-]{0,32}$")
    # ⚡ RUN'ın ANA yolu SÜR (drive) modudur: motor MCP araçlarıyla veri hattını ilerletir.
    # "hunt" AV modunu tetikler (zorunlu Kademe-2 derin av, salt rapor, MCP KAPALI) — av
    # ayrı bir tetikleyici olarak KORUNUR (Kademe-2 taraması için hâlâ gerekli).
    mode: str = Field(default="drive", pattern=r"^(drive|hunt)$")


def _run_autodrive_bg(run_id: str, engine: str, mode: str) -> None:
    """Arka plan: gerçek otonom sürüş (uzun sürer). Olaylar timeline'a yazılır."""
    from app.orchestration.driver import AutoDriver

    AutoDriver().drive(run_id, execute=True, engine=engine, mode=mode)


# ⛔ `human_only`: bu uç gerçek bir `claude -p` ALT-SÜRECİ doğurur. Sür (drive) modunda
# motora MCP erişimi verildiği için, kapı olmasaydı motor bu ucu MCP aracı olarak görüp
# KENDİ KENDİNE yeni sürücüler doğurabilirdi (özyinelemeli spawn + abonelik kotası yakma).
# İnsan (UI/CLI) etkilenmez — yalnız `driver` scope 403 alır (Kural 8; /training/run ile
# aynı desen). Bkz. docs/SCOPE_ISOLATION.md.
@router.post("/autodrive/{run_id}", dependencies=[Depends(require_human)])
def orchestration_autodrive(
    run_id: str, req: OrchestrationAutodriveRequest, background: BackgroundTasks
) -> dict[str, Any]:
    """⚡ RUN — headless `claude -p` (abonelik) ile OTONOM sür → veri hattını ilerlet.

    `mode="drive"` (varsayılan, ⚡ RUN): motora MCP araçları verilir, veri hattı ilerletilir.
    `mode="hunt"`: zorunlu Kademe-2 derin av (salt rapor, MCP kapalı) — ayrı tetikleyici.

    `execute=false` (varsayılan): DRY-RUN — çalıştırılacak komutu döner, spawn YOK.
    `execute=true`: gerçek sürüş ARKA PLANDA başlar (timeline'dan izlenir); yanıt hemen döner.
    Gerçek eğitim yine TAZE insan onayı bekler (Kural 8 — sür promptu eğitimi başlatmaz;
    onay/eğitim uçları sürücü kimliğiyle 403 alır).
    """
    from app.orchestration import engines
    from app.orchestration.driver import AutoDriver

    orch = _orchestrator()
    if orch.store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Koşu bulunamadı: {run_id}")

    # Motor adını çöz — bilinmeyen ad sessizce varsayılana DÜŞMEZ (yanlış motoru
    # doğurmaktansa 400 vermek doğru; `engines.get_engine` ile aynı disiplin).
    engine_name = req.engine or engines.DEFAULT_ENGINE
    try:
        engines.get_engine(engine_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not req.execute:
        return AutoDriver(orchestrator=orch).drive(
            run_id, execute=False, engine=engine_name, mode=req.mode
        )

    # ⛔ SUNUCU TARAFI KAPI: kurulu-değil / süreç-başlatmayan / sertleştirilemeyen motor
    # REDDEDİLİR. UI bu motorları zaten gri gösterir, ama gri buton bir güvenlik sınırı
    # DEĞİLDİR — istek elle (curl/MCP) atılabilir. Tek doğruluk kaynağı
    # `engines.run_blocked_reason`; UI ile uç aynı fonksiyonu kullanır (fail-closed).
    blocked = engines.run_blocked_reason(engine_name)
    if blocked:
        raise HTTPException(status_code=503, detail=blocked)
    # Sür modu MCP erişimli SERTLEŞTİRİLMİŞ şablon ister. `run_blocked_reason` yalnız av
    # profilini (`hardened`) kontrol eder; sür profili AYRI bayrak (`drive_hardened`) — bu
    # yüzden burada AÇIKÇA doğrulanır. Aksi halde "av'da sertleşmiş ama sür'de değil" bir
    # motor (bugün yok, gelecekte olabilir) `run_blocked_reason`'ı geçip arka planda sessizce
    # reddedilirken HTTP yanıtı yanıltıcı "autodrive_started" derdi. Fail-closed + dürüst yanıt.
    if req.mode == "drive":
        eng = engines.get_engine(engine_name)
        if not engines.drive_supported(engine_name) or not eng.drive_hardened:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{eng.label} sür (drive) modunda sertleştirilemiyor — MCP erişimli "
                    "sertleştirilmiş şablonu yok (Kural 8; docs/SCOPE_ISOLATION.md)."
                ),
            )

    background.add_task(_run_autodrive_bg, run_id, engine_name, req.mode)
    return {
        "ok": True,
        "status": "autodrive_started",
        "engine": engine_name,
        "mode": req.mode,
        "message": "Otonom sürüş arka planda başladı; ilerlemeyi timeline'dan izle.",
        "run_id": run_id,
    }
