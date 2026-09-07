"""delegates.py — orkestrasyon aşamalarının GERÇEK fonksiyonlara bağlanması.

Her delege `RunContext` alır, `StageResult` döner. İç-bağımlılıklar SAVUNMACI import
edilir (eksikse aşama makul biçimde blocked/skipped olur, koşu çökmez).

Güvenlik sınırı (CLAUDE.md Kural 8):
  - Salt-okuma aşamaları (preflight/data-gate/curriculum/dry-run) GERÇEK çalışır.
  - `deep-hunt` zorunlu Kademe-2 avı temsil eder → hunt_ack olmadan blocked.
  - `approval` TAZE insan onayını yalnız GÖZLER (tüketmez, üretmez) → onaysız blocked.
  - `train`/`evaluate`/`registry` varsayılan olarak HANDOFF'tur (gerçek detached
    yürütme bilinçli olarak gözetimsiz BAŞLATILMAZ; web tek-tık akışı ayrı, onay-kapılı
    bir "yürüten delege" seti enjekte ederek devralır).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.orchestration.pipeline import StageStatus

if TYPE_CHECKING:
    from app.orchestration.orchestrator import RunContext, StageDelegate, StageResult

log = logging.getLogger(__name__)

_SFT_REL = ("data", "lora_sft", "lora_sft.jsonl")

# Gerçek `train --run` yolunun kullandığı onay anahtarı — TEK anahtar, iki yüzey.
_APPROVAL_AGENT_ID = "lora-trainer"
_APPROVAL_ACTION = "train_run"
_APPROVAL_KEY = f"{_APPROVAL_AGENT_ID}/{_APPROVAL_ACTION}"


def _result(status: StageStatus, message: str, output: dict | None = None) -> StageResult:
    # Lazy import — orchestrator ↔ delegates döngüsünü kırar.
    from app.orchestration.orchestrator import StageResult as _SR

    return _SR(status=status, message=message, output=output or {})


def _deps_available() -> bool:
    """torch + transformers + peft kurulu mu (gerçek eğitim/eval için)."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft"))


def _count_sft_lines() -> tuple[int, Path]:
    from app.config import get_settings

    settings = get_settings()
    jsonl = settings.root.joinpath(*_SFT_REL)
    if not jsonl.exists():
        return 0, jsonl
    n = sum(1 for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip())
    return n, jsonl


# ── aşama delegeleri ─────────────────────────────────────────────────────────


def preflight(ctx: RunContext) -> StageResult:
    """STOP_ALL + veri hazırlığı + bağımlılık durumu — salt-okuma."""
    out: dict = {}
    stop_all = False
    try:
        from app.agents.runtime import supervisor

        stop_all = bool(supervisor.is_stop_all_active())
    except Exception as exc:  # supervisor yüklenemese bile koşu çökmesin
        out["supervisor_error"] = str(exc)
    out["stop_all"] = stop_all

    try:
        from app.training.detached_launch import readiness

        r = readiness()
        out["ready"] = bool(r.get("ready"))
        out["n_examples"] = int(r.get("ready_examples", 0))
        out["ready_label"] = r.get("ready_label", "")
    except Exception as exc:
        out["readiness_error"] = str(exc)

    out["train_deps_ok"] = _deps_available()

    if stop_all:
        return _result(StageStatus.blocked, "STOP_ALL aktif — orkestrasyon durdu.", out)
    return _result(StageStatus.completed, out.get("ready_label", "ön kontrol tamam"), out)


def collision(ctx: RunContext) -> StageResult:
    """Eşzamanlı oturum/worktree çakışması taraması (git durumu) — salt-okuma.

    git yoksa skipped (hat durmaz); kirli ağaç → completed (uyarı event'i); aktif lock /
    aynı-branch worktree / HEAD kayması → blocked (insan çözmeli, Kural 8). Çakışma 'fail'i
    failed DEĞİL blocked'a eşlenir: kurtarılabilir (commit/stash/çöz → resume).
    """
    try:
        from app.orchestration.collision import CollisionDetector
    except Exception as exc:  # modül yüklenemese bile koşu çökmesin
        return _result(StageStatus.skipped, f"Çakışma modülü yüklenemedi: {exc}", {})

    baseline_head = str(ctx.params.get("baseline_head", "") or "")
    result = CollisionDetector(baseline_head=baseline_head or None).run()
    status_map = {
        "pass": StageStatus.completed,
        "warn": StageStatus.completed,  # uyarı hattı durdurmaz
        "skip": StageStatus.skipped,
        "fail": StageStatus.blocked,  # çakışma insan eylemi bekler (kurtarılabilir)
    }
    return _result(
        status_map.get(result.verdict, StageStatus.skipped), result.summary, result.to_dict()
    )


def smoke(ctx: RunContext) -> StageResult:
    """Gerçek runtime uçtan-uca duman testi ("stub≠runtime").

    Çevrimdışıysa skipped (hat durmaz; sonraki insan kapısına geçer), canlı+sağlıklıysa
    completed, canlı ama üretim boş/degenere/hata ise failed. Salt-okuma (üretim atılır).
    """
    try:
        from app.orchestration.smoke import SmokeRunner
    except Exception as exc:  # modül yüklenemese bile koşu çökmesin
        return _result(StageStatus.skipped, f"Smoke modülü yüklenemedi: {exc}", {})

    result = SmokeRunner().run()
    status_map = {
        "pass": StageStatus.completed,
        "skip": StageStatus.skipped,
        "fail": StageStatus.failed,
    }
    return _result(
        status_map.get(result.verdict, StageStatus.skipped), result.summary, result.to_dict()
    )


def deep_hunt(ctx: RunContext) -> StageResult:
    """Eğitim öncesi ZORUNLU Kademe-2 derin av (CLAUDE.md). hunt_ack olmadan blocked."""
    if bool(ctx.params.get("hunt_ack")):
        return _result(
            StageStatus.completed,
            "Kademe-2 derin av onaylandı (hunt_ack=true).",
            {"hunt_ack": True},
        )
    return _result(
        StageStatus.blocked,
        (
            "ZORUNLU: eğitim öncesi Kademe-2 derin adversarial bug-avı çalıştırılmalı "
            "(v5 regresyonu bu yüzden olmuştu). Tamamlanınca hunt_ack=true ile sürdür."
        ),
        {"hunt_ack": False},
    )


def data_gate(ctx: RunContext) -> StageResult:
    """pretrain-gate GO/NO-GO kalite kapısı — salt-okuma (LLM'siz)."""
    n, jsonl = _count_sft_lines()
    if n == 0:
        return _result(
            StageStatus.failed,
            f"SFT verisi yok veya boş: {jsonl.name}",
            {"exists": jsonl.exists(), "n_lines": 0},
        )
    try:
        from app.training.dataset_quality import audit_dataset
        from app.training.discipline_dataset import discipline_jsonl_lines

        lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        report = audit_dataset(lines, discipline_lines=discipline_jsonl_lines())
        out = report.to_dict()
        if report.verdict == "GO":
            return _result(
                StageStatus.completed,
                f"GO — {report.total} örnek, öneri {report.recommended_epochs} epoch.",
                out,
            )
        reason = "; ".join(report.blockers) or "kalite kapısı NO-GO"
        return _result(StageStatus.blocked, f"NO-GO: {reason}", out)
    except Exception as exc:
        return _result(StageStatus.failed, f"Kalite kapısı hatası: {exc}", {"n_lines": n})


def curriculum(ctx: RunContext) -> StageResult:
    """Müfredat seviyelendirme şeması (L0-L4) hazır mı — hafif, salt-okuma."""
    try:
        from app.lora.curriculum import LEVEL_BOUNDS

        return _result(
            StageStatus.completed,
            f"Müfredat şeması hazır — {len(LEVEL_BOUNDS)} seviye (L0-L4).",
            {"levels": list(LEVEL_BOUNDS.keys())},
        )
    except Exception as exc:
        return _result(StageStatus.skipped, f"Müfredat modülü yüklenemedi: {exc}", {})


def dry_run(ctx: RunContext) -> StageResult:
    """Eğitim komutu + örnek sayısı önizleme — gerçek yürütme/yazma YOK."""
    from app.config import get_settings

    settings = get_settings()
    model = ctx.run.get("model", "") or getattr(settings, "peft_base_model", "")
    profile = ctx.run.get("profile", "") or "discipline_safe_local"
    adapter = ctx.run.get("adapter_name", "") or "hektor_lora"
    iters = int(ctx.params.get("iters", 0) or 0)
    n, _ = _count_sft_lines()
    cmd = f"hektor train --run --backend peft --profile {profile} --adapter {adapter}"
    out = {
        "command": cmd,
        "model": model,
        "profile": profile,
        "adapter": adapter,
        "n_examples": n,
        "iters": iters,
    }
    return _result(StageStatus.completed, f"Komut hazır: {cmd}", out)


def regression(ctx: RunContext) -> StageResult:
    """Eğitim/onay öncesi gerileme kapısı (v5 dersi) — salt-okuma.

    Mevcut aday veri setinin v5-ilgili kalite sinyallerini son GEÇEN baseline ile kıyaslar.
    Baseline yoksa skipped (ilk koşu, kıyas yok); gerileme yoksa completed; gerileme varsa
    blocked (insan araştırmadan eğitim ilerlemesin). Baseline oto-güncellenmez (--commit ile).
    """
    try:
        from app.orchestration.regression import RegressionGuard
    except Exception as exc:  # modül yüklenemese bile koşu çökmesin
        return _result(StageStatus.skipped, f"Gerileme modülü yüklenemedi: {exc}", {})

    result = RegressionGuard().run()
    status_map = {
        "pass": StageStatus.completed,
        "skip": StageStatus.skipped,
        "fail": StageStatus.blocked,  # gerileme insan araştırması bekler (Kural 8)
    }
    return _result(
        status_map.get(result.verdict, StageStatus.skipped), result.summary, result.to_dict()
    )


def approval(ctx: RunContext) -> StageResult:
    """Gerçek eğitim için TAZE onay kapısı (Kural 8) — onayı TÜKETMEZ, yalnız gözler.

    Varsayılan `train` delegesi handoff'tur (gerçek eğitimi inline başlatmaz). Onayı burada
    `authorize_training_action` → `require_fresh_approval` ile TÜKETMEK iki sorun doğurur:
    (1) hiçbir gerçek eğitime karşılık gelmeyen onayı boşa harcar (tek-kullanımlık onay
    sözleşmesi ihlali) ve mesaj "eğitim yetkili" diyerek yanıltır; (2) taze onay yokken her
    başarısız resume YENİ bir PENDING onay üretip biriktirir. Bu yüzden bu aşama gerçek
    `train --run` yolunun kullandığı AYNI anahtarı (lora-trainer/train_run) TÜKETMEDEN
    gözler (`has_fresh_approval`): onay o yolda oluşturulur ve gerçek eğitim noktasında
    tüketilir. Böylece tek onay her iki yolu da yetkilendirir ve onay boşa harcanmaz.

    Politika sırası TEK OTORİTE (`unattended_policy.authorize_training_action`) ile aynıdır —
    STOP_ALL > gözetimsiz mod (gate'ler geçti) > taze insan onayı — ama tüketen çağrı hiçbir
    dalda yapılmaz. Kapı GEVŞETİLMEZ: otomatik yetkilendirme yoktur; onay yoksa blocked.
    """
    try:
        from app.agents.runtime import approvals, supervisor
        from app.config import get_settings
    except Exception as exc:
        return _result(StageStatus.failed, f"Onay altyapısı yüklenemedi: {exc}", {})

    out: dict = {"approval_key": _APPROVAL_KEY}
    try:
        if supervisor.is_stop_all_active():
            out["stop_all"] = True
            return _result(
                StageStatus.blocked, "STOP_ALL aktif — eğitim yetkisi verilmez (Kural 8).", out
            )
        if bool(get_settings().unattended_training_enabled):
            out["authorization_mode"] = "unattended_policy"
            return _result(
                StageStatus.completed,
                "Eğitim yetkisi verildi (unattended_policy) — train aşaması devralabilir.",
                out,
            )
        # Salt-okuma: taze (onaylı + tüketilmemiş) onay VAR MI — TÜKETMEZ, yenisini ÜRETMEZ.
        has_fresh = bool(approvals.has_fresh_approval(_APPROVAL_AGENT_ID, _APPROVAL_ACTION))
    except Exception as exc:
        return _result(StageStatus.failed, f"Onay durumu okunamadı: {exc}", out)

    if has_fresh:
        out["authorization_mode"] = "human_approval"
        out["approval_consumed"] = False
        return _result(
            StageStatus.completed,
            (
                f"Taze insan onayı mevcut ({_APPROVAL_KEY}) — TÜKETİLMEDİ; gerçek "
                "'train --run' noktasında tüketilecek."
            ),
            out,
        )
    out["needs_approval"] = True
    out["approval_id"] = ""  # bu aşama onay ÜRETMEZ (pending birikmesin)
    return _result(
        StageStatus.blocked,
        (
            "Gerçek eğitim TAZE insan onayı gerektirir (Kural 8). Bu aşama onay ÜRETMEZ ve "
            "TÜKETMEZ. Onay akışını 'hektor train --run' (veya web'deki eğitim butonu) "
            f"başlatır: {_APPROVAL_KEY} anahtarıyla bir onay oluşturur, 'hektor "
            "approval-approve <id>' ile onaylarsın ve onay gerçek eğitim noktasında "
            "tüketilir. Onaydan sonra orkestrasyonu sürdür."
        ),
        out,
    )


def train_handoff(ctx: RunContext) -> StageResult:
    """Gerçek eğitimi gözetimsiz BAŞLATMAZ — detached devir talimatı verir (Kural 8)."""
    adapter = ctx.run.get("adapter_name", "") or "hektor_lora"
    iters = int(ctx.params.get("iters", 300) or 300)
    cmd = f"hektor train --run --adapter {adapter}"
    from app.config import get_settings

    unattended = get_settings().unattended_training_enabled
    return _result(
        StageStatus.completed if unattended else StageStatus.blocked,
        (
            "Eğitim yürütmesi tek sahip Auto-LoRA denetleyicisine devredildi."
            if unattended
            else f"Detached eğitimi elle başlat: {cmd}"
        ),
        {
            "handoff": True,
            "owner": "auto-lora-pipeline",
            "command": cmd,
            "adapter": adapter,
            "iters": iters,
        },
    )


def eval_handoff(ctx: RunContext) -> StageResult:
    """Eval, eğitim devri sonrası auto_pipeline tarafından yürütülür (handoff)."""
    return _result(
        StageStatus.skipped,
        "Eval, eğitim devri sonrası auto_pipeline._run_eval tarafından yürütülür.",
        {"handoff": True},
    )


def registry_handoff(ctx: RunContext) -> StageResult:
    """Kayıt, eval sonrası auto_pipeline tarafından yürütülür (handoff)."""
    return _result(
        StageStatus.skipped,
        "Adapter kaydı, eval sonrası auto_pipeline._register_adapter tarafından yapılır.",
        {"handoff": True},
    )


def default_delegates() -> dict[str, StageDelegate]:
    """Üretim varsayılanı: salt-okuma aşamaları gerçek; tehlikeli tail handoff."""
    return {
        "preflight": preflight,
        "collision": collision,
        "smoke": smoke,
        "deep-hunt": deep_hunt,
        "data-gate": data_gate,
        "curriculum": curriculum,
        "dry-run": dry_run,
        "regression": regression,
        "approval": approval,
        "train": train_handoff,
        "evaluate": eval_handoff,
        "registry": registry_handoff,
    }
