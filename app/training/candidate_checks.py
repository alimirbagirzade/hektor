"""candidate_checks.py — aday adapter'ın GERÇEKTEN tamamlandığı ve dönüşümünün bütün olduğu (2D).

``run_complete.json`` tek başına kanıt DEĞİLDİR (yalnız adım sayısı + zaman taşır). Tamamlanma
için hepsi gerekir:
- süreç sonucu: ``train_status.json`` bu adapter'a ait, ``finished_at`` var, ``failed_at`` yok,
  süreç artık yaşamıyor;
- adım: ``run_complete.global_step == max_steps`` ve ``run_plan.json`` hedefiyle aynı;
- adapter dosyaları: ``adapter_config.json`` + ``adapter_model.safetensors`` var (özetleri
  kaydedilir);
- temel model kökeni: ``adapter_config.base_model_name_or_path`` reçetedeki temel modelle aynı;
- veri özeti: koşunun ``data_sha256``'sı reçeteyle aynı;
- koşu kimliği: koşunun onay kimliği, kolay akışın başlatma kaydındakiyle aynı.
- onay kaydı: koşunun onay kimliği onay deposunda var, İNSAN tarafından onaylanmış, eğitim
  eylemine ait ve TÜKETİLMİŞ (koşu sahte bir kimlikle başlatılamaz).

Kolay akış dışında (``start-train.ps1`` / doğrudan CLI) eğitilen aday için reçete, birbirinden
bağımsız kayıtlardan kurulur (``recipe_from_records``): kayıt defteri girdisi (temel model),
onay deposu (koşu kimliği), ağırlık kararının tüketim kaydı ve veri özeti — yalnız diskteki
``lora_sft.jsonl`` bugün yeniden özetlendiğinde koşu özetiyle AYNIYSA doğrulanmış sayılır.

Dönüşüm (``adapter_to_ollama.ps1``) için: birleştirme kanıtı (``merge_info.json``: adapter özeti
eşleşiyor, KL kapısı geçti, ilk-token aynı), GGUF köken anahtarları adapter özetini taşıyor,
yarım (``.partial``) çıktı yok, Modelfile başlığı + FROM satırı doğru dosyayı gösteriyor, Ollama
etiketi var (digest kaydedilir). Yarım kalmış çıktı başarılı SAYILMAZ.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import get_settings


def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _result(checks: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"ok": all(c["ok"] for c in checks), "checks": checks, **extra}


def verify_run_completion(adapter: str, recipe: dict[str, Any] | None = None) -> dict[str, Any]:
    from app.training.detached_launch import _pid_alive, read_detached_training_status

    s = get_settings()
    d = s.adapters_dir / Path(adapter).name
    checks: list[dict[str, Any]] = []

    def add(key: str, ok: bool, detail: str) -> None:
        checks.append({"key": key, "ok": bool(ok), "detail": detail})

    st = read_detached_training_status(s.root)
    mine = st.get("adapter") == adapter
    add(
        "surec_sonucu",
        mine and bool(st.get("finished_at")) and not st.get("failed_at"),
        f"train_status: adapter={st.get('adapter')} finished={st.get('finished_at')} "
        f"failed={st.get('failed_at')}",
    )
    pid = st.get("pid")
    alive = isinstance(pid, int) and _pid_alive(pid)
    add("surec_bitti", mine and not alive, f"pid {pid}")
    done = _read(d / "run_complete.json") or {}
    plan = _read(d / "run_plan.json") or {}
    target = plan.get("max_steps") or plan.get("target_steps") or done.get("max_steps")
    add(
        "adim",
        bool(done) and done.get("global_step") == done.get("max_steps") == target,
        f"run_complete {done.get('global_step')}/{done.get('max_steps')}, plan {target}",
    )
    cfg = _read(d / "adapter_config.json") or {}
    weights = d / "adapter_model.safetensors"
    w_sha = sha256_file(weights) if weights.is_file() else ""
    add(
        "adapter_dosyalari",
        bool(cfg) and bool(w_sha),
        f"adapter_config {'var' if cfg else 'YOK'}, ağırlık {w_sha[:12] or 'YOK'}",
    )
    base = str(cfg.get("base_model_name_or_path") or "")
    want_base = str((recipe or {}).get("base_model") or "")
    add(
        "temel_model",
        bool(base)
        and (not want_base or Path(base).name == Path(want_base).name or base == want_base),
        f"adapter_config {base or '?'} · reçete {want_base or '(reçete yok)'}",
    )
    want_data = str((recipe or {}).get("data_sha256") or "")
    add(
        "veri_ozeti",
        bool(want_data) and st.get("data_sha256") == want_data,
        f"koşu {str(st.get('data_sha256'))[:12]} · reçete {want_data[:12] or '(reçete yok)'}",
    )
    want_apr = str((recipe or {}).get("approval_id") or "")
    add(
        "kosu_kimligi",
        bool(want_apr) and st.get("approval_id") == want_apr,
        f"koşu onayı {st.get('approval_id')} · başlatma kaydı {want_apr or '(yok)'}",
    )
    appr = (_approval_record(str(st.get("approval_id") or "")) if mine else None) or {}
    add(
        "onay_kaydi",
        bool(appr)
        and appr.get("status") == "approved"
        and str(appr.get("action", "")).startswith("train_run")
        and bool(appr.get("consumed_at"))
        and str(appr.get("decided_by") or "") not in ("", "driver"),
        "onay deposunda "
        + (
            f"{appr.get('approval_id')} durum={appr.get('status')} eylem={appr.get('action')} "
            f"karar={appr.get('decided_by')} tüketildi={appr.get('consumed_at') or 'HAYIR'}"
            if appr
            else "kayıt YOK"
        ),
    )
    extra = {k: v for k, v in (recipe or {}).items() if k in ("source", "evidence")}
    return _result(checks, adapter=adapter, adapter_sha256=w_sha, base_model=base, **extra)


def _approval_record(approval_id: str) -> dict[str, Any] | None:
    if not approval_id:
        return None
    try:
        from app.agents.runtime import approvals

        a = approvals.get_approval(approval_id)
    except Exception:
        return None
    if a is None:
        return None
    return {
        "approval_id": a.approval_id,
        "status": getattr(a.status, "value", str(a.status)),
        "action": a.action,
        "agent_id": a.agent_id,
        "decided_by": getattr(a, "decided_by", None),
        "consumed_at": a.consumed_at,
    }


def recipe_from_records(adapter: str) -> dict[str, Any] | None:
    """Kolay akış kaydı olmayan (start-train/CLI) koşu için bağımsız kayıtlardan reçete."""
    from app.training.detached_launch import _combined_source, read_detached_training_status

    s = get_settings()
    st = read_detached_training_status(s.root)
    if st.get("adapter") != adapter:
        return None
    evidence: list[str] = []
    base = ""
    reg = s.root / "registry" / "adapters" / "registry.jsonl"
    if reg.is_file():
        for line in reg.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("adapter_name") == adapter:
                base = str(row.get("base_model") or "")
                evidence.append(
                    f"kayıt defteri {row.get('adapter_id')} ({row.get('status')}): temel {base}, "
                    f"{row.get('train_examples')} örnek"
                )
    data_sha = ""
    src = _combined_source(s)
    now_sha = sha256_file(src) if src.is_file() else ""
    if st.get("data_sha256") and now_sha == st.get("data_sha256"):
        data_sha = now_sha
        evidence.append(f"lora_sft.jsonl bugün yeniden özetlendi = koşu özeti ({now_sha[:12]}…)")
    else:
        run_sha = str(st.get("data_sha256"))[:12]
        evidence.append(
            f"lora_sft.jsonl şu an {now_sha[:12] or 'YOK'}… ≠ koşu {run_sha}… — eğitim verisi "
            "bağımsız doğrulanamadı"
        )
    wd = s.root / "registry" / "lora" / "weight_decisions.jsonl"
    if st.get("mix_decision_id") and wd.is_file():
        for line in wd.read_text(encoding="utf-8").splitlines():
            if st["mix_decision_id"] in line:
                with_row = json.loads(line)
                evidence.append(
                    f"ağırlık kararı {with_row.get('decision_id')} tüketen "
                    f"{with_row.get('consumed_by')} @ {with_row.get('consumed_at')}"
                )
    return {
        "base_model": base or s.peft_base_model,
        "data_sha256": data_sha,
        "approval_id": str(st.get("approval_id") or ""),
        "source": "kayıtlar (start-train.ps1 / CLI; kolay akış anlık görüntüsü yok)",
        "evidence": evidence,
    }


def recipe_for_adapter(adapter: str) -> dict[str, Any] | None:
    """Kolay akışın başlatma kaydı (launch.json) + reçetesi; yoksa None."""
    from app.training.easy_train import snapshots_dir

    d = snapshots_dir()
    if not d.is_dir():
        return None
    for p in sorted(d.glob("snap_*/launch.json")):
        launch = _read(p) or {}
        recipe = _read(p.parent / "recipe.json") or {}
        if recipe.get("adapter_name") == adapter and launch.get("ok"):
            return {
                **recipe,
                "approval_id": launch.get("approval_id", ""),
                "source": "kolay akış anlık görüntüsü + başlatma kaydı",
            }
    return recipe_from_records(adapter)


def verify_conversion(
    adapter: str, ollama_tag: str, *, tags: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    from app.feedback.model_identity import match_entry, model_origin, ollama_tags

    s = get_settings()
    checks: list[dict[str, Any]] = []

    def add(key: str, ok: bool, detail: str) -> None:
        checks.append({"key": key, "ok": bool(ok), "detail": detail})

    weights = s.adapters_dir / Path(adapter).name / "adapter_model.safetensors"
    w_sha = sha256_file(weights) if weights.is_file() else ""
    mi = _read(s.root / "models" / "merged" / Path(adapter).name / "merge_info.json") or {}
    kl = mi.get("kl_peft_vs_merged")
    gate = mi.get("kl_gate")
    add(
        "birlestirme",
        bool(mi)
        and mi.get("adapter_sha256") == w_sha
        and w_sha != ""
        and isinstance(kl, int | float)
        and isinstance(gate, int | float)
        and kl <= gate
        and bool(mi.get("top_same")),
        f"adapter {str(mi.get('adapter_sha256'))[:12]} ↔ {w_sha[:12] or 'YOK'}, KL {kl} ≤ {gate}, "
        f"ilk-token aynı {mi.get('top_same')}",
    )
    gguf = s.root / "models" / "gguf"
    partial = sorted(p.name for p in gguf.glob(f"{adapter}*.partial")) if gguf.is_dir() else []
    add("yarim_cikti_yok", not partial, ", ".join(partial) or "yok")
    srcs = sorted(gguf.glob(f"{adapter}-*.gguf.src")) if gguf.is_dir() else []
    keys = [p.read_text(encoding="utf-8").strip() for p in srcs]
    add(
        "gguf_koken",
        bool(keys) and all(k.startswith(f"adapter:{w_sha}") for k in keys),
        f"{len(keys)} köken anahtarı",
    )
    origin = model_origin(ollama_tag)
    mf = gguf / f"Modelfile.{ollama_tag.removesuffix(':latest')}"
    from_ok = False
    if mf.is_file():
        from_line = next(
            (ln for ln in mf.read_text(encoding="utf-8").splitlines() if ln.startswith("FROM ")),
            "",
        )
        from_ok = Path(from_line[5:].strip()).name.startswith(f"{adapter}-")
    add(
        "modelfile",
        origin is not None and origin.get("adapter") == adapter and from_ok,
        f"köken {origin.get('adapter') if origin else 'YOK'}, FROM doğru {from_ok}",
    )
    listing = tags if tags is not None else ollama_tags()
    entry = match_entry(listing, ollama_tag) if listing is not None else None
    digest = str((entry or {}).get("digest") or "").removeprefix("sha256:")
    add("ollama", bool(digest), f"etiket {ollama_tag} digest {digest[:12] or 'YOK'}")
    return _result(
        checks,
        adapter=adapter,
        ollama_tag=ollama_tag,
        digest=digest,
        adapter_sha256=w_sha,
        merge_params={"kl": kl, "kl_gate": gate},
    )
