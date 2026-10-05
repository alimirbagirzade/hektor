"""easy_train.py — kolay ve bağlı eğitim akışı (Faz 2C). Eğitimi KENDİLİĞİNDEN BAŞLATMAZ.

Akış (her adım ayrı ve açık):
1. ``state()``            : son kullanılan ayarlar (profil, karışım, temel model, adapter adı) +
                            SALT-OKUNUR hazırlık kontrolü. Hiçbir dosya yazmaz/bölmez.
2. ``prepare_snapshot()`` : (insan) değişmez veri anlık görüntüsü + reçete. Reçete: veri özeti
                            (lora_sft + deterministik train/valid), temel model, profil, karışım
                            ağırlıkları, hedef adapter adı, sohbet veri seçimi, kod durumu.
                            ``recipe_sha`` bunların özetidir.
3. ``launch()``           : (insan) özet onayı. Sıra: idempotentlik (aynı istek kimliği → aynı
                            sonuç, çift eğitim yok) → UCUZ ön kontroller (anlık görüntü bütün mü,
                            güncel veri hâlâ aynı mı, iptal/sızıntı/kaynak/sohbet kontrolleri,
                            Kademe 2 kaydı, koşan iş) → onay. Onay ``train_run:<recipe_sha>``
                            eylemine BAĞLIDIR: veri/model/profil/karışım/adapter değişirse eski
                            onay tüketilemez. En son ``detached_launch.launch``.

Kademe 2 kaydı TÜM eğitim yollarında zorunludur (``kademe2_check``): web butonu / Auto-LoRA /
kolay akış (``detached_launch.preflight_launch``, onay tüketilmeden önce), ``hektor train --run``
(start-train.ps1, doğrudan CLI, nöbetçi kurtarması, web'in alt süreci) ve ``pretrain-gate``.
Kayıt salt bir "OK" dosyası değildir: denetlenen kod durumu
(app/ scripts/ configs/ pyproject.toml ağaç özetleri + temiz çalışma ağacı), kapsam (reçete ya da
veri özeti), bulgular (her biri kapanmış) ve kapanış kanıtı içerir. Kayıt ``reports/`` altında
olduğundan commit'lenmesi denetlenen kod özetini DEĞİŞTİRMEZ → sonsuz yeniden denetim yok.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import DEFAULT_TRAIN_PROFILE, get_settings
from app.feedback.chat_store import utcnow

CODE_PATHS = ("app", "scripts", "configs", "pyproject.toml")
CLOSED_FINDING = frozenset({"duzeltildi", "reddedildi", "risk_kabul"})


class EasyTrainError(ValueError):
    """Kullanıcıya gösterilecek akış hatası."""


# ── kod durumu ───────────────────────────────────────────────────────────────


def _git(args: list[str], root: Path) -> tuple[int, str]:
    try:
        p = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return p.returncode, p.stdout


def _default_code_state() -> dict[str, Any]:
    root = get_settings().root
    rc, out = _git(["ls-tree", "HEAD", "--", *CODE_PATHS], root)
    if rc != 0 or not out.strip():
        return {"ok": False, "code_sha": "", "dirty": [], "note": "git ağacı okunamadı"}
    code_sha = hashlib.sha256(out.encode("utf-8")).hexdigest()
    rc2, st = _git(["status", "--porcelain", "--", *CODE_PATHS], root)
    dirty = [ln[3:] for ln in st.splitlines() if ln.strip()] if rc2 == 0 else ["?"]
    _rc, head = _git(["rev-parse", "HEAD"], root)
    return {"ok": not dirty, "code_sha": code_sha, "dirty": dirty[:20], "head": head.strip()}


# Testlerde değiştirilebilir.
code_state_provider: Callable[[], dict[str, Any]] = _default_code_state


# ── yollar ───────────────────────────────────────────────────────────────────


def _storage() -> Path:
    return get_settings().root / "storage"


def last_settings_path() -> Path:
    return _storage() / "training_last_settings.json"


def snapshots_dir() -> Path:
    return _storage() / "training_snapshots"


def requests_path() -> Path:
    return _storage() / "training_launch_requests.json"


def kademe2_dir() -> Path:
    return get_settings().reports_dir / "kademe2"


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── ayarlar ──────────────────────────────────────────────────────────────────


def default_settings() -> dict[str, Any]:
    from app.lora.mix_common import load_mix_config

    s = get_settings()
    try:
        profiles = sorted((load_mix_config().get("profiles") or {}).keys())
    except Exception:
        profiles = []
    return {
        "profile": DEFAULT_TRAIN_PROFILE,
        "base_model": s.peft_base_model,
        "mix_profile": profiles[0] if profiles else "",
        "mix_weights": {},
        "adapter_name": "",
        "max_examples": 0,
    }


def last_settings() -> dict[str, Any]:
    saved = _read(last_settings_path())
    out = default_settings()
    if isinstance(saved, dict):
        out.update({k: v for k, v in saved.items() if k in out})
        out["_saved_at"] = saved.get("_saved_at", "")
    out["_source"] = "son kullanılan" if isinstance(saved, dict) else "varsayılan (ilk kullanım)"
    return out


def resolve_weights(settings: dict[str, Any]) -> tuple[dict[str, float], str]:
    from app.lora.mix_common import profile_weights, validate_weights

    if settings.get("mix_weights"):
        return validate_weights(settings["mix_weights"]), "özel"
    name = str(settings.get("mix_profile") or "")
    if not name:
        raise EasyTrainError("Karışım profili ya da ağırlıkları seçilmeli.")
    return profile_weights(name), name


def _normalize(settings: dict[str, Any]) -> dict[str, Any]:
    from app.training.detached_launch import _ADAPTER_RE

    base = last_settings()
    merged = {k: settings.get(k, base.get(k)) for k in default_settings()}
    if not _ADAPTER_RE.match(str(merged.get("adapter_name") or "")):
        raise EasyTrainError("Hedef adapter adı gerekli (harf, rakam, _ ve -; en çok 64).")
    merged["max_examples"] = int(merged.get("max_examples") or 0)
    weights, label = resolve_weights(merged)
    merged["mix_weights_resolved"] = weights
    merged["mix_label"] = label
    return merged


# ── salt-okunur hazırlık ─────────────────────────────────────────────────────


def _lora_sft_lines() -> tuple[list[str], str]:
    from app.training.detached_launch import _combined_source

    src = _combined_source(get_settings())
    if not src.exists():
        return [], ""
    raw = src.read_bytes()
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    return lines, hashlib.sha256(raw).hexdigest()


def readiness() -> dict[str, Any]:
    """Hiçbir dosyayı yazmayan/bölmeyen kontrol listesi."""
    from app.feedback.chat_dataset import chat_selection_blockers
    from app.training import resource_lock
    from app.training.detached_launch import is_running

    items: list[dict[str, Any]] = []

    def add(key: str, ok: bool, detail: str) -> None:
        items.append({"key": key, "ok": bool(ok), "detail": detail})

    lines, data_sha = _lora_sft_lines()
    add("veri", bool(lines), f"lora_sft.jsonl {len(lines)} satır" if lines else "eğitim verisi yok")
    chat = chat_selection_blockers()
    add("sohbet_verisi", not chat, chat[0] if chat else "seçili sohbet sürümü geçerli / seçim yok")
    lock = resource_lock.blocker()
    add("agir_is", not lock and not is_running(), lock or "koşan eğitim/dönüşüm/karşılaştırma yok")
    code = code_state_provider()
    add(
        "kod_durumu",
        bool(code.get("ok")),
        "çalışma ağacı temiz"
        if code.get("ok")
        else f"doğrulanamadı: {code.get('note') or ', '.join(code.get('dirty') or [])}",
    )
    k2 = kademe2_check(data_sha=data_sha)
    add("kademe2", k2 is None, k2 or "bu kod durumu + eğitim verisi için kapanmış kayıt var")
    return {
        "items": items,
        "ready": all(i["ok"] for i in items),
        "data_sha256": data_sha,
        "note": "Salt-okunur kontrol: hiçbir veri yazılmadı/bölünmedi.",
    }


def state() -> dict[str, Any]:
    from app.training.peft_lora_train import load_lora_profile  # noqa: F401  (varlık kontrolü)

    return {"settings": last_settings(), "readiness": readiness(), "snapshots": list_snapshots()}


# ── Kademe 2 kaydı ───────────────────────────────────────────────────────────


def kademe2_blocker(*, recipe_sha: str = "", data_sha: str | None = None) -> str | None:
    """Eğitim için Kademe 2 engeli (yoksa None). TÜM eğitim yolları bunu çağırır.

    Gerekenler: çalışma ağacı temiz (denetlenen kod = çalışacak kod) VE bu kod özeti için,
    kapsamı bu reçeteyi ya da GÜNCEL eğitim verisini (lora_sft.jsonl özeti) kapsayan, bulguları
    kapanmış bir kayıt. Veri ya da kod değişirse yeni kayıt gerekir.
    """
    if data_sha is None:
        _lines, data_sha = _lora_sft_lines()
    code = code_state_provider()
    if not code.get("ok"):
        why = code.get("note") or ", ".join(code.get("dirty") or []) or "bilinmiyor"
        return (
            f"Kademe 2: kod durumu doğrulanamadı ({why}) — denetlenen kod ile çalışacak kod "
            "aynı olmalı (app/ scripts/ configs/ pyproject.toml commit'li ve temiz)."
        )
    if latest_kademe2(code["code_sha"], recipe_sha, data_sha) is None:
        return (
            "Kademe 2 kaydı yok: bu kod durumu "
            f"({code['code_sha'][:12]}…) ve eğitim verisi ({(data_sha or '?')[:12]}…) için "
            "kapanmış derin av kaydı gerekli (her eğitimden önce zorunlu). Derin avdan sonra: "
            "`uv run hektor kademe2-kayit --findings bulgular.json --data-sha "
            f'{data_sha or "<sha>"} --evidence "..."`'
        )
    return None


# Testlerde (git ağacı olmayan geçici kök) değiştirilebilir; üretimde her zaman gerçek kapı.
kademe2_check: Callable[..., str | None] = kademe2_blocker


def record_kademe2(
    *,
    scope: dict[str, str],
    findings: list[dict[str, Any]],
    closure_evidence: str,
    reviewer: str,
) -> dict[str, Any]:
    """Kademe 2 derin av kaydı (kapanış kanıtıyla). Kod durumu temiz olmalı."""
    code = code_state_provider()
    if not code.get("ok"):
        raise EasyTrainError(f"Kod durumu temiz değil, denetlenen kod sabitlenemez: {code}")
    if not (scope.get("recipe_sha") or scope.get("data_sha256")):
        raise EasyTrainError("Kapsam reçete ya da veri özeti içermeli.")
    open_ = [f for f in findings if f.get("status") not in CLOSED_FINDING]
    if open_:
        raise EasyTrainError(f"Kapanmamış bulgu var ({len(open_)}): kayıt kapatılamaz.")
    if len((closure_evidence or "").strip()) < 20:
        raise EasyTrainError("Kapanış kanıtı (testler, commit'ler) en az 20 karakter olmalı.")
    rec = {
        "record_id": "k2_" + secrets.token_hex(6),
        "created_at": utcnow(),
        "code_sha": code["code_sha"],
        "head": code.get("head", ""),
        "code_paths": list(CODE_PATHS),
        "scope": scope,
        "findings": findings,
        "closure": {"status": "kapandi", "evidence": closure_evidence, "reviewer": reviewer},
        "note": "Kayıt reports/ altında; commit'lenmesi denetlenen kod özetini değiştirmez.",
    }
    _write(kademe2_dir() / f"{rec['record_id']}.json", rec)
    return rec


def list_kademe2() -> list[dict[str, Any]]:
    d = kademe2_dir()
    if not d.is_dir():
        return []
    out = [r for p in sorted(d.glob("k2_*.json")) if isinstance(r := _read(p), dict)]
    return sorted(out, key=lambda r: str(r.get("created_at", "")))


def latest_kademe2(code_sha: str, recipe_sha: str = "", data_sha: str = "") -> dict | None:
    """Bu kod durumu (ve verilirse reçete/veri) için kapanmış son kayıt."""
    found = None
    for r in list_kademe2():
        if not code_sha or r.get("code_sha") != code_sha:
            continue
        if (r.get("closure") or {}).get("status") != "kapandi":
            continue
        if any(f.get("status") not in CLOSED_FINDING for f in r.get("findings", [])):
            continue
        sc = r.get("scope") or {}
        covered = (recipe_sha and sc.get("recipe_sha") == recipe_sha) or (
            data_sha and sc.get("data_sha256") == data_sha
        )
        if (recipe_sha or data_sha) and not covered:
            continue
        found = r
    return found


# ── anlık görüntü + reçete ───────────────────────────────────────────────────


def _recipe_sha(recipe: dict[str, Any]) -> str:
    keys = (
        "base_model",
        "profile",
        "mix_weights",
        "adapter_name",
        "max_examples",
        "data_sha256",
        "train_sha256",
        "valid_sha256",
        "chat_selection",
    )
    return _sha_text(json.dumps({k: recipe.get(k) for k in keys}, sort_keys=True))


def prepare_snapshot(settings: dict[str, Any]) -> dict[str, Any]:
    """(İnsan) Değişmez veri anlık görüntüsü + reçete. Aynı reçete → aynı anlık görüntü."""
    from app.feedback.chat_dataset import read_selection
    from app.training.detached_launch import split_lines_by_source
    from app.training.peft_lora_train import load_lora_profile

    norm = _normalize(settings)
    try:
        load_lora_profile(norm["profile"])
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise EasyTrainError(f"Profil hatası: {exc}") from exc
    lines, data_sha = _lora_sft_lines()
    if not lines:
        raise EasyTrainError("Eğitim verisi yok (data/lora_sft/lora_sft.jsonl).")
    train, valid = split_lines_by_source(lines)
    train_text = "\n".join(train) + "\n"
    valid_text = "\n".join(valid) + ("\n" if valid else "")
    sel = read_selection() or {}
    recipe = {
        "base_model": norm["base_model"],
        "profile": norm["profile"],
        "mix_weights": norm["mix_weights_resolved"],
        "mix_label": norm["mix_label"],
        "adapter_name": norm["adapter_name"],
        "max_examples": norm["max_examples"],
        "data_sha256": data_sha,
        "train_sha256": _sha_text(train_text),
        "valid_sha256": _sha_text(valid_text),
        "n_train": len(train),
        "n_valid": len(valid),
        "chat_selection": {
            "version_id": sel.get("version_id", ""),
            "train_sha256": sel.get("train_sha256", ""),
        },
    }
    rsha = _recipe_sha(recipe)
    recipe["recipe_sha"] = rsha
    snap_id = "snap_" + rsha[:16]
    d = snapshots_dir() / snap_id
    if not (d / "recipe.json").exists():
        tmp = snapshots_dir() / f".tmp-{snap_id}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "train.jsonl").write_bytes(train_text.encode("utf-8"))
        (tmp / "valid.jsonl").write_bytes(valid_text.encode("utf-8"))
        _write(tmp / "recipe.json", {**recipe, "snapshot_id": snap_id, "created_at": utcnow()})
        if d.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            os.replace(tmp, d)
    saved = {k: norm[k] for k in default_settings()}
    _write(last_settings_path(), {**saved, "_saved_at": utcnow()})
    return {**(_read(d / "recipe.json") or recipe), "summary": summary(recipe)}


def summary(recipe: dict[str, Any]) -> str:
    from app.lora.mix_common import format_weights

    return (
        f"{recipe['adapter_name']} ← {recipe['base_model']} · profil {recipe['profile']} · "
        f"karışım {recipe.get('mix_label', '')} ({format_weights(recipe['mix_weights'])}) · "
        f"{recipe['n_train']} train / {recipe['n_valid']} valid · veri "
        f"{recipe['data_sha256'][:12]}… · reçete {recipe['recipe_sha'][:12]}…"
    )


def list_snapshots() -> list[dict[str, Any]]:
    d = snapshots_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("snap_*/recipe.json")):
        r = _read(p)
        if isinstance(r, dict):
            out.append({**r, "summary": summary(r)})
    return sorted(out, key=lambda r: str(r.get("created_at", "")), reverse=True)


def _snapshot(snap_id: str) -> dict[str, Any]:
    d = snapshots_dir() / Path(snap_id).name
    r = _read(d / "recipe.json")
    if not isinstance(r, dict):
        raise EasyTrainError(f"Anlık görüntü yok: {snap_id}")
    for name, key in (("train.jsonl", "train_sha256"), ("valid.jsonl", "valid_sha256")):
        p = d / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != r[key]:
            raise EasyTrainError(f"Anlık görüntü bozulmuş ({name} özeti tutmuyor).")
    if _recipe_sha(r) != r["recipe_sha"]:
        raise EasyTrainError("Reçete dosyası değişmiş (özet tutmuyor).")
    return r


# ── başlatma ─────────────────────────────────────────────────────────────────


def approval_action(recipe_sha: str) -> str:
    return f"train_run:{recipe_sha[:16]}"


def precheck(recipe: dict[str, Any]) -> list[str]:
    """Gerçek başlatmadan HEMEN önce yeniden yapılan ucuz kontroller (onay tüketilmeden)."""
    from app.feedback.chat_dataset import chat_selection_blockers, read_selection
    from app.training import resource_lock
    from app.training.detached_launch import _pretrain_gate_blockers, is_running

    problems: list[str] = []
    _lines, data_sha = _lora_sft_lines()
    if data_sha != recipe["data_sha256"]:
        problems.append("eğitim verisi anlık görüntüden sonra değişti — yeni anlık görüntü gerekir")
    sel = read_selection() or {}
    if sel.get("version_id", "") != recipe["chat_selection"]["version_id"]:
        problems.append("sohbet veri seçimi değişti — yeni anlık görüntü gerekir")
    problems += chat_selection_blockers()  # iptal/hariç/düzenlenmiş aday
    problems += _pretrain_gate_blockers(get_settings())  # kalite + kaynak kapısı
    try:
        from app.evals.profile.leakage import check_leakage
        from app.lora.mix_cli import leakage_eval_items

        d = snapshots_dir() / f"snap_{recipe['recipe_sha'][:16]}"
        rows = [json.loads(x) for x in (d / "train.jsonl").read_text("utf-8").splitlines() if x]
        rows += [json.loads(x) for x in (d / "valid.jsonl").read_text("utf-8").splitlines() if x]
        if not check_leakage(leakage_eval_items(), rows).clean:
            problems.append("anlık görüntüde eval sızıntısı var")
    except Exception as exc:
        problems.append(f"sızıntı kontrolü çalıştırılamadı: {exc}")
    lock = resource_lock.blocker()
    if lock or is_running():
        problems.append(lock or "eğitim zaten çalışıyor")
    k2 = kademe2_check(recipe_sha=recipe["recipe_sha"], data_sha=recipe["data_sha256"])
    if k2:
        problems.append(k2)
    return problems


def launch(snapshot_id: str, request_id: str) -> dict[str, Any]:
    """(İnsan) Özeti onayla ve başlat. Aynı ``request_id`` ikinci kez eğitim başlatmaz."""
    from app.agents.runtime import approvals
    from app.lora.weight_decision import WeightDecisionStore
    from app.training import detached_launch

    request_id = (request_id or "").strip()
    if not 8 <= len(request_id) <= 80:
        raise EasyTrainError("Geçersiz istek kimliği.")
    reqs = _read(requests_path()) or {}
    if request_id in reqs and reqs[request_id].get("status") in ("started", "starting"):
        return {**reqs[request_id], "replayed": True}
    recipe = _snapshot(snapshot_id)
    problems = precheck(recipe)
    if problems:
        return {
            "ok": False,
            "status": "blocked",
            "problems": problems,
            "message": "Ön kontroller geçmedi — onay TÜKETİLMEDİ.",
        }
    reqs[request_id] = {"status": "starting", "snapshot_id": snapshot_id, "at": utcnow()}
    _write(requests_path(), reqs)
    decision = approvals.require_fresh_approval(
        "lora-trainer",
        approval_action(recipe["recipe_sha"]),
        "critical",
        "Gerçek LoRA eğitimi (kolay akış): " + summary(recipe),
    )
    if not decision.authorized:
        reqs[request_id] = {"status": "needs_approval", "approval_id": decision.approval_id}
        _write(requests_path(), reqs)
        return {
            "ok": False,
            "status": "needs_approval",
            "approval_id": decision.approval_id,
            "approve_command": f"uv run hektor approval-approve {decision.approval_id}",
            "message": "Bu reçeteye bağlı taze onay gerekli. Onaylayıp aynı özeti tekrar onayla "
            "(reçete değişirse bu onay kullanılamaz).",
        }
    # Ağırlık kararı: kullanıcı özeti (ağırlıklar dahil) onayladı → bu reçetenin ağırlıkları.
    WeightDecisionStore().record(
        recipe["mix_weights"], recipe.get("mix_label") or "özel", source="easy_train:" + snapshot_id
    )
    res = detached_launch.launch(
        adapter_name=recipe["adapter_name"],
        base_model=recipe["base_model"],
        profile=recipe["profile"],
        max_examples=recipe["max_examples"],
        approval_id=decision.approval_id,
    )
    status = "started" if res.get("ok") else "error"
    out = {
        "ok": bool(res.get("ok")),
        "status": status,
        "message": res.get("message", ""),
        "approval_id": decision.approval_id,
        "snapshot_id": snapshot_id,
    }
    reqs[request_id] = {**out, "at": utcnow()}
    _write(requests_path(), reqs)
    if res.get("ok"):
        _write(snapshots_dir() / snapshot_id / "launch.json", {**out, "at": utcnow()})
    return out
