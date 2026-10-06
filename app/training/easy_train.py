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

import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import DEFAULT_TRAIN_PROFILE, get_settings
from app.feedback.chat_store import utcnow

CODE_PATHS = ("app", "scripts", "configs", "pyproject.toml")
CLOSED_FINDING = frozenset({"duzeltildi", "reddedildi", "risk_kabul"})
# risk_kabul kapsamı: yalnız kaydın reçetesi (pilot) ya da kaydın tüm kapsamı (veri/reçete).
RISK_SCOPES = frozenset({"yalniz_recete", "kayit"})


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


# Windows'ta açık bir dosyanın üzerine ``os.replace`` ve değiştirilmekte olan dosyayı okuma
# kısa süre PermissionError verir (eşzamanlı iki başlatma isteği). Geçici pencere: bekleyip
# yeniden denenir. Okumada hatayı "dosya yok" saymak, istek kaydının boş sanılıp üzerine
# yazılmasına (diğer isteklerin kaybına) yol açardı (#36 ile aynı sınıf; c0d6aea avı).
_SHARE_RETRY_S = 5.0


def _read(path: Path) -> Any:
    deadline = time.monotonic() + _SHARE_RETRY_S
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if time.monotonic() > deadline:
                return None
            time.sleep(0.01)
        except (OSError, ValueError):
            return None


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    deadline = time.monotonic() + _SHARE_RETRY_S
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
            time.sleep(0.01)


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
    # Kademe 2 2026-10-06 F1-9: arayüz yalnız `mix_profile` gönderir. Eksik `mix_weights`
    # son kaydedilen ayardan dolarsa, eski ÖZEL ağırlıklar açıkça seçilen profili sessizce
    # eziyordu (resolve_weights özel ağırlığı önceler). Açık profil seçimi = profil ağırlığı.
    if settings.get("mix_profile") and "mix_weights" not in settings:
        merged["mix_weights"] = {}
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
    if recipe_sha:
        # "Reçete özeti yeterli" yolu: reçete GERÇEK, bütün bir anlık görüntüye ait olmalı ve
        # o reçetenin veri özeti şu anki eğitim verisiyle aynı olmalı. Aksi halde reçete
        # kapsamlı kayıt, başka veri/temel model/profil/karışımla eğitime izin verebilirdi.
        problem = _recipe_scope_problem(recipe_sha, data_sha)
        if problem:
            return f"Kademe 2: {problem}"
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


def _recipe_scope_problem(recipe_sha: str, data_sha: str | None) -> str:
    try:
        recipe = _snapshot(f"snap_{recipe_sha[:16]}")
    except EasyTrainError as exc:
        return f"reçete {recipe_sha[:12]}… doğrulanamadı ({exc})"
    if recipe.get("recipe_sha") != recipe_sha:
        return f"reçete özeti anlık görüntüyle tutmuyor ({recipe_sha[:12]}…)"
    if data_sha is not None and recipe.get("data_sha256") != data_sha:
        return (
            f"reçetenin veri özeti ({str(recipe.get('data_sha256'))[:12]}…) şu anki eğitim "
            f"verisiyle ({(data_sha or '?')[:12]}…) aynı değil — yeni anlık görüntü gerekir"
        )
    return ""


def recipe_binding_problems(
    recipe_sha: str,
    *,
    adapter_name: str,
    base_model: str | None,
    profile: str | None,
    max_examples: int,
    weights: dict[str, float] | None = None,
) -> list[str]:
    """Başlatılacak/başlamış koşu onaylanan reçeteyle AYNI mı? (boş liste = aynı)

    Hem web/kolay akış başlatmasında (alt süreç doğmadan) hem ``train --run`` alt sürecinde
    (onay/kilit öncesi) çağrılır: veri (lora_sft + train/valid bölmesi), temel model, profil,
    örnek tavanı, adapter adı ve karışım ağırlıkları.
    """

    def norm_sha(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        except OSError:
            return ""

    try:
        r = _snapshot(f"snap_{recipe_sha[:16]}")
    except EasyTrainError as exc:
        return [f"onaylanan reçete doğrulanamadı: {exc}"]
    out: list[str] = []
    s = get_settings()
    _lines, data_sha = _lora_sft_lines()
    if data_sha != r["data_sha256"]:
        out.append("lora_sft.jsonl onaylanan anlık görüntüden sonra değişti")
    for name, key in (("train.jsonl", "train_sha256"), ("valid.jsonl", "valid_sha256")):
        if norm_sha(s.jsonl_dir / name) != r[key]:  # satır sonu normalize (Windows CRLF)
            out.append(f"{name} onaylanan anlık görüntüyle aynı değil")
    # F3-5 zinciri: seçili sohbet veri sürümü reçetedekiyle aynı olmalı (eskiden yalnız web
    # precheck'te; alt süreçte dolaylıydı — tazelik + veri özeti üzerinden).
    from app.feedback.chat_dataset import read_selection

    try:
        sel = read_selection() or {}
    except ValueError as exc:
        return [*out, f"sohbet veri seçimi okunamadı: {exc}"]
    want_sel = r.get("chat_selection") or {}
    if (sel.get("version_id", ""), sel.get("train_sha256", "")) != (
        want_sel.get("version_id", ""),
        want_sel.get("train_sha256", ""),
    ):
        out.append("seçili sohbet veri sürümü onaylanan reçeteden farklı")
    want = {
        "adapter_name": (adapter_name, r.get("adapter_name")),
        "base_model": (base_model or s.peft_base_model, r.get("base_model") or s.peft_base_model),
        "profile": (profile or "", r.get("profile") or ""),
        "max_examples": (int(max_examples or 0), int(r.get("max_examples") or 0)),
    }
    for key, (got, exp) in want.items():
        if got != exp:
            out.append(f"{key} reçeteden farklı ({got!r} ≠ {exp!r})")
    if weights is not None:
        exp_w = {k: round(float(v), 6) for k, v in (r.get("mix_weights") or {}).items()}
        got_w = {k: round(float(v), 6) for k, v in weights.items()}
        if exp_w != got_w:
            out.append(f"karışım ağırlıkları reçeteden farklı ({got_w} ≠ {exp_w})")
    return out


RECIPE_ENV = "HEKTOR_TRAIN_RECIPE_SHA"


# Testlerde (git ağacı olmayan geçici kök) değiştirilebilir; üretimde her zaman gerçek kapı.
kademe2_check: Callable[..., str | None] = kademe2_blocker


def record_kademe2(
    *,
    scope: dict[str, str],
    findings: list[dict[str, Any]],
    closure_evidence: str,
    reviewer: str,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Kademe 2 derin av kaydı (kapanış kanıtıyla). Kod durumu temiz olmalı.

    Boş bulgu listesi YALNIZ ``allow_empty`` ile (açıkça "bulgusuz av") kabul edilir: yanlış
    biçimli dosya boş listeye dönüşüp "hepsi kapalı" diye kapıyı açmasın. ``risk_kabul``
    yazılı gerekçe (``gerekce``) ister — açık maddeyi gerekçesiz kapatmak kayıtta görünür.
    """
    code = code_state_provider()
    if not code.get("ok"):
        raise EasyTrainError(f"Kod durumu temiz değil, denetlenen kod sabitlenemez: {code}")
    if not (scope.get("recipe_sha") or scope.get("data_sha256")):
        raise EasyTrainError("Kapsam reçete ya da veri özeti içermeli.")
    if not isinstance(findings, list) or not all(isinstance(f, dict) for f in findings):
        raise EasyTrainError("Bulgular bir liste olmalı (her biri id + status).")
    if not findings and not allow_empty:
        raise EasyTrainError(
            "Bulgu listesi boş — bulgusuz bir av ise bunu açıkça belirt (--temiz-av); "
            "yanlış dosya verilmiş olabilir."
        )
    nameless = [f for f in findings if not str(f.get("id") or "").strip()]
    if nameless:
        raise EasyTrainError(f"Kimliksiz bulgu var ({len(nameless)}): her bulgu 'id' taşımalı.")
    # Kademe 2 L-1: risk kabulünün KAPSAMI açık ve tanımlı olmalı — yazım farkı (ör.
    # 'yalnız_reçete') sessizce sınırsız kabul sayılıp tam eğitime taşınıyordu.
    bad_scope = [
        f"{f['id']}={f.get('kapsam')!r}"
        for f in findings
        if f.get("status") == "risk_kabul" and f.get("kapsam") not in RISK_SCOPES
    ]
    if bad_scope:
        raise EasyTrainError(
            f"risk_kabul kapsamı tanımsız ({', '.join(bad_scope)}): 'kapsam' alanı "
            f"{sorted(RISK_SCOPES)} değerlerinden biri olmalı."
        )
    bare_risk = [
        f["id"]
        for f in findings
        if f.get("status") == "risk_kabul" and len(str(f.get("gerekce") or "").strip()) < 20
    ]
    if bare_risk:
        raise EasyTrainError(
            f"risk_kabul gerekçesiz ({', '.join(map(str, bare_risk))}): her biri en az 20 "
            "karakterlik 'gerekce' taşımalı."
        )
    # Reçeteye SINIRLI risk kabulü (ör. F3-4: yalnız 64 örneklik teknik pilot): kayıt yalnız o
    # reçeteyi kapsayabilir — veri özeti kapsamı, aynı veriyle HER reçeteye (tam eğitime)
    # taşınırdı.
    limited = [f["id"] for f in findings if _recipe_limited(f)]
    if limited and (scope.get("data_sha256") or not scope.get("recipe_sha")):
        raise EasyTrainError(
            f"Reçeteye sınırlı risk kabulü ({', '.join(map(str, limited))}): kayıt YALNIZ "
            "--recipe-sha ile yazılabilir (--data-sha verilmez)."
        )
    open_ = [f for f in findings if f.get("status") not in CLOSED_FINDING]
    if open_:
        raise EasyTrainError(
            f"Kapanmamış bulgu var ({len(open_)}): "
            + ", ".join(f"{f.get('id')}={f.get('status')}" for f in open_[:10])
            + " — kayıt kapatılamaz."
        )
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


def _recipe_limited(finding: dict[str, Any]) -> bool:
    return finding.get("status") == "risk_kabul" and finding.get("kapsam") == "yalniz_recete"


def recipe_has_limited_acceptance(recipe_sha: str) -> bool:
    """Bu reçete yalnız-reçete kapsamlı (pilot) bir risk kabulüyle mi kayıtlı?"""
    if not recipe_sha:
        return False
    return any(
        (r.get("scope") or {}).get("recipe_sha") == recipe_sha
        and any(_recipe_limited(f) for f in r.get("findings", []))
        for r in list_kademe2()
    )


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
        if any(_recipe_limited(f) for f in r.get("findings", [])):
            # Sınırlı kabul yalnız kendi reçetesini kapsar (elle düzenlenmiş kayıtta da).
            covered = bool(recipe_sha) and sc.get("recipe_sha") == recipe_sha
        else:
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


def _effective_note(recipe: dict[str, Any]) -> str:
    """Kademe 2 (c0d6aea avı) D-3: onay metni FİİLEN eğitilecek örnek sayısını + seed'i gösterir."""
    try:
        from app.training.detached_launch import plan_iterations
        from app.training.peft_lora_train import PeftTrainConfig

        iters, n_eff, epochs = plan_iterations(
            int(recipe["n_train"]), int(recipe.get("max_examples") or 0), recipe.get("profile")
        )
        seed = PeftTrainConfig.__dataclass_fields__["seed"].default
    except Exception:  # özet metni asla başlatmayı bozmasın
        return ""
    return f" (eğitilecek {n_eff} örnek, seed {seed}, {iters} mikro-adım ≈ {epochs} epoch)"


def summary(recipe: dict[str, Any]) -> str:
    from app.lora.mix_common import format_weights

    return (
        f"{recipe['adapter_name']} ← {recipe['base_model']} · profil {recipe['profile']} · "
        f"karışım {recipe.get('mix_label', '')} ({format_weights(recipe['mix_weights'])}) · "
        f"{recipe['n_train']} train / {recipe['n_valid']} valid"
        f"{_effective_note(recipe)} · veri "
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


def _set_request(request_id: str, value: dict[str, Any]) -> None:
    """F1-6: istek kaydını DOSYANIN güncel hâli üzerine yaz (bayat kopya diğer istekleri
    silmesin)."""
    reqs = _read(requests_path()) or {}
    reqs[request_id] = value
    _write(requests_path(), reqs)


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
    # Kademe 2 (2026-10-06) F1-2: onay TÜKETİLMEDEN önce sohbet kirası, adapter klasörü ve
    # reçete (RAM + hedef modüller) — eskiden onaydan sonra/alt süreçte düşüyordu.
    from app.feedback.resource_guard import chat_lease_blocker
    from app.training.detached_launch import _adapter_dir_blocker, _recipe_blockers

    s = get_settings()
    lease = chat_lease_blocker(s.root)
    if lease:
        problems.append(lease)
    dir_blocker = _adapter_dir_blocker(s.adapters_dir / recipe["adapter_name"])
    if dir_blocker:
        problems.append(dir_blocker)
    problems += _recipe_blockers(recipe.get("base_model"), recipe.get("profile"))
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
    # F1-7: kimlik yalnız kendi biçiminde (yol öğesi yok) — launch.json başka yere yazılmasın.
    if not re.fullmatch(r"snap_[0-9a-f]{16}", snapshot_id or ""):
        raise EasyTrainError("Geçersiz anlık görüntü kimliği.")
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
    _set_request(request_id, {"status": "starting", "snapshot_id": snapshot_id, "at": utcnow()})
    decision = approvals.require_fresh_approval(
        "lora-trainer",
        approval_action(recipe["recipe_sha"]),
        "critical",
        "Gerçek LoRA eğitimi (kolay akış): " + summary(recipe),
    )
    if not decision.authorized:
        _set_request(request_id, {"status": "needs_approval", "approval_id": decision.approval_id})
        return {
            "ok": False,
            "status": "needs_approval",
            "approval_id": decision.approval_id,
            "approve_command": f"uv run hektor approval-approve {decision.approval_id}",
            "message": "Bu reçeteye bağlı taze onay gerekli. Onaylayıp aynı özeti tekrar onayla "
            "(reçete değişirse bu onay kullanılamaz).",
        }
    # Ağırlık kararı: kullanıcı özeti (ağırlıklar dahil) onayladı → bu reçetenin ağırlıkları.
    wd = WeightDecisionStore().record(
        recipe["mix_weights"], recipe.get("mix_label") or "özel", source="easy_train:" + snapshot_id
    )
    res = detached_launch.launch(
        adapter_name=recipe["adapter_name"],
        base_model=recipe["base_model"],
        profile=recipe["profile"],
        max_examples=recipe["max_examples"],
        approval_id=decision.approval_id,
        recipe_sha=recipe["recipe_sha"],
    )
    status = "started" if res.get("ok") else "error"
    out = {
        "ok": bool(res.get("ok")),
        "status": status,
        "message": res.get("message", ""),
        "approval_id": decision.approval_id,
        "snapshot_id": snapshot_id,
    }
    _set_request(request_id, {**out, "at": utcnow()})
    if not res.get("ok"):
        # F1-8: başlamayan koşunun ağırlık kararı açık kalırsa sonraki (başka reçeteli) koşu onu
        # sormadan tüketirdi → geri çek (her eğitimde yeniden sorulur).
        WeightDecisionStore().revoke(wd.decision_id, "kolay akış başlatılamadı")
    if res.get("ok"):
        _write(snapshots_dir() / snapshot_id / "launch.json", {**out, "at": utcnow()})
    return out
