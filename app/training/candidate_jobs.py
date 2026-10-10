"""candidate_jobs.py — aday yaşam döngüsünün AĞIR işleri için ortak servis (web + CLI).

Akış: Adayı doğrula → Ollama'ya hazırla (dönüşüm) → Karşılaştır → Sonuçları incele (kör
inceleme) → Kullanıma al / Geri dön (``model_activation``). Bu modül yalnız iki ağır işi yönetir;
iş mantığını KOPYALAMAZ, mevcut komutları çalıştırır:

- ``conversion`` : ``scripts/adapter_to_ollama.ps1`` (aynı betik; ``-Force`` ASLA verilmez →
  mevcut Ollama etiketi ya da web'in canlı modeli ezilemez).
- ``comparison`` : ``hektor compare-run`` (aynı CLI komutu).

İlkeler:
- Ayrık çalıştırıcı (``python -m app.training.candidate_jobs run <iş>``) alt komutu başlatır,
  bekler ve SONUCU iş dosyasına yazar. Web sunucusu kapansa da iş sürer; sayfa yenilenince ya
  da sunucu yeniden açılınca durum iş dosyası + süreç canlılığıyla UZLAŞTIRILIR (çalıştırıcı
  ölmüşse "kesildi" — tamamlandı sayılmaz).
- Çift tıklama / yeniden deneme: aynı ``request_id`` aynı işi döndürür; bir iş koşarken (ya da
  ortak ağır iş kilidi tutuluyorken) yeni iş başlamaz. Başlatma kısa bir O_EXCL mutex'i
  altındadır.
- Kaynak kilidi web ve CLI için AYNI: alt komutlar ``storage/heavy_job.lock``'u kendileri alır.
- Başarı = çıkış kodu 0 + iş sonrası doğrulama (dönüşüm: ``verify_conversion``; karşılaştırma:
  manifest ``generated``). Yarım (``.partial``) dosya ya da durdurulmuş üretim geçerli aday
  SAYILMAZ.
- Güvenli durdurma: süreç ağacı sonlandırılır, iş "durduruldu" olur; ortak kilit sahibi öldüğü
  için bayat sayılır.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.config.settings import PROJECT_ROOT
from app.feedback.chat_store import utcnow
from app.procutil import CREATE_BREAKAWAY_FROM_JOB, DETACHED_HIDDEN, NO_WINDOW

KINDS = ("conversion", "comparison")
KIND_TR = {"conversion": "Ollama'ya hazırlama (dönüşüm)", "comparison": "Karşılaştırma"}
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,60}$")
ADAPTER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEFAULT_TEMPLATE = "hektor-base-30b-q4a8"
DEFAULT_SET = "evals/candidate_compare/smoke_v1.jsonl"
ACTIVE = ("starting", "running", "stopping")
_STEP_RE = re.compile(r"\b([1-5])/5\b")
_PROG_RE = re.compile(r"İLERLEME\s+(\d+)/(\d+)")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SPIN_RE = re.compile(
    r"^(gathering model components|copying file|parsing GGUF|using existing layer)\b.*"
)


class JobError(ValueError):
    """Kullanıcıya gösterilecek iş hatası."""


def jobs_dir(root: Path | None = None) -> Path:
    return (root or get_settings().root) / "storage" / "candidate_jobs"


def _path(job_id: str) -> Path:
    if not re.fullmatch(r"job_[0-9a-f]{12}", job_id or ""):
        raise JobError(f"Geçersiz iş kimliği: {job_id}")
    return jobs_dir() / f"{job_id}.json"


# Windows'ta açık bir dosyanın üzerine ``os.replace`` ve değiştirilmekte olan dosyayı okuma
# kısa süre PermissionError verir (web yoklaması ↔ çalıştırıcı yazımı). Bu pencere geçicidir:
# bekleyip yeniden denenir; aksi halde iş "yok" görünür ya da sonuç yazılamaz (iş asılı kalır).
_SHARE_RETRY_S = 5.0
# ``os.replace`` sırasında okuyucu hedefi çok kısa süre YOK da görebilir (FileNotFoundError;
# 2026-10-06 stres testinde ölçüldü). Gerçekten olmayan iş için bekleme kısa tutulur.
_MISSING_RETRY_S = 0.5


def _read(p: Path) -> dict[str, Any] | None:
    t0 = time.monotonic()
    while True:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else None
        except PermissionError:
            if time.monotonic() - t0 > _SHARE_RETRY_S:
                return None
            time.sleep(0.01)
        except FileNotFoundError:
            if time.monotonic() - t0 > _MISSING_RETRY_S:
                return None
            time.sleep(0.01)
        except (OSError, ValueError):
            return None


def _write(p: Path, data: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    deadline = time.monotonic() + _SHARE_RETRY_S
    while True:
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
            time.sleep(0.01)


@contextlib.contextmanager
def _job_lock(job_id: str) -> Iterator[None]:
    """İş dosyasının oku-değiştir-yaz güncellemesi için kısa kilit (başlatıcı ↔ çalıştırıcı).

    Windows'ta başka sürecin o an sildiği (silinme-bekleyen) kilit dosyasına ``O_EXCL`` açma
    FileExistsError DEĞİL PermissionError verir; bu da "kilit meşgul" demektir. Yakalanmazsa
    çalıştırıcı sonucunu yazamadan çöker ve iş "kesildi" görünür (2026-10-06 kararsız test).
    """
    p = _path(job_id).with_suffix(".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    # Kademe 2 T-5 (2026-10-10): bekleme süresi bayat eşiğinden (30 sn) uzun — ölen sahibin
    # yetim kilidi kırılabilsin; alınamazsa KİLİTSİZ yazılmaz, hata yükselir (fail-closed).
    deadline = time.monotonic() + 45
    while True:
        try:
            os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except (FileExistsError, PermissionError):
            with contextlib.suppress(OSError):
                if time.time() - p.stat().st_mtime > 30 and _break_stale_lock(p):
                    continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"iş kaydı kilidi alınamadı: {p.name}") from None
            time.sleep(0.02)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            p.unlink()


def _break_stale_lock(p: Path) -> bool:
    """Bayat kilidi ATOMİK sahiplenip sil (T-5 TOCTOU): yalnız tek bekleyen yeniden
    adlandırabilir; taşınan dosya o arada tazelenmişse (yeni sahip) geri konur."""
    tmp = p.with_name(f"{p.name}.break{os.getpid()}_{time.monotonic_ns()}")
    try:
        os.rename(p, tmp)
    except OSError:
        return False
    try:
        if time.time() - tmp.stat().st_mtime <= 30:
            with contextlib.suppress(OSError):
                os.link(tmp, p)  # taze kilit yanlışlıkla alındı → sahibine iade (varsa dokunma)
            return False
        return True
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _update(job_id: str, **fields: Any) -> dict[str, Any]:
    p = _path(job_id)
    with _job_lock(job_id):
        job = _read(p)
        if job is None:
            # Kademe 2 T-3: okunamayan kaydı yalnız yeni alanlarla EZME (iş kimliği/durum/
            # komut kaybolurdu) — hata yükselir; çalıştırıcı düşerse iş "kesildi" görünür.
            raise RuntimeError(f"iş kaydı okunamadı, güncellenmedi: {job_id}")
        job.update(fields)
        _write(p, job)
    return job


@contextlib.contextmanager
def _start_mutex() -> Iterator[None]:
    """Başlatma bölümü için kısa ömürlü O_EXCL kilit (çift tıklama yarışı)."""
    d = jobs_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / "start.mutex"
    deadline = time.monotonic() + 10
    while True:
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except (FileExistsError, PermissionError) as exc:  # Windows: silinme-bekleyen kilit
            with contextlib.suppress(OSError):
                if time.time() - p.stat().st_mtime > 30:  # çökmüş başlatıcı
                    p.unlink()
                    continue
            if time.monotonic() > deadline:
                raise JobError(
                    "Başka bir başlatma sürüyor — birkaç saniye sonra yenileyin."
                ) from exc
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            p.unlink()


# ── süreç yardımcıları ───────────────────────────────────────────────────────


def _alive(pid: Any, create_time: Any = None) -> bool:
    from app.training.resource_lock import pid_alive, process_create_time

    if not pid_alive(pid):
        return False
    if isinstance(create_time, int | float) and isinstance(pid, int):
        got = process_create_time(pid)
        if got is not None and abs(got - float(create_time)) > 1.0:
            return False  # pid başka bir sürece geçmiş
    return True


def _kill_tree(pid: int) -> list[int]:
    killed: list[int] = []
    try:
        import psutil

        from app.training.resource_lock import process_tree

        try:
            procs = process_tree(pid)  # yalnız gerçek alt süreçler (bayat ppid'li yetim değil)
        except psutil.NoSuchProcess:
            # Kademe 2 T-6: kök o arada bitti → pid başka sürece geçmiş olabilir; taskkill
            # /T /F ile körlemesine ağaç öldürme YOK.
            return killed
        for p in procs:
            with contextlib.suppress(psutil.Error):
                p.terminate()
                killed.append(p.pid)
        _gone, alive = psutil.wait_procs(procs, timeout=10)
        for p in alive:
            with contextlib.suppress(psutil.Error):
                p.kill()
    except Exception:  # psutil yoksa ya da ağaç okunamadı
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                creationflags=NO_WINDOW,
            )
        else:
            with contextlib.suppress(OSError):
                os.kill(pid, 15)
        killed.append(pid)
    return killed


# ── durum ────────────────────────────────────────────────────────────────────


def _stage(job: dict[str, Any]) -> dict[str, Any]:
    """Günlükten aşama ilerlemesi (dönüşüm: n/5 adım; karşılaştırma: cevap i/n)."""
    log = Path(job.get("log_path") or "")
    tail = ""
    with contextlib.suppress(OSError):
        data = log.read_bytes()[-400_000:]
        tail = data.decode("utf-8", errors="replace")
    with contextlib.suppress(OSError):  # çalıştırıcının kendi hatası (ör. içe aktarma)
        rlog = log.with_name(log.name.replace(".log", ".runner.log"))
        tail += rlog.read_bytes()[-4000:].decode("utf-8", errors="replace")
    stage: dict[str, Any] = {"label": "", "done": 0, "total": 0}
    if job.get("kind") == "conversion":
        steps = _STEP_RE.findall(tail)
        if steps:
            n = int(steps[-1])
            stage = {"label": f"adım {n}/5", "done": n, "total": 5}
    else:
        prog = _PROG_RE.findall(tail)
        if prog:
            i, n = prog[-1]
            stage = {"label": f"cevap {i}/{n}", "done": int(i), "total": int(n)}
    # Terminal denetim dizileri ve dönen imleç (Ollama ilerleme çubuğu) ekranda gürültüdür.
    clean = _ANSI_RE.sub("\n", tail).replace("\r", "\n")
    lines = [ln for ln in clean.splitlines() if ln.strip() and not _SPIN_RE.match(ln.strip())]
    return {**stage, "log_tail": lines[-15:]}


def _runner_dead(job: dict[str, Any]) -> bool:
    if job.get("runner_pid") is None and job.get("status") == "starting":
        # Başlatıcı henüz çalıştırıcı pid'ini yazmadı (çok kısa pencere) → kesilmiş sayma.
        return False
    return not _alive(job.get("runner_pid"), job.get("runner_create_time"))


def _transition(job_id: str, allowed: tuple[str, ...], **fields: Any) -> dict[str, Any] | None:
    """Durumu YALNIZ güncel kayıt hâlâ ``allowed`` içindeyse değiştir (kilit altında, taze okuma).

    Bayat bir anlık görüntüye dayanan yazım, çalıştırıcının o arada yazdığı gerçek sonucu
    (``failed``/``done``) ezmesin diye. Değişiklik yapılmadıysa ``None``.
    """
    p = _path(job_id)
    with _job_lock(job_id):
        cur = _read(p)
        if cur is None or cur.get("status") not in allowed:
            return None
        cur.update(fields)
        _write(p, cur)
    return cur


def reconcile(job: dict[str, Any]) -> dict[str, Any]:
    """İş dosyasını gerçek süreç durumuyla uzlaştır (sayfa yenileme / sunucu yeniden açılış).

    Sıra önemli: çalıştırıcı önce sonucunu yazar, SONRA çıkar. Okunan anlık görüntü "running"
    iken çalıştırıcı sonucu yazıp çıkmış olabilir; bu yüzden "kesildi" kararı kilit altında
    TAZE kayıt + o kayda göre canlılıkla yeniden verilir (çalıştırıcının yazımı da aynı kilidi
    ister → karar ile yazım arasına sonuç giremez).
    """
    if job.get("status") in ACTIVE and _runner_dead(job):
        job_id = job["job_id"]
        p = _path(job_id)
        with _job_lock(job_id):
            fresh = _read(p)
            if fresh is None:
                # Taze kayıt okunamadı: bayat görüntüye dayanıp "kesildi" yazmak çalıştırıcının
                # gerçek sonucunu ezebilir → bu turda karar verme, sonraki yoklama yeniden dener.
                return {
                    **job,
                    "progress": _stage(job),
                    "kind_label": KIND_TR.get(job.get("kind", ""), ""),
                }
            cur = fresh
            if cur.get("status") in ACTIVE and _runner_dead(cur):
                # Çalıştırıcı sonucu yazamadan öldü (çökme, yeniden başlatma, öldürme).
                if cur.get("status") == "stopping":
                    cur.update(status="stopped", finished_at=cur.get("finished_at") or utcnow())
                else:
                    cur.update(
                        status="lost",
                        finished_at=utcnow(),
                        error="Çalıştırıcı süreç sonucu yazmadan sonlandı — iş TAMAMLANMIŞ "
                        "sayılmaz; kısmi çıktılar geçerli aday değildir. Yeniden deneyin.",
                    )
                _write(p, cur)
        job = cur
    return {**job, "progress": _stage(job), "kind_label": KIND_TR.get(job.get("kind", ""), "")}


def get_job(job_id: str) -> dict[str, Any]:
    job = _read(_path(job_id))
    if job is None:
        raise JobError(f"İş yok: {job_id}")
    return reconcile(job)


def list_jobs(adapter: str = "", limit: int = 30) -> list[dict[str, Any]]:
    d = jobs_dir()
    if not d.is_dir():
        return []
    out = []
    for p in d.glob("job_*.json"):
        j = _read(p)
        if j and (not adapter or j.get("adapter") == adapter):
            out.append(reconcile(j))
    out.sort(key=lambda j: str(j.get("created_at", "")), reverse=True)
    return out[:limit]


def running_job() -> dict[str, Any] | None:
    return next((j for j in list_jobs(limit=200) if j.get("status") in ACTIVE), None)


# ── yetenek (platform) kontrolü ──────────────────────────────────────────────


def is_windows() -> bool:
    return os.name == "nt"


def tools_dir() -> Path:
    return Path(os.environ.get("HEKTOR_CONVERSION_TOOLS", r"C:\HP\tools"))


def capability(kind: str, adapter: str = "") -> dict[str, Any]:
    """Bu makinede iş başlatılabilir mi? Kapalıysa AÇIK gerekçe."""
    s = get_settings()
    reasons: list[str] = []
    if kind == "conversion":
        if not is_windows():
            reasons.append(
                "Dönüşüm betiği (scripts/adapter_to_ollama.ps1) yalnız Windows + PowerShell'de "
                "çalışır; bu platformda kapalı."
            )
        if not (s.root / "scripts" / "adapter_to_ollama.ps1").is_file():
            reasons.append("scripts/adapter_to_ollama.ps1 bulunamadı.")
        gguf = s.root / "models" / "gguf"
        cached = bool(adapter) and any(gguf.glob(f"{adapter}-*.gguf.src"))
        t = tools_dir()
        have_tools = (t / "llama.cpp" / "convert_hf_to_gguf.py").is_file() and (
            t / "llama-bin" / "llama-quantize.exe"
        ).is_file()
        if not have_tools and not cached:
            reasons.append(
                f"llama.cpp araçları yok ({t}) ve bu adapter için önbellekte doğrulanmış GGUF "
                "yok — dönüşüm yapılamaz."
            )
        py = s.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not py.is_file():
            reasons.append(f"Betiğin kullanacağı Python ortamı yok: {py}")
    elif kind == "comparison":
        from app.feedback.model_identity import ollama_tags

        if ollama_tags() is None:
            reasons.append("Ollama sunucusuna ulaşılamıyor — karşılaştırma koşamaz.")
    else:
        reasons.append(f"Bilinmeyen iş türü: {kind}")
    return {"kind": kind, "supported": not reasons, "reasons": reasons}


# ── başlatma / durdurma ──────────────────────────────────────────────────────


def suggest_tag(adapter: str) -> str:
    base = adapter.lower().replace("_", "-").replace("hektor-lora-", "hektor-")
    return f"{base}-aday"[:60]


def conversion_cmd(adapter: str, tag: str, template_from: str) -> list[str]:
    s = get_settings()
    return [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(s.root / "scripts" / "adapter_to_ollama.ps1"),
        "-Adapter",
        adapter,
        "-OllamaName",
        tag,
        "-TemplateFrom",
        template_from,
        "-Tools",
        str(tools_dir()),
    ]


def comparison_cmd(
    *, question_set: str, active: str, candidate: str, base: str, adapter: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "app.main",
        "compare-run",
        "--set",
        question_set,
        "--active",
        active,
        "--candidate",
        candidate,
        "--base",
        base,
        "--adapter",
        adapter,
    ]


def _validate_names(adapter: str, tag: str) -> None:
    if not ADAPTER_RE.match(adapter or ""):
        raise JobError("Geçersiz adapter adı.")
    if not (get_settings().adapters_dir / adapter).is_dir():
        raise JobError(f"Adapter klasörü yok: models/adapters/{adapter}")
    if not TAG_RE.match(tag or ""):
        raise JobError("Geçersiz Ollama etiketi (küçük harf, rakam, . _ -; en çok 61).")


def _existing_tag(tag: str) -> bool:
    from app.feedback.model_identity import match_entry, ollama_tags

    tags = ollama_tags()
    return bool(tags is not None and match_entry(tags, tag))


def _chat_tag(slot: str) -> str:
    """Sohbet yuvasının etiketi; etkinleştirme reddi (ör. ayardaki pilot ana model, E-8a) ham
    500 değil kullanıcıya gösterilen iş hatası olur."""
    from app.feedback.model_activation import ActivationError, resolve_chat_tag

    try:
        return resolve_chat_tag(slot)
    except ActivationError as exc:
        raise JobError(str(exc)) from exc


def start_job(
    kind: str,
    *,
    adapter: str,
    request_id: str,
    params: dict[str, Any],
    spawn: Any = None,
) -> dict[str, Any]:
    """(İnsan) Ağır işi başlat. Aynı ``request_id`` → aynı iş; koşan iş varken yeni iş yok."""
    from app.feedback.resource_guard import chat_lease_blocker
    from app.training import resource_lock

    if kind not in KINDS:
        raise JobError(f"Bilinmeyen iş türü: {kind}")
    request_id = (request_id or "").strip()
    if not 8 <= len(request_id) <= 80:
        raise JobError("Geçersiz istek kimliği.")
    tag = str(params.get("ollama_tag") or "")
    _validate_names(adapter, tag)
    with _start_mutex():
        for j in list_jobs(limit=500):
            if j.get("request_id") == request_id:
                return {**j, "replayed": True}
        cap = capability(kind, adapter)
        if not cap["supported"]:
            raise JobError("Bu makinede başlatılamaz: " + " | ".join(cap["reasons"]))
        busy = running_job()
        if busy:
            raise JobError(
                f"Başka bir ağır iş sürüyor ({busy['kind_label']} · {busy['job_id']}); bitmesini "
                "bekleyin ya da güvenle durdurun."
            )
        lock = resource_lock.blocker()
        if lock:
            raise JobError(lock)
        lease = chat_lease_blocker()
        if lease:
            raise JobError(lease)
        if kind == "conversion":
            if _existing_tag(tag):
                raise JobError(
                    f"Ollama etiketi '{tag}' zaten var — mevcut etiketin üzerine yazılmaz; "
                    "yeni bir aday etiketi seçin."
                )
            live = {_chat_tag("main"), _chat_tag("trial")}
            if tag in live or f"{tag}:latest" in live:
                raise JobError("Bu etiket şu an sohbette kullanılan model — üzerine yazılmaz.")
            template = str(params.get("template_from") or DEFAULT_TEMPLATE)
            if not TAG_RE.match(template.removesuffix(":latest")) and ":" not in template:
                raise JobError("Geçersiz şablon etiketi.")
            cmd = conversion_cmd(adapter, tag, template)
            meta: dict[str, Any] = {"ollama_tag": tag, "template_from": template}
            if params.get("then_compare"):
                # Tek tıklamalı döngü (docs/TASARIM_SUREKLI_DONGU.md): dönüşüm DOĞRULANINCA
                # karşılaştırma aynı kurallarla (kilit, kira, etiket kontrolü) başlatılır.
                meta["then_compare"] = True
                meta["question_set"] = str(params.get("question_set") or DEFAULT_SET)
        else:
            active = str(params.get("active") or _chat_tag("main"))
            base = str(params.get("base") or DEFAULT_TEMPLATE)
            qset = str(params.get("question_set") or DEFAULT_SET)
            qpath = Path(qset) if Path(qset).is_absolute() else get_settings().root / qset
            if not qpath.is_file():
                raise JobError(f"Soru seti yok: {qset}")
            if not _existing_tag(tag):
                raise JobError(f"Aday etiketi Ollama'da yok: {tag} — önce Ollama'ya hazırlayın.")
            for t in (active, base):
                if not _existing_tag(t):
                    raise JobError(f"Ollama etiketi yok: {t}")
            cmd = comparison_cmd(
                question_set=str(qpath), active=active, candidate=tag, base=base, adapter=adapter
            )
            meta = {"ollama_tag": tag, "active": active, "base": base, "question_set": str(qpath)}
        job_id = "job_" + secrets.token_hex(6)
        log = jobs_dir() / f"{job_id}.log"
        job: dict[str, Any] = {
            "job_id": job_id,
            "kind": kind,
            "adapter": adapter,
            "request_id": request_id,
            "params": meta,
            "cmd": cmd,
            "status": "starting",
            "created_at": utcnow(),
            "log_path": str(log),
            "platform": sys.platform,
        }
        _write(_path(job_id), job)
        pid, ctime = (spawn or _spawn_runner)(job_id)
        with _job_lock(job_id):
            cur = _read(_path(job_id))
            # T-3: okunamadıysa bayat 'starting' görüntüsünü YAZMA (çalıştırıcının yazdığı
            # sonuç ezilirdi); çalıştırıcı kendi pid'ini zaten kaydeder.
            if cur is not None and cur.get("runner_pid") is None:
                cur.update(runner_pid=pid, runner_create_time=ctime)
                _write(_path(job_id), cur)
        job = cur if cur is not None else job
    return reconcile(job)


def _popen_detached(cmd: list[str], err_path: Path) -> subprocess.Popen[bytes]:
    err_fh = err_path.open("ab")
    kwargs: dict[str, Any] = {"cwd": str(PROJECT_ROOT), "env": os.environ.copy(), "close_fds": True}
    kwargs.update(stdin=subprocess.DEVNULL, stdout=err_fh, stderr=err_fh)
    try:
        if os.name == "nt":
            # Gizli konsol (DETACHED_PROCESS DEĞİL): torunlar da pencere açamaz — bkz.
            # app/procutil.py. Mümkünse iş nesnesinden de ayrıl.
            try:
                return subprocess.Popen(
                    cmd, creationflags=DETACHED_HIDDEN | CREATE_BREAKAWAY_FROM_JOB, **kwargs
                )
            except OSError:  # iş nesnesi ayrılmaya izin vermiyor
                return subprocess.Popen(cmd, creationflags=DETACHED_HIDDEN, **kwargs)
        return subprocess.Popen(cmd, start_new_session=True, **kwargs)
    finally:
        err_fh.close()


def _spawn_runner(job_id: str) -> tuple[int, float | None]:
    """Çalıştırıcıyı web sunucusunun süreç AĞACININ DIŞINDA başlat (çift başlatma).

    Kısa ömürlü ara başlatıcı çalıştırıcıyı doğurup hemen çıkar → çalıştırıcının ebeveyni
    ölmüş olur; sunucu ağacıyla kapatılan (``taskkill /T``) bir web süreci işi öldürmez.
    Çalışma dizini KOD köküdür (veri kökü değil): ``-m app...`` yanlış paketi içe aktarmasın.
    Çalıştırıcının kendi hataları ayrı günlüğe yazılır.
    """
    from app.training.resource_lock import process_create_time

    err = jobs_dir() / f"{job_id}.runner.log"
    launcher = _popen_detached(
        [sys.executable, "-m", "app.training.candidate_jobs", "spawn", job_id], err
    )
    try:
        launcher.wait(timeout=120)
    except subprocess.TimeoutExpired:
        launcher.kill()
    job = _read(_path(job_id)) or {}
    pid = job.get("runner_pid")
    if isinstance(pid, int):
        return pid, job.get("runner_create_time")
    return launcher.pid, process_create_time(launcher.pid)  # ara başlatıcı çöktüyse uzlaştırma


def _spawn_child(job_id: str) -> int:
    """Ara başlatıcı: çalıştırıcıyı doğur, kimliğini yaz, hemen çık."""
    from app.training.resource_lock import process_create_time

    proc = _popen_detached(
        [sys.executable, "-m", "app.training.candidate_jobs", "run", job_id],
        jobs_dir() / f"{job_id}.runner.log",
    )
    _update(job_id, runner_pid=proc.pid, runner_create_time=process_create_time(proc.pid))
    return 0


def stop_job(job_id: str, reason: str = "") -> dict[str, Any]:
    """(İnsan) Güvenli durdur: süreç ağacı sonlanır; kısmi çıktı geçerli aday sayılmaz."""
    job = get_job(job_id)
    if job["status"] not in ACTIVE:
        return {**job, "note": "İş zaten bitmiş."}
    reason = (reason or "kullanıcı durdurdu")[:300]
    fresh = _transition(job_id, ACTIVE, status="stopping", stop_reason=reason)
    if fresh is None:
        return {**get_job(job_id), "note": "İş zaten bitmiş."}  # o arada sonuç yazıldı
    # PID'ler geçişin TAZE kaydından: ilk anlık görüntüden sonra çalıştırıcı "running"i
    # sahiplenmiş olabilir. "starting" iken durdurulan çalıştırıcı komutu hiç başlatmaz (``run``).
    job = fresh
    killed: list[int] = []
    unverified: list[int] = []
    for key in ("child", "runner"):
        pid = job.get(f"{key}_pid")
        ctime = job.get(f"{key}_create_time")
        if not isinstance(pid, int):
            continue
        # Başlangıç zamanı kayıtlı değilse PID'in hâlâ bu işe ait olduğu doğrulanamaz (yeniden
        # kullanılmış olabilir) → öldürme. Kayıtlı zaman tutmuyorsa PID başka sürece geçmiştir.
        if not isinstance(ctime, int | float):
            unverified.append(pid)
        elif _alive(pid, ctime):
            killed += _kill_tree(pid)
    _after_stop(job)
    _transition(
        job_id,
        ("stopping", "stopped"),
        status="stopped",
        finished_at=utcnow(),
        killed_pids=killed,
        unverified_pids=unverified,
    )
    return get_job(job_id)


def _after_stop(job: dict[str, Any]) -> None:
    """Durdurulan karşılaştırmanın yarım manifestini "kesildi" işaretle (geçerli değil)."""
    if job.get("kind") != "comparison":
        return
    with contextlib.suppress(OSError):
        tail = Path(job.get("log_path") or "").read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"KARŞILAŞTIRMA\s+(cmp_[0-9a-f]+)", tail)
        if m:
            from app.evals.candidate_compare import mark_interrupted

            mark_interrupted(m[-1], "iş durduruldu")


# ── çalıştırıcı (ayrık süreç) ────────────────────────────────────────────────


def _post_verify(job: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    if job["kind"] == "conversion":
        from app.training.candidate_checks import verify_conversion

        v = verify_conversion(job["adapter"], job["params"]["ollama_tag"])
        bad = [c["key"] for c in v["checks"] if not c["ok"]]
        return v["ok"], ("dönüşüm doğrulaması geçmedi: " + ", ".join(bad)) if bad else "", v
    tail = Path(job["log_path"]).read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"KARŞILAŞTIRMA\s+(cmp_[0-9a-f]+)", tail)
    if not m:
        return False, "karşılaştırma kimliği günlükte yok", {}
    from app.evals.candidate_compare import result

    r = result(m[-1])
    st = (r.get("manifest") or r).get("status")
    ok = st in ("generated", "reviewed", "decided")
    return (
        ok,
        "" if ok else f"karşılaştırma üretimi tamamlanmadı (durum {st})",
        {
            "comparison_id": m[-1],
            "status": st,
        },
    )


def run(job_id: str) -> int:
    from app.training.resource_lock import process_create_time

    job = _read(_path(job_id))
    if job is None:
        return 2
    log = Path(job["log_path"])
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # Komutu başlatmadan ÖNCE "starting → running" geçişini kilit altında sahiplen: o arada
    # durdurma geldiyse (stopping/stopped) komut hiç başlamaz — arayüz "durduruldu" derken iş
    # arka planda koşmaz. Çalıştırıcı kendi kimliğini de yazar (başlatıcının yazımıyla
    # yarışmasın; durdurma bu kimlikle ağacı bulur).
    claimed = _transition(
        job_id,
        ("starting",),
        status="running",
        started_at=utcnow(),
        runner_pid=os.getpid(),
        runner_create_time=process_create_time(os.getpid()),
    )
    if claimed is None:
        _transition(job_id, ("stopping",), status="stopped", finished_at=utcnow())
        return 0
    with log.open("ab") as fh:
        fh.write(f"[{utcnow()}] başlıyor: {' '.join(job['cmd'])}\n".encode())
        fh.flush()
        try:
            proc = subprocess.Popen(
                job["cmd"],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=fh,
                stderr=fh,
                creationflags=NO_WINDOW,
            )
        except OSError as exc:
            _update(job_id, status="failed", finished_at=utcnow(), error=f"başlatılamadı: {exc}")
            return 1
        _update(job_id, child_pid=proc.pid, child_create_time=process_create_time(proc.pid))
        rc = proc.wait()
    cur = _read(_path(job_id))
    if cur is None:
        # T-3: eski 'starting' görüntüsüyle karar verme (durdurma görünmez, iş 'failed' olurdu).
        time.sleep(1.0)
        cur = _read(_path(job_id))
    if cur is None:
        raise RuntimeError(f"iş kaydı okunamadı, sonuç yazılmadı: {job_id}")
    if cur.get("status") in ("stopping", "stopped"):
        _update(job_id, status="stopped", exit_code=rc, finished_at=utcnow())
        return rc
    if rc != 0:
        _update(
            job_id,
            status="failed",
            exit_code=rc,
            finished_at=utcnow(),
            error=f"komut başarısız (çıkış {rc}) — ayrıntı günlükte; kısmi çıktı geçerli değil",
        )
        return rc
    try:
        ok, why, detail = _post_verify(cur)
    except Exception as exc:  # doğrulanamayan iş başarılı sayılmaz
        ok, why, detail = False, f"iş sonrası doğrulama çalıştırılamadı: {exc}", {}
    _update(
        job_id,
        status="done" if ok else "failed",
        exit_code=rc,
        finished_at=utcnow(),
        verification=detail,
        error="" if ok else why,
    )
    if ok and cur.get("kind") == "conversion" and (cur.get("params") or {}).get("then_compare"):
        _chain_compare(cur)
    return 0 if ok else 1


def _chain_compare(job: dict[str, Any]) -> None:
    """Doğrulanmış dönüşümün ardından karşılaştırmayı başlat (``start_job`` kurallarıyla).

    Başlatılamazsa (kira, kilit, etiket) sebep iş kaydına yazılır; karşılaştırma elle başlatılır.
    Aynı dönüşüm için istek kimliği sabit → yeniden çalıştırma ikinci iş açmaz."""
    params = job.get("params") or {}
    try:
        nxt = start_job(
            "comparison",
            adapter=job["adapter"],
            request_id=f"chain-{job['job_id']}",
            params={
                "ollama_tag": params.get("ollama_tag"),
                "question_set": params.get("question_set") or DEFAULT_SET,
            },
        )
        _update(job["job_id"], chain={"job_id": nxt["job_id"], "error": ""})
    except Exception as exc:  # zincir hatası dönüşümü başarısız yapmaz
        _update(job["job_id"], chain={"job_id": "", "error": str(exc)[:300]})


def _cli(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) == 2 and args[0] == "run":
        return run(args[1])
    if len(args) == 2 and args[0] == "spawn":
        return _spawn_child(args[1])
    print("kullanım: python -m app.training.candidate_jobs run <job_id>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
