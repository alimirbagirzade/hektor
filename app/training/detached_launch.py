"""Detached LoRA eğitim başlatıcı + canlı durum/hazırlık tespiti.

Neden ayrı modül: web sunucusu (veya Claude Code) kapansa da eğitim sürmeli.
`training_manager` eğitimi web süreci içinde subprocess olarak çalıştırır →
web yeniden başlayınca/çökünce eğitim de ölür (storage/auto_lora_state.json'daki
"Eğitim COMPLETED olmadı" hataları bundandı). Bu modül eğitimi **detached** süreç
olarak başlatır: çıktılar log dosyalarına yazılır, ilerleme `training_status()`
tarafından log'dan okunur. start-train.ps1 ile aynı mekanik.

Veri kaynağı tek: `data/lora_sft/lora_sft.jsonl` (sentetik + kart birleşik, ~1266).
Başlatıcı her seferinde bunu train/valid'e böler → `DatasetBuilder` kaynaklı
clobber (train.jsonl'in 0'a düşmesi) başlatmada otomatik onarılır.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from app.config import get_settings

log = logging.getLogger(__name__)

# "HAZIR" rozetinin görünme eşiği (insan yine de tek tıkla onaylar — Kural 8).
MIN_READY_EXAMPLES = 200
# Determinist split (Kural 6) — lora-split ile aynı.
_SPLIT_SEED = 42
_VALID_RATIO = 0.05
# Log'a son yazımdan bu kadar dakika geçmediyse eğitim "canlı" sayılır.
# Yavaş CPU eğitiminde adım ~dakikalar sürer + log seyrek yazılabilir → 45 dk
# (tamamlanma zaten step>=total ile anında algılanır; bu yalnız ara boşluklar için).
_LIVE_LOG_AGE_MIN = 45.0
# Geçerli adapter adı (path traversal/CLI argüman güvenliği): yalnız harf/rakam/_/-.
_ADAPTER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Çift-başlatma yarışını kapatan atomik kilit (log yazılmadan önceki pencere için).
_LAUNCH_LOCK_TTL = 120.0


def _launch_lock_path(root: Path) -> Path:
    return root / "storage" / ".training_launching"


def _acquire_launch_lock(root: Path) -> bool:
    """Atomik kilit al (O_EXCL). Eş-zamanlı ikinci başlatmayı engeller.

    Bayat kilit (> TTL) temizlenir; eğitim sürerken zaten log-tazeliği guard'ı
    (is_running) korur — bu kilit yalnız spawn ile ilk-log arasındaki pencere için.
    """
    lock = _launch_lock_path(root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and (time.time() - lock.stat().st_mtime) > _LAUNCH_LOCK_TTL:
        with contextlib.suppress(Exception):
            lock.unlink()
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.write(fd, str(int(time.time())).encode())
    finally:
        os.close(fd)
    return True


def _release_launch_lock(root: Path) -> None:
    with contextlib.suppress(Exception):
        _launch_lock_path(root).unlink()


def _count_lines(path: Path) -> int:
    """Dosyadaki boş-olmayan satır sayısı (yoksa 0)."""
    if not path.exists():
        return 0
    try:
        return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
    except Exception:
        return 0


def _combined_source(settings) -> Path:
    return settings.root / "data" / "lora_sft" / "lora_sft.jsonl"


def ensure_train_split(settings=None) -> tuple[int, int]:
    """`lora_sft.jsonl` → `jsonl_dir/{train,valid}.jsonl` (determinist).

    Birleşik kaynak doluysa HER ZAMAN yeniden böler (clobber onarımı). Kaynak
    boş/yoksa mevcut train.jsonl'e dokunmaz. (n_train, n_valid) döndürür.
    """
    import random

    s = settings or get_settings()
    src = _combined_source(s)
    lines = (
        [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if src.exists()
        else []
    )
    if not lines:
        # Kaynak yok → eldeki train/valid neyse onu say (bozma).
        return _count_lines(s.jsonl_dir / "train.jsonl"), _count_lines(s.jsonl_dir / "valid.jsonl")

    random.Random(_SPLIT_SEED).shuffle(lines)
    n_valid = max(1, int(len(lines) * _VALID_RATIO)) if len(lines) > 1 else 0
    valid, train = lines[:n_valid], lines[n_valid:]

    jd = s.jsonl_dir
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "train.jsonl").write_text("\n".join(train) + ("\n" if train else ""), encoding="utf-8")
    (jd / "valid.jsonl").write_text("\n".join(valid) + ("\n" if valid else ""), encoding="utf-8")
    log.info("Eğitim verisi bölündü: train=%d valid=%d → %s", len(train), len(valid), jd)
    return len(train), len(valid)


@dataclass
class TrainingSplit:
    """Kanonik eğitim verisi bölme sonucu (eski `DatasetResult` ile alan-uyumlu)."""

    train_path: Path
    valid_path: Path
    n_train: int
    n_valid: int
    content_hash: str


def _hash_jsonl_lines(lines: list[str]) -> str:
    """Train satırlarından kısa içerik hash'i (adapter provenance; 16 hex)."""
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode("utf-8"))
    return h.hexdigest()[:16]


def _auto_register_dataset(settings, src: Path) -> None:
    """Kanonik veri setini (lora_sft.jsonl) ``DatasetVersion`` olarak sürümle (Modül 8).

    Eğitim verisi hazırlandığında otomatik kayıt → ``model-data-registry`` Kural-8 kapısı
    gerçek veriyle dolar. İçerik-hash ile İDEMPOTENT (aynı veri → tek sürüm). BEST-EFFORT:
    registry hatası eğitimi/önizlemeyi BOZMAZ. Onay/terfi yine İNSAN ELİYLE (CLI
    ``registry-promote-dataset``; Kural 8 — burada eğitim BAŞLATILMAZ, terfi YAPILMAZ).
    """
    if _count_lines(src) <= 0:
        return
    with contextlib.suppress(Exception):
        from app.memory.sqlite_store import SqliteStore
        from app.registry import RegistryStore

        RegistryStore(SqliteStore(settings.sqlite_file)).register_dataset_from_file(
            src, name="lora_sft", source_type="sft"
        )


def build_training_split(settings=None) -> TrainingSplit:
    """TEK KANONİK eğitim verisi: `lora_sft.jsonl` → `jsonl_dir/{train,valid}.jsonl`.

    Web uçları (`/api/training/dataset|dry-run|colab-notebook`) artık `DatasetBuilder`
    (SQLite tabanlı, cılız `{prompt,completion}`) yerine BUNU çağırır → üretilen train.jsonl
    daima kanonik `lora_sft.jsonl` ile AYNI format (`{messages}`) ve sayıda olur; iki-hat
    drifti (CLI zengin / web cılız, aynı dosyayı karşılıklı ezme) kökten kalkar. CLI
    `train`/`launch()` ile AYNI `ensure_train_split` yolu kullanılır (tek doğruluk kaynağı).

    Kanonik kaynak yok/boşsa `assemble_sft_lines` (synth + onaylı kart + ~%25 disiplin;
    determinist seed=0) ile BİR KEZ üretilir; üretilen de boşsa dosyaya DOKUNULMAZ
    (clobber guard). Kaynak doluysa olduğu gibi bölünür — CLI'nin kurduğu zengin seti
    web'den ezme riski yok. Eğitim BAŞLATMAZ (CLAUDE.md kural 8).
    """
    s = settings or get_settings()
    src = _combined_source(s)
    if _count_lines(src) == 0:
        # Kanonik kaynak yok → bir kez birleştir (scripts/assemble_sft.py ile aynı yol).
        with contextlib.suppress(Exception):
            from app.training.sft_assembly import assemble_sft_lines

            res = assemble_sft_lines(s, seed=0)
            if res.lines:  # boşsa yazma → mevcut kaynağı/eğitimi bozma
                src.parent.mkdir(parents=True, exist_ok=True)
                src.write_text("\n".join(res.lines) + "\n", encoding="utf-8")

    n_train, n_valid = ensure_train_split(s)
    train_path = s.jsonl_dir / "train.jsonl"
    valid_path = s.jsonl_dir / "valid.jsonl"
    train_lines = (
        [ln for ln in train_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if train_path.exists()
        else []
    )
    _auto_register_dataset(s, src)  # Modül 8: kanonik veri setini sürümle (best-effort, idempotent)
    return TrainingSplit(
        train_path=train_path,
        valid_path=valid_path,
        n_train=n_train,
        n_valid=n_valid,
        content_hash=_hash_jsonl_lines(train_lines) if train_lines else "",
    )


def readiness(settings=None) -> dict:
    """Eğitime hazırlık: gerçek birleşik veri (lora_sft.jsonl) örnek sayısı."""
    s = settings or get_settings()
    n = _count_lines(_combined_source(s))
    if n == 0:  # kaynak yoksa eldeki bölünmüş train'i baz al
        n = _count_lines(s.jsonl_dir / "train.jsonl")
    ready = n >= MIN_READY_EXAMPLES
    if not n:
        label = "veri yok"
    elif ready:
        label = f"{n} örnek hazır"
    else:
        label = f"{n}/{MIN_READY_EXAMPLES} örnek"
    return {"ready": ready, "ready_examples": n, "ready_label": label}


def _detached_status(settings) -> dict | None:
    """Detached/CLI eğitimi log tazeliği + tqdm satırından algıla (yoksa None)."""
    s = settings
    logf = s.root / "logs" / "train-full-err.log"
    if not logf.exists():
        return None
    if (time.time() - logf.stat().st_mtime) / 60.0 >= _LIVE_LOG_AGE_MIN:
        return None
    try:
        txt = logf.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    # tqdm örn: "30/1203 [1:28:58<48:43:23, 149.53s/it]"
    m = re.findall(r"(\d+)/(\d+)\s*\[[^<\]]*<([^,\]]*)", txt)
    if not m:
        return None
    step, total, eta = int(m[-1][0]), int(m[-1][1]), m[-1][2].strip()
    if total <= 0 or step >= total:
        return None
    adapter = "LoRA"
    st = s.root / "storage" / "train_status.json"
    if st.exists():
        with contextlib.suppress(Exception):
            adapter = json.loads(st.read_text(encoding="utf-8")).get("adapter", "LoRA")
    return {
        "running": True,
        "source": "detached",
        "adapter": adapter,
        "step": step,
        "total": total,
        "pct": round(step * 100 / total, 1),
        "eta": eta,
    }


def training_status() -> dict:
    """Birleşik eğitim durumu (üst-bar rozeti + sekme için).

    Sıra: web (training_manager) → detached log → çalışmıyorsa hazırlık bilgisi.
    """
    s = get_settings()
    # 1) Web'den başlatılan (legacy in-process yol)
    try:
        from app.web.training_manager import get_training_manager

        prog = get_training_manager().progress
        state = getattr(getattr(prog, "state", None), "value", "")
        if state == "running":
            return {
                "running": True,
                "source": "web",
                "adapter": getattr(prog, "adapter_name", "LoRA"),
                "step": getattr(prog, "current_iter", 0),
                "total": getattr(prog, "total_iters", 0),
                "pct": round(getattr(prog, "pct", 0.0), 1),
                "eta": "",
            }
    except Exception:
        pass
    # 2) Detached/CLI eğitim
    det = _detached_status(s)
    if det:
        return det
    # 3) Çalışmıyor → hazırlık
    return {"running": False, **readiness(s)}


def is_running() -> bool:
    """Herhangi bir eğitim (web veya detached) canlı mı?"""
    return bool(training_status().get("running"))


def _find_achilles(root: Path) -> list[str] | None:
    """`achilles` konsol betiğini bul → komut öneki (yoksa uv run yedeği)."""
    exe = "achilles.exe" if os.name == "nt" else "achilles"
    # 1) Aktif venv (web sunucusunun python'ı) yanındaki Scripts/bin
    cand = Path(sys.executable).parent / exe
    if cand.exists():
        return [str(cand)]
    # 2) Proje .venv
    sub = "Scripts" if os.name == "nt" else "bin"
    cand2 = root / ".venv" / sub / exe
    if cand2.exists():
        return [str(cand2)]
    # 3) PATH
    found = shutil.which("achilles")
    if found:
        return [found]
    # 4) uv run yedeği
    uv = shutil.which("uv") or str(
        Path(os.environ.get("USERPROFILE", "")) / ".local" / "bin" / "uv.exe"
    )
    if uv and Path(uv).exists():
        return [uv, "run", "--project", str(root), "achilles"]
    return None


def _build_train_cmd(
    base: list[str],
    adapter_name: str,
    iters: int,
    base_model: str | None,
    profile: str | None,
    max_examples: int,
) -> list[str]:
    """Detached eğitim için `achilles train --run …` komut listesini kur (saf, test edilebilir).

    profile TRUTHY ise `--profile` EKLENİR; boş/None ise EKLENMEZ. Bu ayrım kritik:
    profile geçmezse spawn edilen `train --run` PeftTrainConfig varsayılanlarıyla (vanilya:
    assistant_only_loss=False, NEFTune=0, lr=2e-4) koşar → prompt maskeleme FİİLEN kapalı
    kalır (v5 disiplin-regresyon reçetesi). Bu yüzden `launch()` yerel-güvenli bir profili
    varsayılan yapar; bu helper yalnız listeyi kurar.
    """
    cmd = [
        *base,
        "train",
        "--run",
        "--backend",
        "peft",
        "--adapter-name",
        adapter_name,
        "--iterations",
        str(iters),
    ]
    if base_model:
        cmd += ["--base-model", base_model]
    if profile:
        cmd += ["--profile", profile]
    if max_examples > 0:
        cmd += ["--max-examples", str(max_examples)]
    return cmd


def launch(
    adapter_name: str = "achilles_lora",
    iterations: int = 0,
    dtype: str = "bf16",
    base_model: str | None = None,
    profile: str | None = "discipline_safe_local",
    max_examples: int = 0,
) -> dict:
    """Eğitimi DETACHED başlat (web/terminal kapansa da sürer).

    - Veriyi `lora_sft.jsonl`'den yeniden böler (clobber-proof).
    - iterations<=0 → 1 epoch (train örnek sayısı kadar adım).
    - profile: LoRA reçete profili. VARSAYILAN `discipline_safe_local` (maskeli + NEFTune)
      — web butonu / auto_pipeline / CLI-detached çağıranlarının HİÇBİRİ profil geçmediği
      için varsayılan vanilya olsaydı bu yollarla başlatılan eğitim SESSİZCE maskesiz koşar
      ve tam v5 disiplin-regresyonunu üretirdi (Kademe-2 av bulgusu). Güvenli-varsayılan
      olarak yerel maskeli profil seçilir; farklı reçete isteyen çağıran açıkça geçebilir,
      profili tümüyle atlamak isteyen `profile=""`/`None` verebilir.
    - max_examples>0: yalnız N örnekle eğit (CPU'da makul süre). profile içinde de olabilir.
    - {ok, message, adapter} döndürür. Eğitimi GERÇEKTEN başlatır (Kural 8: bu
      çağrı yalnızca açık kullanıcı eylemiyle — buton/onay — tetiklenir).
    """
    s = get_settings()
    root = s.root

    # Adapter adı güvenliği (path traversal / CLI argüman): yalnız [A-Za-z0-9_-].
    if not _ADAPTER_RE.match(adapter_name or ""):
        return {
            "ok": False,
            "message": "Geçersiz adapter adı — yalnız harf, rakam, _ ve - (en çok 64).",
            "adapter": "",
        }

    if is_running():
        return {"ok": False, "message": "Zaten eğitim çalışıyor.", "adapter": ""}

    # Atomik kilit: iki eş-zamanlı istek (çift-tık/retry) çift süreç başlatmasın.
    if not _acquire_launch_lock(root):
        return {
            "ok": False,
            "message": "Eğitim şu an başlatılıyor — lütfen birkaç saniye bekle.",
            "adapter": "",
        }

    spawned = False
    try:
        n_train, _n_valid = ensure_train_split(s)
        if n_train <= 0:
            return {
                "ok": False,
                "message": (
                    "Eğitim verisi yok (lora_sft.jsonl boş). "
                    "Önce sentetik veri üret (synth-qa / lora-cloud-prep)."
                ),
                "adapter": "",
            }

        iters = iterations if iterations > 0 else n_train  # 1 epoch
        base = _find_achilles(root)
        if not base:
            return {
                "ok": False,
                "message": "achilles çalıştırıcısı bulunamadı (venv/uv yok).",
                "adapter": "",
            }

        cmd = _build_train_cmd(base, adapter_name, iters, base_model, profile, max_examples)

        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["ACHILLES_TRAIN_DTYPE"] = dtype
        # Bu yol (auto_pipeline/web buton) onayı ÜST katmanda alır; spawn edilen
        # `achilles train --run` iç onay kapısını atlasın (çift onay olmasın). STOP_ALL
        # iç komutta yine de geçerlidir. Manuel `achilles train --run` bu env'i ALMAZ.
        env["ACHILLES_TRAIN_SUPERVISED"] = "1"

        popen_kwargs: dict = {"cwd": str(root), "env": env, "close_fds": True}
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
            popen_kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000
        else:
            popen_kwargs["start_new_session"] = True

        # Log dosyaları child sürece devredilir (with-bloğu kullanılamaz: Popen
        # detached çalışacağı için handle'lar spawn'dan sonra kapatılır).
        out_f = open(logs / "train-full.log", "ab")  # noqa: SIM115
        err_f = open(logs / "train-full-err.log", "ab")  # noqa: SIM115
        try:
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, **popen_kwargs)
        except (OSError, ValueError) as exc:
            log.exception("Detached eğitim başlatılamadı")
            return {
                "ok": False,
                "message": f"Eğitim başlatılamadı (süreç oluşturulamadı): {exc}",
                "adapter": "",
            }
        finally:
            out_f.close()
            err_f.close()

        (root / "storage").mkdir(parents=True, exist_ok=True)
        # pid kaydı (Phase 2): /api/training/stop detached koşuyu pid ile durdurabilsin.
        (root / "storage" / "train_status.json").write_text(
            json.dumps(
                {
                    "adapter": adapter_name,
                    "dtype": dtype,
                    "iterations": iters,
                    "pid": proc.pid,
                    "started_at": _utcnow_iso(),
                }
            ),
            encoding="utf-8",
        )
        spawned = True
        log.info("Detached eğitim başlatıldı: %s (%d adım, dtype=%s)", adapter_name, iters, dtype)
        # Süreç spawn edildi; canlı kalıp kalmadığı üst-bar rozetinden/log'dan izlenir
        # (Kural 2: "başladı" değil "başlatıldı" + nereden doğrulanacağı belirtilir).
        return {
            "ok": True,
            "message": (
                f"Eğitim başlatıldı (detached): {adapter_name} — "
                f"{n_train} örnek, {iters} adım, {dtype}. "
                "İlerlemeyi üst-bar rozetinden izle."
            ),
            "adapter": adapter_name,
        }
    finally:
        # Spawn başarısızsa kilidi hemen bırak; başarılıysa TTL ile (log-tazeliği
        # guard'ı devralana kadar) tutulur.
        if not spawned:
            _release_launch_lock(root)


# --------------------------------------------------------------------------
# Detached eğitim DURDURMA (Phase 2) — /api/training/stop gerçek durdurma
# --------------------------------------------------------------------------
def _utcnow_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def read_detached_training_status(root: Path | None = None) -> dict:
    """``storage/train_status.json`` içeriğini döndür (yoksa/bozuksa {})."""
    r = root or get_settings().root
    st = r / "storage" / "train_status.json"
    if not st.exists():
        return {}
    try:
        data = json.loads(st.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
        except Exception:
            return False


def is_detached_training_running(root: Path | None = None) -> bool:
    """Detached eğitim süreci canlı mı? Önce pid (varsa), sonra log tazeliği.

    ``root`` verilirse (test/izolasyon) log-tazeliği yedeği de YALNIZ o root'tan
    okunur; yoksa global ayar kökü kullanılır. Eskiden yedek her zaman
    ``get_settings()`` (gerçek makine kökü) okuyup verilen ``root``'u yok sayıyordu
    → ölü-pid testi gerçek makinedeki taze ``logs/train-full-err.log``'u görüp
    yanlış ``True`` dönüyordu (Linux CI'de log olmadığından gizliydi).
    """
    info = read_detached_training_status(root)
    pid = info.get("pid")
    if isinstance(pid, int) and _pid_alive(pid):
        return True
    try:
        settings = SimpleNamespace(root=Path(root)) if root is not None else get_settings()
        return bool(_detached_status(settings))
    except Exception:
        return False


def _terminate_tree(pid: int) -> tuple[bool, str]:
    """Süreci ve (varsa) çocuklarını nazikçe sonlandır; gerekirse kill. (psutil)."""
    try:
        import psutil
    except Exception:
        # psutil yoksa tek-süreç kaba terminate (yine de çalışır).
        try:
            if _pid_alive(pid):
                os.kill(pid, signal.SIGTERM)
                return True, f"SIGTERM pid {pid} (psutil yok)"
            return False, f"pid {pid} zaten ölü — stop_requested"
        except Exception as exc:
            return False, f"stop_requested (terminate hatası: {exc})"
    try:
        if not psutil.pid_exists(pid):
            return False, f"pid {pid} zaten ölü — stop_requested"
        proc = psutil.Process(pid)
        procs = [*proc.children(recursive=True), proc]
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate()
        _gone, alive = psutil.wait_procs(procs, timeout=8)
        for p in alive:
            with contextlib.suppress(Exception):
                p.kill()
        return True, f"terminated pid {pid} (+{len(procs) - 1} çocuk)"
    except Exception as exc:
        return False, f"stop_requested (terminate hatası: {exc})"


def _stop_event(detail: str, terminated: bool) -> None:
    try:
        from app.agents.runtime.tracker import log_system_event

        log_system_event(
            f"Detached eğitim durdurma: {detail}",
            agent_id="lora-trainer",
            level="warning",
            action="train_stop",
            payload={"terminated": terminated},
        )
    except Exception:
        log.debug("stop event yazılamadı", exc_info=True)


def request_stop_detached_training(root: Path | None = None) -> dict:
    """Detached eğitimi durdurmayı iste (Windows/Linux/macOS uyumlu).

    1) ``storage/STOP_TRAINING`` bırak (loop-script'ler + güvenlik sinyali).
    2) ``train_status.json``'dan pid oku; süreç (ve çocukları) canlıysa terminate
       (gerekirse kill) et. pid yok/ölüyse HATA VERME → 'stop_requested' döndür.
    Olay genel akışa yazılır.
    """
    r = root or get_settings().root
    storage = r / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "STOP_TRAINING").write_text(_utcnow_iso(), encoding="utf-8")

    info = read_detached_training_status(r)
    pid = info.get("pid")
    terminated = False
    if isinstance(pid, int) and pid > 0:
        terminated, detail = _terminate_tree(pid)
    else:
        detail = "stop_requested (pid kaydı yok — STOP_TRAINING bırakıldı)"

    info["stop_requested_at"] = _utcnow_iso()
    info["stop_detail"] = detail
    with contextlib.suppress(Exception):
        (storage / "train_status.json").write_text(json.dumps(info), encoding="utf-8")

    _stop_event(detail, terminated)
    return {"ok": True, "stopped": terminated, "detail": detail, "pid": pid}
