"""resource_lock.py — ağır işler (eğitim / model dönüşümü / karşılaştırma) için ORTAK kilit.

Neden: eğitim, adapter → GGUF dönüşümü ve aday/aktif model karşılaştırması aynı makinede aynı
belleği ister; sohbet cevabı da öyle. Faz 1'de yalnız web başlatma yolu sohbet kirasına
bakıyordu; ``start-train.ps1`` / doğrudan ``hektor train --run`` / dönüşüm / karşılaştırma
bakmıyordu. Bu modül TEK kilit dosyası (``storage/heavy_job.lock``) ile hepsini koordine eder.

Kurallar:
- Edinme ATOMİKTİR (``O_CREAT | O_EXCL``). Aynı anda iki başlatma → yalnız biri kazanır.
- Kilit sahibini taşır: tür, sahip, pid, pid'in başlangıç zamanı (pid yeniden kullanımına
  karşı), makine adı, durum (``launching`` | ``running``) ve rastgele ``token``.
- BAYAT kilit: sahibinin süreci ölmüşse (ya da pid başka bir sürece geçmişse) ya da
  ``launching`` durumu ``LAUNCH_TTL_S``'yi aşmışsa. Bayat kilidi kırmak ayrı, kısa ömürlü bir
  "kırma" mutex'i altında yapılır ve YALNIZ okunan token hâlâ duruyorsa silinir → iki süreç
  aynı bayat kilidi aynı anda kırıp birbirinin yeni kilidini silemez.
- Devir: web başlatması kilidi ``launching`` olarak alır, alt süreci başlatınca pid'i alt
  sürece DEVREDER (``transfer``) ve token'ı ``HEKTOR_HEAVY_LOCK_TOKEN`` ile geçirir; alt süreç
  ``claim`` ile kendi pid'ini yazar. Token'ı tutmayan ``release`` hiçbir şey silmez.
- Başka makinedeki sahip (paylaşılan disk) değerlendirilemez → kilit tutuluyor sayılır.

Sohbet tarafı (``resource_guard``) önce KİRASINI yazar SONRA bu kilide bakar; ağır iş önce bu
kilidi alır SONRA kiralara bakar → ikisi birden ilerleyemez.

CLI (PowerShell betikleri için): ``python -m app.training.resource_lock acquire|release|status``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

KINDS = ("training", "conversion", "comparison")
LAUNCH_TTL_S = 180.0
# Yazar O_EXCL ile dosyayı açıp hemen yazar; bu süreden genç BOŞ/bozuk dosya "yazılıyor"dur.
_PARTIAL_GRACE_S = 5.0
_BREAK_TTL_S = 30.0
TOKEN_ENV = "HEKTOR_HEAVY_LOCK_TOKEN"

_KIND_TR = {"training": "eğitim", "conversion": "model dönüşümü", "comparison": "karşılaştırma"}


def _root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    from app.config import get_settings

    return get_settings().root


def lock_path(root: Path | None = None) -> Path:
    return _root(root) / "storage" / "heavy_job.lock"


def _break_path(root: Path | None = None) -> Path:
    return _root(root) / "storage" / "heavy_job.lock.break"


# ── süreç kimliği ────────────────────────────────────────────────────────────


def process_create_time(pid: int) -> float | None:
    """pid'in başlangıç zamanı (psutil yoksa ya da süreç yoksa None)."""
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def process_tree(pid: int) -> list[Any]:
    """pid'in GERÇEK alt süreçleri + kendisi (``psutil.Process`` listesi; kök EN SONDA).

    ``psutil.Process.children(recursive=True)`` her torunu yalnız KÖKÜN başlangıç zamanıyla
    kıyaslar. Windows'ta ebeveyni ölmüş (yetim) bir sürecin ppid'i bayat kalır; o pid ağaçtaki
    bir alt sürece yeniden verilmişse, kökten sonra doğmuş İLGİSİZ yetim (ör. ayrık başlatılmış
    bir aday-iş çalıştırıcısı) "torun" sayılıp öldürülür. Burada ağaç kat kat yürünür; her
    düğüm DOĞRUDAN ebeveyninin başlangıç zamanıyla kıyaslanır (tek katlı ``children()``
    bunu yapar). psutil yoksa / kök yoksa ``psutil`` hatası yükselir.
    """
    import psutil

    root = psutil.Process(pid)
    found: list[Any] = []
    seen = {root.pid}
    stack = [root]
    while stack:
        parent = stack.pop()
        try:
            kids = parent.children()
        except psutil.Error:
            continue
        for kid in kids:
            if kid.pid not in seen:
                seen.add(kid.pid)
                found.append(kid)
                stack.append(kid)
    return [*found, root]


def pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False


def owner_alive(info: dict[str, Any]) -> bool:
    """Kaydın sahibi hâlâ yaşıyor mu (aynı makinede, aynı süreç)?"""
    if str(info.get("host") or "") not in ("", socket.gethostname()):
        return True  # başka makine → değerlendirilemez, tutuluyor say
    pid = info.get("pid")
    if not pid_alive(pid):
        return False
    want = info.get("pid_create_time")
    if isinstance(want, int | float) and isinstance(pid, int):
        got = process_create_time(pid)
        if got is not None and abs(got - float(want)) > 1.0:
            return False  # pid başka bir sürece geçmiş (yeniden kullanım)
    return True


def stale_reason(info: dict[str, Any], now: float | None = None) -> str:
    """Boş dize → kilit geçerli; aksi halde neden bayat olduğu."""
    now = time.time() if now is None else now
    if not owner_alive(info):
        return f"sahip süreç (pid {info.get('pid')}) yaşamıyor"
    if info.get("state") == "launching":
        age = now - float(info.get("acquired_at") or 0.0)
        if age > LAUNCH_TTL_S:
            return f"başlatma {int(age)} sn'de tamamlanmadı (TTL {int(LAUNCH_TTL_S)} sn)"
    return ""


# ── okuma ────────────────────────────────────────────────────────────────────


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def status(root: Path | None = None) -> dict[str, Any]:
    """Salt-okuma durum: ``{held, info, stale_reason}`` (bayat kilit 'held' sayılmaz)."""
    path = lock_path(root)
    if not path.exists():
        return {"held": False, "info": None, "stale_reason": ""}
    info = _read(path)
    if info is None:
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return {"held": False, "info": None, "stale_reason": ""}
        if age <= _PARTIAL_GRACE_S:
            return {"held": True, "info": {"state": "writing"}, "stale_reason": ""}
        return {"held": False, "info": None, "stale_reason": "okunamayan eski kilit dosyası"}
    why = stale_reason(info)
    return {"held": not why, "info": info, "stale_reason": why}


def describe(info: dict[str, Any] | None) -> str:
    if not info:
        return "ağır iş kilidi tutuluyor"
    kind = _KIND_TR.get(str(info.get("kind")), str(info.get("kind") or "?"))
    since = info.get("acquired_at")
    when = time.strftime("%H:%M:%S", time.localtime(float(since))) if since else "?"
    return (
        f"{kind} sürüyor ({info.get('owner') or '?'}, pid {info.get('pid')}, "
        f"durum {info.get('state')}, {when}'den beri)"
    )


def blocker(root: Path | None = None) -> str | None:
    st = status(root)
    if not st["held"]:
        return None
    return "Ağır iş kilidi: " + describe(st["info"])


# ── yazma ────────────────────────────────────────────────────────────────────


def _write_new(path: Path, info: dict[str, Any]) -> bool:
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.write(fd, json.dumps(info, ensure_ascii=False).encode("utf-8"))
    finally:
        os.close(fd)
    return True


def _rewrite(path: Path, info: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


@contextlib.contextmanager
def _break_mutex(root: Path | None) -> Iterator[bool]:
    bp = _break_path(root)
    got = False
    for _ in range(2):
        try:
            fd = os.open(str(bp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            got = True
            break
        except FileExistsError:
            try:
                if time.time() - bp.stat().st_mtime > _BREAK_TTL_S:
                    bp.unlink()  # kırıcı çöktüyse
                    continue
            except OSError:
                continue
            break
    try:
        yield got
    finally:
        if got:
            with contextlib.suppress(OSError):
                bp.unlink()


def _break_stale(root: Path | None, seen: dict[str, Any] | None) -> bool:
    """Okunan bayat kaydı (aynı token ise) kır. Başkasının taze kilidini ASLA silmez."""
    path = lock_path(root)
    with _break_mutex(root) as got:
        if not got:
            return False
        current = _read(path)
        if seen is None:
            # Okunamayan dosya: hâlâ okunamıyor ve yeterince eskiyse sil.
            if current is not None:
                return False
            try:
                if time.time() - path.stat().st_mtime <= _PARTIAL_GRACE_S:
                    return False
            except OSError:
                return True
        elif current is None or current.get("token") != seen.get("token"):
            return current is None and not path.exists()
        elif not stale_reason(current):
            return False
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return True


def acquire(
    kind: str,
    owner: str,
    *,
    root: Path | None = None,
    pid: int | None = None,
    state: str = "running",
) -> tuple[dict[str, Any] | None, str]:
    """Kilidi al → (kayıt, "") ya da (None, gerekçe). Bayat kilit güvenle kırılır."""
    if kind not in KINDS:
        raise ValueError(f"Geçersiz ağır iş türü: {kind}")
    if state not in ("launching", "running"):
        raise ValueError(f"Geçersiz durum: {state}")
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner_pid = int(pid or os.getpid())
    for _ in range(4):
        info = {
            "token": secrets.token_hex(16),
            "kind": kind,
            "owner": owner[:200],
            "pid": owner_pid,
            "pid_create_time": process_create_time(owner_pid),
            "host": socket.gethostname(),
            "state": state,
            "acquired_at": time.time(),
        }
        if _write_new(path, info):
            return info, ""
        st = status(root)
        if st["held"]:
            return None, "Ağır iş kilidi alınamadı — " + describe(st["info"])
        _break_stale(root, st["info"])
    return None, "Ağır iş kilidi alınamadı (eşzamanlı başlatma ya da kırılamayan bayat kilit)."


def transfer(token: str, pid: int, *, root: Path | None = None, state: str = "running") -> bool:
    """Kilidin sahibini başka bir sürece devret (token tutmalı)."""
    path = lock_path(root)
    info = _read(path)
    if info is None or info.get("token") != token:
        return False
    info.update(
        pid=int(pid),
        pid_create_time=process_create_time(int(pid)),
        state=state,
        transferred_at=time.time(),
    )
    _rewrite(path, info)
    return True


def claim(token: str, *, root: Path | None = None) -> bool:
    """Alt süreç devralır: kilit bu token'a aitse pid'i bu süreç yap."""
    return transfer(token, os.getpid(), root=root, state="running")


def release(token: str | None, *, root: Path | None = None) -> bool:
    """Yalnız token tutuyorsa sil (başkasının kilidine dokunmaz)."""
    if not token:
        return False
    path = lock_path(root)
    info = _read(path)
    if info is None or info.get("token") != token:
        return False
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    return True


class HeavyJobBusy(RuntimeError):
    """Kilit alınamadı ya da sohbet cevabı üretiliyor."""


@contextlib.contextmanager
def hold(
    kind: str, owner: str, *, root: Path | None = None, check_chat: bool = True
) -> Iterator[str]:
    """Kilidi tut (bağlam yöneticisi). Üst süreç aynı kilidi tutuyorsa (``TOKEN_ENV``) devralır.

    Kilit alındıktan SONRA sohbet kiralarına bakılır; cevap üretiliyorsa ``HeavyJobBusy``.
    """
    inherited = os.environ.get(TOKEN_ENV, "").strip()
    if inherited:
        st = status(root)
        if st["held"] and (st["info"] or {}).get("token") == inherited:
            yield inherited
            return
    info, why = acquire(kind, owner, root=root)
    if info is None:
        raise HeavyJobBusy(why)
    token = str(info["token"])
    try:
        if check_chat:
            from app.feedback.resource_guard import chat_lease_blocker

            lease = chat_lease_blocker(root)
            if lease:
                raise HeavyJobBusy(lease)
        yield token
    finally:
        release(token, root=root)


def held_kind(root: Path | None = None) -> str:
    st = status(root)
    if not st["held"]:
        return ""
    return str((st["info"] or {}).get("kind") or "")


# ── CLI (PowerShell betikleri) ───────────────────────────────────────────────


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="resource_lock", description="Ağır iş kilidi")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--kind", required=True, choices=KINDS)
    a.add_argument("--owner", required=True)
    a.add_argument("--pid", type=int, default=0, help="Kilidin bağlanacağı süreç (0 = bu süreç)")
    r = sub.add_parser("release")
    r.add_argument("--token", required=True)
    sub.add_parser("status")
    args = p.parse_args(argv)
    if args.cmd == "acquire":
        info, why = acquire(args.kind, args.owner, pid=args.pid or None)
        if info is None:
            print(why, file=sys.stderr)
            return 9
        from app.feedback.resource_guard import chat_lease_blocker

        lease = chat_lease_blocker()
        if lease:
            release(info["token"])
            print(lease, file=sys.stderr)
            return 9
        print(info["token"])
        return 0
    if args.cmd == "release":
        return 0 if release(args.token) else 1
    print(json.dumps(status(), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - betik girişi
    raise SystemExit(_cli())
