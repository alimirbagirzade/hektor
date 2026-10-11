"""nightly_publish.py — gece raporunun TEMİZLENMİŞ özetini ayrı bir git dalına yayınla.

Amaç: kullanıcı raporları elle kopyalamasın; bulut tarafındaki sabah rutini (Claude Code)
``gece-raporlari`` dalını okuyup analiz etsin ve gerekirse düzeltme PR'ı açsın.

DEPO HERKESE AÇIK OLABİLİR → yalnız İZİN LİSTESİ alanlar gider (``sanitize``):
sayılar, kimlikler, model adları, CSV dosya ADLARI (yol değil), metrikler ve bu kodun kendi
ürettiği kısa durum metinleri (mutlak yollar ve e-postalar silinir). GİTMEYENLER: sohbet
soruları/cevapları, aday hedef metinleri, hakem gerekçeleri, model cevapları (LLM-30 ham
cevap dahil), kaynak metinleri, makine yolları, kullanıcı adı/e-postası (commit kimliği sabit
``Hektor gece``). main dalına DOKUNULMAZ; yalnız ``gece-raporlari`` dalına commit atılır.

Yayın hatası gece koşusunu bozmaz (raporda ``publish.error`` görünür).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.procutil import NO_WINDOW

BRANCH = "gece-raporlari"
SCHEMA = 1
_IDENT = ["-c", "user.name=Hektor gece", "-c", "user.email=hektor-gece@noreply.local"]

_WIN_PATH = re.compile(r"[A-Za-z]:[\\/][^\s'\"`,;)]*")
_POSIX_PATH = re.compile(
    r"(?<![\w.])/(?:home|Users|root|tmp|var|mnt|opt|srv|private)/[^\s'\"`,;)]*"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def scrub(text: Any, limit: int = 300) -> str:
    """Kendi ürettiğimiz durum metninden yol/e-posta sil, kısalt."""
    s = str(text or "")
    s = _WIN_PATH.sub("<yol>", s)
    s = _POSIX_PATH.sub("<yol>", s)
    s = _EMAIL.sub("<eposta>", s)
    return s[:limit]


_ERR_TYPE = re.compile(r"^\s*([A-Za-z_][\w.]{0,60})\s*:")


def scrub_error(text: Any) -> str:
    """Ham istisna metni YAYINLANMAZ — yalnız tür adı (Kademe 2 N-4).

    ``str(exc)`` SQLAlchemy'de ``[parameters: (...)]`` ile hakem gerekçesini / aday metnini
    taşıyabiliyordu; ``scrub`` yalnız yol/e-posta siler. Tür adı yoksa sabit "hata".
    """
    s = str(text or "").strip()
    if not s:
        return ""
    m = _ERR_TYPE.match(s)
    return m.group(1) if m else "hata"


def _num(v: Any) -> Any:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _counts(d: Any) -> dict[str, int]:
    if not isinstance(d, dict):
        return {}
    return {str(k)[:40]: int(v) for k, v in d.items() if isinstance(v, int)}


def _hakem(h: dict[str, Any]) -> dict[str, Any]:
    if not h:
        return {}
    return {
        "ran": bool(h.get("ran")),
        "skipped": scrub(h.get("skipped")) if h.get("skipped") else scrub_error(h.get("error")),
        "model": scrub(h.get("model") or "", 80),
        "pending_total": _num(h.get("pending_total")),
        "judged_fresh": _num(h.get("judged_fresh")),
        "left_for_next_night": _num(h.get("left_for_next_night")),
        "counts": _counts(h.get("counts")),
        # Yalnız kimlik + karar etiketi; gerekçe (kullanıcı metnini alıntılayabilir) GİTMEZ.
        "quarantined": [
            {
                "candidate_id": scrub(q.get("candidate_id"), 40),
                "verdict": scrub(q.get("verdict"), 12),
            }
            for q in h.get("quarantined") or []
        ],
        "seconds": _num(h.get("seconds")),
    }


def _olcum(o: dict[str, Any]) -> dict[str, Any]:
    if not o:
        return {}
    out: dict[str, Any] = {
        "ran": bool(o.get("ran")),
        "skipped": scrub(o.get("skipped")) if o.get("skipped") else scrub_error(o.get("error")),
        "model": scrub(o.get("model") or "", 80),
        "digest": scrub(o.get("digest") or "", 80)[:19],
        "rag": bool(o.get("rag")),
        "reason": scrub(o.get("reason") or "", 120),
    }
    for k in (
        "n",
        "n_questions",
        "n_errors",
        "n_flagged",
        "key_hits",
        "key_total",
        "avg_output_tokens",
        "seconds",
    ):
        out[k] = _num(o.get(k))
    out["flag_counts"] = _counts(o.get("flag_counts"))
    out["regression"] = [scrub(r, 200) for r in o.get("regression") or []]
    rows = []
    report = o.get("report")
    if report:  # soru başına bayrak + iz; HAM CEVAP GİTMEZ
        try:
            data = json.loads(Path(report).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        for r in data.get("rows") or []:
            keys = r.get("keys") or {}
            rows.append(
                {
                    "question_id": scrub(r.get("question_id"), 20),
                    "flags": [scrub(f, 40) for f in r.get("flags") or []],
                    "keys_hit": sum(1 for v in keys.values() if v),
                    "keys_total": len(keys),
                    "output_tokens": _num(r.get("output_tokens")),
                    "latency_s": _num(r.get("latency_s")),
                    "error": bool(r.get("error")),
                }
            )
    out["per_question"] = rows
    return out


def _csv(c: dict[str, Any]) -> dict[str, Any]:
    if not c:
        return {}
    files = []
    for f in c.get("files") or []:
        rep = f.get("report") or {}
        files.append(
            {
                "file": scrub(Path(str(f.get("file") or "")).name, 80),
                "status": scrub(f.get("status"), 12),
                # Beklenmeyen hata (failed) ham istisna metni taşır → yalnız tür adı (N-4).
                "reason": scrub_error(f.get("reason"))
                if f.get("status") == "failed"
                else scrub(f.get("reason") or ""),
                "timeframe": scrub(rep.get("timeframe") or "", 6),
                "profile": scrub(rep.get("profile") or "", 20),
                "n_trials": _num(rep.get("n_trials")),
                "assumptions": [scrub(a, 200) for a in rep.get("assumptions") or []],
                "selected": [
                    {
                        "strategy_id": scrub(e.get("strategy_id") or "", 40),
                        "name": scrub(e.get("name") or "", 80),
                        "verdict": scrub(e.get("verdict") or "", 24),
                        "dev_sharpe": _num(e.get("dev_sharpe")),
                        "dsr": _num(e.get("dsr")),
                        "val_sharpe": _num(e.get("val_sharpe")),
                        "val_trades": _num(e.get("val_trades")),
                    }
                    for e in rep.get("selected") or []
                ],
            }
        )
    return {"error": scrub_error(c.get("error")), "files": files}


def _egitim(e: dict[str, Any]) -> dict[str, Any]:
    if not e:
        return {}
    return {
        "ran": bool(e.get("ran")),
        "ready": bool(e.get("ready")),
        # Yalnız kapı ANAHTARLARI (ör. "kademe2"); ayrıntı metni yol içerebilir.
        "blocker_keys": [scrub(str(b).split(":", 1)[0], 30) for b in e.get("blockers") or []],
        "started_training": False,
        "error": scrub_error(e.get("error")),
    }


def sanitize(rep: dict[str, Any], *, commit: str = "") -> dict[str, Any]:
    st = rep.get("steps") or {}
    return {
        "schema": SCHEMA,
        "status": scrub(rep.get("status"), 12),
        "reason": scrub(rep.get("reason") or ""),
        "started_at": scrub(rep.get("started_at"), 40),
        "finished_at": scrub(rep.get("finished_at"), 40),
        "seconds": _num(rep.get("seconds")),
        "code_commit": scrub(commit, 12),
        "steps": {
            "hakem": _hakem(st.get("hakem") or {}),
            "olcum": _olcum(st.get("olcum") or {}),
            "csv": _csv(st.get("csv") or {}),
            "egitim": _egitim(st.get("egitim") or {}),
        },
        "note": "Temizlenmiş gece özeti (Hektor). Metin/cevap/yol içermez; yatırım "
        "tavsiyesi değildir.",
    }


# ── git ──────────────────────────────────────────────────────────────────────


def _git(args: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=NO_WINDOW,
        # Kademe 2 N-6: zamanlanmış görevde kimlik eksikse GCM gece masaüstüne giriş penceresi
        # açıp 300 sn bekleyebiliyordu → etkileşimsiz: hızlı ve sessiz başarısız olsun.
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )


def publish(summary: dict[str, Any], *, source_root: Path | None = None) -> dict[str, Any]:
    """Özeti ``gece-raporlari`` dalına yaz + push et. Hata fırlatmaz; sonuç sözlüğü döner."""
    s = get_settings()
    src = source_root or s.source_root
    wt = s.state_dir / "nightly" / "publish_repo"
    try:
        url = _git(["remote", "get-url", "origin"], src).stdout.strip()
        if not url:
            return {"ok": False, "error": "origin uzak deposu yok"}
        if not (wt / ".git").is_dir():
            wt.mkdir(parents=True, exist_ok=True)
            _git(["init", "-q"], wt)
            _git(["remote", "add", "origin", url], wt)
        else:
            _git(["remote", "set-url", "origin", url], wt)
        fetched = _git(["fetch", "-q", "--depth", "1", "origin", BRANCH], wt, timeout=300)
        if fetched.returncode == 0:
            _git(["checkout", "-q", "-B", BRANCH, "FETCH_HEAD"], wt)
        else:  # dal henüz yok → yetim dal
            _git(["checkout", "-q", "--orphan", BRANCH], wt)
            _git(["rm", "-rfq", "--cached", "."], wt)
        day = str(summary.get("started_at") or "")[:10] or "bilinmeyen"
        stamp = re.sub(r"[^0-9]", "", str(summary.get("started_at") or ""))[8:14] or "000000"
        body = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        rel = Path("raporlar") / day / f"gece_{stamp}.json"
        (wt / rel).parent.mkdir(parents=True, exist_ok=True)
        (wt / rel).write_text(body, encoding="utf-8")
        (wt / "son.json").write_text(body, encoding="utf-8")
        readme = wt / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Hektor gece raporları\n\nOtomatik, temizlenmiş gece özetleri "
                "(`app/orchestration/nightly_publish.py`). Metin/cevap/yol içermez.\n",
                encoding="utf-8",
            )
        _git(["add", "-A"], wt)
        msg = f"gece: {day} {summary.get('status', '')}"
        c = _git([*_IDENT, "commit", "-q", "-m", msg], wt)
        if c.returncode != 0 and "nothing to commit" not in (c.stdout + c.stderr):
            return {"ok": False, "error": scrub(c.stderr or c.stdout)}
        p = _git(["push", "-q", "origin", f"HEAD:refs/heads/{BRANCH}"], wt, timeout=300)
        if p.returncode != 0:
            return {"ok": False, "error": scrub(p.stderr or p.stdout)}
        return {"ok": True, "branch": BRANCH, "path": rel.as_posix()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": scrub(f"{type(exc).__name__}: {exc}")}


def code_commit(source_root: Path | None = None) -> str:
    try:
        r = _git(["rev-parse", "--short=12", "HEAD"], source_root or get_settings().source_root)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""
