"""nightly.py — gün sonu (gece) döngüsü: yerel hakem → karantina → CSV laboratuvarı → rapor.

Tasarım: ``docs/TASARIM_GECE_DONGUSU.md`` (Faz 1). Gözetimsiz koşar (Windows Görev Zamanlayıcı:
``scripts/install-nightly-task.ps1``; elle: ``hektor gece``). Adımlar:

1. **hakem**   — öğrenme adaylarını YEREL Ollama hakemiyle oku; şüpheli/belirsiz → karantina
                 (``app.orchestration.local_judge``). Bulut KULLANILMAZ.
2. **csv**     — ``data/market/raw`` altındaki YENİ CSV'lerden gösterge/strateji adayı
                 (``app.trading.csv_lab``); final dönemine dokunulmaz.
3. **egitim**  — YALNIZ hazırlık raporu (``easy_train.readiness``). Faz 1'de eğitim BAŞLATILMAZ
                 (Kural 8 + Kademe 2 değişmedi); hazırsa "Bekleyen kararlar" tek tık önerir.
4. **rapor**   — ``reports/nightly/<tarih>/gece.{json,md}`` + ``storage/nightly/latest.json``.

Güvenlik: ``storage/STOP_ALL`` (ya da ``STOP_LEARNING``) varsa hiçbir adım koşmaz. Ağır iş
(eğitim/dönüşüm/karşılaştırma) sürerken hakem adımı atlanır (Ollama/GPU yarışı). Aynı anda iki
gece koşusu olmaz (kilit; 6 saatten eski ya da sahibi ölü kilit bayat sayılır). Bir adımın
hatası diğerlerini durdurmaz; rapor her durumda yazılır.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.feedback.chat_store import utcnow

STEPS = ("hakem", "csv", "egitim")
STALE_LOCK_S = 6 * 3600
NOTE = (
    "Gece döngüsü (Faz 1): yerel hakem şüpheyi karantinaya alır, CSV'den ADAY üretir, eğitim "
    "hazırlığını RAPORLAR. Eğitim başlatmaz, terfi etmez, buluta istek göndermez."
)


def _dir() -> Path:
    d = get_settings().state_dir / "nightly"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lock_path() -> Path:
    return _dir() / "gece.lock"


def _acquire_lock() -> str:
    """Kilit al; alınamazsa sebep döner (boş = alındı)."""
    from app.training.resource_lock import pid_alive

    p = _lock_path()
    if p.exists():
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        age = time.time() - float(info.get("t") or 0)
        if age < STALE_LOCK_S and pid_alive(info.get("pid")):
            return f"Başka bir gece koşusu sürüyor (pid {info.get('pid')}, {int(age)} sn)."
        p.unlink(missing_ok=True)  # bayat kilit
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return "Başka bir gece koşusu az önce başladı."
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "t": time.time(), "at": utcnow()}, fh)
    return ""


def _release_lock() -> None:
    _lock_path().unlink(missing_ok=True)


def stop_marker() -> str:
    sd = get_settings().state_dir
    for m in ("STOP_ALL", "STOP_LEARNING"):
        if (sd / m).exists():
            return m
    return ""


def _step_hakem() -> dict[str, Any]:
    from app.orchestration.local_judge import run_judging
    from app.training import resource_lock

    busy = resource_lock.blocker()
    if busy:
        return {"ran": False, "skipped": f"Ağır iş sürüyor — hakem atlandı: {busy}"}
    return run_judging()


def _step_csv() -> dict[str, Any]:
    from app.trading.csv_lab import run_inbox

    return run_inbox()


def _step_egitim() -> dict[str, Any]:
    from app.training.easy_train import readiness

    r = readiness()
    bad = [i for i in r.get("items", []) if not i.get("ok")]
    return {
        "ran": True,
        "ready": not bad,
        "blockers": [f"{i['key']}: {i['detail']}" for i in bad],
        "data_sha256": r.get("data_sha256", ""),
        "started_training": False,
        "note": "Faz 1: eğitim BAŞLATILMAZ. Hazırsa onay 'Bekleyen kararlar'da sizde.",
    }


STEP_FNS: dict[str, Callable[[], dict[str, Any]]] = {
    "hakem": _step_hakem,
    "csv": _step_csv,
    "egitim": _step_egitim,
}


def run_nightly(steps: tuple[str, ...] | list[str] = STEPS) -> dict[str, Any]:
    """Gece döngüsünü bir kez koş; raporu yazar ve döndürür."""
    unknown = [s for s in steps if s not in STEP_FNS]
    if unknown:
        raise ValueError(f"Bilinmeyen adım: {', '.join(unknown)} (izinli: {', '.join(STEPS)})")
    started = utcnow()
    t0 = time.monotonic()
    rep: dict[str, Any] = {"started_at": started, "steps": {}, "note": NOTE}
    stop = stop_marker()
    if stop:
        rep.update(status="stopped", reason=f"{stop} etkin — hiçbir adım koşmadı.")
        return _finish(rep, t0)
    why = _acquire_lock()
    if why:
        rep.update(status="busy", reason=why)
        return _finish(rep, t0, write=False)
    try:
        for name in steps:
            s0 = time.monotonic()
            try:
                out = STEP_FNS[name]()
                out.setdefault("ran", True)
            except Exception as exc:  # bir adım düşerse diğerleri yine koşar
                out = {"ran": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
            out["seconds"] = round(time.monotonic() - s0, 2)
            rep["steps"][name] = out
        errors = [n for n, o in rep["steps"].items() if o.get("error")]
        rep["status"] = "partial" if errors else "done"
    finally:
        _release_lock()
    return _finish(rep, t0)


def _finish(rep: dict[str, Any], t0: float, *, write: bool = True) -> dict[str, Any]:
    rep["finished_at"] = utcnow()
    rep["seconds"] = round(time.monotonic() - t0, 2)
    if write:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        out = get_settings().reports_dir / "nightly" / day
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%H%M%S")
        jp = out / f"gece_{stamp}.json"
        mp = out / f"gece_{stamp}.md"
        rep["report_json"], rep["report_md"] = str(jp), str(mp)
        jp.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), "utf-8")
        mp.write_text(render_markdown(rep), encoding="utf-8")
        (_dir() / "latest.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return rep


def latest() -> dict[str, Any] | None:
    try:
        return json.loads((_dir() / "latest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render_markdown(rep: dict[str, Any]) -> str:
    lines = [
        f"# Gece döngüsü — {rep.get('started_at', '')}",
        "",
        f"Durum: **{rep.get('status')}** · süre {rep.get('seconds')} sn",
        "",
    ]
    if rep.get("reason"):
        lines += [f"> {rep['reason']}", ""]
    st = rep.get("steps", {})
    h = st.get("hakem")
    if h is not None:
        lines.append("## 1 · Yerel hakem (öğrenme adayları)")
        if not h.get("ran"):
            lines.append(f"- Koşmadı: {h.get('skipped') or h.get('error')}")
        else:
            c = h.get("counts", {})
            lines.append(
                f"- Model `{h.get('model')}` · bekleyen {h.get('pending_total')} · yeni yargı "
                f"{h.get('judged_fresh')} · ertesi geceye {h.get('left_for_next_night')}"
            )
            lines.append(
                f"- tutarlı {c.get('tutarli', 0)} · şüpheli {c.get('supheli', 0)} · belirsiz "
                f"{c.get('belirsiz', 0)} · hata {c.get('failed', 0)}"
            )
            for q in h.get("quarantined", []):
                lines.append(f"  - KARANTİNA `{q['candidate_id']}` ({q['verdict']}): {q['reason']}")
        lines.append("")
    c = st.get("csv")
    if c is not None:
        lines.append("## 2 · CSV laboratuvarı")
        if c.get("error"):
            lines.append(f"- Hata: {c['error']}")
        for f in c.get("files", []):
            if f["status"] == "seen":
                continue
            lines.append(f"- `{f['file']}` → {f['status']} {f.get('reason') or ''}".rstrip())
            for e in (f.get("report") or {}).get("selected", []):
                lines.append(
                    f"  - {e['name']}: `{e['verdict']}` · geliştirme Sharpe {e['dev_sharpe']} "
                    f"(DSR {e['dsr']}) · OOS Sharpe {e['val_sharpe']} ({e['val_trades']} işlem)"
                )
            if (f.get("report") or {}).get("md"):
                lines.append(f"  - Rapor: `{f['report']['md']}`")
        if not [f for f in c.get("files", []) if f["status"] != "seen"]:
            lines.append("- Yeni CSV yok.")
        lines.append("")
    e = st.get("egitim")
    if e is not None:
        lines.append("## 3 · Eğitim hazırlığı (yalnız rapor)")
        if e.get("error"):
            lines.append(f"- Hata: {e['error']}")
        elif e.get("ready"):
            lines.append("- Tüm kapılar geçti — eğitim onayı 'Bekleyen kararlar'da sizde.")
        else:
            for b in e.get("blockers", []):
                lines.append(f"- Engel: {b}")
        lines.append("- Eğitim başlatılmadı (Faz 1).")
        lines.append("")
    lines += ["---", NOTE, "Çıktılar hipotez + test noktasıdır; yatırım tavsiyesi değildir."]
    return "\n".join(lines) + "\n"
