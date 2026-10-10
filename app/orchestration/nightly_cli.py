"""Gece döngüsü CLI: ``hektor gece`` · ``csv-lab`` · ``karantina`` · ``karantina-kaldir``.

``app/main.py`` şişmesin diye ayrı modülde; ``register(app)`` ile ana uygulamaya eklenir.
Tasarım: ``docs/TASARIM_GECE_DONGUSU.md``. Hiçbiri eğitim başlatmaz (Kural 8).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def gece(
    adim: list[str] = typer.Option(
        None, "--adim", "-a", help="Yalnız bu adımlar (hakem, csv, egitim); boş = hepsi."
    ),
) -> None:
    """Gün sonu döngüsü: yerel hakem → karantina → CSV laboratuvarı → eğitim hazırlık raporu.

    Eğitim BAŞLATMAZ, buluta istek göndermez. Rapor: reports/nightly/<tarih>/gece_*.md
    """
    from app.orchestration.nightly import STEPS, run_nightly

    try:
        rep = run_nightly(tuple(adim) if adim else STEPS)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    color = {"done": "green", "partial": "yellow"}.get(str(rep.get("status")), "red")
    console.print(f"[{color}]Gece döngüsü: {rep.get('status')}[/{color}] ({rep.get('seconds')} sn)")
    if rep.get("reason"):
        console.print(rep["reason"])
    for name, out in (rep.get("steps") or {}).items():
        msg = out.get("error") or out.get("skipped") or "tamam"
        console.print(f"  · {name}: {msg}")
    if rep.get("report_md"):
        console.print(f"Rapor: {rep['report_md']}")
    if rep.get("status") not in ("done", "partial"):
        raise typer.Exit(1)


def csv_lab(
    dosya: str = typer.Option("", "--dosya", help="Tek dosya (data/market/raw altındaki ad)."),
    top_k: int = typer.Option(None, "--top-k", help="Doğrulamaya bakacak aday sayısı."),
) -> None:
    """CSV'den gösterge/strateji ADAYI üret; geliştirmede seç, örneklem dışında bir kez doğrula.

    Yeni dosyalar data/market/raw/ altına bırakılır. Maliyet profili/saat dilimi için isteğe
    bağlı yan dosya: <ad>.meta.json {"profile": "kripto_vadeli|kripto_spot|forex_cfd|bist",
    "tz": "UTC", "timeframe": "1h", "costs": {...}}. Final dönemine dokunulmaz.
    """
    from app.trading import csv_lab as lab
    from app.trading.data_quality import DataQualityError
    from app.trading.strategy_testing import StrategyTestError, resolve_data_file

    if dosya:
        try:
            path: Path = resolve_data_file(dosya)
            rep = lab.analyze_file(path, top_k=top_k)
        except (DataQualityError, StrategyTestError) as exc:
            console.print(f"[red]Atlandı:[/red] {exc}")
            raise typer.Exit(1) from exc
        paths = lab._write_report(rep, path.stem)
        console.print(lab.render_markdown(rep))
        console.print(f"Rapor: {paths['md']}")
        return
    res = lab.run_inbox(top_k=top_k)
    t = Table(title=f"CSV laboratuvarı — {res['dir']}")
    for col in ("dosya", "durum", "ayrıntı"):
        t.add_column(col)
    for f in res["files"]:
        sel = (f.get("report") or {}).get("selected") or []
        detail = f.get("reason") or ", ".join(f"{e['name']}={e['verdict']}" for e in sel)
        t.add_row(f["file"], f["status"], detail or "-")
    console.print(t)


def karantina() -> None:
    """Karantinadaki öğrenme adaylarını listele (gece hakemi şüphelendi; eğitime girmezler)."""
    from app.feedback.chat_store import ChatStore

    items = ChatStore().list_candidates(status="quarantined")
    if not items:
        console.print("Karantinada aday yok.")
        return
    t = Table(title=f"Karantina ({len(items)})")
    for col in ("aday", "karar", "model", "gerekçe"):
        t.add_column(col)
    for c in items:
        q = c.get("quarantine") or {}
        t.add_row(
            c["candidate_id"],
            str(q.get("verdict", "")),
            str(q.get("model", "")),
            str(q.get("reason", ""))[:160],
        )
    console.print(t)


def karantina_kaldir(
    aday: str = typer.Argument(..., help="Aday kimliği"),
    gerekce: str = typer.Option(..., "--gerekce", help="En az 10 karakter gerekçe"),
) -> None:
    """(İnsan) Karantinayı gerekçeyle kaldır. Eğitim onayı DEĞİLDİR; aday kurallarına döner."""
    from app.feedback.learning import LearningError, LearningService

    try:
        c = LearningService().lift_quarantine(aday, gerekce)
    except (LearningError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"Karantina kaldırıldı · yeni durum: {c['status']} — {c['status_reason']}")


def register(app: typer.Typer) -> None:
    app.command("gece")(gece)
    app.command("csv-lab")(csv_lab)
    app.command("karantina")(karantina)
    app.command("karantina-kaldir")(karantina_kaldir)
