"""``hektor surekli-dongu …`` — sürekli eğitim döngüsünün CLI'ı (bkz. continuous_loop.py)."""

from __future__ import annotations

import json

import typer

surekli_app = typer.Typer(
    help="Sürekli eğitim döngüsü: tik (tek adım) · durum · duraklat · devam.",
    no_args_is_help=True,
)


@surekli_app.command("tik")
def tik(as_json: bool = typer.Option(False, "--json")) -> None:
    """Döngünün TEK adımı (Görev Zamanlayıcı bunu tekrarlar). Terfi YAPMAZ."""
    from app.orchestration.continuous_loop import tick

    out = tick()
    typer.echo(
        json.dumps(out, ensure_ascii=False)
        if as_json
        else f"{out['action']}: {out.get('detail', '')}"
    )


@surekli_app.command("durum")
def durum(as_json: bool = typer.Option(False, "--json")) -> None:
    """Salt-okuma: ayarlar, durum, son olaylar."""
    from app.orchestration.continuous_loop import overview

    ov = overview()
    if as_json:
        typer.echo(json.dumps(ov, ensure_ascii=False, indent=2))
        return
    st = ov["state"]
    typer.echo(
        f"Döngü: {'AÇIK' if ov['enabled'] else 'kapalı'} · gözetimsiz eğitim: "
        f"{'AÇIK' if ov['unattended'] else 'kapalı'} · Kademe 2 kod-yalnız: "
        f"{ov['kademe2_code_only']} · profil {ov['profile']} · referans {ov['reference']} · "
        f"hakem {ov['judge']}"
    )
    if st.get("paused"):
        typer.echo(f"⏸ DURAKLATILDI: {st.get('pause_reason')}")
    typer.echo(f"Ardışık başarısız aday: {st.get('consecutive_failures', 0)}")
    if st.get("job"):
        typer.echo(f"Arka plan işi: {st['job'].get('kind')} (pid {st['job'].get('pid')})")
    for e in ov["events"]:
        typer.echo(f"  {e['at'][:19]} {e['action']}: {str(e.get('detail', ''))[:160]}")


@surekli_app.command("duraklat")
def duraklat(gerekce: str = typer.Option(..., "--gerekce")) -> None:
    """Döngüyü duraklat (koşan eğitimi DURDURMAZ)."""
    from app.orchestration.continuous_loop import pause

    typer.echo(json.dumps(pause(gerekce), ensure_ascii=False))


@surekli_app.command("devam")
def devam(gerekce: str = typer.Option(..., "--gerekce")) -> None:
    """Duraklatmayı kaldır (insan); ardışık başarısızlık sayacı sıfırlanır."""
    from app.orchestration.continuous_loop import resume

    typer.echo(json.dumps(resume(gerekce), ensure_ascii=False))
