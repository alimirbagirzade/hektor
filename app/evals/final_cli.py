"""``hektor final-set …`` ve ``hektor rubric-gate`` — final test takımı + rubrik kapısı CLI.

Soru YAZMAZ: sorular ve anahtarlar kullanıcıdan gelir (``evals/final/``, yerel). Bu komutlar
yalnız biçimi gösterir, doğrular, kilitler, karşılaştırma setine dışa aktarır ve puanlanmış
koşulardan kapı kararı (NO-GO / yetersiz kanıt / insan onayı bekliyor) çıkarır.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

final_app = typer.Typer(help="Final test takımı (kilitli, görülmemiş; LLM-30'dan ayrı).")
console = Console()

_TEMPLATE_QUESTION = {
    "id": "final-hesap-01",
    "category": "hesap",
    "subcategory": "sharpe",
    "family": "<aile adı — aynı şablondan türeyen sorular aynı aile>",
    "question": "<SORU METNİ — kullanıcı yazar; hesap sorusunda girdiler metinde açıkça "
    "verilir ve 'Son satıra yalnız sonucu yaz' talimatı bulunur>",
    "evidence": [],
    "written_by": "<kullanıcı adı>",
    "source_provenance": "<kaynak/tarih>",
    "human_verified": True,
}
_TEMPLATE_KEY = {
    "id": "final-hesap-01",
    "rubric_points": ["<doğru cevapta bulunması gereken madde>"],
    "critical_checks": ["anahtara_aykiri_hesap"],
    "numeric": {
        "calc": "sharpe",
        "inputs": {
            "returns": [0.01, -0.005, 0.02],
            "rf_per_period": 0.0,
            "periods_per_year": 252,
            "ddof": 1,
        },
        "tolerance": 0.01,
        "unit": "yıllık Sharpe",
        "human_value": None,
        "sign_insensitive": False,
    },
    "reference_answer": "",
    "verified_by": "<anahtarı doğrulayan kişi>",
}


@final_app.command("format")
def format_cmd() -> None:
    """Kategoriler, hesap sözleşmeleri, kritik kodlar ve satır şablonları (soru içermez)."""
    from app.evals.final_set import (
        CALC_CONVENTIONS,
        CALC_INPUTS,
        CATEGORIES,
        MIN_FAMILIES_PER_CATEGORY,
        MIN_TOTAL,
        all_critical_codes,
        final_root,
    )

    console.print(f"[bold]Final takımı[/bold] — dizin {final_root()} (yerel, git'e girmez)")
    console.print(f"Asgari {MIN_TOTAL} soru; kategori başına ≥{MIN_FAMILIES_PER_CATEGORY} aile.")
    for cat, spec in CATEGORIES.items():
        console.print(
            f"  {cat} ({spec['title']}) ≥{spec['min']}: {', '.join(spec['subcategories'])}"
        )
    console.print("[bold]Hesap türleri[/bold] (anahtar bu tanımla HESAPLANIR):")
    for calc, conv in CALC_CONVENTIONS.items():
        console.print(f"  {calc}({', '.join(CALC_INPUTS[calc])}): {conv}")
    console.print("[bold]Kritik hata kodları[/bold]:")
    for code, text in all_critical_codes().items():
        console.print(f"  {code}: {text}")
    console.print("[bold]questions.jsonl satırı[/bold]:")
    console.print_json(json.dumps(_TEMPLATE_QUESTION, ensure_ascii=False))
    console.print("[bold]keys.jsonl satırı[/bold]:")
    console.print_json(json.dumps(_TEMPLATE_KEY, ensure_ascii=False))


@final_app.command("status")
def status_cmd(as_json: bool = typer.Option(False, "--json")) -> None:
    """Salt-okuma durum: yok / taslak / kilitli / bozuk + kategori sayımları."""
    from app.evals.final_set import final_status

    st = final_status()
    if as_json:
        console.print_json(json.dumps(st, ensure_ascii=False))
        return
    console.print(f"Final takımı: [bold]{st['state']}[/bold]")
    for k in ("n_questions", "n_families", "n_keys", "n_numeric", "lock_sha", "error"):
        if st.get(k):
            console.print(f"  {k}: {st[k]}")
    for cat, subs in (st.get("by_category") or {}).items():
        console.print(f"  {cat}: {sum(subs.values())} {subs}")


@final_app.command("validate")
def validate_cmd() -> None:
    """Kilit şartlarını denetle (≥60 soru, kategori/aile, anahtarlar, LLM-30'dan ayrılık)."""
    from app.evals.final_set import load_final, validate_final
    from app.evals.profile.dataset_loader import DatasetIntegrityError

    try:
        fs = load_final(purpose="audit")
    except DatasetIntegrityError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if fs is None:
        console.print("[yellow]Final takımı yok (evals/final/questions.jsonl).[/yellow]")
        raise typer.Exit(1)
    errors = validate_final(fs)
    for e in errors:
        console.print(f"  [red]✗[/red] {e}")
    if errors:
        console.print(f"[red]{len(errors)} sorun — kilitlenemez.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Geçerli[/green]: {len(fs.questions)} soru — kilitlenebilir.")


@final_app.command("lock")
def lock_cmd() -> None:
    """Doğrulanmış seti kilitle (manifest bir kez yazılır; sonrası değişiklik = geçersiz)."""
    from app.evals.final_set import lock_final
    from app.evals.profile.dataset_loader import DatasetIntegrityError

    try:
        manifest = lock_final()
    except DatasetIntegrityError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Kilitlendi[/green] {manifest['lock_sha'][:16]} · {manifest['counts']}")


@final_app.command("export")
def export_cmd(
    out: Path = typer.Option(None, "--out", help="Varsayılan reports/comparisons/final_sets/"),
) -> None:
    """Kilitli seti ``compare-run --role final`` biçimine aktar (anahtar içerir — YEREL)."""
    from app.config import get_settings
    from app.evals.final_set import current_lock_sha, export_compare_set
    from app.evals.profile.dataset_loader import DatasetIntegrityError

    target = out or (
        get_settings().reports_dir
        / "comparisons"
        / "final_sets"
        / f"final_{current_lock_sha()[:12] or 'kilitsiz'}.jsonl"
    )
    try:
        path = export_compare_set(target)
    except DatasetIntegrityError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Yazıldı[/green] {path}\n"
        "Sıradaki: uv run hektor compare-run --set <bu dosya> --role final --active A "
        "--candidate C --base B --adapter <ad>  (final seti TEK kullanımlıktır)"
    )


def rubric_gate_cmd(
    raw: Path = typer.Argument(..., help="Puanlanmış llm30_run raw.jsonl"),
    candidate: str = typer.Option(..., "--candidate", help="Aday koşul harfi (ör. B ya da D)"),
    reference: str = typer.Option("", "--reference", help="Referans koşul (ör. A ya da C)"),
    min_repeats: int = typer.Option(2, "--min-repeats"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Rubrik + kritik hata kapısı: puanlanmış koşudan NO-GO / yetersiz kanıt / insan onayı.

    ``score_by_dimension`` (RUBRIC boyutları) ve ``critical_errors`` (kod listesi) insan/hakem
    tarafından doldurulmuş olmalı. Adayda tek kritik hata → NO-GO. Otomatik GO YOK.
    """
    from app.evals.rubric_gate import RubricError, format_gate, gate, records_from_llm30_raw

    try:
        res = gate(
            records_from_llm30_raw(raw),
            candidate=candidate,
            reference=reference or None,
            min_repeats=min_repeats,
        )
    except RubricError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    if as_json:
        console.print_json(json.dumps(res, ensure_ascii=False))
    else:
        console.print(format_gate(res))
    if res["decision"] == "NO-GO":
        raise typer.Exit(1)
