"""``hektor bilgi …`` ve ``hektor egitim-dongusu …`` — bilgi kalemi defteri + eğitim döngüsü CLI.

Hiçbir komut eğitim, değerlendirme ya da veri üretimi BAŞLATMAZ. ``bilgi incele`` yerel hakemi
(Ollama) çağırır; eğitim/ağır iş sürerken çalışmaz (ortak kilit + durdurma işaretleri).
İnsan kararı (``onayla`` / ``reddet`` / ``duzelt``) gerekçe ve karar verenin adını ister.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

bilgi_app = typer.Typer(
    help="Kaynaklardan gelen bilgi kalemleri: inceleme, karantina, onay, sürüm."
)
loop_app = typer.Typer(help="Kontrollü eğitim döngüsü: bitiş doğrulama, değerlendirme, kapı.")
console = Console()


def _fail(exc: Exception) -> None:
    console.print(f"[red]{exc}[/red]")
    raise typer.Exit(2) from exc


@bilgi_app.command("durum")
def bilgi_durum(as_json: bool = typer.Option(False, "--json")) -> None:
    """Kalem sayıları, son kalemler, veri sürümleri, seçim ve GPT bağlantısı (salt-okuma)."""
    from app.research.knowledge_review import summary

    s = summary(limit=20)
    if as_json:
        console.print_json(json.dumps(s, ensure_ascii=False))
        return
    console.print(
        f"Toplam {s['total']} kalem · " + " · ".join(f"{k} {v}" for k, v in s["counts"].items())
    )
    gpt = s["gpt"]
    console.print(f"GPT bağlantısı: {'var' if gpt['available'] else 'YOK'} — {gpt['detail']}")
    console.print(f"Seçili sürüm: {(s['selection'] or {}).get('version_id', '(yok)')}")
    for it in s["items"]:
        console.print(
            f"  {it['item_id']} [{it['status']}] {it['question'][:80]} — "
            f"{it['status_reason'][:120]}"
        )


@bilgi_app.command("iceri-al")
def bilgi_iceri_al() -> None:
    """Araştırma staging satırlarını kaleme çevir (idempotent; LLM çağırmaz)."""
    from app.research.knowledge_review import import_staging

    console.print_json(json.dumps(import_staging(), ensure_ascii=False))


@bilgi_app.command("incele")
def bilgi_incele(limit: int = typer.Option(6, "--limit", min=1, max=50)) -> None:
    """Bekleyen kalemleri deterministik ölçütler + YEREL hakemle incele.

    Ağır iş (eğitim/dönüşüm/karşılaştırma) sürerken çalışmaz — ortak kilit."""
    from app.research.knowledge_review import review_pending
    from app.training.resource_lock import HeavyJobBusy, hold

    try:
        with hold("comparison", "knowledge_review"):
            res = review_pending(limit=limit)
    except HeavyJobBusy as exc:
        console.print(f"[yellow]Ağır iş sürüyor — inceleme yapılmadı: {exc}[/yellow]")
        raise typer.Exit(1) from exc
    console.print_json(json.dumps(res, ensure_ascii=False))


@bilgi_app.command("goster")
def bilgi_goster(item_id: str) -> None:
    """Tek kalemin tüm kaydı: kaynak satır, sürümler, incelemeler, kararlar."""
    from app.research.knowledge_review import KnowledgeError, load_item

    try:
        console.print_json(json.dumps(load_item(item_id), ensure_ascii=False))
    except KnowledgeError as exc:
        _fail(exc)


@bilgi_app.command("duzelt")
def bilgi_duzelt(
    item_id: str,
    dosya: Path = typer.Option(..., "--dosya", help="Düzeltilmiş cevap metni (UTF-8)"),
    gerekce: str = typer.Option(..., "--gerekce"),
    kim: str = typer.Option(..., "--kim", help="Düzeltmeyi yapan kişi"),
) -> None:
    """Düzeltmeyi AYRI sürüm olarak ekle (orijinal korunur; yeniden inceleme gerekir)."""
    from app.research.knowledge_review import KnowledgeError, correct

    try:
        it = correct(item_id, dosya.read_text(encoding="utf-8"), gerekce, author=kim)
    except KnowledgeError as exc:
        _fail(exc)
    console.print(f"{it['item_id']} → sürüm {len(it['revisions']) - 1}, durum {it['status']}")


def _decide(item_id: str, decision: str, gerekce: str, kim: str) -> None:
    from app.research.knowledge_review import KnowledgeError, decide

    try:
        it = decide(item_id, decision, gerekce, decided_by=kim)
    except KnowledgeError as exc:
        _fail(exc)
    console.print(f"{it['item_id']} → {it['status']} ({it['status_reason'][:160]})")


@bilgi_app.command("onayla")
def bilgi_onayla(
    item_id: str,
    gerekce: str = typer.Option(..., "--gerekce"),
    kim: str = typer.Option(..., "--kim", help="Kalemi bizzat inceleyen kişi"),
) -> None:
    """Gerekçeli İNSAN onayı (yalnız bizzat incelediyseniz; otomatik çürütmeyi geçemez)."""
    _decide(item_id, "onay", gerekce, kim)


@bilgi_app.command("reddet")
def bilgi_reddet(
    item_id: str,
    gerekce: str = typer.Option(..., "--gerekce"),
    kim: str = typer.Option(..., "--kim"),
) -> None:
    """Gerekçeli İNSAN reddi."""
    _decide(item_id, "ret", gerekce, kim)


@bilgi_app.command("surum")
def bilgi_surum(not_: str = typer.Option("", "--not")) -> None:
    """Onaylı kalemlerden değişmez veri sürümü oluştur (eğitim verisine henüz GİRMEZ)."""
    from app.research.knowledge_review import KnowledgeError, create_version

    try:
        m = create_version(not_)
    except KnowledgeError as exc:
        _fail(exc)
    console.print(f"{m['version_id']}: {m['n']} kalem · sha {m['train_sha256'][:12]}")
    console.print(
        f"Seçmek için: uv run hektor bilgi sec {m['version_id']} → sonra "
        "uv run python scripts/assemble_sft.py (kalite/sızıntı kapıları yeniden koşar)."
    )


@bilgi_app.command("sec")
def bilgi_sec(version_id: str) -> None:
    """Sürümü kanonik birleştirmeye seç (data/lora_sft/research_selection.json)."""
    from app.research.knowledge_review import KnowledgeError, select_version

    try:
        sel = select_version(version_id)
    except KnowledgeError as exc:
        _fail(exc)
    console.print_json(json.dumps(sel, ensure_ascii=False))


@bilgi_app.command("secimi-kaldir")
def bilgi_secimi_kaldir() -> None:
    """Bilgi sürümü seçimini kaldır (kanonik birleştirme öncekiyle aynı olur)."""
    from app.research.knowledge_review import clear_selection

    console.print("kaldırıldı" if clear_selection() else "seçim yoktu")


# ── eğitim döngüsü ───────────────────────────────────────────────────────────


@loop_app.command("durum")
def loop_durum(as_json: bool = typer.Option(False, "--json")) -> None:
    """Adaylar: bitiş doğrulaması, geliştirme değerlendirmesi, sonraki adım (salt-okuma*).

    *Koşan/bitmiş eğitimin durum dosyasındaki adayı deftere ekler (yalnız kayıt)."""
    from app.orchestration.training_loop import overview

    o = overview()
    if as_json:
        console.print_json(json.dumps(o, ensure_ascii=False))
        return
    for c in o["candidates"]:
        comp = c["completion"] or {}
        dev = c["dev_eval"] or {}
        console.print(
            f"{c['adapter']} · veri {str(c['data_sha256'])[:12]} · "
            f"bitiş {comp.get('ok', '—')} · değerlendirme {dev.get('decision', '—')}"
            f"/{dev.get('passed', '—')} → {c['next_step']}"
        )


@loop_app.command("bitis")
def loop_bitis(adapter: str) -> None:
    """Bitişi doğrula: çıkış kodu + final durum + adım + ağırlık dosyaları (+ özet) → kaydet."""
    from app.orchestration.training_loop import TrainingLoopError, record_completion

    try:
        rec = record_completion(adapter)
    except TrainingLoopError as exc:
        _fail(exc)
    comp = rec["completion"]
    for c in comp["checks"]:
        console.print(f"  {'✓' if c['ok'] else '✗'} {c['key']}: {c['detail']}")
    console.print(f"BİTİŞ: {'DOĞRULANDI' if comp['ok'] else 'DOĞRULANAMADI'}")
    if not comp["ok"]:
        raise typer.Exit(1)


@loop_app.command("degerlendirme")
def loop_degerlendirme(
    adapter: str,
    aday_raw: Path = typer.Option(..., "--aday-raw", help="Puanlanmış llm30_run raw.jsonl"),
    aday_kosul: str = typer.Option("B", "--aday-kosul"),
    onceki_raw: Path = typer.Option(..., "--onceki-raw", help="Önceki modelin puanlı raw.jsonl'i"),
    onceki_kosul: str = typer.Option("B", "--onceki-kosul"),
    min_repeats: int = typer.Option(2, "--min-repeats"),
) -> None:
    """Puanlanmış geliştirme koşusundan aday ↔ önceki model kapısını hesapla ve KAYDET."""
    from app.evals.rubric_gate import RubricError
    from app.orchestration.training_loop import TrainingLoopError, record_dev_eval

    try:
        rec = record_dev_eval(
            adapter,
            candidate_raw=aday_raw,
            candidate_condition=aday_kosul,
            reference_raw=onceki_raw,
            reference_condition=onceki_kosul,
            min_repeats=min_repeats,
        )
    except (TrainingLoopError, RubricError) as exc:
        _fail(exc)
    dev = rec["dev_eval"]
    console.print_json(json.dumps(dev, ensure_ascii=False))
    if not dev["passed"]:
        raise typer.Exit(1)


@loop_app.command("kapi")
def loop_kapi() -> None:
    """Şu anki veriyle yeni (sıfırdan) eğitime döngü kuralları izin veriyor mu? (salt-okuma*)"""
    from app.config import get_settings
    from app.orchestration.training_loop import next_training_blockers
    from app.training.detached_launch import _combined_source
    from app.training.train_guard import sha256_file

    problems = next_training_blockers(data_sha=sha256_file(_combined_source(get_settings())) or "")
    if problems:
        for p in problems:
            console.print(f"[red]⛔ {p}[/red]")
        raise typer.Exit(1)
    console.print("Döngü kuralları yeni adaya izin veriyor (diğer kapılar ayrıca uygulanır).")


@loop_app.command("serbest-birak")
def loop_serbest_birak(adapter: str, gerekce: str = typer.Option(..., "--gerekce")) -> None:
    """Başarısız/regresyonlu adaydan sonra devam için TAZE insan onayı iste/tüket."""
    from app.orchestration.training_loop import TrainingLoopError, release

    try:
        res = release(adapter, gerekce)
    except TrainingLoopError as exc:
        _fail(exc)
    console.print_json(json.dumps(res, ensure_ascii=False, default=str))


@loop_app.command("tanik")
def loop_tanik(adapter: str) -> None:
    """KOŞAN eğitimin çıkış kodu tanığını bağla (bu sürümden önce başlatılan koşular için).

    Yalnız bekler; eğitimi durdurmaz/başlatmaz. Süreç bitmişse çıkış kodu artık öğrenilemez."""
    from app.config import get_settings
    from app.training import exit_witness
    from app.training.detached_launch import read_detached_training_status

    st = read_detached_training_status(get_settings().root)
    if st.get("adapter") != adapter or not isinstance(st.get("pid"), int):
        console.print(f"[red]Durum dosyası {adapter} için koşan süreç göstermiyor.[/red]")
        raise typer.Exit(2)
    if st.get("finished_at") or st.get("failed_at"):
        console.print("[red]Koşu zaten sonuçlanmış — çıkış kodu artık gözlenemez.[/red]")
        raise typer.Exit(2)
    ok = exit_witness.spawn(adapter, st["pid"], st.get("pid_create_time"), get_settings().root)
    console.print("tanık başlatıldı" if ok else "[red]tanık başlatılamadı[/red]")
