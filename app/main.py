"""Hektor Trader AI — command line interface.

Run:  uv run hektor --help

Pipeline order (per spec):
  ingest -> ask -> card -> extract-formulas -> research -> dataset -> train -> eval -> backtest
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import configure_logging, get_settings

app = typer.Typer(
    help="Hektor Trader AI — local-first trading research system.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging("DEBUG" if verbose else None)


# --------------------------------------------------------------------------
# init / status
# --------------------------------------------------------------------------
@app.command()
def init() -> None:
    """Tüm dizinleri ve SQLite şemasını oluştur."""
    from app.memory.sqlite_store import SqliteStore

    settings = get_settings()
    settings.ensure_dirs()
    store = SqliteStore()
    console.print(
        Panel.fit(
            f"[green]Hazır.[/green]\nSQLite: {store.db_path}\nChroma: {settings.chroma_dir}",
            title="hektor init",
        )
    )


@app.command()
def status() -> None:
    """Sistem durumunu göster (model, embedding modu, kayıt sayıları)."""
    from app.brain.local_llm import LocalLLM
    from app.memory.chroma_store import ChromaStore
    from app.memory.embedding_service import EmbeddingService
    from app.memory.sqlite_store import SqliteStore

    settings = get_settings()
    store = SqliteStore()
    papers = store.list_papers()
    llm = LocalLLM()

    t = Table(title="Hektor Trader AI — durum")
    t.add_column("Bileşen")
    t.add_column("Değer")
    t.add_row("LLM modeli", settings.llm_model)
    t.add_row("Ollama erişilebilir", "✅" if llm.available() else "❌ (ollama serve)")
    t.add_row("Embedding modu", EmbeddingService().mode)
    t.add_row("Makale sayısı", str(len(papers)))
    try:
        t.add_row("Chroma chunk sayısı", str(ChromaStore().count()))
    except Exception:
        t.add_row("Chroma chunk sayısı", "0")
    try:
        pending = store.list_pending_cards()
        approved = store.list_approved_cards()
        t.add_row(
            "Bilgi kartları (onay / onaylı)",
            f"[yellow]{len(pending)} bekliyor[/yellow] / [green]{len(approved)} onaylı[/green]",
        )
    except Exception:
        t.add_row("Bilgi kartları", "—")
    console.print(t)


# --------------------------------------------------------------------------
# doctor — kurulum & sürüm sapması teşhisi (salt-okuma, offline)
# --------------------------------------------------------------------------
def _git_ro(args: list[str], cwd: Path) -> tuple[int, str]:
    """Salt-okuma git yardımcısı (offline; ağ yok). (returncode, stdout)."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return proc.returncode, (proc.stdout or "").strip()


def _task_path_matches(task_name: str, repo: Path) -> tuple[bool | None, str | None]:
    """Windows scheduled-task hedef yolu bu repoyu mu işaret ediyor? (salt-okuma).

    Dönüş: ``(None, None)`` görev kayıtlı değil; aksi halde
    ``(eşleşti_mi, gözlenen_yol)``. Hiçbir mutasyon/ağ yok.
    """
    import subprocess

    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        f"$t=Get-ScheduledTask -TaskName '{task_name}';"
        "if(-not $t){exit 3};"
        "$t.Actions|ForEach-Object{Write-Output ($_.Arguments+'|'+$_.WorkingDirectory)}"
    )
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if proc.returncode == 3:
        return None, None
    out = (proc.stdout or "").strip()
    if not out:
        return None, None
    matched = str(repo).lower() in out.lower()
    return matched, out.replace("\n", " ")[:100]


@app.command()
def doctor() -> None:
    """Kurulum/sürüm sapması teşhisi (SALT-OKUMA, offline).

    Hiçbir şey çekmez/birleştirmez/değiştirmez. Raporlar: repo yolu, dal,
    HEAD vs origin/main yakınsaması, ahead/behind, push'lanmamış yerel dallar;
    Windows'ta HektorWeb/HektorUpdate görev yolu bu repoyla eşleşiyor mu.
    Çıkış kodu: 0 sağlıklı · 2 SAPMA · 1 git yok.

    origin/main YEREL ref'ten okunur (ağ yok). 'behind' büyükse önce
    `git fetch origin main`. Iraksamış dalı asla otomatik merge etme.
    """
    import sys

    repo = Path(__file__).resolve().parents[1]
    rc, _ = _git_ro(["rev-parse", "--is-inside-work-tree"], repo)
    if rc != 0:
        console.print("[red]git bulunamadı ya da bu bir git deposu değil.[/red]")
        raise typer.Exit(1)

    _, branch = _git_ro(["rev-parse", "--abbrev-ref", "HEAD"], repo)
    _, head = _git_ro(["rev-parse", "--short", "HEAD"], repo)
    _, head_full = _git_ro(["rev-parse", "HEAD"], repo)
    rc_om, origin_main = _git_ro(["rev-parse", "--short", "origin/main"], repo)
    _, om_full = _git_ro(["rev-parse", "origin/main"], repo)

    drift = False
    t = Table(title="Hektor doctor — kurulum & sürüm sapması (salt-okuma)")
    t.add_column("Kontrol")
    t.add_column("Değer")
    t.add_row("Repo yolu", str(repo))
    t.add_row("Mevcut dal", branch or "?")
    t.add_row("HEAD", head or "?")

    if rc_om != 0 or not origin_main:
        t.add_row("origin/main", "[yellow]yerel ref yok — git fetch origin main[/yellow]")
        drift = True
    else:
        t.add_row("origin/main", origin_main)
        _, ab = _git_ro(["rev-list", "--left-right", "--count", "origin/main...HEAD"], repo)
        parts = ab.split()
        behind, ahead = (parts[0], parts[1]) if len(parts) == 2 else ("?", "?")
        t.add_row("origin/main gerisinde (behind)", behind)
        t.add_row("origin/main önünde (ahead)", ahead)
        converged = bool(head_full) and head_full == om_full
        t.add_row(
            "HEAD == origin/main",
            "[green]EVET (yakınsamış)[/green]" if converged else "[red]HAYIR — SAPMA[/red]",
        )
        if not converged:
            drift = True

    on_main = branch == "main"
    t.add_row(
        "Dal == main",
        "[green]EVET[/green]" if on_main else f"[yellow]HAYIR ({branch})[/yellow]",
    )
    if not on_main:
        drift = True

    _, refs = _git_ro(
        [
            "for-each-ref",
            "--format=%(refname:short)|%(upstream:short)|%(upstream:track)",
            "refs/heads",
        ],
        repo,
    )
    unpushed: list[str] = []
    for line in refs.splitlines():
        cols = line.split("|")
        if len(cols) < 3 or not cols[0]:
            continue
        name, upstream, track = cols[0], cols[1], cols[2]
        if not upstream:
            unpushed.append(f"{name} (upstream yok)")
        elif "ahead" in track:
            unpushed.append(f"{name} {track}")
    t.add_row(
        "Push'lanmamış yerel dal",
        ("[yellow]" + ", ".join(unpushed[:8]) + "[/yellow]") if unpushed else "yok",
    )

    if sys.platform == "win32":
        for task in ("HektorWeb", "HektorUpdate"):
            matched, detail = _task_path_matches(task, repo)
            if detail is None:
                t.add_row(f"Görev {task}", "[yellow]kayıtlı değil[/yellow]")
            elif matched:
                t.add_row(f"Görev {task}", "[green]bu repoyu işaret ediyor[/green]")
            else:
                t.add_row(f"Görev {task}", f"[red]ÖLÜ/yabancı yol: {detail}[/red]")
                drift = True

    console.print(t)
    if drift:
        console.print(
            "[yellow]SAPMA saptandı. Otomatik düzeltme YOK — iraksamış dalı merge "
            "etme; önce `git fetch origin main` çalıştırıp inceleyin.[/yellow]\n"
            "[yellow]Bu makineyi origin/main'e zorla eşitlemek için: "
            "update.ps1 -Force (Windows) · ./update.sh --force (mac/Linux)[/yellow]"
        )
        raise typer.Exit(2)
    console.print("[green]Yakınsamış: bu makine origin/main'de ve kurulum tutarlı.[/green]")


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------
@app.command()
def ingest(
    directory: Path = typer.Option(None, help="PDF klasörü (varsayılan: data/papers/raw_pdf)"),
    force: bool = typer.Option(False, help="Var olan makaleleri yeniden indexle"),
) -> None:
    """PDF'leri oku → chunk → SQLite + ChromaDB."""
    from app.memory.paper_indexer import PaperIndexer

    results = PaperIndexer().ingest_directory(directory, force=force)
    if not results:
        console.print("[yellow]PDF bulunamadı. data/papers/raw_pdf içine PDF koyun.[/yellow]")
        raise typer.Exit()
    t = Table(title="Ingestion sonucu")
    t.add_column("paper_id")
    t.add_column("başlık")
    t.add_column("chunk")
    t.add_column("durum")
    for r in results:
        t.add_row(r.paper_id, (r.title or "?")[:50], str(r.n_chunks), "skip" if r.skipped else "ok")
    console.print(t)


@app.command()
def papers() -> None:
    """Indexlenmiş makaleleri listele."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_papers()
    t = Table(title="Makaleler")
    t.add_column("paper_id")
    t.add_column("yıl")
    t.add_column("başlık")
    for p in rows:
        t.add_row(p.paper_id, p.year or "?", (p.title or "?")[:60])
    console.print(t)


# --------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------
@app.command()
def ask(question: str, top_k: int = typer.Option(None)) -> None:
    """RAG ile kaynaklı cevap üret."""
    from app.brain.rag_answerer import RagAnswerer

    ans = RagAnswerer().answer(question, top_k=top_k)
    console.print(Panel(ans.answer, title=f"Cevap (LLM={'on' if ans.llm_used else 'off'})"))
    if ans.sources:
        t = Table(title="Kaynaklar")
        t.add_column("citation")
        t.add_column("başlık")
        t.add_column("dist")
        for s in ans.sources:
            t.add_row(s.citation, (s.title or "?")[:40], f"{s.distance:.3f}" if s.distance else "?")
        console.print(t)


# --------------------------------------------------------------------------
# knowledge cards / training
# --------------------------------------------------------------------------
@app.command()
def card(paper_id: str) -> None:
    """Bir makaleden knowledge card üret (LLM gerekir)."""
    from app.brain.knowledge_card_builder import KnowledgeCardBuilder

    kc = KnowledgeCardBuilder().build(paper_id)
    console.print_json(json.dumps(kc.model_dump(), ensure_ascii=False))


@app.command("cards")
def cards_cmd(
    action: str = typer.Argument(..., help="pending | approve | reject"),
    card_id: str = typer.Argument("", help="approve/reject için kart ID'si"),
) -> None:
    """Kart onay yönetimi: pending listesi / approve / reject."""
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()

    if action == "pending":
        rows = store.list_pending_cards()
        if not rows:
            console.print("[yellow]Onay bekleyen kart yok.[/yellow]")
            return
        t = Table(title="Onay bekleyen kartlar")
        t.add_column("card_id")
        t.add_column("paper_id")
        t.add_column("difficulty")
        t.add_column("stage")
        t.add_column("created_at")
        for r in rows:
            t.add_row(
                r["card_id"][:20],
                r["paper_id"][:20],
                f"{r['difficulty']:.2f}",
                r["stage"] or "—",
                (r["created_at"] or "")[:10],
            )
        console.print(t)

    elif action == "approve":
        if not card_id:
            console.print("[red]approve için card_id gereklidir.[/red]")
            raise typer.Exit(1)
        if store.approve_card(card_id):
            console.print(f"[green]Onaylandı:[/green] {card_id}")
        elif store.get_card_by_id(card_id) is None:
            console.print(f"[red]Kart bulunamadı:[/red] {card_id}")
            raise typer.Exit(1)
        else:
            console.print(
                f"[red]Kart boş/içeriksiz — onaylanamaz:[/red] {card_id} "
                "(reddet ya da kartı yeniden üret)"
            )
            raise typer.Exit(1)

    elif action == "reject":
        if not card_id:
            console.print("[red]reject için card_id gereklidir.[/red]")
            raise typer.Exit(1)
        ok = store.reject_card(card_id)
        if ok:
            console.print(f"[yellow]Reddedildi:[/yellow] {card_id}")
        else:
            console.print(f"[red]Kart bulunamadı:[/red] {card_id}")
            raise typer.Exit(1)

    else:
        console.print(f"[red]Bilinmeyen eylem: {action!r}. pending | approve | reject[/red]")
        raise typer.Exit(1)


@app.command()
def dataset(
    phase: int = typer.Option(0, help="1-4: sadece o fazın örnekleri. 0=tümü"),
    all_examples: bool = typer.Option(
        False, "--all", help="lora_eligible filtresi olmadan tümünü al"
    ),
) -> None:
    """Saklanan training örneklerinden train/valid JSONL üret."""
    from app.training.dataset_builder import DatasetBuilder

    res = DatasetBuilder().build(
        phase=phase if phase > 0 else None,
        lora_eligible_only=not all_examples,
    )
    console.print(
        Panel.fit(
            f"train: {res.train_path} ({res.n_train})\n"
            f"valid: {res.valid_path} ({res.n_valid})\n"
            f"hash:  {res.content_hash}",
            title="dataset",
        )
    )


@app.command()
def train(
    base_model: str = typer.Option(None),
    adapter_name: str = typer.Option("hektor_lora_v1"),
    iterations: int = typer.Option(300, help="Eğitim iterasyon sayısı"),
    batch_size: int = typer.Option(2, help="Batch büyüklüğü (8GB için 2 önerilir)"),
    num_layers: int = typer.Option(8, help="LoRA adapter katman sayısı (sadece MLX)"),
    run: bool = typer.Option(False, help="Eğitimi gerçekten başlat"),
    backend: str = typer.Option("auto", help="Backend: auto|mlx|peft"),
    profile: str = typer.Option(
        None,
        help="LoRA profili (configs/lora/lora_profiles.yaml; yalnız PEFT): "
        "standard_reasoning|high_capacity_reasoning|discipline_safe|"
        "discipline_safe_local|small_smoke_test",
    ),
    max_examples: int = typer.Option(
        0,
        "--max-examples",
        help="Yalnız N örnekle eğit (0=profil/varsayılan). CPU süresini sınırlar (yalnız PEFT).",
    ),
) -> None:
    """LoRA eğitim komutunu hazırla — platform otomatik tespit edilir."""
    from app.training.backend import detect_lora_backend

    settings = get_settings()

    resolved = detect_lora_backend() if backend == "auto" else backend

    if run:
        # Phase 2: STOP_ALL + TAZE manuel onay kapısı (CLAUDE.md Kural 8).
        import os as _os

        from app.agents.runtime import supervisor
        from app.training.unattended_policy import authorize_training_action

        if supervisor.is_stop_all_active():
            console.print(
                "[bold red]STOP_ALL aktif[/bold red] — gerçek eğitim bloklandı. "
                "Kaldır: [cyan]uv run hektor clear-stop-all[/cyan]"
            )
            raise typer.Exit(2)

        # auto_pipeline/launch zaten kendi onayını aldıysa (supervised) iç kapı atlanır
        # — çift onay olmasın; ama STOP_ALL her zaman geçerli.
        if not _os.environ.get("HEKTOR_TRAIN_SUPERVISED"):
            decision = authorize_training_action(
                "train_run",
                (f"Gerçek LoRA eğitimi: {adapter_name} ({iterations} adım, backend={backend})"),
                agent_id="lora-trainer",
            )
            if not decision.authorized:
                console.print(
                    Panel.fit(
                        "[bold red]Gerçek eğitim TAZE manuel onay gerektirir.[/bold red]\n"
                        f"Onay isteği oluşturuldu: [yellow]{decision.approval_id}[/yellow]\n"
                        f"Onayla: [cyan]uv run hektor approval-approve "
                        f"{decision.approval_id}[/cyan]\n"
                        "Sonra bu komutu TEKRAR çalıştır. "
                        "(Standing yetki yok — her eğitim ayrı onay ister.)",
                        title="⛔ Onay gerekli (Phase 2)",
                    )
                )
                raise typer.Exit(3)
            console.print(
                Panel.fit(
                    "[bold red]Bu GERÇEK LoRA eğitimidir.[/bold red] "
                    f"Taze onay tüketildi ([yellow]{decision.approval_id}[/yellow]).\n"
                    "Otomatik gözetimsiz döngüde çalıştırmayın (CLAUDE.md Kural 8).",
                    title="⚠ Onaylı gerçek eğitim",
                )
            )
        else:
            console.print("[dim]Denetimli (supervised) eğitim — üst katman onayı kullanıldı.[/dim]")

        # Eğitim verisi tazeliği (S1): lora_sft.jsonl → jsonl_dir/{train,valid}.jsonl SENKRONLA.
        # CLI `train` doğrudan train.jsonl okur; lora-cloud-prep/assemble_sft lora_sft.jsonl yazar
        # ama split'i güncellemez → bayat/boş veride saatlerce eğitim riski (CLI↔web drift).
        # ensure_train_split kaynak doluysa yeniden böler (clobber onarımı), boşsa dokunmaz.
        from app.training.detached_launch import ensure_train_split

        n_train, _n_valid = ensure_train_split(settings)
        if n_train <= 0:
            console.print(
                "[red]Eğitim verisi yok (train.jsonl boş).[/red] Önce veri kur: "
                "[cyan]uv run python scripts/assemble_sft.py && uv run hektor lora-split[/cyan] "
                "ya da [cyan]uv run hektor lora-cloud-prep[/cyan]."
            )
            raise typer.Exit(1)
        console.print(f"[dim]Eğitim verisi tazelendi: train={n_train}, valid={_n_valid}.[/dim]")

    if resolved == "mlx":
        from app.training.mlx_lora_train import TrainConfig
        from app.training.mlx_lora_train import main as train_main
        from app.training.mlx_lora_train import run as train_run

        cfg = TrainConfig(
            base_model=base_model or settings.mlx_base_model,
            train_jsonl=settings.jsonl_dir / "train.jsonl",
            valid_jsonl=settings.jsonl_dir / "valid.jsonl",
            adapter_output_path=settings.adapters_dir / adapter_name,
            iterations=iterations,
            batch_size=batch_size,
            num_layers=num_layers,
        )
        if run:
            train_run(cfg)
        else:
            train_main(
                [
                    "--base-model",
                    cfg.base_model,
                    "--train-jsonl",
                    str(cfg.train_jsonl),
                    "--valid-jsonl",
                    str(cfg.valid_jsonl),
                    "--adapter-output-path",
                    str(cfg.adapter_output_path),
                    "--iterations",
                    str(cfg.iterations),
                ]
            )
    else:
        from app.training.peft_lora_train import PeftTrainConfig, dry_run, load_lora_profile
        from app.training.peft_lora_train import train as peft_train

        prof: dict = {}
        if profile:
            try:
                prof = load_lora_profile(profile)
            except (KeyError, FileNotFoundError) as exc:
                console.print(f"[red]Profil hatası: {exc}[/red]")
                raise typer.Exit(1) from exc
            # epochs PeftTrainConfig alanı değil — ayıkla (epoch ≠ iterasyon). max_examples
            # ARTIK geçerli bir alan (yerel CPU alt-kümesi) → **prof ile aktarılır.
            prof.pop("epochs", None)
            console.print(f"[cyan]LoRA profili uygulandı:[/cyan] {profile}")

        # CLI --max-examples profili EZER (>0 ise): hızlı smoke ↔ tam koşu kontrolü.
        if max_examples > 0:
            prof["max_examples"] = max_examples

        cfg = PeftTrainConfig(  # type: ignore[assignment]
            base_model=base_model or settings.peft_base_model,
            train_jsonl=settings.jsonl_dir / "train.jsonl",
            valid_jsonl=settings.jsonl_dir / "valid.jsonl",
            adapter_output_path=settings.adapters_dir / adapter_name,
            iterations=iterations,
            **prof,
        )
        if run:
            result = peft_train(cfg)  # type: ignore[arg-type]
            if not result.get("ok"):
                console.print(f"[red]Hata: {result.get('error')}[/red]")
                if "Eksik paketler" in str(result.get("error", "")):
                    console.print(
                        "[yellow]Kur: uv pip install torch transformers "
                        "peft datasets accelerate[/yellow]"
                    )
        else:
            result = dry_run(cfg)  # type: ignore[arg-type]
            import json as _json

            console.print(_json.dumps(result, ensure_ascii=False, indent=2))
            if result.get("missing_packages"):
                console.print(f"\n[yellow]Eksik paketler: {result['missing_packages']}[/yellow]")
                console.print(
                    "[yellow]Kur: uv pip install torch transformers "
                    "peft datasets accelerate[/yellow]"
                )


@app.command()
def evaluate(eval_set: Path, adapter_version: str = typer.Option(None)) -> None:
    """Bir eval seti çalıştır ve hata modlarını işaretle."""
    from app.training.evaluate_model import ModelEvaluator

    res = ModelEvaluator().run_eval(eval_set, adapter_version=adapter_version)
    console.print(
        Panel.fit(
            f"set: {res['eval_set']}\nmodel: {res['model']}\n"
            f"score: {res['score']}\nflags: {res['total_flags']}",
            title="evaluation",
        )
    )


@app.command("lora-eval")
def lora_eval(
    adapter: str = typer.Argument(
        ..., help="Adapter klasör adı (models/adapters/<ad>) veya tam yol"
    ),
    eval_set: Path = typer.Option(
        Path("evals/discipline_core.jsonl"), "--eval-set", help="Eval seti (.jsonl)"
    ),
    n: int = typer.Option(0, "--n", help="Yalnız ilk N soru (0=hepsi; hızlı doğrulama için 1-2)"),
    base_model: str = typer.Option(
        "", "--base-model", help="Base model override (boş=adapter_config'ten, yoksa settings)."
    ),
) -> None:
    """Eğitilen PEFT adapter'ı GERÇEKTEN değerlendir: base vs adapter (Kural 2 dürüst gate).

    `evaluate`/ModelEvaluator base Ollama'yı ölçer, adapter'ı yüklemez. Bu komut adapter'ı
    transformers/PEFT ile yükleyip base ile kıyaslar. AĞIR (CPU'da 4B inference). Base, boş
    bırakılırsa adapter'ın kendi config'inden alınır (küçük-model adapter'ı için ŞART).
    """
    from app.config import get_settings
    from app.training.adapter_eval import evaluate_adapter

    s = get_settings()
    adir = adapter if Path(adapter).exists() else str(s.adapters_dir / adapter)
    if not Path(adir).exists():
        console.print(f"[red]Adapter bulunamadı:[/red] {adir}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Değerlendiriliyor (ağır, CPU):[/cyan] {adir} ← {eval_set}")
    res = evaluate_adapter(adir, eval_set, n=(n or None), base_model=base_model or None)
    color = {"accept": "green", "reject": "red", "inconclusive": "yellow"}.get(res.verdict, "white")
    console.print(
        Panel.fit(
            f"adapter: {res.adapter}\neval: {res.eval_set} (n={res.n})\n"
            f"base    skor: {res.base_score}  (flags {res.base_flags})\n"
            f"adapter skor: {res.adapter_score}  (flags {res.adapter_flags})\n"
            f"[{color}]VERDICT: {res.verdict.upper()}"
            + ("  — GERİLEME (terfi etme)" if res.regression else "")
            + "[/]",
            title="adapter eval (base vs adapter)",
        )
    )


@app.command("lora-chat")
def lora_chat(
    adapter: str = typer.Argument("", help="Adapter adı/yolu (boş=YALNIZ base model)"),
    q: str = typer.Option("", "--q", help="Tek soru (boşsa interaktif sohbet döngüsü)."),
    base_model: str = typer.Option(
        "", "--base-model", help="Base override (boş=adapter_config'ten, yoksa settings)."
    ),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Üretilecek maksimum token."),
) -> None:
    """Eğitilen LoRA adapter'ı (veya base) ile LOKAL sohbet — PEFT, Ollama gerektirmez.

    Base, adapter'ın kendi config'inden çözülür (küçük-model adapter'ı için ŞART). Üretim
    greedy/deterministtir (Kural 6). AĞIR: CPU'da model yükleme + üretim dakikalar sürer.
    """
    from app.config import get_settings
    from app.training.adapter_eval import _generate, _load_model, _resolve_base_model

    s = get_settings()
    adapter_dir: str | None = None
    if adapter:
        adir = adapter if Path(adapter).exists() else str(s.adapters_dir / adapter)
        if not Path(adir).exists():
            console.print(f"[red]Adapter bulunamadı:[/red] {adir}")
            raise typer.Exit(code=1)
        adapter_dir = adir

    base = (
        base_model
        or (_resolve_base_model(adapter_dir) if adapter_dir else None)
        or s.peft_base_model
    )
    console.print(
        f"[cyan]Yükleniyor (CPU, ağır):[/cyan] base={base} | adapter={adapter_dir or '(yok)'}"
    )
    tok, model = _load_model(base, adapter_dir)

    def _answer(question: str) -> str:
        return _generate(tok, model, question, max_new_tokens=max_tokens)

    if q:
        console.print(Panel(_answer(q), title=f"cevap ({'adapter' if adapter_dir else 'base'})"))
        return

    console.print("[dim]İnteraktif sohbet. Çıkış için boş satır veya Ctrl-C.[/dim]")
    while True:
        try:
            question = console.input("[bold green]Sen> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Sohbet bitti.[/dim]")
            break
        if not question:
            break
        console.print(Panel(_answer(question), title="model"))


# --------------------------------------------------------------------------
# trading / backtest
# --------------------------------------------------------------------------
@app.command()
def gen_data(
    out: Path = typer.Option(None, help="Çıktı CSV (varsayılan: data/market/raw/synthetic.csv)"),
    n: int = typer.Option(2000),
) -> None:
    """Test için sentetik OHLCV CSV üret."""
    from app.trading.market_data_loader import write_synthetic_csv

    settings = get_settings()
    out = out or (settings.market_raw_dir / "synthetic.csv")
    path = write_synthetic_csv(out, n=n)
    console.print(f"[green]Yazıldı:[/green] {path}")


@app.command()
def backtest(
    data: Path = typer.Argument(..., help="OHLCV CSV yolu"),
    strategy_json: Path = typer.Option(None, help="Strategy IR JSON; yoksa örnek IR kullanılır"),
) -> None:
    """Strategy IR'i bir CSV üzerinde backtest et, evaluator ile yargıla, kaydet."""
    from app.trading.backtester import persist_backtest, run_backtest
    from app.trading.evaluator import evaluate as eval_strategy
    from app.trading.market_data_loader import load_ohlcv
    from app.trading.strategy_ir import StrategyIR, example_ir

    df = load_ohlcv(data)
    ir = (
        StrategyIR.model_validate_json(strategy_json.read_text(encoding="utf-8"))
        if strategy_json
        else example_ir()
    )

    result = run_backtest(df, ir)
    verdict = eval_strategy(df, ir)
    bt_id = persist_backtest(
        result, str(data), verdict=verdict.verdict, notes="; ".join(verdict.reasons)
    )

    m = result.metrics
    t = Table(title=f"Backtest — {ir.name}")
    t.add_column("metrik")
    t.add_column("değer")
    for k, v in m.to_dict().items():
        t.add_row(k, str(v))
    console.print(t)

    color = {"pass": "green", "fail": "red", "inconclusive": "yellow"}.get(verdict.verdict, "white")
    console.print(
        Panel(
            "\n".join(f"- {r}" for r in verdict.reasons),
            title=f"[{color}]YARGI: {verdict.verdict.upper()}[/{color}]  (backtest_id={bt_id})",
        )
    )


# --------------------------------------------------------------------------
# research — trader beyin döngüsü
# --------------------------------------------------------------------------
@app.command("extract-formulas")
def extract_formulas(
    paper_id: str = typer.Argument(None, help="Belirli makale; yoksa tümü"),
    force: bool = typer.Option(False, help="Zaten çıkarılmışları üzerine yaz"),
) -> None:
    """PDF'lerden matematiksel formülleri çıkar ve kavram grafiğini oluştur."""
    from app.research.concept_graph import ConceptGraph
    from app.research.formula_extractor import FormulaExtractor

    extractor = FormulaExtractor()
    graph = ConceptGraph()

    if paper_id:
        formulas = extractor.extract_from_paper(paper_id, force=force)
        console.print(f"[green]{len(formulas)} formül çıkarıldı:[/green] {paper_id}")
    else:
        results = extractor.extract_from_all_papers()
        total = sum(len(v) for v in results.values())
        console.print(f"[green]Toplam {total} formül çıkarıldı ({len(results)} makale)[/green]")

    n_links = graph.build_from_papers()
    console.print(f"[green]Kavram grafiği: {n_links} bağlantı oluşturuldu[/green]")


@app.command()
def formulas(paper_id: str = typer.Argument(None)) -> None:
    """Çıkarılmış formülleri listele."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_formulas(paper_id=paper_id)
    if not rows:
        console.print("[yellow]Formül bulunamadı — önce 'extract-formulas' çalıştır[/yellow]")
        return
    t = Table(title="Formüller")
    t.add_column("Ad")
    t.add_column("Kategori")
    t.add_column("Açıklama")
    t.add_column("Makale")
    for f in rows:
        t.add_row(
            f["name"],
            f.get("category") or "—",
            (f.get("description") or "")[:60],
            f["paper_id"][:16],
        )
    console.print(t)


@app.command()
def research(
    question: str = typer.Argument(..., help="Araştırma sorusu"),
    iterations: int = typer.Option(3, help="Maksimum iterasyon"),
    paper_ids: str = typer.Option(None, help="Virgülle ayrılmış paper_id listesi (opsiyonel)"),
) -> None:
    """Agentic araştırma döngüsünü çalıştır: sentezle → backtest → yansıt → iyileştir."""
    from app.research.orchestrator import ResearchOrchestrator

    pid_list = [p.strip() for p in paper_ids.split(",")] if paper_ids else None
    orchestrator = ResearchOrchestrator(max_iterations=iterations)
    result = orchestrator.run(question, paper_ids=pid_list)

    color = {"pass": "green", "fail": "red", "inconclusive": "yellow"}.get(
        result.final_verdict, "white"
    )
    console.print(Panel(result.summary(), title=f"[{color}]Araştırma Tamamlandı[/{color}]"))


@app.command("research-sessions")
def research_sessions(limit: int = typer.Option(20)) -> None:
    """Kayıtlı araştırma oturumlarını listele."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_research_sessions(limit=limit)
    if not rows:
        console.print("[yellow]Oturum bulunamadı — önce 'research' komutunu çalıştır[/yellow]")
        return
    t = Table(title="Araştırma Oturumları")
    t.add_column("ID")
    t.add_column("Soru")
    t.add_column("İter")
    t.add_column("Yargı")
    t.add_column("Tarih")
    for r in rows:
        color = {"pass": "green", "fail": "red"}.get(r.get("verdict", ""), "yellow")
        t.add_row(
            r["session_id"][:16],
            (r["question"] or "")[:50],
            str(r["iteration"]),
            f"[{color}]{r.get('verdict', '—')}[/{color}]",
            (r["created_at"] or "")[:10],
        )
    console.print(t)


@app.command("chain-dataset")
def chain_dataset(only_successful: bool = typer.Option(False)) -> None:
    """Araştırma zincirlerinden LoRA eğitim verisi üret."""
    from app.research.chain_data_builder import ChainDataBuilder

    result = ChainDataBuilder().build(only_successful=only_successful)
    console.print(
        Panel(
            f"kayıt: {result['n_records']}\n"
            f"dosya: {result['output_path']}\n"
            f"hash: {result['content_hash']}",
            title="[green]Zincir veri seti[/green]",
        )
    )


# --------------------------------------------------------------------------
# RLM Controller (Recursive/Reasoning LM — çok-adımlı kaynaklı cevap)
# --------------------------------------------------------------------------
@app.command("rlm-answer")
def rlm_answer(
    query: str = typer.Argument(..., help="Soru"),
    paper_ids: str = typer.Option(None, help="Virgülle ayrılmış paper_id listesi (opsiyonel)"),
    top_k: int = typer.Option(None, help="Tur başına getirilecek chunk sayısı"),
    rounds: int = typer.Option(None, help="Maksimum retrieval turu"),
) -> None:
    """RLM Controller: çok-adımlı retrieval + iddia doğrulama + kaynaklı nihai cevap.

    Motor TEK ve yereldir (RlmController + Ollama); dış "recursive reasoning" motoru yoktur.
    """
    from rich.markup import escape

    pid_list = [p.strip() for p in paper_ids.split(",")] if paper_ids else None

    from app.rlm.rlm_controller import RlmController

    result = RlmController().answer(query, paper_ids=pid_list, top_k=top_k, max_rounds=rounds)

    color = {
        "answered": "green",
        "answered_with_limitation": "yellow",
        "abstained": "red",
        "no_llm": "yellow",
    }.get(result.status, "white")
    # final_answer LLM/kaynak metni ('[paper:chunk]' atıfları, ham cümleler) içerir →
    # markup olarak yorumlanırsa render çöker (MarkupError) veya atıf sessizce yutulur.
    # İçeriği escape et; renk yalnız kontrollü başlıkta (status/color sabit kümeden).
    meta = (
        f"run_id={result.run_id} · görev={result.task_type} · "
        f"kanıt={result.evidence_score} · tur={result.retrieval_rounds} · "
        f"güven={result.confidence_level} ({result.final_confidence})"
    )
    console.print(
        Panel(
            f"{escape(result.final_answer)}\n\n{escape(meta)}",
            title=f"[{color}]RLM: {escape(result.status)}[/{color}]",
        )
    )


@app.command("rlm-runs")
def rlm_runs(limit: int = typer.Option(20)) -> None:
    """Kayıtlı RLM koşularını listele."""
    from rich.markup import escape

    from app.rlm.rlm_store import RlmStore

    rows = RlmStore().list_runs(limit=limit)
    if not rows:
        console.print("[yellow]Koşu bulunamadı — önce 'rlm-answer' çalıştır[/yellow]")
        return
    t = Table(title="RLM Koşuları")
    t.add_column("run_id")
    t.add_column("Soru")
    t.add_column("Görev")
    t.add_column("Durum")
    t.add_column("Kanıt", justify="right")
    t.add_column("Güven", justify="right")
    for r in rows:
        color = {
            "answered": "green",
            "answered_with_limitation": "yellow",
            "abstained": "red",
            "no_llm": "yellow",
        }.get(r["status"], "white")
        t.add_row(
            r["run_id"][:14],
            escape((r["user_query"] or "")[:40]),  # kullanıcı sorgusu → markup yorumlanmasın
            r["task_type"],
            f"[{color}]{r['status']}[/{color}]",
            str(r["evidence_score"]),
            str(r["final_confidence"]),
        )
    console.print(t)


@app.command("rlm-trajectory")
def rlm_trajectory(run_id: str = typer.Argument(..., help="RLM koşu run_id")) -> None:
    """Bir RLM koşusunun trajektorisini (adım izi) göster."""
    from rich.markup import escape

    from app.rlm.rlm_store import RlmStore

    store = RlmStore()
    if store.get_run(run_id) is None:
        console.print(f"[red]Koşu bulunamadı: {escape(run_id)}[/red]")
        raise typer.Exit(1)
    steps = store.get_steps(run_id)
    if not steps:
        console.print("[yellow]Bu koşuda adım kaydı yok (eski koşu olabilir).[/yellow]")
    else:
        t = Table(title=f"Trajektori — {run_id[:14]}")
        t.add_column("#", justify="right")
        t.add_column("Aşama")
        t.add_column("Çıktı")
        for st in steps:
            t.add_row(
                str(st.get("step_order", "")),
                escape(str(st.get("step_type", ""))),
                escape((str(st.get("output_text", "")) or "")[:80]),
            )
        console.print(t)


@app.command("rlm-lora-candidates")
def rlm_lora_candidates(
    export: str = typer.Option("", help="Aday JSONL çıktı yolu (boşsa yalnız listele)"),
    min_confidence: float = typer.Option(0.85, help="§16 final_confidence eşiği"),
    limit: int = typer.Option(
        1000, min=1, help="Taranacak en yeni koşu sayısı (sessiz-kesme sınırı; >0)"
    ),
) -> None:
    """RLM koşularından LoRA dataset ADAYLARINI seç (salt-okuma; talimat §16).

    ADAY ≠ eğitim verisi: insan onayı olmadan EĞİTİM YOK (kural 8). Yalnız çok yüksek
    güvenli, tam-doğrulanmış (citation≥0.90, grounding≥0.90, desteklenmeyen iddia yok)
    koşular aday olur.
    """
    from rich.markup import escape

    from app.rlm.lora_candidate import export_candidates_jsonl, select_lora_candidates
    from app.rlm.rlm_store import RlmStore

    # Sessiz-kesme uyarısı: limitten fazla koşu varsa eski adaylar atlanır (--limit artır).
    total = RlmStore().count_runs()
    if 0 < limit < total:
        console.print(
            f"[yellow]Uyarı: {total} koşudan yalnız en yeni {limit} tarandı; "
            f"{total - limit} eski koşu atlandı. --limit artırın.[/yellow]"
        )
    cands = select_lora_candidates(min_confidence=min_confidence, limit=limit)
    if not cands:
        console.print(
            "[yellow]Aday bulunamadı — §16 eşiklerini geçen yüksek-güvenli RLM koşusu yok.[/yellow]"
        )
        return
    t = Table(title=f"LoRA Adayları ({len(cands)}) — insan onayı ŞART (eğitim verisi değil)")
    t.add_column("run_id")
    t.add_column("Soru")
    t.add_column("Güven", justify="right")
    t.add_column("Citation", justify="right")
    t.add_column("Grounding", justify="right")
    for c in cands:
        t.add_row(
            c.run_id[:14],
            escape(c.query[:40]),
            f"{c.final_confidence:.2f}",
            f"{c.citation_score:.2f}",
            f"{c.grounding_score:.2f}",
        )
    console.print(t)
    if export:
        n = export_candidates_jsonl(cands, export)
        console.print(
            Panel(
                f"{n} aday yazıldı: {escape(export)}\n"
                "[yellow]UYARI: Bunlar ADAY'dır; eğitimden önce İNSAN ONAYI şart "
                "(kural 8).[/yellow]",
                title="[green]Export[/green]",
            )
        )


@app.command("rlm-engine")
def rlm_engine() -> None:
    """RLM çalışma-zamanı özetini göster (salt-okuma; sır YOK).

    Motor tektir ve yereldir (RlmController + Ollama); sağlayıcı seçimi yoktur.
    """
    from app.rlm.tool_registry import runtime_info

    cfg = runtime_info()
    t = Table(title="RLM çalışma-zamanı")
    t.add_column("Ayar")
    t.add_column("Değer")
    for k in ("engine", "llm_backend", "llm_model", "production_mode", "allow_local_exec", "seed"):
        t.add_row(k, str(cfg[k]))
    t.add_row("allowed_tools", ", ".join(cfg["allowed_tools"]))
    console.print(t)


@app.command("rlm-tools")
def rlm_tools(
    call: str = typer.Option(
        None, "--call", help="İzinli bir tool'u çağır (örn: calculator, formula_check)"
    ),
    expr: str = typer.Option(None, "--expr", help="calculator için ifade (örn: '2*(3+4)')"),
    text: str = typer.Option(None, "--text", help="formula_check için metin"),
) -> None:
    """RLM güvenli tool allowlist'ini göster veya izinli bir tool'u çağır.

    Tool'lar deny-by-default allowlist'tedir; shell/network/secret/fs-write YAPMAZ.
    Argümansız: izinli tool'ları listeler. --call ile saf bir tool'u (örn calculator)
    çağırıp structured sonucu gösterir; izinsiz ad ToolNotAllowed ile reddedilir.
    """
    from rich.markup import escape

    from app.rlm.safe_tools import build_default_registry
    from app.rlm.tool_registry import ToolNotAllowed

    reg = build_default_registry()
    if not call:
        t = Table(title="RLM Güvenli Tool Allowlist")
        t.add_column("Tool")
        t.add_column("İzinli", justify="center")
        for name in reg.available():
            t.add_row(name, "[green]✓[/green]")
        console.print(t)
        console.print(
            "[dim]Çağrı örneği: hektor rlm-tools --call calculator --expr '2*(3+4)'[/dim]"
        )
        return

    kwargs: dict[str, object] = {}
    if expr is not None:
        kwargs["expression"] = expr
    if text is not None:
        kwargs["text"] = text
    try:
        res = reg.call(call, **kwargs)
    except ToolNotAllowed as exc:
        console.print(f"[red]Reddedildi: {escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    ok = bool(res.get("ok"))
    color = "green" if ok else "yellow"
    console.print(Panel(escape(str(res)), title=f"[{color}]rlm-tools: {escape(call)}[/{color}]"))


@app.command("pine")
def pine_export(
    strategy_name: str = typer.Argument(default="", help="Strateji adı (boşsa varsayılan örnek)"),
    output: str = typer.Option("", help="Çıktı dosyası (.pine). Boşsa stdout."),
) -> None:
    """StrategyIR → TradingView Pine Script v5 olarak dışa aktar."""
    from app.memory.sqlite_store import SqliteStore
    from app.trading.strategy_ir import StrategyIR, example_ir

    if strategy_name:
        from sqlalchemy import select

        from app.memory.sqlite_store import SqliteStore, Strategy

        store = SqliteStore()
        with store.session() as s:
            row = s.scalars(select(Strategy).where(Strategy.name == strategy_name)).first()
        if not row:
            console.print(f"[red]Strateji bulunamadı: {strategy_name}[/red]")
            raise typer.Exit(1)
        ir = StrategyIR.model_validate_json(row.ir_json)
    else:
        ir = example_ir()

    pine_code = ir.to_pine()

    if output:
        Path(output).write_text(pine_code, encoding="utf-8")
        console.print(f"[green]Kaydedildi:[/green] {output}")
    else:
        console.print(pine_code)


@app.command("export-package")
def export_package(
    strategy_name: str = typer.Argument(
        default="", help="Strateji adı (boşsa varsayılan örnek kullanılır)"
    ),
    output: str = typer.Option("", help="Çıktı dosyası (.achpkg). Boşsa stdout."),
) -> None:
    """StrategyIR → Entropia-uyumlu .achpkg paketi olarak dışa aktar (Pine + Python kodu içerir).

    En son backtest verdict ve metrikleri otomatik olarak pakete eklenir.
    """
    from sqlalchemy import desc, select

    from app.memory.sqlite_store import Backtest, SqliteStore, Strategy
    from app.trading.package_exporter import export_strategy
    from app.trading.strategy_ir import StrategyIR, example_ir

    store = SqliteStore()
    ir: StrategyIR
    strategy_id: str | None = None

    if strategy_name:
        with store.session() as s:
            row = s.scalars(select(Strategy).where(Strategy.name == strategy_name)).first()
        if not row:
            console.print(f"[red]Strateji bulunamadı: {strategy_name}[/red]")
            raise typer.Exit(1)
        ir = StrategyIR.model_validate_json(row.ir_json)
        strategy_id = row.strategy_id
    else:
        ir = example_ir()

    # En son backtest'i otomatik çek
    verdict: str | None = None
    metrics: dict = {}
    if strategy_id:
        with store.session() as s:
            bt = s.scalars(
                select(Backtest)
                .where(Backtest.strategy_id == strategy_id)
                .order_by(desc(Backtest.created_at))
                .limit(1)
            ).first()
        if bt:
            verdict = bt.verdict
            try:
                import json as _json

                metrics = _json.loads(bt.metrics_json or "{}")
            except Exception:
                metrics = {}
            if bt.total_return_pct:
                metrics.setdefault("total_return_pct", bt.total_return_pct)
            if bt.sharpe is not None:
                metrics.setdefault("sharpe", bt.sharpe)
            if bt.max_drawdown_pct is not None:
                metrics.setdefault("max_drawdown_pct", bt.max_drawdown_pct)
            if bt.n_trades:
                metrics.setdefault("n_trades", bt.n_trades)

    pkg = export_strategy(ir, backtest_verdict=verdict, backtest_metrics=metrics)

    if output:
        out_path = Path(output)
        pkg.save(out_path)
        console.print(f"[green]Paket kaydedildi:[/green] {out_path}")
        console.print(f"  İsim    : {pkg.name}")
        console.print(f"  Tip     : {pkg.package_type}")
        color = "green" if verdict == "pass" else "red" if verdict == "fail" else "yellow"
        verdict_str = f"[{color}]{verdict or '—'}[/]"
        console.print(f"  Backtest: {verdict_str}")
        console.print(f"  Boyut   : {len(pkg.to_json())} karakter")
    else:
        console.print(pkg.to_json())


@app.command("risk")
def risk_analyze(
    backtest_id: str = typer.Argument(..., help="Backtest ID (örn. bt_cd00c55600)"),
    equity: float = typer.Option(10_000.0, help="Başlangıç sermayesi ($)"),
    max_dd: float = typer.Option(-20.0, help="Max drawdown eşiği % (örn. -20)"),
    risk_pct: float = typer.Option(1.0, help="İşlem başına risk %"),
    stop_pct: float = typer.Option(2.0, help="Stop mesafesi %"),
) -> None:
    """Bir backtest için Kelly + drawdown ölçekleme + sabit risk raporu üret."""
    from sqlalchemy import select

    from app.memory.sqlite_store import Backtest, SqliteStore, Strategy
    from app.trading.backtester import _compute_columns, _net_returns, _position_series
    from app.trading.market_data_loader import generate_synthetic_ohlcv
    from app.trading.risk_manager import analyze_risk
    from app.trading.strategy_ir import StrategyIR

    store = SqliteStore()
    with store.session() as s:
        bt = s.scalars(select(Backtest).where(Backtest.backtest_id == backtest_id)).first()
        if bt is None:
            console.print(f"[red]Backtest bulunamadı: {backtest_id}[/red]")
            raise typer.Exit(1)
        strat = s.get(Strategy, bt.strategy_id)
        if strat is None:
            console.print("[red]Strateji bulunamadı.[/red]")
            raise typer.Exit(1)
        ir = StrategyIR.model_validate_json(strat.ir_json)

    df = generate_synthetic_ohlcv(n=2000, seed=42)
    enriched = _compute_columns(df, ir)
    position = _position_series(enriched, ir)
    bar_ret = enriched["close"].pct_change().fillna(0.0)
    # Kanonik maliyet-dahil seri (komisyon+slippage, cost-timing hizalı). Eski hâl
    # maliyetsiz getiriden Kelly/drawdown üretip pozisyonu fazla-iyimser ölçüyordu (Kural 3).
    net_ret = _net_returns(position, bar_ret, ir.costs.commission + ir.costs.slippage)
    equity_curve = (1 + net_ret).cumprod()

    report = analyze_risk(
        strategy_name=ir.name,
        equity_curve=equity_curve,
        position=position,
        returns=net_ret,
        equity_usd=equity,
        max_dd_threshold_pct=max_dd,
        risk_per_trade_pct=risk_pct,
        atr_stop_pct=stop_pct,
    )

    k = report.kelly
    dd = report.drawdown_scale
    fr = report.fixed_risk

    t = Table(title=f"Risk Analizi — {ir.name}")
    t.add_column("Metrik")
    t.add_column("Değer")
    t.add_row("İşlem sayısı", str(report.n_trades))
    t.add_row("Kazanma oranı", f"{k.win_rate:.1%}")
    t.add_row("Ort. kazanç / kayıp", f"+{k.avg_win:.2%} / -{k.avg_loss:.2%}")
    t.add_row("Odds (b)", f"{k.odds:.2f}")
    t.add_row("Tam Kelly", f"{k.full_kelly:.1%}")
    t.add_row("Yarı Kelly (önerilen)", f"[green]{k.half_kelly:.1%}[/green]")
    t.add_row("Sınırlı Kelly (max %25)", f"[cyan]{k.capped_kelly:.1%}[/cyan]")
    t.add_row("Anlık drawdown", f"{dd.current_drawdown_pct:.1f}%")
    t.add_row("Ölçek faktörü", f"{dd.scale_factor:.2f}")
    t.add_row("Sabit risk pozisyonu %", f"{fr.position_size_pct:.1f}%")
    t.add_row("Sabit risk pozisyonu $", f"${fr.position_size_usd:,.0f}")
    console.print(t)

    if report.warnings:
        for w in report.warnings:
            console.print(f"  [yellow]⚠[/yellow] {w}")

    console.print(Panel(report.recommendation, title="Öneri", border_style="cyan"))


@app.command("arxiv")
def arxiv_fetch(
    query: str = typer.Argument(..., help="Arama sorgusu, örn: 'momentum trading volatility'"),
    max_results: int = typer.Option(5, help="İndirilecek maksimum makale sayısı (1-20)"),
    search_only: bool = typer.Option(False, help="Yalnız ara, indirme yapma"),
    auto_ingest: bool = typer.Option(True, help="İndirdikten sonra otomatik indeksle"),
) -> None:
    """arXiv'den trading araştırması ara → indir → otomatik indeksle."""
    from app.ingestion.arxiv_fetcher import fetch_arxiv_papers, search_arxiv

    max_results = max(1, min(max_results, 20))

    if search_only:
        console.print(f"[cyan]arXiv aranıyor:[/cyan] {query!r} (max {max_results})")
        entries = search_arxiv(query, max_results=max_results)
        if not entries:
            console.print("[yellow]Sonuç bulunamadı.[/yellow]")
            raise typer.Exit()
        t = Table(title=f"arXiv sonuçları — '{query}'")
        t.add_column("ID")
        t.add_column("Başlık")
        t.add_column("Tarih")
        t.add_column("PDF")
        for e in entries:
            t.add_row(e.arxiv_id, e.title[:60], e.published, e.pdf_url)
        console.print(t)
        return

    console.print(f"[cyan]arXiv indiriliyor:[/cyan] {query!r} (max {max_results})")
    with console.status("PDF'ler indiriliyor…"):
        results = fetch_arxiv_papers(query, max_results=max_results)

    if not results:
        console.print("[yellow]İndirilecek makale bulunamadı.[/yellow]")
        raise typer.Exit()

    downloaded = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]
    n_dl, n_sk = len(downloaded), len(skipped)
    console.print(f"[green]İndirilen:[/green] {n_dl}  [yellow]Atlanan:[/yellow] {n_sk}")

    if auto_ingest and downloaded:
        from app.memory.paper_indexer import PaperIndexer

        console.print("[cyan]İndeksleniysor…[/cyan]")
        ingest_results = PaperIndexer().ingest_directory()
        t = Table(title="İndirme + İndeksleme sonucu")
        t.add_column("arxiv_id")
        t.add_column("Başlık")
        t.add_column("Durum")
        for r in results:
            status = "skip (zaten vardı)" if r.skipped else "indirildi"
            t.add_row(r.arxiv_id, r.title[:55], status)
        for ir in ingest_results:
            if not ir.skipped:
                console.print(f"  ✓ {ir.paper_id} — {ir.n_chunks} chunk")
        console.print(t)
    else:
        for r in results:
            status = "[yellow]skip[/yellow]" if r.skipped else "[green]ok[/green]"
            console.print(f"  {status}  {r.arxiv_id}  {r.title[:55]}")


@app.command("rag-scan")
def rag_scan(
    max_per_query: int = typer.Option(8, help="Sorgu başına maksimum arXiv sonucu (1-20)"),
    min_score: int = typer.Option(2, help="Minimum heuristik alaka skoru (eşik)"),
    dry_run: bool = typer.Option(False, help="Watchlist'e yazma; yalnız adayları listele"),
) -> None:
    """RAG yöntem trendlerini arXiv'de tara → izleme listesine aday ekle.

    Güncel-RAG döngüsünün **ucuz tarama** katmanı (projeye yerleşik ajan): Claude/kota
    gerektirmez. Pahalı entegrasyon ayrıdır (haftalık kodlama ajanı turu).
    """
    from app.research.rag_trend_scanner import (
        append_candidates,
        scan_rag_trends,
        watchlist_path,
    )

    max_per_query = max(1, min(max_per_query, 20))
    console.print("[cyan]RAG trend taraması (arXiv)…[/cyan]")
    candidates = scan_rag_trends(max_per_query=max_per_query, min_score=min_score)
    if not candidates:
        console.print("[yellow]Aday bulunamadı (ağ erişimi yok veya hepsi eşik altı).[/yellow]")
        raise typer.Exit()

    t = Table(title="RAG trend adayları (arXiv)")
    t.add_column("Skor")
    t.add_column("arXiv")
    t.add_column("Başlık")
    t.add_column("Tarih")
    for c in candidates[:20]:
        t.add_row(str(c.score), c.arxiv_id, c.title[:60], c.published)
    console.print(t)

    if dry_run:
        console.print("[yellow]dry-run: izleme listesine yazılmadı.[/yellow]")
        return

    added = append_candidates(watchlist_path(), candidates)
    if added:
        console.print(f"[green]İzleme listesine eklenen yeni aday:[/green] {added}")
    else:
        console.print("[yellow]Yeni aday yok (hepsi zaten izleme listesinde).[/yellow]")


@app.command("lit-scan")
def lit_scan(
    topic: list[str] = typer.Option(
        None,
        "--topic",
        "-t",
        help="Konu paketi (rag|lora|rlm|math-physics). Tekrarlanabilir; boş → hepsi.",
    ),
    max_per_query: int = typer.Option(8, help="Sorgu başına maksimum arXiv sonucu (1-20)"),
    min_score: int = typer.Option(2, help="Minimum heuristik alaka skoru (eşik)"),
    top_n: int = typer.Option(3, help="Konu başına indirilecek en alakalı makale sayısı"),
    download: bool = typer.Option(True, "--download/--no-download", help="PDF indirilsin mi"),
) -> None:
    """Literatür KEŞİF turu: LoRA/RAG/RLM/matematik-fizik adaylarını tara, indir, listele.

    İndirilenler "gelen kutusu"na düşer (HEKTOR_SCOUT_INBOX_DIR; yoksa
    data/literature_inbox/). **RAG'a ingest ETMEZ, eğitim BAŞLATMAZ** — bunlar SENDE
    kalır (Kural 8). Claude/API kotası kullanmaz: yalnız arXiv + offline skor.
    """
    from app.research.literature_scout import TOPIC_PACKS, run_scout

    try:
        report = run_scout(
            topics=topic or None,
            max_per_query=max(1, min(max_per_query, 20)),
            min_score=min_score,
            top_n_download=top_n,
            download=download,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    if not report.found:
        console.print("[yellow]Eşik üstü yeni aday bulunamadı (veya ağ erişimi yok).[/yellow]")
        raise typer.Exit()

    t = Table(title=f"Bulunan makaleler — {report.ran_at}")
    t.add_column("Konu")
    t.add_column("Skor")
    t.add_column("arXiv")
    t.add_column("Başlık")
    t.add_column("Dosya")
    for f in report.found:
        t.add_row(
            TOPIC_PACKS[f.topic].label.split(" /")[0],
            str(f.candidate.score),
            f.candidate.arxiv_id,
            f.candidate.title[:52],
            "✓ yeni" if f.downloaded else ("· var" if f.pdf_path else "—"),
        )
    console.print(t)
    console.print(f"[green]Gelen kutusu:[/green] {report.inbox}")
    console.print(f"[green]Bu turda indirilen:[/green] {report.downloaded_count}")
    for k, v in sorted(report.watchlist_added.items()):
        if v:
            console.print(f"[cyan]{k}[/cyan] izleme defterine +{v} aday")
    console.print(
        "[yellow]Not:[/yellow] indirilenler yalnız ADAY. RAG'a alma ve eğitim sende "
        "(Kural 8) — istediğini seç, kalanını sil."
    )


@app.command("profile")
def oss_profile(
    as_json: bool = typer.Option(False, "--json", help="JSON formatında çıktı ver"),
) -> None:
    """Sistemin hardware/software profilini tara ve göster."""
    from app.agents.system_profiler.profiler import collect

    profile = collect()
    if as_json:
        console.print_json(profile.model_dump_json(indent=2))
        return

    console.print(Panel("[bold cyan]Sistem Profili[/bold cyan]", border_style="cyan"))
    console.print(f"  OS      : {profile.os} {profile.os_version[:30]}")
    console.print(f"  Arch    : {profile.arch}")
    console.print(f"  CPU     : {profile.cpu.name} ({profile.cpu.cores}c/{profile.cpu.threads}t)")
    console.print(
        f"  RAM     : {profile.memory.ram_total_gb:.1f} GB total  |  "
        f"{profile.memory.ram_available_gb:.1f} GB free"
    )
    console.print(f"  GPU     : {profile.gpu.name}")
    console.print(f"  Vendor  : {profile.gpu.vendor}")
    console.print(f"  VRAM    : {profile.gpu.vram_gb:.1f} GB")
    flags = []
    if profile.gpu.cuda:
        flags.append("CUDA")
    if profile.gpu.metal:
        flags.append("Metal")
    if profile.gpu.rocm:
        flags.append("ROCm")
    console.print(f"  Accel   : {', '.join(flags) if flags else 'CPU only'}")
    console.print(f"  Disk    : {profile.disk.free_gb:.1f} GB free")
    console.print(f"  Python  : {profile.python_version}")
    tools = []
    if profile.tools.ollama:
        tools.append("ollama")
    if profile.tools.git:
        tools.append("git")
    if profile.tools.cmake:
        tools.append("cmake")
    console.print(f"  Tools   : {', '.join(tools) if tools else '—'}")


@app.command("recommend")
def oss_recommend(
    task: str = typer.Option("general", help="Görev türü: coding, general, reasoning, trading"),
    top_k: int = typer.Option(3, help="Kaç öneri gösterilsin"),
) -> None:
    """RAM/VRAM'a göre en uygun OSS modelini öner."""
    from app.agents.model_advisor.advisor import recommend
    from app.agents.system_profiler.profiler import collect

    console.print("[cyan]Sistem taranıyor...[/cyan]")
    profile = collect()
    result = recommend(profile, task=task, top_k=top_k)

    console.print(f"\n[bold]Sistem:[/bold] {result.system_summary}")
    console.print(f"[bold]Görev :[/bold] {task}\n")

    if result.recommended:
        t = Table(title="Önerilen Modeller")
        t.add_column("Sıra")
        t.add_column("Model")
        t.add_column("Ollama")
        t.add_column("Güven")
        t.add_column("Neden")
        for rec in result.recommended:
            t.add_row(
                str(rec.rank),
                rec.display_name,
                rec.ollama_name,
                f"{rec.confidence:.0%}",
                rec.reasons[0] if rec.reasons else "",
            )
        console.print(t)
    else:
        console.print("[yellow]Bu sistem için uygun model bulunamadı.[/yellow]")

    if result.rejected:
        console.print("\n[dim]Reddedilenler:[/dim]")
        for r in result.rejected:
            console.print(f"  [red]✗[/red] {r.display_name} — {r.reason}")


@app.command("install")
def oss_install(
    model: str = typer.Option("", "--model", help="Ollama model adı (örn. qwen2.5-coder:7b)"),
    auto_safe: bool = typer.Option(
        False, "--auto-safe", help="En iyi uygun modeli otomatik seç ve indir"
    ),
    diagnose: bool = typer.Option(False, "--diagnose", help="Sadece tanı yap, kurma"),
) -> None:
    """OSS modeli Ollama ile kur (sadece güvenli whitelist komutlar)."""
    from app.agents.installer import ollama_installer as oi
    from app.agents.model_advisor.advisor import recommend
    from app.agents.system_profiler.profiler import collect

    profile = collect()

    if not oi.is_ollama_installed():
        console.print(f"[red]{oi.install_guide_text()}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Ollama:[/green] {oi.get_ollama_version()}")

    if diagnose:
        result = recommend(profile, task="general")
        console.print(f"\n[bold]Sistem:[/bold] {result.system_summary}")
        if result.recommended:
            top = result.recommended[0]
            console.print(f"[bold]Öneri :[/bold] {top.display_name} ({top.ollama_name})")
            for reason in top.reasons:
                console.print(f"  • {reason}")
        return

    ollama_name = model
    if not ollama_name and auto_safe:
        result = recommend(profile, task="general")
        if not result.recommended:
            console.print("[red]Uygun model bulunamadı.[/red]")
            raise typer.Exit(1)
        ollama_name = result.recommended[0].ollama_name
        console.print(f"[cyan]Otomatik seçildi:[/cyan] {ollama_name}")

    if not ollama_name:
        console.print("[red]--model veya --auto-safe kullan.[/red]")
        raise typer.Exit(1)

    # Manuel seçimde uyarı kontrolü
    if model and not auto_safe:
        result = recommend(profile, task="general")
        if result.recommended and result.recommended[0].ollama_name != model:
            console.print(
                f"[yellow]⚠ Uyarı: Sisteminiz için önerilen model "
                f"'{result.recommended[0].ollama_name}'. "
                f"'{model}' seçildi — çalışmayabilir.[/yellow]"
            )

    console.print(f"[cyan]İndiriliyor:[/cyan] {ollama_name} ...")
    r = oi.pull_model(ollama_name)
    if r.status == "ok":
        console.print(f"[green]✓ İndirildi:[/green] {ollama_name}")
    else:
        console.print(f"[red]Hata:[/red] {r.error}")
        raise typer.Exit(1)


@app.command("benchmark")
def oss_benchmark(
    model: str = typer.Argument(..., help="Ollama model adı (örn. qwen2.5-coder:7b)"),
    quick: bool = typer.Option(False, "--quick", help="Sadece ilk prompt (hızlı)"),
    save: bool = typer.Option(True, help="Sonucu SQLite'a kaydet"),
) -> None:
    """Modeli benchmark et: tokens/sec, kalite, durum."""
    from app.agents.benchmark.runner import run as bench_run
    from app.agents.learning.memory import save_model_trial, save_system_profile
    from app.agents.system_profiler.profiler import collect

    console.print(f"[cyan]Benchmark başlıyor:[/cyan] {model} {'(hızlı)' if quick else ''}")
    result = bench_run(model, quick=quick)

    color_map = {
        "excellent": "green",
        "usable": "cyan",
        "slow_but_usable": "yellow",
        "unstable": "orange3",
        "failed": "red",
    }
    color = color_map.get(result.status, "white")

    t = Table(title=f"Benchmark — {model}")
    t.add_column("Metrik")
    t.add_column("Değer")
    t.add_row("Durum", f"[{color}]{result.status}[/{color}]")
    t.add_row("tokens/sec", f"{result.tokens_per_second:.1f}")
    t.add_row("İlk token gecikmesi", f"{result.first_token_latency_ms:.0f} ms")
    t.add_row("Peak RAM delta", f"{result.peak_ram_gb:.2f} GB")
    t.add_row("Kalite skoru", f"{result.quality_score:.2f}")
    console.print(t)

    for pr in result.prompt_results:
        icon = "[green]✓[/green]" if pr.status == "ok" else "[red]✗[/red]"
        console.print(
            f"  {icon} {pr.prompt_id}: kalite={pr.quality:.1f}  latency={pr.latency_ms:.0f}ms"
        )

    if save and result.status != "failed":
        profile = collect()
        pid = save_system_profile(profile.model_dump())
        save_model_trial(
            system_profile_id=pid,
            model_id=model,
            backend="ollama",
            status=result.status,
            tokens_per_second=result.tokens_per_second,
            first_token_latency_ms=result.first_token_latency_ms,
            peak_ram_gb=result.peak_ram_gb,
            quality_score=result.quality_score,
        )
        console.print("[dim]✓ Sonuç SQLite'a kaydedildi.[/dim]")


@app.command("tool-use-train")
def tool_use_train(
    questions: list[str] = typer.Argument(None, help="Araştırma soruları (boşsa varsayılan set)."),
    n_loops: int = typer.Option(1, "--n-loops", "-n", help="Her soru için iterasyon sayısı."),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Tool-use eğitim döngüsünü çalıştır ve seansları DB'ye kaydet."""
    from app.training.tool_use_trainer import ToolUseTrainer

    _default_questions = [
        "Yüksek volatilitede RSI momentum stratejileri nasıl filtrelenir?",
        "EMA çaprazlaması ile volume spike kombinasyonu backtest'te ne kadar etkili?",
        "Düşük korelasyonlu indikatörler nasıl birleştirilir?",
    ]
    qs = list(questions) if questions else _default_questions
    trainer = ToolUseTrainer(seed=seed)
    console.print(f"[bold cyan]⚙ Tool-use eğitimi başlıyor — {len(qs)} soru[/bold cyan]")
    sessions = trainer.run_batch(qs, max_iterations=n_loops)
    pass_count = sum(1 for s in sessions if s.final_verdict == "pass")
    console.print(
        f"[green]✓ {len(sessions)} seans tamamlandı "
        f"({pass_count} geçti / {len(sessions) - pass_count} başarısız)[/green]"
    )
    console.print("[dim]Veri seti için: hektor tool-use-dataset[/dim]")


@app.command("tool-use-dataset")
def tool_use_dataset(
    output: str = typer.Option("data/training/tool_use_sft.jsonl", "--output", "-o"),
    only_verdict: str = typer.Option("", "--only-verdict", help="pass | fail | (boş=hepsi)"),
) -> None:
    """tool_use_examples → SFT JSONL veri seti oluştur."""
    from app.training.tool_use_dataset_builder import build_tool_use_dataset, get_tool_use_stats

    stats = get_tool_use_stats()
    console.print(
        f"[bold]Tool-use DB:[/bold] {stats['n_sessions']} seans · "
        f"{stats['n_steps']} adım · {stats['sft_eligible']} SFT uygun"
    )
    if stats["n_sessions"] == 0:
        console.print("[yellow]Önce: hektor tool-use-train[/yellow]")
        raise typer.Exit(1)

    examples = build_tool_use_dataset(
        output_path=output,  # type: ignore[arg-type]
        only_verdict=only_verdict or None,
    )
    console.print(f"[green]✓ {len(examples)} örnek → {output}[/green]")


@app.command("auto-research")
def auto_research(
    max_questions: int = typer.Option(5, "--max", "-n", help="İşlenecek maksimum soru sayısı."),
    iterations: int = typer.Option(1, "--iterations", "-i"),
    seed: int = typer.Option(42, "--seed"),
    all_cards: bool = typer.Option(False, "--all-cards", help="Onaysız kartları da dahil et."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Eğitim çalıştırmadan sorular üret."),
) -> None:
    """Onaylı kartlardan otomatik tool-use eğitimi: kart→soru→backtest→ödül."""
    from app.pipeline.auto_researcher import run_pipeline

    console.print("[bold cyan]⚙ Otomatik araştırma pipeline'ı başlıyor…[/bold cyan]")
    run = run_pipeline(
        max_questions=max_questions,
        max_iterations=iterations,
        seed=seed,
        only_approved=not all_cards,
        dry_run=dry_run,
    )
    console.print(f"\n[bold]Özet:[/bold]\n{run.summary()}")
    if run.errors:
        for e in run.errors[:3]:
            console.print(f"  [red]✗[/red] {e}")
    if dry_run:
        console.print(f"\n[dim]Üretilen sorular ({run.n_questions}):[/dim]")
        from app.memory.sqlite_store import SqliteStore
        from app.pipeline.auto_researcher import _extract_questions

        for q in _extract_questions(SqliteStore(), max_questions, not all_cards):
            console.print(f"  • {q}")


@app.command("reward-analyze")
def reward_analyze(
    session_id: str = typer.Option("", "--session", "-s", help="Tek seans ID'si."),
    build_dpo: bool = typer.Option(False, "--build-dpo", help="DPO veri seti oluştur."),
    output: str = typer.Option("data/training/dpo_pairs.jsonl", "--output", "-o"),
) -> None:
    """Tool-use seanslarını ödül sinyaliyle değerlendir; DPO çiftleri üret."""
    from app.training.dpo_dataset_builder import (
        build_dpo_dataset,
        get_dpo_stats,
        score_and_save_sessions,
    )
    from app.training.reward_signal import compute_reward

    if session_id:
        from app.memory.sqlite_store import SqliteStore

        store = SqliteStore()
        steps = store.list_tool_use_examples(session_id=session_id)
        if not steps:
            console.print(f"[red]Seans bulunamadı: {session_id}[/red]")
            raise typer.Exit(1)
        call_steps = [s for s in steps if s["step_type"] == "call"]
        if not call_steps:
            console.print("[yellow]Bu seansda call adımı yok.[/yellow]")
            raise typer.Exit(1)
        last = call_steps[-1]
        metrics = last.get("tool_output", {}).get("metrics", {})
        rc = compute_reward(metrics, verdict=last.get("verdict", "fail"))
        console.print(f"\n[bold]Seans:[/bold] {session_id}")
        console.print(f"Bileşik skor : [bold]{rc.composite:.3f}[/bold] → [{rc.label}]")
        for k, v in rc.to_dict().items():
            if k not in ("composite", "label", "notes"):
                console.print(f"  {k:<16} {v:.2f}")
        if rc.notes:
            console.print("[yellow]Notlar:[/yellow]", " | ".join(rc.notes))
        return

    new = score_and_save_sessions()
    if new:
        console.print(f"[green]✓ {len(new)} seans skorlandı[/green]")

    stats = get_dpo_stats()
    console.print(
        f"Toplam sinyal: {stats['n_signals']} | "
        f"chosen: {stats['label_distribution'].get('chosen', 0)} | "
        f"rejected: {stats['label_distribution'].get('rejected', 0)} | "
        f"DPO çifti potansiyeli: {stats['dpo_eligible_pairs']}"
    )

    if build_dpo:
        pairs = build_dpo_dataset(output_path=Path(output))
        console.print(f"[green]✓ {len(pairs)} DPO çifti → {output}[/green]")
    elif stats["dpo_eligible_pairs"] > 0:
        console.print("[dim]DPO veri seti için: hektor reward-analyze --build-dpo[/dim]")


@app.command("rules-update")
def oss_rules_update(
    dry_run: bool = typer.Option(False, "--dry-run", help="DB'ye yazmadan önerileri göster."),
    approve: str = typer.Option("", "--approve", help="Onaylanacak öneri ID'si."),
    dismiss: str = typer.Option("", "--dismiss", help="Reddedilecek öneri ID'si."),
) -> None:
    """Başarısız trial'lardan kural önerileri üret; bekleyenleri listele."""
    from app.agents.learning.rules_updater import (
        approve_suggestion,
        dismiss_suggestion,
        generate_rule_suggestions,
        list_pending_suggestions,
    )

    if approve:
        # Phase 2: kural uygulama (sistem-değiştiren) TAZE onay gerektirir.
        from app.agents.runtime import approvals, supervisor

        if supervisor.is_stop_all_active():
            console.print("[red]STOP_ALL aktif — kural uygulama bloklandı.[/red]")
            raise typer.Exit(2)
        decision = approvals.require_fresh_approval(
            "rules-updater", "rules_apply", "medium", f"Kural önerisi uygula: {approve}"
        )
        if not decision.authorized:
            console.print(
                f"[yellow]Kural uygulama TAZE onay gerektirir.[/yellow] "
                f"Onay isteği: [cyan]{decision.approval_id}[/cyan]\n"
                f"Onayla: uv run hektor approval-approve {decision.approval_id}, "
                "sonra tekrar çalıştır."
            )
            raise typer.Exit(3)
        ok = approve_suggestion(approve)
        console.print("[green]✓ Onaylandı[/green]" if ok else "[red]Bulunamadı[/red]")
        return

    if dismiss:
        ok = dismiss_suggestion(dismiss)
        console.print("[yellow]✗ Reddedildi[/yellow]" if ok else "[red]Bulunamadı[/red]")
        return

    import json as _json

    new = generate_rule_suggestions(dry_run=dry_run)
    if new:
        label = "[dim](dry-run)[/dim]" if dry_run else ""
        console.print(f"\n[bold cyan]🔧 {len(new)} yeni öneri üretildi {label}[/bold cyan]")
        for s in new:
            console.print(f"  • [yellow]{s.rule_file}[/yellow] — {s.reason[:80]}…")

    pending = list_pending_suggestions()
    if pending:
        console.print(f"\n[bold]Onay bekleyen {len(pending)} öneri:[/bold]")
        for p in pending:
            patch = _json.loads(p.proposed_patch)
            console.print(
                f"  [dim]{p.suggestion_id[:8]}[/dim]  {patch.get('action', '?')} → {p.rule_file}"
            )
            console.print(f"    {p.reason[:100]}")
        console.print("\n[dim]Onaylamak: hektor rules-update --approve <id>[/dim]")
    elif not new:
        console.print("[green]✓ Yeni öneri yok; sistem sağlıklı.[/green]")


# ---------------------------------------------------------------------------
# mastery commands
# ---------------------------------------------------------------------------


@app.command("mastery-run")
def mastery_run(
    paper_id: str = typer.Argument(..., help="Makale paper_id"),
    questions: int = typer.Option(20, help="Soru sayısı"),
) -> None:
    """Bir makale için Paper Mastery testi çalıştır."""
    from app.learning.paper_mastery_agent import PaperMasteryAgent

    console.print(f"[bold]Mastery testi başlatılıyor:[/bold] {paper_id}")
    agent = PaperMasteryAgent()
    result = agent.run(paper_id, question_count=questions)
    if result.error:
        console.print(f"[red]Hata:[/red] {result.error}")
        raise typer.Exit(1)
    console.print(result.summary())
    console.print(f"[dim]JSON rapor:[/dim] {result.report_json}")
    console.print(f"[dim]MD rapor:[/dim]   {result.report_md}")


@app.command("mastery-queue")
def mastery_queue(
    enqueue_all: bool = typer.Option(False, "--enqueue-all", help="Tüm makaleleri kuyruğa ekle"),
    run_next: bool = typer.Option(False, "--run-next", help="Sıradaki makaleyi test et"),
    run_all: bool = typer.Option(False, "--run-all", help="Tüm kuyruğu işle"),
    limit: int = typer.Option(50, help="--run-all için maks limit"),
) -> None:
    """Mastery öğrenme kuyruğunu göster ve yönet."""
    from app.learning.paper_mastery_agent import LearningQueue

    q = LearningQueue()
    if enqueue_all:
        n = q.enqueue_all_papers()
        console.print(f"[green]{n} makale kuyruğa eklendi.[/green]")
    if run_next:
        result = q.run_next()
        if result is None:
            console.print("[yellow]Kuyrukta bekleyen makale yok.[/yellow]")
        else:
            console.print(result.summary())
    elif run_all:
        results = q.run_all(limit=limit)
        console.print(f"[green]{len(results)} makale işlendi.[/green]")
        for r in results:
            status = "[green]✓[/green]" if not r.error else "[red]✗[/red]"
            total = r.score.total_score
            st = r.score.final_status
            console.print(f"  {status} {r.paper_id} — {total:.1f}/100 {st}")
    else:
        entries = q.list_all()
        if not entries:
            console.print("[dim]Kuyruk boş.[/dim]")
            return
        console.print(f"[bold]Mastery kuyruğu ({len(entries)} makale):[/bold]")
        for e in entries:
            console.print(f"  [{e['status']}] {e['paper_id']}  öncelik={e['priority']}")


@app.command("mastery-score")
def mastery_score(
    paper_id: str = typer.Argument(..., help="Makale paper_id"),
) -> None:
    """Bir makalenin son mastery skorunu göster."""
    from app.memory.mastery_store import MasteryStore

    ms = MasteryStore()
    score = ms.get_latest_score(paper_id)
    if score is None:
        console.print(f"[yellow]'{paper_id}' için henüz mastery skoru yok.[/yellow]")
        raise typer.Exit(1)
    ts = score["total_score"]
    fs = score["final_status"]
    console.print(f"[bold]{paper_id}[/bold]  —  {ts:.1f}/100  [{fs}]")
    for k, v in score.items():
        if k.endswith("_score") and k != "total_score":
            console.print(f"  {k}: {v}")


@app.command("mastery-report")
def mastery_report(
    paper_id: str = typer.Argument(..., help="Makale paper_id"),
) -> None:
    """Bir makalenin mastery raporunu göster."""
    import json as _json2
    from pathlib import Path

    report_path = Path("reports/papers/mastery") / f"{paper_id}_mastery_report.json"
    if not report_path.exists():
        console.print(f"[yellow]Rapor bulunamadı:[/yellow] {report_path}")
        raise typer.Exit(1)
    data = _json2.loads(report_path.read_text())
    score = data.get("score", {})
    ts = score.get("total_score", 0)
    fs = score.get("final_status", "?")
    console.print(f"[bold]{paper_id}[/bold]  total={ts:.1f}  status={fs}")
    q_total = data.get("questions", 0)
    q_pass = data.get("passed", 0)
    q_fail = data.get("failed", 0)
    console.print(f"Sorular: {q_total}  Geçti: {q_pass}  Kaldı: {q_fail}")


# ---------------------------------------------------------------------------
# arxiv-sync (Görev D: kayıtlı sorguları otomatik yeniden çalıştır)
# ---------------------------------------------------------------------------


@app.command("arxiv-sync")
def arxiv_sync(
    dry_run: bool = typer.Option(False, "--dry-run", help="Gerçekten çalıştırmadan listele"),
    force: bool = typer.Option(False, "--force", help="Son çalıştırma zamanına bakma"),
) -> None:
    """Kayıtlı arXiv sorgularını yeniden çalıştır, yeni makaleleri ingest et."""
    from app.ingestion.arxiv_fetcher import fetch_arxiv_papers
    from app.memory.sqlite_store import SqliteStore as _S

    store = _S()
    queries = store.list_arxiv_saved_queries()
    if not queries:
        console.print("[yellow]Kayıtlı arXiv sorgusu yok. Önce:[/yellow]")
        console.print('  uv run hektor arxiv "sorgu terimi"')
        return

    console.print(f"[bold]{len(queries)} kayıtlı sorgu bulundu.[/bold]")
    total_new = 0

    for q in queries:
        if dry_run:
            console.print(
                f"  [dim][dry-run][/dim] {q['query'][:60]}  "
                f"(son çalışma: {q['last_run_at'] or 'hiç'})"
            )
            continue
        if not q["auto_ingest"] and not force:
            console.print(f"  [dim]atla (auto_ingest=False):[/dim] {q['query'][:50]}")
            continue

        console.print(f"  ↻  {q['query'][:60]} ...")
        try:
            results = fetch_arxiv_papers(q["query"], max_results=q["max_results"])
            new_count = sum(1 for r in results if not r.skipped)
            total_new += new_count
            store.mark_arxiv_query_ran(q["query_id"])
            console.print(f"     [green]✓[/green] {new_count} yeni makale / {len(results)} toplam")
        except Exception as exc:
            console.print(f"     [red]✗[/red] Hata: {exc}")

    if not dry_run:
        console.print(f"\n[bold green]Toplam {total_new} yeni makale eklendi.[/bold green]")
        if total_new > 0:
            console.print("[dim]Yeni makaleler için: uv run hektor ingest[/dim]")


# ---------------------------------------------------------------------------
# mastery-to-sft (Mastery cevaplarını SFT eğitim verisi olarak dışa aktar)
# ---------------------------------------------------------------------------


@app.command("mastery-to-sft")
def mastery_to_sft(
    min_score: float = typer.Option(75.0, help="Minimum mastery skoru (0-100)"),
    citation: float = typer.Option(0.5, help="Minimum citation_score (0-1)"),
    output: str = typer.Option("", help="Çıktı JSONL dosya yolu"),
) -> None:
    """Mastery test sonuçlarından SFT eğitim JSONL'i üret."""
    from app.training.mastery_sft_builder import MasterySFTBuilder

    builder = MasterySFTBuilder()
    out_path = None if not output else __import__("pathlib").Path(output)
    path, n = builder.build_jsonl(
        output_path=out_path,
        min_mastery_score=min_score,
        citation_threshold=citation,
    )
    if n == 0:
        console.print(
            f"[yellow]Skor ≥ {min_score} olan ve citation ≥ {citation} olan "
            "cevap bulunamadı.[/yellow]"
        )
        console.print(
            "[dim]İpucu: Önce mastery testleri çalıştır:[/dim] "
            "hektor mastery-queue --enqueue-all && hektor mastery-queue --run-all"
        )
    else:
        console.print(f"[green]✓[/green] {n} SFT örneği → [bold]{path}[/bold]")


@app.command("unified-dataset")
def unified_dataset(
    min_score: float = typer.Option(75.0, help="Minimum mastery skoru"),
    citation: float = typer.Option(0.5, help="Minimum citation_score"),
    output: str = typer.Option("", help="Çıktı JSONL dosya yolu"),
) -> None:
    """Tüm SFT kaynaklarını birleştirip unified_sft.jsonl üret (LoRA faz 2)."""
    from pathlib import Path as _P

    from app.training.unified_dataset import UnifiedDatasetBuilder

    builder = UnifiedDatasetBuilder()
    stats = builder.build(
        output_path=_P(output) if output else None,
        min_mastery_score=min_score,
        citation_threshold=citation,
    )
    console.print("[bold green]✓ Unified dataset hazır[/bold green]")
    console.print(f"  {stats.summary()}")
    console.print(f"  → [bold]{stats.output_path}[/bold]")
    console.print()
    console.print("[dim]LoRA eğitimi için:[/dim]")
    console.print(f"  uv run hektor train --run --data {stats.output_path}")


# ---------------------------------------------------------------------------
# LoRA Training Control Plane (Gate 0-8 denetim hattı)
# ---------------------------------------------------------------------------


@app.command("lora-audit")
def lora_audit(
    dry_run: bool = typer.Option(
        True, "--dry-run/--run", help="Yalnız denetle (varsayılan); --run ile tam hat."
    ),
) -> None:
    """LoRA dataset denetim hattını (Gate 0-7, --run ile 0-8) çalıştır."""
    from app.lora.control_plane import LoRAControlPlane
    from app.memory.sqlite_store import SqliteStore

    plane = LoRAControlPlane(store=SqliteStore())
    report = plane.run_audit() if dry_run else plane.run_full()

    table = Table(title="LoRA Denetim — Kapılar")
    table.add_column("Gate", justify="right")
    table.add_column("Ad")
    table.add_column("Sonuç")
    table.add_column("Red", justify="right")
    table.add_column("İnceleme", justify="right")
    for stage in report.stages:
        status = "[green]PASS[/green]" if stage.passed else "[red]FAIL[/red]"
        table.add_row(
            str(stage.gate_id),
            stage.name,
            status,
            str(stage.rejected_count),
            str(stage.review_count),
        )
    console.print(table)
    console.print(
        f"Girdi: {report.total_input} | Onaylanan: {report.total_approved} | "
        f"Reddedilen: {report.total_rejected} | İnceleme: {report.total_review_needed}"
    )
    verdict = "[green]GEÇTİ[/green]" if report.passed else "[red]BAŞARISIZ[/red]"
    console.print(f"Genel sonuç: {verdict}")

    settings = get_settings()
    report_path = settings.root / "reports" / "lora" / "audit_report.md"
    plane.generate_report(report, output_path=report_path)
    console.print(f"[dim]Rapor:[/dim] {report_path}")


@app.command("lora-curate")
def lora_curate(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--run",
        help="Yalnız raporla (varsayılan); --run ile lora_eligible=0 uygula.",
    ),
) -> None:
    """Orphan + çok-versiyon kartları LoRA eğitiminden çıkar (lora_eligible=0).

    Orphan = paper_id papers tablosunda yok (kaynak yok → kural 7). Çok-versiyon =
    aynı paper'dan birden çok çelişkili kart (v5 disiplin kökü); paper başına en zengin
    tek kart tutulur. Mutasyon GERİ ALINABİLİR (review_status korunur, kart silinmez).
    Eğitim BAŞLATMAZ.
    """
    from app.lora.card_curation import apply_curation, curation_markdown
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    report = apply_curation(store, dry_run=dry_run)

    table = Table(title="LoRA Kart Kürasyon")
    table.add_column("Ölçü")
    table.add_column("Değer", justify="right")
    table.add_row("Uygun kart (approved+eligible)", str(report.total_eligible))
    table.add_row(
        "Orphan düşürülen",
        f"{report.orphan_demoted} ({report.distinct_orphan_papers} paper)",
    )
    table.add_row(
        "Çok-versiyon düşürülen",
        f"{report.redundant_demoted} ({report.distinct_collapsed_papers} paper)",
    )
    table.add_row("Tutulan (kanonik)", str(report.kept))
    table.add_row("Toplam düşürülen", str(report.total_demoted))
    console.print(table)

    mode = "[yellow]DRY-RUN[/yellow] (DB'ye yazılmadı)" if dry_run else "[green]UYGULANDI[/green]"
    console.print(f"Mod: {mode}")
    if dry_run and report.total_demoted:
        console.print("[dim]Uygulamak için:[/dim] uv run hektor lora-curate --run")

    settings = get_settings()
    report_path = settings.root / "reports" / "lora" / "curation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(curation_markdown(report), encoding="utf-8")
    console.print(f"[dim]Rapor:[/dim] {report_path}")


@app.command("lora-dataset")
def lora_dataset(
    output: Path = typer.Option(
        Path("data/lora_sft"), "--output", help="Çıktı dizini (JSONL buraya yazılır)."
    ),
) -> None:
    """Onaylı kartlardan LoRA SFT JSONL veri seti üret."""
    from app.lora.dataset_builder import build_dataset, export_jsonl
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    cards = store.list_approved_cards()
    examples = build_dataset(cards)
    settings = get_settings()
    out_dir = output if output.is_absolute() else (settings.root / output)
    out_path = out_dir / "lora_sft.jsonl"
    n = export_jsonl(examples, out_path)
    if n == 0:
        console.print(
            "[yellow]Uygun (approved + eligible) kart bulunamadı. Önce kartları onayla.[/yellow]"
        )
        return
    console.print(f"[green]✓[/green] {n} LoRA örneği → [bold]{out_path}[/bold]")

    # Eğitim split'i (train/valid) → jsonl_dir. PEFT/MLX eğitimi ve
    # auto-pipeline bu dosyaları okur. Çok küçük veri setinde valid boş kalmasın.
    from app.lora.dataset_splitter import split_dataset

    split = split_dataset(examples)
    train_ex = list(split.train) or list(examples)
    valid_ex = list(split.valid)
    if not valid_ex and len(train_ex) > 1:
        valid_ex = train_ex[-1:]
        train_ex = train_ex[:-1]
    jsonl_dir = settings.jsonl_dir
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    nt = export_jsonl(train_ex, jsonl_dir / "train.jsonl")
    nv = export_jsonl(valid_ex, jsonl_dir / "valid.jsonl")
    console.print(f"[green]✓[/green] eğitim split → train={nt}, valid={nv}")
    console.print(f"  → [bold]{jsonl_dir}[/bold]")


@app.command("synth-qa")
def synth_qa(
    per_chunk: int = typer.Option(5, "--per-chunk", help="Chunk başına üretilecek QA sayısı"),
    max_chunks: int = typer.Option(12, "--max-chunks", help="Makale başına maksimum chunk"),
    max_papers: int = typer.Option(0, "--max-papers", help="İşlenecek makale sınırı (0=tümü)"),
    output: Path = typer.Option(
        Path("data/lora_sft"), "--output", help="Çıktı dizini (synthetic_qa.jsonl buraya)."
    ),
    append: bool = typer.Option(
        True, "--append/--overwrite", help="Mevcut JSONL'e ekle (döngüde birikim) / üzerine yaz."
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Determinizm tabanı; örneklemeyi sabitler (CPU'da yaklaşık tekrar-üretim).",
    ),
) -> None:
    """Makale chunk'larından LLM ile sentetik, grounded QA eğitim seti üret.

    15-50 şablon örneğinden ~1000+ çeşitli örneğe geçiş motoru (büyüme motoru).
    Eğitim BAŞLATMAZ (CLAUDE.md kural 8); yalnız veri üretir. Ollama gerektirir.
    Detay: docs/RAG_EGITIM_YENIDEN_TASARIM.md (Faz A6).
    """
    import os

    from app.brain.local_llm import LocalLLM
    from app.brain.synthetic_qa_builder import dedup_jsonl_lines, generate_synthetic_dataset
    from app.memory.sqlite_store import SqliteStore

    llm = LocalLLM()
    if not llm.available():
        console.print(
            "[red]LLM kullanılamıyor.[/red] Ollama'yı başlat veya bir API anahtarı ekle "
            "(sentetik üretim LLM gerektirir)."
        )
        raise typer.Exit(code=1)

    store = SqliteStore()
    console.print(
        f"[cyan]Sentetik QA üretimi başlıyor[/cyan] (backend={llm.active_backend()}, "
        f"chunk başına {per_chunk}, makale başına ≤{max_chunks} chunk)…"
    )
    examples, stats = generate_synthetic_dataset(
        store,
        llm=llm,
        per_chunk=per_chunk,
        max_chunks_per_paper=max_chunks,
        max_papers=(max_papers or None),
        seed=seed,
    )

    settings = get_settings()
    out_dir = output if output.is_absolute() else (settings.root / output)
    out_path = out_dir / "synthetic_qa.jsonl"

    # Birikim: içerik-hash + near-duplicate (token Jaccard) dedup (Faz A7).
    new_lines = [ex.to_jsonl_line() for ex in examples]
    existing: list[str] = []
    if append and out_path.exists():
        existing = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    base = len(dedup_jsonl_lines(existing))
    merged = dedup_jsonl_lines([*existing, *new_lines])
    added = len(merged) - base
    total = len(merged)

    # Atomik yaz: tmp'ye yaz → os.replace. Timeout/SIGKILL birikmiş dosyayı bozmasın.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    tmp_path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
    os.replace(tmp_path, out_path)

    table = Table(title="Sentetik QA Üretimi")
    table.add_column("Metrik")
    table.add_column("Değer", justify="right")
    table.add_row("İşlenen makale", str(stats["papers"]))
    table.add_row("Ham örnek", str(stats["raw"]))
    table.add_row("Net (bu tur)", str(stats["kept"]))
    table.add_row("Elenen (dup+kalite)", str(stats["rejected"]))
    table.add_row("Dosyaya yeni eklenen", str(added))
    table.add_row("Toplam satır (birikmiş)", str(total))
    console.print(table)
    if total == 0:
        console.print(
            "[yellow]Hiç örnek üretilmedi.[/yellow] Önce makale ingest et "
            "(uv run hektor arxiv/ingest); chunk'lar gerekli."
        )
        return
    console.print(f"[green]✓[/green] +{added} yeni → toplam {total} → [bold]{out_path}[/bold]")
    # NOT: Bu yalnız SATIR SAYISI; eğitim hazırlığı DEĞİL (CLAUDE.md kural 2).
    if total >= 1000:
        console.print(
            "[green]≥1000 satır birikti.[/green] Bu yalnız satır-sayısı eşiğidir — "
            "eğitimden önce kalite denetimi (grounding + dedup + OOS bölme) gerekir."
        )
    else:
        console.print(f"[dim]Hedef ≥1000 satır; şu an {total}. Döngü üretmeye devam etsin.[/dim]")


@app.command("synth-qa-bulk")
def synth_qa_bulk(
    per_chunk: int = typer.Option(4, "--per-chunk", help="Chunk başına QA"),
    max_chunks: int = typer.Option(8, "--max-chunks", help="Makale başına maks chunk"),
    batch: int = typer.Option(5, "--batch", help="Kaç makalede bir checkpoint yazılsın"),
    seed: int = typer.Option(0, "--seed"),
    target: int = typer.Option(1000, "--target", help="Bu örnek sayısına ulaşınca dur"),
    output: Path = typer.Option(Path("data/lora_sft"), "--output"),
    resume: bool = typer.Option(
        False, "--resume/--no-resume", help="Zaten işlenmiş makaleleri atla (kaldığı yerden devam)"
    ),
) -> None:
    """TÜM korpustan checkpoint'li bulk sentetik QA üret (Stage 2 eşiğine hızlı ulaşım).

    synth-qa (yalnız en yeni N makale) yerine TÜM makaleleri batch'ler hâlinde işler;
    her batch sonrası dosyaya yazar (çökme-güvenli, atomik). Hedefe ulaşınca durur.
    --resume: mevcut çıktıda paper_id'si bulunan makaleleri atla (kesilme sonrası devam).
    Eğitim BAŞLATMAZ (CLAUDE.md kural 8); yalnız veri üretir.
    """
    import json as _json
    import os

    from app.brain.local_llm import LocalLLM
    from app.brain.synthetic_qa_builder import dedup_jsonl_lines, generate_synthetic_dataset
    from app.memory.sqlite_store import SqliteStore

    llm = LocalLLM()
    if not llm.available():
        console.print("[red]LLM kullanılamıyor.[/red] Ollama'yı başlat veya API anahtarı ekle.")
        raise typer.Exit(code=1)

    store = SqliteStore()
    all_ids = [p.paper_id for p in store.list_papers()]
    if not all_ids:
        console.print("[yellow]Makale yok — önce ingest et.[/yellow]")
        return

    settings = get_settings()
    out_dir = output if output.is_absolute() else (settings.root / output)
    out_path = out_dir / "synthetic_qa.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _read_existing() -> list[str]:
        if out_path.exists():
            return [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return []

    def _atomic_write(lines: list[str]) -> None:
        tmp = out_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, out_path)

    # --resume: mevcut JSONL'daki paper_id'lere bakarak işlenmiş makaleleri atla
    if resume:
        processed_ids: set[str] = set()
        for ln in _read_existing():
            try:
                meta = _json.loads(ln).get("metadata") or {}
                pid = meta.get("paper_id") or meta.get("source_id")
                if pid:
                    processed_ids.add(pid)
            except Exception:
                pass
        if processed_ids:
            n_before = len(all_ids)
            all_ids = [pid for pid in all_ids if pid not in processed_ids]
            skipped = n_before - len(all_ids)
            console.print(f"  --resume: {skipped} işlenmiş makale atlandı → {len(all_ids)} kalan")

    n_batches = (len(all_ids) + batch - 1) // batch
    console.print(
        f"[cyan]Bulk sentetik üretim[/cyan] ({len(all_ids)} makale, {n_batches} batch, "
        f"hedef={target}, seed={seed}, backend={llm.active_backend()})…"
    )
    total = len(dedup_jsonl_lines(_read_existing()))
    for bi, i in enumerate(range(0, len(all_ids), batch), 1):
        if total >= target:
            console.print(f"[green]Hedef {target} aşıldı ({total}) — bulk üretim durdu.[/green]")
            break
        batch_ids = all_ids[i : i + batch]
        try:
            examples, _stats = generate_synthetic_dataset(
                store,
                llm=llm,
                per_chunk=per_chunk,
                max_chunks_per_paper=max_chunks,
                paper_ids=batch_ids,
                seed=seed,
            )
        except Exception as exc:  # bir batch çökerse diğerleri devam
            console.print(f"  [red]batch {bi} HATA[/red]: {exc}")
            continue
        merged = dedup_jsonl_lines([*_read_existing(), *[ex.to_jsonl_line() for ex in examples]])
        added = len(merged) - total
        total = len(merged)
        _atomic_write(merged)  # checkpoint
        console.print(f"  batch {bi}/{n_batches} ({len(batch_ids)} makale): +{added} → {total}")

    console.print(
        f"[green]✓[/green] Bulk üretim bitti → toplam {total} örnek → [bold]{out_path}[/bold]"
    )
    if total >= 1000:
        console.print(
            "[green]≥1000 örnek: Stage 2 nicelik eşiği karşılandı.[/green] Sıra: "
            "[bold]lora-audit[/bold] → onay → [bold]lora-cloud-prep[/bold]."
        )


@app.command("lora-split")
def lora_split(
    source: Path = typer.Option(
        Path("data/lora_sft/lora_sft.jsonl"), "--source", help="Bölünecek birleşik JSONL"
    ),
    valid_ratio: float = typer.Option(0.05, "--valid-ratio", help="Doğrulama oranı"),
    seed: int = typer.Option(42, "--seed", help="Determinist bölme (CLAUDE.md kural 6)"),
) -> None:
    """Birleşik dataset'i (lora_sft.jsonl) train/valid'e böl → jsonl_dir.

    Lokal/uzak PEFT eğitimi `data/training/jsonl/{train,valid}.jsonl` okur. Bu komut
    1266-örneklik birleşik dataset'i (synth-qa + kart) o yola yazar. Eğitim BAŞLATMAZ.
    """
    import random

    settings = get_settings()
    src = source if source.is_absolute() else (settings.root / source)
    if not src.exists():
        console.print(f"[red]Kaynak yok:[/red] {src} — önce 'lora-cloud-prep' çalıştır.")
        raise typer.Exit(code=1)

    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        console.print("[yellow]Kaynak boş.[/yellow]")
        return
    random.Random(seed).shuffle(lines)
    n_valid = max(1, int(len(lines) * valid_ratio)) if len(lines) > 1 else 0
    valid, train = lines[:n_valid], lines[n_valid:]

    jd = settings.jsonl_dir
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "train.jsonl").write_text("\n".join(train) + ("\n" if train else ""), encoding="utf-8")
    (jd / "valid.jsonl").write_text("\n".join(valid) + ("\n" if valid else ""), encoding="utf-8")
    console.print(
        f"[green]✓[/green] train=[bold]{len(train)}[/bold] valid=[bold]{len(valid)}[/bold] → {jd}"
    )


@app.command("lora-readiness")
def lora_readiness(
    threshold: int = typer.Option(
        1000, "--threshold", help="Stage 2 (bulut-GPU) eşik örnek sayısı"
    ),
) -> None:
    """Stage 1→Stage 2 eşik durumu: sentetik + kart örnekleri ≥ eşik mi?

    Aşamalı eğitim kapısı (bkz. docs/PROTOKOL_ASAMALI_EGITIM.md). Eğitim BAŞLATMAZ.
    """
    from app.lora.dataset_builder import build_dataset
    from app.memory.sqlite_store import SqliteStore

    settings = get_settings()
    synth_path = settings.root / "data" / "lora_sft" / "synthetic_qa.jsonl"
    synth_n = 0
    if synth_path.exists():
        synth_n = sum(1 for ln in synth_path.read_text(encoding="utf-8").splitlines() if ln.strip())

    try:
        store = SqliteStore()
        card_n = len(build_dataset(store.list_approved_cards()))
    except Exception:
        card_n = 0

    total = synth_n + card_n
    pct = min(100, round(total * 100 / threshold)) if threshold else 0
    gate_ok = total >= threshold

    table = Table(title="LoRA Eğitim Hazırlığı (Stage 1 → Stage 2)")
    table.add_column("Metrik")
    table.add_column("Değer", justify="right")
    table.add_row("Sentetik örnek (synth-qa)", str(synth_n))
    table.add_row("Kart örneği (approved)", str(card_n))
    table.add_row("Toplam", str(total))
    table.add_row("Eşik (bulut-GPU)", str(threshold))
    table.add_row("İlerleme", f"%{pct}")
    console.print(table)

    if gate_ok:
        console.print(
            "[green]✓ Nicelik eşiği karşılandı.[/green] Sıradaki kapılar (Stage 2'den önce):\n"
            "  1) [bold]uv run hektor lora-audit[/bold] (Gate 0-7 kalite denetimi)\n"
            "  2) Kullanıcı onayı (gerçek eğitim yalnız açık komutla — CLAUDE.md kural 8)\n"
            "  3) [bold]uv run hektor lora-cloud-prep[/bold] → bulut-GPU notebook"
        )
    else:
        kalan = threshold - total
        console.print(
            f"[yellow]Henüz eşik altında[/yellow] — {kalan} örnek daha gerekli. "
            "Stage 1 üretimine devam: [bold]bash scripts/continuous-learning.sh 72[/bold] "
            "veya [bold]uv run hektor synth-qa[/bold]."
        )


@app.command("lora-cloud-prep")
def lora_cloud_prep(
    hf_repo: str = typer.Option(
        "KULLANICI/hektor-lora-sft",
        "--hf-repo",
        help="HF private dataset repo (kendi kullanıcı adınla)",
    ),
    adapter_name: str = typer.Option("hektor_lora_cloud", "--adapter-name"),
    lora_r: int = typer.Option(16, "--lora-r", help="LoRA rank (4B için 16-32)"),
    epochs: int = typer.Option(2, "--epochs", help="Epoch (≥1000 örnek için 2-3)"),
    max_seq_len: int = typer.Option(2048, "--max-seq-len", help="T4 güvenli 2048"),
    output: Path = typer.Option(Path("notebooks"), "--output", help="Notebook + Modelfile dizini"),
    discipline: bool = typer.Option(
        True,
        "--discipline/--no-discipline",
        help="Adversarial disiplin örneklerini karıştır (#4 Fix B)",
    ),
    discipline_ratio: float = typer.Option(
        0.25, "--discipline-ratio", help="Disiplin payı (disiplin/(taban+disiplin)); v5 dersi ~0.25"
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="LoRA reçete profili (configs/lora/lora_profiles.yaml); örn. discipline_safe. "
        "Notebook'a lr/alpha/dropout/rsLoRA/NEFTune/weight_decay/warmup taşır.",
    ),
    seed: int = typer.Option(0, "--seed", help="Determinizm tabanı (karıştırma — kural 6)"),
) -> None:
    """Stage 2 bulut-GPU eğitimini HAZIRLA: veri paketle + notebook + Modelfile üret.

    Eğitim BAŞLATMAZ (CLAUDE.md kural 8). Üretir: birleşik JSONL (HF'e yüklenecek),
    doğrulanmış unsloth notebook'u (Kaggle/Colab), Ollama Modelfile + adım talimatları.
    Birleşik sete ~%25 adversarial disiplin örneği karıştırılır (v5 regresyon fix'i #4 Fix B;
    `--no-discipline` ile kapatılır). Detay: docs/PROTOKOL_BULUT_EGITIM.md.
    """
    from app.brain.synthetic_qa_builder import dedup_jsonl_lines
    from app.lora.dataset_builder import build_dataset
    from app.memory.sqlite_store import SqliteStore
    from app.training.cloud_notebook import build_stage2_notebook, write_modelfile
    from app.training.discipline_dataset import discipline_jsonl_lines, mix_discipline

    settings = get_settings()
    lora_dir = settings.root / "data" / "lora_sft"
    synth_path = lora_dir / "synthetic_qa.jsonl"

    # 1) Birleşik dataset: sentetik + kart örnekleri, hash + near-duplicate dedup (A7).
    lines: list[str] = []
    if synth_path.exists():
        lines += [ln for ln in synth_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    try:
        store = SqliteStore()
        lines += [ex.to_jsonl_line() for ex in build_dataset(store.list_approved_cards())]
    except Exception:
        pass
    merged = dedup_jsonl_lines(lines)

    # 1b) Adversarial disiplin örneklerini karıştır (DEDUP'TAN SONRA — şablon örnekleri
    # near-dup filtresine takılıp toplu elenmesin; v5 REJECT'in asıl fix'i, #4 Fix B).
    disc_stats: dict | None = None
    if discipline and discipline_ratio > 0:
        disc_lines = discipline_jsonl_lines(seed=seed)
        merged, disc_stats = mix_discipline(merged, disc_lines, ratio=discipline_ratio, seed=seed)

    combined = lora_dir / "lora_sft.jsonl"
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
    n = len(merged)

    # 2) Reçete: profil verilirse hiperparametreleri ondan al (discipline_safe = v5 reçetesi:
    #    lr 1e-4 + epoch 1 + dropout 0.1 + NEFTune 5 → catastrophic-forgetting'e karşı).
    recipe: dict = {
        "learning_rate": 2e-4,
        "lora_alpha": None,  # None → build_stage2_notebook 2*r kullanır
        "lora_dropout": 0.0,
        "use_rslora": False,
        "neftune_noise_alpha": 0.0,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
    }
    eff_lora_r, eff_max_seq, eff_epochs = lora_r, max_seq_len, epochs
    if profile:
        from app.training.peft_lora_train import load_lora_profile

        try:
            prof = load_lora_profile(profile)
        except (KeyError, FileNotFoundError) as exc:
            console.print(f"[red]Profil hatası: {exc}[/red]")
            raise typer.Exit(1) from exc
        eff_lora_r = prof.get("lora_r", eff_lora_r)
        eff_max_seq = prof.get("max_seq_length", eff_max_seq)
        eff_epochs = prof.get("epochs", eff_epochs)
        for k in (
            "learning_rate",
            "lora_alpha",
            "lora_dropout",
            "use_rslora",
            "neftune_noise_alpha",
            "weight_decay",
            "warmup_ratio",
        ):
            if k in prof and prof[k] is not None:
                recipe[k] = prof[k]
        console.print(
            f"[cyan]Reçete profili:[/cyan] {profile} (r={eff_lora_r}, epoch={eff_epochs}, "
            f"lr={recipe['learning_rate']}, dropout={recipe['lora_dropout']}, "
            f"NEFTune={recipe['neftune_noise_alpha']}, rsLoRA={recipe['use_rslora']})"
        )

    # 3) Notebook + Modelfile üret (doğrulanmış unsloth şablonu).
    out_dir = output if output.is_absolute() else (settings.root / output)
    nb_path = out_dir / "hektor_lora_stage2.ipynb"
    build_stage2_notebook(
        base_model=settings.peft_base_model,
        adapter_name=adapter_name,
        hf_dataset_repo=hf_repo,
        max_seq_length=eff_max_seq,
        lora_r=eff_lora_r,
        learning_rate=recipe["learning_rate"],
        num_epochs=eff_epochs,
        out_path=nb_path,
        lora_alpha=recipe["lora_alpha"],
        lora_dropout=recipe["lora_dropout"],
        use_rslora=recipe["use_rslora"],
        neftune_noise_alpha=recipe["neftune_noise_alpha"],
        weight_decay=recipe["weight_decay"],
        warmup_ratio=recipe["warmup_ratio"],
    )
    mf_path = write_modelfile(out_dir)

    # 3) Özet + talimat.
    console.print(f"[green]✓[/green] Birleşik veri: [bold]{combined}[/bold] ({n} örnek)")
    if disc_stats is not None:
        console.print(
            f"[green]✓[/green] Disiplin karışımı (#4 Fix B): "
            f"{disc_stats['discipline_used']}/{disc_stats['discipline_pool']} adversarial örnek "
            f"→ pay %{disc_stats['ratio_actual'] * 100:.0f} "
            f"(hedef %{disc_stats['ratio_target'] * 100:.0f})"
        )
    elif not discipline:
        console.print("[yellow]ℹ[/yellow] Disiplin karışımı KAPALI (--no-discipline).")
    if n < 1000:
        console.print(
            f"[yellow]UYARI:[/yellow] {n} örnek < 1000. Az veride overfit eder; önce "
            "[bold]uv run hektor synth-qa[/bold] ile büyüt (Stage 1)."
        )
    console.print(f"[green]✓[/green] Notebook: [bold]{nb_path}[/bold]")
    console.print(f"[green]✓[/green] Modelfile: [bold]{mf_path}[/bold]")
    console.print(
        "\n[bold]Sıradaki adımlar (docs/PROTOKOL_BULUT_EGITIM.md):[/bold]\n"
        f"  1) HF private dataset: [bold]huggingface-cli upload {hf_repo} "
        f"{combined} lora_sft.jsonl --repo-type dataset[/bold]\n"
        "  2) HF READ token → Kaggle Secrets / Colab userdata: ad=HF_TOKEN\n"
        "  2b) (önerilen) discipline_safe reçetesiyle hazırla: "
        "[bold]uv run hektor lora-cloud-prep --profile discipline_safe[/bold]\n"
        "  3) Kaggle (T4×2, Internet ON) veya Colab (T4) → notebook'u Run All\n"
        "  4) İndir: hektor-Q4_K_M.gguf + Modelfile → aynı klasör\n"
        "  5) [bold]ollama create hektor -f Modelfile[/bold]\n"
        "  6) Eval gate: [bold]$env:HEKTOR_LLM_MODEL='hektor'; "
        "uv run hektor evaluate evals/discipline_core.jsonl[/bold]\n"
        "  7) Onaylıysa promote (yalnız kullanıcı onayıyla — kural 8)"
    )


@app.command("discipline-dataset")
def discipline_dataset(
    output: Path = typer.Option(
        Path("data/lora_sft/discipline.jsonl"), "--output", help="JSONL çıktı yolu (yazmak için)."
    ),
    write: bool = typer.Option(False, "--write", help="Dosyaya yaz (varsayılan: yalnız önizleme)."),
    variants: int = typer.Option(
        3, "--variants", help="Her (tuzak, strateji) için varyant sayısı."
    ),
    seed: int = typer.Option(0, "--seed", help="Determinizm tabanı (kural 6)."),
) -> None:
    """Adversarial disiplin SFT örneklerini üret/önizle (#4 Fix B). EĞİTİM BAŞLATMAZ.

    9 tuzak (garanti / backtest'siz / maliyetsiz / kaynak-yok / bağlam-uyumsuz / look-ahead /
    overfit / kaldıraç / grounded-belirsizlik) × stratejiler × varyant → deterministik örnek.
    Bu örnekler `lora-cloud-prep` tarafından otomatik ~%25 karıştırılır; bu komut denetim/
    önizleme içindir (kalite gözden geçirme, offline sınav). Detay: memory/v5-adapter-regression.md.
    """
    import json as _json

    from app.training.discipline_dataset import discipline_jsonl_lines

    lines = discipline_jsonl_lines(seed=seed, variants_per_combo=variants)

    # Tuzak dağılımı + system-siz oran (eval koşuluyla uyum) özeti.
    by_trap: dict[str, int] = {}
    no_system = 0
    for ln in lines:
        msgs = _json.loads(ln).get("messages", [])
        if not any(m.get("role") == "system" for m in msgs):
            no_system += 1
    # trap meta'sı JSONL'de yok (yalnız messages serileşir) → örneklerden say.
    from app.training.discipline_dataset import build_discipline_examples

    for ex in build_discipline_examples(seed=seed, variants_per_combo=variants):
        by_trap[ex.metadata["trap"]] = by_trap.get(ex.metadata["trap"], 0) + 1

    console.print(f"[bold]Disiplin dataset[/bold] — {len(lines)} tekil örnek (seed={seed}):")
    for trap, cnt in sorted(by_trap.items()):
        console.print(f"  • {trap:<16} {cnt}")
    console.print(f"  system-siz örnek: {no_system}/{len(lines)} (eval system-prompt'suz çağırır)")

    if write:
        out = output if output.is_absolute() else (get_settings().root / output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        console.print(f"[green]✓[/green] Yazıldı: [bold]{out}[/bold]")
    else:
        console.print(
            "[dim]Önizleme — yazmak için [bold]--write[/bold]. "
            "Eğitime karıştırma otomatik: [bold]lora-cloud-prep[/bold].[/dim]"
        )


@app.command("reindex-contextual")
def reindex_contextual(
    max_papers: int = typer.Option(0, "--max-papers", help="İşlenecek makale sınırı (0=tümü)"),
) -> None:
    """Korpusu Contextual Retrieval (P2) ile YENİDEN embed et — AĞIR (Ollama).

    Her chunk'a "başlık / bölüm:" ön-eki eklenerek yeniden embed edilir (retrieval
    doğruluğu ↑); Chroma document'ı (orijinal metin) + metadata değişmez. Bittiğinde
    .env'e HEKTOR_RAG_CONTEXTUAL_EMBED=true ekle ki yeni makaleler de eşleşsin.
    Detay: docs/RAG_EGITIM_YENIDEN_TASARIM.md (P2).
    """
    from app.memory.chroma_store import ChromaStore
    from app.memory.embedding_service import EmbeddingService
    from app.memory.paper_indexer import build_embed_text
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    chroma = ChromaStore()
    embedder = EmbeddingService()
    papers = store.list_papers()
    if max_papers:
        papers = papers[:max_papers]
    if not papers:
        console.print("[yellow]Makale yok — önce ingest et.[/yellow]")
        return

    console.print(f"[cyan]Contextual yeniden-embed başlıyor[/cyan] ({len(papers)} makale, Ollama)…")
    total = 0
    for i, p in enumerate(papers, 1):
        chunks = store.list_chunks(p.paper_id)
        if not chunks:
            continue
        embed_texts = [build_embed_text(c.text, p.title, c.section_name, True) for c in chunks]
        try:
            embeddings = embedder.embed(embed_texts)
        except Exception as exc:
            console.print(f"  [red]HATA[/red] {p.paper_id}: {exc}")
            continue
        chroma.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "paper_id": c.paper_id,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number if c.page_number is not None else -1,
                    "section_name": c.section_name or "",
                    "title": p.title or "",
                }
                for c in chunks
            ],
        )
        total += len(chunks)
        console.print(f"  [{i}/{len(papers)}] {p.paper_id}: {len(chunks)} chunk")

    console.print(f"[green]✓[/green] {total} chunk contextual yeniden embed edildi.")
    console.print(
        "[bold].env'e ekle:[/bold] HEKTOR_RAG_CONTEXTUAL_EMBED=true "
        "(yeni makalelerin de eşleşmesi için)."
    )


@app.command("rag-mastery")
def rag_mastery() -> None:
    """RAG'in ne kadar 'öğrendiğini' gösteren ustalık panosu (LLM gerektirmez)."""
    from app.verification.rag_mastery import compute_rag_mastery

    m = compute_rag_mastery()
    comp = m["comprehension_percent"]

    table = Table(title="RAG Ustalık Panosu")
    table.add_column("Metrik")
    table.add_column("Değer", justify="right")
    table.add_row("İçe alınan makale", str(m["n_papers"]))
    table.add_row("Onaylı bilgi kartı", f"{m['n_cards']}  ({m['empty_cards']} içeriksiz/atlandı)")
    table.add_row(
        "Bilgi kapsamı (gerçek kart/makale)",
        f"{m['papers_with_real']}/{m['n_papers']}  %{m['coverage_percent']}",
    )
    table.add_row(
        "LoRA eğitim örneği", f"{m['n_examples']}  (hazırlık %{m['train_readiness_percent']})"
    )
    comp_txt = (
        f"%{comp} ({m['papers_scored']}/{m['n_papers']} makale)"
        if comp is not None
        else f"hesaplanmadı ({m['papers_scored']}/{m['n_papers']}) — LLM ile çalıştır"
    )
    table.add_row("Anlama skoru", comp_txt)
    console.print(table)
    console.print(f"\n[bold]RAG Ustalık (bileşik): %{m['mastery_percent']}[/bold]")
    if comp is None:
        console.print(
            "[dim]Not: Anlama skoru için makale başına LLM doğrulaması gerekir "
            "(eğitim sürerken RAM çakışır — cooldown'da çalıştır).[/dim]"
        )


@app.command("synth-paper")
def synth_paper(
    max_sessions: int = typer.Option(5, help="Makaleye dahil edilecek son oturum sayısı"),
    question: str = typer.Option(None, help="Yalnız bu soruyu içeren oturumlar"),
) -> None:
    """Araştırma oturumlarından sentez makalesi (Markdown) üret — web'den indirilebilir."""
    from app.research.synthesis_paper import generate_synthesis_paper

    path = generate_synthesis_paper(max_sessions=max_sessions, question_filter=question)
    if path is None:
        console.print(
            "[yellow]Araştırma oturumu yok — önce 'hektor research \"soru\"' çalıştır.[/yellow]"
        )
        return
    console.print(f"[green]✓[/green] Sentez makalesi → [bold]{path}[/bold]")
    console.print("Web: http://127.0.0.1:8765 → Araştırma sekmesi → Sentez Makaleleri")


@app.command("lora-registry")
def lora_registry() -> None:
    """Adapter kayıt defterini listele."""
    from app.lora.adapter_registry import AdapterRegistry

    registry = AdapterRegistry()
    records = registry.list_adapters()
    if not records:
        console.print("[yellow]Kayıtlı adapter yok.[/yellow]")
        return

    table = Table(title="LoRA Adapter Kayıt Defteri")
    table.add_column("ID")
    table.add_column("Ad")
    table.add_column("Base Model")
    table.add_column("Durum")
    table.add_column("Eval", justify="right")
    for r in records:
        score = "-" if r.eval_score is None else f"{r.eval_score:.3f}"
        table.add_row(r.adapter_id, r.adapter_name, r.base_model, r.status.value, score)
    console.print(table)

    production = registry.get_production()
    if production:
        console.print(f"[green]Production:[/green] {production.adapter_name}")


@app.command("lora-status")
def lora_status() -> None:
    """LoRA hattının genel durumunu göster (eligible kart sayısı, aşamalar)."""
    from app.lora.adapter_registry import AdapterRegistry
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    approved = store.list_approved_cards()
    pending = store.list_pending_cards()
    eligible = [c for c in approved if c.get("lora_eligible")]

    stage_counts: dict[str, int] = {}
    for card in eligible:
        stage = str(card.get("stage") or "(atanmamış)")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    table = Table(title="LoRA Durum")
    table.add_column("Metrik")
    table.add_column("Değer", justify="right")
    table.add_row("Onaylı kart", str(len(approved)))
    table.add_row("Eligible kart", str(len(eligible)))
    table.add_row("Bekleyen (pending) kart", str(len(pending)))
    for stage, count in sorted(stage_counts.items()):
        table.add_row(f"stage: {stage}", str(count))

    registry = AdapterRegistry()
    table.add_row("Kayıtlı adapter", str(len(registry.list_adapters())))
    production = registry.get_production()
    table.add_row("Production adapter", production.adapter_name if production else "yok")
    console.print(table)


# --------------------------------------------------------------------------
# Anlama Doğrulama sınavları (L3 / L4 / L5 + objektif anlama-skoru)
# --------------------------------------------------------------------------
_STATUS_MARK = {
    "passed": "[green]GEÇTİ[/]",
    "failed": "[red]KALDI[/]",
    "skipped": "[yellow]ATLANDI (LLM yok)[/]",
    "no_data": "[dim]VERİ YOK[/]",
}


def _print_exam_results(results: list, as_json: bool) -> None:
    if as_json:
        console.print_json(json.dumps([r.to_dict() for r in results], ensure_ascii=False))
        return
    table = Table(title="Anlama Sınav Sonuçları")
    table.add_column("Seviye")
    table.add_column("Gösterge")
    table.add_column("Durum")
    table.add_column("Not")
    for r in results:
        mark = _STATUS_MARK.get(r.status, r.status)
        if "max_abs_err" in r.detail:
            note = f"max_hata={r.detail['max_abs_err']:.2g}"
        else:
            note = str(r.detail.get("reason") or r.detail.get("truth") or "")
        table.add_row(r.level, r.name, mark, note[:48])
    console.print(table)


@app.command("exam-l3")
def exam_l3(
    indicator: str = typer.Option("all", help="Gösterge (SMA/EMA/RSI) veya 'all'"),
    period: int = typer.Option(0, help="Periyot (0 = spec varsayılanı)"),
    seed: int = typer.Option(0, help="Determinizm için seed"),
    as_json: bool = typer.Option(False, "--json", help="JSON çıktı"),
) -> None:
    """L3 UYGULAMA sınavı — model formülü TUTULAN sayılara doğru uyguluyor mu (np.allclose)."""
    from app.verification.exams import ApplicationExam, get_spec, list_specs

    exam = ApplicationExam()
    specs = list_specs() if indicator.lower() == "all" else [get_spec(indicator)]
    results = [exam.run(s, period=period or None, seed=seed) for s in specs]
    _print_exam_results(results, as_json)


@app.command("exam-l4")
def exam_l4(
    indicator: str = typer.Option("all", help="Gösterge (SMA/EMA/RSI) veya 'all'"),
    period: int = typer.Option(0, help="Periyot (0 = spec varsayılanı)"),
    factor: int = typer.Option(2, help="Periyot kaç katına çıkarılsın"),
    seed: int = typer.Option(0),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """L4 KARŞIOLGU sınavı — parametre değişiminin yönünü model doğru tahmin ediyor mu."""
    from app.verification.exams import CounterfactualExam, get_spec, list_specs

    exam = CounterfactualExam()
    specs = list_specs() if indicator.lower() == "all" else [get_spec(indicator)]
    results = [exam.run(s, period=period or None, factor=factor, seed=seed) for s in specs]
    _print_exam_results(results, as_json)


@app.command("exam-l5")
def exam_l5(
    data: Path | None = typer.Option(None, help="OHLCV CSV yolu (yoksa sentetik veri)"),
    seed: int = typer.Option(42, help="Sentetik veri seed'i"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """L5 KOMPOZİSYON sınavı — örnek kompozisyon math+novelty+backtest kapılarından geçiyor mu."""
    from app.trading.market_data_loader import generate_synthetic_ohlcv, load_ohlcv
    from app.trading.strategy_ir import example_ir
    from app.verification.exams import CompositionGate

    df = load_ohlcv(data) if data else generate_synthetic_ohlcv(n=2000, seed=seed)
    res = CompositionGate().evaluate_composition(example_ir(), df)
    if as_json:
        console.print_json(json.dumps(res.to_dict(), ensure_ascii=False))
        return
    verdict = "[green]ADAY[/]" if res.candidate else "[red]REDDEDİLDİ[/]"
    table = Table(title=f"L5 Kompozisyon — {res.name}: {verdict}")
    table.add_column("Kapı")
    table.add_column("Sonuç")
    table.add_column("Detay")
    for g in res.gates:
        mark = "[green]GEÇTİ[/]" if g.passed else "[red]KALDI[/]"
        table.add_row(g.gate, mark, "; ".join(g.details)[:60])
    console.print(table)


_LADDER_ORDER = ("Taban", "L1", "L2", "L3", "L4", "L5")


def _ladder_sort_key(level: str) -> tuple[int, str]:
    """Merdiven sırası: Taban→L1→…→L5; listede olmayan seviyeler sona (alfabetik).

    Alfabetik sort 'Taban'ı (merdivenin TABANI) en sona atıyordu — kavramsal olarak yanlış.
    """
    return (
        (_LADDER_ORDER.index(level), "") if level in _LADDER_ORDER else (len(_LADDER_ORDER), level)
    )


def _by_level_summary(by_level: dict) -> str:
    """Seviye kırılımı 'Taban:1/1, L3:2/3' (geçti/notlanan) — merdiven sırasında."""
    return ", ".join(
        f"{k}:{v.get('passed', 0)}/{v.get('passed', 0) + v.get('failed', 0)}"
        for k, v in sorted((by_level or {}).items(), key=lambda kv: _ladder_sort_key(kv[0]))
    )


def _context_summary(ctx: dict) -> str:
    """Snapshot bağlamını 'base·qwen3:4b·rag' gibi kısa özetle (hangi koşulda ölçüldü)."""
    ctx = ctx or {}
    parts = [ctx.get("model_kind") or "", str(ctx.get("llm_model") or "")]
    if ctx.get("with_rag"):
        parts.append("rag")
    return "·".join(p for p in parts if p) or "—"


@app.command("understanding-score")
def understanding_score_cmd(
    seed: int = typer.Option(0),
    full: bool = typer.Option(
        False, "--full", help="Tam merdiven (L5 + L3/L4; --with-rag ile Taban/L1/L2 da)"
    ),
    with_rag: bool = typer.Option(
        False, "--with-rag", help="Taban/L1/L2 için canlı RAG sınavı da koş (LLM + korpus)"
    ),
    record: bool = typer.Option(
        False, "--record", help="Skoru KALICI kaydet (DB + reports/evals/understanding JSON)"
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Objektif ANLAMA SKORU — sınav geçme oranı (kaba %'nin yerine). --record ile kalıcı."""
    from app.verification.exams.understanding_score import (
        score_full_ladder,
        score_indicator_exams,
    )

    score = (
        score_full_ladder(seed=seed, with_rag=with_rag)
        if (full or with_rag)
        else score_indicator_exams(seed=seed)
    )

    rec_info = None
    if record:
        from app.verification.exams.understanding_record import record_understanding

        rec_info = record_understanding(
            score, seed=seed, context={"full": full, "with_rag": with_rag, "source": "cli"}
        )

    if as_json:
        out = score.to_dict()
        if rec_info:
            out["recorded"] = rec_info
        console.print_json(json.dumps(out, ensure_ascii=False))
        return
    rate = (
        "yok (notlanan sınav yok)" if score.pass_rate is None else f"{score.pass_rate * 100:.1f}%"
    )
    levels = _by_level_summary(score.by_level)
    body = (
        f"Geçme oranı: [bold]{rate}[/]\n"
        f"Notlanan: {score.graded}  (geçti {score.passed}, kaldı {score.failed})\n"
        f"Atlanan (LLM yok): {score.skipped} · Veri/yön yok: {score.no_data}\n"
        f"Seviyeler: {levels or '—'}\n"
        f"Durum: {score.status}"
    )
    if rec_info:
        body += f"\n[green]KAYDEDİLDİ[/] · snapshot={rec_info['snapshot_id']}"
    console.print(Panel(body, title="Anlama Skoru (objektif sınav-geçme-oranı)"))


@app.command("understanding-history")
def understanding_history_cmd(
    limit: int = typer.Option(20, help="Kaç anlık görüntü gösterilsin"),
    compare: bool = typer.Option(
        False, "--compare", help="Son iki snapshot'ı kıyasla → regresyon (v5-tipi gerileme) tespiti"
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """KALICI anlama skorlarının geçmişi (zaman serisi) + opsiyonel regresyon kıyası."""
    from app.verification.exams.understanding_record import (
        compare_understanding,
        load_understanding_history,
    )

    rows = load_understanding_history(limit=limit)
    cmp = compare_understanding(rows[1], rows[0]) if (compare and len(rows) >= 2) else None
    if as_json:
        out: dict[str, object] = {"history": rows}
        if cmp:
            out["compare"] = cmp
        console.print_json(json.dumps(out, ensure_ascii=False))
        return
    if not rows:
        console.print(
            "Henüz kayıtlı anlama skoru yok. 'hektor understanding-score --record' ile oluştur."
        )
        return
    table = Table(title="Anlama Skoru Geçmişi (objektif, kalıcı)")
    table.add_column("Zaman (UTC)")
    table.add_column("Oran")
    table.add_column("Notlanan")
    table.add_column("Durum")
    table.add_column("Seviyeler")
    table.add_column("Bağlam")
    for r in rows:
        rate = "—" if r.get("pass_rate") is None else f"{r['pass_rate'] * 100:.0f}%"
        table.add_row(
            str(r.get("created_at", "?"))[:19],
            rate,
            str(r.get("graded", 0)),
            str(r.get("status", "?")),
            _by_level_summary(r.get("by_level") or {}),
            _context_summary(r.get("context") or {}),
        )
    console.print(table)
    if cmp:
        ld = cmp["level_delta"]
        if not cmp["comparable"]:
            console.print(f"[yellow]Kıyas güvenilmez:[/] {cmp['note']}")
        elif cmp["regressed"]:
            console.print(
                f"[red]⚠ REGRESYON:[/] geçme oranı {cmp['delta'] * 100:+.1f}% düştü "
                f"({cmp['pass_rate_prev']:.0%} → {cmp['pass_rate_curr']:.0%})."
            )
            console.print(f"  Seviye Δ: {ld}")
        elif cmp["delta"] is not None:
            console.print(f"[green]Regresyon yok[/] · Δoran {cmp['delta'] * 100:+.1f}% · Δ: {ld}")


# --------------------------------------------------------------------------
# Agent runtime gözlemcisi (Phase 1) — yalnız gözlem; kontrol/onay Phase 2'de
# --------------------------------------------------------------------------
@app.command("agents-list")
def agents_list() -> None:
    """Kayıtlı runtime agent'ları (automation_manifest.yaml) listele."""
    from app.agents.runtime import list_agents

    try:
        agents = list_agents()
    except Exception as exc:
        console.print(f"[red]Manifest okunamadı:[/red] {exc}")
        raise typer.Exit(1) from exc
    t = Table(title="Hektor — runtime agent'lar")
    t.add_column("agent_id")
    t.add_column("otonomi")
    t.add_column("tehlikeli")
    t.add_column("onay")
    t.add_column("vars. açık")
    t.add_column("dosya")
    for a in agents:
        t.add_row(
            a.agent_id,
            a.autonomy.value,
            "⚠️" if a.dangerous else "—",
            "✓" if a.approval_required else "—",
            "✓" if a.default_enabled else "—",
            a.file,
        )
    console.print(t)


@app.command("agents-runs")
def agents_runs(
    limit: int = typer.Option(20, help="Kaç koşu gösterilsin"),
    agent: str = typer.Option(None, "--agent", help="agent_id ile filtrele"),
    status: str = typer.Option(None, "--status", help="status ile filtrele"),
) -> None:
    """Son agent koşularını göster (en yeni önce)."""
    from app.memory.sqlite_store import SqliteStore

    runs = SqliteStore().list_agent_runs(limit=limit, agent_id=agent, status=status)
    if not runs:
        console.print("[yellow]Kayıtlı agent koşusu yok.[/yellow]")
        return
    t = Table(title="Agent koşuları")
    t.add_column("run_id")
    t.add_column("agent")
    t.add_column("durum")
    t.add_column("tetik")
    t.add_column("başladı")
    t.add_column("bitti")
    for r in runs:
        t.add_row(
            r["run_id"],
            r["agent_id"],
            r["status"],
            r.get("trigger_type") or "—",
            str(r.get("started_at") or "")[:19],
            str(r.get("finished_at") or "—")[:19],
        )
    console.print(t)


@app.command("agents-log")
def agents_log(run_id: str) -> None:
    """Bir agent koşusunun olay günlüğünü (events) göster."""
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    run = store.get_agent_run(run_id)
    if not run:
        console.print(f"[red]Koşu bulunamadı:[/red] {run_id}")
        raise typer.Exit(1)
    console.print(
        Panel.fit(
            f"agent: {run['agent_id']}\n"
            f"durum: {run['status']}\n"
            f"tetik: {run.get('trigger_type') or '—'}\n"
            f"başladı: {run.get('started_at') or '—'}\n"
            f"bitti: {run.get('finished_at') or '—'}\n"
            f"hata: {run.get('error') or '—'}",
            title=run_id,
        )
    )
    events = store.list_agent_events(run_id)
    t = Table(title="Olaylar")
    t.add_column("ts")
    t.add_column("kind")
    t.add_column("level")
    t.add_column("mesaj")
    for e in events:
        t.add_row(
            str(e.get("ts") or "")[:23],
            e.get("kind") or "",
            e.get("level") or "",
            e.get("message") or "",
        )
    console.print(t)


@app.command("pretrain-gate")
def pretrain_gate_cmd(
    jsonl: Path = typer.Option(
        Path("data/lora_sft/lora_sft.jsonl"),
        "--jsonl",
        help="Denetlenecek SFT JSONL dosyası.",
    ),
    check_discipline: bool = typer.Option(
        True,
        "--check-discipline/--no-check-discipline",
        help="Disiplin havuzunu da kapsam uyum denetiminden geçir.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """LLM'siz GO / NO-GO ön eğitim kalite kapısı (kural 7 doğrulama, v5 fix'i).

    Veriyi birleştirmeden, eğitim başlatmadan çalışır; salt denetim aracıdır.
    Hard-block: garanti-vaadi regex, açılış-ezberi (>%40 tek bigram), minimum boyut.
    Uyarılar: sızıntı ön-eki, maliyet-token eksikliği, disiplin kapsam açığı.
    """
    from app.training.dataset_quality import audit_dataset
    from app.training.discipline_dataset import discipline_jsonl_lines

    if not jsonl.is_absolute():
        jsonl = get_settings().root / jsonl
    if not jsonl.exists():
        console.print(f"[red]Dosya bulunamadı:[/red] {jsonl}")
        raise typer.Exit(1)

    lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    disc_lines = discipline_jsonl_lines() if check_discipline else None
    report = audit_dataset(lines, discipline_lines=disc_lines)

    if as_json:
        console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
        return

    color = "green" if report.verdict == "GO" else "red"
    opening = (
        f"En sık açılış: '{report.top_opening}' (%{report.top_opening_share * 100:.0f})\n"
        if report.top_opening
        else ""
    )
    console.print(
        Panel(
            f"[bold {color}]{report.verdict}[/bold {color}]\n"
            f"Toplam örnek: {report.total}\n"
            f"Öneri: {report.recommended_epochs} epoch\n"
            + opening
            + (
                "[red]ENGEL:[/red]\n  " + "\n  ".join(report.blockers) + "\n"
                if report.blockers
                else ""
            )
            + (
                "[yellow]UYARI:[/yellow]\n  " + "\n  ".join(report.warnings)
                if report.warnings
                else "[dim]Uyarı yok[/dim]"
            ),
            title=f"Ön Eğitim Kalite Kapısı — {jsonl.name}",
        )
    )


@app.command("local-training-audit")
def local_training_audit_cmd(
    out: Path = typer.Option(
        Path("reports/local_training_orchestrator"),
        "--out",
        help="Rapor çıktısı dizini (markdown + json).",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Bu komut DAİMA salt-rapordur; eğitim başlatmaz.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
    write: bool = typer.Option(True, "--write/--no-write", help="Raporu diske yaz (md+json)."),
) -> None:
    """Lokal eğitim-hazırlık DENETİMİ — SALT RAPOR (Phase 5A).

    Sistem durumunu OKUR (STOP_ALL, bekleyen onaylar, veri, ön-eğitim kalite kapısı,
    AutoLoRA durumu) ve eğitim-hazırlık skoru + riskler üretip reports/ altına yazar.
    GERÇEK EĞİTİM BAŞLATMAZ; onay TÜKETMEZ; `train --run`/`launch`/terfi çağırmaz.
    `--run` gibi bir mod bu fazda DESTEKLENMEZ.
    """
    from app.agents.local_training_orchestrator import (
        REPORT_ONLY_BANNER,
        render_markdown,
        run_audit,
    )

    if not dry_run:
        console.print(
            "[yellow]Not:[/yellow] Bu komut yalnız salt-rapor modunda çalışır; "
            "gerçek eğitim başlatmaz (Phase 5A)."
        )
    out_dir = out if out.is_absolute() else get_settings().root / out
    report = run_audit(out_dir=out_dir, write=write)

    if as_json:
        console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
    else:
        console.print(render_markdown(report))
    console.print(f"[bold]{REPORT_ONLY_BANNER}[/bold]")


@app.command("local-training-request")
def local_training_request_cmd(
    create_approval: bool = typer.Option(
        False,
        "--create-approval",
        help="Readiness READY ise PENDING onay isteği oluştur (onay TÜKETMEZ).",
    ),
    preview: bool = typer.Option(
        False, "--preview", help="Her zaman yalnız ön izleme; onay oluşturmaz."
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
    out: Path = typer.Option(
        Path("reports/local_training_orchestrator"),
        "--out",
        help="İstek raporu çıktı dizini.",
    ),
    write: bool = typer.Option(True, "--write/--no-write", help="İstek raporunu diske yaz."),
) -> None:
    """Onay-kapılı lokal eğitim İSTEĞİ — eğitim BAŞLATMAZ, onay TÜKETMEZ (Phase 5B).

    Varsayılan: audit + ön izleme (onay oluşturmaz). `--create-approval`: readiness
    READY ise PENDING onay isteği oluşturur ve onay komutunu gösterir. STOP_ALL/risk/
    READY-değil → blocked. Hiçbir durumda `launch`/`train --run`/`start_training`/
    `promote`/`require_fresh_approval` çağırmaz.
    """
    from app.agents.local_training_request import REQUEST_BANNER, build_request, render_markdown

    out_dir = out if out.is_absolute() else get_settings().root / out
    result = build_request(
        create_approval=create_approval and not preview,
        preview=preview,
        out_dir=out_dir,
        write=write,
    )

    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
    else:
        console.print(render_markdown(result))
    console.print(f"[bold]{REQUEST_BANNER}[/bold]")


@app.command("local-training-dry-run")
def local_training_dry_run_cmd(
    approval_id: str = typer.Option(
        "", "--approval-id", help="Onay durumunu READ-ONLY kontrol et (tüketmez)."
    ),
    request_json: str = typer.Option(
        "", "--request-json", help="Okunacak 5B istek raporu (json). Boşsa en son istek."
    ),
    mock_adapter_eval: bool = typer.Option(
        True,
        "--mock-adapter-eval/--no-mock-adapter-eval",
        help="Adapter-eval'i mockla (gerçek model çalıştırma bu fazda DESTEKLENMEZ).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
    out: Path = typer.Option(
        Path("reports/local_training_orchestrator"), "--out", help="Rapor çıktı dizini."
    ),
    write: bool = typer.Option(True, "--write/--no-write", help="Dry-run raporunu diske yaz."),
) -> None:
    """Onaylı eğitim isteği DRY-RUN pipeline'ı — EĞİTİM/ONAY-TÜKETİMİ YOK (Phase 5C).

    Audit + 5B isteği okur, (varsa) onayı READ-ONLY kontrol eder (tüketmez), pretrain-gate
    read-only + adapter-eval MOCKED çalıştırır ve bir execution PLANI üretir (uygulamaz).
    `launch`/`train --run`/`start_training`/`promote`/`require_fresh_approval`/
    `request_approval` HİÇBİR ZAMAN çağrılmaz.
    """
    from app.agents.local_training_dryrun import DRYRUN_BANNER, build_dryrun, render_markdown

    out_dir = out if out.is_absolute() else get_settings().root / out
    result = build_dryrun(
        approval_id=approval_id or None,
        request_json=request_json or None,
        mock_adapter_eval=mock_adapter_eval,
        out_dir=out_dir,
        write=write,
    )

    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
    else:
        console.print(render_markdown(result))
    console.print(f"[bold]{DRYRUN_BANNER}[/bold]")


@app.command("local-training-handoff")
def local_training_handoff_cmd(
    approval_id: str = typer.Option(
        "", "--approval-id", help="Onay durumunu READ-ONLY kontrol et (tüketmez)."
    ),
    dryrun_json: str = typer.Option(
        "", "--dryrun-json", help="Okunacak 5C dry-run raporu (json). Boşsa en son dry-run."
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
    out: Path = typer.Option(
        Path("reports/local_training_orchestrator"), "--out", help="Rapor çıktı dizini."
    ),
    write: bool = typer.Option(True, "--write/--no-write", help="Handoff raporunu diske yaz."),
) -> None:
    """İnsan-kapılı gerçek eğitim HANDOFF'u — komutu YAZDIRIR, ÇALIŞTIRMAZ (Phase 5D).

    5C dry-run raporunu + (varsa) onayı READ-ONLY okur. STOP_ALL + dry_run_passed +
    approved_not_consumed ise `ready_for_human_execution` döner ve gerçek eğitim komutunu
    YALNIZ METİN olarak + son checklist verir. `launch`/`train --run`/subprocess/
    `start_training`/`promote`/`require_fresh_approval` HİÇBİR ZAMAN çağrılmaz; onay TÜKETİLMEZ.
    """
    from app.agents.local_training_handoff import build_handoff, render_markdown

    out_dir = out if out.is_absolute() else get_settings().root / out
    result = build_handoff(
        approval_id=approval_id or None,
        dryrun_json=dryrun_json or None,
        out_dir=out_dir,
        write=write,
    )

    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
    else:
        console.print(render_markdown(result))
    if result.get("status") == "ready_for_human_execution":
        console.print(
            "[bold yellow]READY FOR HUMAN EXECUTION[/bold yellow]\nRecommended command:\n"
            f"[cyan]{result.get('recommended_command')}[/cyan]\nThis command was NOT executed."
        )


@app.command("local-training-postcheck")
def local_training_postcheck_cmd(
    handoff_json: str = typer.Option("", "--handoff-json", help="Okunacak handoff raporu (json)."),
    dryrun_json: str = typer.Option("", "--dryrun-json", help="Okunacak dry-run raporu (json)."),
    training_report: str = typer.Option(
        "", "--training-report", help="Okunacak training raporu (json)."
    ),
    adapter_path: str = typer.Option(
        "", "--adapter-path", help="Adapter metadata (yalnız stat — model yüklenmez)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
    out: Path = typer.Option(
        Path("reports/local_training_orchestrator"), "--out", help="Rapor çıktı dizini."
    ),
    write: bool = typer.Option(True, "--write/--no-write", help="Postcheck raporunu diske yaz."),
) -> None:
    """Eğitim-sonrası SALT-OKUMA doğrulama — terfi/eğitim YOK (Phase 5E).

    Handoff/dry-run + training artefaktı + adapter metadata + adapter-eval + understanding-
    score'u READ-ONLY okur. Sonuç yoksa `no_training_run_found`; varsa
    `postcheck_ready_for_human_review` (promotion_recommendation=human_review_required).
    `launch`/`train --run`/`start_training`/`promote`/`require_fresh_approval` çağrılmaz;
    model YÜKLENMEZ, eval ÇALIŞTIRILMAZ, korumalı yollara YAZILMAZ.
    """
    from app.agents.local_training_postcheck import build_postcheck, render_markdown

    out_dir = out if out.is_absolute() else get_settings().root / out
    result = build_postcheck(
        handoff_json=handoff_json or None,
        dryrun_json=dryrun_json or None,
        training_report=training_report or None,
        adapter_path=adapter_path or None,
        out_dir=out_dir,
        write=write,
    )

    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
    else:
        console.print(render_markdown(result))
    console.print("[bold]No adapter was promoted. Human review required.[/bold]")


# --------------------------------------------------------------------------
# Orkestrasyon — dayanıklı eğitim hattı (checkpoint + resume + panic-recovery)
# --------------------------------------------------------------------------
_STAGE_GLYPH = {
    "completed": "[green]✓[/green]",
    "skipped": "[dim]–[/dim]",
    "running": "[cyan]▶[/cyan]",
    "blocked": "[yellow]⏸[/yellow]",
    "failed": "[red]✕[/red]",
    "pending": "[dim]·[/dim]",
}


def _render_orchestration(snap: dict) -> None:
    """Koşu + aşamaları Rich tablo olarak yaz."""
    run = snap.get("run") or {}
    table = Table(title=f"Orkestrasyon — {run.get('run_id', '?')} ({run.get('status', '?')})")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Aşama")
    table.add_column("Durum")
    table.add_column("Mesaj", overflow="fold")
    for st in snap.get("stages") or []:
        glyph = _STAGE_GLYPH.get(st["status"], st["status"])
        table.add_row(str(st["order"]), st["name"], glyph, st.get("message", ""))
    console.print(table)
    if run.get("error"):
        console.print(f"[red]Hata:[/red] {run['error']}")


@app.command("orchestrate-start")
def orchestrate_start_cmd(
    model: str = typer.Option("", "--model", help="LLM/base model (boşsa ayardan)."),
    profile: str = typer.Option("discipline_safe_local", "--profile", help="LoRA profili."),
    adapter: str = typer.Option("hektor_lora", "--adapter", help="Adapter adı."),
    iters: int = typer.Option(300, "--iters", help="Eğitim adım sayısı (öneri/önizleme)."),
    hunt_ack: bool = typer.Option(
        False,
        "--hunt-ack",
        help="Kademe-2 derin av TAMAMLANDI olarak işaretle (ZORUNLU gate).",
    ),
    run: bool = typer.Option(
        True, "--run/--no-run", help="Başlattıktan sonra blocked olana dek ilerlet."
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """Dayanıklı eğitim orkestrasyonu BAŞLAT — checkpoint'li, resume edilebilir.

    Salt-okuma aşamalarını (ön kontrol/veri-kapısı/müfredat/kuru-çalıştırma) gözetimsiz
    yürütür; insan kapılarında (derin-av onayı, eğitim onayı) DURUR. Gerçek eğitim ASLA
    gözetimsiz başlamaz (Kural 8). İlerlemeyi `orchestrate-status <run_id>` ile izle.
    """
    from app.orchestration.orchestrator import TrainingOrchestrator

    model_name = model or getattr(get_settings(), "peft_base_model", "")
    orch = TrainingOrchestrator()
    run_id = orch.start(
        model=model_name,
        profile=profile,
        adapter_name=adapter,
        params={"iters": iters, "hunt_ack": hunt_ack},
    )
    snap = orch.run_until_blocked(run_id) if run else orch.status(run_id)
    if as_json:
        console.print_json(json.dumps(snap, ensure_ascii=False))
    else:
        console.print(f"[bold]Koşu:[/bold] {run_id}")
        _render_orchestration(snap)
        console.print(
            "[dim]Sürdür: hektor orchestrate-resume "
            f"{run_id} --hunt-ack (gerektiğinde onay sonrası)[/dim]"
        )


@app.command("orchestrate-status")
def orchestrate_status_cmd(
    run_id: str = typer.Argument(..., help="Koşu kimliği (orc_...)."),
    timeline: bool = typer.Option(False, "--timeline", help="Olay zaman çizelgesini de göster."),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """Bir orkestrasyon koşusunun aşama durumunu (+isteğe bağlı timeline) göster."""
    from app.orchestration.orchestrator import TrainingOrchestrator

    orch = TrainingOrchestrator()
    snap = orch.status(run_id)
    if snap.get("run") is None:
        console.print(f"[red]Koşu bulunamadı:[/red] {run_id}")
        raise typer.Exit(1)
    if as_json:
        payload = dict(snap)
        if timeline:
            payload["events"] = orch.timeline(run_id)
        console.print_json(json.dumps(payload, ensure_ascii=False))
        return
    _render_orchestration(snap)
    if timeline:
        for ev in orch.timeline(run_id):
            console.print(f"[dim]{ev['created_at']}[/dim] [{ev['level']}] {ev['message']}")


@app.command("orchestrate-resume")
def orchestrate_resume_cmd(
    run_id: str = typer.Argument(..., help="Sürdürülecek koşu kimliği."),
    hunt_ack: bool = typer.Option(
        False, "--hunt-ack", help="Derin av tamamlandı işaretle (blocked deep-hunt'ı geçer)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """Bloke/başarısız bir koşuyu SÜRDÜR — tamamlanan aşamalar atlanır (checkpoint).

    `--hunt-ack` verilirse koşu parametresi güncellenir (derin av onayı). Onay kapısı
    için önce `hektor approval-approve <id>` çalıştırın; sonra bu komutla sürdürün.
    """
    from app.orchestration.orchestrator import TrainingOrchestrator

    orch = TrainingOrchestrator()
    run = orch.store.get_run(run_id)
    if run is None:
        console.print(f"[red]Koşu bulunamadı:[/red] {run_id}")
        raise typer.Exit(1)
    if hunt_ack:
        params = dict(run.get("params") or {})
        params["hunt_ack"] = True
        orch.store.update_run(run_id, params_json=json.dumps(params, ensure_ascii=False))
    snap = orch.run_until_blocked(run_id)
    if as_json:
        console.print_json(json.dumps(snap, ensure_ascii=False))
    else:
        _render_orchestration(snap)


@app.command("orchestrate-list")
def orchestrate_list_cmd(
    limit: int = typer.Option(20, "--limit", help="Gösterilecek koşu sayısı."),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """Son orkestrasyon koşularını listele."""
    from app.orchestration.orchestrator import TrainingOrchestrator

    orch = TrainingOrchestrator()
    runs = orch.list_runs(limit=limit)
    if as_json:
        console.print_json(json.dumps(runs, ensure_ascii=False))
        return
    table = Table(title="Orkestrasyon koşuları")
    table.add_column("run_id")
    table.add_column("model")
    table.add_column("durum")
    table.add_column("aşama")
    table.add_column("oluşturma", style="dim")
    for r in runs:
        table.add_row(r["run_id"], r["model"], r["status"], r["current_stage"], r["created_at"])
    console.print(table)


@app.command("orchestrate-recover")
def orchestrate_recover_cmd(
    timeout_min: float = typer.Option(
        30.0, "--timeout-min", help="Kalp atışı bu kadar dakikadan eskiyse stale say."
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """Panic recovery — kalp atışı durmuş 'running' aşamaları failed'a çevir.

    Detached eğitim/işlem çökünce 'running' aşama sonsuza dek asılı kalmasın diye; sonuç
    failed koşular `orchestrate-resume` ile sürdürülebilir (tamamlanan aşamalar atlanır).
    """
    from app.orchestration.orchestrator import TrainingOrchestrator

    orch = TrainingOrchestrator()
    recovered = orch.recover_stale(timeout_min=timeout_min)
    if as_json:
        console.print_json(json.dumps(recovered, ensure_ascii=False))
        return
    if not recovered:
        console.print("[green]Stale (asılı) aşama yok.[/green]")
        return
    for item in recovered:
        console.print(f"[yellow]Kurtarıldı:[/yellow] {item['run_id']} / {item['stage']} → failed")


@app.command("orchestrate-autodrive")
def orchestrate_autodrive_cmd(
    run_id: str = typer.Argument(..., help="Otonom sürülecek koşu kimliği."),
    execute: bool = typer.Option(
        False,
        "--execute/--dry-run",
        help="GERÇEK `claude -p` spawn et (abonelik). Varsayılan dry-run: yalnız komutu gösterir.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """deep-hunt aşamasını headless `claude -p` ile OTONOM sür → onay kapısına kadar ilerlet.

    Kullanıcının "tek tuş → Claude aboneliğiyle devreye soksun" akışının CLI'si. Varsayılan
    DRY-RUN (spawn yok; çalıştırılacak komutu gösterir). `--execute`: gerçek `claude -p` koşar
    (abonelikli Claude Code PATH'te olmalı; API key DEĞİL). Derin av PASS → hunt_ack + onay
    kapısına ilerletir; GERÇEK EĞİTİM yine TAZE insan onayı bekler (Kural 8 — onayda DURUR).
    """
    from app.orchestration.driver import AutoDriver

    res = AutoDriver().drive(run_id, execute=execute)
    if as_json:
        console.print_json(json.dumps(res, ensure_ascii=False))
        return
    if not res.get("ok"):
        console.print(f"[red]Hata:[/red] {res.get('reason')}")
        raise typer.Exit(1)
    if res.get("dry_run"):
        console.print("[bold]DRY-RUN[/bold] — çalıştırılacak komut:")
        console.print(f'  [cyan]{" ".join(res["command"][:2])} "<derin-av-promptu>"[/cyan]')
        console.print(
            f"[dim]Gerçek çalıştırmak için: hektor orchestrate-autodrive {run_id} --execute[/dim]"
        )
        return
    if res.get("hunt_passed"):
        console.print(
            f"[green]Derin av PASS[/green] → onay kapısına ilerletildi "
            f"(durum: {res['run']['status']} @ {res['run']['current_stage']})."
        )
        console.print("[dim]Gerçek eğitim TAZE onay bekler (Kural 8).[/dim]")
    elif res.get("drove"):
        v = res.get("verdict", {})
        console.print(f"[yellow]Derin av {v.get('verdict')}[/yellow] → deep-hunt bloklu kaldı.")
    else:
        console.print(f"[dim]{res.get('reason')}[/dim]")


@app.command("orchestrate-smoke")
def orchestrate_smoke_cmd(
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
    skip_runtime: bool = typer.Option(
        False, "--skip-runtime", help="Canlı runtime yoklamasını atla (yalnız RUN sözleşmeleri)."
    ),
    skip_run_pipeline: bool = typer.Option(
        False, "--skip-run-pipeline", help="⚡ RUN sözleşme yoklamasını atla (yalnız runtime)."
    ),
) -> None:
    """Uçtan-uca DUMAN TESTİ — canlı runtime + ⚡ RUN hattı sözleşmeleri.

    İKİ BÖLÜM:
      1. RUNTIME ("stub≠runtime") — yapılandırılmış LLM backend'i (yerel-öncelikli Ollama)
         gerçek küçük bir üretim + RAG retrieval ile yoklanır. Runtime kapalıysa `skip`
         (kusur değil, "şimdi test edilemez").
      2. ⚡ RUN SÖZLEŞMELERİ — motor sertleştirme bayrakları, MCP config, kurulu-değil
         hatası, dry-run varsayılanı, Kural-8 onay kapısı, sürücü scope 403'leri ve
         ⛔ DURDUR'un koşan motoru gerçekten kesmesi. Çevrimdışı ve deterministik.

    ⛔ GERÇEK MOTOR DOĞURULMAZ (kota yakmaz) — `live-spawn` bilinçli `skip`'tir ve
    raporda "kanıtlanmadı" der. Salt-okuma; eğitim BAŞLATMAZ.
    Çıkış kodu: pass/skip→0, fail→2.
    """
    from app.orchestration.run_smoke import RunPipelineSmoke, RunSmokeResult
    from app.orchestration.smoke import SmokeResult, SmokeRunner

    # İki bölüm AYNI rapor şeklini paylaşır (verdict/summary/checks) → tek döngüde basılır.
    bolumler: list[tuple[str, SmokeResult | RunSmokeResult]] = []
    if not skip_runtime:
        bolumler.append(("runtime", SmokeRunner().run()))
    if not skip_run_pipeline:
        bolumler.append(("run-pipeline", RunPipelineSmoke().run()))

    if as_json:
        console.print_json(
            json.dumps({ad: res.to_dict() for ad, res in bolumler}, ensure_ascii=False, default=str)
        )
    else:
        for ad, res in bolumler:
            color = {"pass": "green", "skip": "yellow", "fail": "red"}.get(res.verdict, "white")
            console.print(f"[{color}]{ad}: {res.verdict.upper()}[/{color}] — {res.summary}")
            for c in res.checks:
                mark = {"pass": "✓", "fail": "✕", "warn": "≈", "skip": "·"}.get(c.status, "?")
                console.print(f"  {mark} [dim]{c.name}[/dim]: {c.detail}")

    if any(res.verdict == "fail" for _ad, res in bolumler):
        raise typer.Exit(2)


@app.command("orchestrate-drive-live")
def orchestrate_drive_live_cmd(
    allow_live_spawn: bool = typer.Option(
        False,
        "--allow-live-spawn",
        help="GERÇEK sür motoru doğur (abonelik KOTASI yakar). Varsayılan KAPALI → spawn yok.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """ELLE tetiklenen TEK canlı sür duman adımı — sür motoru MCP araçlarını görüyor mu?

    ⚠️ Bu KOMUT OTOMATİK DEĞİLDİR ve CI'da ASLA koşmaz. `--allow-live-spawn` OLMADAN hiçbir
    süreç doğurmaz (yalnız neden kapalı olduğunu açıklar). Bayrakla: `claude`'u SÜR (drive)
    argv'siyle bir kez doğurur, çıktıda Hektor MCP araçlarının (`mcp__*`) GERÇEKTEN
    göründüğünü kanıtlar, sonra biter. Önce `uv run hektor-web` çalışıyor olmalı (MCP proxy
    ona bağlanır) ve `claude` aboneliğinle girişli olmalı. Gerçek eğitim BAŞLATMAZ (Kural 8).
    """
    from app.orchestration.run_smoke import LiveDriveSmoke

    res = LiveDriveSmoke().run(allow_live_spawn=allow_live_spawn)
    if as_json:
        console.print_json(json.dumps(res.to_dict(), ensure_ascii=False, default=str))
    else:
        color = {"pass": "green", "skip": "yellow", "fail": "red"}.get(res.verdict, "white")
        console.print(f"[{color}]live-drive: {res.verdict.upper()}[/{color}] — {res.summary}")
        for c in res.checks:
            mark = {"pass": "✓", "fail": "✕", "warn": "≈", "skip": "·"}.get(c.status, "?")
            console.print(f"  {mark} [dim]{c.name}[/dim]: {c.detail}")
    if res.verdict == "fail":
        raise typer.Exit(2)


@app.command("orchestrate-collision")
def orchestrate_collision_cmd(
    baseline_head: str = typer.Option(
        "", "--baseline-head", help="Beklenen HEAD sha; mevcut HEAD kaymışsa çakışma BLOK."
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """Eşzamanlı oturum/worktree ÇAKIŞMASI taraması (git durumu) — salt-okuma.

    Paylaşılan working tree'de başka oturumun `git add -A`'i uncommitted fix'leri süpürebilir
    ya da HEAD altımdan kayabilir. git deposu değilse atlanır (skip). Çıkış kodu: pass/warn→0,
    skip→0, fail→2 (aktif lock / aynı-branch worktree / HEAD kayması — çözülmeli). Aynı yoklama
    orkestrasyon hattının 'collision' aşamasında da koşar.
    """
    from app.orchestration.collision import CollisionDetector

    result = CollisionDetector(baseline_head=baseline_head or None).run()
    if as_json:
        console.print_json(json.dumps(result.to_dict(), ensure_ascii=False))
    else:
        color = {"pass": "green", "skip": "yellow", "warn": "yellow", "fail": "red"}.get(
            result.verdict, "white"
        )
        console.print(f"[{color}]Çakışma: {result.verdict.upper()}[/{color}] — {result.summary}")
        for f in result.findings:
            mark = {"block": "✕", "warn": "≈", "info": "·"}.get(f.severity, "?")
            console.print(f"  {mark} [dim]{f.name}[/dim]: {f.detail}")
    if result.verdict == "fail":
        raise typer.Exit(2)


@app.command("orchestrate-regression")
def orchestrate_regression_cmd(
    commit: bool = typer.Option(
        False, "--commit", help="Mevcut metrikleri YENİ baseline olarak kaydet (explicit)."
    ),
    note: str = typer.Option("", "--note", help="--commit ile kaydedilecek not."),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """Eğitim/onay öncesi GERİLEME denetimi (v5 dersi) — mevcut veri vs son geçen baseline.

    Aday veri setinin v5-ilgili sinyallerini (açılış ezberi, zehir, sızıntı, disiplin kapsamı)
    baseline ile kıyaslar. Baseline yoksa atlanır (skip, ilk koşu); gerileme varsa fail.
    `--commit` mevcut metrikleri yeni baseline yapar (oto-terfi YOK — Kural 8 ruhu). Çıkış kodu:
    pass/skip→0, fail→2. Aynı yoklama hattın 'regression' aşamasında da koşar.
    """
    from app.orchestration.regression import RegressionGuard

    guard = RegressionGuard()
    if commit:
        metrics = guard.commit_baseline(note=note)
        if as_json:
            console.print_json(json.dumps({"committed": metrics}, ensure_ascii=False))
        elif metrics:
            console.print("[green]Baseline kaydedildi[/green] — mevcut metrikler:")
            for k, v in metrics.items():
                console.print(f"  [dim]{k}[/dim]: {v:g}")
        else:
            console.print(
                "[yellow]Metrik hesaplanamadı (veri yok) — baseline kaydedilmedi.[/yellow]"
            )
        return

    result = guard.run()
    if as_json:
        console.print_json(json.dumps(result.to_dict(), ensure_ascii=False))
    else:
        color = {"pass": "green", "skip": "yellow", "fail": "red"}.get(result.verdict, "white")
        console.print(f"[{color}]Gerileme: {result.verdict.upper()}[/{color}] — {result.summary}")
        for f in result.findings:
            mark = {"regressed": "✕", "improved": "↑", "stable": "=", "new": "·"}.get(f.status, "?")
            console.print(f"  {mark} [dim]{f.name}[/dim]: {f.detail}")
    if result.verdict == "fail":
        raise typer.Exit(2)


# --------------------------------------------------------------------------
# Sentinel (Nöbetçi) — tüm ajan/altsistemleri izleyen sağlık monitörü
# --------------------------------------------------------------------------
_SENTINEL_GLYPH = {
    "ok": "[green]✓[/green]",
    "warn": "[yellow]≈[/yellow]",
    "fail": "[red]✕[/red]",
    "skip": "[dim]–[/dim]",
}


@app.command("sentinel")
def sentinel_cmd(
    history: int = typer.Option(0, "--history", help="Son N geçmiş kaydı da göster (0=yok)."),
    no_persist: bool = typer.Option(
        False, "--no-persist", help="Sonucu geçmişe YAZMA (yalnız canlı bak)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """Sentinel (Nöbetçi) — sistem sağlığını uçtan-uca yokla; SALT RAPOR.

    Ollama/web/eğitim/orkestrasyon/STOP_ALL/disk/SQLite/feedback + CPU-çekişmesi
    (danışman Resource Negotiator) yoklanır; hiçbir şey durdurulmaz/başlatılmaz.
    Verdict: ok/warn/fail/skip. Çıkış kodu: fail→2, diğerleri→0.
    """
    from app.monitoring import Sentinel

    sentinel = Sentinel()
    report = sentinel.run(persist=not no_persist)

    if as_json:
        payload: dict = report.to_dict()
        if history > 0:
            payload["history"] = sentinel.history(limit=history)
        console.print_json(json.dumps(payload, ensure_ascii=False))
    else:
        color = {"ok": "green", "warn": "yellow", "fail": "red"}.get(report.overall, "white")
        console.print(
            f"[bold {color}]NÖBETÇİ: {report.overall.upper()}[/bold {color}] — {report.summary}"
        )
        table = Table(show_header=True)
        table.add_column("yoklama")
        table.add_column("durum", justify="center")
        table.add_column("detay", overflow="fold")
        table.add_column("öneri", overflow="fold", style="dim")
        for p in report.probes:
            table.add_row(p.name, _SENTINEL_GLYPH.get(p.status, p.status), p.detail, p.advice)
        console.print(table)
        if history > 0:
            for h in sentinel.history(limit=history):
                console.print(f"[dim]{h['created_at'][:19]}[/dim] {h['overall']} — {h['summary']}")

    if report.overall == "fail":
        raise typer.Exit(2)


# --------------------------------------------------------------------------
# Echo — feedback döngüsü (kullanıcı düzeltmesi → sentetik SFT adayı)
# --------------------------------------------------------------------------
@app.command("feedback-add")
def feedback_add_cmd(
    correction: str = typer.Option(..., "--correction", help="Doğru/düzeltilmiş cevap."),
    question: str = typer.Option("", "--question", help="Orijinal soru/bağlam."),
    bad_answer: str = typer.Option("", "--bad-answer", help="Yanlış olan cevap."),
    source: str = typer.Option("manual", "--source", help="rlm/eval/backtest/manual/<run_id>."),
    correction_type: str = typer.Option(
        "other",
        "--type",
        help="claim_correction|missing_caveat|wrong_number|advice_language_removed|other.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Makine-okunabilir JSON çıktı."),
) -> None:
    """Bir cevap düzeltmesini kaydet (Echo). Garanti/kesinlik dili → ANINDA reddedilir (Kural 1).

    Eğitim BAŞLATMAZ; yalnız aday üretir. Onaylanan düzeltmeler `feedback-export` ile AYRI
    aday SFT dosyasına yazılır (kanonik veriye dokunmaz, Kural 8).
    """
    from app.feedback import EchoCollector

    rec = EchoCollector().record(
        source=source,
        question=question,
        bad_answer=bad_answer,
        correction=correction,
        correction_type=correction_type,
    )
    if as_json:
        console.print_json(json.dumps(rec, ensure_ascii=False))
        return
    color = "green" if rec["status"] == "pending" else "red"
    console.print(f"[{color}]{rec['status'].upper()}[/{color}] {rec['correction_id']}")
    if rec["reject_reason"]:
        console.print(f"[red]Gerekçe:[/red] {rec['reject_reason']}")


@app.command("feedback-list")
def feedback_list_cmd(
    status: str = typer.Option(
        "", "--status", help="pending|approved|rejected|exported (boş=hepsi)."
    ),
    limit: int = typer.Option(30, "--limit"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Feedback düzeltmelerini listele."""
    from app.feedback import EchoCollector

    items = EchoCollector().store.list(status=status or None, limit=limit)
    if as_json:
        console.print_json(json.dumps(items, ensure_ascii=False))
        return
    table = Table(title="Feedback düzeltmeleri")
    table.add_column("id")
    table.add_column("durum")
    table.add_column("tip")
    table.add_column("düzeltme", overflow="fold")
    for it in items:
        table.add_row(
            it["correction_id"], it["status"], it["correction_type"], (it["correction"] or "")[:80]
        )
    console.print(table)


@app.command("feedback-approve")
def feedback_approve_cmd(
    correction_id: str = typer.Argument(..., help="Onaylanacak düzeltme kimliği."),
) -> None:
    """Bir düzeltmeyi onayla (export'a aday). Onay anında güvenlik tekrar kontrol edilir."""
    from app.feedback import EchoCollector

    ok = EchoCollector().approve(correction_id)
    if ok:
        console.print(f"[green]Onaylandı:[/green] {correction_id}")
    else:
        console.print(
            f"[red]Onaylanamadı[/red] (bulunamadı / boş / Kural-1 zehiri): {correction_id}"
        )
        raise typer.Exit(1)


@app.command("feedback-reject")
def feedback_reject_cmd(
    correction_id: str = typer.Argument(..., help="Reddedilecek düzeltme kimliği."),
    reason: str = typer.Option("", "--reason"),
) -> None:
    """Bir düzeltmeyi reddet."""
    from app.feedback import EchoCollector

    ok = EchoCollector().reject(correction_id, reason)
    console.print(
        f"[yellow]Reddedildi:[/yellow] {correction_id}"
        if ok
        else f"[red]Bulunamadı:[/red] {correction_id}"
    )


@app.command("feedback-export")
def feedback_export_cmd(
    out: str = typer.Option(
        "", "--out", help="Aday SFT dosyası (boş=data/feedback/feedback_sft.jsonl)."
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Onaylanan düzeltmeleri AYRI aday SFT dosyasına export et (kanonik veriye dokunmaz).

    EĞİTİM BAŞLATMAZ (Kural 8). Aday yine pretrain-gate + dataset audit'ten geçmelidir.
    """
    from app.feedback import EchoCollector

    res = EchoCollector().export_approved(out_path=out or None)
    if as_json:
        console.print_json(json.dumps(res, ensure_ascii=False))
        return
    console.print(
        f"[green]{res['n_exported']} aday[/green] → {res['path']} "
        f"(yeni işaretlenen: {res['newly_marked']})"
    )
    console.print(
        "[dim]Bu yalnız ADAY veri; eğitime girmeden önce pretrain-gate + audit "
        "gerekir (Kural 8).[/dim]"
    )


@app.command("feedback-status")
def feedback_status_cmd(as_json: bool = typer.Option(False, "--json")) -> None:
    """Feedback özet sayıları (pending/approved/rejected/exported)."""
    from app.feedback import EchoCollector

    s = EchoCollector().summary()
    if as_json:
        console.print_json(json.dumps(s, ensure_ascii=False))
        return
    console.print(
        f"bekleyen={s['pending']} onaylı={s['approved']} "
        f"reddedilen={s['rejected']} export={s['exported']}"
    )


# --------------------------------------------------------------------------
# Agent runtime — task queue + approvals + supervisor (Phase 2)
# --------------------------------------------------------------------------
@app.command("task-create")
def task_create(
    agent: str = typer.Option(..., "--agent", help="agent_id"),
    title: str = typer.Option(..., "--title", help="Görev başlığı"),
    description: str = typer.Option("", "--description", help="Açıklama"),
    requires_approval: bool = typer.Option(
        False, "--requires-approval", help="Çalıştırma onay gerektirir mi"
    ),
) -> None:
    """Yeni bir otomasyon görevi oluştur (pending)."""
    from app.agents.runtime import task_queue

    t = task_queue.create_task(
        agent_id=agent,
        title=title,
        description=description or None,
        requires_approval=requires_approval,
    )
    console.print(f"[green]Görev oluşturuldu:[/green] {t.task_id} ({t.status.value})")


@app.command("tasks-list")
def tasks_list(
    limit: int = typer.Option(50, help="Kaç görev"),
    status: str = typer.Option(None, "--status", help="status ile filtrele"),
) -> None:
    """Otomasyon görevlerini listele (en yeni önce)."""
    from app.agents.runtime import task_queue

    tasks = task_queue.list_tasks(limit=limit, status=status)
    if not tasks:
        console.print("[yellow]Görev yok.[/yellow]")
        return
    t = Table(title="Otomasyon görevleri")
    for c in ("task_id", "agent", "durum", "onay", "başlık", "oluşturuldu"):
        t.add_column(c)
    for x in tasks:
        t.add_row(
            x.task_id,
            x.agent_id,
            x.status.value,
            "✓" if x.requires_approval else "—",
            (x.title or "")[:40],
            str(x.created_at)[:19],
        )
    console.print(t)


@app.command("task-cancel")
def task_cancel(
    run_id: str, reason: str = typer.Option("", "--reason", help="İptal nedeni")
) -> None:
    """Bir görevi iptal et (task_id)."""
    from app.agents.runtime import task_queue

    t = task_queue.cancel_task(run_id, reason=reason or None)
    if t is None:
        console.print(f"[red]Görev bulunamadı:[/red] {run_id}")
        raise typer.Exit(1)
    console.print(f"[yellow]Görev durumu:[/yellow] {t.status.value}")


@app.command("tasks-run")
def tasks_run(
    limit: int = typer.Option(10, help="En çok kaç bekleyen görev işlenir"),
    retry_blocked: bool = typer.Option(
        False, "--retry-blocked", help="blocked_* görevleri önce yeniden kuyruğa al"
    ),
) -> None:
    """Bekleyen görevleri yürüt (hibrit executor: supervisor + taze-onay kapısından geçirir).

    DAG/cron motoru DEĞİL — kuyruk→supervisor köprüsü. Tehlikeli işler (gerçek
    eğitim, terfi) yine TAZE ONAY ister; STOP_ALL aktifse bloklanır. Yalnız
    handler'ı kayıtlı agent_id'ler çalışır (bilinmeyen → başarısız).
    """
    from app.agents.runtime import executor
    from app.agents.runtime.handlers import register_default_handlers

    register_default_handlers()  # güvenli salt-okuma handler'ları (idempotent; tehlikeli HARİÇ)
    results = executor.run_pending(limit=limit, retry_blocked=retry_blocked)
    if not results:
        console.print("[yellow]İşlenecek bekleyen görev yok.[/yellow]")
        return
    ok = sum(1 for r in results if r.get("ok"))
    blocked = sum(1 for r in results if r.get("blocked"))
    failed = len(results) - ok - blocked
    t = Table(title="Executor sonucu")
    for c in ("task_id", "sonuç", "ayrıntı"):
        t.add_column(c)
    for r in results:
        outcome = "✓ tamam" if r.get("ok") else ("⏸ blok" if r.get("blocked") else "✗ hata")
        detail = r.get("reason") or r.get("blocked_by") or r.get("agent_id") or ""
        t.add_row(r.get("task_id", "—"), outcome, str(detail)[:50])
    console.print(t)
    console.print(
        f"[green]{ok} tamam[/green] · [yellow]{blocked} blok[/yellow] · [red]{failed} hata[/red]"
    )


@app.command("approvals-list")
def approvals_list(
    status: str = typer.Option(None, "--status", help="status ile filtrele"),
    limit: int = typer.Option(50, help="Kaç istek"),
) -> None:
    """Onay isteklerini listele (en yeni önce)."""
    from app.agents.runtime import approvals

    items = approvals.list_approvals(status=status, limit=limit)
    if not items:
        console.print("[yellow]Onay isteği yok.[/yellow]")
        return
    t = Table(title="Onay istekleri")
    for c in ("approval_id", "agent", "aksiyon", "risk", "durum", "tüketildi", "özet"):
        t.add_column(c)
    for a in items:
        t.add_row(
            a.approval_id,
            a.agent_id,
            a.action,
            a.risk.value,
            a.status.value,
            "✓" if a.consumed_at else "—",
            (a.summary or "")[:36],
        )
    console.print(t)


@app.command("approval-approve")
def approval_approve(approval_id: str, note: str = typer.Option("", "--note")) -> None:
    """Bir onay isteğini ONAYLA (tek kullanımlık taze onay — standing yetki yok)."""
    from app.agents.runtime import approvals

    a = approvals.approve(approval_id, note=note or None)
    if a is None:
        console.print(f"[red]Onay bulunamadı:[/red] {approval_id}")
        raise typer.Exit(1)
    console.print(f"[green]Durum:[/green] {a.status.value} ({a.action})")


@app.command("approval-reject")
def approval_reject(approval_id: str, note: str = typer.Option("", "--note")) -> None:
    """Bir onay isteğini REDDET."""
    from app.agents.runtime import approvals

    a = approvals.reject(approval_id, note=note or None)
    if a is None:
        console.print(f"[red]Onay bulunamadı:[/red] {approval_id}")
        raise typer.Exit(1)
    console.print(f"[yellow]Durum:[/yellow] {a.status.value} ({a.action})")


@app.command("stop-all")
def stop_all(reason: str = typer.Option("", "--reason", help="Durdurma nedeni")) -> None:
    """KÜRESEL acil-durdurma: tüm tehlikeli aksiyonları blokla (storage/STOP_ALL)."""
    from app.agents.runtime import supervisor

    supervisor.create_stop_all(reason=reason or None)
    console.print(
        Panel.fit(
            f"[bold red]STOP_ALL ETKİN[/bold red]\nNeden: {reason or '—'}\n"
            "Tüm tehlikeli aksiyonlar (gerçek eğitim, terfi) bloklandı.\n"
            "Kaldır: [cyan]uv run hektor clear-stop-all[/cyan]",
            title="🛑 Acil durdurma",
        )
    )


@app.command("clear-stop-all")
def clear_stop_all_cmd() -> None:
    """Küresel acil-durdurmayı (STOP_ALL) kaldır."""
    from app.agents.runtime import supervisor

    r = supervisor.clear_stop_all()
    console.print(f"[green]STOP_ALL kaldırıldı[/green] (önceden aktifti={r['was_active']})")


@app.command("runtime-init")
def runtime_init() -> None:
    """Ajan-runtime ön-uçuş: manifest + Phase-2 tabloları + STOP_ALL doğrula (taze makine kapısı).

    'hektor init' tabloları zaten oluşturur; bu komut bunu DOĞRULAR. Sorun varsa
    sıfır-dışı çıkar → autostart/agent komutlarından önce kapı olarak kullanılabilir.
    """
    from app.agents.runtime.preflight import runtime_preflight

    r = runtime_preflight()
    tbl = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in r["tables"].items())
    lines = [
        f"Ajanlar (manifest) : {r['agents']}  "
        f"(tehlikeli {r['dangerous']}, onay-gerektiren {r['approval_required']})",
        f"STOP_ALL           : {'AKTİF' if r['stop_all'] else 'kapalı'}",
        f"Tablolar           : {tbl}",
    ]
    if r["errors"]:
        lines.append("[red]Hatalar:[/red]\n  " + "\n  ".join(r["errors"]))
    console.print(
        Panel.fit(
            "\n".join(lines),
            title=("[green]Runtime HAZIR[/green]" if r["ok"] else "[red]Runtime SORUNLU[/red]"),
        )
    )
    if not r["ok"]:
        raise typer.Exit(1)


@app.command("chain-status")
def chain_status(
    live: bool = typer.Option(
        False, "--live", help="Her adım için supervisor kapı durumunu da göster (salt-okuma)"
    ),
) -> None:
    """Çalıştırma zincirini (topolojik sıra) göster — tasarım diyagramının tek kaynağı.

    Sıra automation_manifest.yaml 'chain' bölümünden gelir; gate/otonomi AgentSpec'ten.
    --live: her adım için supervisor.can_run_agent ile ŞU ANKİ kapı durumunu ekler
    (hiçbir şey çalıştırmaz/onay tüketmez) → zincirin NEREDE duracağını gösterir.
    """
    from app.agents.runtime.chain import resolve_chain

    steps = resolve_chain()
    t = Table(title="Çalıştırma zinciri (otonom → onay kapısı)")
    for c in ("#", "adım", "otonomi", "kapı"):
        t.add_column(c)
    if live:
        t.add_column("şu an")
    for s in steps:
        gate = "🔒 ONAY" if s.requires_approval else ("⚠ tehlikeli" if s.dangerous else "—")
        row = [str(s.order), s.step, s.autonomy, gate]
        if live:
            from app.agents.runtime import supervisor

            dec = supervisor.can_run_agent(s.step)
            row.append("izinli" if dec.allowed else f"blok:{dec.blocked_by}")
        t.add_row(*row)
    console.print(t)


# --------------------------------------------------------------------------
# Kayıt defteri (dataset / RAG-index / embedding / ödül sürümleri + terfi)
# --------------------------------------------------------------------------
@app.command("registry-list")
def registry_list(
    kind: str = typer.Option(
        "datasets",
        "--kind",
        help="datasets | indices | embeddings | rewards | decisions",
    ),
    limit: int = typer.Option(50),
) -> None:
    """Kayıt defteri sürümlerini listele (sürümleme + denetim izi)."""
    from app.registry import RegistryStore

    reg = RegistryStore()
    if kind == "datasets":
        rows = reg.list_datasets(limit)
        table = Table(title="Dataset Sürümleri")
        for c in ("ID", "Ad", "Tür", "Kayıt", "Onay", "Hash"):
            table.add_column(c)
        for r in rows:
            table.add_row(
                r["dataset_version_id"],
                r["name"][:28],
                r["source_type"],
                str(r["n_records"]),
                r["approval_status"],
                (r["content_hash"] or "-")[:12],
            )
    elif kind == "indices":
        rows = reg.list_rag_indices(limit)
        table = Table(title="RAG İndeks Sürümleri")
        for c in ("ID", "Collection", "Embedding", "Chunk", "Makale", "Zaman"):
            table.add_column(c)
        for r in rows:
            table.add_row(
                r["rag_index_version_id"],
                r["collection_name"],
                r["embedding_model"][:24],
                str(r["n_chunks"]),
                str(r["n_papers"]),
                r["created_at"][:19],
            )
    elif kind == "embeddings":
        rows = reg.list_embeddings(limit)
        table = Table(title="Embedding Modeli Sürümleri")
        for c in ("ID", "Model", "Boyut", "Sağlayıcı", "Zaman"):
            table.add_column(c)
        for r in rows:
            table.add_row(
                r["embedding_model_id"],
                r["model_name"][:28],
                str(r["dimension"] or "-"),
                r["provider"] or "-",
                r["created_at"][:19],
            )
    elif kind == "rewards":
        rows = reg.list_rewards(limit)
        table = Table(title="RLM Ödül Seti Sürümleri")
        for c in ("ID", "Ad", "Yöntem", "Örnek", "Sır", "PII"):
            table.add_column(c)
        _scan = {0: "—", 1: "temiz", 2: "BULGU"}
        for r in rows:
            table.add_row(
                r["reward_version_id"],
                r["name"][:28],
                r["method"] or "-",
                str(r["n_examples"]),
                _scan.get(r["secret_scanned"], "?"),
                _scan.get(r["pii_scanned"], "?"),
            )
    elif kind == "decisions":
        rows = reg.list_decisions(limit=limit)
        table = Table(title="Terfi Kararları (denetim izi)")
        for c in ("Hedef tür", "Hedef ID", "Karar", "Durum", "Onaylayan", "Gerekçe"):
            table.add_column(c)
        for r in rows:
            table.add_row(
                r["target_type"],
                r["target_id"][:16],
                r["decision"],
                r["to_status"],
                r["approved_by"] or "-",
                (r["reason"] or "")[:36],
            )
    else:
        console.print(f"[red]Bilinmeyen kind: {kind}[/red]")
        raise typer.Exit(1)

    if not rows:
        console.print("[yellow]Kayıt yok.[/yellow]")
        return
    console.print(table)


@app.command("registry-snapshot")
def registry_snapshot() -> None:
    """Mevcut RAG indeksinin + embedding modelinin sürüm anlık görüntüsünü al (çevrimdışı)."""
    from app.registry import RegistryStore

    reg = RegistryStore()
    idx = reg.snapshot_rag_index()
    emb = reg.snapshot_embedding()
    console.print(
        Panel.fit(
            f"[green]RAG indeks:[/green] {idx['rag_index_version_id']} "
            f"({idx['n_chunks']} chunk / {idx['n_papers']} makale)\n"
            f"[green]Embedding:[/green] {emb['embedding_model_id']} "
            f"({emb['model_name']} · {emb['provider']})",
            title="registry snapshot",
        )
    )


@app.command("registry-register-dataset")
def registry_register_dataset(
    path: str = typer.Option(..., "--path", help="Dataset dosyası (JSONL) yolu"),
    name: str = typer.Option("", "--name", help="Sürüm adı (boşsa dosya adı)"),
    source_type: str = typer.Option("sft", "--source-type", help="sft | dpo | tool_use"),
) -> None:
    """Bir dataset dosyasını kayıt defterine sürümle (hash+sayı otomatik; sonra promote)."""
    from app.registry import RegistryStore

    reg = RegistryStore()
    try:
        out = reg.register_dataset_from_file(path, name=name or None, source_type=source_type)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    console.print(
        f"[green]Kayıt edildi:[/green] {out['dataset_version_id']} · "
        f"{out['n_records']} kayıt · durum={out['approval_status']} · "
        f"hash={(out['content_hash'] or '')[:12]}\n"
        f"Onay için: hektor registry-promote-dataset --version {out['dataset_version_id']} "
        f"--approver <kim>"
    )


@app.command("registry-promote-dataset")
def registry_promote_dataset(
    version: str = typer.Option(..., "--version", help="dataset_version_id"),
    approver: str = typer.Option(..., "--approver", help="Onaylayan kişi/kimliği"),
    reject: bool = typer.Option(False, "--reject", help="Onaylamak yerine REDDET"),
    reason: str = typer.Option("", "--reason", help="Gerekçe (red için zorunlu)"),
) -> None:
    """Dataset sürümünü onayla/reddet (Kural 8: onaysız eğitime giremez) — karar loglanır."""
    from app.registry import RegistryStore, approve_dataset, reject_dataset

    reg = RegistryStore()
    try:
        if reject:
            if not reason:
                console.print("[red]--reject için --reason zorunlu.[/red]")
                raise typer.Exit(1)
            out = reject_dataset(reg, version, approver, reason)
            console.print(f"[yellow]REDDEDİLDİ[/yellow] · {version} · {reason}")
        else:
            out = approve_dataset(reg, version, approver, reason)
            if out.get("already"):
                console.print(f"[green]Zaten onaylı[/green] · {version}")
            else:
                console.print(f"[green]ONAYLANDI[/green] · {version} · onaylayan={approver}")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


# --------------------------------------------------------------------------
# Bilimsel araç çalışma zamanı (Monte Carlo / istatistik) — saf numpy, seed'li
# --------------------------------------------------------------------------
@app.command("tools-list")
def tools_list(
    category: str = typer.Option("", "--category", help="probability/statistics/trading/risk/..."),
) -> None:
    """Kayıtlı bilimsel araçları listele (keşif + determinizm sözleşmesi)."""
    from app.tools.tool_registry import list_tools

    tools = list_tools(category or None)
    if not tools:
        console.print("[yellow]Araç yok.[/yellow]")
        return
    table = Table(title="Bilimsel Araçlar")
    for c in ("ID", "Kategori", "Seed?", "Açıklama"):
        table.add_column(c)
    for t in tools:
        table.add_row(t.tool_id, t.category, "evet" if t.requires_seed else "—", t.description[:50])
    console.print(table)


def _parse_returns(returns: str, csv: str | None) -> list[float]:
    """Virgüllü liste veya CSV'den işlem getirisi serisi çıkar."""
    if returns.strip():
        return [float(x) for x in returns.split(",") if x.strip()]
    if csv:
        import pandas as pd

        df = pd.read_csv(csv)
        for col in ("trade_return", "return", "ret", "returns"):
            if col in df.columns:
                return [float(x) for x in df[col].dropna().tolist()]
        # tek sayısal kolon → onu kullan
        num = df.select_dtypes("number")
        if num.shape[1] >= 1:
            return [float(x) for x in num.iloc[:, 0].dropna().tolist()]
    raise typer.BadParameter("--returns (virgüllü) ya da --csv (getiri kolonu) ver.")


@app.command("montecarlo")
def montecarlo_cmd(
    seed: int = typer.Option(42, "--seed", help="Determinizm (Kural 6) — aynı seed aynı sonuç"),
    returns: str = typer.Option("", "--returns", help="Virgüllü işlem getirileri (ör. 0.05,-0.02)"),
    csv: str = typer.Option("", "--csv", help="İşlem getirisi kolonu olan CSV yolu"),
    n_paths: int = typer.Option(1000, "--n", help="Simüle edilecek yol sayısı"),
    ruin_fraction: float = typer.Option(0.5, "--ruin", help="Ruin seviyesi (başlangıcın kesri)"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Monte Carlo equity simülasyonu + risk-of-ruin (hipotez; tavsiye değil). Çalışma loglanır."""
    from app.memory.sqlite_store import SqliteStore
    from app.tools.probability_simulator import monte_carlo_equity

    series = _parse_returns(returns, csv or None)
    store = SqliteStore()
    with store.log_tool_run(
        tool_id="montecarlo",
        params={"n_paths": n_paths, "ruin_fraction": ruin_fraction, "n_returns": len(series)},
        seed=seed,
    ) as run_id:
        result = monte_carlo_equity(series, seed=seed, n_paths=n_paths, ruin_fraction=ruin_fraction)
        store.set_tool_run_output(run_id, result.to_dict())

    if as_json:
        console.print_json(json.dumps(result.to_dict(), ensure_ascii=False))
        return
    console.print(
        Panel(
            f"Yol: {result.n_paths} × {result.n_trades} işlem · seed={result.seed}\n"
            f"Beklenen değer (işlem başı): {result.per_trade_mean:+.4f} "
            f"(std {result.per_trade_std:.4f})\n"
            f"Final equity ort/medyan: {result.mean_final_equity:,.0f} / "
            f"{result.median_final_equity:,.0f}\n"
            f"VaR%95: {result.var_95_pct:.1f}% · ES%95: {result.expected_shortfall_pct:.1f}%\n"
            f"Kayıp olasılığı: {result.prob_loss:.1%} · "
            f"Ruin (≤%{result.ruin_fraction * 100:.0f}): {result.ruin_probability:.1%}\n"
            f"[dim]{result.note}[/dim]",
            title="Monte Carlo (risk-of-ruin)",
        )
    )


@app.command("stats-check")
def stats_check_cmd(
    csv: str = typer.Option(..., "--csv", help="CSV yolu"),
    x: str = typer.Option("", "--x", help="Korelasyon için X kolonu"),
    y: str = typer.Option("", "--y", help="Korelasyon için Y kolonu"),
    seed: int = typer.Option(42, "--seed"),
    n_perm: int = typer.Option(1000, "--n-perm", help="Permütasyon sayısı"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """İstatistik kontrol: iki kolon → korelasyon + permütasyon p-değeri; tek → betimsel."""
    import pandas as pd

    from app.tools.statistics_checker import correlation_report, describe_series

    df = pd.read_csv(csv)
    if x and y:
        rep = correlation_report(df[x].tolist(), df[y].tolist(), seed=seed, n_permutations=n_perm)
        out = rep.to_dict()
    else:
        col = x or y or df.select_dtypes("number").columns[0]
        out = describe_series(df[col].tolist()).to_dict()

    if as_json:
        console.print_json(json.dumps(out, ensure_ascii=False))
        return
    console.print_json(json.dumps(out, ensure_ascii=False))


# --------------------------------------------------------------------------
# Birleşik değerlendirme çalıştırıcısı (eval-runner) — tek giriş + yayın kapısı
# --------------------------------------------------------------------------
@app.command("eval-runner")
def eval_runner_cmd(
    eval_type: str = typer.Option(
        "trading-hypothesis", "--type", help="trading-hypothesis | rag-retrieval"
    ),
    input_path: str = typer.Option(
        "", "--input", help="trading-hypothesis için hipotez JSONL/JSON yolu"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Kapı geçilemezse hata ver (production engelle)"
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Değerlendirmeyi çalıştır ve ReleaseGate'ten geçir; rapor reports/evals'a yazılır."""
    from app.evals.eval_runner import EvalGateError, EvalRunner

    runner = EvalRunner()
    try:
        if eval_type == "trading-hypothesis":
            if not input_path:
                console.print("[red]--input (hipotez JSON/JSONL) gerekli.[/red]")
                raise typer.Exit(1)
            hyps = _load_hypotheses(input_path)
            result = runner.run("trading-hypothesis", hypotheses=hyps, strict=strict)
        else:
            console.print(
                f"[yellow]'{eval_type}' CLI'dan henüz çalıştırılamıyor "
                "(retriever/kaynak gerektirir); programatik EvalRunner kullan.[/yellow]"
            )
            raise typer.Exit(2)
    except EvalGateError as e:
        console.print(f"[red]KAPI GEÇİLEMEDİ:[/red] {e}")
        raise typer.Exit(1) from e

    if as_json:
        console.print_json(json.dumps(result.to_dict(), ensure_ascii=False))
        return
    mark = "[green]GEÇTİ[/green]" if result.passed else "[red]KALDI[/red]"
    body = (
        f"Tip: {result.eval_type} · kalem: {result.n_items} · {mark}\n"
        f"Metrikler: {json.dumps(result.metrics, ensure_ascii=False)}\n"
        f"Rapor: {result.report_path}"
    )
    if result.failures:
        body += "\nEksikler: " + "; ".join(result.failures)
    console.print(Panel(body, title="eval-runner"))


def _load_hypotheses(path: str) -> list:
    """Hipotezleri JSON listesi veya JSONL'den yükle (her satır/öğe str ya da dict)."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    data = json.loads(raw)
    return data if isinstance(data, list) else [data]


# --------------------------------------------------------------------------
# İçe-alım kalite skoru (compute-on-demand; PaperIndexer sıcak yolu değişmez)
# --------------------------------------------------------------------------
@app.command("ingestion-quality")
def ingestion_quality_cmd(
    paper_id: str = typer.Option(..., "--paper-id", help="Skorlanacak makale paper_id"),
    record: bool = typer.Option(
        False, "--record", help="Skoru KALICI yap (paper_ingestion_runs + Paper alanları)"
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Bir makalenin içe-alım kalite skorunu (100 puan) hesapla. --record ile kalıcı."""
    from app.ingestion.quality_scorer import score_paper
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    try:
        result = score_paper(store, paper_id)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    run_id = None
    if record:
        from app.ingestion.quality_scorer import gather_inputs

        inp = gather_inputs(store, paper_id)
        run_id = store.add_ingestion_run(
            paper_id=paper_id,
            status=result.status,
            quality_score=result.total,
            component_scores=result.components,
            n_chunks=inp.n_chunks,
            n_formulas=inp.n_formulas,
            notes=result.notes,
        )

    if as_json:
        out = result.to_dict()
        if run_id:
            out["ingestion_run_id"] = run_id
        console.print_json(json.dumps(out, ensure_ascii=False))
        return
    comp = "  ".join(f"{k}={v}" for k, v in result.components.items())
    body = (
        f"Toplam: [bold]{result.total}/100[/] · durum: [bold]{result.status}[/]\nBileşenler: {comp}"
    )
    if result.notes:
        body += "\nNot: " + "; ".join(result.notes)
    if run_id:
        body += f"\n[green]KAYDEDİLDİ[/] · run={run_id}"
    console.print(Panel(body, title="İçe-alım Kalite Skoru"))


@app.command("ingestion-quality-scan")
def ingestion_quality_scan_cmd(
    record: bool = typer.Option(False, "--record", help="Skorları KALICI yap (tüm korpus)"),
    worst: int = typer.Option(10, "--worst", help="En düşük kaç makaleyi listele"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Tüm korpusu içe-alım kalitesi için tara: durum dağılımı + en zayıf makaleler."""
    from app.ingestion.quality_scorer import score_all_papers
    from app.memory.sqlite_store import SqliteStore

    summary = score_all_papers(SqliteStore(), record=record, worst_n=worst)
    if as_json:
        console.print_json(json.dumps(summary.to_dict(), ensure_ascii=False))
        return
    if summary.total == 0:
        console.print("[yellow]Makale yok.[/yellow]")
        return
    dist = Table(title=f"İçe-alım Kalite Dağılımı ({summary.scored}/{summary.total} skorlandı)")
    dist.add_column("Durum")
    dist.add_column("Sayı", justify="right")
    for status in ("ready_for_rag", "usable", "slow_but_usable", "unstable", "failed"):
        n = summary.by_status.get(status, 0)
        if n:
            dist.add_row(status, str(n))
    console.print(dist)
    console.print(f"Ortalama skor: [bold]{summary.avg_score}/100[/]")
    if summary.worst:
        worst_t = Table(title=f"En zayıf {len(summary.worst)} makale")
        for c in ("paper_id", "Başlık", "Skor", "Durum"):
            worst_t.add_column(c)
        for w in summary.worst:
            worst_t.add_row(w["paper_id"], w["title"], f"{w['total']}", w["status"])
        console.print(worst_t)
    if record:
        console.print("[green]KAYDEDİLDİ[/] (paper_ingestion_runs + papers.quality_score)")


if __name__ == "__main__":  # pragma: no cover
    app()
