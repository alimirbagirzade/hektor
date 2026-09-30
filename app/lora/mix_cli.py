"""``hektor mix …`` — LoRA karışım profilleri + profil eval CLI'ı (Typer alt uygulaması).

Hiçbir komut eğitim BAŞLATMAZ. ``build`` varsayılan dry-run'dır; ``promote --to production``
regression gate + açık ``--approve`` ister.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from app.lora.mix_common import (
    DOMAINS,
    MERGE_METHODS,
    MixConfigError,
    fingerprint_model_dir,
    format_weights,
    git_commit,
    hash_file,
    hash_obj,
    load_mix_config,
    parse_weights,
    profile_weights,
)

mix_app = typer.Typer(
    help="LoRA karışım profilleri: ağırlık kararı, adapter/profil registry, router, eval.",
    no_args_is_help=True,
)
console = Console()


def _print_json(obj: Any) -> None:
    console.print_json(json.dumps(obj, ensure_ascii=False, default=str))


@mix_app.command("weights")
def weights_cmd(
    profile: str = typer.Option(None, "--profile", help="Hazır profil adı (mix_profiles.yaml)"),
    weights: str = typer.Option(None, "--weights", help='Özel: "math=0.3,statistics=0.2,..."'),
    show: bool = typer.Option(False, "--show", help="Yalnız profilleri + bekleyen kararı göster"),
) -> None:
    """Sıradaki LoRA eğitimi için karışım ağırlığı kararını kaydet (eğitim öncesi zorunlu)."""
    from app.lora.weight_decision import WeightDecisionStore, ask_weights_interactively

    store = WeightDecisionStore()
    cfg = load_mix_config()
    if show:
        for name, w in cfg["profiles"].items():
            console.print(f"  [cyan]{name:<22}[/cyan] {format_weights(w)}")
        pending = store.pending()
        console.print(
            f"Bekleyen karar: {pending.decision_id} ({pending.profile_name}: "
            f"{format_weights(pending.weights)})"
            if pending
            else "Bekleyen karar yok."
        )
        return
    try:
        if profile and weights:
            raise MixConfigError("--profile ve --weights birlikte verilemez")
        if weights:
            w, name = parse_weights(weights), "custom"
        elif profile:
            w, name = profile_weights(profile, cfg), profile
        else:
            w, name = ask_weights_interactively(lambda m: typer.prompt(m), console.print, cfg)
    except MixConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    dec = store.record(w, name, "recorded")
    console.print(
        f"[green]Kaydedildi[/green] {dec.decision_id}: {name} → {format_weights(w)}\n"
        "[dim]Katsayılar semantik yüzde değildir; bir sonraki `train --run` bu kararı "
        "tüketir.[/dim]"
    )


@mix_app.command("register-adapter")
def register_adapter_cmd(
    domain: str = typer.Option(..., help=f"{'|'.join(DOMAINS)}"),
    adapter_path: Path = typer.Option(..., help="PEFT adapter dizini (adapter_config.json)"),
    dataset: Path = typer.Option(..., help="Adapter'ın eğitildiği JSONL (hash için)"),
    version: str = typer.Option(..., help="adapter_version (ör. v1)"),
    base_model_dir: Path = typer.Option(None, help="Yerel base model dizini (hash için)"),
    adapter_id: str = typer.Option(None, help="Varsayılan: <domain>_lora"),
    learning_rate: float = typer.Option(..., help="Eğitim lr"),
    epochs: float = typer.Option(..., help="Epoch"),
    max_seq_length: int = typer.Option(2048),
    seed: int = typer.Option(42),
    dataset_version: str = typer.Option("", help="Varsayılan: dataset hash[:12]"),
    serving_model: str = typer.Option("", help="Ollama model adı (eval matrisi için)"),
) -> None:
    """Bağımsız eğitilmiş bir domain adapter'ını kayıt defterine ekle."""
    from app.lora.domain_adapter_registry import (
        AdapterRegistryError,
        DomainAdapterRecord,
        DomainAdapterRegistry,
        read_peft_adapter_config,
    )

    try:
        pcfg = read_peft_adapter_config(adapter_path)
        base_hash, tok_hash = fingerprint_model_dir(base_model_dir) if base_model_dir else ("", "")
        if not base_hash:
            console.print("[red]--base-model-dir gerekli (base_model_hash/tokenizer_hash).[/red]")
            raise typer.Exit(2)
        ds_hash = hash_file(dataset)
        train_cfg = {
            "r": pcfg["r"],
            "lora_alpha": pcfg["lora_alpha"],
            "lora_dropout": pcfg["lora_dropout"],
            "target_modules": pcfg["target_modules"],
            "learning_rate": learning_rate,
            "epochs": epochs,
            "max_seq_length": max_seq_length,
            "seed": seed,
        }
        rec = DomainAdapterRecord(
            adapter_id=adapter_id or f"{domain}_lora",
            adapter_version=version,
            domain=domain,
            adapter_path=str(adapter_path),
            base_model=pcfg["base_model"],
            base_model_revision=pcfg["base_model_revision"],
            base_model_hash=base_hash,
            tokenizer_hash=tok_hash,
            dataset_version=dataset_version or ds_hash[:12],
            dataset_hash=ds_hash,
            training_config_hash=hash_obj(train_cfg),
            r=pcfg["r"],
            lora_alpha=pcfg["lora_alpha"],
            lora_dropout=pcfg["lora_dropout"],
            target_modules=pcfg["target_modules"],
            learning_rate=learning_rate,
            epochs=epochs,
            max_seq_length=max_seq_length,
            training_seed=seed,
            git_commit=git_commit(),
            serving_model=serving_model,
        )
        DomainAdapterRegistry().register(rec)
    except (FileNotFoundError, AdapterRegistryError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    console.print(f"[green]Kaydedildi:[/green] {rec.adapter_id}@{rec.adapter_version} ({domain})")


@mix_app.command("adapters")
def adapters_cmd() -> None:
    """Domain adapter kayıt defterini listele."""
    from app.lora.domain_adapter_registry import DomainAdapterRegistry

    t = Table("adapter", "domain", "r", "base#", "eval", "production", "serving")
    for r in DomainAdapterRegistry().records():
        t.add_row(
            f"{r.adapter_id}@{r.adapter_version}",
            r.domain,
            str(r.r),
            r.base_model_hash[:10],
            r.eval_status.value,
            r.production_status.value,
            r.serving_model or "—",
        )
    console.print(t)


@mix_app.command("build")
def build_cmd(
    profile: str = typer.Option(..., help="mix_profiles.yaml profil adı"),
    merge_method: str = typer.Option(..., help=f"{'|'.join(MERGE_METHODS)}"),
    density: float = typer.Option(None, help="ties/dare_ties yoğunluğu"),
    run: bool = typer.Option(False, help="Gerçekten kur (base model yerelden yüklenir)"),
    base_model_path: str = typer.Option(None, help="Yerel base model dizini"),
) -> None:
    """Profil kur — varsayılan DRY-RUN (uyumluluk planı). Base model DEĞİŞMEZ."""
    from app.lora import profile_builder

    argv = ["--profile", profile, "--merge-method", merge_method]
    if density is not None:
        argv += ["--density", str(density)]
    if run:
        argv.append("--run")
    if base_model_path:
        argv += ["--base-model-path", base_model_path]
    raise typer.Exit(profile_builder.main(argv))


@mix_app.command("profiles")
def profiles_cmd() -> None:
    """Profil kayıt defterini listele (durum + eval skorları)."""
    from app.lora.profile_registry import ProfileRegistry

    t = Table("profile_id", "status", "merge", "eval", "grounding", "halluc.", "serving")
    for r in ProfileRegistry().records():
        t.add_row(
            r.profile_id,
            r.status.value,
            r.merge_method,
            "—" if r.eval_score is None else f"{r.eval_score:.3f}",
            "—" if r.grounding_score is None else f"{r.grounding_score:.3f}",
            "—" if r.hallucination_score is None else f"{r.hallucination_score:.3f}",
            r.serving_model or "—",
        )
    console.print(t)


@mix_app.command("set-serving-model")
def set_serving_model_cmd(profile_id: str, model: str) -> None:
    """Profilin Ollama servis modeli adını ayarla (GGUF dönüşümü sonrası; eval C/D için)."""
    from app.lora.mix_common import write_jsonl
    from app.lora.profile_registry import ProfileRegistry

    reg = ProfileRegistry()
    rows = reg.records()
    hit = next((r for r in rows if r.profile_id == profile_id), None)
    if hit is None:
        console.print(f"[red]profil yok: {profile_id}[/red]")
        raise typer.Exit(2)
    hit.serving_model = model
    write_jsonl(reg.path, (r.to_dict() for r in rows))
    console.print(f"[green]{profile_id}[/green] → {model}")


@mix_app.command("route")
def route_cmd(query: str = typer.Argument(...), no_log: bool = typer.Option(False)) -> None:
    """Sorguyu validated bir profile yönlendir (v1: yalnız validated profiller)."""
    from app.lora.profile_router import ProfileRouter

    _print_json(ProfileRouter().route(query, log=not no_log).to_dict())


@mix_app.command(
    "eval", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def eval_cmd(ctx: typer.Context) -> None:
    """Profil eval (A/B/C/D). Argümanlar eval_runner'a aynen geçer: --profile X [--dry-run]."""
    from app.evals.profile import eval_runner

    raise typer.Exit(eval_runner.main(list(ctx.args)))


@mix_app.command(
    "compare", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def compare_cmd(ctx: typer.Context) -> None:
    """Profil karşılaştırma raporu (CSV/JSON/MD): --profiles a b."""
    from app.evals.profile import comparison

    raise typer.Exit(comparison.main(list(ctx.args)))


@mix_app.command(
    "audit", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def audit_cmd(ctx: typer.Context) -> None:
    """İnsan denetimi: --run-id RUN [--sample-size 100] | --score."""
    from app.evals.profile import human_audit

    raise typer.Exit(human_audit.main(list(ctx.args)))


@mix_app.command("leakage")
def leakage_cmd(
    train_jsonl: Path = typer.Option(None, help="Varsayılan: eğitim train.jsonl"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Eğitim verisi ↔ validation/golden sızıntı denetimi (exact/normalized/near-dup/source)."""
    report = run_leakage_check(train_jsonl)
    if as_json:
        _print_json(report)
    else:
        console.print(
            f"{'[green]TEMİZ[/green]' if report['clean'] else '[red]SIZINTI[/red]'} "
            f"— eğitim={report['n_train']} valid={report['n_valid']} eval={report['n_eval']} "
            f"{report['counts']}"
        )
        for h in report["hits"][:20]:
            console.print(
                f"  {h['kind']:<15} {h['eval_id']:<12} {h['split']}#{h['train_index']} "
                f"{h['detail']}"
            )
    if not report["clean"]:
        raise typer.Exit(1)


def adapter_eval_items(eval_dir: Path | None = None) -> list[Any]:
    """Adapter eval setleri (`evals/*.jsonl`: discipline_core, trader_persona …) → EvalItem.

    Kademe-2 C6 (2026-09-30): sızıntı kapısı yalnız profil validation/golden setlerini
    tarıyordu; v10–v12 terfi kararlarını veren bu 80 kalem hiç taranmıyordu. Soru + (varsa)
    bağlam eğitim verisinde geçmemeli.
    """
    from app.config import get_settings
    from app.evals.profile.schema import EvalItem

    d = eval_dir or (get_settings().root / "evals")
    out: list[Any] = []
    for f in sorted(d.glob("*.jsonl")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            q = str(row.get("question") or "").strip()
            if not q:
                continue
            out.append(
                EvalItem(
                    id=f"adapter_eval:{f.stem}:{i}",
                    domain="trading",
                    question=q,
                    reference_answer=str(row.get("context") or ""),
                    source_provenance=f"evals/{f.name}",
                )
            )
    return out


def run_leakage_check(train_jsonl: Path | None = None) -> dict[str, Any]:
    """train.jsonl (+ yanındaki valid.jsonl varsa) ↔ validation/golden sızıntı denetimi.

    valid.jsonl de taranır: ``ensure_train_split`` her eğitimde yeniden böldüğü için bugün
    valid'de olan bir satır sonraki bölmede train'e düşebilir.
    """
    from app.config import get_settings
    from app.evals.llm30 import load_llm30
    from app.evals.profile.dataset_loader import load_split
    from app.evals.profile.leakage import check_leakage, load_train_jsonl
    from app.evals.profile.schema import Split

    path = train_jsonl or get_settings().jsonl_dir / "train.jsonl"
    train = load_train_jsonl(path) if path.exists() else []
    items = [
        *load_split(Split.VALIDATION, purpose="leakage_check"),
        *load_split(Split.GOLDEN_TEST, purpose="leakage_check"),
        *load_llm30(purpose="leakage_check"),
        *adapter_eval_items(),
    ]
    rep = check_leakage(items, train)
    valid_path = path.with_name("valid.jsonl")
    n_valid = 0
    if valid_path.exists() and valid_path != path:
        valid = load_train_jsonl(valid_path)
        n_valid = len(valid)
        vrep = check_leakage(items, valid)
        for h in vrep.hits:
            h.split = "valid"
        rep.hits.extend(vrep.hits)
    report = rep.to_dict()
    report["train_path"] = str(path)
    report["valid_path"] = str(valid_path) if n_valid else ""
    report["n_valid"] = n_valid
    return report


@mix_app.command("gate")
def gate_cmd(profile_id: str) -> None:
    """Regression gate'i göster (terfi ETMEZ)."""
    res = compute_gate(profile_id)
    _print_json(res)
    if not res.get("passed"):
        raise typer.Exit(1)


#: Aday ve production referans koşusunun karşılaştırılabilir olması için manifestlerde
#: BİREBİR eşleşmesi gereken alanlar (RunManifest). rag_version, indeks hash'ini de kapsar.
COMPARABILITY_KEYS: tuple[str, ...] = (
    "eval_dataset_hash",
    "rag_version",
    "base_model_hash",
    "eval_config_hash",
)


def manifest_mismatches(candidate: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    """İki koşu manifesti arasında karşılaştırılabilirliği bozan alanlar (boşsa uyumlu).

    Eksik / boş / ``unknown`` değer de uyumsuz sayılır (fail-closed): doğrulanamayan eşitlik
    eşitlik değildir.
    """
    out: list[str] = []
    for key in COMPARABILITY_KEYS:
        c, r = str(candidate.get(key) or ""), str(reference.get(key) or "")
        if not c or not r or "unknown" in (c, r):
            out.append(f"{key}: doğrulanamadı (aday={c or '—'}, referans={r or '—'})")
        elif c != r:
            out.append(f"{key}: farklı (aday={c[:16]}, referans={r[:16]})")
    return out


def compute_gate(profile_id: str) -> dict[str, Any]:
    """Aday = profilin son golden koşusundaki D (RAG+profil) metrikleri.

    Referans = production profilinin son golden D metrikleri; production yoksa adayın aynı
    koşusundaki B (base+RAG) baseline'ı. Production referansı ayrı bir koşudan geldiği için
    iki manifestin eval veri seti / RAG sürümü / base model / eval config hash'leri
    eşleşmelidir; eşleşmezse kapı KAPALI kalır (farklı koşulların metrikleri kıyaslanamaz).
    """
    from app.evals.profile.eval_registry import EvalRegistry
    from app.evals.profile.eval_runner import load_eval_config
    from app.evals.profile.regression_eval import evaluate_regression_gate
    from app.lora.profile_registry import ProfileRegistry

    reg = EvalRegistry()
    preg = ProfileRegistry()
    cand = preg.get(profile_id)
    if cand is None:
        return {"passed": False, "error": f"profil yok: {profile_id}"}
    run = reg.latest_for_profile(cand.profile_name, "golden_test") or reg.latest_for_profile(
        cand.profile_id, "golden_test"
    )
    if run is None:
        return {"passed": False, "error": "golden_test koşusu yok (önce --final eval)"}
    c_metrics = run.metrics.get(f"D_rag+{profile_id}")
    if not c_metrics:
        return {"passed": False, "error": f"koşuda D_rag+{profile_id} metrikleri yok"}
    prod = preg.production()
    ref_metrics: dict[str, Any] | None = None
    label = "baseline B_base_rag (production yok)"
    ref_run_id = run.run_id
    if prod is not None and prod.profile_id != profile_id:
        prun = reg.latest_for_profile(prod.profile_name, "golden_test")
        ref_metrics = (prun.metrics.get(f"D_rag+{prod.profile_id}") if prun else None) or None
        label = f"production {prod.profile_id}"
        if ref_metrics is None or prun is None:
            return {"passed": False, "error": f"production {prod.profile_id} golden koşusu yok"}
        mismatches = manifest_mismatches(run.manifest, prun.manifest)
        if mismatches:
            return {
                "passed": False,
                "reference_label": label,
                "error": (
                    "aday ve production koşuları karşılaştırılamaz (aynı eval seti, RAG, base "
                    "model ve eval config ile yeniden koşun)"
                ),
                "blockers": mismatches,
                "candidate_run_id": run.run_id,
                "reference_run_id": prun.run_id,
            }
        ref_run_id = prun.run_id
    else:
        ref_metrics = run.metrics.get("B_base_rag")
        if not ref_metrics:
            return {"passed": False, "error": "baseline B_base_rag metrikleri yok"}
    rules = load_eval_config().get("regression_gate") or {}
    res = evaluate_regression_gate(
        c_metrics,
        ref_metrics,
        rules,
        reference_label=label,
        candidate_overall=(c_metrics.get("overall") or {}).get("overall"),
        reference_overall=(ref_metrics.get("overall") or {}).get("overall"),
    )
    out = res.to_dict()
    out["candidate_run_id"] = run.run_id
    out["reference_run_id"] = ref_run_id
    return out


@mix_app.command("promote")
def promote_cmd(
    profile_id: str,
    to: str = typer.Option(..., "--to", help="evaluating|validated|rejected|production|deprecated"),
    approve: bool = typer.Option(False, "--approve", help="production için açık kullanıcı onayı"),
    note: str = typer.Option(""),
) -> None:
    """Profil durum geçişi. production: regression gate + --approve zorunlu."""
    from app.lora.profile_registry import ProfileRegistry, ProfileRegistryError, ProfileStatus

    try:
        target = ProfileStatus(to)
    except ValueError as exc:
        console.print(f"[red]geçersiz durum: {to}[/red]")
        raise typer.Exit(2) from exc
    gate_passed = False
    if target is ProfileStatus.PRODUCTION:
        gate = compute_gate(profile_id)
        _print_json(gate)
        gate_passed = bool(gate.get("passed"))
        if not gate_passed:
            console.print("[red]Regression gate geçilmedi — production'a terfi YOK.[/red]")
            raise typer.Exit(1)
    try:
        rec = ProfileRegistry().transition(
            profile_id, target, gate_passed=gate_passed, user_approved=approve, note=note
        )
    except ProfileRegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]{rec.profile_id}[/green] → {rec.status.value}")


if __name__ == "__main__":  # pragma: no cover
    mix_app()
