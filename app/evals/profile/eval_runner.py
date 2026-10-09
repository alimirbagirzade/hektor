"""Profil eval çalıştırıcısı — A/B/C/D (+ tekil adapter) sistemlerini AYNI koşullarda ölçer.

Aynı tutulanlar: soru, system prompt, prompt şablonu, üretim parametreleri (temperature,
max_tokens, seed), RAG indeksi + retrieval ayarları (retrieval kalem başına BİR KEZ yapılır ve
B/D sistemleri aynı bağlamı görür), eval veri seti. RAG yapılandırması koşu başında ve sonunda
karşılaştırılır; değiştiyse koşu GEÇERSİZ (RagConfigDrift).

Split kuralı: varsayılan ``validation`` (profil seçimi). ``golden_test`` yalnız ``--final``
ile ve profil ``validated``/``production`` iken (seçim bittikten sonra) koşulabilir.

CLI::

    python -m app.evals.profile.eval_runner --profile balanced_v1 --dry-run
    python -m app.evals.profile.eval_runner --profile balanced_v1
    python -m app.evals.profile.eval_runner --profile balanced_v1 --split golden_test --final
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.evals.profile.abstention_eval import evaluate_abstention
from app.evals.profile.answer_eval import evaluate_answer
from app.evals.profile.consistency_eval import consistency_score
from app.evals.profile.dataset_loader import dataset_version, load_split, split_path
from app.evals.profile.eval_registry import EvalRegistry
from app.evals.profile.failure_diagnosis import cross_system_diagnosis, diagnose_item
from app.evals.profile.generators import (
    GenerationConfig,
    OllamaGenerator,
    SystemSpec,
)
from app.evals.profile.grounding_eval import evaluate_grounding
from app.evals.profile.manifest import RunManifest, new_run_id
from app.evals.profile.retrieval_eval import evaluate_retrieval
from app.evals.profile.schema import EvalItem, ItemScore, Split
from app.evals.profile.score_aggregator import aggregate, overall_score
from app.lora.mix_common import git_commit, hash_file, hash_obj, repo_root
from app.memory.rag_version import RagSnapshot, RetrievalTrace, assert_same_rag
from app.procutil import NO_WINDOW


def default_eval_config_path() -> Path:
    return repo_root() / "configs" / "eval" / "profile_eval.yaml"


def load_eval_config(path: Path | None = None) -> dict[str, Any]:
    p = path or default_eval_config_path()
    return dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})


def build_prompt(item: EvalItem, chunks: list[Any] | None) -> str:
    """Tüm sistemlerde aynı şablon; RAG'lı sistemlerde bağlam bloğu eklenir."""
    parts: list[str] = []
    if chunks is not None:
        if chunks:
            parts.append("BAĞLAM (yalnız bunlara atıf yap):")
            for c in chunks:
                parts.append(f"{c.citation} {str(c.text).strip()[:1200]}")
        else:
            parts.append("BAĞLAM: (retrieval boş — kaynak bulunamadı)")
        parts.append("")
    parts.append(f"SORU: {item.question}")
    return "\n".join(parts)


def _resources() -> dict[str, Any]:
    out: dict[str, Any] = {"ram_mb": None, "vram_mb": None}
    try:
        import psutil

        out["ram_mb"] = round(psutil.virtual_memory().used / 2**20)
    except ImportError:  # pragma: no cover
        pass
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            r = subprocess.run(
                [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                creationflags=NO_WINDOW,
            )
            vals = [float(x) for x in r.stdout.split() if x.strip()]
            out["vram_mb"] = sum(vals) if vals else None
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return out


@dataclass
class EvalRunOutput:
    manifest: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    overall: dict[str, dict[str, Any]]
    items: list[ItemScore]
    not_run: dict[str, str] = field(default_factory=dict)
    run_dir: str | None = None


def run_eval(
    items: list[EvalItem],
    systems: list[SystemSpec],
    *,
    eval_config: dict[str, Any],
    rag_snapshot: RagSnapshot,
    retriever: Any | None = None,
    rag_snapshot_fn: Callable[[], RagSnapshot] | None = None,
    dataset_label: str = "",
    dataset_hash: str = "",
    model_meta: dict[str, Any] | None = None,
    registry: EvalRegistry | None = None,
    resource_fn: Callable[[], dict[str, Any]] = _resources,
) -> EvalRunOutput:
    gen_cfg = GenerationConfig(**(eval_config.get("generation") or {}))
    system_prompt = str(eval_config.get("system_prompt") or "")
    allow_code = bool(eval_config.get("allow_code_execution", False))
    repeats = int(eval_config.get("consistency_repeats") or 0)
    meta = dict(model_meta or {})

    not_run: dict[str, str] = {}
    runnable: list[SystemSpec] = []
    for s in systems:
        if s.generator is None:
            not_run[s.name] = s.unavailable_reason or "üretici yok"
        elif s.use_rag and retriever is None:
            not_run[s.name] = "retriever yok (RAG sistemi koşulamaz)"
        else:
            runnable.append(s)

    # Retrieval: kalem başına BİR KEZ — B ve D aynı bağlamı görür.
    contexts: dict[str, list[Any]] = {}
    traces: dict[str, RetrievalTrace] = {}
    if retriever is not None and any(s.use_rag for s in runnable):
        for it in items:
            try:
                chunks = list(retriever.retrieve(it.question, top_k=rag_snapshot.top_k))
            except Exception:  # retrieval hatası = boş bağlam (teşhiste context_missing)
                chunks = []
            contexts[it.id] = chunks
            traces[it.id] = RetrievalTrace.from_chunks(rag_snapshot, chunks)

    scores: list[ItemScore] = []
    by_item: dict[str, dict[str, ItemScore]] = {}
    for spec in runnable:
        assert spec.generator is not None
        for it in items:
            ctx = contexts.get(it.id) if spec.use_rag else None
            prompt = build_prompt(it, ctx)
            gen = spec.generator.generate(prompt, system=system_prompt, config=gen_cfg)
            ev = evaluate_answer(it, gen.text, allow_code_execution=allow_code)
            trace = traces.get(it.id) if spec.use_rag else None
            ret = (
                evaluate_retrieval(it, trace.retrieved_chunk_ids, trace.retrieved_document_ids)
                if trace is not None
                else {}
            )
            gr = evaluate_grounding(it, gen.text, trace.retrieved_chunk_ids if trace else None)
            ab = evaluate_abstention(it, gen.text, fabricated_citations=int(gr["fabricated"]))
            checks = dict(ev["checks"])
            checks["abstention_correct"] = ab["abstention_correct"]
            checks["citations"] = gr["citations"]
            if repeats > 0:
                answers = [gen.text]
                for i in range(1, repeats + 1):
                    cfg_i = GenerationConfig(
                        temperature=gen_cfg.temperature,
                        max_tokens=gen_cfg.max_tokens,
                        seed=gen_cfg.seed + i,
                        top_p=gen_cfg.top_p,
                    )
                    answers.append(
                        spec.generator.generate(prompt, system=system_prompt, config=cfg_i).text
                    )
                checks["consistency"] = consistency_score(answers)
            sc = ItemScore(
                item_id=it.id,
                domain=it.domain,
                system=spec.name,
                correct=ev["correct"],
                score=ev["score"],
                checks=checks,
                abstained=bool(ab["abstained"]),
                hallucinated=bool(ab["hallucinated"]),
                grounding=gr["grounding"],
                citation_accuracy=gr["citation_accuracy"],
                retrieval_recall=ret.get("recall_at_k"),
                retrieval_precision=ret.get("precision"),
                context_relevance=ret.get("context_relevance"),
                answer=gen.text,
                latency_s=round(gen.latency_s, 4),
                tokens_per_second=gen.tokens_per_second,
                retrieval=trace.to_dict() if trace else None,
            )
            sc.failure_categories = diagnose_item(
                it, sc, uses_rag=spec.use_rag, retrieval=sc.retrieval
            )
            scores.append(sc)
            by_item.setdefault(it.id, {})[spec.name] = sc

    # Sistemler-arası teşhis (LoRA / merge / base kaynaklı hatalar).
    base_of = {s.name: s.base_counterpart for s in runnable if s.base_counterpart}
    indiv = {s.name: list(s.individual_counterparts) for s in runnable if s.individual_counterparts}
    for item_id, per_sys in by_item.items():
        extra = cross_system_diagnosis(item_id, per_sys, base_of=base_of, individual_of=indiv)
        for sys_name, cats in extra.items():
            sc = per_sys[sys_name]
            sc.failure_categories = list(dict.fromkeys([*sc.failure_categories, *cats]))

    if rag_snapshot_fn is not None:
        assert_same_rag(rag_snapshot, rag_snapshot_fn())

    res = resource_fn()
    weights = eval_config.get("eval_weights") or {}
    metrics: dict[str, dict[str, Any]] = {}
    overall: dict[str, dict[str, Any]] = {}
    for spec in runnable:
        m = aggregate([s for s in scores if s.system == spec.name], res)
        metrics[spec.name] = m
        overall[spec.name] = overall_score(m, weights)

    material = {
        "dataset": dataset_label,
        "systems": [s.name for s in systems],
        "config": hash_obj(eval_config),
    }
    manifest = RunManifest(
        run_id=new_run_id(material),
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        git_commit=meta.get("git_commit") or git_commit(),
        base_model=str(meta.get("base_model", "")),
        base_model_hash=str(meta.get("base_model_hash", "unknown")),
        tokenizer_hash=str(meta.get("tokenizer_hash", "unknown")),
        quantization=str(meta.get("quantization", "unknown")),
        rag_version=rag_snapshot.rag_version,
        rag_index_hash=rag_snapshot.index_hash,
        embedding_model=rag_snapshot.embedding_model,
        reranker=f"{rag_snapshot.reranker}:{rag_snapshot.reranker_version}",
        adapter_versions=dict(meta.get("adapter_versions") or {}),
        profile=meta.get("profile"),
        merge_method=meta.get("merge_method"),
        weights=dict(meta.get("weights") or {}),
        eval_dataset=dataset_label,
        eval_dataset_hash=dataset_hash,
        eval_config_hash=hash_obj(eval_config),
        generation_config=gen_cfg.to_dict(),
        seed=gen_cfg.seed,
        systems=[s.name for s in systems],
    ).to_dict()
    manifest["system_labels"] = {s.name: s.label for s in systems}
    manifest["not_run"] = not_run
    manifest["rag_snapshot"] = rag_snapshot.to_dict()

    out = EvalRunOutput(manifest, metrics, overall, scores, not_run)
    if registry is not None:
        out.run_dir = str(
            registry.save(
                manifest,
                {k: {**v, "overall": overall[k]} for k, v in metrics.items()},
                scores,
                summary={"not_run": not_run},
            )
        )
    return out


# ---------------------------------------------------------------- CLI yardımcıları


def build_systems_for_profile(
    profile: str,
    *,
    include: set[str],
    matrix: bool,
) -> tuple[list[SystemSpec], dict[str, Any], list[Any]]:
    """Profil adı/kimliği için A/B/C/D (+ matris: tekil adapter) sistemlerini kur."""
    from app.config import get_settings
    from app.lora.domain_adapter_registry import DomainAdapterRegistry
    from app.lora.profile_registry import ProfileRegistry

    settings = get_settings()
    records = [r for r in ProfileRegistry().records() if profile in (r.profile_name, r.profile_id)]
    base_gen = OllamaGenerator(settings.llm_model)
    systems: list[SystemSpec] = []
    if "A" in include:
        systems.append(SystemSpec("A_base", "Base", base_gen, use_rag=False))
    if "B" in include:
        systems.append(SystemSpec("B_base_rag", "Base + RAG", base_gen, use_rag=True))

    indiv_names: dict[bool, list[str]] = {False: [], True: []}
    if matrix:
        for dom_rec in DomainAdapterRegistry().records():
            gen = OllamaGenerator(dom_rec.serving_model) if dom_rec.serving_model else None
            reason = "" if gen else "serving_model tanımlı değil (GGUF→Ollama yapılmadı)"
            tag = f"{dom_rec.domain}:{dom_rec.adapter_id}@{dom_rec.adapter_version}"
            for rag in (False, True):
                name = f"{'D' if rag else 'C'}_adapter_{dom_rec.domain}_{dom_rec.adapter_version}"
                systems.append(
                    SystemSpec(
                        name,
                        f"{'Base + RAG + ' if rag else 'Base + '}{dom_rec.domain} LoRA",
                        gen,
                        use_rag=rag,
                        lora=tag,
                        lora_kind="adapter",
                        unavailable_reason=reason,
                        base_counterpart="B_base_rag" if rag else "A_base",
                    )
                )
                indiv_names[rag].append(name)

    for rec in records:
        gen = OllamaGenerator(rec.serving_model) if rec.serving_model else None
        reason = "" if gen else "serving_model tanımlı değil (GGUF→Ollama yapılmadı)"
        for rag, letter in ((False, "C"), (True, "D")):
            if letter not in include:
                continue
            systems.append(
                SystemSpec(
                    f"{letter}_{'rag+' if rag else ''}{rec.profile_id}",
                    f"{'Base + RAG + ' if rag else 'Base + '}{rec.profile_id}",
                    gen,
                    use_rag=rag,
                    lora=rec.profile_id,
                    lora_kind="profile",
                    unavailable_reason=reason,
                    base_counterpart="B_base_rag" if rag else "A_base",
                    individual_counterparts=tuple(indiv_names[rag]),
                )
            )
    meta: dict[str, Any] = {"base_model": settings.llm_model, "profile": profile}
    if records:
        r0 = records[0]
        meta.update(
            {
                "adapter_versions": r0.adapter_versions,
                "weights": r0.adapter_weights,
                "merge_method": ",".join(sorted({r.merge_method for r in records})),
                "base_model_hash": r0.base_model_version.rpartition("#")[2] or "unknown",
            }
        )
    return systems, meta, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profil eval (A/B/C/D karşılaştırması)")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--split", default="validation", choices=[s.value for s in Split])
    parser.add_argument("--final", action="store_true", help="golden_test için zorunlu onay")
    parser.add_argument("--systems", default="A,B,C,D")
    parser.add_argument("--matrix", action="store_true", help="tekil adapter satırlarını ekle")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="LLM çağırmadan planı göster")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    from app.lora.profile_registry import ProfileStatus
    from app.memory.rag_version import current_rag_snapshot

    split = Split(args.split)
    cfg = load_eval_config(args.config)
    systems, meta, records = build_systems_for_profile(
        args.profile, include=set(args.systems.split(",")), matrix=args.matrix
    )
    if split is Split.GOLDEN_TEST:
        if not args.final:
            print(
                "HATA: golden_test yalnız --final ile (profil seçimi bittikten sonra).",
                file=sys.stderr,
            )
            return 2
        if not records or not all(
            r.status in (ProfileStatus.VALIDATED, ProfileStatus.PRODUCTION) for r in records
        ):
            print(
                "HATA: golden_test için profil validated/production olmalı "
                "(önce validation split ile seç, sonra `hektor mix promote --to validated`).",
                file=sys.stderr,
            )
            return 2
    purpose = "final_comparison" if split is Split.GOLDEN_TEST else "selection"
    items = load_split(split, purpose=purpose)
    if args.limit > 0:
        items = items[: args.limit]
    top_k = int((cfg.get("retrieval") or {}).get("top_k") or 6)
    snapshot = current_rag_snapshot(top_k=top_k)
    plan = {
        "profile": args.profile,
        "split": split.value,
        "dataset_version": dataset_version(split),
        "n_items": len(items),
        "rag_version": snapshot.rag_version,
        "systems": {
            s.name: ("hazır" if s.generator else f"koşulamaz: {s.unavailable_reason}")
            for s in systems
        },
        "profiles_in_registry": [r.profile_id for r in records],
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, ensure_ascii=False, indent=2))
        return 0

    retriever = None
    if any(s.use_rag and s.generator for s in systems):
        from app.memory.reranking_retriever import RerankingRetriever

        retriever = RerankingRetriever()
    out = run_eval(
        items,
        systems,
        eval_config=cfg,
        rag_snapshot=snapshot,
        retriever=retriever,
        rag_snapshot_fn=lambda: current_rag_snapshot(top_k=top_k),
        dataset_label=dataset_version(split),
        dataset_hash=hash_file(split_path(split)),
        model_meta=meta,
        registry=EvalRegistry(),
    )
    for rec in records:  # profil kaydına D (RAG+profil) sonuçlarını işle
        d_name = f"D_rag+{rec.profile_id}"
        if d_name in out.metrics:
            from app.lora.profile_registry import ProfileRegistry

            m = out.metrics[d_name]
            ProfileRegistry().record_eval(
                rec.profile_id,
                run_id=out.manifest["run_id"],
                eval_dataset_version=out.manifest["eval_dataset"],
                eval_score=out.overall[d_name]["overall"],
                domain_scores={
                    k: v for k, v in m.items() if k.endswith("_accuracy") and v is not None
                },
                grounding_score=m.get("grounding"),
                hallucination_score=m.get("hallucination_rate"),
            )
    print(
        json.dumps(
            {"run_id": out.manifest["run_id"], "run_dir": out.run_dir, "not_run": out.not_run},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
