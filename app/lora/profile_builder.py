"""Profil kurucu — bağımsız domain adapter'larını ağırlıklı TEK adapter'da birleştirir.

Adımlar:
1. Registry'den her domain için adapter bul (ağırlığı 0 olan domain atlanır).
2. Aynı base model (ad + revision + hash) doğrula.
3. Tokenizer uyumluluğu (tokenizer_hash) doğrula.
4. target_modules uyumluluğu doğrula.
5. Rank uyumluluğu (svd dışı yöntemler aynı r ister — PEFT kısıtı).
6. Profil ağırlıklarını ``configs/lora/mix_profiles.yaml``'dan oku.
7. PEFT ``add_weighted_adapter`` ile yeni adapter oluştur (yalnız ``--run``).
8. Merge yöntemi yapılandırılabilir: svd | ties | dare_ties | linear — hiçbiri "en iyi"
   sayılmaz; kimlik ``<profil>_<yöntem>`` (ör. balanced_v1_svd).

GÜVENLİK: Base model ağırlıkları DEĞİŞMEZ — ``merge_and_unload`` asla çağrılmaz; yalnız
yeni adapter ayrı dizine kaydedilir. Base model ``local_files_only=True`` ile yüklenir →
büyük model İNDİRİLMEZ. Varsayılan DRY-RUN (plan + uyumluluk raporu).

CLI::

    python -m app.lora.profile_builder --profile balanced_v1 --merge-method svd
    python -m app.lora.profile_builder --profile balanced_v1 --merge-method ties --run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.lora.domain_adapter_registry import DomainAdapterRecord, DomainAdapterRegistry
from app.lora.mix_common import (
    MERGE_METHODS,
    SAME_RANK_METHODS,
    MixConfigError,
    hash_obj,
    load_mix_config,
    profile_weights,
)
from app.lora.profile_registry import ProfileRecord, ProfileRegistry

log = logging.getLogger(__name__)

DEFAULT_MERGE_SEED = 42


@dataclass
class BuildPlan:
    profile_name: str
    profile_id: str
    merge_method: str
    merge_parameters: dict[str, Any]
    weights: dict[str, float]  # yalnız > 0 olanlar
    adapters: dict[str, dict[str, Any]]  # domain → adapter kaydı (dict)
    base_model: str = ""
    base_model_revision: str = ""
    base_model_hash: str = ""
    tokenizer_hash: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def profile_hash(self) -> str:
        return hash_obj(
            {
                "base_model_hash": self.base_model_hash,
                "tokenizer_hash": self.tokenizer_hash,
                "adapters": {
                    d: f"{a['adapter_id']}@{a['adapter_version']}"
                    for d, a in sorted(self.adapters.items())
                },
                "weights": self.weights,
                "merge_method": self.merge_method,
                "merge_parameters": self.merge_parameters,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        d["profile_hash"] = self.profile_hash
        return d


def resolve_merge_parameters(
    method: str, config: dict[str, Any], overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    if method not in MERGE_METHODS:
        raise MixConfigError(f"desteklenmeyen merge yöntemi: {method} (geçerli: {MERGE_METHODS})")
    params = dict(config.get("merge_methods", {}).get(method, {}))
    params.update(overrides or {})
    if method in ("ties", "dare_ties"):
        density = float(params.get("density", 0.5))
        if not 0.0 < density <= 1.0:
            raise MixConfigError(f"density (0, 1] aralığında olmalı: {density}")
        params["density"] = density
        params.setdefault("majority_sign_method", "total")
    if method == "dare_ties":
        params.setdefault("seed", DEFAULT_MERGE_SEED)  # DARE rastgele budar → seed şart
    return params


def plan_profile(
    profile_name: str,
    merge_method: str,
    *,
    adapter_registry: DomainAdapterRegistry | None = None,
    config: dict[str, Any] | None = None,
    adapter_overrides: dict[str, str] | None = None,
    merge_overrides: dict[str, Any] | None = None,
) -> BuildPlan:
    """Uyumluluk denetimli kurulum planı (hiçbir şey yüklemez/yazmaz)."""
    cfg = config or load_mix_config()
    reg = adapter_registry or DomainAdapterRegistry()
    weights = {d: w for d, w in profile_weights(profile_name, cfg).items() if w > 0}
    params = resolve_merge_parameters(merge_method, cfg, merge_overrides)
    plan = BuildPlan(
        profile_name=profile_name,
        profile_id=f"{profile_name}_{merge_method}",
        merge_method=merge_method,
        merge_parameters=params,
        weights=weights,
        adapters={},
    )

    chosen: dict[str, DomainAdapterRecord] = {}
    for domain in weights:
        override = (adapter_overrides or {}).get(domain)
        rec: DomainAdapterRecord | None
        if override:
            aid, _, ver = override.partition("@")
            rec = reg.get(aid, ver or None)
        else:
            rec = reg.latest_for_domain(domain)
        if rec is None:
            plan.errors.append(f"{domain}: registry'de kullanılabilir adapter yok")
            continue
        if rec.domain != domain:
            plan.errors.append(f"{domain}: seçilen adapter {rec.adapter_id} domain={rec.domain}")
        if rec.parent_adapter:
            plan.errors.append(
                f"{domain}: {rec.adapter_id} başka adapter üstünden eğitilmiş "
                f"({rec.parent_adapter})"
            )
        if not Path(rec.adapter_path).exists():
            plan.warnings.append(f"{domain}: adapter dizini diskte yok: {rec.adapter_path}")
        chosen[domain] = rec
    plan.adapters = {d: r.to_dict() for d, r in chosen.items()}
    if not chosen:
        return plan

    def _uniq(attr: str) -> set[Any]:
        return {getattr(r, attr) for r in chosen.values()}

    for attr, label in (
        ("base_model", "base model adı"),
        ("base_model_revision", "base model revision"),
        ("base_model_hash", "base model hash"),
    ):
        if len(_uniq(attr)) > 1:
            plan.errors.append(f"farklı {label}: {sorted(map(str, _uniq(attr)))}")
    if len(_uniq("tokenizer_hash")) > 1:
        plan.errors.append("tokenizer uyumsuz: adapter'lar farklı tokenizer_hash taşıyor")
    tm_sets = {tuple(sorted(r.target_modules)) for r in chosen.values()}
    if len(tm_sets) > 1:
        plan.errors.append(f"target_modules uyumsuz: {sorted(tm_sets)}")
    ranks = _uniq("r")
    if len(ranks) > 1:
        if merge_method in SAME_RANK_METHODS:
            plan.errors.append(
                f"{merge_method} tüm adapter'larda aynı r ister; bulunan: {sorted(ranks)} "
                "(svd yöntemi farklı rank'leri kabul eder)"
            )
        else:
            plan.warnings.append(f"farklı rank'ler: {sorted(ranks)} (svd ile birleştirilecek)")
    if merge_method == "svd":
        plan.merge_parameters.setdefault("svd_rank", max(ranks))
    first = next(iter(chosen.values()))
    plan.base_model = first.base_model
    plan.base_model_revision = first.base_model_revision
    plan.base_model_hash = first.base_model_hash
    plan.tokenizer_hash = first.tokenizer_hash
    return plan


class MergeBackend(Protocol):
    def merge(self, plan: BuildPlan, output_dir: Path) -> Path: ...


class PeftMergeBackend:
    """HF PEFT ile ağırlıklı adapter kurulumu. Base ağırlıkları DEĞİŞTİRMEZ."""

    def __init__(self, base_model_path: str | None = None) -> None:
        self.base_model_path = base_model_path

    def merge(self, plan: BuildPlan, output_dir: Path) -> Path:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - ortam bağımlı
            raise RuntimeError(
                "PEFT merge için torch/transformers/peft gerekli (uv sync --extra train-cpu)"
            ) from exc

        base_path = self.base_model_path or plan.base_model
        # local_files_only: büyük model İNDİRİLMEZ (açık izin olmadan indirme yasak).
        model = AutoModelForCausalLM.from_pretrained(
            base_path, local_files_only=True, torch_dtype="auto"
        )
        domains = list(plan.weights)
        first = domains[0]
        peft_model = PeftModel.from_pretrained(
            model, plan.adapters[first]["adapter_path"], adapter_name=first, is_trainable=False
        )
        for d in domains[1:]:
            peft_model.load_adapter(plan.adapters[d]["adapter_path"], adapter_name=d)

        params = dict(plan.merge_parameters)
        kwargs: dict[str, Any] = {}
        if plan.merge_method in ("ties", "dare_ties"):
            kwargs["density"] = params["density"]
            kwargs["majority_sign_method"] = params.get("majority_sign_method", "total")
        if plan.merge_method == "svd":
            kwargs["svd_rank"] = int(params["svd_rank"])
            if "svd_clamp" in params:
                kwargs["svd_clamp"] = params["svd_clamp"]
        if plan.merge_method == "dare_ties":
            torch.manual_seed(int(params.get("seed", DEFAULT_MERGE_SEED)))
        peft_model.add_weighted_adapter(
            adapters=domains,
            weights=[plan.weights[d] for d in domains],
            adapter_name=plan.profile_id,
            combination_type=plan.merge_method,
            **kwargs,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        # Yalnız yeni adapter kaydedilir; base model + kaynak adapter'lar dokunulmaz.
        peft_model.save_pretrained(str(output_dir), selected_adapters=[plan.profile_id])
        saved = output_dir / plan.profile_id
        return saved if saved.exists() else output_dir


def default_output_dir(profile_id: str) -> Path:
    from app.config import get_settings

    return get_settings().adapters_dir / "profiles" / profile_id


def build_profile(
    plan: BuildPlan,
    *,
    run: bool = False,
    backend: MergeBackend | None = None,
    output_dir: Path | None = None,
    profile_registry: ProfileRegistry | None = None,
    rag_version: str = "",
    profile_version: str = "1",
) -> dict[str, Any]:
    """Planı uygula. ``run=False`` (varsayılan) → yalnız plan döner, hiçbir şey yazmaz."""
    result: dict[str, Any] = {"run": run, "plan": plan.to_dict()}
    if not plan.ok:
        result["status"] = "blocked"
        return result
    if not run:
        result["status"] = "dry_run"
        return result
    out = output_dir or default_output_dir(plan.profile_id)
    merged_path = (backend or PeftMergeBackend()).merge(plan, out)
    reg = profile_registry or ProfileRegistry()
    record = ProfileRecord(
        profile_id=plan.profile_id,
        profile_name=plan.profile_name,
        profile_version=profile_version,
        base_model_version=(
            f"{plan.base_model}@{plan.base_model_revision or '-'}#{plan.base_model_hash[:12]}"
        ),
        rag_version=rag_version,
        adapter_versions={
            d: f"{a['adapter_id']}@{a['adapter_version']}" for d, a in plan.adapters.items()
        },
        adapter_weights=dict(plan.weights),
        merge_method=plan.merge_method,
        merge_parameters=dict(plan.merge_parameters),
        profile_hash=plan.profile_hash,
        merged_adapter_path=str(merged_path),
    )
    reg.upsert(record)
    result["status"] = "built"
    result["merged_adapter_path"] = str(merged_path)
    result["registry_status"] = record.status.value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA karışım profili kur (varsayılan dry-run)")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--merge-method", required=True, choices=MERGE_METHODS)
    parser.add_argument("--density", type=float, default=None)
    parser.add_argument("--run", action="store_true", help="Gerçekten kur (base model yüklenir)")
    parser.add_argument("--base-model-path", default=None, help="Yerel base model dizini")
    args = parser.parse_args(argv)

    overrides = {"density": args.density} if args.density is not None else None
    try:
        plan = plan_profile(args.profile, args.merge_method, merge_overrides=overrides)
    except MixConfigError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 2
    rag_version = ""
    if args.run:
        from app.memory.rag_version import current_rag_snapshot

        rag_version = current_rag_snapshot().rag_version
    result = build_profile(
        plan,
        run=args.run,
        backend=PeftMergeBackend(args.base_model_path),
        rag_version=rag_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("dry_run", "built") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
