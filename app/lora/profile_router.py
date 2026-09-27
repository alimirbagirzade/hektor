"""Profil router (v1) — sorguyu önceden VALIDATED/PRODUCTION olmuş bir profile yönlendirir.

v1 bilinçli olarak basittir:
- Sürekli (continuous) ağırlık ÜRETMEZ; yalnız registry'deki routable profiller arasından seçer.
- Domain tespiti deterministik anahtar-kelime regex'i (TR + EN); LLM yok.
- Güven eşiğin altındaysa → fallback (balanced) profil.
- Seçilecek profil validated değilse → fallback; o da değilse → profil YOK (base model).
- Her karar loglanır: query_id, detected_domains, selected_profile, router_confidence,
  fallback_profile, created_at.

CLI::

    python -m app.lora.profile_router --query "Sharpe oranının standart hatası nedir?"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.lora.mix_common import append_jsonl, load_mix_config, registry_dir
from app.lora.profile_registry import ProfileRecord, ProfileRegistry, ProfileStatus

# Router domain'leri: istatistik/matematik/trading/kod + "general" (dengeli profil).
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "math": (
        r"türev",
        r"integral",
        r"denklem",
        r"matris",
        r"logaritma",
        r"limit",
        r"hesapla",
        r"kanıtla",
        r"ispat",
        r"derivative",
        r"equation",
        r"matrix",
        r"prove",
        r"\bsolve\b",
        r"\d+\s*[\^*/+-]\s*\d+",
    ),
    "statistics": (
        r"istatisti",
        r"olasılık",
        r"dağılım",
        r"varyans",
        r"standart sapma",
        r"standart hata",
        r"korelasyon",
        r"regresyon",
        r"hipotez test",
        r"p[- ]?değer",
        r"güven aralığı",
        r"örneklem",
        r"monte carlo",
        r"bootstrap",
        r"probabilit",
        r"distribution",
        r"variance",
        r"correlation",
        r"regression",
        r"p-value",
        r"confidence interval",
        r"sharpe",
    ),
    "trading": (
        r"trading",
        r"strateji",
        r"strategy",
        r"piyasa",
        r"market",
        r"hisse",
        r"\bstock",
        r"volatilite",
        r"volatility",
        r"drawdown",
        r"momentum",
        r"likidite",
        r"liquidity",
        r"komisyon",
        r"slippage",
        r"portföy",
        r"portfolio",
        r"\brsi\b",
        r"\bmacd\b",
        r"getiri",
        r"\breturns?\b",
    ),
    "coding": (
        r"python",
        r"\bkod",
        r"\bcode\b",
        r"script",
        r"fonksiyon",
        r"function",
        r"pandas",
        r"numpy",
        r"backtest",
        r"\bdef\b",
        r"hata ayıkla",
        r"debug",
        r"implement",
    ),
}
_COMPILED = {d: [re.compile(p, re.IGNORECASE) for p in ps] for d, ps in _KEYWORDS.items()}


@dataclass
class RouterDecision:
    query_id: str
    detected_domains: dict[str, float]
    primary_domain: str
    selected_profile: str | None  # None → profil yok, base model (+RAG)
    router_confidence: float
    fallback_profile: str | None
    used_fallback: bool
    reason: str
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_domains(query: str) -> dict[str, float]:
    """Domain başına eşleşen anahtar-kelime sayısı → normalize pay (toplam 1 ya da boş)."""
    hits = {d: sum(1 for rx in rxs if rx.search(query)) for d, rxs in _COMPILED.items()}
    total = sum(hits.values())
    if total == 0:
        return {}
    return {d: round(h / total, 4) for d, h in hits.items() if h > 0}


def _confidence(query: str, shares: dict[str, float]) -> tuple[str, float]:
    if not shares:
        return "general", 0.0
    # Eşitlikte alfabetik ilk kazanır (deterministik): coding < math < statistics < trading
    # → "python backtest" gibi kod+trading eşitliği coding_backtest profiline gider.
    primary = max(sorted(shares), key=lambda d: shares[d])
    total_hits = sum(1 for rxs in _COMPILED.values() for rx in rxs if rx.search(query))
    # Tek eşleşme zayıf kanıttır → güveni kıs.
    evidence = min(1.0, total_hits / 2)
    return primary, round(shares[primary] * evidence, 4)


class ProfileRouter:
    def __init__(
        self,
        registry: ProfileRegistry | None = None,
        config: dict[str, Any] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.registry = registry or ProfileRegistry()
        self.config = config or load_mix_config()
        self.log_path = log_path or registry_dir() / "router_decisions.jsonl"
        rcfg = self.config.get("router") or {}
        self.threshold = float(rcfg.get("confidence_threshold", 0.5))
        self.fallback_name: str | None = rcfg.get("fallback_profile")
        self.domain_profiles: dict[str, str] = dict(rcfg.get("domain_profiles") or {})

    def _routable_by_name(self) -> dict[str, ProfileRecord]:
        """profile_name → en uygun routable kayıt (production > son validated)."""
        best: dict[str, ProfileRecord] = {}
        for rec in self.registry.routable():
            cur = best.get(rec.profile_name)
            if (
                cur is None
                or rec.status is ProfileStatus.PRODUCTION
                or (cur.status is not ProfileStatus.PRODUCTION and rec.updated_at >= cur.updated_at)
            ):
                best[rec.profile_name] = rec
        return best

    def route(self, query: str, *, query_id: str | None = None, log: bool = True) -> RouterDecision:
        qid = query_id or "q_" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        shares = detect_domains(query)
        primary, conf = _confidence(query, shares)
        routable = self._routable_by_name()
        fallback_rec = routable.get(self.fallback_name or "")
        fallback_id = fallback_rec.profile_id if fallback_rec else None

        wanted = self.domain_profiles.get(primary) or self.fallback_name
        selected: str | None
        used_fallback = False
        if conf < self.threshold:
            selected, used_fallback = fallback_id, True
            reason = f"güven {conf:.2f} < eşik {self.threshold:.2f} → fallback"
        elif wanted in routable:
            selected = routable[wanted].profile_id
            reason = f"{primary} → {wanted}"
        else:
            selected, used_fallback = fallback_id, True
            reason = f"{wanted} validated değil → fallback"
        if selected is None:
            reason += " (fallback de validated değil → profil yok, base model)"

        decision = RouterDecision(
            query_id=qid,
            detected_domains=shares,
            primary_domain=primary,
            selected_profile=selected,
            router_confidence=conf,
            fallback_profile=fallback_id,
            used_fallback=used_fallback,
            reason=reason,
        )
        if log:
            append_jsonl(self.log_path, decision.to_dict())
        return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sorgu → validated profil seçimi (v1)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)
    decision = ProfileRouter().route(args.query, log=not args.no_log)
    print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
