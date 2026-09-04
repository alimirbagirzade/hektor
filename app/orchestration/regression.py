"""regression.py — eğitim/onay öncesi gerileme (regression) bloklayıcı.

v5 adapteri eğitimi bitirdi ama DİSİPLİNDE GERİLEDİ ve yine de ACCEPT edildi
([[v5-adapter-regression]]). Çünkü hiçbir kapı "bir önceki geçen duruma göre kötüleşme"yi
ölçmüyordu. Bu modül, eğitim/onay aşamasından ÖNCE mevcut aday veri setinin v5-ilgili kalite
sinyallerini son kaydedilen GEÇEN baseline ile kıyaslar; gerileme varsa hattı bloklar.

İzlenen sinyaller (audit_dataset'ten — hepsi v5 mekanizmasıyla doğrudan ilgili):
  - guaranteed_profit_hits ↓ (Kural-1 zehir vaadi sayısı; artış = gerileme)
  - top_opening_share       ↓ (açılış ezberi oranı — v5'in TAM mekanizması)
  - leakage_prefix_share    ↓ (sızıntı öneki oranı)
  - ignores_costs_hits      ↓ (maliyet-körü cevap sayısı)
  - discipline_coverage     ↑ (disiplin havuzu kapsamı; düşüş = gerileme)
  - dataset_go              ↑ (kalite kapısı GO=1/NO-GO=0; GO→NO-GO = gerileme)

Verdict semantiği (delege StageStatus'a çevirir):
  - baseline YOK            → 'skip': ilk koşu, kıyaslanacak geçmiş yok (kusur değil). Hat
                              ilerler; baseline `orchestrate-regression --commit` ile kurulur.
  - gerileme YOK            → 'pass'.
  - gerileme VAR            → 'fail': insan araştırmadan eğitim ilerlememeli (BLOCKED — Kural 8).

Baseline GÜNCELLEMESİ daima EXPLICIT (CLI --commit) — oto-terfi yok (Kural 8 ruhu): kötü bir
set sessizce yeni baseline olup gerilemeyi normalleştirmesin. metrics_provider + baseline_store
ENJEKTE edilebilir → testler gerçek veri/dosya olmadan çalışır (offline, Kural).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MetricsProvider = Callable[[], dict[str, float]]

_BASELINE_REL = ("storage", "orchestration", "regression_baseline.json")


@dataclass(frozen=True)
class MetricSpec:
    """Bir metriğin gerileme yönü + toleransı."""

    name: str
    higher_is_better: bool
    # Mutlak tolerans: |değişim| bunu aşmazsa gerileme SAYILMAZ (gürültü payı).
    abs_tol: float = 0.0
    title: str = ""


# v5-ilgili sinyaller. abs_tol gürültü payı (oran metrikleri için küçük, sayımlar için 0).
METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "guaranteed_profit_hits",
        higher_is_better=False,
        abs_tol=0.0,
        title="garanti/kesinlik vaadi sayısı",
    ),
    MetricSpec(
        "top_opening_share", higher_is_better=False, abs_tol=0.05, title="açılış ezberi oranı"
    ),
    MetricSpec(
        "leakage_prefix_share", higher_is_better=False, abs_tol=0.02, title="sızıntı öneki oranı"
    ),
    MetricSpec(
        "ignores_costs_hits", higher_is_better=False, abs_tol=0.0, title="maliyet-körü cevap sayısı"
    ),
    MetricSpec(
        "discipline_coverage", higher_is_better=True, abs_tol=0.02, title="disiplin kapsamı"
    ),
    MetricSpec("dataset_go", higher_is_better=True, abs_tol=0.0, title="kalite kapısı GO"),
)


@dataclass
class RegressionFinding:
    """Tek bir metriğin baseline'a göre değerlendirmesi."""

    name: str
    status: str  # "regressed" | "improved" | "stable" | "new"
    baseline: float | None
    current: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "baseline": self.baseline,
            "current": self.current,
            "detail": self.detail,
        }


@dataclass
class RegressionResult:
    """Gerileme taramasının bütünsel sonucu."""

    verdict: str  # "pass" | "skip" | "fail"
    summary: str
    findings: list[RegressionFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
        }


def evaluate_regression(
    current: dict[str, float],
    baseline: dict[str, float] | None,
    specs: tuple[MetricSpec, ...] = METRIC_SPECS,
) -> RegressionResult:
    """SAF kıyas: current metrikleri baseline ile karşılaştır → RegressionResult.

    Baseline None/boş → 'skip' (kıyaslanacak geçmiş yok). Aksi halde herhangi bir metrik
    yön+tolerans dışında kötüleşmişse 'fail'; değilse 'pass'.
    """
    findings: list[RegressionFinding] = []
    if not baseline:
        for s in specs:
            if s.name not in current:
                continue
            cur = float(current[s.name])
            findings.append(
                RegressionFinding(
                    s.name, "new", None, cur, f"{s.title or s.name}: {cur:g} (baseline yok)."
                )
            )
        return RegressionResult("skip", "Baseline yok — kıyaslama atlandı (ilk koşu).", findings)

    regressed: list[str] = []
    for spec in specs:
        if spec.name not in current or spec.name not in baseline:
            continue
        cur = float(current[spec.name])
        base = float(baseline[spec.name])
        delta = cur - base
        # "Daha iyi yön"de değişim büyüklüğü; tolerans içindeyse stable.
        if spec.higher_is_better:
            worse = delta < -spec.abs_tol
            better = delta > spec.abs_tol
        else:
            worse = delta > spec.abs_tol
            better = delta < -spec.abs_tol
        label = spec.title or spec.name
        if worse:
            regressed.append(label)
            findings.append(
                RegressionFinding(
                    spec.name,
                    "regressed",
                    base,
                    cur,
                    f"{label}: {base:g} → {cur:g} (kötüleşti, tol={spec.abs_tol:g}).",
                )
            )
        elif better:
            findings.append(
                RegressionFinding(
                    spec.name, "improved", base, cur, f"{label}: {base:g} → {cur:g} (iyileşti)."
                )
            )
        else:
            findings.append(
                RegressionFinding(
                    spec.name, "stable", base, cur, f"{label}: {base:g} → {cur:g} (sabit)."
                )
            )

    if regressed:
        return RegressionResult(
            "fail",
            "Gerileme tespit edildi: " + "; ".join(regressed) + " (v5 dersi — eğitim bloklandı).",
            findings,
        )
    return RegressionResult("pass", "Gerileme yok — tüm sinyaller baseline'a eşit/üstün.", findings)


class BaselineStore:
    """Geçen baseline metriklerini JSON dosyasında saklar (enjekte edilebilir yol)."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            from app.config import get_settings

            path = get_settings().root.joinpath(*_BASELINE_REL)
        self.path = Path(path)

    def load(self) -> dict[str, float] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Regression baseline okunamadı (%s): %s", self.path, exc)
            return None
        metrics = data.get("metrics") if isinstance(data, dict) else None
        if not isinstance(metrics, dict):
            return None
        return {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}

    def save(self, metrics: dict[str, float], *, note: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metrics": {k: float(v) for k, v in metrics.items()}, "note": note}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )


def _default_metrics_provider() -> dict[str, float]:
    """SFT setinden v5-ilgili sinyalleri hesapla (data_gate ile aynı audit, salt-okuma)."""
    from app.orchestration.delegates import _count_sft_lines

    n, jsonl = _count_sft_lines()
    if n == 0 or not jsonl.exists():
        return {}
    from app.training.dataset_quality import audit_dataset
    from app.training.discipline_dataset import discipline_jsonl_lines

    lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    report = audit_dataset(lines, discipline_lines=discipline_jsonl_lines())
    return metrics_from_report(report.to_dict())


def metrics_from_report(report: dict[str, Any]) -> dict[str, float]:
    """audit_dataset raporu (dict) → izlenen metrik sözlüğü."""
    target = float(report.get("discipline_target", 0) or 0)
    present = float(report.get("discipline_present", 0) or 0)
    coverage = (present / target) if target > 0 else 1.0
    return {
        "guaranteed_profit_hits": float(report.get("guaranteed_profit_hits", 0) or 0),
        "top_opening_share": float(report.get("top_opening_share", 0.0) or 0.0),
        "leakage_prefix_share": float(report.get("leakage_prefix_share", 0.0) or 0.0),
        "ignores_costs_hits": float(report.get("ignores_costs_hits", 0) or 0),
        "discipline_coverage": coverage,
        "dataset_go": 1.0 if report.get("verdict") == "GO" else 0.0,
    }


class RegressionGuard:
    """Mevcut veri kalitesini baseline ile kıyaslayıp gerilemeyi bloklayan kapı."""

    def __init__(
        self,
        baseline_store: BaselineStore | None = None,
        metrics_provider: MetricsProvider | None = None,
    ) -> None:
        self._store = baseline_store or BaselineStore()
        self._provider = metrics_provider or _default_metrics_provider

    def current_metrics(self) -> dict[str, float]:
        try:
            return self._provider()
        except Exception as exc:  # provider patlasa bile koşu çökmesin → boş metrik
            log.debug("Regression: metrik sağlayıcı hata verdi: %s", exc)
            return {}

    def run(self) -> RegressionResult:
        current = self.current_metrics()
        if not current:
            return RegressionResult(
                "skip", "Mevcut metrik hesaplanamadı (veri yok) — gerileme taraması atlandı.", []
            )
        baseline = self._store.load()
        return evaluate_regression(current, baseline)

    def commit_baseline(self, *, note: str = "") -> dict[str, float]:
        """Mevcut metrikleri YENİ baseline olarak kaydet (EXPLICIT — oto-terfi yok)."""
        metrics = self.current_metrics()
        if metrics:
            self._store.save(metrics, note=note)
        return metrics
