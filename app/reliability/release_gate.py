"""Release gate — checks minimum quality thresholds.

MVP thresholds: Recall@10 >= 0.70, Citation accuracy >= 0.85, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# MVP minimum eşikleri (üretim hedeflerinden daha düşük)
MINIMUM_THRESHOLDS: dict[str, float] = {
    "recall_at_10": 0.70,
    "citation_accuracy": 0.85,
    "grounding_score": 0.80,
    "abstention_correct": 0.90,
}


@dataclass
class GateResult:
    """Yayın kapısı değerlendirme sonucu."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class ReleaseGate:
    """Tüm minimum eşikler karşılanmışsa yayına izin veren kapı.

    Kullanım:
        gate = ReleaseGate()
        result = gate.check({"recall_at_10": 0.75, "citation_accuracy": 0.88, ...})
        if result.passed:
            deploy()
    """

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self._thresholds = thresholds or MINIMUM_THRESHOLDS.copy()

    def check(self, metrics: dict) -> GateResult:
        """Metriklerin eşikleri karşılayıp karşılamadığını kontrol et.

        Args:
            metrics: {metrik_adı: değer} sözlüğü.

        Returns:
            GateResult (passed=True yalnızca tüm eşikler karşılandığında).
        """
        failures: list[str] = []

        for metric, threshold in self._thresholds.items():
            if metric not in metrics:
                failures.append(f"{metric}: EKSİK (eşik: {threshold:.2f})")
                continue
            value = metrics[metric]
            # Geçersiz/hesaplanamayan metrik (None, NaN, inf, sayısal-olmayan) → FAIL-CLOSED.
            # Eskiden NaN SESSİZCE GEÇERDİ (nan<eşik=False → fail-open: bozuk metrik yayını
            # geçirir, Kural 2) ve None TypeError ile ÇÖKERDİ. Yayın kapısı asla fail-open
            # olmamalı: hesaplanamayan bir metrik eşiği KARŞILAMADI sayılır.
            valid_number = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
            if not valid_number:
                failures.append(
                    f"{metric}: geçersiz/hesaplanamayan değer ({value!r}) — fail-closed "
                    f"(eşik: {threshold:.2f})"
                )
                continue
            if value < threshold:
                failures.append(f"{metric}: {value:.3f} < {threshold:.2f} (eşik karşılanmadı)")

        return GateResult(
            passed=len(failures) == 0,
            failures=failures,
            metrics=dict(metrics),
        )
