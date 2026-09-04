"""LoRA kontrol düzlemi — denetim hattını koordine eden orkestratör.

Onaylı bilgi kartlarını SQLite'tan çeker, Gate 0-8'i sırayla çalıştırır,
bir `PipelineReport` üretir ve Markdown rapor yazar. Ağır eğitim başlatmaz;
yalnızca veri denetimi, dataset bölme ve raporlama yapar.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from app.lora.dataset_builder import build_dataset
from app.lora.dataset_splitter import DatasetSplit
from app.lora.gates import (
    GateResult,
    gate_0_source,
    gate_1_schema,
    gate_2_curriculum,
    gate_3_domain,
    gate_4_quality,
    gate_5_math,
    gate_6_philosophy,
    gate_7_safety,
    gate_8_split,
)
from app.memory.sqlite_store import SqliteStore

DEFAULT_DATA_DIR = Path("data/lora")
DEFAULT_DATASET_DIR = Path("data/lora_sft")
DEFAULT_REPORT_DIR = Path("reports/lora")
DEFAULT_REGISTRY_DIR = Path("registry/adapters")

DEFAULT_MATH_CORRECTNESS = 0.90
DEFAULT_HALLUCINATION_RISK = 0.05


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


@dataclass
class ControlPlaneConfig:
    """Kontrol düzlemi yapılandırması (yollar ve eşikler)."""

    data_dir: Path = DEFAULT_DATA_DIR
    dataset_dir: Path = DEFAULT_DATASET_DIR
    report_dir: Path = DEFAULT_REPORT_DIR
    registry_dir: Path = DEFAULT_REGISTRY_DIR
    math_correctness_threshold: float = DEFAULT_MATH_CORRECTNESS
    hallucination_risk_threshold: float = DEFAULT_HALLUCINATION_RISK


@dataclass
class PipelineReport:
    """Denetim hattının tam çıktısı."""

    stages: list[GateResult] = field(default_factory=list)
    total_input: int = 0
    total_approved: int = 0
    total_rejected: int = 0
    total_review_needed: int = 0
    dataset_split: DatasetSplit | None = None
    timestamp: str = field(default_factory=_utcnow)

    @property
    def passed(self) -> bool:
        """Tüm kapılar geçtiyse True."""
        return all(stage.passed for stage in self.stages)


class LoRAControlPlane:
    """Gate 0-8 denetim hattını koordine eden sınıf."""

    def __init__(
        self,
        store: SqliteStore | None = None,
        config: ControlPlaneConfig | None = None,
    ) -> None:
        self.store = store or SqliteStore()
        self.config = config or ControlPlaneConfig()

    # --- veri yükleme --------------------------------------------------
    def _load_approved_cards(self) -> list[dict]:
        """Onaylı (approved) kartları SQLite'tan çek."""
        return self.store.list_approved_cards()

    # --- denetim -------------------------------------------------------
    def run_audit(self) -> PipelineReport:
        """Gate 0-7'yi çalıştır ve denetim raporu üret (dataset bölme yok)."""
        cards = self._load_approved_cards()
        report = PipelineReport(total_input=len(cards))

        stages, clean_cards, examples = self._run_card_gates(cards)
        report.stages.extend(stages)

        report.total_review_needed = sum(s.review_count for s in stages)
        report.total_rejected = len(cards) - len(clean_cards)
        report.total_approved = len(clean_cards)
        # examples üretildiğini denetlemek için Gate 1'i de çalıştır
        report.stages.insert(1, gate_1_schema([self._example_to_dict(e) for e in examples]))
        return report

    def run_full(self) -> PipelineReport:
        """Tüm hattı (Gate 0-8) çalıştır; dataset bölme dahil.

        Bu sınıf hiçbir koşulda ağır eğitim başlatmaz, yalnızca denetler ve böler.
        """
        cards = self._load_approved_cards()
        report = PipelineReport(total_input=len(cards))

        stages, clean_cards, examples = self._run_card_gates(cards)
        report.stages.extend(stages)

        example_dicts = [self._example_to_dict(e) for e in examples]
        report.stages.insert(1, gate_1_schema(example_dicts))

        gate8, split = gate_8_split(example_dicts)
        report.stages.append(gate8)
        report.dataset_split = split

        report.total_review_needed = sum(s.review_count for s in report.stages)
        report.total_rejected = len(cards) - len(clean_cards)
        report.total_approved = len(clean_cards)
        return report

    def _run_card_gates(self, cards: list[dict]) -> tuple[list[GateResult], list[dict], list]:
        """Kart bazlı kapıları (0,2-7) çalıştır, temiz kart ve örnekleri döndür."""
        # title/içerik boş olan kartları gate'lere sokmadan ele — bunlar DB'de
        # hatalı onaylanmış (içeriksiz) kartlardır ve pipeline'ı bloklamalı değil.
        from app.lora.gates import _card_text

        nonempty = [c for c in cards if _card_text(c)]

        # Gate 0 kaynak-varlık denetimi: papers tablosundaki gerçek paper_id seti.
        # Bir kartın paper_id'si bu sette yoksa 'orphan'dır (kaynak yok → kural 7).
        valid_paper_ids = self.store.list_paper_ids()
        stages: list[GateResult] = [gate_0_source(nonempty, valid_paper_ids=valid_paper_ids)]
        stages.append(gate_2_curriculum(nonempty))
        stages.append(gate_3_domain(nonempty))

        gate4, clean_cards = gate_4_quality(nonempty)
        stages.append(gate4)
        stages.append(gate_5_math(clean_cards))
        stages.append(gate_6_philosophy(clean_cards))
        # Gate 7 (safety) BLOCKER → Gate 4 elemesinden BAĞIMSIZ, içerikli kartların
        # TAMAMINI tara. Aksi halde kısa/duplicate diye Gate 4'te elenen ama sır/PII
        # taşıyan bir kart (örn. implementation_notes'ta) güvenlik taramasını atlardı.
        stages.append(gate_7_safety(nonempty))

        examples = build_dataset(clean_cards)
        return stages, clean_cards, examples

    @staticmethod
    def _example_to_dict(example: object) -> dict:
        """LoRAExample'ı gate_1/gate_8 için dict'e dönüştür."""
        messages = getattr(example, "messages", [])
        metadata = getattr(example, "metadata", {})
        return {"messages": messages, "metadata": metadata}

    # --- raporlama -----------------------------------------------------
    def generate_report(self, report: PipelineReport, output_path: Path | None = None) -> str:
        """Pipeline raporundan Markdown üret; istenirse dosyaya yaz."""
        lines: list[str] = [
            "# LoRA Denetim Raporu",
            "",
            f"- Zaman: {report.timestamp}",
            f"- Girdi kart sayısı: {report.total_input}",
            f"- Onaylanan: {report.total_approved}",
            f"- Reddedilen: {report.total_rejected}",
            f"- İnceleme gereken: {report.total_review_needed}",
            f"- Genel sonuç: {'GEÇTİ' if report.passed else 'BAŞARISIZ'}",
            "",
            "## Kapılar",
            "",
            "| Gate | Ad | Sonuç | Red | İnceleme |",
            "|------|----|-------|-----|----------|",
        ]
        for stage in report.stages:
            status = "PASS" if stage.passed else "FAIL"
            lines.append(
                f"| {stage.gate_id} | {stage.name} | {status} | "
                f"{stage.rejected_count} | {stage.review_count} |"
            )

        for stage in report.stages:
            if stage.details:
                lines.append("")
                lines.append(f"### Gate {stage.gate_id} — {stage.name} ayrıntıları")
                lines.extend(f"- {detail}" for detail in stage.details)

        if report.dataset_split is not None:
            split = report.dataset_split
            lines.extend(
                [
                    "",
                    "## Dataset Bölme",
                    "",
                    f"- train: {len(split.train)}",
                    f"- valid: {len(split.valid)}",
                    f"- test: {len(split.test)}",
                ]
            )

        markdown = "\n".join(lines) + "\n"
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
        return markdown
