"""LoRA Training Control Plane — kontrollü eğitim yaşam döngüsü.

Bu paket, RAG'da onaylanmış bilgi kartlarını LoRA SFT veri setine
dönüştürmek için disiplinli bir pipeline sağlar. Ağır eğitim başlatmaz;
yalnızca veri denetimi, kalite kapıları (Gate 0-8), dataset üretimi ve
adapter kayıt yönetimi yapar.

RAG = bilgi deposu (kaynak, belge). LoRA = davranış (muhakeme, format).
"""

from __future__ import annotations

from app.lora.adapter_registry import (
    AdapterRecord,
    AdapterRegistry,
    AdapterStatus,
)
from app.lora.control_plane import (
    ControlPlaneConfig,
    LoRAControlPlane,
    PipelineReport,
)
from app.lora.curriculum import (
    CurriculumLevel,
    classify_curriculum,
    is_curriculum_valid,
)
from app.lora.dataset_builder import (
    SYSTEM_PROMPT,
    LoRAExample,
    build_dataset,
    card_to_lora_example,
    export_jsonl,
)
from app.lora.dataset_splitter import (
    DatasetSplit,
    check_leakage,
    split_dataset,
)
from app.lora.domain_classifier import Domain, classify_domains
from app.lora.gates import GateResult
from app.lora.math_verifier import MathVerifyResult, verify_math_content
from app.lora.quality_filter import QualityFilter, QualityResult, check_quality
from app.lora.safety_scanner import SafetyResult, scan_for_secrets

__all__ = [
    "SYSTEM_PROMPT",
    "AdapterRecord",
    "AdapterRegistry",
    "AdapterStatus",
    "ControlPlaneConfig",
    "CurriculumLevel",
    "DatasetSplit",
    "Domain",
    "GateResult",
    "LoRAControlPlane",
    "LoRAExample",
    "MathVerifyResult",
    "PipelineReport",
    "QualityFilter",
    "QualityResult",
    "SafetyResult",
    "build_dataset",
    "card_to_lora_example",
    "check_leakage",
    "check_quality",
    "classify_curriculum",
    "classify_domains",
    "export_jsonl",
    "is_curriculum_valid",
    "scan_for_secrets",
    "split_dataset",
    "verify_math_content",
]
