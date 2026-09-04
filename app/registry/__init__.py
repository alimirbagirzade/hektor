"""app/registry — model/veri kayıt defteri (sürümleme + terfi kararları).

EKSİK olan yatay katman: dataset / RAG-index / embedding / RLM-reward sürümleri ve
her terfi kararının denetlenebilir kaydı. Adapter yaşam döngüsü ayrı kalır
(``app/lora/adapter_registry`` + ``app/training/adapter_registry``).
"""

from __future__ import annotations

from app.registry.promotion_gates import (
    ScanResult,
    approve_dataset,
    check_rag_index_eval,
    gate_reward_dataset,
    reject_dataset,
    scan_secret_pii,
)
from app.registry.version_store import RegistryStore

__all__ = [
    "RegistryStore",
    "ScanResult",
    "approve_dataset",
    "check_rag_index_eval",
    "gate_reward_dataset",
    "reject_dataset",
    "scan_secret_pii",
]
