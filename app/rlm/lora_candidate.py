"""RLM koşularından LoRA dataset ADAYI seçimi (talimat §16).

Talimat §16: RLM/Paper Mastery logları ileride LoRA dataset KAYNAĞI olabilir — ama
yalnız çok yüksek güvenli, tam-doğrulanmış koşular. Eşikler (§16):

    final_confidence ≥ 0.85
    citation_score   ≥ 0.90
    grounding_score  ≥ 0.90
    unsupported_claims = []

§16 ayrıca `human_review_status = approved` der; ama RLM'de otomatik onay mekanizması
YOKTUR. Bu yüzden seçim YALNIZ yukarıdaki SAYISAL kapıları uygular; insan onayı seçimin
PARÇASI değil, AŞAĞI-AKIŞ manuel adımdır (her aday `requires_human_approval=True` taşır).

ÖNEMLİ — ADAY ≠ EĞİTİM VERİSİ:
- Bu modül SALT-OKUMA seçim + export yapar. Hiçbir eğitim başlatmaz (CLAUDE.md kural 8).
- Hiçbir koşuyu OTOMATİK onaylamaz. Sayısal eşikleri geçen koşular yalnız "aday"dır;
  herhangi bir eğitimden ÖNCE İNSAN ONAYI şarttır (`requires_human_approval=True`).
- Amaç: yüksek-kaliteli kaynaklı/doğrulanmış cevap davranışını ileride LoRA ile
  öğretmek için aday havuzu hazırlamak — bilgiyi ezberletmek değil (§16).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.rlm.rlm_store import RlmStore

log = logging.getLogger(__name__)

# §16 eşikleri (sabit; talimattan). Salt-okuma seçim eşikleri — config'e bağlanmadı
# (dead-config riskinden kaçınmak için fonksiyon parametresi olarak verilir).
MIN_CONFIDENCE = 0.85
MIN_CITATION = 0.90
MIN_GROUNDING = 0.90

# Yalnız GERÇEK cevap üreten koşular aday olabilir (çekimser/no_llm/failed hariç).
_ELIGIBLE_STATUS = ("answered", "answered_with_limitation")


@dataclass
class LoraCandidate:
    """§16 sayısal eşiklerini geçen tek bir RLM koşusu (insan onayı BEKLEYEN aday)."""

    run_id: str
    task_type: str
    query: str
    answer: str
    final_confidence: float
    citation_score: float
    grounding_score: float
    supported_claims: list[str] = field(default_factory=list)
    requires_human_approval: bool = True  # MUTLAK — onaysız eğitim YOK (kural 8)
    note: str = "RLM aday — eğitim verisi DEĞİL; insan onayı şart (§16)."


def select_lora_candidates(
    store: RlmStore | None = None,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    min_citation: float = MIN_CITATION,
    min_grounding: float = MIN_GROUNDING,
    limit: int = 1000,
) -> list[LoraCandidate]:
    """§16 eşiklerini geçen RLM koşularını aday olarak seç (salt-okuma).

    Bir koşu aday olur ancak ve ancak: status gerçek-cevap ∧ final_confidence ≥ eşik ∧
    citation ≥ eşik ∧ grounding ≥ eşik ∧ unsupported_claims = [].
    """
    # limit POZİTİF olmalı: SQLite'ta LIMIT -1 = 'sınırsız' (tümü), LIMIT 0 = sıfır satır.
    # Negatif/0 limit, sessiz-kesme uyarısını yanlış/çelişkili tetikler (total-limit overcount).
    if limit <= 0:
        raise ValueError("limit pozitif olmalı (>0)")
    store = store or RlmStore()
    # Sessiz-kesme uyarısı: list_runs yalnız en yeni `limit` koşuyu tarar; daha fazla
    # koşu varsa eşik geçen ESKİ adaylar atlanır → kullanıcı/çağıran bunu bilmeli.
    total = store.count_runs()
    if 0 < limit < total:
        log.warning(
            "RLM koşu sayısı (%d) tarama limitini (%d) aşıyor — en eski %d koşu aday "
            "seçiminde ATLANDI; daha yüksek limit verin.",
            total,
            limit,
            total - limit,
        )
    candidates: list[LoraCandidate] = []
    for run in store.list_runs(limit=limit):
        if run["status"] not in _ELIGIBLE_STATUS:
            continue
        if float(run["final_confidence"]) < min_confidence:
            continue
        ver = store.get_verification(run["run_id"])
        if ver is None:
            continue
        if float(ver["citation_score"]) < min_citation:
            continue
        if float(ver["grounding_score"]) < min_grounding:
            continue
        if ver["unsupported_claims"]:  # desteklenmeyen iddia VARSA aday olamaz (§16)
            continue
        candidates.append(
            LoraCandidate(
                run_id=run["run_id"],
                task_type=run["task_type"],
                query=run["user_query"],
                answer=run["final_answer"],
                final_confidence=float(run["final_confidence"]),
                citation_score=float(ver["citation_score"]),
                grounding_score=float(ver["grounding_score"]),
                supported_claims=list(ver["supported_claims"]),
            )
        )
    return candidates


def export_candidates_jsonl(candidates: Iterable[LoraCandidate], path: str | Path) -> int:
    """Adayları JSONL olarak yaz (her satır bir aday). EĞİTİM VERİSİ DEĞİL — inceleme için.

    Returns: yazılan aday sayısı.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n
