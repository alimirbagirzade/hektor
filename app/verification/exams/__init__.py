"""Anlama Doğrulama sınavları — L3 (uygulama), L4 (karşıolgu), L5 (kompozisyon).

Çekirdek fikir: "anlama" yüzdeyle değil, TEST EDİLEBİLİR davranışla kanıtlanır.
Her sınav çevrimdışı + deterministik (seed) koşar; referans daima güvenli
``compute_indicator`` registry'si veya whitelist'li ``safe_eval`` ile üretilir —
``eval``/``exec`` YOK (CLAUDE.md Kural 5). LLM kapalıyken sınav 'skipped' döner,
asla sahte 'pass' üretmez (Kural 2).
"""

from __future__ import annotations

from app.verification.exams.discipline_exam import run_discipline_exam
from app.verification.exams.l3_application import ApplicationExam, ExamResult
from app.verification.exams.l4_counterfactual import CounterfactualExam
from app.verification.exams.l5_composition import (
    CompositionGate,
    CompositionResult,
    GateResult,
)
from app.verification.exams.reference_oracle import ReferenceOracle
from app.verification.exams.registry import ExamSpec, get_spec, list_specs
from app.verification.exams.safe_eval import UnsafeExpressionError, safe_eval

__all__ = [
    "ApplicationExam",
    "CompositionGate",
    "CompositionResult",
    "CounterfactualExam",
    "ExamResult",
    "ExamSpec",
    "GateResult",
    "ReferenceOracle",
    "UnsafeExpressionError",
    "get_spec",
    "list_specs",
    "run_discipline_exam",
    "safe_eval",
]
