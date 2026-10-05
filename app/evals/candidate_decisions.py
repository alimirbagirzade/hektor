"""candidate_decisions.py — aday modeller için karşılaştırma KARARLARININ deposu (eklemeli).

Kararı üreten tek yol aday/aktif karşılaştırmasıdır (``app.evals.candidate_compare``). Model
etkinleştirme (``app.feedback.model_activation``) yalnız bu depodan okur; karar uydurmaz.

Karar değerleri (sabit):
- ``kabul``          : önceden sabitlenmiş ölçütlere göre aday aktif modelden kötü değil ve
                       kritik hata yok → ana model olarak etkinleştirilebilir (insan onayıyla).
- ``yetersiz_kanit`` : kanıt karar vermeye yetmiyor → YALNIZ belirgin etiketli deneme sohbeti.
- ``ret``            : aday aktif modelden kötü → etkinleştirilemez.
- ``kritik_ret``     : kritik hata → hiçbir yuvada kullanılamaz, üretime geçirilemez.

Her kayıt adayın Ollama etiketini VE karşılaştırılan digest'ini taşır: etiket sonradan başka
ağırlıklarla yeniden oluşturulursa eski karar o modele uygulanmaz.

Bu kararlar bir LLM soru-cevap karşılaştırmasıdır; trading performansı iddiası DEĞİLDİR.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.feedback.chat_store import utcnow

DECISIONS = ("kabul", "yetersiz_kanit", "ret", "kritik_ret")
MAIN_OK = frozenset({"kabul"})
TRIAL_OK = frozenset({"kabul", "yetersiz_kanit"})

DECISION_TR = {
    "kabul": "Kabul",
    "yetersiz_kanit": "Yetersiz kanıt",
    "ret": "Ret",
    "kritik_ret": "Kritik hatadan ret",
}


def decisions_path(root: Path | None = None) -> Path:
    return (root or get_settings().root) / "storage" / "candidate_decisions.jsonl"


def norm_digest(digest: str) -> str:
    return str(digest or "").removeprefix("sha256:").strip().lower()


def norm_tag(tag: str) -> str:
    tag = str(tag or "").strip()
    return tag[: -len(":latest")] if tag.endswith(":latest") else tag


def record_decision(rec: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Kararı ekle (eklemeli; eski kayıt silinmez/değişmez)."""
    decision = str(rec.get("decision") or "")
    if decision not in DECISIONS:
        raise ValueError(f"Geçersiz karar: {decision!r} (izinli: {DECISIONS})")
    tag = norm_tag(str(rec.get("candidate_tag") or ""))
    digest = norm_digest(str(rec.get("candidate_digest") or ""))
    if not tag or not digest:
        raise ValueError("Karar adayın etiketini VE karşılaştırılan digest'ini taşımalı.")
    row = {
        "decision_id": "dec_" + secrets.token_hex(6),
        "decided_at": utcnow(),
        **rec,
        "decision": decision,
        "candidate_tag": tag,
        "candidate_digest": digest,
    }
    path = decisions_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    return row


def list_decisions(root: Path | None = None) -> list[dict[str, Any]]:
    path = decisions_path(root)
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(ln)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("decision") in DECISIONS:
            out.append(row)
    return out


def latest_decision(tag: str, digest: str = "", root: Path | None = None) -> dict[str, Any] | None:
    """Bu etiket (+ verilirse digest) için EN SON karar. Digest uyuşmayan karar sayılmaz."""
    want_tag, want_dig = norm_tag(tag), norm_digest(digest)
    found = None
    for row in list_decisions(root):
        if row["candidate_tag"] != want_tag:
            continue
        if want_dig and row["candidate_digest"] != want_dig:
            continue
        found = row
    return found
