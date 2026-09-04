"""Assemble train/valid JSONL datasets from stored training examples.

Reads from the ``training_examples`` table, deduplicates, shuffles
deterministically, splits, and writes MLX-LM compatible JSONL files. Also
computes a content hash used for adapter provenance.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.memory.sqlite_store import KnowledgeCard, SqliteStore, TrainingExample

# LoRA fazı → stage eşlemesi
STAGE_TO_PHASE: dict[str, int] = {
    "lora_phase_1": 1,
    "lora_phase_2": 2,
    "lora_phase_3": 3,
    "lora_phase_4": 4,
}


@dataclass
class DatasetResult:
    train_path: Path
    valid_path: Path
    n_train: int
    n_valid: int
    content_hash: str


def _to_mlx_record(instruction: str, input_text: str, output_text: str) -> dict:
    # MLX-LM 'completions'-style: prompt + completion
    prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return {"prompt": prompt, "completion": output_text}


class DatasetBuilder:
    def __init__(self, store: SqliteStore | None = None) -> None:
        self.store = store or SqliteStore()
        self.settings = get_settings()

    def collect(
        self,
        *,
        phase: int | None = None,
        lora_eligible_only: bool = True,
    ) -> list[dict]:
        """Training örneklerini topla.

        phase: 1-4 → sadece o fazın stage'indeki kartlardan gelen örnekler
               None → tüm örnekler (lora_eligible_only=True ise sadece uygunlar)
        lora_eligible_only: True → sadece lora_eligible=1 VE review_status=approved örnekler
        """
        # LoRA uygun paper_id setini oluştur
        approved_paper_ids: set[str] | None = None
        # Faz→stage eşlemesi (phase belirtilmişse kullanılır)
        phase_stages: set[str] = set()
        if phase is not None:
            phase_stages = {s for s, p in STAGE_TO_PHASE.items() if p == phase}

        if lora_eligible_only:
            with self.store.session() as s:
                cards = list(
                    s.scalars(
                        select(KnowledgeCard).where(
                            KnowledgeCard.lora_eligible == 1,
                            KnowledgeCard.review_status == "approved",
                        )
                    )
                )
            if phase is not None:
                # Sadece istenen fazın stage'indeki kartlar
                approved_paper_ids = {c.paper_id for c in cards if c.stage in phase_stages}
            else:
                approved_paper_ids = {c.paper_id for c in cards}

        with self.store.session() as s:
            rows = list(s.scalars(select(TrainingExample)))

        seen: set[str] = set()
        records: list[dict] = []
        for r in rows:
            # cross_paper_synthesis örnekleri source_paper_id TAŞIMAZ → faz/uygunluk
            # filtresinden MUAF ("her zaman dahil" sözleşmesi). Eskiden buradaki
            # KOŞULSUZ filtre bloğu sentez satırlarını is_synthesis'e ulaşmadan eliyordu
            # (alttaki istisna ölü koddu); blok kaldırıldı, filtre yalnız makale-kaynaklı örneklere.
            is_synthesis = r.example_type == "cross_paper_synthesis"

            if not is_synthesis:
                if phase is not None and approved_paper_ids is not None:
                    if r.source_paper_id not in approved_paper_ids:
                        continue
                elif lora_eligible_only and approved_paper_ids is not None:
                    if r.source_paper_id is None:
                        continue
                    if r.source_paper_id not in approved_paper_ids:
                        continue

            key = f"{r.instruction}||{r.input_text}||{r.output_text}"
            if key in seen:
                continue
            seen.add(key)
            records.append(_to_mlx_record(r.instruction, r.input_text, r.output_text))
        return records

    def build(
        self,
        valid_ratio: float = 0.15,
        seed: int = 13,
        phase: int | None = None,
        lora_eligible_only: bool = True,
    ) -> DatasetResult:
        """Train/valid JSONL dosyalarını oluştur.

        phase belirtilmişse curriculum pacing uygulanır:
        - %60 mevcut faz örnekleri
        - %30 alt faz örnekleri (phase-1, varsa)
        - %10 üst faz örnekleri (phase+1, varsa)
        """
        rng = random.Random(seed)

        if phase is not None:
            current = self.collect(phase=phase, lora_eligible_only=lora_eligible_only)
            prev = (
                self.collect(phase=phase - 1, lora_eligible_only=lora_eligible_only)
                if phase > 1
                else []
            )
            nxt = (
                self.collect(phase=phase + 1, lora_eligible_only=lora_eligible_only)
                if phase < 4
                else []
            )

            # Curriculum pacing (yalnızca toplam > 20 ise)
            total_available = len(current) + len(prev) + len(nxt)
            if total_available > 20:
                n_current = max(1, int(total_available * 0.60))
                n_prev = max(0, int(total_available * 0.30))
                n_next = max(0, int(total_available * 0.10))
                rng.shuffle(current)
                rng.shuffle(prev)
                rng.shuffle(nxt)
                records = current[:n_current] + prev[:n_prev] + nxt[:n_next]
            else:
                records = current + prev + nxt
        else:
            records = self.collect(lora_eligible_only=lora_eligible_only)

        # GLOBAL dedup: faz yolunda cross_paper_synthesis örnekleri faz-muaf olduğundan
        # current/prev/nxt'in her birinde tekrar gelir (ayrı 'seen' set'leri). Bölmeden
        # önce tekilleştir → ağırlık şişmesi + valid/train sızıntısı (sahte OOS) önlenir.
        _seen_keys: set[str] = set()
        _deduped: list[dict] = []
        for _rec in records:
            _k = f"{_rec.get('prompt', '')}||{_rec.get('completion', '')}"
            if _k not in _seen_keys:
                _seen_keys.add(_k)
                _deduped.append(_rec)
        records = _deduped

        rng.shuffle(records)
        # Valid seti train'den AYRIK olmalı (OOS garantisi, CLAUDE.md Kural 2):
        # eski "bootstrap" kopyalama (valid ⊂ train) KALDIRILDI → sahte OOS yok.
        # Az veride valid azınlıkta kalır; train daima çoğunluk.
        if len(records) < 2:
            n_valid = 0
        elif len(records) < 8:
            n_valid = max(1, int(len(records) * valid_ratio))
        else:
            n_valid = max(4, int(len(records) * valid_ratio))
        valid = records[:n_valid]
        train = records[n_valid:]

        out_dir = self.settings.jsonl_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / "train.jsonl"
        valid_path = out_dir / "valid.jsonl"

        # CLOBBER KORUMASI: üretilecek train boşsa (collect() geçici boş döndüyse)
        # mevcut hazır train.jsonl/valid.jsonl dosyalarını SIFIRLAMA — veri kaybı önlenir.
        if not train:
            return DatasetResult(
                train_path=train_path,
                valid_path=valid_path,
                n_train=0,
                n_valid=0,
                content_hash="",
            )

        hasher = hashlib.sha256()
        with open(train_path, "w", encoding="utf-8") as f:
            for rec in train:
                line = json.dumps(rec, ensure_ascii=False)
                f.write(line + "\n")
                hasher.update(line.encode("utf-8"))
        with open(valid_path, "w", encoding="utf-8") as f:
            for rec in valid:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return DatasetResult(
            train_path=train_path,
            valid_path=valid_path,
            n_train=len(train),
            n_valid=len(valid),
            content_hash=hasher.hexdigest()[:16],
        )
