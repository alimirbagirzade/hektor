"""Birleşik SFT seti kurma — sentetik QA + onaylı kart + adversarial disiplin karışımı.

`lora-cloud-prep` (eğitim paketi) ve `pretrain-gate` (offline kalite kapısı) ORTAK yolu;
ikisi aynı birleştirme mantığını kullansın diye buraya çıkarıldı (drift önlenir).

Sıra: sentetik QA dosyası + onaylı kart örnekleri → hash + near-duplicate dedup (A7) →
disiplin örneklerini DEDUP'TAN SONRA ~%25 karıştır (#4 Fix B; şablon örnekleri near-dup
filtresine toplu takılmasın). EĞİTİM BAŞLATMAZ (kural 8).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.brain.synthetic_qa_builder import dedup_jsonl_lines
from app.lora.dataset_builder import build_dataset
from app.training.discipline_dataset import discipline_jsonl_lines, mix_discipline

_PII_MASK = "[kişisel-veri]"
# json.dumps(ensure_ascii=False) bunları ham bırakır ama str.splitlines() onları satır sonu
# sayar (LoRAExample.to_jsonl_line ile aynı koruma).
_LINE_SEPARATOR_ESCAPES = (
    (chr(0x2028), "\\u2028"),
    (chr(0x2029), "\\u2029"),
    (chr(0x85), "\\u0085"),
)


def _mask_strings(value: Any, patterns: list[re.Pattern[str]]) -> Any:
    """JSON değerindeki TÜM string'lerde PII desenlerini maskele (anahtarlara dokunmaz)."""
    if isinstance(value, str):
        for pat in patterns:
            value = pat.sub(_PII_MASK, value)
        return value
    if isinstance(value, list):
        return [_mask_strings(v, patterns) for v in value]
    if isinstance(value, dict):
        return {k: _mask_strings(v, patterns) for k, v in value.items()}
    return value


def redact_pii_line(line: str) -> str:
    """JSONL satırındaki e-posta / uluslararası telefon desenlerini maskele.

    Makale başlık bloklarındaki yazar e-postaları ("Corresponding author: …@…") sentetik QA
    bağlamına sızıyordu (2026-09-14 ölçümü: 1701 satırın 92'sinde 191 adres). Eğitim verisine
    kişisel veri girmez → birleştirmede maskelenir. Desenler pretrain-gate'in taradığı
    `_PII_PATTERNS` ile AYNI kaynaktan gelir (kapı ile maskeleme sapmasın); kapı yine son
    savunma olarak taramaya devam eder.

    Maskeleme ÇÖZÜMLENMİŞ string değerlerinde yapılır, ham JSON metninde DEĞİL (2026-09-15
    hatası): ham metinde satır sonu `\\n` olarak yazılır ve e-posta deseni kaçışın `n`
    harfini adresin parçası sanıyordu (`\\ndasashreeya@…` → `\\[kişisel-veri]`); geriye kalan
    `\\[` geçersiz JSON kaçışıdır → 1117 sentetik satırın 47'si bozuldu, pretrain-gate
    "okunamayan satır" ile NO-GO verdi. Eşleşme yoksa satır BAYT-ÖZDEŞ döner (dedup ve diff
    kararlı kalsın); JSON olmayan satır olduğu gibi bırakılır (kapı onu okunamayan diye bloklar).
    """
    from app.registry.promotion_gates import _PII_PATTERNS

    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return line
    masked = _mask_strings(obj, list(_PII_PATTERNS.values()))
    if masked == obj:
        return line
    out = json.dumps(masked, ensure_ascii=False)
    for ch, esc in _LINE_SEPARATOR_ESCAPES:
        out = out.replace(ch, esc)
    return out


@dataclass
class AssemblyResult:
    """Birleşik SFT satırları + nereden geldiği şeffaflığı."""

    lines: list[str]
    synth_n: int
    card_n: int
    deduped: int  # synth + kart, dedup sonrası
    discipline: dict[str, Any] | None = field(default=None)
    low_value_dropped: int = 0  # atılan çekimser / "pasaj" atıflı sentetik örnek

    @property
    def total(self) -> int:
        return len(self.lines)


def assemble_sft_lines(
    settings: Any,
    *,
    discipline: bool = True,
    discipline_ratio: float = 0.25,
    seed: int = 0,
) -> AssemblyResult:
    """Birleşik SFT JSONL satırlarını kur (eğitim BAŞLATMAZ).

    Args:
        settings: `get_settings()` çıktısı (`.root` kullanılır).
        discipline: Adversarial disiplin örneklerini karıştır (#4 Fix B).
        discipline_ratio: Disiplin payı (disiplin/(taban+disiplin)); v5 dersi ~0.25.
        seed: Determinizm tabanı (karıştırma — kural 6).
    """
    lora_dir = settings.root / "data" / "lora_sft"
    synth_path = lora_dir / "synthetic_qa.jsonl"

    lines: list[str] = []
    synth_n = 0
    low_value_dropped = 0
    if synth_path.exists():
        synth_lines = [
            ln for ln in synth_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        synth_n = len(synth_lines)
        # Üretici filtresinden ÖNCE yazılmış çekimser / "pasaj" atıflı örnekler de eğitime
        # girmesin (eski veride %3-5; kitaplardan üretilen ilk partide %33-50).
        kept_synth = [ln for ln in synth_lines if not _is_low_value_line(ln)]
        low_value_dropped = synth_n - len(kept_synth)
        lines += kept_synth

    card_n = 0
    try:
        from app.memory.sqlite_store import SqliteStore

        card_lines = [
            ex.to_jsonl_line() for ex in build_dataset(SqliteStore().list_approved_cards())
        ]
        card_n = len(card_lines)
        lines += card_lines
    except Exception:
        pass

    merged = dedup_jsonl_lines([redact_pii_line(ln) for ln in lines])
    deduped = len(merged)

    disc_stats: dict[str, Any] | None = None
    if discipline and discipline_ratio > 0:
        disc_lines = discipline_jsonl_lines(seed=seed)
        merged, disc_stats = mix_discipline(merged, disc_lines, ratio=discipline_ratio, seed=seed)

    return AssemblyResult(
        lines=merged,
        synth_n=synth_n,
        card_n=card_n,
        deduped=deduped,
        discipline=disc_stats,
        low_value_dropped=low_value_dropped,
    )


def _is_low_value_line(line: str) -> bool:
    """Sentetik JSONL satırının assistant cevabı düşük değerli mi? Okunamayan satır tutulur
    (pretrain-gate onu ayrıca 'okunamayan' diye engeller — sessiz atma yok)."""
    import json

    from app.brain.synthetic_qa_builder import is_low_value_answer

    try:
        msgs = json.loads(line).get("messages") or []
    except (json.JSONDecodeError, AttributeError, TypeError):
        return False
    answer = next(
        (str(m.get("content", "")) for m in reversed(msgs) if m.get("role") == "assistant"), ""
    )
    return bool(answer) and is_low_value_answer(answer)
