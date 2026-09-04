"""LoRA dataset builder — onaylı bilgi kartlarını SFT formatına çevir.

KnowledgeCard JSON'undan soru-cevap üretir ve OpenAI-uyumlu `messages`
formatında JSONL örnekleri verir. Yalnız `lora_eligible=1` ve
`review_status=approved` kartlar dahil edilir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SYSTEM_PROMPT = (
    "Sen Achilles yerel AI asistanısın. Matematik, fizik, istatistik, felsefe, "
    "trading ve AI sistem tasarımı konularında adım adım, kaynak temelli, kontrollü "
    "ve belirsizliği doğru ifade eden cevaplar ver. RAG bağlamı varsa kullan. "
    "Kaynak yoksa emin görünme."
)


@dataclass
class LoRAExample:
    """Tek bir SFT eğitim örneği (system→user→assistant)."""

    messages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_jsonl_line(self) -> str:
        """JSONL satırı olarak serileştir (messages + opsiyonel metadata).

        `json.dumps(ensure_ascii=False)` Unicode satır ayırıcılarını (U+2028 LINE
        SEPARATOR, U+2029 PARAGRAPH SEPARATOR, U+0085 NEXT LINE) HAM bırakır. Bunlar
        `json.loads` için satır sonu DEĞİL ama `str.splitlines()` için satır sonudur →
        downstream lora-split / sft_assembly `.splitlines()` ile tek geçerli kaydı İKİ
        geçersiz parçaya böler (trainer çöker / sessiz bozuk veri). Bu üç karakteri
        `\\uXXXX` kaçışına çevir: JSON anlamı korunur (loads aynı string'i üretir), satır
        bütünlüğü bozulmaz. Kart + feedback yollarını tek noktada korur (Kademe-2 av)."""
        data: dict = {"messages": self.messages}
        if self.metadata:
            data["metadata"] = self.metadata
        line = json.dumps(data, ensure_ascii=False)
        # U+2028/U+2029/U+0085: json.dumps'in ham bıraktığı, splitlines'ı bölen ayırıcılar.
        for ch, esc in ((chr(0x2028), "\\u2028"), (chr(0x2029), "\\u2029"), (chr(0x85), "\\u0085")):
            line = line.replace(ch, esc)
        return line


def _build_answer(card_json: dict) -> str:
    """Kart içeriğinden cevap metni kur.

    summary → main_claim → trading_relevance → possible_strategy_hypotheses
    sırasıyla kontrol eder; boş olmayan tüm bölümleri birleştirir.
    """
    parts: list[str] = []

    for key in ("summary", "main_claim"):
        val = str(card_json.get(key) or "").strip()
        if val:
            parts.append(val)

    trading_rel = str(card_json.get("trading_relevance") or "").strip()
    if trading_rel:
        parts.append(f"Trading önemi: {trading_rel}")

    hypotheses = card_json.get("possible_strategy_hypotheses") or []
    if isinstance(hypotheses, list) and hypotheses:
        lines = ["Olası strateji hipotezleri:"]
        lines.extend(f"- {str(h).strip()}" for h in hypotheses if h)
        if len(lines) > 1:
            parts.append("\n".join(lines))

    formulas = card_json.get("formulas") or []
    if isinstance(formulas, list) and formulas:
        lines = ["İlgili formüller:"]
        for f in formulas:
            if isinstance(f, dict):
                name = str(f.get("name") or "").strip()
                plain = str(f.get("plain") or f.get("latex") or "").strip()
                desc = str(f.get("description") or "").strip()
                entry = " — ".join(p for p in (name, plain, desc) if p)
                if entry:
                    lines.append(f"- {entry}")
            elif isinstance(f, str) and f.strip():
                lines.append(f"- {f.strip()}")
        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n\n".join(parts).strip()


def card_to_lora_example(card: dict, paper_id: str) -> LoRAExample | None:
    """Bir KnowledgeCard kaydını SFT örneğine çevir.

    `card` formatı `SqliteStore._card_to_dict` çıktısıdır:
    en üst düzey 'card_json' anahtarı parse edilmiş içeriği taşır.
    Uygun değilse (eligible/approved değil veya içerik boş) None döner.
    """
    if card.get("review_status") != "approved":
        return None
    if not card.get("lora_eligible"):
        return None

    card_json = card.get("card_json")
    if not isinstance(card_json, dict):
        return None

    title = str(card_json.get("title") or "").strip()
    if not title:
        return None

    answer = _build_answer(card_json)
    if not answer:
        return None

    question = f"{title} konusunu açıkla."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    metadata = {
        "paper_id": paper_id,
        "card_id": card.get("card_id", ""),
        "source_id": paper_id,
        "difficulty": card.get("difficulty", 0.0),
        "stage": card.get("stage", ""),
    }
    return LoRAExample(messages=messages, metadata=metadata)


def build_dataset(cards: list[dict]) -> list[LoRAExample]:
    """Kart listesinden geçerli LoRA örnekleri üret (None'ları ele)."""
    examples: list[LoRAExample] = []
    for card in cards:
        paper_id = str(card.get("paper_id", ""))
        example = card_to_lora_example(card, paper_id)
        if example is not None:
            examples.append(example)
    return examples


def export_jsonl(examples: list[LoRAExample], path: Path) -> int:
    """Örnekleri JSONL dosyasına yaz; yazılan satır sayısını döndür."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [ex.to_jsonl_line() for ex in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)
