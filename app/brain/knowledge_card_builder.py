"""Build structured knowledge cards from papers.

Output schema (per spec):
    paper_id, title, year, domain, main_claim, methods[], datasets[],
    trading_relevance, limitations[], possible_strategy_hypotheses[],
    risk_warnings[], implementation_notes[]

The LLM is asked to return strict JSON. We parse defensively (strip code
fences, fall back to an empty-but-valid card on parse failure).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.brain.local_llm import LocalLLM
from app.brain.prompt_loader import load_prompt
from app.config import get_settings
from app.memory.sqlite_store import SqliteStore

log = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Kart üretmek için gereken asgari kaynak metni. Bunun altındaki makaleler (ör. PDF
# çıkarımı bozuk: 0-2KB) LLM'e GÖNDERİLMEZ — yalnız boş kart üretip CPU'yu boşa
# harcarlar (yerel CPU'da ~5 dk/çağrı × retry). Gerçek bir araştırma makalesinin
# çıkarılmış metni her zaman bundan çok daha uzundur (en küçükleri ~11KB ölçüldü).
_MIN_SOURCE_CHARS = 1500


class KnowledgeCard(BaseModel):
    paper_id: str
    title: str | None = None
    year: str | None = None
    domain: str | None = None
    main_claim: str = ""
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    trading_relevance: str = ""
    limitations: list[str] = Field(default_factory=list)
    possible_strategy_hypotheses: list[str] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)


def _extract_json(text: str) -> dict[str, Any]:
    """Modelin çıktısından JSON nesnesini çıkar; küçük modellerin tipik
    bozulmalarını (kod çiti, akıllı tırnak, sondaki virgül) toleranslı onar."""
    m = _JSON_FENCE.search(text)
    raw = m.group(1) if m else text
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)  # sondaki virgüller
        return json.loads(repaired)


class KnowledgeCardBuilder:
    def __init__(self, store: SqliteStore | None = None, llm: LocalLLM | None = None) -> None:
        self.store = store or SqliteStore()
        self.llm = llm or LocalLLM()
        self.settings = get_settings()

    def _classify_card(self, card: KnowledgeCard) -> tuple[str, float, str]:
        """Kartı otomatik sınıflandır → (trust_level, difficulty, stage).

        Basit kural tabanlı sınıflandırma (LLM çağrısı gerekmez, 8GB'da güvenli):
        - difficulty: hypotheses sayısı + methods derinliğine göre 0.0-1.0
        - stage: difficulty aralığına göre
        - trust_level: her zaman "draft" (insan onayı gerekir)
        """
        hypotheses = card.possible_strategy_hypotheses
        methods = card.methods
        impl_notes = card.implementation_notes
        risk_warnings = card.risk_warnings

        if not hypotheses:
            difficulty = 0.1
        elif len(hypotheses) <= 2 and len(methods) < 3:
            difficulty = 0.2
        else:
            difficulty = 0.4

        # implementation_notes dolu VE risk_warnings dolu → +0.2 ekle
        if impl_notes and risk_warnings:
            difficulty = min(1.0, difficulty + 0.2)

        if difficulty <= 0.25:
            stage = "lora_phase_1"
        elif difficulty <= 0.50:
            stage = "lora_phase_2"
        elif difficulty <= 0.75:
            stage = "lora_phase_3"
        else:
            stage = "lora_phase_4"

        return "draft", difficulty, stage

    def _load_text(self, paper_id: str) -> str:
        path = self.settings.extracted_text_dir / f"{paper_id}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "\n\n".join(c.text for c in self.store.list_chunks(paper_id))

    _SKELETON = (
        '{"title":"","year":"","domain":"","main_claim":"","methods":[],'
        '"datasets":[],"trading_relevance":"","limitations":[],'
        '"possible_strategy_hypotheses":[],"risk_warnings":[],'
        '"implementation_notes":[]}'
    )

    def _card_json(self, text: str, system: str, *, max_tokens: int) -> dict[str, Any]:
        """Modelden tek bir JSON kart üret; başarısızsa boş dict döner."""
        prompt = (
            f"MAKALE:\n{text}\n\n"
            f"Bu JSON şemasını makaleye göre DOLDUR, yalnızca JSON döndür:\n{self._SKELETON}\n"
            "Bilinmeyen alanı boş bırak; makalede olmayan şeyi uydurma."
        )
        try:
            out = self.llm.generate(
                prompt,
                system=system,
                temperature=0.1,
                fmt="json",
                max_tokens=max_tokens,
                timeout=180,
            )
        except Exception:  # LLM yok / ağ / zaman aşımı — kart boş kalır
            return {}
        try:
            return _extract_json(out)
        except (json.JSONDecodeError, ValueError):
            return {}

    def build(self, paper_id: str, max_chars: int = 6000) -> KnowledgeCard:
        """8GB-dostu: kısa girdi + Ollama JSON modu + num_predict cap + retry.

        Büyük metni tek seferde modele vermek (eski 14000 krk) küçük modellerde
        bozuk JSON / 7B'de timeout üretiyordu. Artık odaklı bir alıntı +
        ``fmt="json"`` ile geçerli JSON garanti altına alınır; boş kalırsa daha
        kısa metinle bir kez daha denenir.
        """
        # Windows/CRLF savunması: paper_id'ye takılı kalan \r/\n/boşluk hem DB kaydını
        # hem 'reports/papers/<id>_card.json' dosya adını kirletir; ikincisi Windows'ta
        # OSError [Errno 22] Invalid argument verir. Girişte tek noktada temizle.
        paper_id = paper_id.strip()
        try:
            system = load_prompt("knowledge_card")
        except FileNotFoundError:
            system = (
                "Akademik makaleden yapılandırılmış bilgi kartı çıkar. "
                "SADECE geçerli JSON döndür (markdown/açıklama yok). Kaynak uydurma."
            )

        full_text = self._load_text(paper_id)
        data: dict[str, Any]
        if len(full_text.strip()) < _MIN_SOURCE_CHARS:
            # Kaynak metni yetersiz (ör. PDF çıkarımı bozuk) → LLM'e HİÇ gitme: yalnız boş
            # kart üretip CPU'yu boşa harcar. Boş kart döner; _approve_if_content onaylamaz,
            # rebuild deneme tavanı bunu kalıcı bırakır → sonsuz/boşa LLM yok.
            log.warning(
                "Kart atlandı — kaynak metni yetersiz (%d krk < %d): %s",
                len(full_text.strip()),
                _MIN_SOURCE_CHARS,
                paper_id,
            )
            data = {}
        else:
            # num_predict tavanı düşük: dolu bir kart ~300-500 token; yerel CPU'da üretim
            # ~4 tok/s olduğundan gereksiz yüksek tavan (eski 900) çağrı başına ~1 dk boşa
            # harcıyordu. fmt=json zaten JSON kapanışında durur; tavan yalnız gevezeliği keser.
            data = self._card_json(full_text[:max_chars], system, max_tokens=700)
            if not str(data.get("main_claim") or "").strip():
                # daha kısa alıntıyla tek retry (8GB'da hız + JSON sağlamlığı)
                data = self._card_json(full_text[:3000], system, max_tokens=500) or data
            if not str(data.get("main_claim") or "").strip() and len(full_text) > max_chars * 2:
                # Büyük belgelerde (kitaplar) ilk 6000 krk kapak/içindekiler/ön-madde olabilir →
                # gerçek içerik için belge BOYUNCA orantılı birkaç kesit dene. (Sabit 8000 krk
                # ofset, devasa kitaplarda HÂLÂ ön-madde kalıyordu — boş kart darboğazı fix.)
                for frac in (0.25, 0.55):
                    offset = min(int(len(full_text) * frac), len(full_text) - max_chars)
                    data = (
                        self._card_json(
                            full_text[offset : offset + max_chars], system, max_tokens=700
                        )
                        or data
                    )
                    if str(data.get("main_claim") or "").strip():
                        break

        data["paper_id"] = paper_id
        card = KnowledgeCard.model_validate(data)

        trust_level, difficulty, stage = self._classify_card(card)

        card_id = f"card_{uuid.uuid4().hex[:12]}"
        self.store.save_knowledge_card(
            card_id=card_id,
            paper_id=paper_id,
            model=self.llm.model,
            card=card.model_dump(),
            trust_level=trust_level,
            review_status="pending",
            lora_eligible=0,
            difficulty=difficulty,
            stage=stage,
        )
        out_path: Path = self.settings.reports_dir / "papers" / f"{paper_id}_card.json"
        out_path.write_text(
            json.dumps(card.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return card
