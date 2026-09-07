"""Build structured knowledge cards from papers.

Output schema (per spec):
    paper_id, title, year, domain, main_claim, methods[], datasets[],
    trading_relevance, limitations[], possible_strategy_hypotheses[],
    risk_warnings[], implementation_notes[]

The LLM is asked to return strict JSON. We parse defensively (strip code
fences, fall back to an empty-but-valid card on parse failure) and normalize
field TYPES before validation (small models emit ``"year": 2021`` or
``"methods": "GARCH"``, which pydantic would otherwise reject).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.brain.local_llm import LocalLLM
from app.brain.prompt_loader import load_prompt
from app.config import get_settings
from app.memory.sqlite_store import SqliteStore, card_has_content

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

    @property
    def has_content(self) -> bool:
        """Kart gerçekten içerik taşıyor mu (title VEYA main_claim alfanümerik)?

        `False` ise `build()` kartı KAYDETMEMİŞTİR — LLM boş/parse edilemez yanıt
        döndürmüştür (yerel CPU'da tipik olarak zaman aşımı). Çağıran bunu başarısızlık
        olarak ele almalı, "kart üretildi" dememeli.
        """
        return card_has_content(self.model_dump())


def _extract_json(text: str) -> Any:
    """Modelin çıktısından JSON değerini çıkar; küçük modellerin tipik
    bozulmalarını (kod çiti, akıllı tırnak, sondaki virgül) toleranslı onar.

    Dönüş tipi bilinçli olarak ``Any``: model bazen nesne yerine dizi/skaler döndürür,
    "dict garantisi" çağıran tarafta (``_card_json``) verilir.
    """
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


# --- LLM tip sapmalarına karşı savunma -------------------------------------------------
# Küçük modeller (ör. qwen3:4b) şemayı sık ihlal eder: `"year": 2021` (int, şema
# `str | None`), `"methods": "GARCH"` (tek string, şema `list[str]`),
# `"limitations": null`. pydantic v2 gevşek modda bile bunları REDDEDER
# (ValidationError) → içerikli kart KAYDEDİLMEDEN `build()` çöker, dakikalarca süren
# yerel LLM emeği boşa gider. Aşağısı yalnız TİPİ düzeltir; Kural 7 gereği hiçbir değer
# UYDURULMAZ: çevrilemeyen/anlamsız değer boş bırakılır.

# Skaler alan → şema varsayılanı (normalizasyon boş metin ürettiğinde buraya düşülür).
_SCALAR_DEFAULTS: dict[str, str | None] = {
    "title": None,
    "year": None,
    "domain": None,
    "main_claim": "",
    "trading_relevance": "",
}

_LIST_FIELDS = (
    "methods",
    "datasets",
    "limitations",
    "possible_strategy_hypotheses",
    "risk_warnings",
    "implementation_notes",
)

# İç içe geçmiş yapıları düzleştirirken azami derinlik (patolojik/döngüsel girdi koruması).
_MAX_NEST_DEPTH = 3


def _as_text(value: Any, *, depth: int = 0) -> str:
    """Herhangi bir JSON değerini okunabilir TEK SATIR metne indir; çeviremezse "" döner.

    ``bool`` ve boş kapsayıcılar bilinçli olarak "" verir: "True" gibi bir değer alfanümerik
    olduğu için kartı "doluymuş" gibi gösterip boş-kart kapısını (``card_has_content``)
    delerdi. Anlamsız değer içerik sayılmaz (Kural 7).
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # 2021.0 → "2021" (yıl alanında ".0" kuyruğu gürültü).
        return str(int(value)) if value.is_integer() else str(value)
    if depth >= _MAX_NEST_DEPTH:
        return ""
    if isinstance(value, list | tuple | set):
        return "; ".join(t for t in (_as_text(v, depth=depth + 1) for v in value) if t)
    if isinstance(value, dict):
        # Tek anahtarlı sözlükte anahtar gürültüdür ({"name": "GARCH"} → "GARCH");
        # çok anahtarlıda "anahtar: değer" çiftleri korunur.
        single = len(value) == 1
        parts: list[str] = []
        for key, raw in value.items():
            text = _as_text(raw, depth=depth + 1)
            if not text:
                continue
            parts.append(text if single else f"{_as_text(key, depth=depth + 1)}: {text}")
        return "; ".join(parts)
    return ""


def _as_text_list(value: Any) -> list[str]:
    """Liste alanını `list[str]`e indir: None/bool → [], tek string → tek elemanlı liste,
    iç öğeler (sözlük dâhil) okunabilir tek satıra düşürülür. Boş öğeler atılır."""
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list | tuple | set):
        return [t for t in (_as_text(v, depth=1) for v in value) if t]
    text = _as_text(value)
    return [text] if text else []


def _normalize_card_data(data: Any) -> dict[str, Any]:
    """LLM'den gelen ham sözlüğü ``KnowledgeCard`` şemasına uygun TİPLERE çevir.

    Yalnız gelen anahtarlara dokunur (eksik alan pydantic varsayılanında kalır) ve
    hiçbir alanı doldurmaz — boş kartı "dolu" göstermez, boş-kart sözleşmesi korunur.
    """
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = dict(data)
    for field, default in _SCALAR_DEFAULTS.items():
        if field in out:
            text = _as_text(out[field])
            out[field] = text or default
    for field in _LIST_FIELDS:
        if field in out:
            out[field] = _as_text_list(out[field])
    return out


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
                # Paylaşımlı Ollama'da (tek slot) istek önce kuyrukta bekler; tavan ayardan
                # (HEKTOR_CARD_LLM_TIMEOUT_S). Bkz. settings.card_llm_timeout_s.
                timeout=max(30, int(getattr(self.settings, "card_llm_timeout_s", 180) or 180)),
            )
        except Exception:  # LLM yok / ağ / zaman aşımı — kart boş kalır
            return {}
        try:
            parsed = _extract_json(out)
        except (json.JSONDecodeError, ValueError):
            return {}
        # Model bazen nesne yerine dizi/skaler döndürür ("[...]", "null", "3"); sözleşme
        # dict olduğundan (çağıran `data[...]` ile yazıyor) dict olmayanı boş kart say.
        return parsed if isinstance(parsed, dict) else {}

    def build(self, paper_id: str, max_chars: int | None = None) -> KnowledgeCard:
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
        # Girdi tavanı: açık parametre > settings.card_max_chars (HEKTOR_CARD_MAX_CHARS) > 6000.
        # Prompt işleme CPU'da ~30 tok/sn olduğundan bu sayı kart süresinin yarısını belirler.
        if max_chars is None:
            max_chars = int(getattr(self.settings, "card_max_chars", 6000) or 6000)
        max_chars = max(500, max_chars)
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

        # Şemaya sokmadan ÖNCE tipleri düzelt (int yıl, tek-string methods, null liste...);
        # aksi hâlde pydantic ValidationError fırlatır ve İÇERİKLİ kart kaydedilmeden
        # build() çöker — çağıran (web /api/card, `hektor read-all`, RAG döngüsü) hata alır,
        # yerel CPU'da dakikalar süren LLM emeği boşa gider.
        data = _normalize_card_data(data)
        data["paper_id"] = paper_id
        try:
            card = KnowledgeCard.model_validate(data)
        except ValidationError as exc:
            # Normalizasyonun öngörmediği bir sapma kaldıysa: çökme yerine BOŞ kart üret.
            # Boş kart aşağıda KAYDEDİLMEZ; çağıran "kart üretilemedi" görür.
            log.warning("Kart şemaya uymadı, boş kart döndürülüyor (%s): %s", paper_id, exc)
            card = KnowledgeCard(paper_id=paper_id)

        # Boş kart KAYDEDİLMEZ. Eskiden LLM zaman aşımı/parse hatası `{}` döndürünce
        # kart yine de `pending` olarak yazılıyordu: onay kuyruğu boş kartla doluyor
        # (ölçüldü: 24 karttan 21'i boş), CLI listesi boşu doludan ayırt etmiyor, onaylansa
        # eğitim verisine sızıyordu. Kural 7: içerik yoksa kart da yok — yalnız uyarı.
        # Kaynak-yetersizlik yolu da buradan geçer (LLM'e gitmedi, yine de yazılmaz).
        if not card.has_content:
            log.warning(
                "Kart üretilemedi — LLM boş/parse edilemez yanıt döndürdü "
                "(yerel CPU'da tipik sebep zaman aşımı); KAYDEDİLMEDİ: %s",
                paper_id,
            )
            return card

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
