"""Web için tembel-yüklemeli PEFT adapter sohbet servisi.

Eğitilen LoRA adapter'ı (veya base) transformers/PEFT ile YEREL yükler, chat-template
uygular ve cevap üretir — Ollama gerektirmez. Model bir kez yüklenir, bellekte tutulur
(yeniden yükleme maliyeti CPU'da yüksek). Üretim greedy/deterministtir (Kural 6).

Eğitim ↔ çıkarım eşleşmesi: adapter, SYSTEM_PROMPT'lu ve çoğunlukla "BAĞLAM: … SORU: …"
biçimli örneklerle eğitildi. Sohbet de aynı sistem prompt'unu gönderir; `use_context=True`
ise soru korpustan retrieval ile getirilen parçalar eğitimdeki BİREBİR biçimde gömülür.
Retrieval boşsa model hiç çağrılmaz (Kural 7 — kaynak uydurma yok).

Kaynaklı modda cevap HİBRİTTİR (`app/brain/hybrid_answer.py`): model yalnız Kısa Cevap'ı
yazar; Kaynaklar / Bağlam Kalitesi / Akademik Bulgu / Trading Hipotezi / Test Planı /
Riskler / Sonraki Adım retrieval, bilgi kartları ve kurallardan deterministik kurulur.

`adapter_eval._load_model`/`_generate`/`_resolve_base_model` tekrar kullanılır (eğitim
doğrulamasıyla AYNI yükleme/üretim yolu → tutarlılık). Üretim ağırdır (CPU'da dakikalar);
endpoint senkron `def` olmalı ki FastAPI'nin threadpool'unda koşsun, event loop'u bloklamasın.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.brain.hybrid_answer import CardLookup
    from app.memory.retrieval_service import RetrievedChunk

log = logging.getLogger(__name__)

# Tek-örnek model önbelleği: aynı (base, adapter) tekrar istenirse yeniden yüklenmez.
_CACHE: dict[str, Any] = {"key": None, "tok": None, "model": None}
_LOCK = Lock()  # üretimi serileştir (tek kullanıcı; eşzamanlı model erişimini önle)

# Bağlam karakter bütçesi. Eğitim `max_seq_length=1024` token ile yapıldı
# (configs/lora/lora_profiles.yaml); sistem prompt'u + soru + cevap payı düşülünce bağlamın
# ~600 token'ı (≈2400 karakter) aşmaması modeli eğitildiği uzunluk dağılımında tutar.
CONTEXT_CHAR_BUDGET = 2400

NO_SOURCE_ANSWER = (
    "Kaynak bulunamadı: korpusta bu soruya dayanak olacak parça yok. Uydurmamak için model "
    "çağrılmadı. Soruyu daraltın veya ilgili makaleyi ingest edin."
)


def list_adapters() -> list[str]:
    """models/adapters altındaki TAM adapter'ları (config + ağırlık var) listele.

    Sıra EN YENİ ÖNCE (ağırlık dosyasının değişme zamanı): arayüz ilk elemanı varsayılan
    seçer → en son eğitilen adapter. Alfabetik sıra v8 varken v7'yi seçtiriyordu.
    """
    from app.config import get_settings

    d = get_settings().adapters_dir
    if not d.exists():
        return []
    found: list[tuple[float, str]] = []
    for p in d.glob("*"):
        weights = p / "adapter_model.safetensors"
        if (p / "adapter_config.json").exists() and weights.exists():
            found.append((weights.stat().st_mtime, p.name))
    found.sort(key=lambda t: (-t[0], t[1]))
    return [name for _, name in found]


def build_user_content(
    question: str, chunks: list[RetrievedChunk], budget: int = CONTEXT_CHAR_BUDGET
) -> str:
    """Soruyu eğitim verisindeki biçimle kur: parça yoksa çıplak soru, varsa BAĞLAM + SORU.

    Biçim `synthetic_qa_builder` ve `discipline_dataset` ile birebir aynıdır. Parçalar
    retrieval sırasıyla eklenir; toplam bağlam `budget` karakteri aşmaz.
    """
    parts: list[str] = []
    used = 0
    for c in chunks:
        remaining = budget - used
        if remaining <= 0:
            break
        text = c.text.strip()[:remaining]
        if text:
            parts.append(text)
            used += len(text)
    if not parts:
        return question
    context = "\n\n".join(parts)
    return f"BAĞLAM:\n{context}\n\nSORU: {question}"


def _source_dicts(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "paper_id": c.paper_id,
            "chunk_id": c.chunk_id,
            "title": c.title,
            "page": c.page_number,
            "distance": c.distance,
        }
        for c in chunks
    ]


def chat(
    question: str,
    adapter: str | None,
    *,
    max_tokens: int = 256,
    use_context: bool = False,
    top_k: int | None = None,
    retriever: Any = None,
    card_lookup: CardLookup | None = None,
) -> dict:
    """Adapter (veya base) ile cevap üret. adapter=None/"" → yalnız base model.

    Dönüş: {answer, adapter, base_model, used_context, llm_used, sources, sections}.
    Adapter yoksa FileNotFoundError. `use_context=True` ve retrieval boşsa model
    yüklenmez; `llm_used=False` ile açık "kaynak bulunamadı" döner. Kaynaklı modda
    `sections` 8 bölümlü hibrit cevaptır; kaynaksız modda boştur (şablon sahte güven vermesin).
    """
    from app.config import get_settings
    from app.lora.dataset_builder import SYSTEM_PROMPT
    from app.training.adapter_eval import _generate, _load_model, _resolve_base_model

    s = get_settings()
    adapter_dir: str | None = None
    if adapter:
        ap = s.adapters_dir / adapter
        if not (ap / "adapter_config.json").exists():
            raise FileNotFoundError(f"Adapter bulunamadı: {adapter}")
        adapter_dir = str(ap)

    # Base önceliği: adapter'ın kendi config'i → settings (küçük-model 4B'ye yüklenmesin).
    base = (_resolve_base_model(adapter_dir) if adapter_dir else None) or s.peft_base_model
    result: dict[str, Any] = {
        "adapter": adapter or "(base)",
        "base_model": base,
        "used_context": use_context,
        "sources": [],
        "sections": [],
    }

    user_content = question
    chunks: list[RetrievedChunk] = []
    cards: dict[str, dict] = {}
    if use_context:
        from app.brain.hybrid_answer import load_cards

        if retriever is None:
            from app.memory.reranking_retriever import RerankingRetriever

            retriever = RerankingRetriever()
        chunks = [c for c in retriever.retrieve(question, top_k=top_k) if c.text.strip()]
        if not chunks:
            return {**result, "answer": NO_SOURCE_ANSWER, "llm_used": False}
        if card_lookup is None:
            from app.memory.sqlite_store import SqliteStore

            card_lookup = SqliteStore().get_latest_knowledge_card
        # Kartlar PAHALI üretimden ÖNCE çekilir: DB hatası dakikalarca bekledikten sonra değil
        # hemen görünsün.
        cards = load_cards(chunks, card_lookup)
        user_content = build_user_content(question, chunks)
        result["sources"] = _source_dicts(chunks)

    key = f"{base}|{adapter_dir or ''}"
    with _LOCK:
        if _CACHE["key"] != key:
            log.info("lora-chat: model yükleniyor (key=%s) — ilk istek yavaş.", key)
            tok, model = _load_model(base, adapter_dir)
            _CACHE.update(key=key, tok=tok, model=model)
        answer = _generate(
            _CACHE["tok"],
            _CACHE["model"],
            user_content,
            max_new_tokens=max_tokens,
            system=SYSTEM_PROMPT,
        )

    if use_context:
        from app.brain.hybrid_answer import build_sections

        sections = build_sections(
            answer,
            chunks,
            cards,
            min_similarity=s.rag_abstain_min_similarity,
            min_margin=s.rag_abstain_min_margin,
        )
        result["sections"] = [sec.to_dict() for sec in sections]

    return {**result, "answer": answer, "llm_used": True}
