"""Web için tembel-yüklemeli PEFT adapter sohbet servisi.

Eğitilen LoRA adapter'ı (veya base) transformers/PEFT ile YEREL yükler, chat-template
uygular ve cevap üretir — Ollama gerektirmez. Model bir kez yüklenir, bellekte tutulur
(yeniden yükleme maliyeti CPU'da yüksek). Üretim greedy/deterministtir (Kural 6).

Eğitim ↔ çıkarım eşleşmesi: adapter, SYSTEM_PROMPT'lu ve çoğunlukla "BAĞLAM: … SORU: …"
biçimli örneklerle eğitildi. Sohbet de aynı sistem prompt'unu gönderir; `use_context=True`
ise soru korpustan retrieval ile getirilen parçalar eğitimdeki BİREBİR biçimde gömülür.
Retrieval boşsa model hiç çağrılmaz (Kural 7 — kaynak uydurma yok).

İstem biçimi ve bağlam bütçesi ADAPTER BAŞINA seçilir (`app/training/adapter_recipe.py`):
  * ``synthetic_qa`` (v11–v13; 1024 token, sentetik QA) → SYSTEM_PROMPT + "BAĞLAM: … SORU: …",
    bağlam karakter bütçesiyle kırpılır.
  * ``live_rag`` (v14; `moe30b_attn_long`, 6144 token, öz-damıtma) → canlı RAG istemi
    (`build_rag_prompt`, bayt-aynı); parçalar tokenizer ile ÖLÇÜLEREK
    `max_seq_length - max_tokens` bütçesine sığana dek en düşük sıralıdan bütün olarak düşülür.
    Kaynaksız modda eğitimdeki "plain" biçimi: SYSTEM_PROMPT + PLAIN_NO_SOURCE_NOTE.
  Reçete bulunamazsa eski (synthetic_qa, 2400 karakter) davranışa düşülür ve cevapta
  ``recipe_warning`` ile görünür uyarı döner.

Kaynaklı modda cevap HİBRİTTİR (`app/brain/hybrid_answer.py`): model yalnız Kısa Cevap'ı
yazar; Kaynaklar / Bağlam Kalitesi / Akademik Bulgu / Trading Hipotezi / Test Planı /
Riskler / Sonraki Adım retrieval, bilgi kartları ve kurallardan deterministik kurulur.

`adapter_eval._load_model`/`_generate`/`_resolve_base_model` tekrar kullanılır (eğitim
doğrulamasıyla AYNI yükleme/üretim yolu → tutarlılık). Üretim ağırdır (CPU'da dakikalar);
endpoint senkron `def` olmalı ki FastAPI'nin threadpool'unda koşsun, event loop'u bloklamasın.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from app.training.adapter_recipe import LIVE_RAG, AdapterRecipe, unknown_recipe

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
# Yukarıdaki türetmenin sabit kısmı: 1024 − 600 = 424 token (sistem + soru + cevap payı) ve
# ~4 karakter/token. Sentetik QA biçimli ama daha uzun max_seq_length'le eğitilmiş adapter'ın
# bütçesi aynı formülle ölçeklenir (1024 → tam 2400).
_SYNTHETIC_RESERVED_TOKENS = 424
_CHARS_PER_TOKEN = 4

# Reçetesiz base model sohbeti: eski davranış (uyarısız — adapter yok, eşleşecek reçete yok).
BASE_RECIPE = AdapterRecipe(
    prompt_format="synthetic_qa", max_seq_length=None, profile=None, source="base"
)

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


class ContextBudgetError(ValueError):
    """Bağlam, adapter'ın eğitim uzunluğuna (max_seq_length − max_tokens) sığmıyor."""


def synthetic_char_budget(max_seq_length: int | None) -> int:
    """Sentetik QA biçimi için bağlam karakter bütçesi (bilinmeyen/≤1024 → CONTEXT_CHAR_BUDGET)."""
    if not max_seq_length or max_seq_length <= 1024:
        return CONTEXT_CHAR_BUDGET
    return (max_seq_length - _SYNTHETIC_RESERVED_TOKENS) * _CHARS_PER_TOKEN


def count_prompt_tokens(tok: Any, system: str, user: str) -> int:
    """Chat şablonu UYGULANMIŞ istemin token sayısı (üretimdeki `_generate` ile aynı yol)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return len(tok(text)["input_ids"])


def fit_live_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    tok: Any,
    *,
    max_prompt_tokens: int,
    reorder: bool = True,
) -> tuple[str, str, list[RetrievedChunk]]:
    """Canlı RAG istemini (`build_rag_prompt`, bayt-aynı) token bütçesine sığdır.

    Eğitimde (öz-damıtma) parçalar BÜTÜN halde ve retrieval sırasıyla verildi; bu yüzden
    sığmazsa en düşük sıralı parça bütün olarak düşülür (kırpılmış parça eğitimde yoktu).
    Tek parça bile sığmıyorsa o parça sığacak en uzun önekine kısaltılır. Dönüş:
    (sistem, kullanıcı, modele GERÇEKTEN giden parçalar). Hiçbir bağlam sığmazsa
    ContextBudgetError.
    """
    from dataclasses import replace

    from app.brain.rag_answerer import build_rag_prompt

    def _fits(cs: list[RetrievedChunk]) -> tuple[str, str] | None:
        system, user = build_rag_prompt(question, cs, reorder=reorder)
        return (
            (system, user) if count_prompt_tokens(tok, system, user) <= max_prompt_tokens else None
        )

    used = list(chunks)
    while len(used) > 1:
        hit = _fits(used)
        if hit is not None:
            return hit[0], hit[1], used
        used.pop()  # en düşük sıralı parça
    if not used:
        raise ContextBudgetError("Bağlam parçası yok.")
    hit = _fits(used)
    if hit is not None:
        return hit[0], hit[1], used
    # Tek parça sığmıyor: sığan en uzun önek (ikili arama; token sayısı uzunlukla monoton).
    text = used[0].text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _fits([replace(used[0], text=text[:mid])]) is not None:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0 or not text[:lo].strip():
        raise ContextBudgetError(
            f"Bağlam token bütçesine sığmıyor (istem için {max_prompt_tokens} token kaldı). "
            "max_tokens değerini düşürün."
        )
    log.warning(
        "lora-chat: tek parça bile bütçeye sığmadı; %d/%d karaktere kısaltıldı.", lo, len(text)
    )
    cut = [replace(used[0], text=text[:lo])]
    system, user = build_rag_prompt(question, cut, reorder=reorder)
    return system, user, cut


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

    Dönüş: {answer, adapter, base_model, used_context, llm_used, sources, sections,
    prompt_format, recipe_source, recipe_warning, context_budget}.
    Adapter yoksa FileNotFoundError. `use_context=True` ve retrieval boşsa model
    yüklenmez; `llm_used=False` ile açık "kaynak bulunamadı" döner. Kaynaklı modda
    `sections` 8 bölümlü hibrit cevaptır; kaynaksız modda boştur (şablon sahte güven vermesin).
    """
    from app.config import get_settings
    from app.lora.dataset_builder import SYSTEM_PROMPT
    from app.training.adapter_eval import (
        _generate,
        _is_degenerate,
        _load_model,
        _resolve_base_model,
    )

    s = get_settings()
    adapter_dir: str | None = None
    recipe = BASE_RECIPE
    if adapter:
        ap = s.adapters_dir / adapter
        if not (ap / "adapter_config.json").exists():
            raise FileNotFoundError(f"Adapter bulunamadı: {adapter}")
        adapter_dir = str(ap)
        recipe = _recipe_for(ap, s)
    if recipe.warning:
        log.warning("lora-chat: %s", recipe.warning)
    live_rag = recipe.prompt_format == LIVE_RAG
    if live_rag and not recipe.max_seq_length:
        # Biçim biliniyor ama uzunluk bilinmiyor: ölçmeden bütçe uydurulmaz → eski davranış.
        recipe = unknown_recipe(adapter or "", "canlı RAG biçimi biliniyor ama max_seq_length yok")
        live_rag = False

    # Base önceliği: adapter'ın kendi config'i → settings (küçük-model 4B'ye yüklenmesin).
    base = (_resolve_base_model(adapter_dir) if adapter_dir else None) or s.peft_base_model
    result: dict[str, Any] = {
        "adapter": adapter or "(base)",
        "base_model": base,
        "used_context": use_context,
        "sources": [],
        "sections": [],
        "degenerate": False,
        "prompt_format": recipe.prompt_format,
        "recipe_source": recipe.source,
        "recipe_warning": recipe.warning,
        "context_budget": None,
    }

    user_content = question
    system = SYSTEM_PROMPT
    if live_rag:
        from app.training.self_distill import PLAIN_NO_SOURCE_NOTE

        # Eğitimdeki RAG'sız ("plain") öz-damıtma satırlarıyla aynı sistem istemi.
        system = SYSTEM_PROMPT + PLAIN_NO_SOURCE_NOTE
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
        if not live_rag:
            budget = synthetic_char_budget(recipe.max_seq_length)
            user_content = build_user_content(question, chunks, budget)
            result["context_budget"] = f"{budget} karakter"
            result["sources"] = _source_dicts(chunks)

    key = f"{base}|{adapter_dir or ''}"
    with _LOCK:
        if _CACHE["key"] != key:
            log.info("lora-chat: model yükleniyor (key=%s) — ilk istek yavaş.", key)
            # Önce eski referansları bırak: iki büyük model aynı anda RAM'de tutulmasın.
            _CACHE.update(key=None, tok=None, model=None)
            gc.collect()
            tok, model = _load_model(base, adapter_dir)
            _CACHE.update(key=key, tok=tok, model=model)
        if live_rag and use_context and recipe.max_seq_length:
            # Bütçe gerçek tokenizer'la ölçülür → model (tokenizer) yüklendikten sonra kurulur.
            # Eğitimde istem + cevap ≤ max_seq_length idi; cevap payı max_tokens kadar ayrılır.
            prompt_budget = recipe.max_seq_length - max_tokens
            system, user_content, chunks = fit_live_rag_prompt(
                question,
                chunks,
                _CACHE["tok"],
                max_prompt_tokens=prompt_budget,
                reorder=bool(getattr(s, "rag_reorder_context", True)),
            )
            result["context_budget"] = f"{prompt_budget} token"
            result["sources"] = _source_dicts(chunks)
            kept = {c.paper_id for c in chunks}  # düşülen parçaların kartı bölümlere girmesin
            cards = {pid: card for pid, card in cards.items() if pid in kept}
        answer = _generate(
            _CACHE["tok"],
            _CACHE["model"],
            user_content,
            max_new_tokens=max_tokens,
            system=system,
        )

    # Tekrar döngüsü (v8 eval'de ~%19): eval'deki AYNI dedektörle tespit edilir ve açıkça
    # bayraklanır; üretim ayarıyla (repetition_penalty vb.) gizlenmez — kusur ölçülebilir kalsın.
    degenerate = _is_degenerate(answer)
    if degenerate:
        log.warning("lora-chat: dejenere (tekrar döngüsü) çıktı — adapter=%s", adapter or "(base)")

    if use_context:
        from app.brain.hybrid_answer import build_sections

        sections = build_sections(
            answer,
            chunks,
            cards,
            min_similarity=s.rag_abstain_min_similarity,
            min_margin=s.rag_abstain_min_margin,
            degenerate=degenerate,
        )
        result["sections"] = [sec.to_dict() for sec in sections]

    return {**result, "answer": answer, "llm_used": True, "degenerate": degenerate}


def _recipe_for(adapter_path: Path, s: Any) -> AdapterRecipe:
    """Adapter'ın eğitim reçetesi; okuma hatası sohbeti düşürmez (uyarıyla eski davranış)."""
    from app.training.adapter_recipe import detect_adapter_recipe

    root = getattr(s, "root", None)
    registry = Path(root) / "registry" / "adapters" / "registry.jsonl" if root else None
    try:
        return detect_adapter_recipe(adapter_path, registry_path=registry)
    except Exception as exc:  # bozuk YAML vb. — sohbet yine çalışsın, ama görünür uyarıyla
        log.warning("lora-chat: reçete okunamadı (%s): %s", adapter_path.name, exc)
        return unknown_recipe(adapter_path.name, f"okuma hatası: {exc}")
