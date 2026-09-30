"""Retrieval için sorgu çevirisi (TR → EN) — yalnız yerel Ollama.

Korpusun ~%89'u İngilizce; ``nomic-embed-text`` İngilizce-merkezli. Türkçe sorgu hem BM25'te
hem dense'te korpustaki tek Türkçe metinlere (proje kılavuzları) çekiliyor (ölçüm:
``app.evals.llm30_rag_probe``). Çeviri yalnız RETRIEVAL sorgusunu değiştirir; modele giden
soru metni aynı kalır.

Determinizm (Kural 6): temperature 0 + seed; sonuç ``storage/rag_query_translations.json``
önbelleğine (model + sorgu hash'i) yazılır → aynı sorgu aynı çeviriyi alır. Çeviri modeli
ayardan (``rag_translate_model``) gelir, ``HEKTOR_LLM_MODEL``'den DEĞİL: o bir LoRA'lı model
olabilir (``hektor-*``); adapter'lı model deney değişkenini kirletir → reddedilir.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Protocol

from app.config import get_settings

_TR_CHARS = re.compile(r"[çğışöüÇĞİŞÖÜ]")
_TR_WORDS = re.compile(r"\b(ve|bir|için|ile|nedir|nasıl|neden|değil|hangi|göre|olan)\b", re.I)
_LOCK = threading.Lock()

SYSTEM = (
    "You translate Turkish questions about quantitative finance, statistics and time series "
    "into English for a document search engine. Keep all formulas, numbers, symbols, variable "
    "names, code identifiers and established technical terms. Output ONLY the English "
    "translation, with no explanation, notes or quotes."
)


class TranslationError(RuntimeError):
    """Çeviri üretilemedi ya da model kabul edilmedi."""


class _Generator(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = ...,
        temperature: float = ...,
        max_tokens: int | None = ...,
        seed: int | None = ...,
    ) -> str: ...


def looks_turkish(text: str) -> bool:
    return bool(_TR_CHARS.search(text)) or len(_TR_WORDS.findall(text)) >= 2


def check_model(model: str) -> None:
    if model.lower().startswith("hektor-"):
        raise TranslationError(
            f"çeviri modeli adapter'lı olamaz: {model} — base Ollama modeli ver "
            "(HEKTOR_RAG_TRANSLATE_MODEL)"
        )


def cache_path() -> Path:
    return get_settings().root / "storage" / "rag_query_translations.json"


def _key(model: str, query: str) -> str:
    return hashlib.sha256(f"{model}\x00{SYSTEM}\x00{query}".encode()).hexdigest()


def _read_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _clean(text: str) -> str:
    out = re.sub(r"^\s*(english|translation)\s*:\s*", "", text.strip(), flags=re.I)
    return out.strip().strip("\"'“”").strip()


def translate_query(
    query: str,
    *,
    model: str | None = None,
    llm: _Generator | None = None,
    path: Path | None = None,
    seed: int = 42,
) -> str:
    """Türkçe görünen sorguyu İngilizceye çevir (önbellekli). Türkçe değilse aynen döner."""
    if not looks_turkish(query):
        return query
    model = model or get_settings().rag_translate_model
    check_model(model)
    path = path or cache_path()
    key = _key(model, query)
    with _LOCK:
        hit = _read_cache(path).get(key)
    if hit:
        return hit["en"]
    if llm is None:
        from app.brain.local_llm import LocalLLM

        llm = LocalLLM(model=model)
    raw = llm.generate(
        query, system=SYSTEM, temperature=0.0, seed=seed, max_tokens=max(256, len(query))
    )
    en = _clean(raw)
    if not en or looks_turkish(en):
        raise TranslationError(f"çeviri başarısız ya da hâlâ Türkçe: {en[:120]!r}")
    with _LOCK:
        cache = _read_cache(path)
        cache[key] = {"model": model, "tr": query, "en": en}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    return en
