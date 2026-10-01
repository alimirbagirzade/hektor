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
# Tek başına yeterli: İngilizcede geçmeyen Türkçe soru/bağ kelimeleri — Türkçe harfsiz (ASCII)
# yazımlar dahil (Kademe 2 F3-4: "Kernel makinesi tanimi nedir?" çevrilmiyordu; v14
# öz-damıtma iş listesinde 25/800 soru).
_TR_STRONG = re.compile(
    r"\b(nedir|nelerdir|neler|neyi|neye|nasil|nasıl|neden|nicin|niçin|hangi|hangisi|icin|"
    r"degil|gore|midir|mıdır|mudur|müdür|veya|ile)\b",
    re.I,
)
_LOCK = threading.Lock()
_TRANSLATION_MAX_RATIO = 2.5  # çıktı/girdi uzunluğu; ölçülen medyan 1.12 (Kademe 2 F3-9)

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
    return (
        bool(_TR_CHARS.search(text))
        or bool(_TR_STRONG.search(text))
        or len(_TR_WORDS.findall(text)) >= 2
    )


def _still_turkish(src: str, out: str) -> bool:
    """Çeviri çıktısı hâlâ Türkçe mi? Girdiyle ORTAK kelimeler (özel adlar: "Borsa İstanbul",
    "Gödel") sayılmaz — çevirmene bunları koru dendi (Kademe 2 F3-5)."""
    # Yalnız ÖZEL AD görünümlü ortak kelimeler (büyük harfle başlayan ya da Türkçe harf taşıyan)
    # düşülür; "ve", "bir" gibi bağlaçlar girdide de geçse Türkçe kanıtı olarak kalır.
    names = {w for w in re.findall(r"\w+", src) if w[:1].isupper() or _TR_CHARS.search(w)}
    rest = " ".join(w for w in re.findall(r"\w+", out) if w not in names)
    return looks_turkish(rest)


def _answer_like(src: str, out: str) -> bool:
    """Çevirmen soruyu çevirmek yerine CEVAPLADI mı? (Kademe 2 F3-9: alt maddeli sorularda
    2/17 gerçek vakada çıktı girdinin 2.3-3.5 katıydı ve cevap retrieval sorgusu oldu.)"""
    too_long = len(out) > _TRANSLATION_MAX_RATIO * len(src) + 60
    return too_long or out.count("\n") > src.count("\n") + 2


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
    out = re.sub(r"^\s*(english|translation)\s*:\s*", "", text.strip(), flags=re.I).strip()
    # Yalnız TÜM çıktıyı saran tırnak soyulur; sondaki meşru tırnak ("…'alpha'") korunur.
    if len(out) >= 2 and out[0] in "\"'“" and out[-1] in "\"'”":
        out = out[1:-1].strip()
    return out


def _write_cache(path: Path, key: str, entry: dict[str, str]) -> None:
    """Önbelleğe atomik ekle. Süreçler arası güvenli değildir (kilit iş parçacığı düzeyinde);
    bu yüzden: süreç başına geçici dosya, Windows paylaşım ihlalinde kısa yeniden deneme,
    okunamayan önbelleğin üzerine YAZILMAZ (Kademe 2 F3-6: eskiden {} okunup silinirdi).
    Yazılamazsa çeviri yine döner — önbellek yardımcıdır."""
    import os
    import time

    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(cache, dict):
            return
    else:
        cache = {}
    cache[key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        for attempt in range(5):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
    except OSError:
        pass
    tmp.unlink(missing_ok=True)


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
    # Önbellekteki eski "cevap gibi" girdiler (F3-9 öncesi yazıldı) isabet sayılmaz.
    if hit and not _answer_like(query, hit["en"]):
        return hit["en"]
    if llm is None:
        from app.brain.local_llm import LocalLLM

        llm = LocalLLM(model=model)
    en = ""
    # 2. deneme yalnız cevap-benzeri çıktıda: istem açıkça "cevaplama" der (sistem istemi ve
    # önbellek anahtarı değişmez).
    for prompt in (query, f"Translate this question into English. Do NOT answer it.\n\n{query}"):
        raw = llm.generate(
            prompt, system=SYSTEM, temperature=0.0, seed=seed, max_tokens=max(256, len(query))
        )
        en = _clean(raw)
        if not en or _still_turkish(query, en):
            raise TranslationError(f"çeviri başarısız ya da hâlâ Türkçe: {en[:120]!r}")
        if not _answer_like(query, en):
            break
    else:
        raise TranslationError(f"çeviri değil cevap gibi (uzunluk {len(en)}/{len(query)})")
    with _LOCK:
        _write_cache(path, key, {"model": model, "tr": query, "en": en})
    return en
