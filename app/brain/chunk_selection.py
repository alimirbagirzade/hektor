"""İçerik chunk seçimi — kitap ön sayfalarını (kapak, telif, içindekiler, şekil listesi) ele.

Kitabın ilk chunk'ları kapak, telif, içindekiler ve şekil listesidir (ölçüldü: López de
Prado #1-#7 "Table 1.1 11 Equation 26 23…"). Bunlardan üretilen sentetik QA çekimser/boş,
bilgi kartı başlıksız çıkıyordu. Sentetik QA (``synthetic_qa_builder``) ve bilgi kartı
(``knowledge_card_builder``) aynı ölçütü buradan kullanır.
"""

from __future__ import annotations

import re
from typing import Any

FRONT_MATTER_RE = re.compile(
    r"\b(?:table of contents|contents|copyright|all rights reserved|isbn|"
    r"library of congress|printed in|published by|acknowledg\w*|praise for|oceanofpdf)\b",
    re.I,
)
MIN_CONTENT_CHARS = 400
MAX_NUMERIC_WORD_SHARE = 0.3


def is_content_chunk(text: str) -> bool:
    """Chunk gerçek içerik mi (ön sayfa / içindekiler / şekil listesi değil)?"""
    flat = " ".join((text or "").split())
    if len(flat) < MIN_CONTENT_CHARS or FRONT_MATTER_RE.search(flat[:300]):
        return False
    words = flat.split()
    numeric = sum(1 for w in words if any(ch.isdigit() for ch in w))
    return numeric / len(words) < MAX_NUMERIC_WORD_SHARE


def select_content_chunks(chunks: list[Any], max_chunks: int) -> list[Any]:
    """İçerik chunk'larından en çok `max_chunks` tanesini belgeye EŞİT aralıkla seç (determinist).

    Hiç içerik chunk'ı yoksa (kısa belge / test) boş olmayan chunk'lara aynı yayılım uygulanır.
    """
    pool = [c for c in chunks if is_content_chunk(getattr(c, "text", ""))] or [
        c for c in chunks if len((getattr(c, "text", "") or "").strip()) >= 40
    ]
    if max_chunks <= 0 or len(pool) <= max_chunks:
        return pool
    step = len(pool) / max_chunks
    return [pool[int(i * step + step / 2)] for i in range(max_chunks)]
