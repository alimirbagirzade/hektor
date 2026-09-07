"""Makale Okuyucu ajanı — korpustaki HER makaleyi "okunmuş" hâle getirir.

"Okunmuş" = (1) indekslenmiş (chunk + vektör), (2) içerikli bilgi kartı var,
(3) anlama skoru hesaplanmış. Arka plandaki RAG öğrenme döngüsü bunu tur başına küçük
bütçelerle (3 kart / 5 skor) ve yalnız açıksa yapar; bu ajan aynı makineyi **elle
tetiklenen, bütçesi belirlenen tek bir koşuda** çalıştırır — "tüm PDF'leri okut" komutu.

Yeniden yazmaz: kart üretimi ve skorlama `RagLearningLoop`'un kendi adımlarıdır
(`_build_missing_cards` / `_score_missing`); burada yalnız planlanır, sayılır ve izlenir
(`track_agent_run` → 10 · AGENTS panelinde koşu, 15 · AJAN HARİTASI'nda düğüm).

Kural 8: eğitim BAŞLATMAZ. Kural 7: boş kart yazılmaz (builder), skor uydurulmaz.
Maliyet: kart + LLM'li skor yerel Ollama'da ~40 sn/çağrı; bütçeler bu yüzden açık parametre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.runtime import log_step, track_agent_run

AGENT_ID = "makale-okuyucu"


@dataclass
class ReaderPlan:
    """Korpusun okunmuşluk fotoğrafı (saf veri; testler elle kurar)."""

    toplam: int
    kartsiz: list[str]
    skorsuz: list[str]  # kartı var ama anlama skoru yok

    @property
    def okunmus(self) -> int:
        return self.toplam - len(set(self.kartsiz) | set(self.skorsuz))

    @property
    def yuzde(self) -> float:
        return round(100.0 * self.okunmus / self.toplam, 1) if self.toplam else 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "toplam": self.toplam,
            "okunmus": self.okunmus,
            "yuzde": self.yuzde,
            "kartsiz": len(self.kartsiz),
            "skorsuz": len(self.skorsuz),
        }


def plan_reading(paper_ids: list[str], carded_ids: set[str], scored_ids: set[str]) -> ReaderPlan:
    """Saf planlayıcı: kimin kartı, kimin skoru eksik? Skor yalnız KARTLI makalede anlamlıdır
    (skorlayıcı kart alanlarına bakar) → kartsız makale 'skorsuz' listesine GİRMEZ."""
    kartsiz = [p for p in paper_ids if p not in carded_ids]
    skorsuz = [p for p in paper_ids if p in carded_ids and p not in scored_ids]
    return ReaderPlan(toplam=len(paper_ids), kartsiz=kartsiz, skorsuz=skorsuz)


def topla_plan(store: Any = None) -> ReaderPlan:
    """Canlı fotoğraf (SQLite; LLM yok)."""
    from app.memory.sqlite_store import SqliteStore

    st = store or SqliteStore()
    ids = [p.paper_id for p in st.list_papers()]
    carded = {p for p in ids if st.has_knowledge_card(p)}
    scored = {p for p in carded if st.get_comprehension_score(p) is not None}
    return plan_reading(ids, carded, scored)


def run_reader(
    *,
    cards: int = 20,
    scores: int = 20,
    use_llm_score: bool = True,
    dry_run: bool = False,
    loop: Any = None,
    store: Any = None,
    trigger_type: str = "manual",
) -> dict[str, Any]:
    """Tek koşu: en çok `cards` kart üret, en çok `scores` skor hesapla; öncesi/sonrası say.

    Döngüye GİRMEZ (kalıcı olarak kartlanamayan makale sonsuz tekrar üretmesin); kalan iş
    `sonra` alanında raporlanır, çağıran isterse yeniden tetikler.

    `kart` alanı GERÇEKTEN üretilen (içerikli → kaydedilmiş) kart sayısıdır, deneme sayısı
    değil (CLAUDE.md kural 2). Boş/parse edilemez LLM yanıtı kart SAYILMAZ; o makale
    `kalan_kartsiz` içinde görünmeye devam eder. `cards`/`scores` bütçesi deneme sayısını
    kelepçeler, bu yüzden `kart <= cards`'tır ama eşit olmayabilir. Kalıcı olarak
    kartlanamayan makale döngünün deneme tavanına ulaşınca atlanır (sıra diğerlerine geçer).
    """
    with track_agent_run(AGENT_ID, trigger_type=trigger_type):
        once = topla_plan(store)
        log_step(
            f"Okunmuşluk: {once.okunmus}/{once.toplam} (%{once.yuzde}) — "
            f"kartsız {len(once.kartsiz)}, skorsuz {len(once.skorsuz)}",
            payload=once.to_dict(),
        )
        if dry_run:
            return {"ok": True, "dry_run": True, "once": once.to_dict(), "kart": 0, "skor": 0}

        if loop is None:
            from app.research.rag_learning_loop import RagLearningLoop

            loop = RagLearningLoop()
        # Skorda LLM kullanımı döngünün kalıcı ayarına dokunmadan bu koşuya özgü seçilir
        # (set_config diske yazar; burada yalnız bellek).
        loop._state.score_use_llm = use_llm_score

        kart = loop._build_missing_cards(max(0, cards)) if once.kartsiz else 0
        # "üretildi" = içerikli + kaydedilmiş kart (boş yanıt sayılmaz); denenip içerik
        # gelmeyen makaleler kartsız kalır ve `kalan_kartsiz`'da raporlanır.
        log_step(f"İçerikli kart üretildi: {kart}")
        skor = loop._score_missing(max(0, scores))
        log_step(f"Anlama skoru hesaplandı: {skor}")

        sonra = topla_plan(store)
        log_step(
            f"Okunmuşluk: {sonra.okunmus}/{sonra.toplam} (%{sonra.yuzde}) — "
            f"hâlâ kartsız: {len(sonra.kartsiz)}",
            payload=sonra.to_dict(),
        )
        return {
            "ok": True,
            "dry_run": False,
            "kart": kart,
            "skor": skor,
            "once": once.to_dict(),
            "sonra": sonra.to_dict(),
            "kalan_kartsiz": sonra.kartsiz[:20],
            "kalan_skorsuz": sonra.skorsuz[:20],
        }
