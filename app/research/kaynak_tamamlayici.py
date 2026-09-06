"""Kaynak Tamamlayıcı ajanı — okunamayan makaleye ÖN-KOŞUL/DESTEK makalesi bulur.

Bir makale anlaşılamıyorsa (anlama skoru eşiğin altında ya da kartı üretilemiyor) çoğu
zaman sebep RAG'da ön-koşul bilgisinin olmamasıdır. Bu ajan kaynak makalenin başlık + kart
terimlerinden sorgu üretir, arXiv'de arar, **çevrimdışı alaka kapısından** geçen adayları
indirir ve korpusa alır (ingest, LLM'siz).

Alaka kapısı çekirdektir — ölçüldü (2026-09-06): ham anahtar-kelime araması "path integral"
için parçacık fiziği, "risk management" için şehir sel yönetimi, "decision making" için
AI etiği getirdi. Kapı: kaynak terimleri ile adayın başlık+özeti arasında en az `min_ortak`
anlamlı ortak terim; yoksa indirilmez, raporda `alaka yok` diye görünür.

Sözleşme: eğitim BAŞLATMAZ; kart/skor üretmez (makale-okuyucu'ya bırakır); arXiv ID'si
korpusta olanı atlar; makale başına deneme sayar (defter) → sonsuz arama yok. Saf çekirdek
(`sorgu_uret`, `alaka_skoru`, `secilecek_makaleler`) çevrimdışı test edilir; ağ ve disk
yalnız `run_tamamlayici` içinde, enjekte edilebilir.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agents.runtime import log_step, track_agent_run
from app.memory.paper_indexer import is_boilerplate_title

AGENT_ID = "kaynak-tamamlayici"

# Sorgu/alaka için anlamsız ya da fazla genel kelimeler (EN + TR + arXiv boilerplate).
# "time/series/model/learning/network" gibi çok genel alan sözcükleri BİLEREK dışarıda:
# tek başlarına arXiv'in yarısıyla eşleşir, alaka kapısını anlamsızlaştırırlar.
_STOP: frozenset[str] = frozenset(
    [
        "about",
        "above",
        "after",
        "again",
        "against",
        "along",
        "among",
        "analysis",
        "another",
        "approach",
        "around",
        "based",
        "because",
        "before",
        "behind",
        "being",
        "below",
        "between",
        "beyond",
        "during",
        "either",
        "every",
        "except",
        "first",
        "further",
        "general",
        "generalized",
        "generic",
        "given",
        "however",
        "improved",
        "improving",
        "including",
        "instead",
        "introduction",
        "large",
        "learning",
        "machine",
        "method",
        "methods",
        "model",
        "models",
        "modern",
        "network",
        "networks",
        "neural",
        "novel",
        "other",
        "paper",
        "papers",
        "preprint",
        "problem",
        "problems",
        "proposed",
        "published",
        "recent",
        "research",
        "results",
        "review",
        "second",
        "series",
        "several",
        "should",
        "simple",
        "since",
        "small",
        "study",
        "studies",
        "their",
        "there",
        "these",
        "those",
        "through",
        "toward",
        "towards",
        "under",
        "unified",
        "using",
        "various",
        "versus",
        "where",
        "whether",
        "which",
        "while",
        "within",
        "without",
        "would",
        "applications",
        "application",
        "framework",
        "frameworks",
        "theory",
        "theoretical",
        "empirical",
        "evidence",
        "approachs",
        "ilişkin",
        "üzerine",
        "yöntem",
        "yöntemler",
        "model",
        "modeller",
        "analiz",
        "çalışma",
        "çalışması",
    ]
)
_TOKEN_RE = re.compile(r"[a-zçğıöşüâîû]{5,}")
_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def tokenize(text: str) -> list[str]:
    """Küçük harf, ≥5 harfli alfabetik jetonlar, stop-list dışı, çoğul 's' kırpılmış;
    ilk görülme sırası korunur, tekrar yok (sorgu üretimi sıraya duyarlı)."""
    seen: set[str] = set()
    out: list[str] = []
    for t in _TOKEN_RE.findall((text or "").lower()):
        if t.endswith("s") and len(t) > 5:
            t = t[:-1]
        if t in _STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def sorgu_uret(title: str | None, card_json: dict[str, Any] | None, *, max_terim: int = 8) -> str:
    """Kaynak makaleden arXiv sorgusu: başlık (şablonsa atlanır) + kart domain + ilk metodlar.

    Boş string → çağıran o makaleyi 'sorgu üretilemedi' diye atlar (uydurma sorgu yok).
    """
    terimler: list[str] = []
    if title and not is_boilerplate_title(title):
        terimler += tokenize(title)
    if card_json:
        terimler += tokenize(str(card_json.get("domain") or ""))
        for m in (card_json.get("methods") or [])[:3]:
            terimler += tokenize(str(m))
    # tekrar sil, sırayı koru
    tekil = list(dict.fromkeys(terimler))[:max_terim]
    return " ".join(tekil) if len(tekil) >= 2 else ""


def kaynak_terimleri(title: str | None, card_json: dict[str, Any] | None) -> set[str]:
    """Alaka karşılaştırması için kaynak terim kümesi (sorgudan daha geniş: özet de girer)."""
    s: set[str] = set(tokenize(title or "")) if not is_boilerplate_title(title or "") else set()
    if card_json:
        for k in ("domain", "main_claim", "trading_relevance"):
            s.update(tokenize(str(card_json.get(k) or "")))
        for m in card_json.get("methods") or []:
            s.update(tokenize(str(m)))
    return s


def alaka_skoru(kaynak: set[str], aday_baslik: str, aday_ozet: str) -> int:
    """Kaynak terimleri ile adayın başlık+özeti arasındaki anlamlı ortak terim sayısı."""
    aday = set(tokenize(f"{aday_baslik} {aday_ozet}"))
    return len(kaynak & aday)


@dataclass
class Makale:
    paper_id: str
    title: str | None
    card_json: dict[str, Any] | None
    skor: float | None  # anlama skoru (None = hesaplanmamış)


def secilecek_makaleler(
    makaleler: Iterable[Makale],
    *,
    esik: float,
    defter: dict[str, int],
    max_deneme: int,
    max_paper: int,
) -> list[Makale]:
    """Okunamayanları seç: kartı yok VEYA skoru eşiğin altında; deneme hakkı dolmamış.

    Deterministik sıra: en düşük skor önce, sonra skorsuzlar (kartsız), sonra paper_id.
    """
    aday = [
        m
        for m in makaleler
        if defter.get(m.paper_id, 0) < max_deneme
        and (m.card_json is None or (m.skor is not None and m.skor < esik))
    ]
    aday.sort(key=lambda m: (m.skor is None, m.skor if m.skor is not None else 0.0, m.paper_id))
    return aday[:max_paper]


@dataclass
class MakaleRaporu:
    paper_id: str
    baslik: str
    sorgu: str
    aday: int = 0
    alinan: list[dict[str, Any]] = field(default_factory=list)
    reddedilen: list[dict[str, str]] = field(default_factory=list)


def _defter_yolu(state_path: Path | None) -> Path:
    if state_path is not None:
        return state_path
    from app.config import get_settings

    return get_settings().state_dir / "kaynak_tamamlayici_state.json"


def _defter_oku(p: Path) -> dict[str, int]:
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in (raw.get("deneme") or {}).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def _defter_yaz(p: Path, defter: dict[str, int]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"deneme": defter}, ensure_ascii=False, indent=2), encoding="utf-8")


def _korpustaki_idler(raw_dir: Path) -> set[str]:
    if not raw_dir.exists():
        return set()
    ids: set[str] = set()
    for f in raw_dir.glob("*.pdf"):
        m = _ID_RE.search(f.name)
        if m:
            ids.add(m.group(1))
    return ids


def _makaleleri_topla(store: Any) -> list[Makale]:
    """SQLite'tan kaynak makaleler: başlık, canlı (rejected olmayan) kart, anlama skoru."""
    kartlar: dict[str, dict[str, Any]] = {}
    for c in store.list_cards():
        if c.get("review_status") == "rejected":
            continue
        pid = str(c.get("paper_id") or "")
        cj = c.get("card_json")
        if pid and isinstance(cj, dict) and pid not in kartlar:
            kartlar[pid] = cj
    out: list[Makale] = []
    for p in store.list_papers():
        row = store.get_comprehension_score(p.paper_id)
        skor = float(row.total_score) if row is not None else None
        out.append(Makale(p.paper_id, p.title, kartlar.get(p.paper_id), skor))
    return out


def run_tamamlayici(
    *,
    esik: float = 50.0,
    max_paper: int = 5,
    max_per_paper: int = 2,
    min_ortak: int = 3,
    max_deneme: int = 2,
    dry_run: bool = False,
    searcher: Callable[..., list[Any]] | None = None,
    fetcher: Callable[..., list[Any]] | None = None,
    ingest: Callable[[], Any] | None = None,
    store: Any = None,
    raw_dir: Path | None = None,
    state_path: Path | None = None,
    trigger_type: str = "manual",
) -> dict[str, Any]:
    """Tek koşu: seç → sorgu → ara → alaka kapısı → indir → ingest → defter."""
    with track_agent_run(AGENT_ID, trigger_type=trigger_type):
        from app.config import get_settings

        st = store
        if st is None:
            from app.memory.sqlite_store import SqliteStore

            st = SqliteStore()
        if searcher is None:
            from app.ingestion.arxiv_fetcher import search_arxiv

            searcher = search_arxiv
        if fetcher is None:
            from app.ingestion.arxiv_fetcher import fetch_arxiv_papers

            fetcher = fetch_arxiv_papers
        raw = raw_dir or get_settings().raw_pdf_dir
        defter_p = _defter_yolu(state_path)
        defter = _defter_oku(defter_p)
        korpus = _korpustaki_idler(raw)

        makaleler = _makaleleri_topla(st)
        secim = secilecek_makaleler(
            makaleler, esik=esik, defter=defter, max_deneme=max_deneme, max_paper=max_paper
        )
        log_step(
            f"Okunamayan aday: {len(secim)} makale (eşik {esik}, defter dolu olanlar hariç)",
            payload={"secilen": [m.paper_id for m in secim]},
        )

        raporlar: list[MakaleRaporu] = []
        indirilen = 0
        for m in secim:
            sorgu = sorgu_uret(m.title, m.card_json)
            r = MakaleRaporu(m.paper_id, m.title or "?", sorgu)
            raporlar.append(r)
            if not sorgu:
                r.reddedilen.append({"sebep": "sorgu üretilemedi (şablon başlık / boş kart)"})
                if not dry_run:
                    defter[m.paper_id] = defter.get(m.paper_id, 0) + 1
                continue
            try:
                adaylar = list(searcher(sorgu, max_results=max(3, max_per_paper * 3)))
            except Exception as exc:  # ağ yok → o makale için boş, koşu sürer
                log_step(f"arXiv araması başarısız ({m.paper_id}): {exc}", level="warning")
                adaylar = []
            r.aday = len(adaylar)
            kaynak = kaynak_terimleri(m.title, m.card_json)
            uygun: list[tuple[int, Any]] = []
            for a in adaylar:
                aid = str(getattr(a, "arxiv_id", ""))
                skor = alaka_skoru(
                    kaynak, str(getattr(a, "title", "")), str(getattr(a, "abstract", ""))
                )
                if _ID_RE.search(aid) and _ID_RE.search(aid).group(1) in korpus:  # type: ignore[union-attr]
                    r.reddedilen.append({"id": aid, "sebep": "zaten korpusta"})
                elif skor < min_ortak:
                    r.reddedilen.append(
                        {"id": aid, "sebep": f"alaka yok (ortak terim {skor} < {min_ortak})"}
                    )
                else:
                    uygun.append((skor, a))
            uygun.sort(key=lambda t: (-t[0], str(getattr(t[1], "arxiv_id", ""))))
            secilen = uygun[:max_per_paper]
            for _skor, a in uygun[max_per_paper:]:
                r.reddedilen.append(
                    {"id": str(getattr(a, "arxiv_id", "")), "sebep": "makale başına sınır"}
                )
            if dry_run:
                r.alinan = [
                    {
                        "id": str(a.arxiv_id),
                        "baslik": str(a.title)[:90],
                        "alaka": skor,
                        "indirildi": False,
                    }
                    for skor, a in secilen
                ]
                continue
            if secilen:
                sonuclar = fetcher(sorgu, entries=[a for _s, a in secilen], dest_dir=raw)
                skorlar = {str(a.arxiv_id): skor for skor, a in secilen}
                for fr in sonuclar:
                    ok = not getattr(fr, "skipped", True)
                    kayit = {
                        "id": str(fr.arxiv_id),
                        "baslik": str(fr.title)[:90],
                        "alaka": skorlar.get(str(fr.arxiv_id), 0),
                        "indirildi": ok,
                    }
                    if ok:
                        indirilen += 1
                        korpus.add(_ID_RE.search(str(fr.arxiv_id)).group(1))  # type: ignore[union-attr]
                        r.alinan.append(kayit)
                    else:
                        r.reddedilen.append(
                            {
                                "id": str(fr.arxiv_id),
                                "sebep": f"indirme: {getattr(fr, 'error', None) or 'atlandı'}",
                            }
                        )
            defter[m.paper_id] = defter.get(m.paper_id, 0) + 1
            log_step(
                f"{m.paper_id}: aday {r.aday}, alınan {len(r.alinan)}, red {len(r.reddedilen)}"
            )

        indekslenen = 0
        if not dry_run:
            _defter_yaz(defter_p, defter)
            if indirilen:
                if ingest is None:
                    from app.memory.paper_indexer import PaperIndexer

                    def ingest() -> Any:
                        return PaperIndexer().ingest_directory()

                try:
                    sonuc = ingest()
                    indekslenen = sum(1 for x in (sonuc or []) if not getattr(x, "skipped", False))
                except Exception as exc:
                    log_step(f"Ingest başarısız: {exc}", level="error")
        log_step(f"İndirilen {indirilen}, indekslenen {indekslenen}")
        return {
            "ok": True,
            "dry_run": dry_run,
            "secilen": len(secim),
            "indirilen": indirilen,
            "indekslenen": indekslenen,
            "makaleler": [
                {
                    "paper_id": r.paper_id,
                    "baslik": r.baslik[:80],
                    "sorgu": r.sorgu,
                    "aday": r.aday,
                    "alinan": r.alinan,
                    "reddedilen": r.reddedilen,
                }
                for r in raporlar
            ],
            "defter": dict(sorted(defter.items())),
        }
