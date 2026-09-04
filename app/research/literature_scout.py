"""literature_scout.py — konu-paketli periyodik literatür KEŞİF ajanı.

arXiv'de dört konu paketinde **yöntem** araması yapar (trading sinyali değil),
deterministik/offline heuristikle puanlar, yeni adayları konu-bazlı izleme defterine
işler ve isteğe bağlı olarak PDF'leri "gelen kutusu" klasörüne indirir.

Konu paketleri:
  - ``rag``           → retrieval/reranking/chunking/GraphRAG/RAFT …
  - ``lora``          → LoRA/DoRA/rsLoRA/QLoRA/kurriculum/veri karışımı/KL-reg …
  - ``rlm``           → muhakeme, çok-adımlı retrieval, iddia doğrulama, çekimserlik …
  - ``math-physics``  → stokastik süreç, entropi, rastgele matris, ağır kuyruk, istatistik …

**Ne YAPMAZ (kasıtlı):** RAG'a ingest ETMEZ, eğitim BAŞLATMAZ, reçete/hiperparametre
DEĞİŞTİRMEZ. `indirmek ≠ RAG'a almak ≠ eğitmek` — üçü ayrı kapıdır (CLAUDE.md Kural 8).
Bulunanlar yalnız *aday*dır; RAG'a alma ve eğitim İNSAN elindedir.

**Maliyet:** Claude/LLM kotası KULLANMAZ — yalnız arXiv genel API'si + anahtar-kelime
skoru (CLAUDE.md "API ASLA YOK" kısıtına uyar).

**Çevrimdışı/test:** `searcher` ve `downloader` enjekte edilebilir; ağ yoksa graceful
(boş döner, çökmez). Skor ve sıralama deterministiktir.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.ingestion.arxiv_fetcher import ArxivEntry, search_arxiv
from app.research.rag_trend_scanner import (
    _RELEVANCE_TERMS as _RAG_TERMS,
)
from app.research.rag_trend_scanner import (
    DEFAULT_QUERIES as _RAG_QUERIES,
)
from app.research.rag_trend_scanner import (
    TrendCandidate,
    append_candidates,
    repo_root,
    scan_rag_trends,
)

logger = logging.getLogger(__name__)

_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class TopicPack:
    """Bir konu paketi: arama sorguları + alaka anahtar kelimeleri."""

    key: str
    label: str
    queries: tuple[str, ...]
    terms: tuple[str, ...]


TOPIC_PACKS: dict[str, TopicPack] = {
    "rag": TopicPack(
        key="rag",
        label="RAG / retrieval",
        queries=_RAG_QUERIES,
        terms=_RAG_TERMS,
    ),
    "lora": TopicPack(
        key="lora",
        label="LoRA / SFT / eğitim yöntemleri",
        queries=(
            "LoRA low-rank adaptation parameter efficient fine-tuning",
            "DoRA rsLoRA LoRA+ adapter rank stabilization",
            "QLoRA quantized low-rank fine-tuning",
            "instruction tuning data mixture curriculum learning",
            "NEFTune noisy embedding fine-tuning regularization",
            "catastrophic forgetting KL regularized supervised fine-tuning",
            "loss masking assistant only training language model",
            "small language model domain adaptation distillation",
        ),
        terms=(
            "lora",
            "low-rank",
            "adapter",
            "fine-tun",
            "finetun",
            "peft",
            "instruction tun",
            "sft",
            "curriculum",
            "distill",
            "quantiz",
            "forgetting",
            "regulariz",
            "data mixture",
            "rank",
            "masking",
        ),
    ),
    "rlm": TopicPack(
        key="rlm",
        label="RLM / muhakeme + doğrulama",
        queries=(
            "reasoning language model multi-step retrieval",
            "claim verification attribution grounding evidence",
            "abstention calibration uncertainty language model",
            "self-consistency verification reasoning chain",
            "hallucination detection benchmark factuality",
            "iterative retrieval multi-hop question answering",
        ),
        terms=(
            "reason",
            "multi-hop",
            "multi-step",
            "claim",
            "verif",
            "attribut",
            "ground",
            "abstain",
            "abstention",
            "calibrat",
            "uncertain",
            "self-consistency",
            "halluc",
            "factual",
            "evidence",
            "citation",
        ),
    ),
    "math-physics": TopicPack(
        key="math-physics",
        label="Matematiksel fizik + olasılık/istatistik",
        queries=(
            "stochastic processes Markov regime switching",
            "entropy information theory time series",
            "random matrix theory correlation matrix cleaning",
            "heavy tailed distributions extreme value theory",
            "statistical inference multiple testing false discovery",
            "statistical physics of complex systems scaling",
        ),
        terms=(
            "stochastic",
            "markov",
            "entropy",
            "information theor",
            "random matrix",
            "eigenvalue",
            "heavy tail",
            "extreme value",
            "probabilit",
            "statistic",
            "inference",
            "false discovery",
            "hypothesis test",
            "scaling",
            "ergodic",
            "diffusion",
        ),
    ),
}


@dataclass
class FoundPaper:
    """Bir konu paketinde bulunan (ve isteğe bağlı indirilmiş) aday makale."""

    topic: str
    candidate: TrendCandidate
    pdf_path: Path | None = None
    downloaded: bool = False  # True: bu turda indirildi, False: zaten vardı / indirilmedi


@dataclass
class ScoutReport:
    """Bir tarama turunun özeti (CLI/web/manifest bunu gösterir)."""

    ran_at: str
    inbox: Path
    found: list[FoundPaper] = field(default_factory=list)
    watchlist_added: dict[str, int] = field(default_factory=dict)

    @property
    def downloaded_count(self) -> int:
        return sum(1 for f in self.found if f.downloaded)


def resolve_topics(topics: Sequence[str] | None) -> list[TopicPack]:
    """İstenen konu anahtarlarını paketlere çevir; None/boş → HEPSİ (stabil sıra)."""
    if not topics:
        return [TOPIC_PACKS[k] for k in sorted(TOPIC_PACKS)]
    out: list[TopicPack] = []
    for t in topics:
        pack = TOPIC_PACKS.get(t.strip().lower())
        if pack is None:
            raise ValueError(f"Bilinmeyen konu paketi: {t!r} (geçerli: {sorted(TOPIC_PACKS)})")
        if pack not in out:
            out.append(pack)
    return out


def inbox_root() -> Path:
    """PDF 'gelen kutusu' kökü.

    Öncelik: ayar `scout_inbox_dir` (.env: ACHILLES_SCOUT_INBOX_DIR) → yoksa repo-içi
    `data/literature_inbox/`. Varsayılan repo-içidir; test/CI Desktop'a YAZMAZ.
    """
    try:
        from app.config import get_settings

        configured = (getattr(get_settings(), "scout_inbox_dir", "") or "").strip()
    except Exception as exc:  # ayar okunamasa bile tarama çökmesin
        logger.debug("scout_inbox_dir okunamadı: %s", exc)
        configured = ""
    if configured:
        return Path(configured).expanduser()
    return repo_root() / "data" / "literature_inbox"


def watchlist_dir_default() -> Path:
    """İzleme defterlerinin varsayılan kökü (`docs/egitim/`)."""
    return repo_root() / "docs" / "egitim"


def watchlist_path_for(topic: str, base: Path | None = None) -> Path:
    """Konu başına izleme defteri.

    `rag` geriye uyum için mevcut `rag-watchlist.md` dosyasını kullanır. `base`
    enjekte edilebilir → testler repo'ya YAZMAZ (tmp_path ile izole edilir).
    """
    root = base or watchlist_dir_default()
    name = "rag-watchlist.md" if topic == "rag" else f"{topic}-watchlist.md"
    return root / name


def _safe_name(arxiv_id: str) -> str:
    return _SAFE_RE.sub("_", arxiv_id)


def _default_downloader(url: str) -> bytes:
    """Gerçek indirici (ağ). Testlerde enjekte edilerek devre dışı bırakılır."""
    import httpx

    with httpx.Client(timeout=90, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return bytes(resp.content)


def download_candidate(
    cand: TrendCandidate,
    dest_dir: Path,
    downloader: Callable[[str], bytes] = _default_downloader,
) -> tuple[Path | None, bool]:
    """Bir adayın PDF'ini `dest_dir`e indir.

    Idempotent: dosya varsa yeniden indirmez. İçerik `%PDF` ile başlamıyorsa yazmaz
    (bozuk/anti-bot HTML sayfası kaydedilmesin). Hata hâlinde (None, False) döner —
    tek makale patlasa da tur devam eder.

    Returns:
        (yerel yol | None, bu turda indirildi mi)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"arxiv_{_safe_name(cand.arxiv_id)}.pdf"
    if dest.exists():
        return dest, False
    try:
        blob = downloader(_PDF_URL.format(arxiv_id=cand.arxiv_id))
    except Exception as exc:
        logger.warning("PDF indirilemedi (%s): %s", cand.arxiv_id, exc)
        return None, False
    if not blob.startswith(b"%PDF"):
        logger.warning("PDF değil, atlandı: %s", cand.arxiv_id)
        return None, False
    dest.write_bytes(blob)
    return dest, True


def _manifest_path(inbox: Path) -> Path:
    return inbox / "BULUNANLAR.md"


def write_manifest(inbox: Path, report: ScoutReport) -> Path:
    """Gelen kutusuna insan-okur özet yaz (ne bulundu, nerede, neden ilgili).

    Ekleme (append) mantığı: her tur kendi başlığı altında eklenir; önceki turlar durur.
    """
    path = _manifest_path(inbox)
    lines: list[str] = []
    if not path.exists():
        lines += [
            "# Bulunan makaleler — literatür keşif ajanı",
            "",
            "> Bu klasöre ajan yalnızca **aday** makale indirir.",
            "> **İndirilmiş ≠ RAG'a alınmış ≠ eğitilmiş.** RAG'a alma ve eğitim SENDE",
            "> (CLAUDE.md Kural 8). İstediğini seç, kalanını sil.",
            "",
        ]
    lines += [f"## Tur: {report.ran_at}", ""]
    if not report.found:
        lines += ["_Bu turda eşik üstü yeni aday bulunamadı._", ""]
    else:
        lines += [
            "| Konu | Skor | arXiv | Başlık | Yerel dosya |",
            "|---|---|---|---|---|",
        ]
        for f in report.found:
            c = f.candidate
            local = f.pdf_path.name if f.pdf_path else "—"
            title = c.title.replace("|", "/").replace("\n", " ")[:80]
            lines.append(
                f"| {f.topic} | {c.score} | "
                f"[{c.arxiv_id}](https://arxiv.org/abs/{c.arxiv_id}) | {title} | {local} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def run_scout(
    topics: Sequence[str] | None = None,
    max_per_query: int = 8,
    min_score: int = 2,
    top_n_download: int = 3,
    download: bool = True,
    inbox: Path | None = None,
    watchlist_dir: Path | None = None,
    searcher: Callable[[str, int], list[ArxivEntry]] = search_arxiv,
    downloader: Callable[[str], bytes] = _default_downloader,
    today: str | None = None,
) -> ScoutReport:
    """Bir keşif turu: tara → izleme defterine işle → (opsiyonel) PDF indir → özet yaz.

    Args:
        topics: Konu anahtarları; None → hepsi.
        max_per_query: Sorgu başına arXiv sonucu.
        min_score: Heuristik alaka eşiği.
        top_n_download: Konu başına en çok kaç PDF indirilsin (gelen kutusu boğulmasın).
        download: False → yalnız defter güncellenir, dosya indirilmez.
        inbox: PDF kökü; None → `inbox_root()`.
        watchlist_dir: İzleme defteri kökü; None → `docs/egitim/` (testler tmp_path verir).
        searcher/downloader: test/çevrimdışı için enjekte edilir.
        today: Tarih damgası (test determinizmi).

    Returns:
        Turun `ScoutReport`u. Ağ yoksa boş rapor (çökmez).
    """
    packs = resolve_topics(topics)
    stamp = today or dt.date.today().isoformat()
    root = inbox or inbox_root()
    report = ScoutReport(ran_at=stamp, inbox=root)

    for pack in packs:
        cands = scan_rag_trends(
            queries=pack.queries,
            max_per_query=max_per_query,
            min_score=min_score,
            searcher=searcher,
            terms=pack.terms,
        )
        if not cands:
            continue
        # 1) İzleme defteri (idempotent; zaten bilinen id atlanır)
        try:
            wl_path = watchlist_path_for(pack.key, watchlist_dir)
            # Kök yoksa oluştur: yeni makinede/yeni konuda defter sessizce yazılmadan
            # kalmasın (append_candidates dizini kendisi açmaz).
            wl_path.parent.mkdir(parents=True, exist_ok=True)
            added = append_candidates(wl_path, cands, today=stamp)
            report.watchlist_added[pack.key] = added
        except Exception as exc:  # defter yazılamasa bile tur sürsün
            logger.warning("Watchlist yazılamadı (%s): %s", pack.key, exc)
            report.watchlist_added[pack.key] = 0
        # 2) En alakalı ilk N adayı indir (deterministik sıra: scan_rag_trends sıralı döner)
        for cand in cands[: max(0, top_n_download)]:
            pdf_path: Path | None = None
            was_new = False
            if download:
                pdf_path, was_new = download_candidate(cand, root / pack.key, downloader)
            report.found.append(
                FoundPaper(
                    topic=pack.key,
                    candidate=cand,
                    pdf_path=pdf_path,
                    downloaded=was_new,
                )
            )

    if report.found or download:
        try:
            write_manifest(root, report)
        except Exception as exc:  # özet yazılamasa da rapor döner
            logger.warning("BULUNANLAR.md yazılamadı: %s", exc)
    return report
