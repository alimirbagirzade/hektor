"""RAG Öğrenme Döngüsü — otonom korpus büyütme + öğrenme arka plan döngüsü.

"RAG eğitimi" bu projede model ağırlığı eğitmek DEĞİL; RAG'ın korpusu öğrenme
çevrimidir. Döngü her turda (kullanıcı etkin kıldıysa):

  1. [opsiyonel] Kayıtlı arXiv sorgularından yeni makale çek → indeksle.
  2. Kartı olmayan makalelere bilgi kartı üret (LLM).
  3. Kartı olup skoru olmayan makaleleri comprehension ile skorla.
  4. RAG ustalık panosunu (türetilmiş %) güncelle.

KULLANICI ONAYI (CLAUDE.md kural 8 ruhu): döngü VARSAYILAN KAPALI; yalnız web'den
etkinleştirilince çalışır. Ağır LLM/ağ kullanımı otomatik başlamaz. LoRA eğitimi
sürerken döngü kendini DURAKLATIR (RAM/LLM çakışmasını önlemek için — bkz.
rag_mastery notu: ustalık paneli ağır LLM çağırmaz, ama kart/skor üretimi çağırır).

State dosyası: storage/rag_learning_state.json (config + çalışma durumu birlikte).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agents.runtime import log_step, tracked

log = logging.getLogger(__name__)


def _state_file() -> Path:
    """Durum dosyası — CWD'ye değil `settings.state_dir`'e bağlı."""
    from app.config import get_settings

    return get_settings().state_dir / "rag_learning_state.json"


# Boş kart rebuild deneme tavanı: builder içerik üretemeyen makaleyi (ör. kaynak metni
# bozuk/0KB) sonsuza dek denemesin → bu kadar denemeden sonra bırakılır (boşa LLM yok).
_MAX_REBUILD_ATTEMPTS = 3


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


_ALNUM_RE = re.compile(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü]")


def is_substantive_card(card_json: dict[str, Any]) -> bool:
    """Kart ANLAMLI içerik taşıyor mu — dejenere ('...', tek kelime) kartları eler.

    Yalnız non-empty bakmak yetmez: LLM ön-madde/çekişmede `"..."` döndürebilir ve bu
    onaylanırsa coverage çöple şişer (v5-tipi risk). Anlamlı (alfanümerik) karakter say:
    title ≥ 8 ve main_claim ≥ 40 → gerçek içerik kabul edilir.
    """
    if not isinstance(card_json, dict):
        return False
    title = _ALNUM_RE.sub("", str(card_json.get("title") or ""))
    main_claim = _ALNUM_RE.sub("", str(card_json.get("main_claim") or ""))
    return len(title) >= 8 and len(main_claim) >= 40


@dataclass
class RagLoopState:
    """Döngü ayarları (kalıcı) + anlık çalışma durumu."""

    # --- ayarlar (kullanıcı kontrolü; kalıcı) ---
    enabled: bool = False
    interval_min: int = 30  # turlar arası bekleme
    fetch_enabled: bool = True  # her turda yeni makale çekmeyi dene
    fetch_interval_hours: int = 24  # arXiv'e nazik ol: çekim turu seyrek
    max_fetch_per_cycle: int = 5  # çekim turunda en çok kaç makale
    cards_per_cycle: int = 3  # turda en çok kaç kart (CPU sınırı)
    scores_per_cycle: int = 5  # turda en çok kaç skor
    score_use_llm: bool = True  # comprehension skorunda LLM kullan (yavaş ama kaliteli)
    rebuild_empty: bool = False  # içeriksiz (boş) kartı olan makaleleri YENİDEN üret (opt-in)

    # --- çalışma durumu (anlık) ---
    stage: str = "idle"  # idle|fetching|carding|rebuilding|scoring|paused_training|error
    running: bool = False  # şu an bir tur yürütülüyor mu
    last_cycle_at: str = ""
    last_fetch_at: str = ""
    cycles_completed: int = 0
    last_error: str = ""

    # son tur özeti
    last_fetched: int = 0
    last_ingested: int = 0
    last_cards: int = 0
    last_rebuilt: int = 0
    last_scored: int = 0

    # kümülatif sayaçlar
    total_fetched: int = 0
    total_cards: int = 0
    total_rebuilt: int = 0
    total_scored: int = 0

    # boş kart rebuild edilmeye ÇALIŞILMIŞ makaleler (başarılı olsa da içerik gelmese de —
    # aynı makaleyi sonsuz tekrar denememek için; bkz. _rebuild_empty_cards).
    rebuilt_paper_ids: list[str] = field(default_factory=list)
    # (legacy) eski bayat-skor defteri — artık _score_missing içerik-değişimiyle (kart
    # created_at > skor computed_at) tetikler; alan geriye dönük uyumluluk için tutulur.
    rescored_paper_ids: list[str] = field(default_factory=list)
    # paper_id → boş-kart rebuild deneme sayısı; _MAX_REBUILD_ATTEMPTS'e ulaşınca bırakılır
    # (builder düzelince eski boş kartlar yeniden denensin ama bozuk-kaynak sonsuz dönmesin).
    # (legacy `rebuilt_paper_ids` yerine geçer.)
    rebuild_attempts: dict[str, int] = field(default_factory=dict)

    mastery_percent: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)  # son ~20 tur özeti

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RagLoopState:
        s = cls()
        for f in dataclasses.fields(cls):
            if f.name in d and d[f.name] is not None:
                setattr(s, f.name, d[f.name])
        return s


class RagLearningLoop:
    """Sunucu-taraflı RAG öğrenme döngüsü yöneticisi (singleton ile kullanılır)."""

    def __init__(self) -> None:
        self._state = self._load_state()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> RagLoopState:
        path = _state_file()
        if path.exists():
            try:
                return RagLoopState.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                log.warning("RAG loop: state okunamadı, varsayılana dönülüyor")
        return RagLoopState()

    def _save_state(self) -> None:
        path = _state_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state.to_dict(), indent=2), encoding="utf-8")

    def get_status(self) -> dict[str, Any]:
        return self._state.to_dict()

    def set_enabled(self, enabled: bool) -> None:
        self._state.enabled = bool(enabled)
        self._save_state()
        log.info("RAG loop: enabled=%s", self._state.enabled)

    def set_config(self, **kw: Any) -> dict[str, Any]:
        """Ayarları doğrula + kelepçele + kalıcı yap. Bilinmeyen/None alanlar yok sayılır."""
        bounds = {
            "interval_min": (5, 1440),
            "fetch_interval_hours": (1, 720),
            "max_fetch_per_cycle": (0, 50),
            "cards_per_cycle": (0, 50),
            "scores_per_cycle": (0, 50),
        }
        for key, (lo, hi) in bounds.items():
            if kw.get(key) is not None:
                try:
                    val = int(kw[key])
                except (TypeError, ValueError):
                    continue
                setattr(self._state, key, max(lo, min(hi, val)))
        for flag in ("fetch_enabled", "score_use_llm", "enabled", "rebuild_empty"):
            if kw.get(flag) is not None:
                setattr(self._state, flag, bool(kw[flag]))
        self._save_state()
        return self.get_status()

    # ------------------------------------------------------------------ zaman yardımcıları

    @staticmethod
    def _minutes_since(iso: str) -> float:
        if not iso:
            return float("inf")
        try:
            then = dt.datetime.fromisoformat(iso)
        except ValueError:
            return float("inf")
        if then.tzinfo is None:
            then = then.replace(tzinfo=dt.UTC)
        return (dt.datetime.now(dt.UTC) - then).total_seconds() / 60.0

    @staticmethod
    def _is_newer(card_iso: str, score_iso: str) -> bool:
        """``card_iso`` zaman damgası ``score_iso``'dan SONRA mı (kart skordan yeni mi).

        Boş/bozuk damga → False (güvenli: gereksiz yeniden skorlama yok).
        """
        try:
            ca = dt.datetime.fromisoformat(card_iso)
            sa = dt.datetime.fromisoformat(score_iso)
        except (TypeError, ValueError):
            return False
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=dt.UTC)
        if sa.tzinfo is None:
            sa = sa.replace(tzinfo=dt.UTC)
        return ca > sa

    def _fetch_due(self) -> bool:
        return (
            self._minutes_since(self._state.last_fetch_at) >= self._state.fetch_interval_hours * 60
        )

    async def _update_stage(self, stage: str) -> None:
        async with self._lock:
            self._state.stage = stage
            self._save_state()

    # ------------------------------------------------------- tur adımları (senkron; to_thread ile)

    @staticmethod
    def _training_running() -> bool:
        """LoRA eğitimi (web veya detached) sürüyor mu — ağır işi atlamak için."""
        try:
            from app.training.detached_launch import training_status

            return bool(training_status().get("running"))
        except Exception:
            return False

    def _fetch_new_papers(self) -> tuple[int, int]:
        """Kayıtlı (auto_ingest) arXiv sorgularından yeni makale çek + indeksle.

        Dönüş: (indirilen, indekslenen). Sorgu yoksa (0, 0).
        """
        from app.ingestion.arxiv_fetcher import fetch_arxiv_papers
        from app.ingestion.paper_loader import DiscoveredPaper, compute_file_hash
        from app.memory.paper_indexer import PaperIndexer
        from app.memory.sqlite_store import SqliteStore

        store = SqliteStore()
        queries = [q for q in store.list_arxiv_saved_queries() if q.get("auto_ingest")]
        if not queries:
            return 0, 0

        budget = self._state.max_fetch_per_cycle
        if budget <= 0:
            return 0, 0

        indexer = PaperIndexer()
        fetched = 0
        ingested = 0
        for q in queries:
            if fetched >= budget:
                break
            remaining = budget - fetched
            try:
                results = fetch_arxiv_papers(
                    q["query"], max_results=min(int(q["max_results"]), remaining)
                )
            except Exception as exc:
                log.warning("RAG loop: arXiv çekimi başarısız (%s): %s", q.get("query"), exc)
                continue
            downloaded = [r for r in results if not r.skipped]
            fetched += len(downloaded)
            for r in downloaded:
                try:
                    disc = DiscoveredPaper(path=r.pdf_path, file_hash=compute_file_hash(r.pdf_path))
                    indexer.ingest_one(disc)
                    ingested += 1
                except Exception:
                    log.warning("RAG loop: indeksleme başarısız: %s", r.arxiv_id)
            with contextlib.suppress(Exception):
                store.mark_arxiv_query_ran(q["query_id"])
        return fetched, ingested

    @staticmethod
    def _approve_if_content(store: Any, paper_id: str) -> bool:
        """En yeni pending kartı ANLAMLI içerik taşıyorsa onayla.

        continuous-learning.sh sadece `title+main_claim` non-empty bakıyor; bu ÇOK ZAYIF —
        dejenere `"..."` kartları (LLM ön-madde/çekişmede "..." döndürür) onaylanıp coverage'ı
        ÇÖPLE şişiriyor (v5-tipi risk). Burada `is_substantive_card` ile gerçek-uzunluk eşiği
        uygulanır: anlamlı (alfanümerik) title ≥8 ve main_claim ≥40 karakter. Onaylanan kart
        `papers_with_real`'e girer → coverage GERÇEK içerikle yükselir.
        """
        pend = [c for c in store.list_pending_cards() if str(c.get("paper_id", "")) == paper_id]
        if not pend:
            return False
        newest = max(pend, key=lambda c: str(c.get("created_at", "")))
        if is_substantive_card(newest.get("card_json") or {}):
            return bool(store.approve_card(newest["card_id"]))
        return False

    def _build_missing_cards(self, limit: int) -> int:
        """Kartı olmayan makalelere bilgi kartı üret + içerikliyse onayla (en çok `limit`)."""
        if limit <= 0:
            return 0
        from app.brain.knowledge_card_builder import KnowledgeCardBuilder
        from app.memory.sqlite_store import SqliteStore

        store = SqliteStore()
        builder = KnowledgeCardBuilder()
        built = 0
        for p in store.list_papers():
            if built >= limit:
                break
            if store.has_knowledge_card(p.paper_id):
                continue
            try:
                builder.build(p.paper_id)
                self._approve_if_content(store, p.paper_id)
                built += 1
            except Exception as exc:
                log.warning("RAG loop: kart üretilemedi (%s): %s", p.paper_id, exc)
        return built

    def _rebuild_empty_cards(self, limit: int) -> int:
        """İçeriksiz (boş) kartı olan makaleleri yeniden kartla (opt-in; en çok `limit`).

        `_build_missing_cards` bir makaleyi 'kartı var' diye atlar; ama o kart İÇERİKSİZ
        olabilir (build_dataset örnek üretmez → coverage düşük, LoRA darboğazı). Burada
        içerik üretmeyen makaleleri saptayıp YENİDEN kartlarız.

        Önceki sürüm makaleyi BİR denemeden sonra (başarılı olsun olmasın) kalıcı atlıyordu;
        bu, builder düzeldikten sonra bile eski boş kartların donmasına yol açtı (kök sorun:
        eski builder büyük belgelerde boş kart üretmiş, yeni builder üretiyor ama defter
        yeniden denemeyi engelliyordu). Artık deneme SAYISI tutulur (`rebuild_attempts`):
        bir makale içerik üretene kadar en çok `_MAX_REBUILD_ATTEMPTS` kez denenir. Başaran
        makale `with_real`'e girip doğal elenir; üretemeyen (ör. kaynak metni bozuk/0KB)
        tavana ulaşınca bırakılır → boşa/sonsuz LLM çağrısı yok.
        """
        if limit <= 0 or not self._state.rebuild_empty:
            return 0
        from app.brain.knowledge_card_builder import KnowledgeCardBuilder
        from app.lora.dataset_builder import build_dataset
        from app.memory.sqlite_store import SqliteStore

        store = SqliteStore()
        cards = store.list_approved_cards()
        if not cards:
            return 0
        examples = build_dataset(cards)
        with_real = {
            str(e.metadata.get("paper_id", "")) for e in examples if e.metadata.get("paper_id")
        }
        carded = {str(c["paper_id"]) for c in cards if c.get("paper_id")}
        attempts = dict(self._state.rebuild_attempts)
        gave_up = {pid for pid, n in attempts.items() if n >= _MAX_REBUILD_ATTEMPTS}
        empty = [pid for pid in carded if pid not in with_real and pid not in gave_up]
        if not empty:
            return 0

        builder = KnowledgeCardBuilder()
        done = 0
        for pid in empty:
            if done >= limit:
                break
            attempts[pid] = attempts.get(pid, 0) + 1  # denemeyi say (tavan için)
            try:
                builder.build(pid)
                self._approve_if_content(store, pid)  # içerikliyse onayla → coverage'a girsin
                done += 1
            except Exception as exc:
                log.warning("RAG loop: boş kart rebuild başarısız (%s): %s", pid, exc)
        self._state.rebuild_attempts = attempts
        return done

    def _score_missing(self, limit: int) -> int:
        """Skorsuz VEYA kartı skordan SONRA değişmiş makaleleri yeniden skorla.

        Comprehension skoru bir kez hesaplanıp kalıcı kalıyordu; kart boşken düşük skor
        alıp sonra içerik kazanan makale o düşük skorda DONUYORDU (kök sorun). Eski çözüm
        yalnız ``total_score ≤ 20`` makaleyi BİR kez yeniden skorluyordu → ortalama ~59
        iken 21-99 arasındaki tüm makaleler sonsuza donuyordu.

        Artık ölçüt İÇERİK DEĞİŞİMİ: en güncel onaylı kartın ``created_at``'i skorun
        ``computed_at``'inden yeni ise (kart skordan sonra yeniden üretildi) yeniden
        skorla. Bu kendini sınırlar — skor güncellenince ``computed_at`` kartı geçer,
        kart tekrar değişene dek tetiklenmez (sonsuz döngü yok, defter gerekmez). Skoru
        hiç olmayanlar da skorlanır. Determinizm: skorlama LLM çağrısı sabit seed'li.
        """
        if limit <= 0:
            return 0
        from app.memory.sqlite_store import SqliteStore
        from app.verification.comprehension_scorer import ComprehensionScorer

        store = SqliteStore()
        scorer = ComprehensionScorer()
        # paper_id → en güncel onaylı kart zaman damgası (içerik değişimi tespiti için).
        latest_card_at: dict[str, str] = {}
        for card in store.list_approved_cards():
            pid = str(card.get("paper_id") or "")
            created = str(card.get("created_at") or "")
            if pid and created > latest_card_at.get(pid, ""):
                latest_card_at[pid] = created

        scored = 0
        for p in store.list_papers():
            if scored >= limit:
                break
            if not store.has_knowledge_card(p.paper_id):
                continue
            row = store.get_comprehension_score(p.paper_id)
            card_changed = row is not None and self._is_newer(
                latest_card_at.get(p.paper_id, ""), getattr(row, "computed_at", "") or ""
            )
            if row is not None and not card_changed:
                continue
            try:
                result = scorer.score(p.paper_id, use_llm=self._state.score_use_llm)
                store.save_comprehension_score(result)
                scored += 1
            except Exception as exc:
                log.warning("RAG loop: skor hesaplanamadı (%s): %s", p.paper_id, exc)
        return scored

    @staticmethod
    def _refresh_mastery() -> int | None:
        try:
            from app.verification.rag_mastery import compute_rag_mastery

            return compute_rag_mastery().get("mastery_percent")
        except Exception:
            return None

    # ------------------------------------------------------------------ tur

    @tracked("rag-learning-loop", trigger_type="background_loop")
    async def run_one_cycle(self) -> dict[str, Any]:
        """Tek bir öğrenme turunu yürüt. Eşzamanlı ikinci tur reddedilir."""
        async with self._lock:
            if self._state.running:
                return {"ok": False, "reason": "Zaten bir tur yürütülüyor"}
            self._state.running = True
            self._state.last_error = ""
            self._save_state()

        try:
            # LoRA eğitimi sürüyorsa LLM/RAM çakışmasını önle — ağır işi atla.
            if await asyncio.to_thread(self._training_running):
                mastery = await asyncio.to_thread(self._refresh_mastery)
                async with self._lock:
                    self._state.stage = "paused_training"
                    self._state.mastery_percent = mastery
                    self._state.last_cycle_at = _utcnow()
                    self._save_state()
                log.info("RAG loop: LoRA eğitimi sürüyor → tur atlandı (duraklatıldı)")
                return {"ok": True, "skipped": "training_running", "mastery": mastery}

            fetched = ingested = cards = rebuilt = scored = 0
            log_step("RAG turu başladı (çek→kart→skor→ustalık)")

            # 1) yeni makale çek (nazik kadans; yalnız çekim aralığı dolduysa)
            if self._state.fetch_enabled and self._fetch_due():
                await self._update_stage("fetching")
                fetched, ingested = await asyncio.to_thread(self._fetch_new_papers)
                async with self._lock:
                    self._state.last_fetch_at = _utcnow()
                    self._save_state()

            # 2) eksik kartları üret (kartı HİÇ olmayan makaleler)
            await self._update_stage("carding")
            cards = await asyncio.to_thread(self._build_missing_cards, self._state.cards_per_cycle)

            # 2b) içeriksiz (boş) kartları yeniden üret (opt-in — coverage darboğazı)
            if self._state.rebuild_empty:
                await self._update_stage("rebuilding")
                rebuilt = await asyncio.to_thread(
                    self._rebuild_empty_cards, self._state.cards_per_cycle
                )

            # 3) eksik skorları hesapla
            await self._update_stage("scoring")
            scored = await asyncio.to_thread(self._score_missing, self._state.scores_per_cycle)

            # 4) RAG ustalığını güncelle (türetilmiş; ağır LLM yok)
            mastery = await asyncio.to_thread(self._refresh_mastery)

            now = _utcnow()
            async with self._lock:
                self._state.last_fetched = fetched
                self._state.last_ingested = ingested
                self._state.last_cards = cards
                self._state.last_rebuilt = rebuilt
                self._state.last_scored = scored
                self._state.total_fetched += fetched
                self._state.total_cards += cards
                self._state.total_rebuilt += rebuilt
                self._state.total_scored += scored
                self._state.mastery_percent = mastery
                self._state.cycles_completed += 1
                self._state.last_cycle_at = now
                self._state.stage = "idle"
                self._state.history.append(
                    {
                        "at": now,
                        "ingested": ingested,
                        "cards": cards,
                        "rebuilt": rebuilt,
                        "scored": scored,
                        "mastery": mastery,
                    }
                )
                self._state.history = self._state.history[-20:]
                self._save_state()

            log.info(
                "RAG loop turu bitti: +%d makale, +%d kart, +%d rebuild, +%d skor (ustalık %s%%)",
                ingested,
                cards,
                rebuilt,
                scored,
                mastery,
            )
            return {
                "ok": True,
                "fetched": fetched,
                "ingested": ingested,
                "cards": cards,
                "rebuilt": rebuilt,
                "scored": scored,
                "mastery": mastery,
            }
        except Exception as exc:
            async with self._lock:
                self._state.stage = "error"
                self._state.last_error = str(exc)
                self._save_state()
            log.exception("RAG loop: tur hatası")
            return {"ok": False, "reason": str(exc)}
        finally:
            async with self._lock:
                self._state.running = False
                self._save_state()

    def trigger_once_bg(self) -> dict[str, Any]:
        """Bir turu arka plan görevi olarak başlat (anında döner — UI durumdan izler)."""
        if self._state.running:
            return {"ok": False, "reason": "Zaten bir tur yürütülüyor"}
        task = asyncio.create_task(self.run_one_cycle())
        task.add_done_callback(lambda _t: None)
        return {"ok": True, "started": True}

    # ------------------------------------------------------------------ arka plan döngüsü

    async def background_loop(self) -> None:
        """15 sn nabızla çalışır; etkinse ve aralık dolduysa bir tur yürütür."""
        log.info("RAG öğrenme döngüsü arka planı başladı (enabled=%s)", self._state.enabled)
        while True:
            await asyncio.sleep(15)
            if not self._state.enabled or self._state.running:
                continue
            if self._minutes_since(self._state.last_cycle_at) < self._state.interval_min:
                continue
            try:
                await self.run_one_cycle()
            except Exception:
                log.exception("RAG loop: arka plan turu hatası")


# ---------- singleton ----------

_loop: RagLearningLoop | None = None


def get_rag_loop() -> RagLearningLoop:
    global _loop
    if _loop is None:
        _loop = RagLearningLoop()
    return _loop
