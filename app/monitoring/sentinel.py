"""sentinel.py — Sentinel (Nöbetçi): tüm ajan/altsistemleri izleyen sağlık monitörü.

Tasarım (smoke.py evi stili): probe'lar ENJEKTE edilebilir → testler %100 çevrimdışı;
varsayılan probe'lar savunmacı lazy-import'lu (modül eksikse skip, koşu çökmez).

Probe verdicti: "ok" | "warn" | "fail" | "skip". Bütünsel agregasyon: herhangi bir
fail → fail; yoksa herhangi bir warn → warn; yoksa en az bir ok → ok; hepsi skip → skip.

GÜVENLİK: tüm probe'lar SALT-OKUMA — hiçbir şeyi durdurmaz/başlatmaz/mutasyona uğratmaz
(ör. stale orkestrasyon aşamasını recover ETMEZ, yalnız `orchestrate-recover` önerir).
Resource Negotiator DANIŞMAN modda `contention` probe'udur: CPU çekişmesini raporlar,
eğitimi duraklatmaz (detached PEFT eğitiminde güvenli duraklat/sürdür yok — riskli).
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.monitoring.store import MonitoringStore, utcnow

log = logging.getLogger(__name__)

_WEB_TIMEOUT_S = 2.0
_DISK_WARN_GB = 10.0
_DISK_FAIL_GB = 2.0
_FEEDBACK_BACKLOG_WARN = 50
_ORCH_STALE_MIN = 30.0

# Verdict önceliği (agregasyon): büyük olan kazanır.
_SEVERITY = {"skip": 0, "ok": 1, "warn": 2, "fail": 3}


@dataclass
class ProbeResult:
    """Tek bir yoklamanın sonucu."""

    name: str
    status: str  # "ok" | "warn" | "fail" | "skip"
    detail: str = ""
    advice: str = ""  # insan için önerilen eylem (boş = eylem gerekmez)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "advice": self.advice,
        }


Probe = Callable[[], ProbeResult]


@dataclass
class SentinelReport:
    """Nöbetçi koşusunun bütünsel sonucu."""

    overall: str  # "ok" | "warn" | "fail" | "skip"
    summary: str
    probes: list[ProbeResult] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "summary": self.summary,
            "probes": [p.to_dict() for p in self.probes],
            "created_at": self.created_at,
        }


def _guard(name: str, fn: Callable[[], ProbeResult]) -> ProbeResult:
    """Probe istisnası nöbetçiyi düşürmesin → skip (sessiz yutma değil: detayda hata)."""
    try:
        return fn()
    except Exception as exc:
        log.debug("Sentinel: %s probe hatası: %s", name, exc)
        return ProbeResult(name, "skip", f"yoklanamadı: {type(exc).__name__}: {exc}")


# ── varsayılan probe'lar (hepsi salt-okuma, savunmacı) ───────────────────────


def probe_llm() -> ProbeResult:
    def _run() -> ProbeResult:
        from app.brain.local_llm import LocalLLM

        llm = LocalLLM()
        backend = llm.active_backend()
        if backend != "none" and llm.available():
            return ProbeResult("llm", "ok", f"LLM backend canlı (backend={backend}).")
        return ProbeResult(
            "llm",
            "fail",
            f"LLM backend erişilemez (backend={backend}).",
            "Ollama'yı başlat (runtime kalbi; RAG/RLM/eval buna dayanır).",
        )

    return _guard("llm", _run)


def probe_web() -> ProbeResult:
    def _run() -> ProbeResult:
        import httpx

        from app.config import get_settings

        port = int(getattr(get_settings(), "web_port", 8765) or 8765)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/", timeout=_WEB_TIMEOUT_S)
            if r.status_code < 500:
                return ProbeResult("web", "ok", f"Web sunucusu canlı (:{port}).")
            return ProbeResult("web", "warn", f"Web sunucusu {r.status_code} döndü (:{port}).")
        except Exception:
            return ProbeResult(
                "web",
                "warn",
                f"Web sunucusu yanıt vermiyor (:{port}).",
                "Gerekliyse başlat: uv run hektor-web (CLI onsuz da çalışır).",
            )

    return _guard("web", _run)


def probe_training() -> ProbeResult:
    def _run() -> ProbeResult:
        from app.training.detached_launch import training_status

        ts = training_status() or {}
        if ts.get("running"):
            step, total = ts.get("step"), ts.get("total")
            pos = f" (adım {step}/{total})" if step and total else ""
            return ProbeResult("training", "ok", f"Eğitim sürüyor{pos}.")
        return ProbeResult("training", "ok", "Eğitim boşta (koşu yok).")

    return _guard("training", _run)


def probe_orchestration() -> ProbeResult:
    def _run() -> ProbeResult:
        from app.orchestration.store import OrchestrationStore, iso_minutes_ago

        store = OrchestrationStore()
        stale = store.find_stale_running_stages(iso_minutes_ago(_ORCH_STALE_MIN))
        if stale:
            names = ", ".join(f"{rid[:12]}…/{st}" for rid, st in stale[:3])
            return ProbeResult(
                "orchestration",
                "warn",
                f"{len(stale)} asılı (stale) 'running' aşama: {names}",
                "İncele ve kurtar: hektor orchestrate-recover",
            )
        return ProbeResult("orchestration", "ok", "Asılı orkestrasyon aşaması yok.")

    return _guard("orchestration", _run)


def probe_stop_all() -> ProbeResult:
    def _run() -> ProbeResult:
        from app.agents.runtime import supervisor

        if supervisor.is_stop_all_active():
            return ProbeResult(
                "stop_all",
                "warn",
                "STOP_ALL aktif — tehlikeli işlemler bloklu.",
                "Bilinçli değilse kaldır: uv run hektor clear-stop-all",
            )
        return ProbeResult("stop_all", "ok", "STOP_ALL kapalı.")

    return _guard("stop_all", _run)


def probe_disk() -> ProbeResult:
    def _run() -> ProbeResult:
        from app.config import get_settings

        usage = shutil.disk_usage(get_settings().root)
        free_gb = usage.free / (1024**3)
        if free_gb < _DISK_FAIL_GB:
            return ProbeResult(
                "disk",
                "fail",
                f"Disk kritik: {free_gb:.1f} GB boş.",
                "Yer aç — eğitim/checkpoint yazamayabilir.",
            )
        if free_gb < _DISK_WARN_GB:
            return ProbeResult("disk", "warn", f"Disk azalıyor: {free_gb:.1f} GB boş.")
        return ProbeResult("disk", "ok", f"Disk yeterli: {free_gb:.0f} GB boş.")

    return _guard("disk", _run)


def probe_sqlite() -> ProbeResult:
    def _run() -> ProbeResult:
        from app.config import get_settings

        db = get_settings().sqlite_file
        if not db.exists():
            return ProbeResult("sqlite", "warn", "SQLite dosyası henüz yok (init edilmemiş).")
        conn = sqlite3.connect(str(db), timeout=5.0)
        try:
            # busy_timeout: WAL yazıcılarıyla eşzamanlıyken 'database is locked' düşmesin
            # (per-bağlantı ayar, DB'ye yazmaz). journal_mode'a BİLEREK dokunulmaz —
            # o kalıcı DB özelliğidir; salt-okuma probe DB durumunu değiştirmemeli.
            conn.execute("PRAGMA busy_timeout=5000")
            row = conn.execute("PRAGMA quick_check").fetchone()
        finally:
            conn.close()
        verdict = str(row[0]) if row else "?"
        if verdict == "ok":
            return ProbeResult("sqlite", "ok", "SQLite bütünlüğü sağlam (quick_check ok).")
        return ProbeResult(
            "sqlite",
            "fail",
            f"SQLite quick_check: {verdict[:120]}",
            "Yedek al; bozulmayı araştır (WAL checkpoint / kopya-onar).",
        )

    return _guard("sqlite", _run)


def probe_feedback() -> ProbeResult:
    def _run() -> ProbeResult:
        from app.feedback import FeedbackStore

        pending = FeedbackStore().counts().get("pending", 0)
        if pending > _FEEDBACK_BACKLOG_WARN:
            return ProbeResult(
                "feedback",
                "warn",
                f"{pending} bekleyen düzeltme birikti.",
                "Gözden geçir: 13·GERİ BİLDİRİM sekmesi veya hektor feedback-list",
            )
        return ProbeResult("feedback", "ok", f"{pending} bekleyen düzeltme.")

    return _guard("feedback", _run)


def probe_contention() -> ProbeResult:
    """Resource Negotiator (DANIŞMAN): eğitim + canlı sorgu yükü çakışıyor mu?

    CPU-only makinede eğitim sürerken Ollama sorguları yavaşlar. Bu probe yalnız
    UYARIR (eğitimi duraklatmaz — detached PEFT'te güvenli resume yok). rag-learning-loop
    zaten eğitim sürerken kendini duraklatır; bu, insan kullanıcı sorguları içindir."""

    def _run() -> ProbeResult:
        from app.training.detached_launch import training_status

        if not (training_status() or {}).get("running"):
            return ProbeResult("contention", "ok", "CPU çekişmesi yok (eğitim boşta).")
        return ProbeResult(
            "contention",
            "warn",
            "Eğitim sürüyor → CPU çekişmesi: sorgular/eval yavaşlayabilir.",
            "Acil sorgu gerekiyorsa eğitimin bitmesini bekle (rag-loop zaten duraklar).",
        )

    return _guard("contention", _run)


def probe_rag_loop() -> ProbeResult:
    """RAG döngüsü hata verdi mi; eğitim nedeniyle duraklama sağlıklı sayılır."""

    def _run() -> ProbeResult:
        from app.research.rag_learning_loop import get_rag_loop

        state = get_rag_loop().get_status()
        stage = str(state.get("stage", "idle"))
        if stage == "error":
            return ProbeResult("rag_loop", "fail", str(state.get("last_error") or "RAG loop error"))
        return ProbeResult(
            "rag_loop", "ok", f"RAG loop stage={stage}, enabled={bool(state.get('enabled'))}."
        )

    return _guard("rag_loop", _run)


def default_probes() -> list[Probe]:
    """Üretim varsayılanı — sıra UI'daki gösterim sırasıdır."""
    return [
        probe_llm,
        probe_web,
        probe_training,
        probe_orchestration,
        probe_stop_all,
        probe_disk,
        probe_sqlite,
        probe_feedback,
        probe_rag_loop,
        probe_contention,
    ]


class Sentinel:
    """Sağlık nöbetçisi — probe'ları koşar, agregasyon yapar, geçmişe yazar."""

    def __init__(
        self,
        probes: Sequence[Probe] | None = None,
        store: MonitoringStore | None = None,
    ) -> None:
        self._probes: list[Probe] = list(probes) if probes is not None else default_probes()
        self.store = store or MonitoringStore()

    def run(self, *, persist: bool = True) -> SentinelReport:
        """Tüm probe'ları koş → bütünsel verdict; persist=True ise geçmişe yaz."""
        results = [_guard(getattr(p, "__name__", "probe"), p) for p in self._probes]

        worst = max(results, key=lambda r: _SEVERITY.get(r.status, 0), default=None)
        # Sözleşme: overall ∈ {ok,warn,fail,skip}. Enjekte probe geçersiz status dönerse
        # (ör. "unknown") sözleşmeyi bozmasın → skip'e normalize (review bulgusu).
        overall = worst.status if worst is not None and worst.status in _SEVERITY else "skip"
        n_fail = sum(1 for r in results if r.status == "fail")
        n_warn = sum(1 for r in results if r.status == "warn")
        if overall == "fail":
            summary = f"{n_fail} kritik + {n_warn} uyarı — müdahale gerekli."
        elif overall == "warn":
            summary = f"{n_warn} uyarı — sistem çalışıyor, dikkat önerilir."
        elif overall == "ok":
            summary = "Tüm yoklamalar sağlıklı."
        else:
            summary = "Hiçbir yoklama koşulamadı (skip)."

        report = SentinelReport(
            overall=overall, summary=summary, probes=results, created_at=utcnow()
        )
        if persist:
            try:
                self.store.record(
                    overall=overall, summary=summary, probes=[r.to_dict() for r in results]
                )
            except Exception:  # geçmiş yazılamasa bile canlı rapor dönsün
                log.warning("Sentinel: geçmiş kaydedilemedi", exc_info=True)
        return report

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.history(limit=limit)
