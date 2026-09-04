"""Sentinel (Nöbetçi) çekirdeği — çevrimdışı testler (sahte probe + tmp SQLite).

Gerçek Ollama/web/eğitim YOK. Verdict agregasyonu, probe-istisna güvenliği, geçmiş
kaydı + budama ve salt-okuma sözleşmesi doğrulanır.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.monitoring.sentinel import ProbeResult, Sentinel
from app.monitoring.store import MonitoringStore


@pytest.fixture
def store(tmp_path: Path) -> MonitoringStore:
    return MonitoringStore(db_path=tmp_path / "sentinel.db")


def _p(name: str, status: str, detail: str = "") -> ProbeResult:
    return ProbeResult(name, status, detail)


def _mk(result: ProbeResult):
    def probe() -> ProbeResult:
        return result

    probe.__name__ = result.name
    return probe


# ── agregasyon ───────────────────────────────────────────────────────────────


def test_all_ok_overall_ok(store: MonitoringStore) -> None:
    s = Sentinel(probes=[_mk(_p("a", "ok")), _mk(_p("b", "ok"))], store=store)
    r = s.run()
    assert r.overall == "ok"
    assert "sağlıklı" in r.summary


def test_warn_beats_ok(store: MonitoringStore) -> None:
    s = Sentinel(probes=[_mk(_p("a", "ok")), _mk(_p("b", "warn"))], store=store)
    assert s.run().overall == "warn"


def test_fail_beats_warn(store: MonitoringStore) -> None:
    s = Sentinel(
        probes=[_mk(_p("a", "warn")), _mk(_p("b", "fail")), _mk(_p("c", "ok"))], store=store
    )
    r = s.run()
    assert r.overall == "fail"
    assert "müdahale" in r.summary


def test_all_skip_overall_skip(store: MonitoringStore) -> None:
    s = Sentinel(probes=[_mk(_p("a", "skip")), _mk(_p("b", "skip"))], store=store)
    assert s.run().overall == "skip"


def test_no_probes_overall_skip(store: MonitoringStore) -> None:
    assert Sentinel(probes=[], store=store).run().overall == "skip"


# ── istisna güvenliği ─────────────────────────────────────────────────────────


def test_raising_probe_becomes_skip_not_crash(store: MonitoringStore) -> None:
    def boom() -> ProbeResult:
        raise RuntimeError("patladı")

    s = Sentinel(probes=[boom, _mk(_p("b", "ok"))], store=store)
    r = s.run()
    assert r.overall == "ok"  # patlayan probe skip sayılır, nöbetçi düşmez
    assert any(p.status == "skip" and "patladı" in p.detail for p in r.probes)


# ── geçmiş + budama ───────────────────────────────────────────────────────────


def test_run_persists_history_roundtrip(store: MonitoringStore) -> None:
    s = Sentinel(probes=[_mk(_p("a", "warn", "detay"))], store=store)
    s.run()
    hist = s.history(limit=5)
    assert len(hist) == 1
    assert hist[0]["overall"] == "warn"
    assert hist[0]["probes"][0]["name"] == "a"
    assert hist[0]["probes"][0]["detail"] == "detay"


def test_persist_false_writes_nothing(store: MonitoringStore) -> None:
    s = Sentinel(probes=[_mk(_p("a", "ok"))], store=store)
    s.run(persist=False)
    assert s.history() == []


def test_history_pruned_to_keep_last(tmp_path: Path) -> None:
    store = MonitoringStore(db_path=tmp_path / "prune.db", keep_last=3)
    s = Sentinel(probes=[_mk(_p("a", "ok"))], store=store)
    for _ in range(6):
        s.run()
    assert len(store.history(limit=100)) <= 3


def test_store_failure_does_not_break_live_report(tmp_path: Path) -> None:
    """Geçmiş yazılamasa bile canlı rapor döner (best-effort persist)."""

    class BrokenStore(MonitoringStore):
        def record(self, **kw):  # type: ignore[override]
            raise OSError("disk dolu")

    s = Sentinel(probes=[_mk(_p("a", "ok"))], store=BrokenStore(db_path=tmp_path / "broken.db"))
    assert s.run().overall == "ok"  # istisna yutuldu, rapor sağlam


def test_prune_same_timestamp_ties_keep_new_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aynı-timestamp bağında (mikrosaniye çakışması) yeni kayıt SİLİNMEZ (review fix: <= → <)."""
    import app.monitoring.store as store_mod

    monkeypatch.setattr(store_mod, "utcnow", lambda: "2026-01-01T00:00:00+00:00")
    store = MonitoringStore(db_path=tmp_path / "ties.db", keep_last=3)
    ids = [store.record(overall="ok", summary=f"koşu {i}", probes=[]) for i in range(5)]
    hist = store.history(limit=100)
    # < ile: bağdaki kayıtlar silinmez → hepsi durur; kritik değişmez: SON kayıt kayıpsız
    remaining = {h["check_id"] for h in hist}
    assert ids[-1] in remaining
    assert len(hist) >= 3


def test_invalid_probe_status_normalized_to_skip(store: MonitoringStore) -> None:
    """Enjekte probe geçersiz status dönerse overall sözleşmesi bozulmaz (→ skip)."""
    s = Sentinel(probes=[_mk(_p("x", "unknown"))], store=store)
    r = s.run(persist=False)
    assert r.overall == "skip"


# ── gerçek varsayılan probe'lar (offline: çökmemeli) ─────────────────────────


def test_default_probes_run_offline_without_crash(store: MonitoringStore) -> None:
    """Gerçek default_probes çevrimdışı ortamda istisnasız koşar; verdict ne olursa olsun
    yapı sağlamdır (Ollama kapalıyken llm=fail olması NORMAL)."""
    r = Sentinel(store=store).run(persist=False)
    assert r.overall in {"ok", "warn", "fail", "skip"}
    names = {p.name for p in r.probes}
    assert "rag_loop" in names
    assert {"llm", "training", "disk", "sqlite"} <= names
    assert all(p.status in {"ok", "warn", "fail", "skip"} for p in r.probes)
