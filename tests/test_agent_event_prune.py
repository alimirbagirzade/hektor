"""Agent olay retention budaması (``prune_agent_events``) — Kademe-2 regresyon testleri.

Kapsam:
- Tek ``IN(...)`` yerine parçalı silme: >SQLITE_MAX_VARIABLE_NUMBER (~32766) olayda
  ``OperationalError`` fırlamamalı (retention'ın asıl tetiklendiği büyük backlog).
- Eşit-``ts`` sınırında budama deterministik olmalı (CLAUDE.md Kural 6).
- Yaş/overflow budaması doğru olmalı (en yeni ``max_events`` korunur, eski silinir).

Tümü çevrimdışı. Her test kendi ``tmp_path`` DB'sini kullanır (conftest'in oturum-kapsamlı
paylaşımlı DB'sinden bağımsız izolasyon).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select

from app.memory.sqlite_store import AgentEventRow, SqliteStore


def _store(tmp_path, name: str = "t.db") -> SqliteStore:
    return SqliteStore(db_path=tmp_path / name)


def _seed_events(store: SqliteStore, specs: list[tuple[str, str]]) -> None:
    """specs: (event_id, ts) listesi → toplu ekle (hızlı, tek transaction)."""
    with store.session() as s:
        s.add_all(AgentEventRow(event_id=eid, run_id="r", ts=ts, kind="step") for eid, ts in specs)


def _count(store: SqliteStore) -> int:
    with store.session() as s:
        return s.scalar(select(func.count()).select_from(AgentEventRow))


def _ids(store: SqliteStore) -> set[str]:
    with store.session() as s:
        return set(s.scalars(select(AgentEventRow.event_id)))


def test_prune_buyuk_backlog_sql_degisken_limiti_asilmaz(tmp_path) -> None:
    # REGRESYON: tek IN(...) ile 32766'dan fazla event budamak OperationalError fırlatır
    # ve çağıran tarafça sessizce yutulurdu → retention kalıcı çalışmazdı.
    store = _store(tmp_path)
    old_ts = (dt.datetime.now(dt.UTC) - dt.timedelta(days=60)).isoformat()
    n = 33_000  # > SQLITE_MAX_VARIABLE_NUMBER (32766)
    _seed_events(store, [(f"e{i:06d}", old_ts) for i in range(n)])

    deleted = store.prune_agent_events(max_events=50_000, max_age_days=30)

    assert deleted == n  # hepsi 30 günden eski → hepsi silinmeli, hata fırlamadan
    assert _count(store) == 0


def test_prune_overflow_en_yeniyi_korur(tmp_path) -> None:
    store = _store(tmp_path)
    base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    # 10 olay, artan ts (e0 en eski, e9 en yeni)
    _seed_events(
        store,
        [(f"e{i}", (base + dt.timedelta(minutes=i)).isoformat()) for i in range(10)],
    )

    deleted = store.prune_agent_events(max_events=3, max_age_days=3650)

    assert deleted == 7  # en yeni 3 korunur (e7,e8,e9)
    assert _ids(store) == {"e7", "e8", "e9"}


def test_prune_esit_ts_deterministik(tmp_path) -> None:
    # Eşit ts'li olaylarda kesme noktasında hangi olayın budanacağı deterministik olmalı.
    same_ts = "2026-01-01T00:00:00+00:00"
    keep_runs = []
    for k in range(2):
        store = _store(tmp_path, name=f"run{k}.db")
        _seed_events(store, [(f"e{i:02d}", same_ts) for i in range(10)])
        store.prune_agent_events(max_events=4, max_age_days=3650)
        keep_runs.append(frozenset(_ids(store)))

    assert keep_runs[0] == keep_runs[1]  # iki çalıştırma aynı kümeyi korumalı
    assert len(keep_runs[0]) == 4
    # ikincil anahtar event_id.desc() → en büyük 4 event_id korunur
    assert keep_runs[0] == {"e06", "e07", "e08", "e09"}
