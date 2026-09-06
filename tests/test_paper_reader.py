"""Makale Okuyucu ajanı testleri — çevrimdışı (LLM/Ollama yok).

Planlayıcı saf; koşu, RagLearningLoop adımları stub'lanarak izlenir. Koşunun kendisi
agent_runs'a düşer (track_agent_run) — tracker hataları koşuyu bozmaz.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.research import paper_reader
from app.research.paper_reader import ReaderPlan, plan_reading, run_reader


def test_plan_reading_saf() -> None:
    p = plan_reading(["a", "b", "c", "d"], carded_ids={"a", "b"}, scored_ids={"a"})
    assert p.toplam == 4
    assert p.kartsiz == ["c", "d"]
    assert p.skorsuz == ["b"]  # kartlı ama skorsuz; kartsızlar burada DEĞİL
    assert p.okunmus == 1 and p.yuzde == 25.0
    assert p.to_dict()["okunmus"] == 1


def test_plan_bos_korpus_yuzde_100() -> None:
    assert ReaderPlan(0, [], []).yuzde == 100.0


class _Store:
    def __init__(self, ids, carded, scored) -> None:
        self._ids, self._carded, self._scored = ids, set(carded), set(scored)

    def list_papers(self):
        return [SimpleNamespace(paper_id=i) for i in self._ids]

    def has_knowledge_card(self, pid: str) -> bool:
        return pid in self._carded

    def get_comprehension_score(self, pid: str):
        return object() if pid in self._scored else None


class _Loop:
    """RagLearningLoop stub: çağrıları sayar, ilerlemeyi store'a yansıtır."""

    def __init__(self, store: _Store) -> None:
        self._state = SimpleNamespace(score_use_llm=True)
        self._store = store
        self.calls: list[tuple[str, int]] = []

    def _build_missing_cards(self, limit: int) -> int:
        self.calls.append(("kart", limit))
        n = 0
        for pid in self._store._ids:
            if pid not in self._store._carded and n < limit:
                self._store._carded.add(pid)
                n += 1
        return n

    def _score_missing(self, limit: int) -> int:
        self.calls.append(("skor", limit))
        n = 0
        for pid in self._store._ids:
            if pid in self._store._carded and pid not in self._store._scored and n < limit:
                self._store._scored.add(pid)
                n += 1
        return n


def test_run_reader_kart_ve_skor_uretir(monkeypatch) -> None:
    store = _Store(["a", "b", "c"], carded=["a"], scored=[])
    loop = _Loop(store)
    r = run_reader(cards=5, scores=5, loop=loop, store=store, use_llm_score=False)
    assert r["ok"] and r["kart"] == 2 and r["skor"] == 3
    assert r["once"]["okunmus"] == 0 and r["sonra"]["okunmus"] == 3
    assert loop.calls == [("kart", 5), ("skor", 5)]
    assert loop._state.score_use_llm is False  # koşuya özgü, diske yazılmadan


def test_run_reader_dry_run_hicbir_sey_yapmaz() -> None:
    store = _Store(["a", "b"], carded=[], scored=[])
    loop = _Loop(store)
    r = run_reader(dry_run=True, loop=loop, store=store)
    assert r["dry_run"] is True and r["kart"] == 0 and r["skor"] == 0
    assert loop.calls == []
    assert r["once"]["kartsiz"] == 2


def test_run_reader_butce_sinirlar() -> None:
    store = _Store(["a", "b", "c", "d"], carded=[], scored=[])
    loop = _Loop(store)
    r = run_reader(cards=1, scores=1, loop=loop, store=store)
    assert r["kart"] == 1 and r["skor"] == 1
    assert r["sonra"]["kartsiz"] == 3  # kalan iş raporlanır, döngüye girilmez


def test_agent_id_manifestte_kayitli() -> None:
    from app.agents.runtime import list_agents

    assert paper_reader.AGENT_ID in {a.agent_id for a in list_agents()}
