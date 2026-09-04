"""Duman testi koşucusu — çevrimdışı testler (enjekte edilen sahte LLM/retriever).

Gerçek Ollama YOK: tüm yoklamalar enjekte edilen fake'lerle sürülür. Verdict semantiği
(skip/pass/fail), determinizm (Kural 6 seed), retrieval'in kritik-olmaması ve delege'nin
verdict→StageStatus eşlemesi doğrulanır.
"""

from __future__ import annotations

from typing import Any

import pytest

import app.orchestration.smoke as smoke_mod
from app.orchestration import delegates
from app.orchestration.orchestrator import RunContext
from app.orchestration.pipeline import StageStatus
from app.orchestration.smoke import SmokeCheck, SmokeResult, SmokeRunner


class FakeLLM:
    def __init__(
        self,
        *,
        backend: str = "ollama",
        available: bool = True,
        output: str = "TAMAM",
        raise_exc: Exception | None = None,
    ) -> None:
        self._backend = backend
        self._available = available
        self._output = output
        self._raise = raise_exc
        self.generate_calls: list[tuple[str, dict[str, Any]]] = []

    def available(self) -> bool:
        return self._available

    def active_backend(self) -> str:
        return self._backend

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.generate_calls.append((prompt, kwargs))
        if self._raise is not None:
            raise self._raise
        return self._output


class FakeRetriever:
    def __init__(self, n: int = 3, raise_exc: Exception | None = None) -> None:
        self._n = n
        self._raise = raise_exc

    def retrieve(self, query: str, top_k: int | None = None) -> list[Any]:
        if self._raise is not None:
            raise self._raise
        return [object() for _ in range(self._n)]


def _check(result: SmokeResult, name: str) -> SmokeCheck | None:
    return next((c for c in result.checks if c.name == name), None)


# ── verdict semantiği ─────────────────────────────────────────────────────────


def test_pass_when_live_and_sane() -> None:
    r = SmokeRunner(llm=FakeLLM(output="TAMAM"), retriever=FakeRetriever(n=2)).run()
    assert r.verdict == "pass"
    assert _check(r, "backend").status == "pass"
    assert _check(r, "generation").status == "pass"
    assert _check(r, "retrieval").status == "pass"


def test_skip_when_backend_unavailable_does_not_generate() -> None:
    llm = FakeLLM(available=False, backend="none")
    r = SmokeRunner(llm=llm, retriever=FakeRetriever()).run()
    assert r.verdict == "skip"
    assert _check(r, "backend").status == "skip"
    assert _check(r, "generation") is None  # üretim DENENMEZ
    assert llm.generate_calls == []  # canlı değilken model çağrılmaz


def test_skip_when_no_llm_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = SmokeRunner(retriever=FakeRetriever())
    monkeypatch.setattr(runner, "_get_llm", lambda: None)  # type: ignore[method-assign]
    r = runner.run()
    assert r.verdict == "skip"
    assert _check(r, "backend").status == "skip"


def test_fail_when_generation_empty() -> None:
    r = SmokeRunner(llm=FakeLLM(output="   "), retriever=FakeRetriever()).run()
    assert r.verdict == "fail"
    assert _check(r, "generation").status == "fail"


def test_fail_when_generation_raises() -> None:
    r = SmokeRunner(
        llm=FakeLLM(raise_exc=RuntimeError("model çekilmemiş")), retriever=FakeRetriever()
    ).run()
    assert r.verdict == "fail"
    assert "model çekilmemiş" in _check(r, "generation").detail


def test_fail_when_degenerate() -> None:
    r = SmokeRunner(
        llm=FakeLLM(output="döngü döngü döngü"),
        retriever=FakeRetriever(),
        degenerate_fn=lambda _t: True,
    ).run()
    assert r.verdict == "fail"
    assert "degenere" in _check(r, "generation").detail


def test_reuses_real_degenerate_heuristic() -> None:
    """degenerate_fn verilmezse adapter_eval._is_degenerate yeniden kullanılır."""
    looping = ". ".join(["aynı cümle tekrar ediyor burada"] * 5)
    r = SmokeRunner(llm=FakeLLM(output=looping), retriever=FakeRetriever()).run()
    assert r.verdict == "fail"
    assert _check(r, "generation").status == "fail"


# ── retrieval kritik DEĞİL ─────────────────────────────────────────────────────


def test_empty_retrieval_warns_not_fails() -> None:
    r = SmokeRunner(llm=FakeLLM(output="TAMAM"), retriever=FakeRetriever(n=0)).run()
    assert r.verdict == "pass"  # boş korpus verdict'i düşürmez
    assert _check(r, "retrieval").status == "warn"


def test_retrieval_exception_warns_not_fails() -> None:
    r = SmokeRunner(
        llm=FakeLLM(output="TAMAM"), retriever=FakeRetriever(raise_exc=RuntimeError("x"))
    ).run()
    assert r.verdict == "pass"
    assert _check(r, "retrieval").status == "warn"


# ── determinizm + şekil ────────────────────────────────────────────────────────


def test_generation_probe_is_deterministic() -> None:
    llm = FakeLLM(output="TAMAM")
    SmokeRunner(llm=llm, retriever=FakeRetriever()).run()
    _prompt, kwargs = llm.generate_calls[0]
    assert kwargs.get("seed") == 42  # Kural 6
    assert kwargs.get("temperature") == 0.0


def test_result_to_dict_shape() -> None:
    d = SmokeRunner(llm=FakeLLM(output="TAMAM"), retriever=FakeRetriever()).run().to_dict()
    assert set(d) == {"verdict", "summary", "checks"}
    assert all(set(c) == {"name", "status", "detail"} for c in d["checks"])


# ── delege verdict → StageStatus eşlemesi ──────────────────────────────────────


def _smoke_ctx() -> RunContext:
    return RunContext(run_id="r", stage="smoke", run={}, params={}, store=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("pass", StageStatus.completed),
        ("skip", StageStatus.skipped),
        ("fail", StageStatus.failed),
    ],
)
def test_delegate_maps_verdict_to_status(
    monkeypatch: pytest.MonkeyPatch, verdict: str, expected: StageStatus
) -> None:
    monkeypatch.setattr(
        smoke_mod.SmokeRunner,
        "run",
        lambda self: SmokeResult(verdict, "özet", [SmokeCheck("backend", "pass")]),
    )
    res = delegates.smoke(_smoke_ctx())
    assert res.status == expected
    assert res.output["verdict"] == verdict
    assert res.message == "özet"
