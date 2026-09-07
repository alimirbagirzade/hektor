"""Regresyon: orkestrasyonun `approval` aşaması insan onayını TÜKETMEZ (Kural 8).

Bulgu: aşama `authorize_training_action(...)` çağırıyordu; bu da `require_fresh_approval`
üzerinden tek-kullanımlık onayı TÜKETİYOR ve onay yokken YENİ bir PENDING onay ÜRETİYORDU.
Sonuç: (1) hiçbir gerçek eğitime karşılık gelmeyen onay boşa harcanır — gerçek
`train --run` yolunda onay artık yoktur; (2) her başarısız resume yeni pending biriktirir.

Testler tamamen ÇEVRİMDIŞI: onay deposuna dokunulmaz, tüketen/üreten tüm fonksiyonlar
"çağrılırsa patla" tuzağıyla değiştirilir (monkeypatch). Gerçek eğitim BAŞLATILMAZ.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.orchestration.orchestrator import RunContext
from app.orchestration.pipeline import StageStatus

_KEY = "lora-trainer/train_run"


def _ctx() -> RunContext:
    return RunContext(
        run_id="r-test",
        stage="approval",
        run={"adapter_name": "a"},
        params={},
        store=None,  # type: ignore[arg-type]
    )


def _boom(name: str):
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name} çağrılmamalıydı (onay tüketimi/üretimi yasak)")

    return _raise


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch):
    """STOP_ALL kapalı, gözetimsiz mod KAPALI; tüketen/üreten uçlar tuzaklı."""
    import app.agents.runtime.approvals as approvals_mod
    import app.agents.runtime.supervisor as supervisor_mod
    import app.training.unattended_policy as policy_mod
    from app.config import settings as settings_mod

    monkeypatch.setattr(supervisor_mod, "is_stop_all_active", lambda *a, **k: False)
    # Onayı TÜKETEN veya YENİ pending ÜRETEN her yol tuzaklı:
    monkeypatch.setattr(approvals_mod, "require_fresh_approval", _boom("require_fresh_approval"))
    monkeypatch.setattr(approvals_mod, "request_approval", _boom("request_approval"))
    # Tek otorite tüketen çağrıyı içerir → aşama onu HİÇ çağırmamalı.
    monkeypatch.setattr(policy_mod, "authorize_training_action", _boom("authorize_training_action"))
    monkeypatch.setenv("HEKTOR_UNATTENDED_TRAINING_ENABLED", "false")
    settings_mod.get_settings.cache_clear()
    yield approvals_mod
    settings_mod.get_settings.cache_clear()


def test_fresh_approval_advances_stage_without_consuming(
    gate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) TAZE onay VARKEN aşama ilerler ve onay TÜKETİLMEZ (salt-okuma bakış)."""
    from app.orchestration import delegates

    seen: list[tuple[str, str]] = []

    def _peek(agent_id: str, action: str, store: Any = None) -> bool:
        seen.append((agent_id, action))
        return True

    monkeypatch.setattr(gate, "has_fresh_approval", _peek)

    res = delegates.approval(_ctx())

    assert res.status == StageStatus.completed
    assert res.output["authorization_mode"] == "human_approval"
    assert res.output["approval_consumed"] is False
    assert res.output["approval_key"] == _KEY
    # Gerçek eğitim yolu ile AYNI anahtar salt-okuma gözlendi (tüketim yok).
    assert seen == [("lora-trainer", "train_run")]


def test_missing_approval_blocks_and_creates_no_pending(
    gate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) Onay YOKKEN blocked döner ve YENİ pending onay OLUŞTURULMAZ."""
    from app.orchestration import delegates

    monkeypatch.setattr(gate, "has_fresh_approval", lambda *a, **k: False)

    res = delegates.approval(_ctx())

    assert res.status == StageStatus.blocked
    assert res.output["needs_approval"] is True
    assert res.output["approval_id"] == ""  # üretilmiş bir pending kimliği YOK
    assert "authorization_mode" not in res.output
    assert "train --run" in res.message


def test_repeated_resume_never_accumulates_pending(
    gate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Başarısız resume tekrarı pending biriktirmemeli (gözlenen 4 pending regresyonu)."""
    from app.orchestration import delegates

    monkeypatch.setattr(gate, "has_fresh_approval", lambda *a, **k: False)

    for _ in range(4):
        res = delegates.approval(_ctx())
        assert res.status == StageStatus.blocked
        assert res.output["approval_id"] == ""


def test_stop_all_blocks_even_with_fresh_approval(
    gate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STOP_ALL, taze onay olsa bile bloklar ve onaya DOKUNMAZ (kapı gevşemedi)."""
    import app.agents.runtime.supervisor as supervisor_mod
    from app.orchestration import delegates

    monkeypatch.setattr(supervisor_mod, "is_stop_all_active", lambda *a, **k: True)
    monkeypatch.setattr(gate, "has_fresh_approval", _boom("has_fresh_approval"))

    res = delegates.approval(_ctx())

    assert res.status == StageStatus.blocked
    assert res.output["stop_all"] is True


def test_unattended_mode_authorizes_without_touching_approvals(
    gate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gözetimsiz mod korunur: yetki verir ama onay deposuna DOKUNMAZ (tüketim taşınmadı)."""
    from app.config import settings as settings_mod
    from app.orchestration import delegates

    monkeypatch.setattr(gate, "has_fresh_approval", _boom("has_fresh_approval"))
    monkeypatch.setenv("HEKTOR_UNATTENDED_TRAINING_ENABLED", "true")
    settings_mod.get_settings.cache_clear()

    res = delegates.approval(_ctx())

    assert res.status == StageStatus.completed
    assert res.output["authorization_mode"] == "unattended_policy"
