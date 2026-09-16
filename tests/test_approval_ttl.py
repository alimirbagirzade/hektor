"""Taze onay ÖMRÜ (Kural 8) — bayat onay gerçek eğitimi yetkilendiremez. Çevrimdışı.

Kademe 2 bulgusu (2026-09-15, gerçek DB kanıtı): 2026-09-08'de ``hektor_lora_v8_4b``
(600 adım) için verilip hiç tüketilmemiş onay, BİR HAFTA sonra web'den başlatılan BAŞKA
bir eğitim (``hektor_lora``, 500 adım) tarafından tüketildi. "Tek kullanımlık taze onay"
fiilen kalıcı yetkiye dönüşmüştü: tüketim damgası vardı ama yaş kontrolü yoktu.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from app.memory.sqlite_store import APPROVAL_TTL_HOURS, SqliteStore

_AGENT = "lora-trainer"
_ACTION = "train_run"


def _store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(db_path=tmp_path / "approvals.db")


def _approved_request(store: SqliteStore, approval_id: str, *, age_hours: float) -> None:
    """`age_hours` saat ÖNCE onaylanmış bir istek kur (istek anı da o kadar eski)."""
    decided = dt.datetime.now(dt.UTC) - dt.timedelta(hours=age_hours)
    store.create_approval_request(
        approval_id=approval_id,
        agent_id=_AGENT,
        action=_ACTION,
        summary="Gerçek LoRA eğitimi: test_adapter",
        risk="critical",
        requested_at=decided.isoformat(),
    )
    store.update_approval_request(
        approval_id,
        status="approved",
        decided_at=decided.isoformat(),
        decided_by="user",
    )


def test_fresh_approval_is_consumed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _approved_request(store, "apr_fresh", age_hours=0.5)

    assert store.find_fresh_approval(_AGENT, _ACTION) is not None
    consumed = store.consume_fresh_approval(_AGENT, _ACTION)
    assert consumed is not None
    assert consumed["approval_id"] == "apr_fresh"
    # Tek kullanımlık: ikinci çağrı aynı onayı tüketemez.
    assert store.consume_fresh_approval(_AGENT, _ACTION) is None


def test_stale_approval_is_not_consumable(tmp_path: Path) -> None:
    """TTL'den eski onay ne bulunur ne tüketilir (v8 onayının bir hafta sonra kullanılması)."""
    store = _store(tmp_path)
    _approved_request(store, "apr_stale", age_hours=APPROVAL_TTL_HOURS + 1)

    assert store.find_fresh_approval(_AGENT, _ACTION) is None
    assert store.consume_fresh_approval(_AGENT, _ACTION) is None
    # Bayat onay "tüketildi" diye damgalanmaz — kullanıcı onu hâlâ görebilir/inceleyebilir.
    assert store.get_approval_request("apr_stale")["consumed_at"] is None


def test_stale_approval_does_not_shadow_fresh_one(tmp_path: Path) -> None:
    """Bayat onay sıradaki taze onayı gölgelemez (en yeni kayıt seçilir)."""
    store = _store(tmp_path)
    _approved_request(store, "apr_old", age_hours=APPROVAL_TTL_HOURS + 24)
    _approved_request(store, "apr_new", age_hours=0.1)

    consumed = store.consume_fresh_approval(_AGENT, _ACTION)
    assert consumed is not None and consumed["approval_id"] == "apr_new"


def test_approval_without_decision_time_is_not_fresh(tmp_path: Path) -> None:
    """Karar zamanı yoksa onay taze SAYILMAZ (fail-closed — Kural 2)."""
    store = _store(tmp_path)
    store.create_approval_request(
        approval_id="apr_nodate",
        agent_id=_AGENT,
        action=_ACTION,
        summary="karar zamanı yazılmamış",
        risk="critical",
    )
    store.update_approval_request("apr_nodate", status="approved", decided_by="user")

    assert store.find_fresh_approval(_AGENT, _ACTION) is None
    assert store.consume_fresh_approval(_AGENT, _ACTION) is None
