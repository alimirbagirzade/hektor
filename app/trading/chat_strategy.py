"""chat_strategy.py — sohbet turu → strateji taslağı → kayıtlı (değişmez) strateji (Faz 2B).

- ``draft_from_turn`` : yerel model taslak çıkarır. Strateji KAYDEDİLMEZ; yalnız orijinal öneri
  (metin + deterministik okuma) ve modelin taslağı ayrı bir ÇEVİRİ kaydına yazılır. Model
  çağrısı sohbetle aynı kaynak korumasından (kira + ortak kilit) geçer.
- ``save_spec``       : kullanıcının onayladığı formu doğrulayıp kaydeder. İçerik aynıysa aynı
  kimlik döner. ``parent_id`` verilirse aynı AİLE (varyant); yoksa yeni aile.
- ``simplify``        : desteklenmeyen/istenmeyen kuralları AÇIKÇA çıkaran ayrı kimlikli
  "basitleştirilmiş strateji" (aynı aile, ``simplified_from`` + ``removed_rules``).
- ``review`` / ``approve`` : orijinal → nihai ve taslak → nihai farkları; kullanıcı HEPSİNİ
  görüp işaretlemeden test koşmaz (``strategy_testing.run_stage`` onayı ve özetini denetler).
Hiçbiri test koşturmaz; test ``strategy_testing.run_stage`` ile ayrı ve açık adımdır.
"""

from __future__ import annotations

import secrets
from typing import Any

from app.feedback.chat_store import ChatStore, utcnow
from app.trading.strategy_draft import DraftError, extract_draft
from app.trading.strategy_spec import TestableStrategy
from app.trading.strategy_store import StrategyStore


def draft_from_turn(
    turn_id: str,
    *,
    llm: Any = None,
    store: ChatStore | None = None,
    strategy_store: StrategyStore | None = None,
) -> dict:
    from app.feedback.model_activation import resolve_chat_tag
    from app.feedback.resource_guard import (
        acquire_chat_lease,
        check_chat_resources,
        release_chat_lease,
    )

    store = store or ChatStore()
    turn = store.get_turn(turn_id)
    if turn is None:
        raise KeyError(f"Tur bulunamadı: {turn_id}")
    answer = turn.get("raw_answer") or turn.get("answer") or ""
    if turn.get("status") != "answered" or not answer.strip():
        raise DraftError("Bu turda model cevabı yok; strateji çıkarılamaz.")
    tag = resolve_chat_tag("main")
    lease = None
    if llm is None:
        decision = check_chat_resources(tag=tag)
        if not decision.allowed:
            raise DraftError(f"Yerel model şu an kullanılamıyor: {decision.reason}")
        lease, why = acquire_chat_lease(f"draft-{turn_id}-{secrets.token_hex(3)}")
        if lease is None:
            raise DraftError(why)
        from app.brain.local_llm import LocalLLM

        llm = LocalLLM(model=tag)
    try:
        out = extract_draft(turn["question"], answer, llm=llm)
    finally:
        release_chat_lease(lease)
    from app.trading.strategy_translation import proposal_sha

    source = {
        "turn_id": turn_id,
        "conversation_id": turn["conversation_id"],
        "answer_model_tag": turn.get("model_tag", ""),
        "answer_model_digest": turn.get("model_digest", ""),
        "draft_model_tag": tag,
        "extracted_at": utcnow(),
    }
    tr = (strategy_store or StrategyStore()).save_translation(
        turn_id=turn_id,
        original_text=answer,
        original_sha=proposal_sha(answer),
        proposal=out["proposal"],
        model_draft=out["model_draft"],
        draft=out["draft"],
        pending=out["pending"],
        raw_model_output=out["raw_model_output"],
        source=source,
        created_at=utcnow(),
    )
    source["translation_id"] = tr["translation_id"]
    out["source"] = source
    out["translation_id"] = tr["translation_id"]
    return out


def save_spec(
    spec_data: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
    parent_id: str = "",
    origin: str = "draft",
    store: StrategyStore | None = None,
) -> dict[str, Any]:
    store = store or StrategyStore()
    spec = TestableStrategy.model_validate(spec_data)
    family = ""
    src = dict(source or {})
    if parent_id:
        parent = store.get_strategy(parent_id)
        if parent is None:
            raise KeyError(f"Üst strateji yok: {parent_id}")
        family = parent["family_id"]
        src = {**parent["source"], **src}
    tr_id = str(src.get("translation_id") or "")
    if tr_id:
        # Çeviri kaydı sunucuda doğrulanır: başka bir turun önerisi bu stratejiye bağlanamaz.
        tr = store.get_translation(tr_id)
        if tr is None:
            raise KeyError(f"Çeviri kaydı yok: {tr_id}")
        if src.get("turn_id") and tr["turn_id"] != src["turn_id"]:
            raise ValueError("Çeviri kaydı bu sohbet turuna ait değil.")
        src["turn_id"] = tr["turn_id"]
    family = family or "sfam_" + secrets.token_hex(6)
    rec, created = store.save_strategy(
        spec.strategy_id(),
        family_id=family,
        parent_id=parent_id,
        name=spec.name,
        spec=spec.model_dump(mode="json"),
        source=src,
        origin=origin,
    )
    return {
        **rec,
        "created": created,
        "readable": spec.readable(),
        "testable": not spec.important_unsupported(),
    }


def simplify(
    strategy_id: str,
    *,
    remove_rules: list[str],
    drop_unsupported: list[str],
    store: StrategyStore | None = None,
) -> dict[str, Any]:
    """Kuralları AÇIKÇA çıkaran ayrı kimlikli basitleştirilmiş strateji (aynı aile)."""
    store = store or StrategyStore()
    rec = store.get_strategy(strategy_id)
    if rec is None:
        raise KeyError(f"Strateji yok: {strategy_id}")
    spec = TestableStrategy.model_validate(rec["spec"])
    remove = {r.strip() for r in remove_rules if r.strip()}
    drop = {u.strip() for u in drop_unsupported if u.strip()}
    unknown = remove - set(spec.entry_rules) - set(spec.exit_rules)
    unknown |= drop - {u.text for u in spec.unsupported_rules}
    if unknown:
        raise ValueError(f"Stratejide olmayan kural: {sorted(unknown)[:3]}")
    if not remove and not drop:
        raise ValueError("Basitleştirmek için en az bir kural seçin.")
    data = spec.model_dump(mode="json")
    data["entry_rules"] = [r for r in spec.entry_rules if r not in remove]
    data["exit_rules"] = [r for r in spec.exit_rules if r not in remove]
    data["unsupported_rules"] = [
        u.model_dump() for u in spec.unsupported_rules if u.text not in drop
    ]
    data["simplified_from"] = strategy_id
    data["removed_rules"] = sorted(remove | drop)
    data["name"] = (spec.name + "_basitlestirilmis")[:120]
    if not data["entry_rules"]:
        raise ValueError("Giriş kuralı kalmadı; basitleştirilmiş strateji tanımsız.")
    return save_spec(data, parent_id=strategy_id, origin="simplified", store=store)


# ── nihai strateji onayı (orijinal öneri ↔ taslak ↔ nihai) ───────────────────


def review(strategy_id: str, *, store: StrategyStore | None = None) -> dict[str, Any]:
    """Onay ekranı: farklar + mevcut onayın hâlâ geçerli olup olmadığı."""
    from app.trading.strategy_translation import build_review

    store = store or StrategyStore()
    rec = store.get_strategy(strategy_id)
    if rec is None:
        raise KeyError(f"Strateji yok: {strategy_id}")
    tr = store.get_translation(str((rec.get("source") or {}).get("translation_id") or ""))
    out = build_review(tr, rec["spec"])
    appr = store.latest_approval(strategy_id)
    out["strategy_id"] = strategy_id
    out["approval"] = appr
    out["approved"] = bool(appr and appr["review_sha"] == out["review_sha"])
    out["n_changes"] = len(out["items"])
    return out


def approve(
    strategy_id: str,
    *,
    review_sha: str,
    acknowledged: list[str],
    note: str = "",
    store: StrategyStore | None = None,
) -> dict[str, Any]:
    """(İnsan) Nihai stratejiyi onayla: HER fark öğesi işaretlenmiş olmalı."""
    store = store or StrategyStore()
    rv = review(strategy_id, store=store)
    if review_sha != rv["review_sha"]:
        raise ValueError("Fark listesi değişti — güncel listeyi okuyup yeniden onaylayın.")
    want = {i["key"] for i in rv["items"]}
    got = {str(k) for k in acknowledged}
    missing = want - got
    if missing:
        raise ValueError(
            f"{len(missing)} fark işaretlenmedi — test başlatmadan önce her değişikliği görün."
        )
    if got - want:
        raise ValueError("Listede olmayan fark işaretlendi.")
    if rv["conflicts"] and len((note or "").strip()) < 10:
        raise ValueError(
            "Orijinal öneride çelişki var: nihai değeri neden seçtiğinizi yazın (≥10 karakter)."
        )
    tr_id = rv.get("translation_id", "")
    return store.save_approval(
        strategy_id=strategy_id,
        translation_id=tr_id,
        review_sha=rv["review_sha"],
        review={k: rv[k] for k in ("original_vs_final", "draft_vs_final", "conflicts", "note")},
        acknowledged=sorted(got),
        n_changes=len(rv["items"]),
        note=(note or "").strip()[:1000],
    )
