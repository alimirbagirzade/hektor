"""`has_knowledge_card` reddedilmiş kartı SAYMAMALI.

Bulgu (2026-09-06): 21 boş kart `cards reject` ile reddedildi; ama `has_knowledge_card`
her kartı (rejected dahil) sayıyordu → o makaleler "kartı var" görünüp bir daha
kartlanmıyor, sonsuza dek okunmamış kalıyordu. Reddetmek = "yeniden üret" demektir.
"""

from __future__ import annotations

from pathlib import Path

from app.memory.sqlite_store import SqliteStore


def _store(tmp_path: Path) -> SqliteStore:
    st = SqliteStore(db_path=tmp_path / "k.db")
    st.upsert_paper(
        paper_id="paper_x",
        file_hash="h" * 20,
        source_path="x.pdf",
        title="X",
        authors="[]",
        year="2024",
        source="manual",
        n_pages=1,
        n_chars=100,
    )
    return st


def _kart(st: SqliteStore, cid: str) -> None:
    st.save_knowledge_card(
        card_id=cid,
        paper_id="paper_x",
        model="test",
        card={"paper_id": "paper_x", "title": "X", "main_claim": "iddia"},
        trust_level="draft",
        review_status="pending",
        lora_eligible=0,
        difficulty=0.1,
        stage="lora_phase_1",
    )


def test_reddedilen_kart_sayilmaz(tmp_path: Path) -> None:
    st = _store(tmp_path)
    assert st.has_knowledge_card("paper_x") is False
    _kart(st, "card_1")
    assert st.has_knowledge_card("paper_x") is True
    assert st.reject_card("card_1") is True
    assert st.has_knowledge_card("paper_x") is False  # ← makale yeniden kartlanabilir


def test_onayli_veya_bekleyen_kart_sayilir(tmp_path: Path) -> None:
    st = _store(tmp_path)
    _kart(st, "card_1")
    _kart(st, "card_2")
    st.reject_card("card_1")
    assert st.has_knowledge_card("paper_x") is True  # card_2 hâlâ pending
    assert st.approve_card("card_2") is True
    assert st.has_knowledge_card("paper_x") is True


def _bos_kart(st: SqliteStore, cid: str) -> None:
    st.save_knowledge_card(
        card_id=cid,
        paper_id="paper_x",
        model="test",
        card={"paper_id": "paper_x", "title": None, "main_claim": ""},
        trust_level="draft",
        review_status="pending",
        lora_eligible=0,
        difficulty=0.1,
        stage="lora_phase_1",
    )


def test_bos_pending_kart_sayilmaz(tmp_path: Path) -> None:
    """Bulgu (2026-09-06): boş `pending` kart makaleyi arayüzde 'KARTI GÖR' gösteriyor,
    kart '(başlıksız)' açılıyor ve 'ÜRET' düğmesi kaybolduğu için makale yeniden
    kartlanamıyordu. Boş kart = kart yok."""
    st = _store(tmp_path)
    _bos_kart(st, "card_bos")
    assert st.has_knowledge_card("paper_x") is False
    assert st.get_latest_knowledge_card("paper_x") is None


def test_en_yeni_icerikli_kart_doner(tmp_path: Path) -> None:
    """Boş kart daha yeni olsa bile içerikli canlı kart döner; reddedilmiş kart atlanır."""
    st = _store(tmp_path)
    _kart(st, "card_dolu")
    _bos_kart(st, "card_bos")  # daha yeni ama boş
    card = st.get_latest_knowledge_card("paper_x")
    assert card is not None and card["title"] == "X"
    assert st.has_knowledge_card("paper_x") is True
    st.reject_card("card_dolu")
    assert st.get_latest_knowledge_card("paper_x") is None
    assert st.has_knowledge_card("paper_x") is False
