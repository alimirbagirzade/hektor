"""Sohbetten öğrenme Faz 1 — değişmez veri sürümleri, eğitim verisi bağlantısı, kapılar.

Kabul şartları: eval örnekleri eğitim JSONL'ine ve kanonik SFT birleşimine girmez; sohbet payı
hem satır hem eğitim hedef token payında üst sınırlı (aile düzeyinde, tekrarla şişirme yok);
token yöntemi kaydedilir; hariç tutma eski anlık görüntüyü değiştirmez ama seçili sürümle
eğitim başlatmayı durdurur; aynı içerik ikinci sürüm üretmez; trading bölmesi zaman sıralı;
hiçbir adım eğitim başlatmaz.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from tests.chat_learning_helpers import (
    SUPPORTED_SENTENCE,
    iso,  # noqa: F401
    send,
)

from app.feedback.chat_store import ChatStore


@pytest.fixture
def store(iso):  # noqa: F811
    return ChatStore()


def _eligible(store, n: int, *, split_cycle=("train", "train", "eval"), domain=None):
    from app.feedback.learning import LearningService

    svc = LearningService(store)
    conv = store.create_conversation()["conversation_id"]
    out = []
    for i in range(n):
        t = send(store, conv, f"Benzersiz soru {i} {'x' * i} hakkında ayrıntı")
        c, _ = svc.correct(t["turn_id"], SUPPORTED_SENTENCE, domain=domain)
        assert c["status"] == "eligible", c["status_reason"]
        if split_cycle:
            store.upsert_family(c["family_id"], split=split_cycle[i % len(split_cycle)])
        out.append(store.get_candidate(c["candidate_id"]))
    return svc, out


def _counter(text: str) -> int:
    return len(text.split())


def test_version_files_train_eval_separate(store) -> None:
    from app.feedback.chat_dataset import create_version

    _eligible(store, 6)
    v, created = create_version(store, token_counter=_counter, token_method="test")
    assert created and v["version_id"] == "chat_v1"
    from app.config import get_settings

    d = get_settings().root / v["dir_path"]
    train = [json.loads(x) for x in (d / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    evals = [json.loads(x) for x in (d / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == 4 and len(evals) == 2
    assert all("messages" in r and r["metadata"]["source"] == "chat" for r in train)
    assert all("messages" not in r for r in evals)  # eval satırı SFT biçiminde değil
    fam_train = {r["metadata"]["family_id"] for r in train}
    fam_eval = {r["family_id"] for r in evals}
    assert not fam_train & fam_eval
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_sha256"] == hashlib.sha256((d / "train.jsonl").read_bytes()).hexdigest()
    assert manifest["stats"]["token_method"] == "test"


def test_same_content_does_not_create_second_version(store) -> None:
    from app.feedback.chat_dataset import create_version

    _eligible(store, 3)
    v1, c1 = create_version(store, token_counter=_counter, token_method="t")
    v2, c2 = create_version(store, token_counter=_counter, token_method="t")
    assert (c1, c2) == (True, False) and v1["version_id"] == v2["version_id"]
    assert len(store.list_versions()) == 1


def test_fewer_than_threshold_still_creates_version(store) -> None:
    from app.config import get_settings
    from app.feedback.chat_dataset import create_version

    assert get_settings().learning_min_families == 50
    _eligible(store, 1, split_cycle=("train",))
    v, created = create_version(store, token_counter=_counter, token_method="t")
    assert created and v["n_train"] == 1


def test_exclusion_keeps_snapshot_but_blocks_training(store) -> None:
    from app.config import get_settings
    from app.feedback.chat_dataset import (
        chat_selection_blockers,
        create_version,
        select_version,
    )

    svc, cands = _eligible(store, 3, split_cycle=("train",))
    v, _ = create_version(store, token_counter=_counter, token_method="t")
    d = get_settings().root / v["dir_path"]
    before = (d / "train.jsonl").read_bytes()
    select_version(v["version_id"], store)
    assert chat_selection_blockers() == []

    svc.set_excluded(cands[0]["turn_id"], True, "artık istemiyorum")
    assert (d / "train.jsonl").read_bytes() == before  # anlık görüntü değişmedi
    blockers = chat_selection_blockers()
    assert blockers and v["version_id"] in blockers[0] and "yeni veri sürümü" in blockers[0]
    from app.training import detached_launch

    assert any(
        v["version_id"] in b for b in detached_launch._pretrain_gate_blockers(get_settings())
    )
    with pytest.raises(ValueError):
        select_version(v["version_id"], store)  # geçersiz üyeli sürüm yeniden seçilemez
    # Yeni sürüm hariç tutulanı içermez ve seçilebilir.
    v2, created = create_version(store, token_counter=_counter, token_method="t")
    assert created and v2["n_train"] == 2
    select_version(v2["version_id"], store)
    assert chat_selection_blockers() == []


def test_edit_after_snapshot_blocks_training(store) -> None:
    from app.feedback.chat_dataset import chat_selection_blockers, create_version, select_version

    svc, cands = _eligible(store, 2, split_cycle=("train",))
    v, _ = create_version(store, token_counter=_counter, token_method="t")
    select_version(v["version_id"], store)
    svc.edit(cands[1]["candidate_id"], SUPPORTED_SENTENCE + " Ek cümle ama kaynakta yok burada.")
    assert "düzenlendi" in chat_selection_blockers()[0] or "durum" in chat_selection_blockers()[0]


def test_tampered_selection_file_blocks(store) -> None:
    from app.config import get_settings
    from app.feedback.chat_dataset import chat_selection_blockers, create_version, select_version

    _eligible(store, 2, split_cycle=("train",))
    v, _ = create_version(store, token_counter=_counter, token_method="t")
    select_version(v["version_id"], store)
    p = get_settings().root / v["dir_path"] / "train.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "\n{}\n", encoding="utf-8")
    assert "değişmiş" in chat_selection_blockers()[0]


# ── pay sınırı ───────────────────────────────────────────────────────────────


def _line(fid: str, answer: str, cid: str = "") -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": "s"},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {"source": "chat", "family_id": fid, "candidate_id": cid or fid},
        }
    )


def test_share_cap_rows_and_tokens_family_level() -> None:
    from app.feedback.chat_dataset import apply_chat_share

    base = [
        json.dumps({"messages": [{"role": "assistant", "content": "a " * 10}]}) for _ in range(40)
    ]  # 40 satır, 400 hedef token
    chat = [_line(f"f{i}", "b " * 5) for i in range(10)]  # 10 aile × 5 token
    chat += [_line("big", "c " * 100, "big1"), _line("big", "c " * 100, "big2")]
    kept, st = apply_chat_share(
        base, chat, max_share=0.10, seed=42, token_counter=_counter, token_method="kelime"
    )
    assert st["row_share"] <= 0.10 and st["token_share"] <= 0.10
    assert st["used_rows"] == len(kept) <= 4  # floor(0.111*40)=4
    assert "big" in st["dropped_families"]  # token sınırı aileyi bütün olarak dışarıda bıraktı
    assert len(set(kept)) == len(kept)  # tekrar yok
    assert st["token_method"] == "kelime"
    fams = {json.loads(x)["metadata"]["family_id"] for x in kept}
    assert all(sum(1 for x in kept if json.loads(x)["metadata"]["family_id"] == f) for f in fams)


def test_share_cap_zero_base_takes_nothing() -> None:
    from app.feedback.chat_dataset import apply_chat_share

    kept, st = apply_chat_share(
        [], [_line("f", "x y z")], max_share=0.1, seed=1, token_counter=_counter, token_method="k"
    )
    assert kept == [] and st["used_rows"] == 0


def test_assembly_reads_only_train_and_respects_cap(store, iso) -> None:  # noqa: F811
    from app.config import get_settings
    from app.feedback.chat_dataset import create_version, select_version
    from app.training.sft_assembly import assemble_sft_lines

    _eligible(store, 6)  # 4 train + 2 eval
    v, _ = create_version(store, token_counter=_counter, token_method="t")
    s = get_settings()
    # Taban yokken pay 0 → sohbet satırı girmez (tek başına eğitim seti olamaz).
    select_version(v["version_id"], store)
    res0 = assemble_sft_lines(s, discipline=False, distill=False, token_counter=_counter)
    assert res0.chat is not None and res0.chat["used_rows"] == 0
    # Sentetik taban ekle (60 satır) → sohbet ≤ %10.
    lora = s.root / "data" / "lora_sft"
    lora.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"Soru {i} nedir ve nasıl açıklanır?"},
                    {"role": "assistant", "content": f"Ayrıntılı cevap {i} " + "kelime " * 30},
                ],
                "metadata": {"source_id": f"paper_{i}", "enriched": True},
            }
        )
        for i in range(60)
    ]
    (lora / "synthetic_qa.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    res = assemble_sft_lines(s, discipline=False, distill=False, token_counter=_counter)
    chat_rows = [json.loads(x) for x in res.lines if '"source": "chat"' in x]
    assert res.chat["used_rows"] == len(chat_rows) == 4
    assert res.chat["row_share"] <= 0.10 and res.chat["token_share"] <= 0.10
    eval_ids = {
        json.loads(x)["id"]
        for x in (s.root / v["dir_path"] / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert not eval_ids & {r["metadata"]["candidate_id"] for r in chat_rows}
    assert all("messages" in json.loads(x) for x in res.lines)
    # Seçim kaldırılınca birleştirme öncekiyle aynı (sohbet yok).
    from app.feedback.chat_dataset import clear_selection

    clear_selection()
    res2 = assemble_sft_lines(s, discipline=False, distill=False, token_counter=_counter)
    assert res2.chat is None and not any('"source": "chat"' in x for x in res2.lines)


def test_binding_and_run_observation(store, iso) -> None:  # noqa: F811
    from app.config import get_settings
    from app.feedback.chat_dataset import (
        create_version,
        note_assembly,
        observe_runs,
        select_version,
        version_overview,
    )

    _eligible(store, 3, split_cycle=("train",))
    v, _ = create_version(store, token_counter=_counter, token_method="t")
    select_version(v["version_id"], store)
    s = get_settings()
    out = s.root / "data" / "lora_sft" / "lora_sft.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("x\n", encoding="utf-8")
    ids = [m["candidate_id"] for m in store.version_members(v["version_id"])]
    note_assembly(out, {"version_id": v["version_id"], "used_rows": 3, "used_candidate_ids": ids})
    ov = version_overview(store)
    assert ov["training_use"]["planned"] == 3 and ov["training_use"].get("completed", 0) == 0
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (s.root / "storage").mkdir(parents=True, exist_ok=True)
    (s.root / "storage" / "train_status.json").write_text(
        json.dumps(
            {
                "adapter": "hektor_lora_t",
                "started_at": "2026-10-05T10:00:00",
                "data_sha256": sha,
                "pid": 0,
            }
        ),
        encoding="utf-8",
    )
    observe_runs(store)
    ov = version_overview(store)
    assert ov["training_use"]["failed"] == 3 and ov["training_use"]["planned"] == 0
    (s.adapters_dir / "hektor_lora_t").mkdir(parents=True)
    (s.adapters_dir / "hektor_lora_t" / "run_complete.json").write_text("{}", encoding="utf-8")
    observe_runs(store)
    assert version_overview(store)["training_use"]["completed"] == 3


def test_trading_families_time_ordered(store) -> None:
    from app.feedback.chat_dataset import build_payload
    from app.feedback.learning import LearningService

    svc = LearningService(store)
    conv = store.create_conversation()["conversation_id"]
    for i in range(5):
        t = send(store, conv, f"RSI EMA trading stratejisi sorusu numara {i} {'y' * i}")
        c, _ = svc.correct(t["turn_id"], SUPPORTED_SENTENCE, domain="trading")
        assert store.get_family(c["family_id"])["split"] == "time"
        store.update_candidate(c["candidate_id"], as_of=f"2026-10-0{i + 1}T00:00:00+00:00")
    p = build_payload(store, token_counter=_counter, token_method="t")
    tts = p["stats"]["trading_time_split"]
    assert tts["families"] == 5 and tts["eval_families"] == 1
    assert tts["train_max_as_of"] < tts["eval_min_as_of"]


def test_no_training_started_and_no_rag_write(store, iso) -> None:  # noqa: F811
    from pathlib import Path

    from app.config import get_settings
    from app.feedback.chat_dataset import create_version

    _eligible(store, 3)
    create_version(store, token_counter=_counter, token_method="t")
    root = get_settings().root
    assert not (root / "storage" / "train_status.json").exists()
    assert not (root / "storage" / ".training_launching").exists()
    assert not (root / "data" / "lora_sft" / "lora_sft.jsonl").exists()  # kanonik dosyaya yazmaz
    src = "\n".join(p.read_text(encoding="utf-8") for p in Path("app/feedback").glob("*.py"))
    for banned in ("PaperIndexer", "ChromaStore", "add_documents", ".upsert(", "subprocess"):
        assert banned not in src, banned


def test_rollback_drops_only_chat_tables(tmp_path) -> None:
    import sqlite3

    from app.feedback.chat_store import ChatStore as CS
    from app.feedback.chat_store import drop_chat_tables
    from app.feedback.store import FeedbackStore

    db = tmp_path / "r.db"
    FeedbackStore(db).add(source="manual", question="q", bad_answer="", correction="c")
    CS(db).create_conversation()
    dropped = drop_chat_tables(db)
    assert "chat_turns" in dropped
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "feedback_corrections" in names and "chat_turns" not in names
        assert conn.execute("SELECT COUNT(*) FROM feedback_corrections").fetchone()[0] == 1
