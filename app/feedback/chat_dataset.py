"""chat_dataset.py — sohbet adaylarından DEĞİŞMEZ veri sürümleri + eğitim verisine bağlantı.

Akış (hiçbiri eğitim başlatmaz — Kural 8):
1. ``preview`` : uygun adaylardan train/eval bölmesi + kanonik setle birleşince satır/token
                 payı tahmini (yazma yok).
2. ``create_version`` (insan): ``data/learning/chat_datasets/<chat_vN>/`` altına
   ``train.jsonl`` (SFT biçimi), ``eval.jsonl`` (SFT DEĞİL — eğitime girmez) ve
   ``manifest.json`` yazar; ``version_store``'a kaydeder. Aynı içerik → aynı sürüm.
3. ``select_version`` (``scripts/assemble_sft.py --chat-dataset``): seçimi
   ``data/lora_sft/chat_selection.json``'a yazar. Kanonik birleştirme (``assemble_sft_lines``)
   YALNIZ bu dosya varsa ve YALNIZ ``train.jsonl``'i okur; pay sınırı aile düzeyinde uygulanır.
4. ``chat_selection_blockers``: eğitim başlatılırken (pretrain-gate + web/CLI başlatma kapısı)
   seçili sürümde sonradan reddedilen / hariç tutulan / düzenlenen kayıt varsa DURDURUR.
   Anlık görüntüler değiştirilmez; yeni sürüm oluşturulması istenir.

Bölme: train/eval ataması aile düzeyinde ve kalıcıdır (``learning_families.split``). Trading
aileleri (``time``) her sürümde zaman sırasıyla bölünür: en yeni aileler değerlendirmeye gider,
eğitim ailelerinin hepsi değerlendirme ailelerinden ESKİDİR.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.feedback.chat_store import ChatStore, dumps, utcnow

TokenCounter = Callable[[str], int]

SELECTION_FILE = "chat_selection.json"


# ── token sayımı ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _hf_tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, local_files_only=True)


def make_token_counter(mode: str | None = None) -> tuple[TokenCounter, str]:
    """(sayaç, yöntem açıklaması). Tokenizer yalnız YEREL önbellekten; yoksa yaklaşık."""
    s = get_settings()
    mode = (mode or s.learning_tokenizer).strip().lower()
    if mode != "approx":
        try:
            tok = _hf_tokenizer(s.peft_base_model)

            def _count(text: str) -> int:
                return len(tok.encode(text or "", add_special_tokens=False))

            return _count, f"tokenizer: {s.peft_base_model} (yerel önbellek)"
        except Exception:
            pass

    def _approx(text: str) -> int:
        return math.ceil(len(text or "") / 4)

    why = (
        "ayar: HEKTOR_LEARNING_TOKENIZER=approx"
        if mode == "approx"
        else "base tokenizer yerelde bulunamadı"
    )
    return _approx, f"yaklaşık: karakter/4 ({why})"


def _assistant_text(line: str) -> str:
    try:
        msgs = json.loads(line).get("messages") or []
    except (ValueError, AttributeError):
        return ""
    return "".join(str(m.get("content", "")) for m in msgs if m.get("role") == "assistant")


def _meta(line: str) -> dict[str, Any]:
    try:
        meta = json.loads(line).get("metadata") or {}
    except (ValueError, AttributeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def is_chat_line(line: str) -> bool:
    return _meta(line).get("source") == "chat"


# ── pay sınırı (aile düzeyinde) ──────────────────────────────────────────────


def apply_chat_share(
    base_lines: list[str],
    chat_lines: list[str],
    *,
    max_share: float,
    seed: int,
    token_counter: TokenCounter,
    token_method: str,
) -> tuple[list[str], dict[str, Any]]:
    """Sohbet satırlarını hem SATIR hem EĞİTİM HEDEF TOKEN payında ``max_share``'e kadar al.

    Pay = sohbet / (taban + sohbet). Eksiltme AİLE düzeyinde, seed'li deterministik sırayla;
    sığmayan aile atlanır. Tekrarla çoğaltma YOK.
    """
    base_rows = len(base_lines)
    base_tok = sum(token_counter(_assistant_text(ln)) for ln in base_lines)
    ratio = max_share / (1.0 - max_share) if 0 < max_share < 1 else 0.0
    allow_rows = math.floor(ratio * base_rows + 1e-9)
    allow_tok = math.floor(ratio * base_tok + 1e-9)

    fams: dict[str, list[str]] = {}
    for ln in chat_lines:
        fams.setdefault(str(_meta(ln).get("family_id") or ""), []).append(ln)
    order = sorted(fams, key=lambda f: hashlib.sha256(f"{seed}:{f}".encode()).hexdigest())
    kept: list[str] = []
    used_rows = used_tok = 0
    dropped: list[str] = []
    for fid in order:
        lines = fams[fid]
        rows = len(lines)
        tok = sum(token_counter(_assistant_text(ln)) for ln in lines)
        if used_rows + rows <= allow_rows and used_tok + tok <= allow_tok:
            kept.extend(lines)
            used_rows += rows
            used_tok += tok
        else:
            dropped.append(fid)
    kept_set = set(kept)
    kept = [ln for ln in chat_lines if ln in kept_set]  # özgün sıra korunur
    offered_tok = sum(token_counter(_assistant_text(ln)) for ln in chat_lines)
    total_rows = base_rows + used_rows
    total_tok = base_tok + used_tok
    return kept, {
        "max_share": max_share,
        "rule": "satır payı VE eğitim hedef token payı ≤ üst sınır; eksiltme aile düzeyinde, "
        "tekrarla çoğaltma yok",
        "token_method": token_method,
        "base_rows": base_rows,
        "base_target_tokens": base_tok,
        "offered_rows": len(chat_lines),
        "offered_target_tokens": offered_tok,
        "offered_families": len(fams),
        "used_rows": used_rows,
        "used_target_tokens": used_tok,
        "used_families": len(fams) - len(dropped),
        "dropped_families": dropped,
        "row_share": round(used_rows / total_rows, 4) if total_rows else 0.0,
        "token_share": round(used_tok / total_tok, 4) if total_tok else 0.0,
        "used_candidate_ids": sorted(
            {str(_meta(ln).get("candidate_id") or "") for ln in kept} - {""}
        ),
    }


# ── sürüm içeriği ────────────────────────────────────────────────────────────


def _train_line(cand: dict[str, Any], turn: dict[str, Any], vclass: str) -> str:
    from app.lora.dataset_builder import LoRAExample

    return LoRAExample(
        messages=[
            {"role": "system", "content": turn["system_prompt"]},
            {"role": "user", "content": turn["user_prompt"]},
            {"role": "assistant", "content": cand["target_text"]},
        ],
        metadata={
            "source": "chat",
            "source_id": f"chat_family:{cand['family_id']}",
            "family_id": cand["family_id"],
            "candidate_id": cand["candidate_id"],
            "turn_id": cand["turn_id"],
            "kind": cand["kind"],
            "domain": cand["domain"],
            "verification_class": vclass,
            "model_tag": turn["model_tag"],
            "as_of": cand["as_of"],
        },
    ).to_jsonl_line()


def _eval_line(cand: dict[str, Any], turn: dict[str, Any], vclass: str) -> str:
    # Bilinçli olarak SFT biçiminde DEĞİL (``messages`` yok) → yanlışlıkla eğitime giremez.
    return json.dumps(
        {
            "id": cand["candidate_id"],
            "source": "chat",
            "family_id": cand["family_id"],
            "question": turn["question"],
            "prompt": {"system": turn["system_prompt"], "user": turn["user_prompt"]},
            "reference": cand["target_text"],
            "domain": cand["domain"],
            "verification_class": vclass,
            "model_tag": turn["model_tag"],
            "as_of": cand["as_of"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_payload(
    store: ChatStore,
    *,
    include_human: bool = True,
    token_counter: TokenCounter | None = None,
    token_method: str = "",
) -> dict[str, Any]:
    """Uygun adaylardan train/eval satırlarını kur (yazma yok, deterministik)."""
    s = get_settings()
    if token_counter is None:
        token_counter, token_method = make_token_counter()
    rows: list[tuple[dict, dict, dict, str]] = []  # (cand, turn, fam, class)
    skipped_human = 0
    for c in store.list_candidates(status="eligible"):
        vclass = (c["verification"] or {}).get("class") or "auto"
        if vclass == "human" and not include_human:
            skipped_human += 1
            continue
        turn = store.get_turn(c["turn_id"])
        fam = store.get_family(store.resolve_family(c["family_id"])) if c["family_id"] else None
        if turn is None or fam is None or turn["excluded"]:
            continue
        rows.append((c, turn, fam, vclass))

    # Trading aileleri: zaman sırasıyla (aile = ilk görüldüğü an), en yeni pay → eval.
    time_fams: dict[str, str] = {}
    for c, _t, fam, _v in rows:
        if fam["split"] == "time":
            fid = fam["family_id"]
            time_fams[fid] = min(time_fams.get(fid, c["as_of"]), c["as_of"])
    ordered_time = sorted(time_fams, key=lambda f: (time_fams[f], f))
    n_eval_time = math.floor(len(ordered_time) * s.learning_eval_ratio)
    time_eval = set(ordered_time[len(ordered_time) - n_eval_time :]) if n_eval_time else set()

    train: list[tuple[str, str, dict]] = []
    evals: list[tuple[str, str, dict]] = []
    members: list[dict[str, Any]] = []
    for c, turn, fam, vclass in sorted(rows, key=lambda r: (r[0]["as_of"], r[0]["candidate_id"])):
        fid = fam["family_id"]
        c = {**c, "family_id": fid}
        split = fam["split"]
        if split == "time":
            split = "eval" if fid in time_eval else "train"
        member = {
            "candidate_id": c["candidate_id"],
            "split": split,
            "family_id": fid,
            "target_sha": c["target_sha"],
            "verification_class": vclass,
        }
        members.append(member)
        if split == "eval":
            evals.append((c["as_of"], _eval_line(c, turn, vclass), member))
        else:
            train.append((c["as_of"], _train_line(c, turn, vclass), member))

    train_lines = [ln for _, ln, _ in train]
    eval_lines = [ln for _, ln, _ in evals]
    time_train = [a for a, _, m in train if m["family_id"] in time_fams]
    time_eval_asof = [a for a, _, m in evals if m["family_id"] in time_fams]
    tokens = sum(token_counter(_assistant_text(ln)) for ln in train_lines)
    params = {
        "include_human": include_human,
        "eval_ratio": s.learning_eval_ratio,
        "seed": s.learning_seed,
        "family_jaccard": s.learning_family_jaccard,
    }
    return {
        "train_lines": train_lines,
        "eval_lines": eval_lines,
        "members": members,
        "params": params,
        "stats": {
            "n_train": len(train_lines),
            "n_eval": len(eval_lines),
            "n_train_families": len({m["family_id"] for _, _, m in train}),
            "n_eval_families": len({m["family_id"] for _, _, m in evals}),
            "skipped_human": skipped_human,
            "train_target_tokens": tokens,
            "token_method": token_method,
            "trading_time_split": {
                "families": len(ordered_time),
                "eval_families": len(time_eval),
                "train_max_as_of": max(time_train) if time_train else "",
                "eval_min_as_of": min(time_eval_asof) if time_eval_asof else "",
            },
        },
    }


def _content_sha(train_lines: list[str], eval_lines: list[str], params: dict) -> str:
    h = hashlib.sha256()
    h.update("\n".join(train_lines).encode("utf-8"))
    h.update(b"\n--eval--\n")
    h.update("\n".join(eval_lines).encode("utf-8"))
    h.update(b"\n--params--\n")
    h.update(dumps(params).encode("utf-8"))
    return h.hexdigest()


def _canonical_base_lines() -> list[str]:
    p = get_settings().root / "data" / "lora_sft" / "lora_sft.jsonl"
    if not p.exists():
        return []
    return [
        ln
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not is_chat_line(ln)
    ]


def preview(
    store: ChatStore | None = None,
    *,
    include_human: bool = True,
    token_counter: TokenCounter | None = None,
    token_method: str = "",
) -> dict[str, Any]:
    """Sürüm oluşturmadan önce özet (yazma yok)."""
    store = store or ChatStore()
    if token_counter is None:
        token_counter, token_method = make_token_counter()
    payload = build_payload(
        store, include_human=include_human, token_counter=token_counter, token_method=token_method
    )
    s = get_settings()
    _kept, share = apply_chat_share(
        _canonical_base_lines(),
        payload["train_lines"],
        max_share=s.learning_chat_max_share,
        seed=s.learning_seed,
        token_counter=token_counter,
        token_method=token_method,
    )
    content_sha = _content_sha(payload["train_lines"], payload["eval_lines"], payload["params"])
    existing = store.find_version_by_sha(content_sha)
    excluded = sum(1 for c in store.list_candidates() if c["status"] == "excluded")
    leaks = sum(1 for c in store.list_candidates() if c["status"] == "leak")
    return {
        **payload["stats"],
        "params": payload["params"],
        "projected_share": share,
        "excluded_candidates": excluded,
        "leak_candidates": leaks,
        "existing_version": existing["version_id"] if existing else "",
        "min_families_setting": s.learning_min_families,
        "note": "Eğitim BAŞLATMAZ. Eval örnekleri eğitim dosyasına ve kanonik SFT birleşimine "
        "girmez.",
    }


def create_version(
    store: ChatStore | None = None,
    *,
    include_human: bool = True,
    token_counter: TokenCounter | None = None,
    token_method: str = "",
    recheck: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Değişmez sürüm yaz (aynı içerik → mevcut sürüm döner, yeni klasör açılmaz)."""
    from app.feedback.learning import LearningService

    store = store or ChatStore()
    if recheck:
        LearningService(store).recheck_all()  # sızıntı/çatışma/hariç güncel kurallarla
    if token_counter is None:
        token_counter, token_method = make_token_counter()
    payload = build_payload(
        store, include_human=include_human, token_counter=token_counter, token_method=token_method
    )
    if not payload["train_lines"] and not payload["eval_lines"]:
        raise ValueError("Uygun (eligible) aday yok — veri sürümü oluşturulmadı.")
    content_sha = _content_sha(payload["train_lines"], payload["eval_lines"], payload["params"])
    existing = store.find_version_by_sha(content_sha)
    if existing is not None:
        return existing, False

    s = get_settings()
    base = s.learning_dir / "chat_datasets"
    base.mkdir(parents=True, exist_ok=True)
    train_text = "\n".join(payload["train_lines"]) + ("\n" if payload["train_lines"] else "")
    eval_text = "\n".join(payload["eval_lines"]) + ("\n" if payload["eval_lines"] else "")
    train_sha = hashlib.sha256(train_text.encode("utf-8")).hexdigest()
    for _attempt in range(5):
        seq = store.next_version_seq()
        version_id = f"chat_v{seq}"
        final = base / version_id
        tmp = base / f".tmp-{version_id}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "train.jsonl").write_bytes(train_text.encode("utf-8"))  # LF sabit (özet)
        (tmp / "eval.jsonl").write_bytes(eval_text.encode("utf-8"))
        manifest = {
            "version_id": version_id,
            "created_at": utcnow(),
            "content_sha256": content_sha,
            "train_sha256": train_sha,
            "eval_sha256": hashlib.sha256(eval_text.encode("utf-8")).hexdigest(),
            "params": payload["params"],
            "stats": payload["stats"],
            "members": payload["members"],
            "note": "Değişmez anlık görüntü. eval.jsonl eğitim dosyası DEĞİLDİR. Hariç tutma/ret "
            "bu dosyaları değiştirmez; seçiliyken eğitim başlatma kapısı geçersiz kayıtları "
            "tespit edip durdurur.",
        }
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        version = {
            "version_id": version_id,
            "seq": seq,
            "content_sha256": content_sha,
            "dir_path": str(final.relative_to(s.root))
            if final.is_relative_to(s.root)
            else str(final),
            "train_sha256": train_sha,
            "n_train": payload["stats"]["n_train"],
            "n_eval": payload["stats"]["n_eval"],
            "n_train_families": payload["stats"]["n_train_families"],
            "n_eval_families": payload["stats"]["n_eval_families"],
            "params": {**payload["params"], "token_method": token_method},
            "created_at": manifest["created_at"],
        }
        if final.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            continue  # yarım kalmış eski bir klasör seq'i tutuyor — sıradakini dene
        row, inserted = store.insert_version(version, payload["members"])
        if not inserted:
            shutil.rmtree(tmp, ignore_errors=True)
            if row["content_sha256"] == content_sha:
                return row, False
            continue  # seq yarışı — yeniden dene
        os.replace(tmp, final)
        with contextlib.suppress(Exception):
            from app.registry.version_store import RegistryStore

            reg = RegistryStore().register_dataset(
                name=version_id,
                path=str(final / "train.jsonl"),
                source_type="chat_sft",
                content_hash=content_sha,
                n_records=payload["stats"]["n_train"],
            )
            store.set_version_registry(version_id, str(reg.get("dataset_version_id", "")))
        got = store.get_version(version_id)
        assert got is not None
        return got, True
    raise RuntimeError("Veri sürümü kimliği ayrılamadı (eşzamanlı oluşturma).")


# ── seçim + kanonik birleştirme bağlantısı ───────────────────────────────────


def selection_path() -> Path:
    return get_settings().root / "data" / "lora_sft" / SELECTION_FILE


def read_selection() -> dict[str, Any] | None:
    p = selection_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{SELECTION_FILE} okunamadı: {exc}") from exc
    return data if isinstance(data, dict) else None


def select_version(version_id: str, store: ChatStore | None = None) -> dict[str, Any]:
    store = store or ChatStore()
    v = store.get_version(version_id)
    if v is None:
        raise KeyError(f"Sohbet veri sürümü bulunamadı: {version_id}")
    blockers = version_blockers(version_id, store)
    if blockers:
        raise ValueError(blockers[0])
    sel = {
        "version_id": version_id,
        "train_path": str(Path(v["dir_path"]) / "train.jsonl"),
        "train_sha256": v["train_sha256"],
        "selected_at": utcnow(),
    }
    p = selection_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")
    return sel


def clear_selection() -> bool:
    p = selection_path()
    if p.exists():
        p.unlink()
        return True
    return False


def load_selected_train_lines() -> tuple[list[str], dict[str, Any]] | None:
    """Seçili sürümün train satırları (seçim yoksa None). Özet uyuşmazsa ValueError."""
    sel = read_selection()
    if not sel:
        return None
    root = get_settings().root
    path = Path(sel["train_path"])
    path = path if path.is_absolute() else root / path
    if not path.exists():
        raise ValueError(f"Seçili sohbet sürümünün train dosyası yok: {path}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sel.get("train_sha256"):
        raise ValueError(
            f"Seçili sohbet sürümü {sel.get('version_id')} dosyası değişmiş (özet uyuşmuyor)."
        )
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
    return lines, sel


def version_invalid_members(version_id: str, store: ChatStore) -> list[dict[str, Any]]:
    """Sürümde artık geçerli olmayan üyeler (ret/hariç/sızıntı/çatışma/düzenlendi/silindi)."""
    bad = []
    for m in store.version_members(version_id):
        c = store.get_candidate(m["candidate_id"])
        if c is None:
            bad.append({**m, "why": "aday yok"})
            continue
        turn = store.get_turn(c["turn_id"])
        if turn is None or turn["excluded"]:
            bad.append({**m, "why": "hariç tutuldu"})
        elif c["target_sha"] != m["target_sha"]:
            bad.append({**m, "why": "hedef metin sonradan düzenlendi"})
        elif c["status"] != "eligible":
            bad.append({**m, "why": f"durum artık '{c['status']}'"})
    return bad


def version_blockers(version_id: str, store: ChatStore) -> list[str]:
    bad = version_invalid_members(version_id, store)
    if not bad:
        return []
    whys = sorted({b["why"] for b in bad})
    return [
        f"Seçili sohbet veri sürümü {version_id} içinde artık geçersiz {len(bad)} kayıt var "
        f"({'; '.join(whys)}). Eğitim DURDURULDU — Öğrenme panelinden yeni veri sürümü oluşturup "
        "`uv run python scripts/assemble_sft.py --chat-dataset <yeni>` ile seçin "
        "(ya da `--no-chat` ile sohbet verisini çıkarın)."
    ]


def chat_selection_blockers() -> list[str]:
    """Eğitim başlatma kapıları için: seçili sürüm geçerli mi? (seçim yoksa boş liste)."""
    try:
        sel = read_selection()
        if not sel:
            return []
        load_selected_train_lines()
        return version_blockers(str(sel.get("version_id", "")), ChatStore())
    except Exception as exc:  # okunamıyorsa kapalı kal (Kural 2)
        return [f"Sohbet veri seçimi doğrulanamadı — eğitim durduruldu: {exc}"]


def note_assembly(out_path: Path, chat_stats: dict[str, Any] | None) -> None:
    """Kanonik dosya sohbet sürümüyle yazıldıysa bağlantıyı kaydet (= "planlandı")."""
    if not chat_stats or not chat_stats.get("version_id") or not chat_stats.get("used_rows"):
        return
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    ChatStore().add_binding(str(chat_stats["version_id"]), sha, chat_stats)


# ── koşu gözlemi + panel özeti ───────────────────────────────────────────────


def observe_runs(store: ChatStore | None = None) -> list[dict[str, Any]]:
    """Bağlanmış veriyle başlatılan koşuları kaydet/güncelle (``train_status.json`` +
    ``run_complete.json``). Salt dosya okur; eğitimi etkilemez."""
    from app.training.detached_launch import _pid_alive, read_detached_training_status

    store = store or ChatStore()
    s = get_settings()
    bindings = store.list_bindings()
    by_sha: dict[str, list[str]] = {}
    for b in bindings:
        by_sha.setdefault(b["lora_sft_sha256"], []).append(b["version_id"])
    st = read_detached_training_status(s.root)
    sha = str(st.get("data_sha256") or "")
    current_key = ""
    if sha and sha in by_sha:
        adapter = str(st.get("adapter") or "")
        started = str(st.get("started_at") or "")
        current_key = f"{adapter}|{started}"
        pid = st.get("pid")
        complete = (s.adapters_dir / adapter / "run_complete.json").is_file()
        state = (
            "completed"
            if complete
            else ("running" if isinstance(pid, int) and _pid_alive(pid) else "failed")
        )
        for vid in by_sha[sha]:
            store.upsert_run(
                f"{current_key}|{vid}",
                version_id=vid,
                lora_sft_sha256=sha,
                adapter=adapter,
                started_at=started,
                state=state,
            )
    for r in store.list_runs():
        if r["run_key"].startswith(current_key + "|") and current_key:
            continue
        if r["state"] == "running":
            complete = (s.adapters_dir / r["adapter"] / "run_complete.json").is_file()
            store.upsert_run(r["run_key"], state="completed" if complete else "failed")
    return store.list_runs()


def version_overview(store: ChatStore) -> dict[str, Any]:
    """Panel için sürümler + seçim + koşu durumuna göre eğitim kullanımı (yazma yok)."""
    try:
        sel = read_selection()
    except ValueError:
        sel = {"version_id": "?", "error": "seçim dosyası okunamadı"}
    bindings = store.list_bindings()
    runs = store.list_runs()
    versions = []
    use: dict[str, set[str]] = {
        "planned": set(),
        "running": set(),
        "completed": set(),
        "failed": set(),
    }
    for v in store.list_versions():
        invalid = version_invalid_members(v["version_id"], store)
        v_bind = [b for b in bindings if b["version_id"] == v["version_id"]]
        v_runs = [r for r in runs if r["version_id"] == v["version_id"]]
        used_rows = max((int(b["stats"].get("used_rows", 0)) for b in v_bind), default=0)
        # Sayılan aday = o kanonik dosyaya GERÇEKTEN giren (pay sınırı sonrası) train üyeleri.
        run_shas = {r["lora_sft_sha256"] for r in v_runs}
        for b in v_bind:
            ids = set(b["stats"].get("used_candidate_ids", []))
            if b["lora_sft_sha256"] in run_shas:
                for r in v_runs:
                    if r["lora_sft_sha256"] == b["lora_sft_sha256"]:
                        use.setdefault(r["state"], set()).update(ids)
            else:
                use["planned"].update(ids)
        versions.append(
            {
                **v,
                "invalid_members": len(invalid),
                "invalid_reasons": sorted({b["why"] for b in invalid}),
                "bindings": v_bind,
                "runs": v_runs,
                "selected": bool(sel and sel.get("version_id") == v["version_id"]),
                "bound_rows_max": used_rows,
            }
        )
    return {
        "versions": versions,
        "selection": sel,
        "training_use": {k: len(ids) for k, ids in use.items()},
        "training_use_note": "Sayılar o koşuya bağlanan sürümün train üyeleridir; hariç tutma "
        "önceden eğitilmiş bir modelden bilgiyi silmez.",
    }
