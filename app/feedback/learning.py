"""learning.py — sohbetten öğrenme: geri bildirim, adaylar, aileler, sızıntı ve sayaçlar.

Kullanıcı eylemleri → etki:
- Faydalı  : yalnız tura geri bildirim. Aday ÜRETMEZ.
- Hatalı   : hata kuyruğu (tur). Aday ÜRETMEZ, puanlanabilir test de sayılmaz.
- Öğrensin : model cevabı hedef olan aday (tek tıklama; "Faydalı" şartı yok).
- Düzelt   : kullanıcının yazdığı metin hedef olan aday (aynı turda tek aktif düzeltme;
             yeniden gönderim düzeltmeyi günceller, yeni aday açmaz).
- Hariç tut: tur ve adayları eğitimden çıkar. Eski veri sürümlerini DEĞİŞTİRMEZ; yeni bir
             eğitim başlatılırken seçili sürümdeki geçersiz kayıtlar tespit edilip durdurulur.

Durum önceliği: hariç > sızıntı > aile köprüsü çatışması > doğrulama kararı > aile denetimi
(yinelenen / farklı sayısal sonuç). Hiçbir adım eğitim başlatmaz (Kural 8).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from app.config import get_settings
from app.feedback.chat_store import ChatStore, utcnow
from app.feedback.verify import DOMAINS, decide, detect_domain, sha256_text, verify_target

LABELS = ("useful", "wrong", "")


class LearningError(ValueError):
    """Kullanıcıya gösterilecek (422) iş kuralı hatası."""


# Sızıntı denetiminin eval kalemleri — tek kaynak `mix_cli.leakage_eval_items` (testte
# değiştirilebilir). Eval kalemleri yalnız DENETİM için okunur, hiçbir yere yazılmaz.
def _default_leak_items() -> list[Any]:
    from app.lora.mix_cli import leakage_eval_items

    return leakage_eval_items()


leak_items_provider: Callable[[], list[Any]] = _default_leak_items


# ── aile benzerliği ──────────────────────────────────────────────────────────


def _norm(text: str) -> str:
    from app.evals.profile.leakage import normalize_text

    return normalize_text(text or "")


def _grams(text: str) -> set[tuple[str, ...]]:
    words = _norm(text).split()
    n = 3 if len(words) >= 4 else 1
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def question_similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if na and na == nb:
        return 1.0
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _hash_split(family_id: str, seed: int, eval_ratio: float) -> str:
    h = int(hashlib.sha256(f"{seed}:{family_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if h < eval_ratio else "train"


class LearningService:
    """Aday yaşam döngüsü (eğitim başlatmaz)."""

    def __init__(self, store: ChatStore | None = None) -> None:
        self.store = store or ChatStore()
        self.settings = get_settings()

    # ── tur geri bildirimi ───────────────────────────────────────────────────

    def set_feedback(
        self, turn_id: str, label: str, note: str = "", spans: list[dict] | None = None
    ) -> dict[str, Any]:
        if label not in LABELS:
            raise LearningError(f"Geçersiz etiket: {label}")
        turn = self._turn(turn_id)
        fields: dict[str, Any] = {"feedback": label, "feedback_note": note[:4000]}
        if spans is not None:
            fields["flagged_spans"] = _clean_spans(spans, len(turn["answer"]))
        updated = self.store.update_turn(turn_id, **fields)
        assert updated is not None
        return updated

    def set_excluded(self, turn_id: str, excluded: bool, reason: str = "") -> dict[str, Any]:
        self._turn(turn_id)
        self.store.update_turn(turn_id, excluded=excluded, exclude_reason=reason[:2000])
        for c in self.store.candidates_for_turn(turn_id):
            self.recompute(c["candidate_id"])
        turn = self.store.get_turn(turn_id)
        assert turn is not None
        return turn

    # ── aday oluşturma ───────────────────────────────────────────────────────

    def learn(self, turn_id: str, domain: str | None = None) -> tuple[dict[str, Any], bool]:
        """'Öğrensin': model cevabı hedef. Aynı tur için ikinci tıklama aynı adayı döndürür."""
        turn = self._trainable_turn(turn_id)
        target = turn["raw_answer"] or turn["answer"]
        return self._create(turn, "learn", target, domain, spans=[])

    def correct(
        self,
        turn_id: str,
        text: str,
        *,
        domain: str | None = None,
        spans: list[dict] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """'Düzelt': kullanıcının metni hedef. Turda düzeltme varsa GÜNCELLENİR."""
        text = (text or "").strip()
        if not text:
            raise LearningError("Düzeltme metni boş.")
        turn = self._trainable_turn(turn_id)
        existing = self.store.get_candidate_for_turn(turn_id, "correct")
        clean_spans = _clean_spans(spans or [], len(turn["answer"]))
        if existing is not None:
            if sha256_text(text) == existing["target_sha"] and (
                domain in (None, existing["domain"])
            ):
                return existing, False  # aynı içerik tekrar gönderildi — değişiklik yok
            return self.edit(
                existing["candidate_id"], text, domain=domain, spans=clean_spans
            ), False
        return self._create(turn, "correct", text, domain, spans=clean_spans)

    def _create(
        self, turn: dict[str, Any], kind: str, target: str, domain: str | None, spans: list[dict]
    ) -> tuple[dict[str, Any], bool]:
        if domain is not None and domain not in DOMAINS:
            raise LearningError(f"Geçersiz alan: {domain}")
        dom = domain or detect_domain(turn["question"], target, bool(turn["sources"]))
        cand, created = self.store.insert_candidate(
            turn_id=turn["turn_id"],
            kind=kind,
            target_text=target,
            target_sha=sha256_text(target),
            flagged_spans=spans,
            domain=dom,
            domain_source="user" if domain else "auto",
            as_of=turn["created_at"],
        )
        if created:
            cand = self.recompute(cand["candidate_id"])
        return cand, created

    # ── düzenleme / onay ─────────────────────────────────────────────────────

    def edit(
        self,
        candidate_id: str,
        text: str,
        *,
        domain: str | None = None,
        spans: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Hedefi düzelt / eksik kısmı çıkar → yeniden doğrula; insan onayı DÜŞER."""
        cand = self._cand(candidate_id)
        text = (text or "").strip()
        if not text:
            raise LearningError("Hedef metin boş olamaz.")
        if domain is not None and domain not in DOMAINS:
            raise LearningError(f"Geçersiz alan: {domain}")
        fields: dict[str, Any] = {
            "target_text": text,
            "target_sha": sha256_text(text),
            "revision": int(cand["revision"]) + 1,
            "human_approval": {},
        }
        if domain:
            fields.update(domain=domain, domain_source="user")
        if spans is not None:
            fields["flagged_spans"] = spans
        self.store.update_candidate(candidate_id, **fields)
        return self.recompute(candidate_id)

    def approve(self, candidate_id: str, reason: str) -> dict[str, Any]:
        """Gerekçeli insan onayı. Çürütülmüş / backtest'siz performans iddiası onaylanamaz."""
        from app.feedback.verify import human_approval_blocker

        reason = (reason or "").strip()
        if len(reason) < 10:
            raise LearningError("Onay gerekçesi en az 10 karakter olmalı.")
        cand = self._cand(candidate_id)
        if cand["status"] in ("excluded", "leak"):
            raise LearningError(f"Bu aday onaylanamaz (durum: {cand['status']}).")
        blocker = human_approval_blocker(cand["verification"])
        if blocker:
            raise LearningError(blocker)
        self.store.update_candidate(
            candidate_id,
            human_approval={"reason": reason, "at": utcnow(), "target_sha": cand["target_sha"]},
        )
        return self.recompute(candidate_id)

    # ── yeniden hesaplama ────────────────────────────────────────────────────

    def recompute(self, candidate_id: str) -> dict[str, Any]:
        cand = self._cand(candidate_id)
        turn = self._turn(cand["turn_id"])
        verification = verify_target(
            cand["target_text"],
            question=turn["question"],
            sources=turn["sources"],
            domain=cand["domain"],
        )
        status, reason, codes, vclass = decide(verification, cand["human_approval"])
        verification["class"] = vclass
        fields: dict[str, Any] = {"verification": verification}

        family_id = cand["family_id"]
        if turn["excluded"]:
            status, reason, codes = (
                "excluded",
                turn["exclude_reason"] or "Eğitimden hariç.",
                ["haric"],
            )
        else:
            leak = self.leak_hits(turn, cand["target_text"])
            if leak:
                status, reason, codes = "leak", "Korunan eval setiyle çakışma: " + leak, ["sizinti"]
            else:
                family_id, bridge = self._assign_family(cand, turn)
                if bridge:
                    status, reason, codes = "conflict", bridge, ["aile_kopru"]
        fields.update(status=status, status_reason=reason, reason_codes=codes, family_id=family_id)
        self.store.update_candidate(candidate_id, **fields)
        if family_id:
            self.audit_family(family_id)
        return self._cand(candidate_id)

    def recheck_all(self) -> None:
        """Tüm aktif adayları güncel kurallarla yeniden değerlendir (sürüm öncesi)."""
        for c in self.store.list_candidates():
            self.recompute(c["candidate_id"])

    # ── sızıntı ──────────────────────────────────────────────────────────────

    def leak_hits(self, turn: dict[str, Any], target: str) -> str:
        """Eğitim satırının (gerçek istem + hedef) korunan eval kalemleriyle çakışması."""
        from app.evals.profile.leakage import check_leakage

        prompt = turn["user_prompt"] or turn["question"]
        example = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]
        }
        try:
            items = leak_items_provider()
        except Exception as exc:  # denetlenemiyorsa güvenli taraf: aday eğitime gitmesin
            return f"sızıntı denetimi çalıştırılamadı ({exc})"
        report = check_leakage(items, [example])
        if report.clean:
            return ""
        kinds = sorted({h.kind for h in report.hits})
        ids = sorted({h.eval_id for h in report.hits})[:3]
        return f"{', '.join(kinds)} ({', '.join(ids)})"

    # ── aileler ──────────────────────────────────────────────────────────────

    def _assign_family(self, cand: dict[str, Any], turn: dict[str, Any]) -> tuple[str, str]:
        """(aile_kimliği, köprü-çatışması gerekçesi). Kalıcı split ataması burada yapılır."""
        s = self.settings
        q = turn["question"]
        matched: dict[str, dict[str, Any]] = {}
        for other in self.store.list_candidates():
            if other["candidate_id"] == cand["candidate_id"] or not other["family_id"]:
                continue
            if other["status"] == "excluded":
                continue
            o_turn = self.store.get_turn(other["turn_id"])
            if o_turn is None:
                continue
            if question_similarity(q, o_turn["question"]) >= s.learning_family_jaccard:
                fid = self.store.resolve_family(other["family_id"])
                fam = self.store.get_family(fid)
                if fam is not None:
                    matched[fid] = fam
        if not matched:
            fid = "fam_" + hashlib.sha256(_norm(q).encode()).hexdigest()[:12]
            fam = self.store.get_family(fid)
            if fam is None:
                split = (
                    "time"
                    if cand["domain"] == "trading"
                    else _hash_split(fid, s.learning_seed, s.learning_eval_ratio)
                )
                self.store.upsert_family(fid, split=split, domain=cand["domain"], anchor_question=q)
            return self.store.resolve_family(fid), ""
        splits = {f["split"] for f in matched.values()}
        if len(splits) > 1:
            names = ", ".join(f"{fid}={f['split']}" for fid, f in sorted(matched.items()))
            return "", (
                "Sızıntı çatışması: bu örnek farklı bölmelerdeki aileleri birbirine bağlıyor "
                f"({names}). Otomatik eğitime alınmadı; soruyu ayırın ya da birini hariç tutun."
            )
        ordered = sorted(matched.values(), key=lambda f: (f["created_at"], f["family_id"]))
        root = ordered[0]["family_id"]
        for f in ordered[1:]:
            self.store.upsert_family(f["family_id"], merged_into=root)
            for member in self.store.list_candidates():
                if member["family_id"] == f["family_id"]:
                    self.store.update_candidate(member["candidate_id"], family_id=root)
        return root, ""

    def audit_family(self, family_id: str) -> None:
        """Aile içi denetim: yinelenen hedef → 'duplicate'; farklı sayısal sonuç → 'conflict'.

        Anlamsal çelişki otomatik tespit EDİLMEZ; yalnız yeniden hesaplanmış sayısal sonuçlar
        karşılaştırılır.
        """
        members = [
            c
            for c in self.store.list_candidates()
            if c["family_id"] == family_id
            and c["status"] in ("eligible", "duplicate", "conflict")
            and "aile_kopru" not in c["reason_codes"]
        ]
        # Önce temel kararı geri yükle (önceki denetimin etiketlerini temizle).
        base: list[dict[str, Any]] = []
        for c in members:
            if c["status"] in ("duplicate", "conflict"):
                st, reason, codes, _vclass = decide(c["verification"], c["human_approval"])
                c = (
                    self.store.update_candidate(
                        c["candidate_id"], status=st, status_reason=reason, reason_codes=codes
                    )
                    or c
                )
            if c["status"] == "eligible":
                base.append(c)
        seen: dict[str, str] = {}
        results: dict[str, list[str]] = {}
        for c in base:
            turn = self.store.get_turn(c["turn_id"]) or {}
            key = _norm(turn.get("question", "")) + "\x00" + _norm(c["target_text"])
            if key in seen:
                self.store.update_candidate(
                    c["candidate_id"],
                    status="duplicate",
                    status_reason=f"Aynı soru+hedef zaten var ({seen[key]}); tekrar eklenmez.",
                    reason_codes=["yinelenen"],
                )
                continue
            seen[key] = c["candidate_id"]
            results[c["candidate_id"]] = _math_results(c["verification"])
        with_nums = {cid: r for cid, r in results.items() if r}
        values = {tuple(r) for r in with_nums.values()}
        if len(values) > 1:
            for cid in with_nums:
                self.store.update_candidate(
                    cid,
                    status="conflict",
                    status_reason="Aile içinde farklı sayısal sonuçlar var; birini düzeltin "
                    "ya da hariç tutun.",
                    reason_codes=["aile_ici_celiski"],
                )

    # ── sayaçlar / listeler ──────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Panel sayaçları — adayların GÜNCEL durumundan hesaplanır (yazma yok)."""
        from app.feedback.chat_dataset import version_overview

        cands = self.store.list_candidates()
        by_status: dict[str, int] = {}
        rejected_by: dict[str, int] = {}
        auto = human = 0
        for c in cands:
            by_status[c["status"]] = by_status.get(c["status"], 0) + 1
            if c["status"] == "rejected":
                code = (c["reason_codes"] or ["diger"])[0]
                rejected_by[code] = rejected_by.get(code, 0) + 1
            if c["status"] == "eligible":
                if (c["verification"] or {}).get("class") == "human":
                    human += 1
                else:
                    auto += 1
        eligible = [c for c in cands if c["status"] == "eligible"]
        fam_split: dict[str, str] = {}
        for c in eligible:
            fam = self.store.get_family(c["family_id"]) if c["family_id"] else None
            fam_split[c["family_id"]] = fam["split"] if fam else "?"
        train_like = sum(1 for sp in fam_split.values() if sp in ("train", "time"))
        errors = self.error_queue()
        versions = version_overview(self.store)
        return {
            "collected": len(cands),
            "by_status": by_status,
            "eligible": {"total": len(eligible), "auto": auto, "human": human},
            "rejected_by": rejected_by,
            "errors_open": sum(1 for e in errors if not e["has_correction"]),
            "families": {
                "eligible_total": len(fam_split),
                "eligible_train_or_time": train_like,
                "threshold": self.settings.learning_min_families,
                "threshold_note": "Eğitim hazırlığı için BAŞLANGIÇ ayarıdır; kalite kuralı "
                "değildir. Daha az örnekle de veri sürümü oluşturulabilir.",
            },
            "max_share": self.settings.learning_chat_max_share,
            **versions,
        }

    def error_queue(self) -> list[dict[str, Any]]:
        out = []
        for t in self.store.list_turns_by_feedback("wrong"):
            corr = self.store.get_candidate_for_turn(t["turn_id"], "correct")
            out.append(
                {
                    "turn_id": t["turn_id"],
                    "conversation_id": t["conversation_id"],
                    "turn_index": t["turn_index"],
                    "question": t["question"],
                    "answer": t["answer"],
                    "note": t["feedback_note"],
                    "flagged_spans": t["flagged_spans"],
                    "model_tag": t["model_tag"],
                    "excluded": t["excluded"],
                    "has_correction": corr is not None,
                    "correction_candidate_id": corr["candidate_id"] if corr else "",
                    "created_at": t["created_at"],
                }
            )
        return out

    def list_candidates(self, status: str | None = None, limit: int = 200) -> list[dict]:
        out = []
        for c in reversed(self.store.list_candidates(status=status)):
            turn = self.store.get_turn(c["turn_id"]) or {}
            fam = self.store.get_family(c["family_id"]) if c["family_id"] else None
            out.append(
                {
                    **c,
                    "question": turn.get("question", ""),
                    "model_answer": turn.get("answer", ""),
                    "model_tag": turn.get("model_tag", ""),
                    "conversation_id": turn.get("conversation_id", ""),
                    "turn_index": turn.get("turn_index"),
                    "split": fam["split"] if fam else "",
                }
            )
            if len(out) >= limit:
                break
        return out

    # ── yardımcılar ──────────────────────────────────────────────────────────

    def _turn(self, turn_id: str) -> dict[str, Any]:
        turn = self.store.get_turn(turn_id)
        if turn is None:
            raise KeyError(f"Tur bulunamadı: {turn_id}")
        return turn

    def _cand(self, candidate_id: str) -> dict[str, Any]:
        cand = self.store.get_candidate(candidate_id)
        if cand is None:
            raise KeyError(f"Aday bulunamadı: {candidate_id}")
        return cand

    def _trainable_turn(self, turn_id: str) -> dict[str, Any]:
        turn = self._turn(turn_id)
        if turn["status"] != "answered" or not turn["user_prompt"]:
            raise LearningError(
                "Bu turda modele istem gönderilmedi (kaynak yok / çekimser / engellendi); "
                "eğitim örneğine çevrilemez."
            )
        if turn["excluded"]:
            raise LearningError("Tur eğitimden hariç tutulmuş; önce hariç tutmayı kaldırın.")
        return turn


def _clean_spans(spans: list[dict], limit: int) -> list[dict]:
    out = []
    for sp in spans[:50]:
        try:
            a, b = int(sp.get("start", -1)), int(sp.get("end", -1))
        except (TypeError, ValueError, AttributeError):
            continue
        if 0 <= a < b <= limit:
            out.append({"start": a, "end": b, "note": str(sp.get("note", ""))[:500]})
    return out


def _math_results(verification: dict[str, Any]) -> list[str]:
    for c in verification.get("checks", []):
        if c.get("kind") == "hesap":
            return sorted(
                re.sub(r"\s", "", str(it.get("rhs", "")))
                for it in c.get("items", [])
                if it.get("status") == "gecti"
            )
    return []
