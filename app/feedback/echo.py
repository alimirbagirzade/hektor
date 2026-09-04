"""echo.py — Echo: kullanıcı düzeltmelerini sentetik SFT adayına çevirir (feedback döngüsü).

Akış: kullanıcı "bu cevap yanlıştı, doğrusu X" der → Echo kaydeder + Kural-1 güvenlik
kontrolü (düzeltme metni garanti/kesinlik dili içeriyorsa ANINDA reddedilir, zehir girmesin)
→ insan onayı → onaylananlar lora_sft FORMATINDA AYRI bir aday dosyaya export edilir.

GÜVENLİK (Kural 8): export edilen aday `data/feedback/feedback_sft.jsonl`'e yazılır;
kanonik `lora_sft.jsonl`'e OTO-MERGE YOK. Eğitim ASLA tetiklenmez; aday yine pretrain-gate
+ dataset audit'ten geçmelidir. Çıktı determinist (Kural 6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.feedback.store import FeedbackStore

log = logging.getLogger(__name__)

# Geçerli düzeltme türleri (serbest metin yerine sınırlı küme → tutarlı metadata).
VALID_CORRECTION_TYPES: frozenset[str] = frozenset(
    {
        "claim_correction",  # yanlış iddia düzeltildi
        "missing_caveat",  # eksik uyarı/çekince eklendi
        "wrong_number",  # hatalı sayı/metrik düzeltildi
        "advice_language_removed",  # tavsiye/kesinlik dili giderildi
        "other",
    }
)

# Durum sabitleri.
PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
EXPORTED = "exported"


# Feedback'e ÖZEL ek yönlendirme/kesinlik kalıpları (alt-dize, tr_fold ile aksan/büyük-harf
# bilinçli). Paylaşılan safety_scanner.FINANCIAL_DIRECTIVES'in feedback için genişletilmesidir;
# küratörlü kartların Gate-7 eşiğini DEĞİŞTİRMEZ → kart yanlış-pozitifi yaratmaz. Feedback HAM
# kullanıcı girdisi olduğundan asimetri bilinçli: yanlış-pozitif UCUZ (kullanıcı yeniden yazar),
# yanlış-negatif PAHALI (eğitim verisine zehir → v5-sınıfı disiplin regresyonu). Bu yüzden
# olumsuzlama içinde geçse bile (ör. 'garantili değildir') reddedilir — feedback için güvenli
# taraf. 'risksiz/risk-free' EKLENMEDİ (akademik 'risk-free rate' FP'si); imperatif
# 'risk yok/no risk' zaten scan_for_secrets'te. Kademe-2 av (ECHO-POISON-01) HIGH bulgusu.
_FEEDBACK_EXTRA_DIRECTIVES: tuple[str, ...] = (
    "garantili",
    "kâr garanti",
    "kar garanti",
    "asla kaybetme",
    "kaybetmezsin",
    "kaybetmezsiniz",
    "never lose",
    "cannot lose",
    "can't lose",
    "cant lose",
    "her zaman kazan",
    "kesin getiri",
    "getiri kesin",
    "canlı sinyal",
    "sinyal canlı",
    "live signal",
    "%100 kar",
    "%100 kâr",
    "100% profit",
)


def correction_safety_reason(*texts: str) -> str | None:
    """Verilen metinlerden BİRİ Kural-1 zehiri (advice/garanti/kesinlik/canlı-sinyal + sır/PII)
    içeriyorsa gerekçe.

    KRİTİK: SFT satırına giren TÜM alanlar (soru=user-turn + düzeltme=assistant-turn) ve saklanan
    bad_answer denetlenmeli. GENİŞ tarama (Kademe-2 av ECHO-POISON-01): önceden YALNIZ
    RED_FLAGS['guaranteed_profit'] taranıyordu → advice ('şimdi al'/'buy now'), risk-free
    ('risk yok'/'no risk'), canlı-sinyal ve 'garantili'/'asla kaybetmezsin' gibi kesinlik dili
    SFT adayına sızabiliyordu. Artık paylaşılan safety_scanner.scan_for_secrets (FINANCIAL_
    DIRECTIVES + sır/PII) + guaranteed_profit + feedback'e özel _FEEDBACK_EXTRA_DIRECTIVES
    birlikte uygulanır. Diğer (advice-dışı) kalite export sonrası pretrain-gate/audit'te."""
    from app.lora.safety_scanner import scan_for_secrets, tr_fold
    from app.training.evaluate_model import RED_FLAGS

    gp = RED_FLAGS["guaranteed_profit"]
    for raw in texts:
        t = raw or ""
        if not t.strip():
            continue
        if gp.search(t):
            return "Girdi garanti/kesinlik dili içeriyor (Kural 1 zehiri) — reddedildi."
        sr = scan_for_secrets(t)
        if not sr.passed:
            return f"Girdi yasak/yönlendirme deseni içeriyor ({sr.violations[0]}) — Kural 1 reddi."
        folded = tr_fold(t)
        for directive in _FEEDBACK_EXTRA_DIRECTIVES:
            if tr_fold(directive) in folded:
                return f"Girdi yönlendirme/kesinlik dili içeriyor ('{directive}') — Kural 1 reddi."
    return None


def _ensure_within(path: Path, roots: tuple[Path, ...]) -> Path:
    """Export yolunu izinli köklerin İÇİNE kısıtla (yol-geçişi savunması).

    Adversarial review bulgusu: out_path ileride güvensiz bir girdiye bağlanırsa
    '../../etc/passwd' gibi keyfi konuma yazmayı engeller. İzinli kökler proje kökü +
    deponun kendi dizinidir (üretimde sqlite kök altında; testte tmp_path) → kırılgan
    OS-temp varsayımı yok. Web ucu out_path'i HİÇ ifşa etmez; bu ek savunma katmanıdır."""
    resolved = path.resolve()
    if not any(resolved.is_relative_to(r.resolve()) for r in roots):
        raise ValueError(f"Export yolu izinli kök dışında reddedildi: {resolved}")
    return resolved


class EchoCollector:
    """Feedback toplama + onay + SFT-aday export'u (eğitim başlatmaz)."""

    def __init__(self, store: FeedbackStore | None = None) -> None:
        self.store = store or FeedbackStore()

    # ── kayıt ────────────────────────────────────────────────────────────────

    def record(
        self,
        *,
        source: str,
        question: str,
        bad_answer: str,
        correction: str,
        correction_type: str = "other",
    ) -> dict[str, Any]:
        """Bir düzeltmeyi kaydet. Boş/zehirli düzeltme ANINDA reddedilir (dürüstlük)."""
        ct = correction_type if correction_type in VALID_CORRECTION_TYPES else "other"

        if not (correction or "").strip():
            cid = self.store.add(
                source=source,
                question=question,
                bad_answer=bad_answer,
                correction=correction,
                correction_type=ct,
                status=REJECTED,
                reject_reason="Boş düzeltme.",
            )
            return self.store.get(cid)  # type: ignore[return-value]

        reason = correction_safety_reason(correction, question, bad_answer)
        status = REJECTED if reason else PENDING
        cid = self.store.add(
            source=source,
            question=question,
            bad_answer=bad_answer,
            correction=correction,
            correction_type=ct,
            status=status,
            reject_reason=reason or "",
        )
        return self.store.get(cid)  # type: ignore[return-value]

    # ── onay/ret ──────────────────────────────────────────────────────────────

    def approve(self, correction_id: str) -> bool:
        """Bir düzeltmeyi onayla. Onay anında güvenlik tekrar kontrol edilir (Kural 1)."""
        c = self.store.get(correction_id)
        if c is None:
            return False
        reason = correction_safety_reason(c["correction"], c["question"], c["bad_answer"])
        if reason:
            self.store.set_status(correction_id, REJECTED, reason)
            return False
        if not (c["question"] or "").strip() or not (c["correction"] or "").strip():
            self.store.set_status(correction_id, REJECTED, "Boş soru veya düzeltme.")
            return False
        return self.store.set_status(correction_id, APPROVED)

    def reject(self, correction_id: str, reason: str = "") -> bool:
        return self.store.set_status(correction_id, REJECTED, reason or "Elle reddedildi.")

    # ── SFT export ────────────────────────────────────────────────────────────

    def to_sft_line(self, correction: dict[str, Any]) -> str:
        """Bir düzeltmeyi lora_sft uyumlu JSONL satırına çevir (system→user→assistant)."""
        from app.lora.dataset_builder import SYSTEM_PROMPT, LoRAExample

        ex = LoRAExample(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": correction["question"]},
                {"role": "assistant", "content": correction["correction"]},
            ],
            metadata={
                "source": "feedback",
                "correction_type": correction["correction_type"],
                "feedback_id": correction["correction_id"],
            },
        )
        return ex.to_jsonl_line()

    def export_approved(self, out_path: str | Path | None = None) -> dict[str, Any]:
        """Onaylanan (ve daha önce export edilen) düzeltmeleri AYRI aday dosyaya yaz.

        Kanonik `lora_sft.jsonl`'e DOKUNMAZ (Kural 8). İdempotent: approved ∪ exported
        kümesini yazar, approved → exported işaretler → tekrar çağrı aynı dosyayı üretir
        (export sonrası 'approved' boşalıp dosyanın silinmesi bug'ına karşı)."""
        approved = self.store.list(status=APPROVED, limit=1_000_000)
        already = self.store.list(status=EXPORTED, limit=1_000_000)
        # id'ye göre birleştir + dedup. Determinizm (Kural 6): created_at EŞİTLİĞİNDE (kaba
        # Windows saati aynı tick'i üretebilir) correction_id İKİNCİL anahtar → satır sırası
        # snapshot bölünmesinden + SQLite tie davranışından bağımsız, içerikten türeyen total sıra.
        by_id: dict[str, dict[str, Any]] = {c["correction_id"]: c for c in already + approved}
        usable: list[dict[str, Any]] = []
        newly_approved_ids: list[str] = []
        skipped_poison = 0
        for c in sorted(by_id.values(), key=lambda c: (c["created_at"], c["correction_id"])):
            # İçerik seçiminde durumu TAZE oku: eşzamanlı /reject, snapshot ile yazım arasına
            # girip kaydı REJECTED yaparsa onu dosyaya YAZMA (reddedilen düzeltme adaya sızmasın —
            # eski 'race savunması' yalnız işaretlemeyi koruyordu, İÇERİĞİ değil).
            cur = self.store.get(c["correction_id"])
            if cur is None or cur["status"] not in (APPROVED, EXPORTED):
                continue
            if not (cur["question"] or "").strip() or not (cur["correction"] or "").strip():
                continue
            # Geç-yakalanan zehir: filtre güçlenmeden ÖNCE onaylanmış bayat kayıt adaya sızmasın —
            # yazmadan ÖNCE tekrar tara (ECHO-POISON-02: feedback satırları aksi halde geniş
            # güvenlik tarayıcısına HİÇ uğramıyor).
            if correction_safety_reason(cur["correction"], cur["question"], cur["bad_answer"]):
                skipped_poison += 1
                continue
            usable.append(cur)
            if cur["status"] == APPROVED:
                newly_approved_ids.append(cur["correction_id"])

        path = (
            Path(out_path)
            if out_path
            else get_settings().root / "data" / "feedback" / "feedback_sft.jsonl"
        )
        # yol-geçişi savunması: proje kökü + deponun kendi dizini izinli (kırılgan değil)
        path = _ensure_within(path, (get_settings().root, self.store.db_path.parent))
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [self.to_sft_line(c) for c in usable]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # Yalnız dosyaya YAZILAN (usable) + hâlâ APPROVED kayıtları EXPORTED işaretle; işaretlemeden
        # hemen ÖNCE durumu TEKRAR oku (yazım↔işaretleme penceresinde reddedilen kaydı geçirme).
        marked = 0
        for cid in newly_approved_ids:
            cur = self.store.get(cid)
            if cur and cur["status"] == APPROVED:
                self.store.set_status(cid, EXPORTED)
                marked += 1

        if skipped_poison:
            log.warning(
                "export: %d onaylı kayıt geç-yakalanan Kural-1 zehiri nedeniyle ATLANDI",
                skipped_poison,
            )
        return {
            "n_exported": len(usable),
            "newly_marked": marked,
            "skipped_poison": skipped_poison,
            "path": str(path),
        }

    # ── özet ──────────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        counts = self.store.counts()
        return {
            "counts": counts,
            "pending": counts.get(PENDING, 0),
            "approved": counts.get(APPROVED, 0),
            "rejected": counts.get(REJECTED, 0),
            "exported": counts.get(EXPORTED, 0),
        }
