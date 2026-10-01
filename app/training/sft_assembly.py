"""Birleşik SFT seti kurma — sentetik QA + onaylı kart + adversarial disiplin karışımı.

`lora-cloud-prep` (eğitim paketi) ve `pretrain-gate` (offline kalite kapısı) ORTAK yolu;
ikisi aynı birleştirme mantığını kullansın diye buraya çıkarıldı (drift önlenir).

Sıra: sentetik QA dosyası + onaylı kart örnekleri → hash + near-duplicate dedup (A7) →
disiplin örneklerini DEDUP'TAN SONRA ~%25 karıştır (#4 Fix B; şablon örnekleri near-dup
filtresine toplu takılmasın). EĞİTİM BAŞLATMAZ (kural 8).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.brain.synthetic_qa_builder import dedup_jsonl_lines
from app.lora.dataset_builder import build_dataset
from app.training.discipline_dataset import discipline_jsonl_lines, mix_discipline

_PII_MASK = "[kişisel-veri]"
# json.dumps(ensure_ascii=False) bunları ham bırakır ama str.splitlines() onları satır sonu
# sayar (LoRAExample.to_jsonl_line ile aynı koruma).
_LINE_SEPARATOR_ESCAPES = (
    (chr(0x2028), "\\u2028"),
    (chr(0x2029), "\\u2029"),
    (chr(0x85), "\\u0085"),
)


def _mask_strings(value: Any, patterns: list[re.Pattern[str]]) -> Any:
    """JSON değerindeki TÜM string'lerde PII desenlerini maskele (anahtarlara dokunmaz)."""
    if isinstance(value, str):
        for pat in patterns:
            value = pat.sub(_PII_MASK, value)
        return value
    if isinstance(value, list):
        return [_mask_strings(v, patterns) for v in value]
    if isinstance(value, dict):
        return {k: _mask_strings(v, patterns) for k, v in value.items()}
    return value


def redact_pii_line(line: str) -> str:
    """JSONL satırındaki e-posta / uluslararası telefon desenlerini maskele.

    Makale başlık bloklarındaki yazar e-postaları ("Corresponding author: …@…") sentetik QA
    bağlamına sızıyordu (2026-09-14 ölçümü: 1701 satırın 92'sinde 191 adres). Eğitim verisine
    kişisel veri girmez → birleştirmede maskelenir. Desenler pretrain-gate'in taradığı
    `_PII_PATTERNS` ile AYNI kaynaktan gelir (kapı ile maskeleme sapmasın); kapı yine son
    savunma olarak taramaya devam eder.

    Maskeleme ÇÖZÜMLENMİŞ string değerlerinde yapılır, ham JSON metninde DEĞİL (2026-09-15
    hatası): ham metinde satır sonu `\\n` olarak yazılır ve e-posta deseni kaçışın `n`
    harfini adresin parçası sanıyordu (`\\ndasashreeya@…` → `\\[kişisel-veri]`); geriye kalan
    `\\[` geçersiz JSON kaçışıdır → 1117 sentetik satırın 47'si bozuldu, pretrain-gate
    "okunamayan satır" ile NO-GO verdi. Eşleşme yoksa satır BAYT-ÖZDEŞ döner (dedup ve diff
    kararlı kalsın); JSON olmayan satır olduğu gibi bırakılır (kapı onu okunamayan diye bloklar).
    """
    from app.registry.promotion_gates import _PII_PATTERNS

    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return line
    masked = _mask_strings(obj, list(_PII_PATTERNS.values()))
    if masked == obj:
        return line
    out = json.dumps(masked, ensure_ascii=False)
    for ch, esc in _LINE_SEPARATOR_ESCAPES:
        out = out.replace(ch, esc)
    return out


@dataclass
class AssemblyResult:
    """Birleşik SFT satırları + nereden geldiği şeffaflığı."""

    lines: list[str]
    synth_n: int
    card_n: int
    deduped: int  # synth + kart, dedup sonrası
    discipline: dict[str, Any] | None = field(default=None)
    low_value_dropped: int = 0  # atılan çekimser / "pasaj" atıflı sentetik örnek
    distill_n: int = 0  # eklenen öz-damıtma satırı (yeniden doğrulama sonrası, dedup öncesi)

    @property
    def total(self) -> int:
        return len(self.lines)


# v14: kısa sentetik QA payı ~%60 → ~%30. Varsayılan olarak TÜM yazıcılar (assemble_sft.py,
# lora-cloud-prep, detached_launch) ve tazelik denetimi aynı sınırı kullanır (drift yok).
CANONICAL_SYNTH_CAP = 400


def _cap_synth(lines: list[str], cap: int, seed: int) -> list[str]:
    """Sentetik QA'yı ``cap`` satıra indir: önce zenginleştirilmiş (açıklamalı) satırlar,
    kalan kota seed'li rastgele (Kural 6). Seçilenler ORİJİNAL sırasını korur.

    v14 (2026-10-01): kısa sentetik cevaplar setin ~%60'ıydı ve LoRA'yı kısa/atıfsız cevaba
    çekiyordu (LLM-30 2×2) → pay düşürülür, yerine öz-damıtma örnekleri girer.
    """
    import random

    if cap <= 0 or len(lines) <= cap:
        return lines

    def _enriched(ln: str) -> bool:
        try:
            return bool((json.loads(ln).get("metadata") or {}).get("enriched"))
        except (json.JSONDecodeError, AttributeError, TypeError):
            return False

    idx = list(range(len(lines)))
    random.Random(seed).shuffle(idx)
    idx.sort(key=lambda i: not _enriched(lines[i]))  # kararlı sıralama: zenginler öne
    keep = sorted(idx[:cap])
    return [lines[i] for i in keep]


def assemble_sft_lines(
    settings: Any,
    *,
    discipline: bool = True,
    discipline_ratio: float = 0.25,
    seed: int = 0,
    synth_cap: int = CANONICAL_SYNTH_CAP,
    distill: bool = True,
) -> AssemblyResult:
    """Birleşik SFT JSONL satırlarını kur (eğitim BAŞLATMAZ).

    Args:
        settings: `get_settings()` çıktısı (`.root` kullanılır).
        discipline: Adversarial disiplin örneklerini karıştır (#4 Fix B).
        discipline_ratio: Disiplin payı (disiplin/(taban+disiplin)); v5 dersi ~0.25.
        seed: Determinizm tabanı (karıştırma — kural 6).
        synth_cap: >0 → sentetik QA en çok bu kadar satır (`_cap_synth`); 0 = sınırsız.
            Varsayılan kanonik sınır (`CANONICAL_SYNTH_CAP`).
        distill: `distill_qa.jsonl` (base öz-damıtma, `hektor synth-distill`) varsa ekle.
    """
    lora_dir = settings.root / "data" / "lora_sft"
    synth_path = lora_dir / "synthetic_qa.jsonl"

    lines: list[str] = []
    synth_n = 0
    low_value_dropped = 0
    if synth_path.exists():
        synth_lines = [
            ln for ln in synth_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        synth_n = len(synth_lines)
        # Üretici filtresinden ÖNCE yazılmış çekimser / "pasaj" atıflı örnekler de eğitime
        # girmesin (eski veride %3-5; kitaplardan üretilen ilk partide %33-50).
        kept_synth = [ln for ln in synth_lines if not _is_low_value_line(ln)]
        low_value_dropped = synth_n - len(kept_synth)
        lines += _cap_synth(kept_synth, synth_cap, seed)

    distill_n = 0
    if distill:
        from app.training.self_distill import distill_files, revalidate_line

        distill_lines = [
            ln
            for p in distill_files(lora_dir)
            for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        # Güncel içerik kapılarından yeniden geçir: kapı sıkılaştıysa eski kabul edilmiş
        # satırlar yeniden üretim gerektirmeden elenir (Kademe 2 F1/F3, 2026-10-01).
        distill_lines = [ln for ln in distill_lines if revalidate_line(ln) is None]
        distill_n = len(distill_lines)
        lines += distill_lines

    card_n = 0
    try:
        from app.memory.sqlite_store import SqliteStore

        # `build_dataset` yalnız approved + lora_eligible=1 kartı örneğe çevirir; lora-audit
        # TÜM approved kartları denetler (üst küme) → eğitime giren her kart denetlenmiş olur.
        # Bağın ZAMAN boyutu (dosya eski DB'den mi) `check_assembly_freshness`'tadır.
        card_lines = [
            ex.to_jsonl_line() for ex in build_dataset(SqliteStore().list_approved_cards())
        ]
        card_n = len(card_lines)
        lines += card_lines
    except Exception:
        pass

    merged = dedup_jsonl_lines([redact_pii_line(ln) for ln in lines])
    deduped = len(merged)

    disc_stats: dict[str, Any] | None = None
    if discipline and discipline_ratio > 0:
        disc_lines = discipline_jsonl_lines(seed=seed)
        merged, disc_stats = mix_discipline(merged, disc_lines, ratio=discipline_ratio, seed=seed)

    return AssemblyResult(
        lines=merged,
        synth_n=synth_n,
        card_n=card_n,
        deduped=deduped,
        discipline=disc_stats,
        low_value_dropped=low_value_dropped,
        distill_n=distill_n,
    )


# `scripts/assemble_sft.py` varsayılanları — tazelik denetimi AYNI parametrelerle kurar.
CANONICAL_DISCIPLINE_RATIO = 0.25
CANONICAL_SEED = 0
_REASSEMBLE_HINT = "`uv run python scripts/assemble_sft.py` ile yeniden birleştir"


@dataclass
class FreshnessResult:
    """Eğitim dosyası ↔ kanonik birleştirme bağı (pretrain-gate tazelik denetimi)."""

    status: str  # "güncel" | "BAYAT" | "doğrulanamadı"
    deterministic: bool = True
    scope: str = "tüm satırlar"  # birleştirme determinist değilse "kart satırları"
    extra: int = 0  # dosyada olup güncel birleştirmede olmayan satır
    missing: int = 0  # güncel birleştirmede olup dosyada olmayan satır
    card_extra: int = 0
    card_missing: int = 0
    order_only: bool = False  # aynı satırlar, farklı sıra
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def fresh(self) -> bool:
        return self.status == "güncel"


def _card_line_ids(lines: list[str]) -> Counter[str]:
    """Kart kaynaklı satırlar (metadata.card_id taşıyan) — kimlik: satırın tam metni."""
    out: Counter[str] = Counter()
    for ln in lines:
        try:
            meta = json.loads(ln).get("metadata") or {}
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
        if isinstance(meta, dict) and meta.get("card_id"):
            out[ln] += 1
    return out


def check_assembly_freshness(
    file_lines: list[str],
    settings: Any,
    *,
    assemble: Callable[..., AssemblyResult] | None = None,
) -> FreshnessResult:
    """Eğitilecek dosya, ŞU ANKİ DB + sentetik QA'dan kanonik birleştirmeyle aynı mı?

    Kademe 2 B1 (2026-09-28): lora-audit GÜNCEL DB kartlarını, pretrain-gate ise diskteki
    `lora_sft.jsonl`'i denetliyordu; ikisini hiçbir şey bağlamıyordu → dosya eski bir DB
    durumundan (sonradan reddedilen/düzeltilen kartlarla) kurulmuşsa denetimden geçen kart
    kümesi eğitilen küme DEĞİLDİ. Burada kanonik `assemble_sft_lines` (assemble_sft.py
    varsayılanları: disiplin %25, seed 0) bellekte İKİ KEZ kurulur: çıktı aynıysa (determinist)
    tüm satırlar, değilse yalnız kart satırları dosyayla karşılaştırılır. Uyuşmazlık → NO-GO.
    Hiçbir şey yazmaz; eğitim başlatmaz.
    """
    build = assemble or assemble_sft_lines

    def _once() -> AssemblyResult:
        return build(
            settings,
            discipline=True,
            discipline_ratio=CANONICAL_DISCIPLINE_RATIO,
            seed=CANONICAL_SEED,
            synth_cap=CANONICAL_SYNTH_CAP,
        )

    try:
        first = _once()
        second = _once()
    except Exception as exc:  # kurulamadıysa bağ kanıtlanamaz → kapalı kal (fail-closed)
        return FreshnessResult(
            status="doğrulanamadı",
            blockers=[
                f"tazelik doğrulanamadı: kanonik birleştirme kurulamadı ({exc}) — "
                "eğitim verisinin denetlenen DB ile bağı kanıtlanamıyor"
            ],
        )

    result = FreshnessResult(status="güncel", deterministic=first.lines == second.lines)
    file_cards = _card_line_ids(file_lines)
    expected_cards = _card_line_ids(first.lines)
    result.card_extra = sum((file_cards - expected_cards).values())
    result.card_missing = sum((expected_cards - file_cards).values())

    if result.deterministic:
        file_count, expected_count = Counter(file_lines), Counter(first.lines)
        result.extra = sum((file_count - expected_count).values())
        result.missing = sum((expected_count - file_count).values())
        stale = bool(result.extra or result.missing)
        result.order_only = not stale and file_lines != first.lines
        if result.order_only:
            result.warnings.append(
                "eğitim dosyası güncel birleştirmeyle aynı satırları farklı sırada taşıyor "
                f"(içerik aynı) — sırayı sabitlemek için {_REASSEMBLE_HINT}"
            )
    else:
        result.scope = "kart satırları"
        stale = bool(result.card_extra or result.card_missing)
        result.warnings.append(
            "kanonik birleştirme iki kurulumda farklı çıktı verdi (determinist değil) — "
            "tazelik yalnız KART satırları üzerinden karşılaştırıldı"
        )

    if stale:
        result.status = "BAYAT"
        detail = (
            f"{result.extra} fazla / {result.missing} eksik satır; " if result.deterministic else ""
        ) + f"kart satırı: {result.card_extra} fazla / {result.card_missing} eksik"
        db_hint = (
            " (birleştirmede hiç kart yok — DB okunamamış olabilir)"
            if first.card_n == 0 and file_cards
            else ""
        )
        result.blockers.append(
            "eğitim verisi BAYAT: lora_sft.jsonl şu anki DB + sentetik QA'dan kurulan kanonik "
            f"birleştirmeyle uyuşmuyor ({detail}){db_hint} — lora-audit'in denetlediği kart "
            f"kümesi eğitilecek küme değil; {_REASSEMBLE_HINT}"
        )
    return result


def _is_low_value_line(line: str) -> bool:
    """Sentetik JSONL satırının assistant cevabı düşük değerli mi? Okunamayan satır tutulur
    (pretrain-gate onu ayrıca 'okunamayan' diye engeller — sessiz atma yok)."""
    import json

    from app.brain.synthetic_qa_builder import is_low_value_answer

    try:
        msgs = json.loads(line).get("messages") or []
    except (json.JSONDecodeError, AttributeError, TypeError):
        return False
    answer = next(
        (str(m.get("content", "")) for m in reversed(msgs) if m.get("role") == "assistant"), ""
    )
    return bool(answer) and is_low_value_answer(answer)
