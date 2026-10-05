"""strategy_draft.py — sohbet cevabından YEREL modelle strateji taslağı çıkar (Faz 2B).

Akış: yerel Ollama modeli cevabı JSON taslağa çevirir → deterministik kontroller → okunur form.
Kullanıcı formu okur/düzeltir; test YALNIZ onun başlatmasıyla koşar.

İlkeler:
- Şablon/örnek strateji (``example_ir`` vb.) ile SESSİZ İKAME YOK. Çıkarım başarısızsa hata
  döner; eksik alan boş kalır ve formda kullanıcıdan istenir.
- Model değer UYDURMAMALI: cevapta olmayan maliyet/stop alanı ``null`` kalır. Maliyetler hiçbir
  zaman varsayılanla doldurulmaz.
- Kural dili desteklemeyen kural sessizce düşmez: ``unsupported_rules``'a ÖNEMLİ olarak yazılır
  → asıl strateji testi engellenir.
- Deterministik çapraz kontrol: cevapta stop / hedef / kısa yön / seans-saat kuralı geçiyor ama
  taslakta yoksa "çıkarılamadı" diye önemli desteklenmeyen kural eklenir.
- Metnin deterministik okumasıyla (``strategy_translation``) ÇELİŞEN ya da metinde dayanağı
  olmayan model değeri (ör. long için "fiyat + 2 ATR" stop'u girişin altına çevirmek) taslakta
  boşaltılır ve "karar gerekli" listelenir; orijinal → taslak farkları ayrıca döner.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.trading.strategy_spec import TestableStrategy, rule_problem

SYSTEM = (
    "Sen bir strateji ÇEVİRMENİSİN. Verilen metindeki trading stratejisini JSON'a çevirirsin. "
    "Metinde olmayan hiçbir değeri UYDURMA: bilinmeyen alan null olur. Yatırım tavsiyesi verme."
)

PROMPT = """Aşağıdaki SORU ve CEVAP'taki stratejiyi şu JSON şemasına çevir. YALNIZ JSON döndür.

{{
  "name": "kısa_ad",
  "market": "XAUUSD gibi ya da null",
  "timeframe": "1m|5m|15m|30m|1h|4h|1d ya da null",
  "direction": "long|short ya da null",
  "indicators": [{{"name": "EMA|SMA|RSI|ATR|MACD|BB", "period": 20}}],
  "entry_rules": ["ema_20 > ema_50", "rsi_14 > 55"],
  "exit_rules": ["ema_20 < ema_50"],
  "stop": {{"type": "none|fixed_pct|atr_initial|atr_trailing", "value": 2.0, "atr_period": 14}},
  "take_profit": {{"type": "none|fixed_pct|atr_multiple|r_multiple", "value": null}},
  "sizing": {{"type": "fixed_fraction|risk_per_trade", "fraction": null, "risk_pct": null}},
  "costs": {{"commission_bps_per_side": null, "slippage_bps_per_side": null,
             "spread_bps": null, "funding_bps_per_day": null}},
  "unsupported": [{{"text": "kural metni", "why": "neden çevrilemedi", "important": true}}]
}}

KURAL DİLİ: her kural "<kolon> <op> <kolon|sayı>"; op: < <= > >= == !=. Kolonlar: open, high,
low, close, volume ya da <gösterge>_<periyot> (ema_20, sma_50, rsi_14, atr_14, macd_12, bb_20).
Kural listesi VE ile birleşir. Bu dile çevrilemeyen her koşulu (formasyon, haber, hacim
patlaması, "yeni zirve", saat filtresi, VEYA mantığı, çapraz/kesişim ifadesi vb.) "unsupported"
listesine yaz; atlama. Stop yüzde ise fixed_pct değeri oran olarak (yüzde 2 → 0.02). ATR stop
sabitse atr_initial, izleyen/trailing ise atr_trailing. Metinde maliyet yoksa costs alanları null.

SORU:
{question}

CEVAP:
{answer}
"""

_STOP_RE = re.compile(r"\bstop\b|stop[- ]?loss|zarar[ıi]? (kes|durdur)|zarar kes", re.I)
_TP_RE = re.compile(r"take[- ]?profit|kâr al|kar al|hedef fiyat|kâr hedef", re.I)
_SHORT_RE = re.compile(r"\bshort\b|açığa sat|kısa pozisyon|sat pozisyon", re.I)
_JSON_RE = re.compile(r"\{.*\}", re.S)


class DraftError(ValueError):
    """Taslak çıkarılamadı (şablonla doldurulmaz; kullanıcıya gösterilir)."""


def _parse_json(raw: str) -> dict[str, Any]:
    m = _JSON_RE.search(raw or "")
    if not m:
        raise DraftError("Yerel model JSON döndürmedi; taslak çıkarılamadı (şablon kullanılmaz).")
    try:
        data = json.loads(m.group(0))
    except ValueError as exc:
        raise DraftError(f"Yerel modelin JSON'u okunamadı: {exc}") from exc
    if not isinstance(data, dict):
        raise DraftError("Yerel model geçerli bir nesne döndürmedi.")
    return data


def normalize_draft(data: dict[str, Any], *, answer: str) -> tuple[dict[str, Any], list[str]]:
    """Model çıktısını forma uygun taslağa çevir: geçersiz kuralı 'desteklenmeyen'e taşı."""
    notes: list[str] = []
    unsupported: list[dict[str, Any]] = []
    for u in data.get("unsupported") or []:
        if isinstance(u, dict) and str(u.get("text") or "").strip():
            unsupported.append(
                {
                    "text": str(u["text"])[:1000],
                    "why": str(u.get("why") or "")[:500],
                    "important": bool(u.get("important", True)),
                }
            )
        elif isinstance(u, str) and u.strip():
            unsupported.append({"text": u[:1000], "why": "", "important": True})
    rules: dict[str, list[str]] = {}
    for key in ("entry_rules", "exit_rules"):
        kept = []
        for r in data.get(key) or []:
            r = str(r or "").strip()
            why = rule_problem(r)
            if why:
                unsupported.append({"text": r, "why": why, "important": True})
                notes.append(f"'{r}' desteklenmeyen kurallara taşındı ({why}).")
            elif r:
                kept.append(r)
        rules[key] = kept
    stop = data.get("stop") if isinstance(data.get("stop"), dict) else None
    tp = data.get("take_profit") if isinstance(data.get("take_profit"), dict) else None
    stop_type = str((stop or {}).get("type") or "none")
    tp_type = str((tp or {}).get("type") or "none")
    if _STOP_RE.search(answer) and stop_type == "none":
        unsupported.append(
            {
                "text": "Cevapta stop-loss geçiyor ama taslağa çevrilemedi.",
                "why": "stop tanımı eksik/anlaşılmadı — elle girin",
                "important": True,
            }
        )
    if _TP_RE.search(answer) and tp_type == "none":
        unsupported.append(
            {
                "text": "Cevapta kâr hedefi geçiyor ama taslağa çevrilemedi.",
                "why": "hedef tanımı eksik — elle girin ya da bilinçli olarak çıkarın",
                "important": True,
            }
        )
    from app.trading.strategy_translation import _SESSION_RE

    have = " ".join(u["text"] for u in unsupported)
    for m in _SESSION_RE.finditer(answer or ""):
        if not _SESSION_RE.search(have):
            unsupported.append(
                {
                    "text": f"Seans/saat kuralı: '{m.group(0)}'",
                    "why": "motor seans/saat filtresini desteklemiyor — sessizce düşürülmedi",
                    "important": True,
                }
            )
            have += " " + m.group(0)
            notes.append(f"Cevaptaki seans/saat kuralı ('{m.group(0)}') desteklenmeyen kural.")
        break
    direction = data.get("direction")
    if _SHORT_RE.search(answer) and direction != "short":
        notes.append("Cevapta kısa (short) yön geçiyor; taslak yönü kontrol edin.")
    raw_costs = data.get("costs")
    costs: dict[str, Any] = raw_costs if isinstance(raw_costs, dict) else {}
    draft = {
        "name": str(data.get("name") or "sohbet_stratejisi")[:120],
        "market": data.get("market") or None,
        "timeframe": data.get("timeframe") or None,
        "direction": direction if direction in ("long", "short") else None,
        "indicators": [
            {"name": str(i.get("name", "")).upper(), "period": int(i.get("period") or 14)}
            for i in (data.get("indicators") or [])
            if isinstance(i, dict) and i.get("name")
        ],
        "entry_rules": rules["entry_rules"],
        "exit_rules": rules["exit_rules"],
        "stop": {
            "type": stop_type,
            "value": (stop or {}).get("value"),
            "atr_period": int((stop or {}).get("atr_period") or 14),
        },
        "take_profit": {
            "type": tp_type,
            "value": (tp or {}).get("value"),
            "atr_period": int((tp or {}).get("atr_period") or 14),
        },
        "sizing": data.get("sizing") if isinstance(data.get("sizing"), dict) else None,
        # Maliyet ASLA varsayılanla doldurulmaz: modelin vermediği alan boş kalır.
        "costs": {
            k: costs.get(k)
            for k in (
                "commission_bps_per_side",
                "slippage_bps_per_side",
                "spread_bps",
                "funding_bps_per_day",
            )
        },
        "unsupported_rules": unsupported,
    }
    return draft, notes


def validate_draft(draft: dict[str, Any]) -> tuple[TestableStrategy | None, list[str]]:
    """Taslak test edilebilir mi? (strateji | None, eksik/hatalı alan listesi)."""
    try:
        return TestableStrategy.model_validate(draft), []
    except ValidationError as exc:
        problems = []
        for e in exc.errors():
            loc = ".".join(str(x) for x in e.get("loc", ()))
            msg = str(e.get("msg", "")).removeprefix("Value error, ")
            problems.append(f"{loc}: {msg}" if loc else msg)
        return None, problems


def extract_draft(question: str, answer: str, *, llm: Any) -> dict[str, Any]:
    """Yerel modelle taslak çıkar. Dönüş: ``{draft, notes, problems, valid}``."""
    if not (answer or "").strip():
        raise DraftError("Cevap boş; strateji çıkarılamaz.")
    raw = llm.generate(
        PROMPT.format(question=question[:2000], answer=answer[:8000]),
        system=SYSTEM,
        temperature=0.0,
        fmt="json",
        seed=42,
        max_tokens=1500,
    )
    from app.trading.strategy_translation import (
        compare_to_proposal,
        extract_proposal,
        reconcile_draft,
    )

    model_draft, notes = normalize_draft(_parse_json(raw), answer=answer)
    proposal = extract_proposal(answer)
    # Model ↔ metin çelişkisi ya da metinde dayanağı olmayan değer TAHMİNLE bırakılmaz.
    draft, pending = reconcile_draft(model_draft, proposal)
    for p in pending:
        notes.append(f"KARAR GEREKLİ ({p['field']}): {p['why']}")
    spec, problems = validate_draft(draft)
    return {
        "draft": draft,
        "model_draft": model_draft,
        "proposal": proposal,
        "pending": pending,
        "original_vs_draft": compare_to_proposal(proposal, draft),
        "notes": notes,
        "problems": problems,
        "valid": spec is not None and not pending,
        "raw_model_output": raw[:6000],
    }
