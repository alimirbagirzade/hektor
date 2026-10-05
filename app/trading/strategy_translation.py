"""strategy_translation.py — sohbet önerisi → çıkarılan taslak → onaylı nihai strateji farkları.

Sorun (2026-10-05 canlı denemesi): sohbet modeli long stop'u "fiyat + 2 ATR" yazdı; taslak
çıkarma aşaması bunu SESSİZCE girişin altındaki 2×ATR stop'a çevirdi. Böyle düzeltmeler artık
görünür olur:

1. **Orijinal öneri** (``extract_proposal``) — cevap metninden DETERMİNİSTİK (regex, LLM yok)
   okunan iddialar: yön, stop (ATR/yüzde, kat, fiyata göre yönü), hedef, işlem başına risk,
   pozisyon büyüklüğü/kaldıraç, seans/saat kuralları ve metinde geçen göstergeler.
2. **Çıkarılan taslak** — yerel modelin JSON'u (``strategy_draft``). Model ile metnin
   deterministik okuması ÇELİŞİRSE ya da model metinde olmayan bir değer verirse o alan
   taslakta BOŞ bırakılır ve "karar gerekli" olarak listelenir (``reconcile_draft``) — tahminle
   tamamlanıp onaylanmış gibi sunulmaz.
3. **Nihai strateji** — kullanıcının kaydettiği tanım. Test YALNIZ, orijinal → nihai ve
   taslak → nihai farklarının tamamını gördüğünü işaretleyen insan onayından sonra koşar
   (``build_review`` + ``StrategyStore.save_approval``).

Fark kategorileri: giriş, çıkış, yön, stop, hedef, risk, pozisyon büyüklüğü, seans (+ piyasa /
zaman dilimi / maliyet "diğer"). Tür: eklendi | değişti | çıkarıldı; çelişkili orijinal ifade
ayrıca ``conflict`` ile işaretlenir.

Sınır: metin okuması regex tabanlıdır; kapsamadığı ifadeleri (ör. eşik değerli serbest giriş
cümleleri) tanımaz. Bu yüzden giriş/çıkış için yalnız gösterge kümesi karşılaştırılır ve
orijinal cümleler kullanıcıya AYNEN gösterilir; eksik okuma "fark yok" anlamına gelmez.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

CAT_TR = {
    "giris": "Giriş",
    "cikis": "Çıkış",
    "yon": "Yön",
    "stop": "Stop",
    "hedef": "Hedef",
    "risk": "Risk",
    "boyut": "Pozisyon büyüklüğü",
    "seans": "Seans",
    "diger": "Diğer",
}
KIND_TR = {"eklendi": "eklendi", "degisti": "değişti", "cikarildi": "çıkarıldı"}

_NUM = r"(\d+(?:[.,]\d+)?)"
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_STOP_KW = re.compile(
    r"stop[- ]?loss|\bstop\b|zarar[ıi]?\s*(?:kes|durdur)\w*|zarar\s*kes\w*|\bsl\b", re.I
)
_TP_KW = re.compile(r"take[- ]?profit|k[âa]r\s*al\w*|k[âa]r\s*hedef\w*|hedef\w*|\btp\b", re.I)
_LONG_RE = re.compile(
    r"\blong\b|uzun\s+pozisyon|al[ıi]m\s+pozisyon\w*|\bbuy\b|al[ıi]ş\s+sinyal\w*", re.I
)
_SHORT_RE = re.compile(
    r"\bshort\b|açığa\s+sat\w*|kısa\s+pozisyon\w*|sat[ıi]ş\s+pozisyon\w*|\bsell\b", re.I
)
_SESSION_RE = re.compile(
    r"(?:londra|london|new\s*york|\bny\b|asya|asia|tokyo|sydney|frankfurt)\s*"
    r"(?:seans\w*|session\w*|açılış\w*)|\bseans\w*|\bsession\w*|"
    r"\b\d{1,2}[:.]\d{2}\s*[-–]\s*\d{1,2}[:.]\d{2}\b|\bsaat\s+\d{1,2}\b",
    re.I,
)
_TRAIL_RE = re.compile(r"takip\s+eden|trailing|izleyen|iz\s*süren", re.I)
_ATR_SIGNED = re.compile(
    r"(fiyat\w*|giriş\w*|entry|price|kapanış\w*|close)\s*(?:fiyat\w*\s*)?"
    r"([+\-−–])\s*" + _NUM + r"\s*(?:[x×*]\s*)?ATR",
    re.I,
)
_ATR_MULT = re.compile(_NUM + r"\s*(?:[x×*]\s*)?(?:kat\s+)?ATR|ATR\s*(?:[x×*]\s*)" + _NUM, re.I)
_PCT = re.compile(r"%\s*" + _NUM + r"|" + _NUM + r"\s*%|yüzde\s+" + _NUM, re.I)
_R_MULT = re.compile(_NUM + r"\s*R\b|1\s*:\s*" + _NUM, re.I)
_BELOW = re.compile(r"alt[ıi]n[ae]|alt[ıi]nda|aşağ[ıi]s[ıi]n[ae]|below|under", re.I)
_ABOVE = re.compile(r"üst[üu]n[ae]|üst[üu]nde|üzerin[ae]|üzerinde|yukar[ıi]s[ıi]n[ae]|above", re.I)
_IND_RE = re.compile(r"\b(EMA|SMA|RSI|ATR|MACD|BB|Bollinger)\s*[\(\-]?\s*(\d{1,4})\b", re.I)
_LEV_RE = re.compile(_NUM + r"\s*[x×]\s*kaldıraç|kaldıraç\w*\s*" + _NUM + r"\s*[x×]?", re.I)
_SIZE_RE = re.compile(
    r"pozisyon\s+büyüklü\w*|\blot\w*|özsermaye\w*|sermaye\w*\s+%|kaldıraç\w*|position\s+size",
    re.I,
)
_ENTRY_RE = re.compile(r"\bgir\w*|\bal\b|alım|entry|\bbuy\b|long\s+aç|short\s+aç", re.I)
_EXIT_RE = re.compile(r"\bçık\w*|\bkapat\w*|\bexit\b|pozisyonu\s+kapa", re.I)


def _f(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _first_num(m: re.Match[str]) -> float | None:
    for g in m.groups():
        if g is not None and re.fullmatch(_NUM, g):
            return _f(g)
    return None


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]


def _clause(sentence: str, kw: re.Pattern[str], other: re.Pattern[str]) -> str:
    """Anahtar kelimeden (ör. stop) diğerine (ör. hedef) kadar olan parça."""
    m = kw.search(sentence)
    if not m:
        return ""
    rest = sentence[m.start() :]
    o = other.search(rest, len(m.group(0)))
    return rest[: o.start()] if o else rest


def _level(clause: str, sentence: str) -> dict[str, Any]:
    """Stop/hedef ifadesinden tür, kat/oran ve fiyata göre yön."""
    out: dict[str, Any] = {"excerpt": sentence[:300], "clause": clause[:200]}
    signed = _ATR_SIGNED.search(clause)
    if signed:
        sign = signed.group(2)
        out.update(
            kind="atr",
            value=_f(signed.group(3)),
            side="above" if sign == "+" else "below",
            expr=signed.group(0),
        )
        return out
    atr = _ATR_MULT.search(clause)
    pct = _PCT.search(clause)
    rm = _R_MULT.search(clause)
    if atr:
        out.update(kind="atr", value=_first_num(atr), expr=atr.group(0))
    elif rm:
        out.update(kind="r", value=_first_num(rm), expr=rm.group(0))
    elif pct:
        v = _first_num(pct)
        out.update(kind="pct", value=(v / 100.0 if v is not None else None), expr=pct.group(0))
    else:
        out.update(kind="unknown", value=None, expr="")
    if _BELOW.search(clause):
        out["side"] = "below"
    elif _ABOVE.search(clause):
        out["side"] = "above"
    else:
        out["side"] = None
    out["trailing"] = bool(_TRAIL_RE.search(clause))
    return out


def extract_proposal(answer: str) -> dict[str, Any]:
    """Cevap metninden deterministik okunan strateji iddiaları (LLM YOK)."""
    sents = sentences(answer)
    long_hits = [s for s in sents if _LONG_RE.search(s)]
    short_hits = [s for s in sents if _SHORT_RE.search(s)]
    stops = []
    targets = []
    risk = None
    risk_excerpt = ""
    sizing_excerpts = []
    leverage = None
    sessions = []
    entry_ex = []
    exit_ex = []
    inds: set[str] = set()
    for s in sents:
        if _STOP_KW.search(s):
            c = _clause(s, _STOP_KW, _TP_KW)
            if c:
                lv = _level(c, s)
                lv["trailing"] = bool(_TRAIL_RE.search(c))
                stops.append(lv)
        if _TP_KW.search(s):
            c = _clause(s, _TP_KW, _STOP_KW)
            if c:
                targets.append(_level(c, s))
        if re.search(r"\brisk\w*", s, re.I) and risk is None:
            pm = _PCT.search(s)
            if pm:
                v = _first_num(pm)
                risk = v / 100.0 if v is not None else None
                risk_excerpt = s[:300]
        if _SIZE_RE.search(s):
            sizing_excerpts.append(s[:300])
            lm = _LEV_RE.search(s)
            if lm and leverage is None:
                leverage = _first_num(lm)
        for m in _SESSION_RE.finditer(s):
            sessions.append({"text": m.group(0), "excerpt": s[:300]})
        if _ENTRY_RE.search(s):
            entry_ex.append(s[:300])
        if _EXIT_RE.search(s):
            exit_ex.append(s[:300])
        for m in _IND_RE.finditer(s):
            name = m.group(1).lower()
            name = "bb" if name == "bollinger" else name
            inds.add(f"{name}_{int(m.group(2))}")
    directions = []
    if long_hits:
        directions.append("long")
    if short_hits:
        directions.append("short")
    return {
        "directions": directions,
        "direction_excerpts": (long_hits + short_hits)[:4],
        "stops": stops,
        "targets": targets,
        "risk_pct": risk,
        "risk_excerpt": risk_excerpt,
        "sizing_excerpts": sizing_excerpts[:3],
        "leverage": leverage,
        "sessions": sessions,
        "indicators": sorted(inds),
        "entry_excerpts": entry_ex[:4],
        "exit_excerpts": exit_ex[:4],
        "method": "regex (deterministik); kapsamadığı ifade 'fark yok' anlamına gelmez",
    }


def proposal_sha(answer: str) -> str:
    return hashlib.sha256((answer or "").encode("utf-8")).hexdigest()


# ── okunur değer biçimleri ──────────────────────────────────────────────────


def _lvl_text(lv: dict[str, Any] | None) -> str:
    if not lv:
        return "yok"
    side = {"above": "fiyatın/girişin ÜSTÜ", "below": "fiyatın/girişin ALTI"}.get(
        lv.get("side") or ""
    )
    base = lv.get("expr") or lv.get("clause") or lv.get("excerpt") or ""
    return f"'{base.strip()}'" + (f" ({side})" if side else "")


def _stop_text(stop: dict[str, Any] | None, direction: str | None) -> str:
    st = stop or {}
    t = st.get("type")
    if t in (None, ""):
        return "(karar gerekli — boş)"
    if t == "none":
        return "stop yok"
    side = "altı" if direction == "long" else "üstü" if direction == "short" else "aleyhe tarafı"
    v = st.get("value")
    if t == "fixed_pct":
        return f"girişin %{(v or 0) * 100:g} {side} (sabit)"
    kind = "takip eden" if t == "atr_trailing" else "sabit"
    return (
        f"girişin {v:g}×ATR({st.get('atr_period', 14)}) {side} ({kind})"
        if v
        else f"{t} (değer yok)"
    )


def _tp_text(tp: dict[str, Any] | None, direction: str | None) -> str:
    t = (tp or {}).get("type")
    if t in (None, ""):
        return "(karar gerekli — boş)"
    if t == "none":
        return "hedef yok"
    v = (tp or {}).get("value")
    side = "üstü" if direction == "long" else "altı" if direction == "short" else "lehe tarafı"
    if t == "fixed_pct":
        return f"girişin %{(v or 0) * 100:g} {side}"
    if t == "r_multiple":
        return f"{v} R"
    return f"{v}×ATR {side}"


def _size_text(sz: dict[str, Any] | None) -> str:
    if not sz or not sz.get("type"):
        return "(karar gerekli — boş)"
    if sz.get("type") == "fixed_fraction":
        lev = sz.get("max_leverage", 1)
        return f"özsermayenin {sz.get('fraction')} katı nominal (kaldıraç ≤ {lev})"
    risk = float(sz.get("risk_pct") or 0) * 100
    return f"işlem başına %{risk:g} risk (kaldıraç ≤ {sz.get('max_leverage', 1)})"


def _rule_cols(spec: dict[str, Any]) -> set[str]:
    cols: set[str] = set()
    for r in [*(spec.get("entry_rules") or []), *(spec.get("exit_rules") or [])]:
        for tok in re.findall(r"[a-z]+_\d+", str(r)):
            cols.add(tok)
    for i in spec.get("indicators") or []:
        if isinstance(i, dict) and i.get("name"):
            name = str(i["name"]).lower()
            cols.add(f"{'bb' if name == 'bollinger' else name}_{int(i.get('period') or 14)}")
    return cols


def _expected_side(direction: str | None, what: str) -> str | None:
    if direction not in ("long", "short"):
        return None
    adverse = "below" if direction == "long" else "above"
    favorable = "above" if direction == "long" else "below"
    return adverse if what == "stop" else favorable


def _item(cat: str, kind: str, original: str, now: str, note: str = "", **kw: Any) -> dict:
    key_src = json.dumps([cat, kind, original, now], ensure_ascii=False)
    return {
        "key": hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:12],
        "category": cat,
        "category_label": CAT_TR[cat],
        "kind": kind,
        "kind_label": KIND_TR[kind],
        "original": original,
        "now": now,
        "note": note,
        **kw,
    }


def _session_unsupported(spec: dict[str, Any]) -> list[str]:
    return [
        str(u.get("text") or "")
        for u in (spec.get("unsupported_rules") or [])
        if isinstance(u, dict) and _SESSION_RE.search(str(u.get("text") or ""))
    ]


def _cmp_level(
    cat: str,
    levels: list[dict[str, Any]],
    spec_lvl: dict[str, Any],
    direction: str | None,
    to_text: Any,
) -> list[dict[str, Any]]:
    out = []
    t = (spec_lvl or {}).get("type")
    now = to_text(spec_lvl, direction)
    if not levels:
        if t not in (None, "", "none"):
            out.append(_item(cat, "eklendi", "metinde yok", now, "Öneride olmayan değer eklendi."))
        return out
    lv = levels[0]
    orig = _lvl_text(lv)
    want = _expected_side(direction, cat)
    conflict = bool(want and lv.get("side") and lv["side"] != want)
    conflict_note = ""
    if conflict:
        conflict_note = (
            f"ÇELİŞKİ: {direction} için {'stop' if cat == 'stop' else 'hedef'} "
            f"{'girişin üstünde' if lv['side'] == 'above' else 'girişin altında'} yazılmış "
            f"({'anında stop' if cat == 'stop' else 'hedef zararda'} olur). Orijinal öneri bu "
            "haliyle test EDİLEMEZ; nihai değer bir yorumdur."
        )
    if t in (None, ""):
        out.append(
            _item(cat, "cikarildi", orig, now, conflict_note or "Karar gerekli.", conflict=conflict)
        )
        return out
    if t == "none":
        out.append(_item(cat, "cikarildi", orig, now, conflict_note, conflict=conflict))
        return out
    v = spec_lvl.get("value")
    kind_map = {"atr_initial": "atr", "atr_trailing": "atr", "atr_multiple": "atr"}
    kind_map.update({"fixed_pct": "pct", "r_multiple": "r"})
    spec_kind = kind_map.get(str(t), "")
    changed = conflict
    notes = [conflict_note] if conflict_note else []
    if lv.get("kind") not in ("unknown", None) and lv["kind"] != spec_kind:
        changed = True
        notes.append(f"Tür değişti ({lv['kind']} → {spec_kind}).")
    elif (
        lv.get("value") is not None and v is not None and abs(float(lv["value"]) - float(v)) > 1e-9
    ):
        changed = True
        notes.append(f"Değer değişti ({lv['value']:g} → {float(v):g}).")
    if cat == "stop" and bool(lv.get("trailing")) != (t == "atr_trailing"):
        changed = True
        notes.append("Takip eden/sabit davranışı değişti.")
    if changed:
        out.append(_item(cat, "degisti", orig, now, " ".join(notes), conflict=conflict))
    return out


def compare_to_proposal(proposal: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Orijinal öneri (metin okuması) ↔ taslak/nihai strateji farkları."""
    items: list[dict[str, Any]] = []
    direction = spec.get("direction")
    dirs = proposal.get("directions") or []
    if len(dirs) == 1 and direction != dirs[0]:
        items.append(_item("yon", "degisti", dirs[0], str(direction or "(karar gerekli — boş)")))
    elif len(dirs) > 1:
        other = [d for d in dirs if d != direction]
        items.append(
            _item(
                "yon",
                "cikarildi",
                "long + short",
                str(direction or "(karar gerekli — boş)"),
                f"Metinde iki yön var; motor tek yön test eder — {', '.join(other)} TEST EDİLMEZ.",
            )
        )
    elif not dirs and direction:
        items.append(_item("yon", "eklendi", "metinde yön yok", str(direction)))
    eff_dir = direction or (dirs[0] if len(dirs) == 1 else None)
    items += _cmp_level(
        "stop", proposal.get("stops") or [], spec.get("stop") or {}, eff_dir, _stop_text
    )
    items += _cmp_level(
        "hedef", proposal.get("targets") or [], spec.get("take_profit") or {}, eff_dir, _tp_text
    )
    sz = spec.get("sizing") or {}
    risk = proposal.get("risk_pct")
    if risk is not None:
        if sz.get("type") != "risk_per_trade" or sz.get("risk_pct") is None:
            items.append(
                _item(
                    "risk",
                    "degisti" if sz else "cikarildi",
                    f"işlem başına %{risk * 100:g} risk",
                    _size_text(sz or None),
                )
            )
        elif abs(float(sz["risk_pct"]) - risk) > 1e-9:
            items.append(
                _item("risk", "degisti", f"%{risk * 100:g}", f"%{float(sz['risk_pct']) * 100:g}")
            )
    elif sz.get("type") == "risk_per_trade":
        items.append(_item("risk", "eklendi", "metinde risk yüzdesi yok", _size_text(sz)))
    if not proposal.get("sizing_excerpts") and risk is None and sz.get("type"):
        items.append(
            _item(
                "boyut",
                "eklendi",
                "metinde pozisyon büyüklüğü yok",
                _size_text(sz),
                "Varsayım eklendi — sonuç bu varsayıma bağlıdır.",
            )
        )
    lev = proposal.get("leverage")
    if lev is not None and sz and abs(float(sz.get("max_leverage") or 1) - lev) > 1e-9:
        items.append(
            _item(
                "boyut", "degisti", f"kaldıraç {lev:g}x", f"kaldıraç ≤ {sz.get('max_leverage', 1)}"
            )
        )
    sess = proposal.get("sessions") or []
    if sess:
        kept = _session_unsupported(spec)
        removed = [str(r) for r in (spec.get("removed_rules") or []) if _SESSION_RE.search(str(r))]
        orig = "; ".join(sorted({s["text"] for s in sess}))
        if kept:
            pass  # hâlâ ÖNEMLİ desteklenmeyen kural olarak duruyor → test engelli, fark yok
        elif removed:
            items.append(
                _item(
                    "seans",
                    "cikarildi",
                    orig,
                    "kural yok (basitleştirmede açıkça çıkarıldı)",
                    "Seans filtresi motor tarafından desteklenmiyor; test filtre OLMADAN yapılır.",
                )
            )
        else:
            items.append(
                _item(
                    "seans",
                    "cikarildi",
                    orig,
                    "kural yok",
                    "Seans/saat kuralı nihai tanımda yok; desteklenmeyen olarak da kayıtlı değil.",
                )
            )
    stop_atr = {f"atr_{(spec.get('stop') or {}).get('atr_period', 14)}"}
    tp_atr = {f"atr_{(spec.get('take_profit') or {}).get('atr_period', 14)}"}
    text_inds = set(proposal.get("indicators") or [])
    text_inds = {i for i in text_inds if not i.startswith("atr_")} | (
        {i for i in text_inds if i.startswith("atr_")} - stop_atr - tp_atr
    )
    cols = _rule_cols(spec)
    for ind in sorted(text_inds - cols):
        items.append(
            _item(
                "giris",
                "cikarildi",
                f"metinde {ind}",
                "kurallarda yok",
                "Metinde geçen gösterge giriş/çıkış kurallarına çevrilmedi.",
            )
        )
    used = {c for c in cols if not c.startswith("atr_")}
    for col in sorted(used - text_inds):
        items.append(
            _item(
                "giris",
                "eklendi",
                "metinde yok",
                f"kurallarda {col}",
                "Kurallarda metinde geçmeyen gösterge var.",
            )
        )
    return items


def _lines(spec: dict[str, Any], key: str) -> list[str]:
    return [str(x).strip() for x in (spec.get(key) or []) if str(x).strip()]


def compare_specs(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Taslak (ya da üst varyant) ↔ nihai strateji: kullanıcının yaptığı değişiklikler."""
    items: list[dict[str, Any]] = []
    for key, cat in (("entry_rules", "giris"), ("exit_rules", "cikis")):
        b, a = _lines(before, key), _lines(after, key)
        for r in [x for x in b if x not in a]:
            items.append(_item(cat, "cikarildi", r, "—"))
        for r in [x for x in a if x not in b]:
            items.append(_item(cat, "eklendi", "—", r))
    if before.get("direction") != after.get("direction"):
        items.append(
            _item(
                "yon", "degisti", str(before.get("direction") or "boş"), str(after.get("direction"))
            )
        )
    d = after.get("direction")
    for key, cat, fn in (("stop", "stop", _stop_text), ("take_profit", "hedef", _tp_text)):
        bl: dict[str, Any] = before.get(key) or {}
        al: dict[str, Any] = after.get(key) or {}
        bt = (bl.get("type"), bl.get("value"), bl.get("atr_period", 14) if bl.get("type") else None)
        at = (al.get("type"), al.get("value"), al.get("atr_period", 14) if al.get("type") else None)
        if (bt[0] or "none") == "none" and (at[0] or "none") == "none":
            continue
        if bt != at:
            items.append(_item(cat, "degisti", fn(bl, before.get("direction")), fn(al, d)))
    bs, as_ = before.get("sizing") or {}, after.get("sizing") or {}
    if (bs.get("type"), bs.get("risk_pct")) != (as_.get("type"), as_.get("risk_pct")) and (
        "risk_per_trade" in (bs.get("type"), as_.get("type"))
    ):
        items.append(_item("risk", "degisti", _size_text(bs or None), _size_text(as_ or None)))
    elif (bs.get("type"), bs.get("fraction"), bs.get("max_leverage", 1)) != (
        as_.get("type"),
        as_.get("fraction"),
        as_.get("max_leverage", 1),
    ):
        items.append(_item("boyut", "degisti", _size_text(bs or None), _size_text(as_ or None)))
    bu = {str(u.get("text")) for u in before.get("unsupported_rules") or [] if isinstance(u, dict)}
    au = {str(u.get("text")) for u in after.get("unsupported_rules") or [] if isinstance(u, dict)}
    for t in sorted(bu - au):
        cat = "seans" if _SESSION_RE.search(t) else "diger"
        items.append(_item(cat, "cikarildi", t, "desteklenmeyen kural listesinden çıkarıldı"))
    for t in sorted(au - bu):
        cat = "seans" if _SESSION_RE.search(t) else "diger"
        items.append(_item(cat, "eklendi", "—", t))
    for key, label in (("market", "piyasa"), ("timeframe", "zaman dilimi")):
        if (before.get(key) or None) != (after.get(key) or None):
            items.append(
                _item(
                    "diger",
                    "degisti",
                    f"{label}: {before.get(key) or 'boş'}",
                    f"{label}: {after.get(key) or 'boş'}",
                )
            )
    bc, ac = before.get("costs") or {}, after.get("costs") or {}
    for k in (
        "commission_bps_per_side",
        "slippage_bps_per_side",
        "spread_bps",
        "funding_bps_per_day",
    ):
        if bc.get(k) != ac.get(k):
            items.append(_item("diger", "degisti", f"{k}: {bc.get(k)}", f"{k}: {ac.get(k)}"))
    return items


# ── taslak uzlaştırma (çıkarım aşaması) ──────────────────────────────────────


def reconcile_draft(
    draft: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Model taslağını metnin deterministik okumasıyla uzlaştır.

    Çelişen ya da metinde dayanağı olmayan değer TAHMİNLE BIRAKILMAZ: alan boşaltılır, modelin
    önerisi ve metnin ifadesi ``pending`` listesinde kullanıcıya gösterilir. Dönüş:
    ``(taslak, bekleyen kararlar)``.
    """
    d = json.loads(json.dumps(draft))
    pending: list[dict[str, Any]] = []
    dirs = proposal.get("directions") or []

    def hold(field: str, why: str, model_value: Any, text: str) -> None:
        pending.append(
            {"field": field, "why": why, "model_value": model_value, "original_text": text[:300]}
        )

    direction = d.get("direction")
    if len(dirs) > 1:
        hold(
            "direction",
            "Metinde hem long hem short var; hangisinin test edileceğini seçin "
            "(diğeri test EDİLMEZ).",
            direction,
            "; ".join(proposal.get("direction_excerpts") or []),
        )
        d["direction"] = None
    elif len(dirs) == 1 and direction and direction != dirs[0]:
        hold(
            "direction",
            f"Model yönü '{direction}' çıkardı, metin '{dirs[0]}' diyor.",
            direction,
            "; ".join(proposal.get("direction_excerpts") or []),
        )
        d["direction"] = None
    elif not dirs and direction:
        hold("direction", "Metinde yön yok; modelin seçimi tahmindir.", direction, "")
        d["direction"] = None
    eff = d.get("direction") or (dirs[0] if len(dirs) == 1 else None)

    for key, cat, levels in (
        ("stop", "stop", proposal.get("stops") or []),
        ("take_profit", "hedef", proposal.get("targets") or []),
    ):
        spec_lvl = d.get(key) or {}
        t = spec_lvl.get("type") or "none"
        if not levels:
            if t != "none":
                hold(
                    key,
                    f"Metinde {cat} yok; model '{t} {spec_lvl.get('value')}' önerdi — eklenmedi.",
                    dict(spec_lvl),
                    "",
                )
                d[key] = {
                    "type": "none",
                    "value": None,
                    "atr_period": spec_lvl.get("atr_period", 14),
                }
            continue
        lv = levels[0]
        want = _expected_side(eff, cat)
        if want and lv.get("side") and lv["side"] != want:
            hold(
                key,
                f"ÇELİŞKİ: {eff} için {cat} metinde {_lvl_text(lv)}. Model bunu "
                f"'{t} {spec_lvl.get('value')}' olarak çevirdi (yön düzeltmesi). Düzeltme "
                "tahmindir — değeri siz seçin.",
                dict(spec_lvl),
                lv.get("excerpt", ""),
            )
            d[key] = {"type": "", "value": None, "atr_period": spec_lvl.get("atr_period", 14)}
            continue
        if t == "none":
            continue  # strategy_draft zaten "çevrilemedi" desteklenmeyen kuralı ekler
        diffs = _cmp_level(cat, [lv], spec_lvl, eff, _stop_text if cat == "stop" else _tp_text)
        if diffs:
            hold(
                key,
                f"Model {cat} değerini metinden farklı çıkardı: {diffs[0]['note']}",
                dict(spec_lvl),
                lv.get("excerpt", ""),
            )
            d[key] = {"type": "", "value": None, "atr_period": spec_lvl.get("atr_period", 14)}

    sz = d.get("sizing") if isinstance(d.get("sizing"), dict) else None
    risk = proposal.get("risk_pct")
    if sz and sz.get("type"):
        bad = ""
        if risk is not None and (
            sz.get("type") != "risk_per_trade"
            or sz.get("risk_pct") is None
            or abs(float(sz["risk_pct"]) - risk) > 1e-9
        ):
            bad = (
                f"Metin işlem başına %{risk * 100:g} risk diyor; model '{_size_text(sz)}' çıkardı."
            )
        elif risk is None and not proposal.get("sizing_excerpts"):
            bad = "Metinde pozisyon büyüklüğü/risk yok; modelin değeri tahmindir — eklenmedi."
        if bad:
            hold("sizing", bad, dict(sz), proposal.get("risk_excerpt") or "")
            d["sizing"] = None
    return d, pending


def review_sha(items: list[dict[str, Any]]) -> str:
    keys = sorted(i["key"] for i in items)
    return hashlib.sha256(json.dumps(keys).encode("utf-8")).hexdigest()


def build_review(
    translation: dict[str, Any] | None,
    spec: dict[str, Any],
    *,
    parent_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Onay ekranı: orijinal → nihai ve taslak → nihai farkları + çelişkiler + özet.

    Onay bu listedeki HER öğenin görüldüğünü (``key``) içermelidir; ``review_sha`` liste
    değişince değişir → eski onay geçersizleşir.
    """
    if translation is None:
        items: list[dict[str, Any]] = []
        return {
            "translation_id": "",
            "original_text": "",
            "original_vs_final": [],
            "draft_vs_final": [],
            "items": items,
            "conflicts": [],
            "review_sha": review_sha(items),
            "note": "Bu strateji bir sohbet önerisinden çıkarılmadı (kaynak öneri kaydı yok); "
            "yine de nihai tanım açıkça onaylanmalı.",
        }
    proposal = translation.get("proposal") or {}
    ovf = [dict(i, stage="orijinal→nihai") for i in compare_to_proposal(proposal, spec)]
    base = parent_spec if parent_spec is not None else (translation.get("draft") or {})
    dvf = [dict(i, stage="taslak→nihai") for i in compare_specs(base, spec)]
    items = ovf + dvf
    return {
        "translation_id": translation.get("translation_id", ""),
        "original_text": translation.get("original_text", ""),
        "original_sha": translation.get("original_sha", ""),
        "proposal": proposal,
        "original_vs_final": ovf,
        "draft_vs_final": dvf,
        "items": items,
        "conflicts": [i for i in items if i.get("conflict")],
        "review_sha": review_sha(items),
        "note": "Orijinal öneri metinden deterministik okunur (regex); okunamayan ifadeler "
        "listelenmez — orijinal metni de okuyun.",
    }
