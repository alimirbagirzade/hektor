"""verify.py — eğitim adayının deterministik kontrolleri ve durum kararı (LLM'siz).

İlke: tek bir kontrol cevabın tamamını doğrulamaz. Her kontrol KAPSAMINI raporlar
("2/3 ifade", "4/6 iddia") ve kontrol edilemeyen kısımlar ayrıca listelenir.

Kontroller:
- ``hesap``        : ``a op b = c`` biçimli ifadeler ``safe_eval`` ile yeniden hesaplanır
                     (eval/exec yok — Kural 5). Tutmayan ifade ÇÜRÜTÜR.
- ``kaynak``       : her ifade/cümle turun KENDİ getirdiği parçalara karşı sözcüksel örtüşme
                     (``GroundingVerifier``) ile denetlenir. Desteklenmemek çürütme DEĞİLDİR;
                     yalnız "kontrol edilemedi" sayılır. Önceki model cevapları kanıt değildir.
- ``atif_kimligi`` : ``[paper:chunk]`` kimliklerinin turun getirdiği parçalarda olup olmadığı.
                     Geçerli kimlik iddianın desteklendiği anlamına GELMEZ; getirilmeyen kimlik
                     uydurma atıftır ve ÇÜRÜTÜR.
- ``guvenlik``     : Kural 1 (garanti/tavsiye/kesinlik dili, sır/PII) — ÇÜRÜTÜR.
- ``backtest``     : trading performans iddiası → Faz 2'ye kadar yapılamaz; insan onayı da
                     bunu geçersiz kılamaz (Kural 2: test edilmeden başarı denmez).
- ``kod_testi``    : kod içeren hedef → test koşucusu yok, yapılamaz.

Karar: çürüten kontrol → ``rejected`` (insan onayı bunu geçemez; önce düzeltilmeli). Tüm
ifadeler kapsandıysa → ``eligible`` (otomatik). Değilse geçerli gerekçeli insan onayı varsa →
``eligible`` (insan onaylı) — kısmi kapsam bilgisi korunur. Aksi halde ``review``.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.memory.retrieval_service import RetrievedChunk

# ── durum sabitleri ──────────────────────────────────────────────────────────

PASSED = "gecti"
PARTIAL = "kismi"
FAILED = "kaldi"  # çürütüldü
UNSUPPORTED = "desteklenmedi"  # çürütme değil, kanıt yok
UNAVAILABLE = "yapilamadi"
NOT_APPLICABLE = "uygulanmadi"

REFUTING_KINDS = ("hesap", "atif_kimligi", "guvenlik")


@dataclass
class Check:
    kind: str
    status: str
    scope: str = ""
    detail: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ── metni ifadelere böl ──────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_WORD4 = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}")
_BRACKETS = re.compile(r"\[[^\[\]\n]{1,400}\]")


def split_units(text: str) -> list[str]:
    """Cümle/satır birimleri; başlık ve çok kısa parçalar (<15 kr) atlanır — kısa da olsa
    hesaplanabilir ifade içeren birim (ör. ``2 + 2 = 4``) KORUNUR."""
    out: list[str] = []
    for raw in _SENT_SPLIT.split(text or ""):
        u = _BULLET.sub("", raw).strip()
        if not u or u.startswith("#"):
            continue
        core = _BRACKETS.sub("", u).strip()  # yalnız atıftan oluşan satır iddia değildir
        short = len(core) < 15 or (core.endswith(":") and len(core) < 60)
        if short and not extract_math(u):
            continue
        out.append(u)
    return out


# ── hesap ────────────────────────────────────────────────────────────────────

_EQ_RE = re.compile(
    r"(?<![\w.,])(?P<lhs>[\d(][\d\s.,+\-*/×·÷^()−]{1,118}?)\s*=\s*"
    r"(?P<rhs>-?\d+(?:[.,]\d+)?)(?P<pct>\s*%)?(?![\d])"
)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_AMBIGUOUS_NUM = re.compile(r"(?<![\d.,])\d{1,3}[.,]\d{3}(?![\d])")
_OPS = re.compile(r"\d\s*[+\-*/×·÷^−]\s*[\d(]|\)\s*[+\-*/×·÷^−]")


def _norm_expr(s: str) -> str:
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("^", "**")
    return re.sub(r"(?<=\d),(?=\d)", ".", s).strip()


def _safe_shape(expr: str) -> bool:
    """Üs patlamasına karşı: üs yalnız küçük sabit olabilir (|üs| ≤ 64)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            r = node.right
            if isinstance(r, ast.UnaryOp):
                r = r.operand
            if not isinstance(r, ast.Constant) or not isinstance(r.value, int | float):
                return False
            if abs(float(r.value)) > 64:
                return False
    return True


def extract_math(text: str) -> list[dict[str, Any]]:
    """``a op b = c`` ifadelerini bul (sonuç sayıyla biten, en az bir işlem içeren)."""
    found: list[dict[str, Any]] = []
    for m in _EQ_RE.finditer(text or ""):
        lhs = m.group("lhs").strip()
        if len(_NUM_RE.findall(lhs)) < 2 or not _OPS.search(lhs):
            continue
        found.append(
            {
                "raw": m.group(0).strip(),
                "lhs": lhs,
                "rhs": m.group("rhs"),
                "percent": bool(m.group("pct")),
                "span": [m.start(), m.end()],
            }
        )
    return found


def check_expression(item: dict[str, Any]) -> dict[str, Any]:
    """Tek ifadeyi yeniden hesapla → status: gecti | kaldi | yapilamadi."""
    from app.verification.exams.safe_eval import UnsafeExpressionError, safe_eval

    raw_lhs, raw_rhs = item["lhs"], item["rhs"]
    if _AMBIGUOUS_NUM.search(raw_lhs) or _AMBIGUOUS_NUM.search(raw_rhs):
        return {**item, "status": UNAVAILABLE, "detail": "binlik/ondalık ayracı belirsiz sayı"}
    expr = _norm_expr(raw_lhs)
    if not _safe_shape(expr):
        return {**item, "status": UNAVAILABLE, "detail": "ifade ayrıştırılamadı/izin dışı"}
    try:
        value = safe_eval(expr, {})
    except (UnsafeExpressionError, ZeroDivisionError, OverflowError, ValueError) as exc:
        return {**item, "status": UNAVAILABLE, "detail": f"hesaplanamadı: {exc}"}
    rhs_text = _norm_expr(raw_rhs)
    claimed = float(rhs_text)
    decimals = len(rhs_text.split(".")[1]) if "." in rhs_text else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9 * max(1.0, abs(value))
    candidates = [claimed]
    if item.get("percent"):
        candidates.append(claimed / 100.0)
    ok = any(math.isfinite(value) and abs(value - c) <= tol for c in candidates)
    detail = f"hesaplanan {value:.10g}, yazılan {raw_rhs}{'%' if item.get('percent') else ''}"
    return {**item, "status": PASSED if ok else FAILED, "value": value, "detail": detail}


# ── alan tespiti ─────────────────────────────────────────────────────────────

_CODE_RE = re.compile(r"```|^\s*(?:def |class |import |from \w+ import )", re.M)
_TRADING_RE = re.compile(
    r"\b(strateji\w*|backtest\w*|sharpe|sortino|drawdown|getiri\w*|al-sat|pozisyon\w*|"
    r"indikatör\w*|rsi|ema|sma|macd|trading|xauusd|komisyon|slippage|stop[- ]loss|kâr|kar\b)",
    re.I,
)
_PERF_RE = re.compile(
    r"(getiri|return|sharpe|sortino|kazan\w*|kâr|kar\b|win[ -]?rate|isabet|drawdown|cagr)"
    r"[^.\n]{0,60}?-?\d+(?:[.,]\d+)?\s*%?|%\s*\d+[^.\n]{0,40}(getiri|kâr|kazanç|return)",
    re.I,
)

DOMAINS = ("math", "code", "paper", "trading", "general")


def detect_domain(question: str, target: str, has_sources: bool) -> str:
    text = f"{question}\n{target}"
    if _CODE_RE.search(target or ""):
        return "code"
    if _TRADING_RE.search(text):
        return "trading"
    if extract_math(target) and len(_WORD4.findall(target)) < 40:
        return "math"
    if has_sources:
        return "paper"
    return "general"


# ── doğrulama ────────────────────────────────────────────────────────────────


def _chunk_objs(sources: list[dict[str, Any]]) -> list[RetrievedChunk]:
    out = []
    for s in sources:
        text = str(s.get("text") or "")
        if not text:
            continue
        out.append(
            RetrievedChunk(
                chunk_id=str(s.get("chunk_id") or ""),
                paper_id=str(s.get("paper_id") or ""),
                text=text,
                page_number=s.get("page"),
                section_name=s.get("section"),
                title=s.get("title"),
                distance=s.get("distance"),
            )
        )
    return out


def verify_target(
    target: str, *, question: str, sources: list[dict[str, Any]], domain: str
) -> dict[str, Any]:
    """Hedef metnin tüm kontrolleri + kapsam özeti (yalnız turun kendi kaynakları)."""
    from app.brain.answer_quality import verify_citations
    from app.feedback.echo import correction_safety_reason
    from app.verification.grounding_verifier import GroundingLevel, GroundingVerifier

    chunks = _chunk_objs(sources)
    units = split_units(target)
    checks: list[Check] = []

    # 1) güvenlik (Kural 1)
    reason = correction_safety_reason(target, question)
    checks.append(
        Check("guvenlik", FAILED if reason else PASSED, "tüm metin", reason or "Kural 1 temiz")
    )

    # 2) hesap
    math_items = [check_expression(it) for it in extract_math(target)]
    n_ok = sum(1 for it in math_items if it["status"] == PASSED)
    n_fail = sum(1 for it in math_items if it["status"] == FAILED)
    if math_items:
        st = FAILED if n_fail else (PASSED if n_ok == len(math_items) else PARTIAL)
        checks.append(
            Check(
                "hesap",
                st,
                f"{n_ok}/{len(math_items)} ifade",
                "Yalnız bu ifadeler yeniden hesaplandı; açıklamanın geri kalanını kanıtlamaz.",
                [{k: v for k, v in it.items() if k != "span"} for it in math_items],
            )
        )
    else:
        checks.append(Check("hesap", NOT_APPLICABLE, "0 ifade", "Hesaplanabilir ifade yok."))

    # 3) kaynak desteği (birim başına) + kapsam
    gv = GroundingVerifier()
    covered: list[bool] = []
    unit_rows: list[dict[str, Any]] = []
    n_supported = n_partial = n_claims = 0
    for u in units:
        maths_here = [it for it in math_items if it["raw"] in u]
        rest = u
        for it in maths_here:
            rest = rest.replace(it["raw"], " ")
        pure_math = bool(maths_here) and len(_WORD4.findall(rest)) <= 2
        if pure_math:
            ok = all(it["status"] == PASSED for it in maths_here)
            covered.append(ok)
            unit_rows.append({"text": u, "by": "hesap", "ok": ok})
            continue
        n_claims += 1
        level = "yok"
        if chunks:
            levels = [g.level for g in gv.verify(u, chunks)]
            if levels and all(lv == GroundingLevel.SUPPORTED for lv in levels):
                level = "destekli"
            elif any(
                lv in (GroundingLevel.SUPPORTED, GroundingLevel.PARTIALLY_SUPPORTED)
                for lv in levels
            ) or any(lv == GroundingLevel.SPECULATIVE for lv in levels):
                level = "kismi"
        n_supported += level == "destekli"
        n_partial += level == "kismi"
        covered.append(level == "destekli")
        unit_rows.append({"text": u, "by": "kaynak", "level": level, "ok": level == "destekli"})
    if not chunks:
        checks.append(
            Check(
                "kaynak",
                UNAVAILABLE,
                f"0/{n_claims} iddia",
                "Bu turda getirilen kaynak yok; kaynak desteği ölçülemedi.",
            )
        )
    elif n_claims == 0:
        checks.append(
            Check("kaynak", NOT_APPLICABLE, "0 iddia", "Kaynakla denetlenecek iddia yok.")
        )
    else:
        st = PASSED if n_supported == n_claims else (PARTIAL if n_supported else UNSUPPORTED)
        checks.append(
            Check(
                "kaynak",
                st,
                f"{n_supported}/{n_claims} iddia",
                "Sözcüksel örtüşme ölçütü (turun kendi parçaları); anlamsal doğruluk kanıtı "
                f"değildir. Kısmi: {n_partial}.",
            )
        )

    # 4) atıf kimliği (destekle KARIŞTIRILMAZ)
    cites = verify_citations(target, chunks, strict=True) if chunks else None
    if cites is None or cites.n_cited == 0:
        unresolved = bool(re.search(r"\[[A-Za-z][\w\-]+\s*:\s*[A-Za-z][\w\-]+\]", target or ""))
        checks.append(
            Check(
                "atif_kimligi",
                FAILED if unresolved else NOT_APPLICABLE,
                "0 atıf" if not unresolved else "kaynaksız atıf",
                "Kaynak yokken atıf verilmiş." if unresolved else "Atıf yok.",
            )
        )
    else:
        bad = cites.unsupported
        checks.append(
            Check(
                "atif_kimligi",
                FAILED if bad else PASSED,
                f"{cites.n_unique - len(bad)}/{cites.n_unique} kimlik",
                ("Getirilmeyen kimlik(ler): " + ", ".join(bad[:6]))
                if bad
                else "Kimlikler getirilen parçalarda var; bu, iddianın desteklendiği anlamına "
                "gelmez.",
            )
        )

    # 5) alan-özel engeller
    if domain == "trading" and _PERF_RE.search(target or ""):
        checks.append(
            Check(
                "backtest",
                UNAVAILABLE,
                "performans iddiası",
                "Trading performans iddiası backtest olmadan doğrulanamaz (Faz 2). İddiayı "
                "çıkarın veya test edilecek hipotez diline çevirin.",
            )
        )
    if domain == "code":
        checks.append(
            Check("kod_testi", UNAVAILABLE, "kod", "Kod test koşucusu yok; otomatik doğrulanamaz.")
        )

    uncovered = [r["text"] for r, ok in zip(unit_rows, covered, strict=True) if not ok]
    return {
        "domain": domain,
        "target_sha": sha256_text(target),
        "checks": [asdict(c) for c in checks],
        "coverage": {
            "units": len(units),
            "covered": sum(covered),
            "uncovered": uncovered,
            "rows": unit_rows,
        },
    }


# ── karar ────────────────────────────────────────────────────────────────────


def _by_kind(verification: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["kind"]: c for c in verification.get("checks", [])}


def human_approval_blocker(verification: dict[str, Any]) -> str | None:
    """İnsan onayının GEÇERSİZ olduğu durumlar (gerekçe döner)."""
    k = _by_kind(verification)
    for kind in REFUTING_KINDS:
        if k.get(kind, {}).get("status") == FAILED:
            return (
                f"'{kind}' kontrolü bu metni çürüttü — insan onayı yanlışlığı gösterilmiş ifadeyi "
                "geçerli kılamaz. Önce metni düzeltin."
            )
    if k.get("backtest", {}).get("status") == UNAVAILABLE:
        return k["backtest"]["detail"]
    return None


def decide(
    verification: dict[str, Any], human_approval: dict[str, Any] | None
) -> tuple[str, str, list[str], str]:
    """(status, gerekçe, gerekçe kodları, doğrulama sınıfı: auto|human|"")."""
    k = _by_kind(verification)
    refuted = [kind for kind in REFUTING_KINDS if k.get(kind, {}).get("status") == FAILED]
    if refuted:
        detail = "; ".join(f"{kind}: {k[kind]['detail']}" for kind in refuted)
        return "rejected", detail, refuted, ""
    cov = verification.get("coverage", {})
    units = int(cov.get("units", 0))
    uncovered = len(cov.get("uncovered", []))
    blocking = [
        kind for kind in ("backtest", "kod_testi") if k.get(kind, {}).get("status") == UNAVAILABLE
    ]
    math_partial = k.get("hesap", {}).get("status") == PARTIAL
    if units > 0 and uncovered == 0 and not blocking and not math_partial:
        return "eligible", "Tüm ifadeler otomatik kontrollerle kapsandı.", [], "auto"
    approval = human_approval or {}
    approved = bool(approval.get("reason")) and approval.get("target_sha") == verification.get(
        "target_sha"
    )
    if approved and human_approval_blocker(verification) is None:
        return "eligible", "Gerekçeli insan onayı (otomatik kapsam kısmi).", [], "human"
    if units == 0:
        reason = "Kontrol edilecek ifade/iddia bulunamadı — bu bir doğrulama başarısı değildir."
        return "review", reason, ["kapsam_yok"], ""
    if blocking:
        return "review", k[blocking[0]]["detail"], blocking, ""
    return (
        "review",
        f"{uncovered}/{units} ifade otomatik kontrollerle kapsanamadı — inceleme bekliyor.",
        ["kismi_kapsam"],
        "",
    )
