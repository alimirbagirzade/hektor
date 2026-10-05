"""claim_check.py — metindeki performans iddiasını KAYITLI bir test koşusuyla karşılaştır.

Sonuç en fazla "kayıtlı hesapla eşleşti"dir: iddia edilen sayıların, belirtilen strateji/veri/
dönem/maliyet/metrik tanımı/motor sürümüyle yapılmış kayıtlı hesapla tutarlı olduğu. Stratejinin
gelecekte başarılı olacağı ya da modelin daha iyi trader olduğu anlamına GELMEZ.

Yalnız sayı eşleşmesi YETMEZ; hepsi gerekir:
1. Bütünlük: koşunun parmak izi, KAYITLI strateji tanımından + temiz veri özetinden + dönemden +
   maliyetlerden + metrik/motor sürümünden yeniden hesaplanır ve kayıtla aynı olmalı.
2. Motor sürümü güncel olmalı (eski motorla hesap → "yeniden koşun").
3. Metin dönemi belirtmeli (YYYY-AA-GG başlangıç ve bitiş) ve koşunun dönemiyle aynı olmalı.
4. Metindeki her metrik sayısı (getiri %, düşüş %, Sharpe, işlem sayısı, kazanma oranı %,
   profit factor) kayıtlı değere yazıldığı hassasiyette eşit olmalı; en az bir metrik olmalı.
5. Geliştirme dönemi sonucu "örneklem dışı/OOS" diye sunulamaz; maliyetli hesap "maliyetsiz"
   diye sunulamaz.
"""

from __future__ import annotations

import re
from typing import Any

MATCHED = "eslesti"
MISMATCHED = "eslesmedi"

_METRICS = (
    ("total_return_pct", r"(?:toplam\s+)?getiri|return"),
    ("max_drawdown_pct", r"(?:azami\s+|maks(?:imum)?\s+)?d[üu]ş[üu]ş|drawdown"),
    ("sharpe", r"sharpe"),
    ("n_trades", r"i[şs]lem\s+say[ıi]s[ıi]|trade(?:s| say[ıi]s[ıi])"),
    ("win_rate_pct", r"kazan(?:an|ma)\s+(?:i[şs]lem|oran[ıi])|win[ -]?rate"),
    ("profit_factor", r"profit\s+factor|k[âa]r\s+fakt[öo]r[üu]"),
)
_NUM = r"(-?\d+(?:[.,]\d+)?)"
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_OOS_RE = re.compile(r"örneklem\s+dışı|oos\b|out[- ]of[- ]sample", re.I)
_COSTLESS_RE = re.compile(r"maliyetsiz|komisyonsuz|masrafsız", re.I)


def extract_metric_claims(text: str) -> list[dict[str, Any]]:
    out = []
    for key, label in _METRICS:
        pat = re.compile(rf"(?:{label})[^0-9\n%-]{{0,25}}%?\s*{_NUM}", re.I)
        for m in pat.finditer(text or ""):
            raw = m.group(1).replace(",", ".")
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            out.append(
                {
                    "metric": key,
                    "raw": m.group(0).strip(),
                    "value": float(raw),
                    "decimals": decimals,
                }
            )
    return out


def _close(claimed: float, actual: float | None, decimals: int) -> bool:
    if actual is None:
        return False
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return abs(float(actual) - claimed) <= tol


def check_claim(text: str, run: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    """→ ``{status, scope, detail, items}`` (verify.Check alanlarıyla uyumlu)."""
    from app.trading import event_engine as ee
    from app.trading.strategy_spec import TestableStrategy
    from app.trading.strategy_testing import fingerprint

    problems: list[str] = []
    try:
        spec = TestableStrategy.model_validate(strategy["spec"])
        fp = fingerprint(spec, run["clean_sha256"], run["period_start"], run["period_end"])
    except Exception as exc:  # kayıt bozuk → eşleşme iddia edilemez
        return {
            "status": MISMATCHED,
            "scope": "bütünlük",
            "detail": f"Kayıt okunamadı: {exc}",
            "items": [],
        }
    if fp != run.get("fingerprint"):
        problems.append("parmak izi tutmuyor (strateji/veri/dönem/maliyet/sürüm değişmiş)")
    if run.get("engine_version") != ee.ENGINE_VERSION:
        problems.append(
            f"eski motor sürümü ({run.get('engine_version')} ≠ {ee.ENGINE_VERSION}); yeniden koşun"
        )
    dates = set(_DATE_RE.findall(text or ""))
    want = {str(run["period_start"])[:10], str(run["period_end"])[:10]}
    if not want <= dates:
        problems.append(
            f"metin koşunun dönemini belirtmiyor (beklenen {sorted(want)}, metinde {sorted(dates)})"
        )
    extra = dates - want - {str(run.get("data_report", {}).get("start", ""))[:10]}
    extra -= {str(run.get("data_report", {}).get("end", ""))[:10]}
    if extra:
        problems.append(f"metinde koşuyla uyuşmayan tarih(ler): {sorted(extra)}")
    if run.get("stage") == "gelistirme" and _OOS_RE.search(text or ""):
        problems.append("geliştirme dönemi sonucu örneklem dışı diye sunulmuş")
    costs = spec.costs
    any_cost = any(
        float(getattr(costs, f) or 0) > 0
        for f in (
            "commission_bps_per_side",
            "slippage_bps_per_side",
            "spread_bps",
            "funding_bps_per_day",
        )
    )
    if any_cost and _COSTLESS_RE.search(text or ""):
        problems.append("maliyetli hesap 'maliyetsiz' diye sunulmuş")
    metrics = (run.get("result") or {}).get("metrics") or {}
    claims = extract_metric_claims(text)
    items = []
    for c in claims:
        ok = _close(c["value"], metrics.get(c["metric"]), c["decimals"])
        items.append({**c, "recorded": metrics.get(c["metric"]), "ok": ok})
        if not ok:
            problems.append(f"{c['raw']} ≠ kayıtlı {metrics.get(c['metric'])}")
    if not claims:
        problems.append("metinde karşılaştırılabilir metrik sayısı yok")
    n_ok = sum(1 for i in items if i["ok"])
    if problems:
        return {
            "status": MISMATCHED,
            "scope": f"{n_ok}/{len(items)} metrik",
            "detail": "Kayıtlı hesapla EŞLEŞMEDİ: " + "; ".join(problems[:5]),
            "items": items,
        }
    return {
        "status": MATCHED,
        "scope": f"{n_ok}/{len(items)} metrik + dönem + parmak izi",
        "detail": f"Kayıtlı hesapla eşleşti (koşu {run['run_id']}, {run['stage']}, motor "
        f"{run['engine_version']}). Bu, stratejinin gelecekte başarılı olacağı ya da modelin "
        "daha iyi trader olduğu anlamına GELMEZ.",
        "items": items,
    }
