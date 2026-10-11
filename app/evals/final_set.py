"""Final test takımı — kilitli, görülmemiş, LLM-30'dan AYRI (protokol §10, denetim 2026-10-11 E2).

Neden: LLM-30 bir GELİŞTİRME setidir (okunur, teşhiste kullanılır); terfi kararı onunla
verilemez. Final takımı ≥60 soru, üç hedef alan (``CATEGORIES``), soru içeriği KULLANICIDAN
gelir. Bu modül yalnız biçimi, doğrulamayı, kilidi ve deterministik sayısal anahtarları sağlar;
soru yazmaz.

Dosyalar (``evals/final/`` — TÜMÜ ``.gitignore``'da; v15 deseni: final anahtarları açık depoya
girmez, yerelde kalır):

- ``questions.jsonl`` — ``FinalQuestion`` satırları (soru + aile + kategori [+ dondurulmuş kanıt])
- ``keys.jsonl``      — ``FinalKey`` satırları (rubrik noktaları, kritik kontroller, sayısal
  anahtar SPESİFİKASYONU — değer elle yazılmaz, ``compute_numeric`` hesaplar)
- ``manifest.json``   — ``lock_final`` yazar: iki dosyanın sha256'sı + sayımlar (üzerine yazılmaz)

Erişim kuralları:
- ``purpose="selection"`` YASAK (final seti seçimde kullanılamaz; ``GoldenAccessError``).
- ``purpose="final_comparison"`` kilit + hash doğrulaması ister; kullanım ``candidate_compare``
  final erişim günlüğüne yazılır (ikinci kullanım geliştirme sayılır).
- ``purpose="leakage_check"`` kilitsiz taslağı da okur (taslak sorular da eğitime GİREMEZ);
  dosya var ama manifest ile uyuşmuyorsa HATA (sessiz değişiklik = geçersiz kapı).

Eğitim tarafı bu modülü kullanmaz; sızıntı kapısı (``mix_cli.leakage_eval_items``) ve öz-damıtma
soru filtresi soruları + anahtar metinlerini tarar. RAG indeksi ``evals/`` dizinini okumaz.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.evals.profile.dataset_loader import DatasetIntegrityError, GoldenAccessError
from app.lora.mix_common import repo_root

MIN_TOTAL = 60
MIN_FAMILIES_PER_CATEGORY = 5

#: Kullanıcının üç hedef alanı → alt kategoriler + asgari soru sayısı + kör paket boyutu.
CATEGORIES: dict[str, dict[str, Any]] = {
    "bicim_disiplin": {
        "title": "Biçim / disiplin",
        "min": 20,
        "subcategories": (
            "kural1_dili",  # tavsiye/kesinlik isteyen soruya hipotez + test noktası
            "maliyet_oos_uyarisi",  # maliyetsiz / OOS'suz "başarı" iddiasına doğru tepki
            "kaynakli_uzun_cevap",  # dondurulmuş kanıtla uzun, atıflı cevap
            "cekimserlik",  # kaynak/veri yetersiz → neyin eksik olduğunu söyle
        ),
    },
    "hesap": {
        "title": "Hesap doğruluğu",
        "min": 20,
        "subcategories": ("sharpe", "drawdown", "net_getiri_maliyet", "diger_metrik"),
    },
    "alan_bilgisi": {
        "title": "Alan bilgisi",
        "min": 20,
        "subcategories": (
            "backtest_metodoloji",
            "risk_yonetimi",
            "piyasa_yapisi_maliyet",
            "istatistik_zaman_serisi",
            "strateji_indikator",
        ),
    },
}

#: Final takımına özgü kritik hatalar (LLM-30 listesine EK; rubrik kapısı ikisini birleştirir).
FINAL_CRITICAL_ERRORS: dict[str, str] = {
    "kural1_tavsiye_dili": "Al/sat tavsiyesi, garanti/kesinlik dili (Kural 1)",
    "maliyet_yok_sayildi": "Komisyon/slippage'ı yok sayıp net sonuç gibi sunmak (Kural 3)",
    "oos_olmadan_basari": "Örneklem-dışı test olmadan 'başarılı/çalışıyor' demek (Kural 2)",
    "look_ahead": "Gelecek bilgisini kullanan kural/hesap önermek (Kural 4)",
    "uydurma_kaynak": "Kaynakta olmayan bilgiyi kaynağa atfetmek / uydurma atıf (Kural 7)",
    "anahtara_aykiri_hesap": "Deterministik anahtara aykırı sayısal sonuç",
}

Calc = Literal[
    "sharpe",
    "sortino",
    "annualized_volatility",
    "max_drawdown",
    "net_return_after_costs",
    "cagr",
    "profit_factor",
    "expectancy",
]

#: Hesap türü → zorunlu girdi alanları (eksik/fazla alan = HATA; sessiz varsayılan YOK).
CALC_INPUTS: dict[str, tuple[str, ...]] = {
    "sharpe": ("returns", "rf_per_period", "periods_per_year", "ddof"),
    "sortino": ("returns", "target_per_period", "periods_per_year"),
    "annualized_volatility": ("returns", "periods_per_year", "ddof"),
    "max_drawdown": ("equity",),
    "net_return_after_costs": (
        "gross_trade_returns",
        "commission_per_side",
        "slippage_per_side",
        "compounding",
    ),
    "cagr": ("start_value", "end_value", "years"),
    "profit_factor": ("trade_pnls",),
    "expectancy": ("win_rate", "avg_win", "avg_loss", "cost_per_trade"),
}

#: Sözleşmeler (soru metni bunlarla UYUMLU yazılmalı; anahtar yalnız bu tanımla hesaplanır).
CALC_CONVENTIONS: dict[str, str] = {
    "sharpe": "mean(r−rf)/std(r−rf, ddof)·√periods_per_year; r ondalık getiri (0.01 = %1)",
    "sortino": "mean(r−T)·√P / √(mean(min(r−T,0)²)) (aşağı yön sapması TÜM gözlemlerle)",
    "annualized_volatility": "std(r, ddof)·√P, yüzde puanı (20.0 = %20)",
    "max_drawdown": "max_t (1 − equity_t / max_{s≤t} equity_s), POZİTİF yüzde puanı",
    "net_return_after_costs": (
        "işlem başına net = brüt − 2·(komisyon+slippage) (gidiş-dönüş, toplamsal); "
        "compounding=true → Π(1+net)−1, false → Σnet; yüzde puanı"
    ),
    "cagr": "(end/start)^(1/years) − 1, yüzde puanı",
    "profit_factor": "Σ(pozitif pnl) / |Σ(negatif pnl)|",
    "expectancy": "win_rate·avg_win − (1−win_rate)·avg_loss − cost_per_trade (avg_loss pozitif)",
}

FINAL_PURPOSES: frozenset[str] = frozenset({"final_comparison", "audit", "leakage_check"})


class NumericKey(BaseModel):
    """Sayısal anahtarın SPESİFİKASYONU — değer ``compute_numeric`` ile hesaplanır."""

    model_config = ConfigDict(extra="forbid")

    calc: Calc
    inputs: dict[str, Any]
    tolerance: float = Field(gt=0)
    unit: str = ""
    #: İnsanın elle bulduğu değer (opsiyonel çapraz kontrol): hesapla uyuşmazsa set KİLİTLENMEZ.
    human_value: float | None = None
    #: Yalnız işaret farkı (ör. drawdown "−25" vs "25") hata sayılmasın mı?
    sign_insensitive: bool = False

    @model_validator(mode="after")
    def _inputs_exact(self) -> NumericKey:
        need = set(CALC_INPUTS[self.calc])
        have = set(self.inputs)
        if have != need:
            raise ValueError(
                f"{self.calc}: girdiler tam olarak {sorted(need)} olmalı "
                f"(eksik {sorted(need - have)}, fazla {sorted(have - need)})"
            )
        return self


class FinalQuestion(BaseModel):
    """``questions.jsonl`` satırı. Soru metni KULLANICIDAN gelir (``written_by``)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    subcategory: str
    family: str
    question: str
    #: Dondurulmuş kaynak metni (kaynaklı uzun cevap sorularında zorunlu). Modele sorudan önce
    #: verilir; RAG koşusundan bağımsızdır → sonuç retrieval değişiminden etkilenmez.
    evidence: list[str] = Field(default_factory=list)
    written_by: str
    source_provenance: str = ""
    human_verified: bool = False

    @field_validator("id")
    @classmethod
    def _id_prefix(cls, v: str) -> str:
        if not v.startswith("final-") or v != v.strip():
            raise ValueError("kimlik 'final-' ile başlamalı (LLM-30 'llm30-' ile karışmasın)")
        return v

    @model_validator(mode="after")
    def _category_ok(self) -> FinalQuestion:
        cat = CATEGORIES.get(self.category)
        if cat is None:
            raise ValueError(f"kategori {self.category!r} geçersiz; geçerli: {sorted(CATEGORIES)}")
        if self.subcategory not in cat["subcategories"]:
            raise ValueError(
                f"{self.category}: alt kategori {self.subcategory!r} geçersiz; "
                f"geçerli: {list(cat['subcategories'])}"
            )
        for field in ("family", "question", "written_by"):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} boş olamaz")
        if self.subcategory == "kaynakli_uzun_cevap" and not any(e.strip() for e in self.evidence):
            raise ValueError("kaynakli_uzun_cevap sorusu dondurulmuş 'evidence' taşımalı")
        return self


class FinalKey(BaseModel):
    """``keys.jsonl`` satırı (YEREL — açık depoya girmez)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    #: Doğru cevapta bulunması gerekenler (inceleyiciye rubrik maddesi olarak gösterilir).
    rubric_points: list[str] = Field(default_factory=list)
    #: Bu soruda özellikle denetlenecek kritik hata kodları (``all_critical_codes``).
    critical_checks: list[str] = Field(default_factory=list)
    numeric: NumericKey | None = None
    reference_answer: str = ""
    verified_by: str = ""


class FinalSet(BaseModel):
    questions: list[FinalQuestion]
    keys: dict[str, FinalKey]
    locked: bool = False
    lock_sha: str = ""


# ── deterministik sayısal anahtarlar ─────────────────────────────────────────


def _arr(values: Any, name: str) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name}: boş olmayan, sonlu sayı listesi olmalı")
    return a


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name}: sonlu sayı olmalı")
    return float(value)


def compute_numeric(calc: str, inputs: dict[str, Any]) -> float:
    """Anahtar değeri ``CALC_CONVENTIONS`` tanımıyla hesapla (elle yazılmış sabit DEĞİL)."""
    if calc in ("sharpe", "annualized_volatility", "sortino"):
        r = _arr(inputs["returns"], "returns")
        p = _num(inputs["periods_per_year"], "periods_per_year")
        if p <= 0:
            raise ValueError("periods_per_year > 0 olmalı")
        if calc == "sortino":
            ex = r - _num(inputs["target_per_period"], "target_per_period")
            downside = math.sqrt(float(np.mean(np.minimum(ex, 0.0) ** 2)))
            if downside == 0:
                raise ValueError("sortino: aşağı yön sapması sıfır (tanımsız)")
            return float(ex.mean()) / downside * math.sqrt(p)
        ddof = int(_num(inputs["ddof"], "ddof"))
        if ddof not in (0, 1) or r.size <= ddof:
            raise ValueError("ddof 0 ya da 1 olmalı ve gözlem sayısından küçük olmalı")
        if calc == "annualized_volatility":
            return float(r.std(ddof=ddof)) * math.sqrt(p) * 100.0
        ex = r - _num(inputs["rf_per_period"], "rf_per_period")
        sd = float(ex.std(ddof=ddof))
        if sd == 0:
            raise ValueError("sharpe: standart sapma sıfır (tanımsız)")
        return float(ex.mean()) / sd * math.sqrt(p)
    if calc == "max_drawdown":
        eq = _arr(inputs["equity"], "equity")
        if (eq <= 0).any():
            raise ValueError("equity pozitif olmalı")
        peak = np.maximum.accumulate(eq)
        return float(np.max(1.0 - eq / peak)) * 100.0
    if calc == "net_return_after_costs":
        g = _arr(inputs["gross_trade_returns"], "gross_trade_returns")
        cost = _num(inputs["commission_per_side"], "commission_per_side") + _num(
            inputs["slippage_per_side"], "slippage_per_side"
        )
        if cost < 0:
            raise ValueError("maliyet negatif olamaz")
        net = g - 2.0 * cost
        compounding = inputs["compounding"]
        if not isinstance(compounding, bool):
            raise ValueError("compounding true/false olmalı")
        total = float(np.prod(1.0 + net) - 1.0) if compounding else float(net.sum())
        return total * 100.0
    if calc == "cagr":
        start, end = (
            _num(inputs["start_value"], "start_value"),
            _num(inputs["end_value"], "end_value"),
        )
        years = _num(inputs["years"], "years")
        if start <= 0 or end <= 0 or years <= 0:
            raise ValueError("cagr: start, end, years > 0 olmalı")
        return ((end / start) ** (1.0 / years) - 1.0) * 100.0
    if calc == "profit_factor":
        pnl = _arr(inputs["trade_pnls"], "trade_pnls")
        loss = float(-pnl[pnl < 0].sum())
        if loss == 0:
            raise ValueError("profit_factor: zararlı işlem yok (tanımsız)")
        return float(pnl[pnl > 0].sum()) / loss
    if calc == "expectancy":
        w = _num(inputs["win_rate"], "win_rate")
        if not 0 <= w <= 1:
            raise ValueError("win_rate [0,1] olmalı")
        loss = _num(inputs["avg_loss"], "avg_loss")
        if loss < 0:
            raise ValueError("avg_loss pozitif büyüklük olarak verilmeli")
        return (
            w * _num(inputs["avg_win"], "avg_win")
            - (1 - w) * loss
            - _num(inputs["cost_per_trade"], "cost_per_trade")
        )
    raise ValueError(f"bilinmeyen hesap: {calc}")


def numeric_value(key: NumericKey) -> float:
    return compute_numeric(key.calc, key.inputs)


# ── dosyalar ─────────────────────────────────────────────────────────────────


def final_root() -> Path:
    return repo_root() / "evals" / "final"


def _paths(root: Path | None) -> tuple[Path, Path, Path]:
    base = root or final_root()
    return base / "questions.jsonl", base / "keys.jsonl", base / "manifest.json"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(f"{path.name}:{n} JSON hatası: {exc}") from exc
        if not isinstance(obj, dict):
            raise DatasetIntegrityError(f"{path.name}:{n} nesne olmalı")
        rows.append(obj)
    return rows


def all_critical_codes() -> dict[str, str]:
    from app.evals.llm30 import CRITICAL_ERRORS

    return {**CRITICAL_ERRORS, **FINAL_CRITICAL_ERRORS}


def load_final(*, purpose: str, root: Path | None = None) -> FinalSet | None:
    """Final takımını yükle. Dosya yoksa ``None`` (henüz yazılmadı).

    ``final_comparison`` kilit ister; ``leakage_check``/``audit`` taslağı da okur. Manifest varsa
    her amaçta hash doğrulanır (kilitten sonra değişen set sessizce kullanılmaz).
    """
    if purpose not in FINAL_PURPOSES:
        if purpose == "selection":
            raise GoldenAccessError(
                "final test takımı model/ağırlık/checkpoint SEÇİMİNDE kullanılamaz — LLM-30 "
                "geliştirme setini kullan"
            )
        raise ValueError(f"bilinmeyen amaç: {purpose} (geçerli: {sorted(FINAL_PURPOSES)})")
    qpath, kpath, mpath = _paths(root)
    if not qpath.exists() and not kpath.exists():
        if mpath.exists():
            raise DatasetIntegrityError("final manifest var ama soru/anahtar dosyaları yok")
        return None
    if not qpath.exists():
        raise DatasetIntegrityError(f"anahtar var ama soru dosyası yok: {qpath}")
    lock_sha = ""
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        files = manifest.get("files") or {}
        for path in (qpath, kpath):
            expected = (files.get(path.name) or {}).get("sha256")
            if not path.exists() or expected != _sha_file(path):
                raise DatasetIntegrityError(
                    f"final {path.name} kilitli manifest ile uyuşmuyor — kilitten sonra "
                    "değişiklik final kanıtını geçersiz kılar"
                )
        lock_sha = str(manifest.get("lock_sha") or "")
    elif purpose == "final_comparison":
        raise DatasetIntegrityError("final takımı kilitlenmemiş (hektor final-set lock)")
    questions: list[FinalQuestion] = []
    seen: set[str] = set()
    for n, row in enumerate(_read_jsonl(qpath), 1):
        try:
            q = FinalQuestion.model_validate(row)
        except ValueError as exc:
            raise DatasetIntegrityError(f"questions.jsonl:{n} şema hatası: {exc}") from exc
        if q.id in seen:
            raise DatasetIntegrityError(f"questions.jsonl:{n} yinelenen id: {q.id}")
        seen.add(q.id)
        questions.append(q)
    keys: dict[str, FinalKey] = {}
    if kpath.exists():
        for n, row in enumerate(_read_jsonl(kpath), 1):
            try:
                k = FinalKey.model_validate(row)
            except ValueError as exc:
                raise DatasetIntegrityError(f"keys.jsonl:{n} şema hatası: {exc}") from exc
            if k.id in keys:
                raise DatasetIntegrityError(f"keys.jsonl:{n} yinelenen id: {k.id}")
            keys[k.id] = k
    return FinalSet(questions=questions, keys=keys, locked=bool(lock_sha), lock_sha=lock_sha)


# ── doğrulama + kilit ────────────────────────────────────────────────────────


def validate_final(fs: FinalSet, *, check_separation: bool = True) -> list[str]:
    """Kilit öncesi TÜM şartlar. Boş liste = kilitlenebilir."""
    errors: list[str] = []
    qs = fs.questions
    if len(qs) < MIN_TOTAL:
        errors.append(f"soru sayısı {len(qs)} < {MIN_TOTAL}")
    for cat, spec in CATEGORIES.items():
        rows = [q for q in qs if q.category == cat]
        if len(rows) < spec["min"]:
            errors.append(f"{cat}: {len(rows)} soru < {spec['min']}")
        fams = {q.family for q in rows}
        if len(fams) < MIN_FAMILIES_PER_CATEGORY:
            errors.append(f"{cat}: {len(fams)} aile < {MIN_FAMILIES_PER_CATEGORY}")
    owner: dict[str, str] = {}
    for q in qs:
        if q.family in owner and owner[q.family] != q.category:
            errors.append(f"aile {q.family} iki kategoride ({owner[q.family]}, {q.category})")
        owner.setdefault(q.family, q.category)
    texts: dict[str, str] = {}
    for q in qs:
        norm = " ".join(q.question.casefold().split())
        if norm in texts:
            errors.append(f"{q.id}: soru metni {texts[norm]} ile aynı")
        texts[norm] = q.id
        if not q.human_verified:
            errors.append(f"{q.id}: insan doğrulaması yok (human_verified=false)")
    ids = {q.id for q in qs}
    codes = all_critical_codes()
    for orphan in sorted(set(fs.keys) - ids):
        errors.append(f"{orphan}: anahtarı var ama sorusu yok")
    for q in qs:
        key = fs.keys.get(q.id)
        if key is None:
            errors.append(f"{q.id}: anahtar yok")
            continue
        if not key.verified_by.strip():
            errors.append(f"{q.id}: anahtarı doğrulayan yok (verified_by)")
        if not key.rubric_points and key.numeric is None:
            errors.append(f"{q.id}: rubrik noktası ya da sayısal anahtar gerekli")
        for c in key.critical_checks:
            if c not in codes:
                errors.append(f"{q.id}: bilinmeyen kritik hata kodu {c}")
        if q.category == "hesap" and key.numeric is None:
            errors.append(f"{q.id}: hesap sorusu deterministik sayısal anahtar taşımalı")
        if key.numeric is not None:
            try:
                value = numeric_value(key.numeric)
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f"{q.id}: anahtar hesaplanamadı: {exc}")
                continue
            hv = key.numeric.human_value
            if hv is not None and abs(hv - value) > key.numeric.tolerance:
                errors.append(
                    f"{q.id}: insan değeri {hv} hesaplanan {value:.6g} ile uyuşmuyor "
                    f"(tolerans {key.numeric.tolerance})"
                )
    if check_separation:
        errors.extend(separation_errors(fs))
    return errors


def separation_errors(fs: FinalSet) -> list[str]:
    """Final soruları LLM-30 / profil / adapter eval setleriyle çakışmamalı (sızıntı kapısıyla
    AYNI ölçüt: tam, normalize, içerme, yakın kopya)."""
    from app.evals.profile.leakage import check_leakage
    from app.lora.mix_cli import leakage_eval_items

    others = leakage_eval_items(include_final=False)
    rows = [{"messages": [{"role": "user", "content": q.question}]} for q in fs.questions]
    rep = check_leakage(others, rows)
    return [
        f"{fs.questions[h.train_index].id}: {h.eval_id} ile çakışma ({h.kind}) — final seti "
        "geliştirme setlerinden AYRI olmalı"
        for h in rep.hits
    ]


def lock_final(root: Path | None = None) -> dict[str, Any]:
    """Doğrulanmış seti kilitle (manifest bir kez yazılır; üzerine yazılmaz)."""
    qpath, kpath, mpath = _paths(root)
    if mpath.exists():
        raise DatasetIntegrityError(
            "final takımı zaten kilitli; değiştirmek yeni sürüm (yeni dizin) gerektirir"
        )
    fs = load_final(purpose="audit", root=root)
    if fs is None or not kpath.exists():
        raise DatasetIntegrityError("final soru ve anahtar dosyaları gerekli")
    errors = validate_final(fs)
    if errors:
        raise DatasetIntegrityError("kilitlenemez:\n- " + "\n- ".join(errors[:40]))
    files = {p.name: {"sha256": _sha_file(p)} for p in (qpath, kpath)}
    lock_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {
        "version": "1",
        "locked_at": datetime.now(UTC).isoformat(),
        "lock_sha": lock_sha,
        "files": files,
        "counts": counts(fs),
        "note": "Final takımı: seçimde kullanılmaz; her kullanım erişim günlüğüne yazılır.",
    }
    with mpath.open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return manifest


def counts(fs: FinalSet) -> dict[str, Any]:
    by_cat: dict[str, dict[str, int]] = {}
    for q in fs.questions:
        by_cat.setdefault(q.category, {}).setdefault(q.subcategory, 0)
        by_cat[q.category][q.subcategory] += 1
    return {
        "n_questions": len(fs.questions),
        "n_families": len({q.family for q in fs.questions}),
        "n_keys": len(fs.keys),
        "n_numeric": sum(1 for k in fs.keys.values() if k.numeric is not None),
        "by_category": by_cat,
    }


def final_status(root: Path | None = None) -> dict[str, Any]:
    """Salt-okuma durum (sızıntı raporuna ve CLI'ye): yok / taslak / kilitli / bozuk."""
    try:
        fs = load_final(purpose="audit", root=root)
    except DatasetIntegrityError as exc:
        return {"state": "bozuk", "error": str(exc)}
    if fs is None:
        return {"state": "yok"}
    return {"state": "kilitli" if fs.locked else "taslak", "lock_sha": fs.lock_sha, **counts(fs)}


# ── dışa aktarım ─────────────────────────────────────────────────────────────


def leakage_items(fs: FinalSet) -> list[Any]:
    """Sızıntı kapısı için ``EvalItem``'lar: soru + referans cevap + rubrik noktaları.

    Dondurulmuş kanıt (``evidence``) BİLEREK dahil edilmez: makale metnidir ve meşru eğitim
    verisinde (aynı makalenin parçaları) bulunabilir; soru ve anahtar ise bulunamaz.
    """
    from app.evals.profile.schema import EvalItem

    out = []
    for q in fs.questions:
        key = fs.keys.get(q.id)
        out.append(
            EvalItem(
                id=q.id,
                domain="trading",
                subdomain=q.subcategory,
                question=q.question,
                reference_answer=key.reference_answer if key else "",
                accepted_answers=list(key.rubric_points) if key else [],
                source_provenance="evals/final (yerel, kilitli final takımı)",
            )
        )
    return out


def to_compare_rows(fs: FinalSet) -> list[dict[str, Any]]:
    """``candidate_compare`` soru seti satırları (anahtar dahil — YEREL rapor dizinine yazılır).

    Hesap soruları ``type=math`` + hesaplanmış ``answer_key`` ile otomatik puanlanır; hepsi ayrıca
    rubrikle kör incelenir. ``final_lock_sha`` final rolünde kilit doğrulaması içindir.
    """
    if not fs.locked:
        raise DatasetIntegrityError("yalnız kilitli final takımı dışa aktarılır")
    rows: list[dict[str, Any]] = []
    for q in fs.questions:
        key = fs.keys[q.id]
        row: dict[str, Any] = {
            "id": q.id,
            "family": q.family,
            "question": q.question,
            "dimension": q.category,
            "subcategory": q.subcategory,
            "rubric": list(key.rubric_points),
            "critical_checks": list(key.critical_checks),
            "final_lock_sha": fs.lock_sha,
        }
        if q.evidence:
            row["evidence"] = list(q.evidence)
        if key.numeric is not None:
            row.update(
                type="math",
                answer_key=numeric_value(key.numeric),
                tolerance=key.numeric.tolerance,
                unit=key.numeric.unit,
                sign_insensitive=key.numeric.sign_insensitive,
            )
        rows.append(row)
    return rows


def export_compare_set(out: Path, root: Path | None = None) -> Path:
    fs = load_final(purpose="final_comparison", root=root)
    if fs is None:
        raise DatasetIntegrityError("final takımı yok")
    rows = to_compare_rows(fs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return out


def current_lock_sha(root: Path | None = None) -> str:
    _, _, mpath = _paths(root)
    if not mpath.exists():
        return ""
    return str(json.loads(mpath.read_text(encoding="utf-8")).get("lock_sha") or "")
