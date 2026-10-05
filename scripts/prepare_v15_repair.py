# ruff: noqa: E501
"""Çalıştırılmış küçük kod/oracle örnekleriyle ayrı v15 onarım verisini üret."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_v15_candidate import LESSONS, VALID  # noqa: E402

from app.evals.v15_protocol import common_system_prompt, sha256, verify_lock  # noqa: E402


def examples():
    """Referanslarda kesir/kapalı form; cevap kodlarında numpy/iteratif hesap."""
    for i in range(6):
        p = [100 + i, 103 + i, 101 + i, 105 + i]
        weights = [0, 0.5, -0.25, 0.25]
        rate = Fraction(1 + i, 10000)
        net = [0.0]
        for j in range(1, 4):
            net.append(
                float(
                    Fraction(str(weights[j - 1])) * Fraction(p[j] - p[j - 1], p[j - 1])
                    - rate * abs(Fraction(str(weights[j])) - Fraction(str(weights[j - 1])))
                )
            )
        yield (
            "lag_cost_code",
            i,
            f"p={p}; karar sonrası ağırlıklar={weights}; tek yön komisyon={float(rate)}. Bar getirisi için önceki ağırlığı kullan, aynı bardaki turnover maliyetini düş. Kod ve sonuç ver; çalıştırmadığın piyasa performansı iddiası ekleme.",
            f"import numpy as np\np=np.array({p},dtype=float)\nw=np.array({weights})\nr=np.r_[0,np.diff(p)/p[:-1]]\nlag=np.r_[0,w[:-1]]\nnet=lag*r-{float(rate)}*np.abs(np.diff(w,prepend=0))\nprint(net.tolist())",
            net,
            "Karar kapanışta oluşur; yeni ağırlık sonraki bar getirisine uygulanır. Komisyon tek yön ağırlık değişiminden alınır. Sentetik hesap gerçek piyasa kazancı değildir.",
        )

        changes = [3 + i, -2, 4, -1 - i]
        gains = sum(max(v, 0) for v in changes)
        losses = sum(max(-v, 0) for v in changes)
        value = float(Fraction(100 * gains, gains + losses))
        yield (
            "rsi_vector_code",
            i,
            f"Değişimler={changes}; basit pencere RSI'ını pozitif kayıp büyüklüğüyle kodla ve hesapla. Wilder smoothing ile eşit olduğunu varsayma.",
            f"import numpy as np\nd=np.array({changes},dtype=float)\ng=np.maximum(d,0).mean()\nl=np.maximum(-d,0).mean()\nrsi=100*g/(g+l) if g+l>0 else float('nan')\nprint(float(rsi))",
            value,
            "Bu basit pencere yöntemidir; Wilder başlangıcı/smoothing farklıdır. G=L=0 için NaN politikası açık seçildi.",
        )

        prior, variance, observation, noise = 70 + i, 3 + i, 79 + i, 5
        gain = Fraction(variance, variance + noise)
        updated = Fraction(prior) + gain * (observation - prior)
        yield (
            "kalman_update_code",
            i,
            f"Local-level ölçüm güncellemesi: prior={prior}, Pprior={variance}, y={observation}, R={noise}. Pprior verilmiştir. K ve posterior için kod/sonuç üret.",
            f"p={prior}\nP={variance}\ny={observation}\nR={noise}\nK=P/(P+R)\npost=p+K*(y-p)\nprint([K,post])",
            [float(gain), float(updated)],
            "K boyutsuz, Pprior ve R fiyat biriminin karesidir. Pprior zaten verildiğinde Q ayrıca gerekli değildir.",
        )

        a, b, cost = 12 + i, 7 + i, 1
        threshold = Fraction(b + cost, a + b)
        yield (
            "payoff_code",
            i,
            f"Kazanç={a}, kayıp büyüklüğü={b}, iki sonucu da etkileyen toplam sabit maliyet={cost}. Başa baş p'yi türet ve kodla; p=0.5 için net beklentiyi hesapla.",
            f"a,b,c={a},{b},{cost}\np=(b+c)/(a+b)\nexpect=.5*a-.5*b-c\nprint([p,expect])",
            [float(threshold), float(Fraction(a - b, 2) - cost)],
            "E=p*a-(1-p)*b-c; p=(b+c)/(a+b). Yüzde50 doğruluk ekonomik fayda garantisi değildir.",
        )

        prices = [20 + i, 22 + i, 18 + i, 25 + i]
        alpha = Fraction(1, 4)
        ema = (1 - alpha) ** 3 * prices[0] + sum(
            alpha * (1 - alpha) ** (3 - j) * prices[j] for j in range(1, 4)
        )
        yield (
            "ema_recurrence_code",
            i,
            f"Fiyatlar={prices}, alpha=.25, ilk EMA ilk fiyattır. adjust=False reküransını kodla ve son EMA'yı hesapla.",
            f"p={prices}\ne=p[0]\nfor x in p[1:]:\n    e=.25*x+.75*e\nprint(e)",
            float(ema),
            "Sabit alpha, doğrusal ilk değerle doğrusal nedensel filtredir. Bu tanım adjust=True sonlu ağırlık normalizasyonuyla karıştırılmaz.",
        )

        high, low, previous = 40 + i, 35 + i, 44 + i
        tr = max(high - low, abs(high - previous), abs(low - previous))
        yield (
            "tr_code",
            i,
            f"H={high}, L={low}, önceki kapanış={previous}. TR'yi kodla ve birimini açıklayarak hesapla.",
            f"h,l,c={high},{low},{previous}\nprint(max(h-l,abs(h-c),abs(l-c)))",
            tr,
            "TR fiyat birimindedir; hacim verisi üretmez. Standart VWAP/OBV için gerçek hacim ayrıca gerekir.",
        )

        counts = [[3 + i, 2], [1, 4 + i]]
        matrix = [[float(Fraction(n, sum(row))) for n in row] for row in counts]
        yield (
            "transition_counts_code",
            i,
            f"İki durumlu geçiş sayımları={counts}. Satır bazlı MLE matrisi için kod ve sonuç üret; denominator'ı açıkla.",
            f"import numpy as np\nc=np.array({counts},dtype=float)\nP=c/c.sum(axis=1,keepdims=True)\nprint(P.tolist())",
            matrix,
            "Payda ilgili durumdan gözlenen çıkış sayısıdır; global toplam değildir. Burada her satırda çıkış vardır; tahmin belirsizliği ayrıca değerlendirilir.",
        )

        n, price, factor = 5 + i, 120, 2 + i
        yield (
            "wealth_split_code",
            i,
            f"{n} adet * {price} fiyatlı payda {factor}:1 split oluyor. Son adet, fiyat ve serveti kodla; ham fiyat getirisiyle serveti ayır.",
            f"n,p,s={n},{price},{factor}\nprint([n*s,p/s,(n*s)*(p/s)])",
            [n * factor, price / factor, n * price],
            "Splitin kendisi serveti değiştirmez. Ham fiyat değişimi 1/s-1'dir; toplam getiri ve temettü ayrı sözleşmedir.",
        )

        mean, sigma, k = 30 + i, 2 + i, 2
        yield (
            "band_numeric_code",
            i,
            f"M={mean}, std={sigma}, k={k} için standart Bollinger alt/üst bandını kodla.",
            f"m,s,k={mean},{sigma},{k}\nassert s>=0 and k>=0\nprint([m-k*s,m+k*s])",
            [mean - k * sigma, mean + k * sigma],
            "Üst-alt=2*k*std>=0; negatif std kabul edilmez. Diverjans band sırasının ters dönmesi değildir.",
        )

        returns = [Fraction(1 + i, 100), Fraction(-2, 100), Fraction(3, 100)]
        cumulative = math.prod(1 + r for r in returns) - 1
        yield (
            "compound_code",
            i,
            f"Basit getiriler={[float(r) for r in returns]}. Bileşik getiriyi kodla; aritmetik toplamla farkını açıkla.",
            f"import numpy as np\nr=np.array({[float(r) for r in returns]})\nprint(float(np.prod(1+r)-1))",
            float(cumulative),
            "Bileşik getiri prod(1+r)-1'dir; basit toplam çapraz terimleri ihmal eder. Bu örnek sentetiktir.",
        )

        changes_w = [Fraction(0), Fraction(1 + i, 10), Fraction(-2, 10)]
        turnover = sum(abs(changes_w[j] - changes_w[j - 1]) for j in range(1, 3))
        yield (
            "turnover_code",
            i,
            f"Ağırlık yolu={[float(v) for v in changes_w]}. Toplam mutlak ağırlık değişimini kodla; komisyon oranı .001 ise net varlık değerine maliyet oranını hesapla.",
            f"import numpy as np\nw=np.array({[float(v) for v in changes_w]})\nt=float(np.abs(np.diff(w)).sum())\nprint([t,t*.001])",
            [float(turnover), float(turnover / 1000)],
            "Turnover işlem sayısı değildir. Notional tanımı sabit NAV varsayımıdır; gerçek spread/slippage ayrıca modellenir.",
        )

        probability = Fraction(2 + i, 10)
        yield (
            "binary_score_code",
            i,
            f"Gerçek ikili etiket y=1, p={float(probability)}. Tek gözlem negatif log-likelihood ve Brier'ı kodla; bir gözlemden kalibrasyon sonucuna varma.",
            f"import math\np={float(probability)}\nprint([-math.log(p),(p-1)**2])",
            [-math.log(float(probability)), float((probability - 1) ** 2)],
            "Bu tek gözlem skorudur; düşük skor tek başına genel kalibrasyon veya kârlılık kanıtı değildir.",
        )


def close(actual, expected):
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(close(a, b) for a, b in zip(actual, expected, strict=True))
        )
    return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-suffix", default="repair_r1", choices=["repair_r1", "repair_r2", "repair_r3"]
    )
    args = parser.parse_args()
    if errors := verify_lock(ROOT / "reports/v15/protocol_v2/locks/lock.json"):
        raise ValueError(errors)
    data = ROOT / f"data/training/v15_{args.out_suffix}"
    report = ROOT / f"reports/v15/{args.out_suffix}"
    data.mkdir(parents=True, exist_ok=True)
    code_dir = report / "verified_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    if (data / "train.jsonl").exists():
        raise FileExistsError(data / "train.jsonl")
    rows, evidence = [], []
    system = common_system_prompt()
    for family, i, question, code, expected, explanation in examples():
        if family in {"turnover_code", "binary_score_code", "lag_cost_code"}:
            continue
        identifier = f"repair_{family}_{i:02}"
        path = code_dir / f"{identifier}.code.txt"
        path.write_text(code + "\n", encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, check=True, timeout=30
        )
        actual = json.loads(completed.stdout)
        if not close(actual, expected):
            raise ValueError((identifier, actual, expected))
        answer = f"{explanation}\n```python\n{code}\n```\nBeklenen sentetik çıktı: {completed.stdout.strip()}. Burada gerçek piyasa/backtest performansı sonucu verilmemiştir."
        rows.append(
            {
                "id": identifier,
                "question": question,
                "source_group": f"oracle_{family}",
                "template_family": family,
                "verification_status": "executed_code_numeric_reference_check",
                "curriculum_level": "applied",
                "domain": "quantitative_research",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
        evidence.append(
            {
                "id": identifier,
                "code_sha256": sha256(path),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "expected_independent_oracle": expected,
                "passed": True,
            }
        )
    for family, question, answer in LESSONS:
        rows.append(
            {
                "id": f"repair_explain_{family}",
                "question": question,
                "source_group": "pilot_reviewed_explanations",
                "template_family": f"explain_{family}",
                "verification_status": "agent_reviewed_explanation_not_executed_code",
                "curriculum_level": "foundational",
                "domain": "quantitative_research",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    valid = [
        {
            "id": family,
            "question": q,
            "source_group": "distinct_validation_concepts",
            "template_family": family,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        }
        for family, q, a in VALID
    ]
    for name, items in (("train", rows), ("valid", valid)):
        (data / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8"
        )
    (report / "code_oracle_evidence.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "executed": len(evidence),
                "passed": len(evidence),
                "evidence": evidence,
                "oracle_scope": "Fraction/closed-form references plus same-formula consistency checks",
                "excluded_final_overlap_families": [
                    "turnover_code",
                    "binary_score_code",
                    "lag_cost_code",
                ],
                "development_scope": "seen numeric concepts diagnostic, not independent generalization",
                "market_performance_measured": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"train": len(rows), "valid": len(valid), "executed_code": len(evidence)}))


if __name__ == "__main__":
    main()
