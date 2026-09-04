"""Domain sınıflandırma — anahtar kelime eşleşmesiyle deterministik.

LLM gerektirmez. Bir metni Türkçe ve İngilizce anahtar kelimelere göre
ilgili bilgi alanlarına eşler. Eğitim verisinin alan dengesini denetlemek
için kullanılır.
"""

from __future__ import annotations

from enum import StrEnum


class Domain(StrEnum):
    """Achilles eğitim alanları."""

    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    STATISTICS = "statistics"
    PHILOSOPHY = "philosophy"
    TRADING = "trading"
    CODING = "coding"
    AI_SYSTEM_DESIGN = "ai_system_design"
    RISK_MANAGEMENT = "risk_management"


# Her domain için küçük harf anahtar kelimeler (TR + EN).
DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.MATHEMATICS: [
        "calculus",
        "integral",
        "derivative",
        "formula",
        "matrix",
        "vector",
        "equation",
        "theorem",
        "türev",
        "integral",
        "formül",
        "matris",
        "denklem",
        "hesap",
        "fonksiyon",
        "olasılık",
        "probability",
        # Kademe-2 domain-verifier kör noktası (2026-06-22): stokastik süreç kartları.
        "markov",
        "stochastic",
        "stokastik",
    ],
    Domain.PHYSICS: [
        "force",
        "energy",
        "acceleration",
        "velocity",
        "momentum",
        "thermodynamics",
        "kuvvet",
        "enerji",
        "ivme",
        "hız",
        "hareket",
        "termodinamik",
        "birim",
        "kütle",
    ],
    Domain.STATISTICS: [
        "distribution",
        "regression",
        "correlation",
        "p-value",
        "p-değeri",
        "hypothesis",
        "variance",
        "standard deviation",
        "dağılım",
        "regresyon",
        "korelasyon",
        "hipotez",
        "varyans",
        "ortalama",
        "sample",
    ],
    Domain.PHILOSOPHY: [
        "epistemology",
        "causality",
        "uncertainty",
        "evidence",
        "epistemoloji",
        "nedensellik",
        "belirsizlik",
        "kanıt",
        "ontoloji",
        "argument",
        "akıl yürütme",
    ],
    Domain.TRADING: [
        "backtest",
        "strategy",
        "indicator",
        "market",
        "position",
        "candlestick",
        "sharpe",
        "portfolio",
        "option pricing",
        "option",
        "return",
        "alpha",
        "beta",
        "equity",
        "stock",
        "finance",
        "financial",
        "trading",
        "strateji",
        "indikatör",
        "piyasa",
        "pozisyon",
        "mum",
        "fiyat",
        "trend",
        "momentum",
        "portföy",
        "hisse",
        "getiri",
    ],
    Domain.CODING: [
        "python",
        "algorithm",
        "function",
        "code",
        "class",
        "loop",
        "algoritma",
        "fonksiyon",
        "kod",
        "sınıf",
        "döngü",
        "değişken",
    ],
    Domain.AI_SYSTEM_DESIGN: [
        "rag",
        "lora",
        "embedding",
        "model",
        "fine-tuning",
        "fine tuning",
        "transformer",
        "prompt",
        "retrieval",
        # Kademe-2 domain-verifier kör noktası (2026-06-22): olasılıksal-ML kartları
        # ("machine/deep learning", "neural network") hiçbir domain'e eşlenmiyordu.
        "machine learning",
        "deep learning",
        "neural network",
        "lstm",
        "makine öğrenmesi",
        "derin öğrenme",
        "sinir ağı",
        "gömme",
        "ince ayar",
        "vektör",
    ],
    Domain.RISK_MANAGEMENT: [
        "stop-loss",
        "stop loss",
        "drawdown",
        "leverage",
        "position size",
        "kelly",
        "kaldıraç",
        "pozisyon büyüklüğü",
        "risk",
        "zarar durdur",
        "sermaye",
    ],
}


def classify_domains(text: str) -> list[Domain]:
    """Metni anahtar kelime eşleşmesiyle ilgili domainlere sınıfla.

    Eşleşen tüm domainleri `Domain` enum sırasına göre döndürür.
    Eşleşme yoksa boş liste döner (zorla bir domain atanmaz).
    """
    if not text:
        return []
    from app.lora.safety_scanner import tr_fold

    # tr_fold: 'hız'/'olasılık' gibi 'ı'lı anahtarlar büyük harf metinde ('HIZ'.lower()=='hiz')
    # str.lower() ile kaçıyordu; bu fonksiyon gate_0/gate_3'te BLOKLAYICI olduğundan önemli.
    folded = tr_fold(text)
    matched: list[Domain] = []
    for domain in Domain:
        keywords = DOMAIN_KEYWORDS[domain]
        if any(tr_fold(keyword) in folded for keyword in keywords):
            matched.append(domain)
    return matched
