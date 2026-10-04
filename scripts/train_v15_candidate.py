# ruff: noqa: E501
"""Ayrı, sabit yerel base üzerinde küçük v15 araştırma pilotu; varsayılan dry-run."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evals.v15_protocol import (  # noqa: E402
    common_system_prompt,
    sha256,
    split_errors,
    verify_lock,
    write_lock,
)
from app.training.peft_lora_train import PeftTrainConfig, dry_run, train  # noqa: E402

SYSTEM_PROMPT = common_system_prompt()

LESSONS = [
    (
        "schema_identifiability",
        "Fiyat kapanışı gözlenen veriden intrabar uçların özdeşlenebilirliğini açıkla.",
        "Tek kapanış farklı intrabar yollarla üretilebilir; gerçek açılış/yüksek/düşük belirlenemez. Proxy ayrı varsayımdır. Doğrulama: aynı kapanışlı farklı yollar kur; veri görülmediyse incelendi deme.",
    ),
    (
        "validation_contract",
        "Sayısal veri kabul sözleşmesini tasarla, doğrulama aşamalarını ayır.",
        "Sütun varlığı, sayısal parse, NaN, sonluluk, sıralama ve timezone ayrı kontrollerdir. Nokta ve rakam regex'i sayı parser'ı değildir. OHLC için H>=max(O,C),L<=min(O,C),H>=L. Hatalı metin sessiz kabul edilmez.",
    ),
    (
        "time_contract",
        "Yerel saatten olay zamanına dönüşümde gerekli metadata ve DST hata politikası nedir?",
        "Naive zamanın kaynak timezone'u önce belirlenir. Lokalizasyon ve UTC dönüşümü ayrıdır. Ambiguous/nonexistent için açık hata veya kaynak politikası gerekir. Modern İstanbul UTC+3 kullanır; mevsimsel DST ekleme. Bu açıklama, çalıştırılmış veri denetimi değildir.",
    ),
    (
        "asof_revision",
        "Revize edilen piyasa verisinin geçmiş bilgi erişimi sözleşmesini tasarla.",
        "symbol+timestamp olay anahtarı olabilir. Yayın/revizyon erişim zamanı ayrı saklanır. As-of sorgu yalnız o karar anında yayımlanmış revizyonu görür. Günlük gerçekten değişen kayıtları belirtir; farklı sembollerin aynı zamanı mükerrer sayılmaz.",
    ),
    (
        "resample_contract",
        "OHLCV yeniden örneklemesinin veri sözleşmesini tarif et.",
        "open first,high max,low min,close last,volume sum. Timezone,label,closed ve bar availability kaydedilir. Gözlem sayısı size veya uygun sütun count ile hesaplanır. Negatif frekans kaydırması kapalı barı geçmişte erişilebilir yapmaz.",
    ),
    (
        "rsi_bounds_proof",
        "Pozitif ortalama kazanç G ve kayıp büyüklüğü L'den RSI sınırlarını türet.",
        "G,L>=0 için RSI=100*G/(G+L),dolayısıyla 0..100. L negatif imzalı kayıp değildir. G=L=0 için açık50/NaN politikası; L=0,G>0 için100. Wilder smoothing ile basit rolling yöntemini karıştırma; aynı başlangıç yöntemiyle referans test et.",
    ),
    (
        "band_order_proof",
        "Standart Bollinger bandı sırasını cebirle doğrula.",
        "Üst-alt=2*k*std>=0 (k>=0,std>=0). RSI diverjansı band sırası değil fiyat/momentum ilişkisidir. Bu cebirsel açıklama piyasa avantajı ölçümü değildir.",
    ),
    (
        "volume_identifiability",
        "Hacim olmayan veride hangi hacim ağırlıklı nicelikler özdeşlenemez?",
        "Standart OBV/VWAP için gerçek hacim gerekir. RSI/getiri kapanıştan; ATR OHLC ve önceki kapanıştan hesaplanabilir. TR=max(H-L,|H-Cprev|,|L-Cprev|). ATR hacim değildir; proxy gerçek hacimle ayrı karşılaştırılır. Quote tick her kaynakta trade değildir.",
    ),
    (
        "corporate_action_identity",
        "Splitin servet etkisini genel s katsayısıyla kanıtla.",
        "N adet fiyatP, s:1 split sonrası sN adet fiyatP/s: sN*P/s=NP. Ham fiyat getirisi1/s-1,split servet getirisi0. Son dönem referanslı geçmiş düzeltmesinde eski fiyatlar s'ye bölünür; temettü/toplam getiri ayrı tanımdır.",
    ),
    (
        "affine_shape_proof",
        "Standardizasyonun dağılım şekli üzerindeki etkisini açıkla.",
        "z=(x-mu)/sigma afin dönüşümdür,normal dağılım yaratmaz. Tek z gözlemi kendiliğinden parametre güven aralığı veya işlem avantajı değildir. Fit parametreleri yalnız trainingden dondurulur veya online geçmişle güncellenir.",
    ),
    (
        "linear_filter_proof",
        "Sabit alpha EMA'nın doğrusallığını ve gecikmesini açıklayan kısa türetim ver.",
        "E_t=alpha*x_t+(1-alpha)*E_{t-1}. Sabit alpha ve doğrusal ilk değer politikasıyla her çıktı girdilerin doğrusal birleşimidir. adjust boolean sonlu ağırlık normalizasyonudur,öğrenilen alpha değildir. Nedensellik sıfır gecikmeyi garanti etmez.",
    ),
    (
        "kalman_dimensional",
        "Skaler local-level Kalman modelinde kazancın boyutunu türet.",
        "F=H=1,bağımsız gürültüler altında Pprior=Pprev+Q,K=Pprior/(Pprior+R). P,Q,R gözlem biriminin karesi; K boyutsuz. posterior=prior+K*(y-prior). Q/(Q+R) ancak Pprev=0 gibi ek varsayımla. Veri verilmediyse ölçüm iddiası yok.",
    ),
    (
        "fourier_window",
        "Sonlu Fourier analizinin zaman erişim sınırını ve leakage'ı açıkla.",
        "Tam periyodiklik zorunlu değildir. DFT sonlu pencereyi periyodik uzatır;pencere sınırları spectral leakage yaratabilir. Online özellikte pencere yalnız erişilmiş gözlemlerden oluşur. Bir spektrum grafiği trading avantajı kanıtı değildir.",
    ),
    (
        "hurst_assumptions",
        "D=2-H ilişkisinin kapsamını açıkla.",
        "fBm grafik boyutunda ilgili varsayımlar altında D=2-H; genel her süreçte körlemesine uygulanmaz. fBm seviyeleri ile durağan artımlar ayrıdır. Seri bağımlılığı belirsizlik yönteminde korunur; bir Hurst tahmini rejim garantisi değildir.",
    ),
    (
        "adf_logic",
        "Hipotez testinde reddetmeme ile ispatı ayır; ADF ve eşbütünleşmeyi örnek ver.",
        "ADF H0 birim köktür; yüksek p reddetmeme,ne H0 ispatı ne durağanlık kanıtıdır. Eşbütünleşme seviyelerin uygun residual'ı ve Engle-Granger kritik değerleriyle sınanır; farkların durağanlığı seviyelerin cointegration kanıtı değildir.",
    ),
    (
        "entropy_limit_proof",
        "Ayrık entropide sıfır olasılığın katkısını limit üzerinden anlat.",
        "lim(p->0+) p*log(p)=0; sıfır katkısı0. Entropi belirsizlik ölçüsüdür,yön tahmini değildir. Düşük otokorelasyon tek başına mean-reversion mekanizmasını ispatlamaz.",
    ),
    (
        "markov_mle_derivation",
        "Multinomial satır olasılıklarının maksimum likelihood tahminini açıklayıp destek yokluğunu ayır.",
        "Pij=Nij/sum_j Nij; payda durumdan gözlenen çıkış sayısıdır,global toplam değil. Sıfır çıkışta MLE tanımsız; NaN/unsupported veya açık prior kullan. K unrestricted satırlarda K(K-1) serbest parametre vardır; yeterlilik çıkış frekansı ve belirsizlik hedefine bağlı.",
    ),
    (
        "probability_independence",
        "IID,Markov özelliği ve marjinal baseline arasındaki mantıksal ilişkileri ayır.",
        "IID özdeş dağılım ve bağımsızlık,olasılıkların eşitliği değil. Marjinal baseline eğitim koşulsuz frekansıdır,momentum veya zorunlu50/50 değil. İlk derece modelin baseline'ı yenmesi Markov özelliğini ispatlamaz; ikinci derece heldout uygun skorla karşılaştırılmalı.",
    ),
    (
        "hmm_conditioning",
        "HMM'de üç farklı koşullu olasılığın bilgi kümelerini tanımla.",
        "Emission P(x_t|z_t),filtering P(z_t|x_1:t),smoothing P(z_t|x_1:T). Smoothing gelecek kullanabilir. Emission Gaussian olmak zorunda değil; Gaussian farklı varyansları modelleyebilir. AIC/BIC toplam serbest parametreyi cezalandırır.",
    ),
    (
        "payoff_derivation",
        "İkili payoff ve sabit notional maliyette başa baş başarı olasılığını türet.",
        "Kazança,kayıp büyüklüğüb,toplam maliyetc: Enet=p*a-(1-p)*b-c; başa baş p=(b+c)/(a+b). p>.5 tek başına yeterli değil. Tahmin başarısı ile ekonomik fayda ayrı ölçülür; günlük yokken Sharpe/net getiri uydurulmaz.",
    ),
    (
        "cost_units",
        "Komisyon,funding,turnover ve işlem adedi birimlerini ayır.",
        "Tek yön komisyon referans notional*abs(delta_weight)*rate; gerekçesiz ikiyle çarpılmaz. Funding notional*süre*uygulanmış oranlarla hesaplanır; exposure portföy etkisini belirler. Turnover toplam mutlak ağırlık değişimi; işlem adedi değildir. Spread,slippage ayrı kalemdir.",
    ),
    (
        "causality_contract",
        "Gelecek perturbation deneyinin kontrol grubunu ve karşılaştırma kapsamını tarif et.",
        "t sonrası veri değiştir/sil; yalnız t'ye kadarki çıktıları karşılaştır. Fit trainingde freeze veya online geçmişle yapılmalı. Gelecek çıktının değişmesi leakage değil. shift0 kaydırmaz; shift1 full-sample scaler ya da centered window sızıntısını çözmez.",
    ),
    (
        "purge_contract",
        "Etiket erişim aralığıyla split sözleşmesini tanımla.",
        "Purge train etiket bilgi aralığının değerlendirme aralığıyla çakışmasını önler. Gap/embargo label ufku ve split yapısına bağlıdır; her etiket sonrası bütün işlemleri durdurma kuralı değildir. İleri hedef etiket olabilir,karar featurelarına sızmamalı.",
    ),
    (
        "reporting_contract",
        "Deney raporunda başarısız üretim,kurtarma ve test kanıtını ayır.",
        "İlk deneme puanlanır; tekrar ve token limitindeki eksik cevap başarı değil. İçerik rescue ayrı,HTTP altyapı retry ayrı kayıtlanır. Varsayım,çalıştırılmamış kod ve ölçülmüş sonuç ayrı etiketlenir; hedefi tutturmak için maliyet/getiri değiştirilmez.",
    ),
]

VALID = [
    (
        "valid_units",
        "Fiyat varyansı hangi birimdedir?",
        "Fiyat biriminin karesi; standart sapma fiyat biriminde. Boyutsuz oranla varyansı karıştırma.",
    ),
    (
        "valid_probabilities",
        "Olasılık vektörünün sayısal kabul koşulları nelerdir?",
        "Sonlu,her bileşen0..1,toplam toleransla1; bu koşullar kalibrasyon kanıtı değildir.",
    ),
    (
        "valid_availability",
        "Bir raporun event tarihi ile publication tarihi neden farklı önem taşır?",
        "Karar yalnız yayımlandığında erişilmiş bilgiye dayanır,event tarihine geriye taşıma sızıntı yaratabilir.",
    ),
    (
        "valid_metric",
        "Aynı sınavın alt maddelerini bağımsız bootstrap örneği saymak neyi bozar?",
        "Soru içi bağımlılığı yok sayarak belirsizliği küçültebilir;soru düzeyinde eşlenmiş örnekleme gerekir.",
    ),
]

# Bağımsız denetimin final ile yakın tanım şablonu saydığı örnekler eğitimden çıkarılır.
EXCLUDED_FAMILIES = {
    "hmm_conditioning",
    "causality_contract",
    "reporting_contract",
    "probability_independence",
    "entropy_limit_proof",
}
LESSONS = [lesson for lesson in LESSONS if lesson[0] not in EXCLUDED_FAMILIES]
FINAL_TARGET_CLAUSES = (
    " Eşbütünleşme seviyelerin uygun residual'ı ve Engle-Granger kritik değerleriyle sınanır; farkların durağanlığı seviyelerin cointegration kanıtı değildir.",
    " Sıfır çıkışta MLE tanımsız; NaN/unsupported veya açık prior kullan.",
    " İleri hedef etiket olabilir,karar featurelarına sızmamalı.",
    " Fit parametreleri yalnız trainingden dondurulur veya online geçmişle güncellenir.",
    "; günlük yokken Sharpe/net getiri uydurulmaz",
)
for clause in FINAL_TARGET_CLAUSES:
    LESSONS = [
        (family, question, answer.replace(clause, "")) for family, question, answer in LESSONS
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    report = ROOT / "reports/v15/pilot_v4"
    data = ROOT / "data/training/v15_pilot_v4"
    report.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "reports/v15/protocol_v2/locks/lock.json"
    if errors := verify_lock(lock):
        raise ValueError(errors)
    splits = {}
    for name, lessons in (("train", LESSONS), ("valid", VALID)):
        rows = []
        for family, question, answer in lessons:
            rows.append(
                {
                    "id": family,
                    "source_group": f"curated_{name}",
                    "template_family": family,
                    "question": question,
                    "verification_status": "verified_explanation_agent_review",
                    "example_type": "explanation_not_executed_code",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                }
            )
        splits[name] = rows
    for name in ("development", "final"):
        splits[name] = [
            json.loads(s)
            for s in (ROOT / f"evals/v15/protocol_v2/{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
    if errors := split_errors(splits):
        raise ValueError(errors)
    for name in ("train", "valid"):
        target = data / f"{name}.jsonl"
        if not target.exists():
            target.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in splits[name]),
                encoding="utf-8",
            )
    base = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
    )
    cfg = PeftTrainConfig(
        base_model=str(base),
        train_jsonl=data / "train.jsonl",
        valid_jsonl=data / "valid.jsonl",
        adapter_output_path=ROOT / "models/adapters/hektor_lora_v15_localrev_pilot",
        iterations=len(LESSONS),
        batch_size=1,
        learning_rate=0.0002,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        max_seq_length=6144,
        assistant_only_loss=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        neftune_noise_alpha=5,
        warmup_ratio=0.05,
        eval_every_examples=8,
        eval_max_examples=4,
        load_best_model_at_end=False,
        loss_weighting="token",
        seed=42,
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
    config.update(
        experiment="v15_localrev_pilot_not_identical_v14_base_proven",
        checkpoint_selection="last_checkpoint_fixed_in_advance_pilot_no_loss_selection",
        rationale="v14 r/alpha/dropout/attention/lr/GA/seq korunur; 19 kısa doğrulanmış açıklamalık pipeline pilotu, yeterli eğitim iddiası değil",
        rag="unchanged_off",
        authorization="user_explicit_training_and_separate_local_revision",
        started_at=datetime.now(UTC).isoformat(),
    )
    if not (report / "training_config.json").exists():
        (report / "training_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    dataset_lock = report / "locks/lock.json"
    if not dataset_lock.exists():
        write_lock(
            report / "locks",
            [data / "train.jsonl", data / "valid.jsonl", report / "training_config.json"],
        )
    if errors := verify_lock(dataset_lock):
        raise ValueError(errors)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HEKTOR_TRAIN_DTYPE"] = "bf16"
    if args.run:
        import numpy as np
        import torch

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
    result = train(cfg) if args.run else dry_run(cfg)
    target = report / ("training_result.json" if args.run else "dry_run.json")
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if args.run and result.get("ok"):
        artifact = cfg.adapter_output_path / "adapter_model.safetensors"
        (report / "adapter_hash.txt").write_text(sha256(artifact), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0 if not args.run or result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
