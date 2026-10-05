# ruff: noqa: E501
"""Yerel v15 denetim paketi. Model eğitmez; tarihsel cevapları doğruluk etiketi yapmaz."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evals.v15_protocol import (  # noqa: E402
    ACCEPTANCE,
    DECODING,
    sha256,
    split_errors,
    write_lock,
)

# Her satır bağımsız bir görev ailesidir; sayısal varyantlar farklı splitlere dağıtılmaz.
# Anahtarlar taslaktır: teknik fixture ile doğrulanmayan kavramlar bağımsız hakem bekler.
APPLIED = [
    (
        "ohlc",
        "O=54,H=58,L=52,C=53 mumunu doğrula; H=52 olursa hangi eşitsizlik bozulur?",
        "İlki geçerli düşüş mumu; ikincide H<max(O,C).",
    ),
    (
        "time",
        "İstanbul 20 Şubat 2026 16:45 zamanını UTC'ye çevir; naive zamanı nasıl ele alırsın?",
        "13:45 UTC; önce İstanbul'a lokalize et, sonra UTC'ye çevir.",
    ),
    (
        "duplicates",
        "A/10:00 kapanış=8, B/10:00=9, A/10:00 revizyon=10 kayıtlarında as-of ve mükerrer anahtarı tasarla.",
        "Anahtar symbol+timestamp; revizyon erişim zamanını ayrı tut, geçmişte sonradan gelen revizyonu kullanma.",
    ),
    (
        "resampling",
        "09:00-09:15 ve 09:15-09:30 OHLCV barlarını 30 dk bara birleştir; kapanış erişim zamanını belirt.",
        "open first,high max,low min,close last,volume sum; bar 09:30 kapanışından önce erişilebilir değildir; timezone/closed/label açık olmalı.",
    ),
    (
        "rsi",
        "Basit 4 dönem RSI için değişimler +6,-2,+3,-1. Kazanç/kayıp ortalaması ve RSI'ı hesapla.",
        "Kazanç 9/4, pozitif kayıp 3/4; RSI=75; Wilder ile aynı yöntem değildir.",
    ),
    (
        "bollinger",
        "Ortalama=80,std=4,k=1.5 için Bollinger alt/üst bandını hesapla.",
        "74 ve 86; std>=0,k>=0 altında alt<=üst.",
    ),
    (
        "tr",
        "H=63,L=57,önceki C=66 için TR hesapla; ölçü birimi nedir?",
        "max(6,3,9)=9 fiyat birimi; hacim değildir.",
    ),
    (
        "split",
        "3:1 split öncesi 12 adet*90; sonrası kaç adet/fiyat/değer? Geçmiş düzeltmesini belirt.",
        "36 adet*30=1080; son dönem referansıyla eski fiyatlar 3'e bölünür; ham fiyat -2/3, servet değişmez.",
    ),
    (
        "ema",
        "EMA alpha=.25,önceki değer=40,yeni fiyat=48. Yeni EMA ve sabit alpha doğrusallığını ver.",
        "42; sabit alpha ve doğrusal başlangıç politikasıyla doğrusal filtredir, gecikmesi olabilir.",
    ),
    (
        "kalman",
        "Önsel=50,y=56,P_prior=6,R=3 Kalman güncellemesini ve varyans birimini yaz.",
        "K=2/3,posterior=54; P,Q,R gözlem biriminin karesi.",
    ),
    (
        "entropy",
        "Ayrık olasılıklar [.25,.75,0] için bit cinsinden entropiyi hesapla; 0 log0 politikasını yaz.",
        "-.25log2(.25)-.75log2(.75)=.8112781245 bit; sıfır katkısı 0.",
    ),
    (
        "transition",
        "[2,0,2,1,2,0,1] dizisinin 3 durumlu geçiş matrisini sayımlarla kur.",
        "0:[0,.5,.5],1:[0,0,1],2:[2/3,1/3,0]; çıkış sayısıyla satır normalize.",
    ),
    (
        "freedom",
        "4 ve 7 durumlu unrestricted geçiş matrisinin serbest parametre sayısı nedir?",
        "K(K-1):12 ve42; sayılar örneklem yeterliliği garantisi vermez.",
    ),
    (
        "half_life",
        "lambda=.90 unutmanın yarı ömrünü hesapla; 5dk gözlemlerde süreyi belirt.",
        "ln(.5)/ln(.9)=6.5788 dönem,32.894dk; takvim ve gözlem sıklığı ayrıdır.",
    ),
    (
        "payoff",
        "Kazanç %2,kayıp %3,p=.65,maliyet %.2. Brüt/net beklenti ve başa baş p hesapla.",
        "Brüt .25%,net .05%; başa baş(3+.2)/(2+3)=.64.",
    ),
    (
        "turnover",
        "10000 notional portföyde ağırlık -.2 -> .3, tek yön komisyon .08%. Maliyeti hesapla.",
        "abs(delta)=.5,işlem notional5000,maliyet4; gerekçesiz ikiyle çarpılmaz.",
    ),
    (
        "funding",
        "32 dönem,4 funding/dönem,her biri .025% ödeniyorsa sabit notional basit toplam maliyet kaçtır?",
        "32*4*.00025=.032=%3.2; portföy etkisi exposure'a bağlı; bileşik değil basit toplam.",
    ),
    (
        "log_derivative",
        "f(x)=ln(x^2+1) için x=2'de türevi analitik ve h=1e-5 merkezi farkla sınayan kod tasarla.",
        "f'=2x/(x²+1)=.8; merkezi fark(f(2+h)-f(2-h))/(2h),yaklaşık .8; kod çalıştırılmadıysa öyle etiketle.",
    ),
    (
        "centered_terms",
        "x=[1,2,4,8,16] üç terimli merkezli ortalamanın iç noktalarını hesapla; online erişimi belirt.",
        "7/3,14/3,28/3; payda3; her merkez bir gelecek gözleme ihtiyaç duyar.",
    ),
    (
        "zscore",
        "Eğitim ortalaması=12,std=3; yeni x=18 için z ve güven aralığı ilişkisini yaz.",
        "z=2; tek gözlemin standardizasyonu, parametre güven aralığı veya avantaj değildir.",
    ),
    (
        "adf",
        "ADF p=.12,alpha=.05 sonucunu H0 adıyla yorumla; kesin durağanlık kararı verilebilir mi?",
        "H0 birim kök; reddedilemez. H0 doğru ispatlanmaz, durağanlık da kanıtlanmaz.",
    ),
    (
        "bic",
        "HMM logL=-140,n=500,toplam serbest parametre=11. AIC ve BIC hesapla.",
        "AIC=302;BIC=280+11ln500=348.360689; rejim sayısı değil toplam parametre.",
    ),
    (
        "hmm_bayes",
        "İki rejim önseli [.7,.3], bir gözlemin emission likelihoodları [.2,.8]. Filtrelenmiş posterior nedir?",
        "Normalize [.14,.24]:[7/19,12/19]; likelihood posterior değildir.",
    ),
    (
        "brier",
        "Üç ikili hedef [1,0,1],olasılıklar [.8,.4,.6]. Ortalama Brier hesapla.",
        "(.04+.16+.16)/3=.12; yön doğruluğundan farklı uygun skordur.",
    ),
    (
        "baseline",
        "Eğitimde 80 yükseliş,20 düşüş; test için marjinal baseline ve tek yükseliş log-loss'u nedir?",
        "p(up)=.8,p(down)=.2; yükselişte -ln(.8)=.22314355; test frekansıyla fit edilmez.",
    ),
    (
        "purge",
        "Train etiket aralıkları [1,4],[4,7],[7,10]; değerlendirme bilgi aralığı [6,9]. Hangilerini purge et?",
        "[4,7] ve[7,10] çakışır; [1,4] çakışmaz; uç noktaların kapalı/açık politikası belirtilmeli.",
    ),
    (
        "execution",
        "t barı kapanışında hesaplanan ağırlık .4; sonraki close-to-close getiri -.02. Sürtünmesiz portföy katkısı ve execution varsayımı?",
        "-.008=-.8%; t kapanışında aynı fiyatla fill varsayımı açık ve uygulanabilir olmalı, erişim/fill ayrılmalı.",
    ),
    (
        "fourier",
        "N=8,örnek aralığı .5 saniye sinyalinde DFT frekans çözünürlüğü ve Nyquist nedir?",
        "1/(N*dt)=.25Hz;Nyquist=1/(2dt)=1Hz; pencere/leakage ayrı.",
    ),
    (
        "hurst",
        "fBm varsayımları altında H=.35 için grafik fraktal boyutu ve artımların durağanlığı nedir?",
        "D=2-H=1.65; fBm artımları durağandır; süreç seviyeleri genel olarak durağan değil.",
    ),
    (
        "dirichlet",
        "3 kategori çıkış sayımı[2,1,0],Dirichlet(1,1,1) prior için posterior predictive nedir?",
        "[3,2,1]/6=[.5,1/3,1/6]; multinomial prior, üç bağımsız Beta değil.",
    ),
]

EDGE = [
    (
        "number_parser",
        "'1.2.3','NaN','inf','-2.5','1e3' girdilerinde ^[0-9.]+$ niçin güvenli değil?",
        "Hatalı noktaları kabul eder, geçerli işaret/üsleri reddeder; sayısal parse ve sonluluk ayrı kontrol edilir.",
    ),
    (
        "close_only",
        "Yalnız timestamp+close'dan bir mum grafiği üretildi. Aynı kapanışla iki farklı intrabar yolu verip OHLC belirlenebilirliğini çürüt.",
        "Aynı C için farklı O/H/L olabilir; gerçek OHLC belirlenemez, varsayımsal proxy ayrı etiketlenir.",
    ),
    (
        "dst_gap",
        "New York'ta 8 Mart 2026 02:30 yerel saat neden sorunlu? Sessizce bir saat kaydırmak neyi değiştirir?",
        "DST ileri geçişinde nonexistent; açık politika/hata gerekir; sessiz kaydırma gerçek olay zamanını değiştirir.",
    ),
    (
        "dst_fold",
        "New York 1 Kasım 2026 01:30 iki kez oluşursa tek naive timestamp olay sırasını belirler mi?",
        "Hayır, ambiguous; UTC offset/fold veya kaynak sıra bilgisi gerekir.",
    ),
    (
        "calendar",
        "Kripto serisindeki pazar boşluğunu borsa tatili gibi doldurmak rolling ortalamaya ne yapar?",
        "Kripto çoğunlukla sürekli; gerçek veri kaybı olabilir. Ffill sentetik gözlem/rolling ve EMA ağırlıkları yaratır.",
    ),
    (
        "volume_zero",
        "İki veri sağlayıcıda volume=0 aynı mı yorumlanmalı? Birinde volume alanı placeholder.",
        "Kaynak sözleşmesine bağlı; işlem yokluğu ile bilinmeyen hacmi ayır, placeholder gerçek hacim sayılmaz.",
    ),
    (
        "rsi_zero",
        "Düz fiyatlar için RSI formülü 0/0 verir. 'RSI daima 100' sonucunu değerlendir.",
        "Sıfır kazanç ve kayıpta politika gerekir (ör.50/NaN); yalnız kayıp0,kazanç>0 ise100; tutarlı referans.",
    ),
    (
        "rsi_divergence",
        "Fiyat yeni tepe yaparken RSI önceki tepeden düşük. Bollinger bantlarının ters döndüğü söylenebilir mi?",
        "Hayır, farklı kavramlar; momentum azalırken fiyat artabilir; k>=0,std>=0 band sırası korunur.",
    ),
    (
        "vwap_proxy",
        "Hacim yokken ATR ağırlıklı fiyat ortalamasını 'gerçek VWAP' diye adlandırmak doğru mu?",
        "Hayır ATR fiyat volatilitesi; standart VWAP hacim ister, proxy ayrı etiketlenip gerçek hacimle bağımsız sınanmalı.",
    ),
    (
        "tick_trade",
        "Sağlayıcı sadece en iyi bid değişince tick yayınlıyor. Tick sayısını işlem hacmi sayabilir misin?",
        "Hayır quote update trade değildir; kaynak tanımı ve gerçek trade verisi gerekir.",
    ),
    (
        "ema_adjust",
        "pandas ewm(adjust=True) bir öğrenilen adaptif alpha mıdır? İki başlangıç politikasını ayır.",
        "Hayır boolean ağırlık normalizasyonu; adjustFalse rekürsif ilk gözlem başlangıcı,True normalize sonlu ağırlıklar.",
    ),
    (
        "normal_shape",
        "İki tepeli bir dağılım z-score sonrası normal oldu deniyor. Karşı örnekle açıkla.",
        "Afin dönüşüm iki tepeli şekli korur; ortalama/std standardizasyonu normal dağılım yaratmaz.",
    ),
    (
        "kalman_q",
        "Q=1,R=2,önceki posterior P=9. K=Q/(Q+R)=1/3 iddiasını kontrol et.",
        "P_prior=9+1=10,K=10/12=5/6; Q/(Q+R) önceki P=0 gibi ek varsayımla.",
    ),
    (
        "dft_nonperiodic",
        "Tek seferlik darbe Fourier ile incelenemez çünkü periyodik değil iddiasına cevap ver.",
        "Fourier periyodiklik şartı koymaz; DFT sonlu pencereyi periyodik uzatır,leakage ve online pencere erişimi önemli.",
    ),
    (
        "adf_power",
        "Kısa seride ADF reddetmedi; yazar bunun kalıcı random walk ispatı olduğunu söylüyor.",
        "Reddetmeme ispat değildir; test gücü,lag/trend ve örneklem etkiler; H0 birim kök.",
    ),
    (
        "cointegration",
        "İki bağımsız random walk'un farkları durağan. Seviyeler cointegrated mı?",
        "Bu sonuç yeterli değil; seviyelerde uygun lineer birleşimin residual'ı EG kritik değerleriyle sınanır.",
    ),
    (
        "entropy_direction",
        "Yüksek entropili durum dizisi yarın düşüş garanti eder mi? Aynı entropili ters etiketli diziyi düşün.",
        "Hayır entropi yön bilgisi değil; etiket permütasyonu entropiyi korur, yön ters olabilir.",
    ),
    (
        "autocorr",
        "Lag1 otokorelasyon yaklaşık0 olan IID seri mean-reverting indikatör kanıtı mı?",
        "Hayır düşük otokorelasyon ortalamaya dönüş mekanizmasını veya ekonomik avantajı ispatlamaz.",
    ),
    (
        "unvisited",
        "Üç durumdan biri hiç çıkış yapmadı. O satırı üç sıfırla normalize etmek neden sorunlu?",
        "0/0; NaN/unsupported veya açık prior politikasını uygula; veri gözlenmiş olasılık diye sunma.",
    ),
    (
        "iid_bias",
        "IID yazı tura P(yazı)=.85 olabilir mi? IID olmayı .5/.5 ile ayır.",
        "Olabilir; bağımsız ve özdeş dağılım eşit olasılık şartı değildir.",
    ),
    (
        "markov_order",
        "Birinci derece model marjinali yendi. İkinci dereceyi denemeden Markov özelliğini ispatladım denebilir mi?",
        "Hayır; heldout logloss/Brier ve koşullu bağımlılık,belirsizlik karşılaştırması gerekir.",
    ),
    (
        "sample_size",
        "K=10 matris için 1000 toplam gözlem her satırın güvenilir olduğu anlamına gelir mi?",
        "Hayır 90 serbest parametre; çıkış frekansları/nadir durumlar/belirsizlik hedefine bağlı; evrensel eşik yok.",
    ),
    (
        "hmm_smoothing",
        "Tüm yıl üzerinde smoothing posterior'u günlük online sinyale kondu. Emission/filtering/smoothing'i ayır.",
        "P(x_t|z_t),P(z_t|x_1:t),P(z_t|x_1:T); smoothing gelecek gözlemleri içerir.",
    ),
    (
        "gaussian_variance",
        "Getiri-only Gaussian HMM iki rejimde aynı ortalama farklı variance öğrenemez iddiasını değerlendir.",
        "Öğrenebilir; emission variance parametreleri farklı olabilir; Gaussian tek seçenek değil.",
    ),
    (
        "shift_zero",
        "Bir feature shift(0) kullanıyor. Başka bilgi olmadan geleceği kullandığını ispatlar mı?",
        "Hayır shift0 değişiklik yapmaz; feature üretimi/fit ve erişilebilirlik zamanı incelenmeli.",
    ),
    (
        "shift_scaler",
        "Full-sample scaler ile normalize edip shift(1) yaptım; leakage çözüldü mü?",
        "Hayır scaler gelecekle fit edilmiştir; train-fit freeze veya online yalnız geçmiş gerekir.",
    ),
    (
        "perturb_scope",
        "Gelecek veriyi bozunca yalnız gelecekteki EMA değişti; bu leakage mi?",
        "Hayır yalnız t'ye kadarki çıktılar karşılaştırılır,gelecek çıktı değişimi beklenir.",
    ),
    (
        "target_legitimate",
        "İleri 3 günlük getiri eğitim etiketi. Bu tek başına leakage midir?",
        "Hayır etiket meşru; karar featurelarına veya çakışan yanlış splite sızması sorun,purge etiket aralıklarına göre.",
    ),
    (
        "unsupported_sharpe",
        "İşlem günlüğü verilmedi; aday model Sharpe=1.8 ölçtüm diyor. Nasıl raporlamalı?",
        "Ölçüm iddiası reddedilir; veri/yöntem/kayıt eksik,varsayımsal örnek ve çalıştırılmamış kod açık etiketlenir.",
    ),
    (
        "retry_honesty",
        "İlk cevap token limitinde döngüde; ikinci cevap farklı penalty ile doğru. İlk denemeye 2 puan verilir mi?",
        "Hayır ilk başarısızlık ayrı puanlanır; rescue ayrı,altyapı HTTP retry farklı sınıf; ayarlar saklanır.",
    ),
]

# Bağımsız yazım denetimi sonrası, eğitim/final koşusu başlamadan düzeltilen anahtarlar.
APPLIED[11] = (
    "transition",
    APPLIED[11][1],
    "Sayımlar [[0,1,1],[0,0,1],[2,1,0]]; matris [[0,.5,.5],[0,0,1],[2/3,1/3,0]]; çıkış sayısıyla satır normalize.",
)
EDGE[1] = (
    "close_only",
    EDGE[1][1],
    "Aynı kapanış10: yolA [8,12,7,10], yolB [11,15,9,10]; O/H/L farklı. Gerçek OHLC belirlenemez; proxy varsayım olarak etiketlenir.",
)
APPLIED[20] = (
    "bootstrap_mean",
    "IID gözlemler [1,2,3] için örneklem ortalaması ve bootstrap yeniden örnekleme [3,3,1] ortalamasını hesapla. Zaman serisine IID bootstrap uygulama koşulunu belirt.",
    "Örneklem ortalaması2; yeniden örneklem7/3; seri bağımlılığı varsa IID bootstrap yerine bağımlılığı koruyan yöntem gerekir.",
)
EDGE[21] = (
    "probability_density",
    "density=True histogramında yoğunluk3 ve bin genişliği.1 olan hücrenin olasılık kütlesi3 müdür?",
    "Hayır kütle density*width=.3; tüm binlerin integrali1, yoğunluk1'den büyük olabilir.",
)
APPLIED[9] = (
    "kalman",
    APPLIED[9][1],
    APPLIED[9][2] + " Skaler local-level F=H=1, bağımsız süreç/ölçüm gürültüsü varsayımı.",
)
EDGE[12] = ("kalman_q", EDGE[12][1], EDGE[12][2] + " Skaler F=H=1 ve bağımsız gürültü varsayımı.")
APPLIED[15] = (
    "turnover",
    APPLIED[15][1].replace("10000 notional portföyde", "NAV=10000 portföyde işaretli"),
    APPLIED[15][2],
)
EDGE[15] = (
    "cointegration",
    EDGE[15][1],
    EDGE[15][2]
    + " Bağımsız ve dejenere olmayan random-walk yenilikleri altında nontrivial durağan birleşim yoktur, cointegrated değiller.",
)
EDGE[20] = (
    "markov_order",
    EDGE[20][1],
    EDGE[20][2]
    + " Sonlu heldout karşılaştırması da Markov özelliğini ispatlamaz, göreli model yeterliliği kanıtıdır.",
)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    out = ROOT / "reports/v15/protocol_v2"
    evaluation = ROOT / "evals/v15/protocol_v2"
    if (out / "locks/lock.json").exists():
        raise SystemExit("Kilit zaten var; mevcut deney üzerine yazılmaz")
    splits: dict[str, list[dict]] = {"development": [], "final": []}
    for kind, catalog in (("applied", APPLIED), ("edge", EDGE)):
        for index, (family, question, reference) in enumerate(catalog):
            split = "development" if index < 15 else "final"
            row = {
                "id": f"v15-{kind}-{index + 1:02}",
                "kind": kind,
                "split": split,
                "source_group": f"authored-{family}",
                "template_family": family,
                "question": question,
                "reference_answer": reference,
                "rubric": [
                    {
                        "id": "core",
                        "max": 2,
                        "expected": reference,
                        "anchors": {
                            "2": "İstenen hesap/örnek ve çekirdek iddialar doğru ve yeterli",
                            "1": "Çekirdek kısmen doğru, istenen hesap/alt madde eksik",
                            "0": "Yanlış veya yanıtsız",
                        },
                    },
                    {
                        "id": "assumptions_evidence",
                        "max": 2,
                        "anchors": {
                            "2": "Gerekli varsayımlar ve kanıt durumu açık, çelişki yok",
                            "1": "Varsayım/test durumu eksik, kanıtsız ölçüm iddiası yok",
                            "0": "Çelişki, yanlış varsayım veya kanıtsız ölçüm",
                        },
                    },
                ],
                "numeric_tolerance": 0.0001 if kind == "applied" else None,
                "critical_flags": [
                    "yanlis_cekirdek",
                    "calismayan_kod",
                    "tekrar",
                    "kanitsiz_olcum",
                    "celiski",
                ],
                "key_status": "independent_agent_reviewed_not_human_verified",
                "human_verified": False,
            }
            splits[split].append(row)
    errors = split_errors(splits)
    if errors:
        raise SystemExit(errors)
    paths = []
    for split, rows in splits.items():
        path = evaluation / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        paths.append(path)
    dump(out / "acceptance.json", ACCEPTANCE)
    dump(out / "decoding.json", DECODING)
    dump(
        out / "split_manifest.json",
        {
            "counts": {name: len(rows) for name, rows in splits.items()},
            "kind_counts": {"applied": 30, "edge": 30},
            "group_overlap_errors": errors,
            "exact_dedup": "passed",
            "semantic_dedup": "independent_review_pending",
            "training": "not_prepared",
            "final_access": "authoring_only_not_model_evaluated",
            "review_note": "Farklı aile kimliği semantik ayrıklığı tek başına kanıtlamaz.",
        },
    )
    write_lock(out / "locks", [*paths, out / "acceptance.json", out / "decoding.json"])
    runtime = Path("C:/HP/hektor")
    inventory = {
        "time_utc": datetime.now(UTC).isoformat(),
        "runtime_root": str(runtime),
        "worktree_root": str(ROOT),
        "versions": {},
        "artifacts": {},
        "missing": [],
    }
    for package in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "tokenizers",
        "pytest",
        "numpy",
        "pandas",
    ):
        inventory["versions"][package] = importlib.metadata.version(package)
    for version in ("v13", "v14"):
        directory = runtime / "models/adapters" / f"hektor_lora_{version}_30b"
        inventory["artifacts"][version] = {}
        for name in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "run_plan.json",
            "run_complete.json",
            "README.md",
        ):
            p = directory / name
            if p.exists():
                inventory["artifacts"][version][name] = {"path": str(p), "sha256": sha256(p)}
            else:
                inventory["missing"].append(str(p))
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
    )
    inventory["cached_base"] = {
        "path": str(snapshot),
        "commit": snapshot.name,
        "v14_same_revision_proven": False,
        "files": {},
    }
    for p in snapshot.iterdir():
        if p.is_file() and p.suffix != ".safetensors":
            inventory["cached_base"]["files"][p.name] = sha256(p)
    from pypdf import PdfReader

    for version in ("v13", "v14"):
        p = Path(f"C:/HP/Hektor_{version}_30_Soru_Cevaplari.pdf")
        reader = PdfReader(p)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        target = out / "historical" / f"{version}_pdf_extracted.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        inventory["artifacts"][version]["pdf"] = {
            "path": str(p),
            "sha256": sha256(p),
            "pages": len(reader.pages),
            "extraction": str(target),
            "status": "historical_not_fresh_inference",
            "warning": "PDF extraction may replace glyphs; original raw JSONL is preferred",
        }
    dump(out / "inventory.json", inventory)
    print(
        json.dumps(
            {
                "questions": 60,
                "splits": {k: len(v) for k, v in splits.items()},
                "training_executed": False,
                "inventory": str(out / "inventory.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
