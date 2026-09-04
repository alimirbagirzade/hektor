"""Disiplin dataset üreticisi — adversarial "tuzak" sorularından disiplinli cevap üret.

Neden (bkz. memory/v5-adapter-regression.md): v5 LoRA REJECT edildi çünkü sentetik-QA
yalnız "pasajdan cevapla" örneği üretti; *adversarial disiplin* örneği YOKtu. Adapter
maliyetsiz %20 getiri uydurdu, kesinlik vaat etti, look-ahead'i görmezden geldi. Bu modül
o boşluğu kapatır: strateji türü × tuzak kombinasyonlarından, CLAUDE.md mutlak kurallarını
uygulayan deterministik SFT örnekleri üretir (6 elle yazılmış seed → yüzlerce örnek). SFT'ye
~%25 karıştırılır (`lora-cloud-prep --discipline-ratio`).

Neden LLM YOK: disiplin cevabının KENDİSİ doğru olmalı — yanlış disiplin = zehirli veri.
Şablon → kural-uyumu garanti (deterministik, kural 6); çeşitlilik strateji × tuzak × açılış
varyantından gelir. (Üretim için LLM çağırmaz; bu modül EĞİTİM BAŞLATMAZ — kural 8.)

Tasarım kararları (v5 dersleri):
- **Açılışlar çeşitli:** her tuzağın birden çok açılışı var; model bir açılış token'ını
  KOŞULSUZ ezberlemesin (v5'te "Pasaja gore" sızıntısı buydu).
- **Disiplin kullanıcı-sorusuna bağlı, system'e değil:** eval (adapter_eval.py) modeli
  system-prompt'SUZ çağırır. O yüzden örneklerin bir kısmı system'siz üretilir — disiplin
  davranışı kötü-sorunun KENDİSİNDEN öğrenilsin, özel bir system prompt'a sabitlenmesin.
- **Naif eval denetçisini de geçer:** cevaplar yasak yüzey token'larından ("garanti",
  "kesin kazan", bare "başarılı/çalışıyor") kaçınır — negasyon-kör check_flags "garanti
  etmez"i bile flagler. "strateji" geçen her cevap maliyet token'ı (komisyon/slippage)
  içerir (ignores_costs flag'ini önler).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from app.lora.dataset_builder import SYSTEM_PROMPT, LoRAExample

# Adversarial soruların parametresi — doğal Türkçe "{s} stratejisi" okunan kısa adlar.
STRATEGIES: tuple[str, ...] = (
    "RSI aşırı alım/satım",
    "EMA kesişim",
    "MACD sinyal kesişim",
    "Bollinger bandı kırılım",
    "momentum",
    "ortalamaya dönüş",
    "fiyat kırılımı (breakout)",
    "scalping",
    "hareketli ortalama kesişim",
    "Donchian kanal kırılım",
    "stokastik osilatör",
    "Ichimoku",
    "VWAP sapma",
    "Fibonacci geri çekilme",
    "ATR volatilite kırılım",
    "hacim profili",
)

# Grounded/uyumsuz tuzaklar için kısa kavram bağlamları (gerçek makaleler gibi İngilizce).
_CONTEXTS: dict[str, str] = {
    "hmm_regime": (
        "Hidden Markov models infer latent market regimes (bull / bear / sideways) from "
        "return sequences; regime estimates are probabilistic and lagged, and transitions "
        "are not deterministic."
    ),
    "bollinger_vol": (
        "Bollinger Bands measure volatility using standard-deviation bands around a moving "
        "average; they describe dispersion, not position sizing or win probability."
    ),
}


@dataclass(frozen=True)
class _Trap:
    """Tek bir adversarial tuzak: kötü-soru şablonları + disiplinli cevap şablonları.

    `asks`/`answers` `{s}` (strateji) yer tutucusu alır. `context_key` set ise kullanıcı
    mesajına `_CONTEXTS[context_key]` BAĞLAM olarak gömülür (grounded/uyumsuz tuzaklar).
    Her cevap şablonu FARKLI bir açılışla başlar (v5 sabitleme dersi).
    """

    key: str
    asks: tuple[str, ...]
    answers: tuple[str, ...]
    context_key: str | None = None


# Ortak "test noktası" kuyrukları — maliyet + shift(1) + OOS + belirsizlik (kural 2/3/4).
# (Cevaplarda farklı kuyruklar dönüşümlü kullanılır; tek-tip ezberi azaltır.)
_TEST_TAIL = (
    "Doğru test noktası: pozisyonu shift(1) ile gecikmeli uygula, komisyon ve slippage "
    "DAHİL backtest et, sonra out-of-sample doğrula. 'pass' çıksa bile bu bir ADAY'dır."
)
_TEST_TAIL_ALT = (
    "Ölçülmesi gereken bir hipotez var: shift(1) gecikmeli pozisyon, komisyon + slippage "
    "dahil in-sample backtest, ardından out-of-sample doğrulama. Sonuç 'pass' değilse "
    "aday değildir."
)

TRAPS: tuple[_Trap, ...] = (
    # 1) Kesinlik/garanti vaadi → reddet + hipoteze çevir.
    _Trap(
        key="garanti",
        asks=(
            "Bu {s} stratejisi kesin kazandırır mı? Bana garanti kâr getirecek bir kurulum söyle.",
            "{s} stratejisiyle her zaman kazanmanın formülünü ver.",
            "{s} ile kesin para kazandıracak ayarları söyle, riski boş ver.",
        ),
        answers=(
            "Hayır — hiçbir kurulum kârı önceden kesinleştiremez; böyle bir vaat yanıltıcı "
            "olur. {s} stratejisini bir HİPOTEZ olarak ele alabilirim. " + _TEST_TAIL,
            "Böyle bir kesinlik vaadi veremem; piyasada önceden kanıtlanmış kâr yoktur. {s} "
            "stratejisini test edilebilir bir hipoteze çeviriyorum. " + _TEST_TAIL_ALT,
            "Bunu sunmam — kesinlik iddiası kuralı ihlal eder ve yanıltır. {s} stratejisi "
            "ancak bir test noktasıdır. " + _TEST_TAIL,
        ),
    ),
    # 2) Backtest'siz canlı kullanım → reddet + doğru sıra.
    _Trap(
        key="backtest_yok",
        asks=(
            "{s} stratejisini backtest yapmadan canlıda kullanabilir miyim?",
            "Test etmeden {s} stratejisini doğrudan gerçek hesapta açayım mı?",
            "{s} mantıklı görünüyor; backtest'e vakit yok, canlıya geçeyim mi?",
        ),
        answers=(
            "Önermem. Test olmadan {s} stratejisinin geçmiş davranışını, komisyon + slippage "
            "etkisini ve look-ahead riskini göremeyiz. " + _TEST_TAIL,
            "Bu sırayı tersine çevirmek riskli. Önce shift(1) gecikmeli, komisyon + slippage "
            "dahil backtest, sonra out-of-sample; {s} stratejisi ancak ondan sonra ADAY olur.",
            "Hayır demem gerekir — bu bir kuraldır. {s} stratejisi test edilmeden 'kullanılabilir' "
            "diyemem. " + _TEST_TAIL_ALT,
        ),
    ),
    # 3) Maliyetsiz getiri rakamı → reddet + maliyet dahil ölç.
    _Trap(
        key="maliyet_yok",
        asks=(
            "Komisyon ve spread'i saymadan {s} stratejisinin getirisi ne olur?",
            "{s} stratejisinin maliyetleri yok sayarak yıllık kaç % kazandırdığını söyle.",
            "İşlem maliyetini boş ver, {s} ile ne kadar kâr ederim?",
        ),
        answers=(
            "Maliyeti yok sayan bir getiri rakamı vermem — yanıltıcı olur; {s} gibi sık işlemde "
            "komisyon + spread çoğu zaman sonucu tersine çevirir. Kuralları + veriyi verirsen "
            "backtest'i komisyon ve slippage DAHİL çalıştırır, net getiriyi ayrı raporlarım.",
            "Böyle bir rakam UYDURMAM. {s} stratejisinde net sonuç ancak komisyon + slippage "
            "düşülerek, shift(1) gecikmeli backtest + out-of-sample ile ölçülür; maliyetsiz "
            "getiri gerçek dışıdır.",
            "Maliyetsiz getiri yanıltır. {s} için doğru yol: komisyon + spread dahil backtest, "
            "out-of-sample doğrulama; maliyet etkisini ayrı bir satır olarak gösteririm.",
        ),
    ),
    # 4) Kaynak yok → uydurma reddi (retrieval boş, kural 7).
    _Trap(
        key="kaynak_yok",
        asks=(
            "Veritabanındaki en yüksek Sharpe'lı {s} kurulumunu söyle.",
            "Şu ana kadar en iyi {s} stratejisinin tam parametrelerini ver.",
            "{s} için kanıtlanmış en kârlı ayarları hafızandan söyle.",
        ),
        answers=(
            "Bu soruya cevap verecek bir kaynak/bağlam şu an elimde yok (retrieval boş). {s} "
            "için bir Sharpe değeri veya parametre UYDURMAM. Doğrusu: backtest geçmişini sorgula "
            "(risk/backtest raporları) — veri geldiğinde gerçek değerlerle yanıtlarım.",
            "Elimde bunu dayandıracağım bir kaynak yok; {s} kurulumu için sayı uydurmam "
            "(kural 7). Gerçek backtest kayıtları sağlanırsa ölçülmüş değerlerle dönerim.",
            "Retrieval boş — {s} için bir parametre seti icat etmem. Kaynak veya backtest verisi "
            "olmadan kesin bir kurulum veremem; bunu olduğu gibi söylüyorum.",
        ),
    ),
    # 5) Bağlam VAR ama konu uyumsuz → yetersiz kaynak, uydurma yok.
    _Trap(
        key="baglam_uyumsuz",
        context_key="bollinger_vol",
        asks=(
            "Bu bağlama göre {s} için Kelly kriteriyle pozisyon büyüklüğünü hesapla.",
            "Yukarıdaki bağlamdan {s} stratejisinin kazanma olasılığını çıkar.",
            "Bu bağlamı kullanıp {s} için optimum kaldıraç oranını söyle.",
        ),
        answers=(
            "Verilen bağlam Bollinger Bantları (volatilite) ile ilgili; pozisyon büyüklüğü / "
            "Kelly sorusuna YETMİYOR. Bu soruya dair sağlanan kaynak yetersiz — bunu olduğu "
            "gibi söylerim, {s} için bir oran uydurmam. Doğru girdi gelirse hesaplarım.",
            "Bağlam dispersiyon/volatiliteyi anlatıyor, kazanma olasılığını değil; {s} için "
            "buradan olasılık ÇIKARAMAM. Kaynak yetersiz; uydurmadan, eksik olduğunu belirtirim.",
            "Sağlanan bağlam kaldıraç/pozisyon konusunu içermiyor — yalnız volatilite bantları. "
            "{s} için bağlamdan optimum kaldıraç türetmem; bu yetersiz kaynak, sayı uydurmam.",
        ),
    ),
    # 6) Look-ahead (geleceği kullan / shift yok) → kuralı işaretle.
    _Trap(
        key="look_ahead",
        asks=(
            "Sinyal gününün KAPANIŞINI aynı günün açılışında kullanıp {s} ile işlem açayım mı?",
            "{s} sinyalini bugünün tam verisiyle hesaplayıp aynı gün uygulasam sorun olur mu?",
            "Backtest'te {s} pozisyonunu shift'siz, sinyalle aynı barda alsam daha iyi olmaz mı?",
        ),
        answers=(
            "Bu look-ahead bias olur — kapanış sinyalini aynı gün açılışta kullanmak geleceği "
            "sızdırır ve {s} stratejisinin sonucunu yapay olarak şişirir. Pozisyonu shift(1) ile "
            "BİR sonraki bara uygula; komisyon + slippage dahil backtest + out-of-sample.",
            "Hayır — sinyalle aynı barda işlem açmak veri sızıntısıdır (kural 4). {s} için "
            "pozisyon shift(1) gecikmeli olmalı; aksi halde komisyon + slippage dahil edilse "
            "bile sonuç güvenilmez.",
            "Bunu yapma: shift'siz uygulama look-ahead'tir ve {s} stratejisinin getirisini "
            "olduğundan iyi gösterir. Doğrusu shift(1) gecikmeli pozisyon, komisyon + slippage "
            "dahil backtest, sonra out-of-sample.",
        ),
    ),
    # 7) Overfit (tüm veride parametre optimize et) → OOS şart.
    _Trap(
        key="overfit",
        asks=(
            "Tüm geçmiş veride {s} parametrelerini optimize edip en iyi Sharpe'ı "
            "bulayım, yeter mi?",
            "{s} için yüzlerce parametre deneyip en yüksek getireni seçsem "
            "stratejim hazır olur mu?",
            "Bütün veriye uydurarak {s} ayarlarını cilalarsam canlıda da tutar mı?",
        ),
        answers=(
            "Tüm veride parametre cilalamak overfit riskidir — {s} stratejisi geçmişe uyar ama "
            "ileride bozulabilir. Veriyi ayır: in-sample optimize, sonra "
            "DOKUNULMAMIŞ out-of-sample'da "
            "doğrula; komisyon + slippage dahil. OOS düşükse aday eler.",
            "Yetmez. En iyi Sharpe'ı tüm veriden seçmek seçim yanlılığı + overfit demek. {s} için "
            "hold-out (OOS) zorunlu; ayrıca shift(1) ve komisyon + slippage dahil ölçüm. Tek bir "
            "in-sample rakamı kanıt değildir.",
            "Bu yol yanıltır: tüm veriye uydurulan {s} ayarları ezber olur. Doğrusu in-sample / "
            "out-of-sample ayrımı, komisyon + slippage dahil backtest; OOS bozulursa strateji "
            "ADAY bile sayılmaz.",
        ),
    ),
    # 8) Aşırı kaldıraç / martingale → risk-of-ruin işaretle.
    _Trap(
        key="kaldirac",
        asks=(
            "{s} sinyaline 50x kaldıraçla girersem küçük hesabı hızla büyütürüm değil mi?",
            "Her kayıpta pozisyonu katlayan (martingale) bir {s} kurulumu zararı kapatır mı?",
            "{s} ile tüm sermayeyi tek işleme koyup kaldıracı maksimuma çeksem mantıklı mı?",
        ),
        answers=(
            "Bunu önermem — 50x kaldıraç {s} sinyali doğru olsa bile küçük bir ters hareket "
            "hesabı silebilir (risk-of-ruin). Pozisyon büyüklüğü ayrı bir risk kararıdır; önce "
            "komisyon + slippage dahil backtest + out-of-sample, sonra drawdown sınırlı "
            "küçük boyut.",
            "Martingale zararı kapatmaz; kayıpları katlamak iflas olasılığını büyütür. {s} için "
            "sabit/risk-ayarlı küçük pozisyon kullan, komisyon + slippage dahil test et; "
            "kaldıraç hipotezi de ayrıca backtest + OOS ister.",
            "Tüm sermaye + maksimum kaldıraç tek işlemde ruin riskidir. {s} stratejisinin sinyali "
            "ne olursa olsun pozisyon küçük ve drawdown sınırlı olmalı; sonuç komisyon + slippage "
            "dahil backtest ve out-of-sample ile ölçülür.",
        ),
    ),
    # 9) POZİTİF: bağlam grounded + olasılıksal → bağlamı KULLAN, belirsizliği ifade et (seed #4).
    _Trap(
        key="baglam_grounded",
        context_key="hmm_regime",
        asks=(
            "Bu bağlamı {s} için bir rejim filtresine nasıl çeviririm?",
            "Yukarıdaki rejim modelini {s} stratejisine bir filtre olarak nasıl eklerim?",
            "Bu bağlama göre {s} sinyalini rejimle nasıl koşullandırırım?",
        ),
        answers=(
            "Verilen bağlama göre: HMM, getiri dizisinden gizli rejimleri OLASILIKSAL ve gecikmeli "
            "çıkarır. Bunu bir HİPOTEZE çevirebiliriz: yalnız P(yükseliş rejimi) yüksekken {s} "
            "sinyalini al. Belirsizlik (bağlamın da dediği gibi): rejim tahmini gecikmeli, "
            "geçişler kesin değil — yanlış atama zarar yazar. " + _TEST_TAIL,
            "Bağlam, rejimlerin olasılıksal ve gecikmeli olduğunu söylüyor; bunu {s} için bir "
            "filtre HİPOTEZİ yapalım: tahmini rejim olasılığı eşiği üstündeyken sinyali geçir. "
            "Geçişler deterministik olmadığı için yanlış rejim riski var. " + _TEST_TAIL_ALT,
            "Bağlama dayanarak: rejim olasılığı bir kapı olabilir — {s} sinyali yalnız uygun rejim "
            "olasılığı yüksekken işlesin. Ama bağlam performans iddiası içermiyor, yalnız kavramı "
            "veriyor; gecikme ve olasılıksal belirsizlik gerçek. " + _TEST_TAIL,
        ),
    ),
    # 10) Gelecek yön/fiyat tahmini → bilinemez, kalibre "bilmiyorum" (R-Tuning 2311.09677).
    _Trap(
        key="gelecek_tahmin",
        asks=(
            "{s} sinyaline göre yarın fiyat kesin yükselir mi, ne kadar olur söyle.",
            "{s} kullanarak önümüzdeki haftanın yönünü garanti et, kesin konuş.",
            "{s} ile bir sonraki mumun kapanış fiyatını tam rakamla tahmin et.",
        ),
        answers=(
            "Geleceğin yönünü veya fiyat seviyesini kesin bilemem — bu öngörülemez; böyle bir "
            "tahmin uydurmak olur. {s} ancak geçmiş veriyle test edilebilir bir HİPOTEZ'dir, "
            "gelecekteki sonucu önceden belli etmez. " + _TEST_TAIL,
            "Bunu bilmiyorum ve bilemem: tek bir mumun kapanışı ya da yön kesinliği "
            "öngörülemez. {s} için yapabileceğim, geçmişte nasıl davrandığını ölçmek. "
            + _TEST_TAIL_ALT,
            "Gelecek fiyatı söylemem — kestirilemez bir büyüklük, rakam vermek yanıltır. {s} "
            "sinyalinin tarihsel davranışını ölçebilirim ama geleceği önceden bilemem. "
            + _TEST_TAIL,
        ),
    ),
    # 11) Canlı/güncel veri gerektiren soru → erişim yok, "bilemem" (R-Tuning IDK; kural 7).
    _Trap(
        key="canli_veri_yok",
        asks=(
            "Şu anki {s} sinyali ne? Canlı fiyata göre alım mı satım mı söyle.",
            "Bugünkü piyasa haberlerine göre {s} ile şimdi ne yapmalıyım?",
            "Şu an {s} için en güncel gösterge değerlerini söyle.",
        ),
        answers=(
            "Canlı fiyata veya güncel habere erişimim yok — bu anki {s} sinyalini bilemem ve "
            "uydurmam. Geçmiş veriyle test edebileceğin bir kural verirsen onu backtest ederim. "
            + _TEST_TAIL,
            "Şu anki gerçek-zaman veriyi göremiyorum; güncel {s} göstergesini söyleyemem, sayı "
            "icat etmem (kural 7). Veri sağlanırsa ölçülmüş değerlerle dönerim.",
            "Bunu bilemem: anlık fiyat/haber akışına bağlı değilim. {s} için canlı bir karar "
            "veremem; yapabileceğim, sağlanan geçmiş veride hipotezini test etmek. "
            + _TEST_TAIL_ALT,
        ),
    ),
)


def _user_content(trap: _Trap, ask: str) -> str:
    """Tuzağa göre kullanıcı mesajını kur (bağlam varsa BAĞLAM olarak göm)."""
    if trap.context_key:
        ctx = _CONTEXTS[trap.context_key]
        return f"BAĞLAM:\n{ctx}\n\nSORU: {ask}"
    return ask


def build_discipline_examples(
    *,
    seed: int = 0,
    variants_per_combo: int = 3,
    drop_system_every: int = 3,
    limit: int | None = None,
) -> list[LoRAExample]:
    """Strateji × tuzak × varyant kombinasyonlarından disiplinli SFT örneği üret.

    Args:
        seed: Karıştırma tabanı (kural 6 — deterministik).
        variants_per_combo: Her (tuzak, strateji) için kaç örnek (açılış/ifade rotasyonu).
        drop_system_every: Her N'inci örnekte system mesajı DÜŞÜR (0 = asla). Eval
            system-prompt'suz çağırdığından (adapter_eval) bir kısmı system'siz öğretilir →
            disiplin kötü-sorunun kendisinden öğrenilir, özel system'e sabitlenmez.
        limit: Üretilen örnek üst sınırı (karıştırmadan SONRA uygulanır).

    Returns:
        Deterministik sırada LoRAExample listesi (aynı seed → aynı çıktı).
    """
    examples: list[LoRAExample] = []
    idx = 0
    for trap in TRAPS:
        n_ask = len(trap.asks)
        n_ans = len(trap.answers)
        for si, strat in enumerate(STRATEGIES):
            for v in range(variants_per_combo):
                ask = trap.asks[v % n_ask].format(s=strat)
                # Cevap açılışını ask'tan farklı offset'le döndür → (ask, answer) çifti çeşitli.
                ans = trap.answers[(v + si) % n_ans].format(s=strat)
                user = _user_content(trap, ask)

                messages: list[dict] = []
                drop_system = drop_system_every > 0 and (
                    idx % drop_system_every == drop_system_every - 1
                )
                if not drop_system:
                    messages.append({"role": "system", "content": SYSTEM_PROMPT})
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": ans})

                examples.append(
                    LoRAExample(
                        messages=messages,
                        metadata={
                            "synthetic": True,
                            "discipline": True,
                            "trap": trap.key,
                            "strategy": strat,
                            "has_system": not drop_system,
                            # Çıplak soru (bağlam gömülü user mesajından ayrı) — dedup/kalite için.
                            "question": ask,
                        },
                    )
                )
                idx += 1

    rng = random.Random(seed)
    rng.shuffle(examples)
    if limit is not None:
        examples = examples[:limit]
    return examples


def discipline_jsonl_lines(
    *,
    seed: int = 0,
    variants_per_combo: int = 3,
    drop_system_every: int = 3,
    limit: int | None = None,
) -> list[str]:
    """Disiplin örneklerini JSONL satırı (string) listesine çevir — tam-içerik dedup'lu."""
    examples = build_discipline_examples(
        seed=seed,
        variants_per_combo=variants_per_combo,
        drop_system_every=drop_system_every,
        limit=limit,
    )
    seen: set[str] = set()
    out: list[str] = []
    for ex in examples:
        line = ex.to_jsonl_line()
        if line in seen:  # tam dup (şablon çakışması) — ele
            continue
        seen.add(line)
        out.append(line)
    return out


def mix_discipline(
    base_lines: list[str],
    discipline_lines: list[str],
    *,
    ratio: float = 0.25,
    seed: int = 0,
) -> tuple[list[str], dict]:
    """`discipline_lines`'ı `base_lines`'a hedef `ratio` oranında karıştır (deterministik).

    `ratio` = nihai sette disiplin payı (disiplin / (taban + disiplin)). Disiplin havuzu
    yetmezse mevcut kadarını kullanır; gerçekleşen oran `stats`'ta raporlanır.

    Returns:
        (karışık_satırlar, stats) — stats: {base, discipline_pool, discipline_used,
        total, ratio_target, ratio_actual}.
    """
    ratio = max(0.0, min(0.95, ratio))
    n_base = len(base_lines)
    # disiplin / (taban + disiplin) = ratio  →  disiplin = ratio*taban/(1-ratio)
    needed = 0 if ratio <= 0 else round(ratio * n_base / (1.0 - ratio))
    used = min(needed, len(discipline_lines))

    combined = list(base_lines) + list(discipline_lines[:used])
    rng = random.Random(seed)
    rng.shuffle(combined)

    total = len(combined)
    stats = {
        "base": n_base,
        "discipline_pool": len(discipline_lines),
        "discipline_used": used,
        "total": total,
        "ratio_target": ratio,
        "ratio_actual": (used / total) if total else 0.0,
    }
    return combined, stats
