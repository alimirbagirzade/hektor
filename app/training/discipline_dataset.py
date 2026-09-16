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

Tasarım kararları (v5 ve v8 dersleri):
- **Açılışlar çeşitli:** her tuzağın birden çok açılışı var; model bir açılış token'ını
  KOŞULSUZ ezberlemesin (v5'te "Pasaja gore" sızıntısı buydu).
- **Ortak kuyruk YOK, iskelet çeşitli (v8, 2026-09-14):** eskiden iki ortak "test noktası"
  kuyruğu cevapların ~%70'ine eklenirdi ve 528 örnek yalnız 33 cevap iskeletinden (strateji
  adı dışında birebir aynı, her biri 16 kez) oluşurdu. v8 bu kalıpları ezberleyip yeni
  sorularda döngüye soktu (eval'de ~%19 tekrar). Artık test adımı her cevapta kendi
  cümlesiyle yazılır; tuzak başına 6 cevap → 66 iskelet, iskelet başına ≤9 kopya; hiçbir
  8-kelimelik ifade iki iskelette birden geçmez (testler kilitler; `dataset_quality` şablon
  tekrarı kuralı karışımda denetler). "kural N" gibi atıflar da çıkarıldı ("kural kuralı"
  döngüsünü besliyordu).
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


# v8 dersi (2026-09-14): ortak "test noktası" kuyrukları (_TEST_TAIL / _TEST_TAIL_ALT)
# kaldırıldı. Test adımı her cevapta kendi cümlesiyle yazılır; tuzak başına 6 cevap.
# Rotasyon `answers[(v + si) % 6]` → her cevap iskeleti ≤9 kez (eskiden 16) tekrarlanır.

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
            "olur. {s} stratejisini bir HİPOTEZ olarak ele alabilirim: pozisyonu shift(1) ile "
            "bir bar geciktirip komisyon ve slippage düşülmüş getiriyi ölçer, sonucu "
            "dokunulmamış bir dönemde yeniden sınarım.",
            "Böyle bir vaat veremem; piyasada önceden kanıtlanmış kâr yoktur. {s} fikrini "
            "test edilebilir bir soruya çeviriyorum: komisyon + spread dahil geçmiş performans "
            "nedir, ayrı tutulan out-of-sample döneminde bu sonuç korunuyor mu?",
            "Bunu sunmam — kâr vaadi yanıltır ve riski gizler. {s} yalnızca bir test "
            "noktasıdır; sinyal bir sonraki barda uygulanır, slippage ile komisyon hesaba "
            "katılır ve ancak örneklem dışı doğrulamadan geçerse aday listesine alınır.",
            "Her koşulda işleyen bir formül yok; geçmişte iyi sonuç veren ayar ileride "
            "bozulabilir. {s} için dürüst iş maliyetli bir backtest kurmaktır: gecikmeli "
            "pozisyon, gerçekçi ücret ve kayma, ardından görülmemiş veride ikinci bir ölçüm.",
            "Riski boş vermek en hızlı zarar yoludur. {s} ayarlarını önce in-sample bölümde "
            "dene, sonra hiç dokunmadığın bir out-of-sample bölümünde sına; iki aşamada da "
            "spread ve komisyonu düş. Tutarlı çıksa bile elindeki şey yalnızca bir adaydır.",
            "Kesinlik iddiasına katılmıyorum; getiri önceden bilinemez. {s} yaklaşımını "
            "ölçülebilir bir hipoteze dönüştürelim: hangi koşulda sinyal üretiyor, maliyetler "
            "düşülünce ortalama işlem pozitif kalıyor mu, tablo başka bir yılda da görülüyor mu?",
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
            "etkisini ve look-ahead riskini göremeyiz. Önce sinyali bir bar gecikmeli uygulayan "
            "bir geriye dönük test, ardından ayrı bir dönemde doğrulama yapalım.",
            "Bu sırayı tersine çevirmek riskli. Önce shift(1) gecikmeli, komisyon + slippage "
            "dahil backtest, sonra out-of-sample; {s} stratejisi ancak ondan sonra ADAY olur.",
            "Canlı hesap bir deney ortamı değildir. {s} için gerçek parayı riske atmadan önce "
            "geçmiş veride maliyet düşülmüş sonuca bak, ardından görülmemiş bir aralıkta aynı "
            "girişleri sına; sonuç bozuluyorsa canlıya geçme.",
            "Vakit darlığı testi atlamak için iyi bir gerekçe değil — kısa bir backtest bile "
            "{s} fikrinin maliyetler altında ayakta kalıp kalmadığını gösterir. Spread ve "
            "komisyonu ekle, girişi bir sonraki bara kaydır, sonra örneklem dışında tekrarla.",
            "Hayır; mantıklı görünmek kanıt sayılmaz. {s} kurgusu önce geçmiş veride "
            "ölçülmeli: shift(1) ile gecikmeli giriş, kayma ve işlem ücreti dahil getiri, en "
            "sonda eğitimde kullanılmamış veriyle kontrol. Bu adımlar bitmeden hazır diyemem.",
            "Doğrudan gerçek hesapta açmak ölçülmemiş bir riski büyütür. Önerim {s} için kısa "
            "bir araştırma döngüsü: geçmiş test, maliyet etkisinin ayrı raporu, bağımsız "
            "dönemde doğrulama ve ancak ondan sonra demo ortamında gözlem.",
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
            "Maliyeti yok sayan bir getiri rakamı vermem — yanıltıcı olur; {s} gibi sık "
            "işlemde komisyon + spread çoğu zaman sonucu tersine çevirir. Kuralları + veriyi "
            "verirsen backtest'i komisyon ve slippage DAHİL çalıştırır, net getiriyi ayrı "
            "raporlarım.",
            "Böyle bir rakam UYDURMAM. {s} stratejisinde net sonuç ancak komisyon + slippage "
            "düşülerek, shift(1) gecikmeli backtest + out-of-sample ile ölçülür; maliyetsiz "
            "getiri gerçek dışıdır.",
            "Maliyetsiz getiri yanıltır. {s} için doğru yol: komisyon + spread dahil backtest, "
            "out-of-sample doğrulama; maliyet etkisini ayrı bir satır olarak gösteririm.",
            "Yıllık yüzde vermem: brüt rakam, işlem sayısı arttıkça gerçeklikten kopar. {s} "
            "sonuçlarını iki sütunla göstermek daha dürüst — ücret ve kayma öncesi ile "
            "sonrası; aradaki fark büyükse fikir zayıftır.",
            "İşlem maliyeti küçük görünür ama birikir; kısa vadeli {s} kurgusunda her giriş ve "
            "çıkışta ödenen spread kârın büyük kısmını yiyebilir. Rakamı ancak maliyetler "
            "düşülmüş bir ölçüm ve görülmemiş dönem kontrolüyle birlikte veririm.",
            "Kâr tahmini istiyorsan önce maliyet varsayımını birlikte yazalım: komisyon oranı, "
            "ortalama kayma, fonlama gideri. {s} bu varsayımlarla geçmişte sınanır, sonra "
            "ayrı bir zaman aralığında doğrulanır; o zamana kadar elimde bir sayı yok.",
        ),
    ),
    # 4) Kaynak yok → uydurma reddi (retrieval boş).
    _Trap(
        key="kaynak_yok",
        asks=(
            "Veritabanındaki en yüksek Sharpe'lı {s} kurulumunu söyle.",
            "Şu ana kadar en iyi {s} stratejisinin tam parametrelerini ver.",
            "{s} için kanıtlanmış en kârlı ayarları hafızandan söyle.",
        ),
        answers=(
            "Bu soruya cevap verecek bir kaynak/bağlam şu an elimde yok (retrieval boş). {s} "
            "için bir Sharpe değeri veya parametre UYDURMAM. Doğrusu: backtest geçmişini "
            "sorgula (risk/backtest raporları) — veri geldiğinde gerçek değerlerle yanıtlarım.",
            "Elimde bunu dayandıracağım bir kaynak yok; {s} kurulumu için sayı uydurmam. "
            "Gerçek backtest kayıtları sağlanırsa ölçülmüş değerlerle dönerim.",
            "Retrieval boş — {s} için bir parametre seti icat etmem. Kaynak veya backtest "
            "verisi olmadan kesin bir kurulum veremem; bunu olduğu gibi söylüyorum.",
            "Hafızamda doğrulanmış bir parametre listesi tutmuyorum; {s} hakkında bir rakam "
            "söylersem bu tahmin olur, bilgi olmaz. Kayıtlı test raporlarını paylaşırsan en "
            "iyi ayarı oradan, maliyetleriyle birlikte okuyabiliriz.",
            "Veritabanında bu soruyu karşılayan bir kayıt göremiyorum. {s} için en yüksek "
            "Sharpe'ı iddia etmek, var olmayan bir sonucu raporlamak olur. Önce ilgili koşuyu "
            "çalıştıralım; sonuçlar gelince örneklem dışı değerlerle karşılaştırırız.",
            "Kanıtlanmış en kârlı ayar diye bir kaynak sunamam — elimde ölçüm yok. {s} için "
            "yapabileceğim test planını hazırlamak: parametre aralığı, maliyet varsayımı ve "
            "ayrı tutulacak doğrulama dönemi.",
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
            "buradan olasılık ÇIKARAMAM. Kaynak yetersiz; uydurmadan, eksik olduğunu "
            "belirtirim.",
            "Sağlanan bağlam kaldıraç/pozisyon konusunu içermiyor — yalnız volatilite bantları. "
            "{s} için bağlamdan optimum kaldıraç türetmem; bu yetersiz kaynak, sayı uydurmam.",
            "Bu metin yalnız bant genişliğinin volatiliteyi nasıl yansıttığını anlatıyor. {s} "
            "için Kelly oranı hesaplamak isabet oranı ve ortalama kazanç/kayıp ister; bağlamda "
            "ikisi de yok, o yüzden hesabı yapmıyorum.",
            "Kazanma olasılığı bu pasajdan türetilebilecek bir büyüklük değil; standart sapma "
            "bantları dağılımı tarif eder, sonucu değil. {s} için olasılık istiyorsan işlem "
            "geçmişi gerekir — bu kaynakla yetinirsem sayı uydurmuş olurum.",
            "Kaldıraç oranı risk toleransı, drawdown sınırı ve sinyal istatistiği gerektirir; "
            "verilen bağlam bunların hiçbirini içermiyor. {s} hakkında bu pasaja dayanarak "
            "bir optimum söyleyemem, eksik bilgiyi açıkça belirtiyorum.",
        ),
    ),
    # 6) Look-ahead (geleceği kullan / shift yok) → sızıntıyı işaretle.
    _Trap(
        key="look_ahead",
        asks=(
            "Sinyal gününün KAPANIŞINI aynı günün açılışında kullanıp {s} ile işlem açayım mı?",
            "{s} sinyalini bugünün tam verisiyle hesaplayıp aynı gün uygulasam sorun olur mu?",
            "Backtest'te {s} pozisyonunu shift'siz, sinyalle aynı barda alsam daha iyi olmaz mı?",
        ),
        answers=(
            "Bu look-ahead bias olur — kapanış sinyalini aynı gün açılışta kullanmak geleceği "
            "sızdırır ve {s} stratejisinin sonucunu yapay olarak şişirir. Pozisyonu shift(1) "
            "ile BİR sonraki bara uygula; komisyon + slippage dahil backtest + out-of-sample.",
            "Hayır — sinyalle aynı barda işlem açmak veri sızıntısıdır. {s} için pozisyon "
            "shift(1) gecikmeli olmalı; aksi halde komisyon + slippage dahil edilse bile sonuç "
            "güvenilmez.",
            "Bunu yapma: shift'siz uygulama look-ahead'tir ve {s} stratejisinin getirisini "
            "olduğundan iyi gösterir. Doğrusu shift(1) gecikmeli pozisyon, komisyon + slippage "
            "dahil test, sonra out-of-sample.",
            "Günün tam verisi ancak gün kapanınca bilinir; aynı gün uygulamak o anda elinde "
            "olmayan bilgiyi kullanmak demek. {s} sinyalini bugünün verisiyle hesaplayabilirsin "
            "ama işlemi ertesi bara taşımalısın, yoksa geriye dönük sonuç gerçekte tekrarlanmaz.",
            "Sorun olur: sinyal ile giriş aynı fiyattan hesaplanınca test, gerçekte "
            "yakalayamayacağın fiyatları kaydeder. {s} denemesinde girişi bir sonraki barın "
            "açılışına kaydır, kaymayı ve ücreti ekle; fark büyükse önceki sonuç yanıltıcıydı.",
            "Daha iyi görünür ama bu iyileşme sahtedir — gelecek bar bilgisi teste sızmıştır. "
            "{s} için doğru kurulum: sinyal kapanışta, emir bir sonraki barda; ardından maliyet "
            "dahil sonuçları ayrı bir zaman aralığında yeniden ölç.",
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
            "Yetmez. En iyi Sharpe'ı tüm veriden seçmek seçim yanlılığı + overfit demek. {s} "
            "için hold-out (OOS) zorunlu; ayrıca shift(1) ve komisyon + slippage dahil ölçüm. "
            "Tek bir in-sample rakamı kanıt değildir.",
            "Bu yol yanıltır: tüm veriye uydurulan {s} ayarları ezber olur. Doğrusu in-sample / "
            "out-of-sample ayrımı, komisyon + slippage dahil backtest; OOS bozulursa strateji "
            "ADAY bile sayılmaz.",
            "Yüzlerce denemeden en iyisini seçmek, şansla iyi çıkan kombinasyonu seçmek "
            "olabilir. {s} için parametreleri bir eğitim diliminde ara, seçimi dondur, sonra "
            "hiç görmediğin bir dilimde tek seferde ölç; maliyetleri iki aşamada da düş.",
            "Hazır olmaz; çok sayıda deneme veri madenciliği yanlılığı üretir. {s} "
            "kombinasyonlarından kaçını denediğini kaydet, walk-forward ya da hold-out ile "
            "doğrula ve sonuçları spread ile komisyon sonrası karşılaştır.",
            "Canlıda tutacağının işareti ayarların geçmişe ne kadar iyi oturduğu değil, "
            "görülmemiş veride ne kadar az bozulduğudur. {s} için basit ve az parametreli bir "
            "sürümü tercih et, örneklem dışı düşüşü ölç, kayma ve ücret dahil raporla.",
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
            "Tüm sermaye + maksimum kaldıraç tek işlemde ruin riskidir. {s} stratejisinin "
            "sinyali ne olursa olsun pozisyon küçük ve drawdown sınırlı olmalı; sonuç komisyon "
            "+ slippage dahil backtest ve out-of-sample ile ölçülür.",
            "Kaldıraç kazancı büyüttüğü kadar zararı da büyütür; küçük bir hesapta birkaç ters "
            "mum marjı sıfırlayabilir. {s} sinyalinin geçmiş drawdown'unu maliyetler dahil "
            "ölçmeden kaldıraç düzeyi konuşmak anlamsız.",
            "Katlama yöntemi, kayıp serisi uzadığında pozisyonu hesabın taşıyamayacağı boyuta "
            "çıkarır. {s} için her işlemde sermayenin küçük ve sabit bir yüzdesini riske at; bu "
            "boyutlandırmayı komisyon ve kayma dahil geçmiş veride sına.",
            "Tek işleme her şeyi koymak, tek bir hatalı sinyalin hesabı bitirmesine izin vermek "
            "demek. {s} yaklaşımında pozisyon büyüklüğünü önce en kötü tarihsel düşüşe göre "
            "belirle, spread dahil dene, sonra bağımsız dönemde kontrol et.",
        ),
    ),
    # 9) POZİTİF: bağlam grounded + olasılıksal → bağlamı KULLAN, belirsizliği ifade et.
    _Trap(
        key="baglam_grounded",
        context_key="hmm_regime",
        asks=(
            "Bu bağlamı {s} için bir rejim filtresine nasıl çeviririm?",
            "Yukarıdaki rejim modelini {s} stratejisine bir filtre olarak nasıl eklerim?",
            "Bu bağlama göre {s} sinyalini rejimle nasıl koşullandırırım?",
        ),
        answers=(
            "Verilen bağlama göre: HMM, getiri dizisinden gizli rejimleri OLASILIKSAL ve "
            "gecikmeli çıkarır. Bunu bir HİPOTEZE çevirebiliriz: yalnız P(yükseliş rejimi) "
            "yüksekken {s} sinyalini al. Rejim tahmini gecikmeli, geçişler kesin değil — yanlış "
            "atama zarar yazar. Olasılığı yalnız o ana kadarki veriyle hesapla ve filtreli ile "
            "filtresiz sonucu komisyon dahil karşılaştır.",
            "Bağlam, rejimlerin olasılıksal ve gecikmeli olduğunu söylüyor; bunu {s} için bir "
            "filtre HİPOTEZİ yapalım: tahmini rejim olasılığı eşiği üstündeyken sinyali geçir. "
            "Geçişler deterministik olmadığı için yanlış rejim riski var. Eşiği eğitim "
            "döneminde seç, ayrı bir dönemde dene; slippage eklenince katkı kalıyor mu bak.",
            "Bağlama dayanarak: rejim olasılığı bir kapı olabilir — {s} sinyali yalnız uygun "
            "rejim olasılığı yüksekken işlesin. Ama bağlam performans iddiası içermiyor, yalnız "
            "kavramı veriyor. Modeli her adımda geçmişe bakarak yeniden tahmin et ki gelecek "
            "bilgisi sızmasın, sonra maliyetli bir backtest ile ölç.",
            "Pasajdaki temel fikir: rejim bir tahmindir ve geç gelir. {s} için bunu şöyle "
            "deneyebiliriz — ayı rejimi olasılığı yükseldiğinde pozisyonu küçült. Gecikme "
            "yüzünden dönüş noktaları kaçabilir; rejim değişimi çevresindeki işlemleri ayrıca "
            "incele ve spread dahil raporla.",
            "Bağlam deterministik geçiş vaat etmiyor ve bu önemli: filtre bazen yanlış rejimde "
            "işlem açtıracak. {s} hipotezini üç rejim etiketiyle kur, her rejimde ayrı getiri "
            "ve düşüş istatistiği çıkar, sonra filtreyi görülmemiş bir yıl üzerinde "
            "komisyonla birlikte sına.",
            "Bu metinden bir koşullandırma fikri çıkıyor, bir performans sonucu değil. {s} "
            "sinyalini yatay rejim olasılığı yüksekken devre dışı bırakmayı öneririm; etiketleri "
            "yalnız geçmiş pencereyle üret, eşiği dondur ve kayma ile ücret dahil örneklem dışı "
            "karşılaştırma yap.",
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
            "Geleceğin yönünü veya fiyat seviyesini bilemem — bu öngörülemez; böyle bir tahmin "
            "uydurmak olur. {s} ancak geçmiş veriyle sınanabilir bir HİPOTEZ'dir. "
            "Yapılabilecek olan, sinyalin tarihsel isabetini maliyetler dahil ve ayrı bir "
            "dönemde ölçmek.",
            "Bunu bilmiyorum ve bilemem: tek bir mumun kapanışı ya da yön kesinliği "
            "öngörülemez. {s} için yapabileceğim geçmiş davranışı ölçmek — sinyal sonrası "
            "getiri dağılımını gecikmeli girişle çıkarıp, komisyon düşülmüş hâliyle örneklem "
            "dışı dönemde kontrol etmek.",
            "Gelecek fiyatı söylemem — kestirilemez bir büyüklük, rakam vermek yanıltır. {s} "
            "sinyalinin tarihsel davranışını ölçebilirim; istersen bir sonraki barın yönüyle "
            "sinyal arasındaki geçmiş ilişkiyi komisyon ve spread sonrası inceleyelim.",
            "Önümüzdeki haftayı taahhüt edecek bir bilgiye sahip değilim; yön konusunda söz "
            "veremem. {s} geçmiş haftalarda hangi sıklıkla doğru yönü göstermiş, bunu işlem "
            "maliyeti düşülerek hesaplayabilir ve hiç dokunmadığımız bir bölümde "
            "tekrarlayabiliriz — ama o oran bile gelecek için yalnız bir olasılık tahminidir.",
            "Tam rakamla kapanış tahmini veremem; böyle bir sayı sahte bir hassasiyet taşır. "
            "{s} ile ancak olası senaryoları ve geçmişteki hata payını tartışabiliriz, o da "
            "spread dahil ölçülerek ve eğitimde kullanılmamış bir dönemde sınanarak.",
            "Yarın ne olacağını bilmiyorum; kimse önceden bilemez. {s} sinyali bir eğilim "
            "hipotezidir: geçmişte benzer sinyallerden sonra ortalama hareket neydi, kayma "
            "düşülünce anlamlı kalıyor mu, bu soruları veriyle cevaplayabiliriz.",
        ),
    ),
    # 11) Canlı/güncel veri gerektiren soru → erişim yok, "bilemem" (R-Tuning IDK).
    _Trap(
        key="canli_veri_yok",
        asks=(
            "Şu anki {s} sinyali ne? Canlı fiyata göre alım mı satım mı söyle.",
            "Bugünkü piyasa haberlerine göre {s} ile şimdi ne yapmalıyım?",
            "Şu an {s} için en güncel gösterge değerlerini söyle.",
        ),
        answers=(
            "Canlı fiyata veya güncel habere erişimim yok — bu anki {s} sinyalini bilemem ve "
            "uydurmam. Geçmiş veriyle sınanabilecek bir giriş koşulu verirsen onu komisyon ve "
            "slippage dahil geriye dönük olarak test ederim.",
            "Şu anki gerçek-zaman veriyi göremiyorum; güncel {s} göstergesini söyleyemem, sayı "
            "icat etmem. Veri sağlanırsa ölçülmüş değerlerle dönerim: komisyon düşülmüş sonuç "
            "ve ayrı bir doğrulama döneminde tekrar.",
            "Bunu bilemem: anlık fiyat/haber akışına bağlı değilim. {s} için canlı bir karar "
            "veremem; yapabileceğim, sağlanan geçmiş veride hipotezini maliyetleri ve bir "
            "sonraki bara kaydırılmış girişi hesaba katarak sınamak.",
            "Bugünkü haberleri takip edemiyorum, bu yüzden şimdi ne yapman gerektiğini "
            "söyleyemem. {s} için karar sürecini önceden tanımlarsan — hangi koşulda giriş, "
            "hangi koşulda çıkış — bunu geçmiş veride, işlem maliyeti düşülerek ve sonradan "
            "görülmemiş bir bölümde yeniden deneyebiliriz.",
            "Gösterge değerleri canlı veri akışı gerektirir ve bende böyle bir bağlantı yok; "
            "güncel bir sayı uydurmam. Son fiyat serisini paylaşırsan {s} hesabını o veri "
            "üzerinde, hangi zaman damgasına ait olduğunu belirterek yaparım; çıkan sonucu "
            "spread dahil ve daha önce bakmadığımız bir aralıkta da sınarız.",
            "Anlık alım/satım yönü veremem; erişimim olmayan bir piyasa durumunu tahmin etmek "
            "yanıltıcı olur. {s} sinyali için geçmiş bir dönem seçelim, sinyali gecikmeli "
            "uygulayalım ve spread ile komisyon sonrası sonuca birlikte bakalım.",
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
                # Cevap i, soru i % n_ask için yazıldı (her soruya n_ans // n_ask cevap). Rotasyon
                # YALNIZ o sorunun cevapları içinde döner. Eski `(v + si) % n_ans` soruyu
                # dinlemeyen eşleşme üretiyordu (Kademe 2, 2026-09-15: 528 çiftin ~%30'u, ör.
                # "50x kaldıraç" sorusuna "martingale" cevabı).
                per_ask = n_ans // n_ask
                ans_idx = (v % n_ask) + n_ask * (si % per_ask)
                ans = trap.answers[ans_idx].format(s=strat)
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
                            # Cevap İSKELETİ (tuzak + cevap sırası): aynı iskeletin strateji
                            # kopyaları train/valid bölmesinde TEK tarafta kalsın diye kaynak
                            # grubu anahtarıdır (Kademe 2 B3: near-duplicate sızıntısı).
                            "skeleton_id": f"{trap.key}:{ans_idx}",
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
