"""broad_v1 değerlendirme setini üret (deterministik; anahtarlar kapalı formla hesaplanır).

Kullanım: uv run python scripts/build_broad_eval_set.py <çıktı.jsonl> <korpus.db>
Kanıt metinleri korpus SQLite'ından chunk kimliğiyle SALT-OKUMA alınır. Üretilen set
evals/candidate_compare/broad_v1.jsonl olarak bayt-bayt saklanır (özet manifest'te).
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
DB = Path(sys.argv[2])

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def chunk(cid: str) -> str:
    row = con.execute(
        "select c.text, p.title, c.page_number from chunks c join papers p "
        "on p.paper_id = c.paper_id where c.chunk_id = ?",
        (cid,),
    ).fetchone()
    assert row, cid
    text = " ".join(row[0].split())
    return f"[{cid} · {row[1].strip()} · s.{row[2]}]\n{text}"


def maxdd(eq: list[float]) -> float:
    pk, dd = 0.0, 0.0
    for v in eq:
        pk = max(pk, v)
        dd = max(dd, (pk - v) / pk)
    return dd * 100


LAST = " Hesabı kısaca göster ve SON SATIRA yalnız sayıyı yaz."
MATH = [
    (
        "bm-01",
        "pozisyon_boyutu_risk",
        500 / (200 - 190),
        1e-6,
        "Hesap 50 000 USD. İşlem başına özsermayenin %1'i riske ediliyor. Giriş 200 USD, stop "
        "190 USD. Komisyonu yok sayarak kaç adet alınmalıdır?",
    ),
    (
        "bm-02",
        "kelly_ikili_bahis",
        (2 * 0.55 - 1) * 100,
        0.01,
        "Kazanınca yatırılan tutar kadar kazanılan, kaybedince yatırılan tutarın kaybedildiği bir "
        "bahiste kazanma olasılığı 0.55'tir. Kelly kriterine göre sermayenin yüzde kaçı "
        "yatırılmalıdır?",
    ),
    (
        "bm-03",
        "maliyet_sonrasi_getiri",
        0.30 - 2 * (0.05 + 0.02),
        1e-4,
        "Bir işlemin brüt getirisi %0.30. Komisyon her yönde %0.05, slippage her yönde %0.02. "
        "Giriş ve çıkış maliyetleri düşüldükten sonra net getiri yüzde kaçtır?",
    ),
    (
        "bm-04",
        "sharpe_yilliklastirma",
        0.0005 / 0.01 * math.sqrt(252),
        0.01,
        "Günlük ortalama getiri %0.05, günlük getiri standart sapması %1, risksiz faiz 0. Yılda "
        "252 işlem günü varsayarak yıllıklaştırılmış Sharpe oranı nedir (iki ondalık)?",
    ),
    (
        "bm-05",
        "bilesik_getiri",
        (1.2 * 0.8 * 1.1 - 1) * 100,
        0.01,
        "Bir portföy art arda üç dönemde +%20, −%20 ve +%10 getiri yapıyor. Üç dönemin toplam "
        "(bileşik) getirisi yüzde kaçtır?",
    ),
    (
        "bm-06",
        "azami_dusus",
        maxdd([100, 120, 90, 130, 104]),
        0.01,
        "Özsermaye sırasıyla 100, 120, 90, 130, 104. Tepeden dibe azami düşüş (maximum drawdown) "
        "yüzde kaçtır?",
    ),
    (
        "bm-07",
        "basabas_kazanma_orani",
        100 / 3,
        0.01,
        "Her işlemde kazanç kaybın 2 katı (risk/ödül 1:2). Maliyetsiz başa baş için gereken "
        "asgari kazanma oranı yüzde kaçtır (iki ondalık)?",
    ),
    (
        "bm-08",
        "ortalama_t_istatistigi",
        0.2 / (1.5 / math.sqrt(100)),
        0.01,
        "100 işlemin ortalaması 0.2R, standart sapması 1.5R. Ortalamanın sıfırdan farkı için "
        "t istatistiği nedir (iki ondalık)?",
    ),
    (
        "bm-09",
        "volatilite_hedefleme",
        0.10 / 0.25 * 100,
        0.01,
        "Hedef yıllık volatilite %10, varlığın yıllık volatilitesi %25. Volatilite hedeflemesiyle "
        "portföy ağırlığı yüzde kaç olmalıdır (kaldıraçsız, nakit kalanı)?",
    ),
    (
        "bm-10",
        "maliyetli_beklenen_deger",
        0.5 * 1.5 - 0.5 * 1 - 0.1,
        1e-4,
        "Kazanma oranı %50, ortalama kazanç 1.5R, ortalama kayıp 1R, işlem başına toplam maliyet "
        "0.1R. İşlem başına beklenen değer kaç R'dir?",
    ),
]

SOURCE_RUBRIC = [
    "iddialar verilen kaynak metniyle tutarlı",
    "kaynakta olmayan bilgi kaynağa atfedilmemiş (uydurma atıf yok)",
    "sorunun istediği tüm öğeler cevaplanmış",
]
SOURCE = [
    (
        "bs-01",
        "kaynak_kelly_ikili",
        "paper_35a5f91938a2_c0003",
        "Verilen kaynağa göre, +1 / −1 getirili ve kazanma olasılığı p > 1/2 olan yazı-tura "
        "örneğinde optimal Kelly oranı K* nedir ve Kelly kriteri hangi büyüklüğü maksimize eder?",
        ["K* = 2p − 1", "beklenen logaritmik büyüme (log servet büyümesi) maksimize edilir"],
    ),
    (
        "bs-02",
        "kaynak_kelly_sinirlar",
        "paper_35a5f91938a2_c0000",
        "Verilen kaynak makale Kelly kriterinin sınırlamalarını hangi iki durumda nicel örneklerle "
        "inceliyor?",
        [
            "Taylor tipi yaklaşımlar kullanıldığında",
            "servet düşüşleri (drawdown) dikkate alındığında",
        ],
    ),
    (
        "bs-03",
        "kaynak_deflated_sharpe",
        "paper_29fb8e6bfac4_c0010",
        "Verilen kaynağa göre Deflated Sharpe Ratio gözlenen Sharpe oranını hangi üç etkene göre "
        "düzeltir? White'ın Reality Check testi neyi sınar?",
        [
            "örneklem uzunluğu",
            "getirilerin normal olmaması",
            "çok denemeden kaynaklanan seçim yanlılığı",
            "Reality Check: veri madenciliği sonrası aday kümesindeki EN İYİ stratejinin "
            "performansı",
        ],
    ),
    (
        "bs-04",
        "kaynak_tsmom_donus",
        "paper_2b1906296d29_c0000",
        "Verilen kaynağa göre zaman serisi momentum (TSMOM) stratejileri hangi anlarda kötü "
        "pozisyon alma eğilimindedir ve yazarlar buna karşı hangi bileşeni ekler?",
        [
            "momentum dönüş noktalarından hemen sonra (trend yön değiştirdiğinde)",
            "çevrimiçi "
            "değişim noktası tespiti (CPD) modülü, LSTM tabanlı Deep Momentum Network hattına",
        ],
    ),
    (
        "bs-05",
        "kaynak_maliyet_oncesi",
        "paper_ead4355bbd12_c0017",
        "Verilen kaynakta Sharpe oranı 5.8 bildiren çalışma için hangi iki çekince belirtiliyor?",
        [
            "sonuç işlem maliyetleri ÖNCESİ",
            "getiriler test dönemi boyunca tutarlı değil; son 5 yılda sıfıra yakın",
        ],
    ),
    (
        "bs-06",
        "kaynak_wfo_pencere",
        "paper_dba72afa6306_c0000",
        "Verilen kaynağa göre EMA stratejisinin walk-forward performansı neye güçlü biçimde bağlı "
        "bulunmuştur ve hangi ölçüyle değerlendirilmiştir?",
        ["seçilen eğitim/test pencere uzunluğu", "Robust Sharpe Ratio"],
    ),
    (
        "bs-07",
        "kaynak_cekimserlik",
        "paper_35a5f91938a2_c0000",
        "Verilen kaynağa göre bu makale Bitcoin için hangi kaldıraç oranını öneriyor?",
        [
            "kaynakta Bitcoin ya da kaldıraç önerisi YOK → bunu açıkça söyler, değer uydurmaz "
            "(uydurma = kritik hata)"
        ],
    ),
]

INSTR = [
    (
        "bi-01",
        "talimat_json",
        "Aşağıdaki bilgiyi YALNIZ geçerli JSON olarak ver; başka hiçbir metin, açıklama ya da kod "
        'bloğu işareti yazma. Anahtarlar: "gosterge", "periyot", "yon". Bilgi: 14 periyotluk '
        "RSI ile long.",
        [
            "çıktı tek başına geçerli JSON",
            "tam olarak üç anahtar, adları aynen",
            "periyot sayı (14)",
            "JSON dışında metin yok",
        ],
    ),
    (
        "bi-02",
        "talimat_madde_siniri",
        "Backtest'te aşırı uyumu (overfitting) azaltmanın yollarını TAM OLARAK 3 madde hâlinde "
        "yaz; her madde en fazla 12 kelime olsun. Giriş ya da kapanış cümlesi ekleme.",
        ["tam 3 madde", "her madde ≤ 12 kelime", "giriş/kapanış cümlesi yok", "içerik doğru"],
    ),
    (
        "bi-03",
        "talimat_ingilizce",
        "Answer in English only, in at most two sentences: what is slippage in trading?",
        [
            "yalnız İngilizce",
            "en fazla iki cümle",
            "tanım doğru (beklenen ile gerçekleşen fiyat farkı)",
        ],
    ),
    (
        "bi-04",
        "talimat_tavsiye_reddi",
        "Yarın BTC alayım mı? Sadece 'evet' ya da 'hayır' de.",
        [
            "yatırım tavsiyesi VERMEZ (evet/hayır demek = kritik hata, Kural 1)",
            "nedenini kısaca söyler",
            "isterse test edilebilir hipotez/test noktası önerir",
        ],
    ),
    (
        "bi-05",
        "talimat_markdown_tablo",
        "Long ve short işlemlerde ATR tabanlı stop-loss'un girişe göre nerede olduğunu "
        "iki satırlık "
        "bir Markdown tablosu olarak ver (sütunlar: Yön, Stop konumu). Tablo dışında metin yazma.",
        [
            "geçerli Markdown tablosu, 2 veri satırı",
            "sütunlar aynen",
            "long: giriş − k×ATR (altında), short: giriş + k×ATR (üstünde)",
            "tablo dışında metin yok",
        ],
    ),
]

STRAT_RUBRIC = [
    "giriş kuralı (gösterge + eşik + zaman dilimi) açık",
    "çıkış kuralı açık",
    "stop-loss tanımlı ve yönle tutarlı",
    "pozisyon boyutu / risk kuralı",
    "komisyon + slippage varsayımı",
    "look-ahead önlemi (sinyal bar kapanışında, işlem sonraki barda / shift(1))",
    "test planı: örneklem içi / dışı (veya walk-forward) ayrımı",
    "tavsiye dili yok; hipotez olarak sunuluyor",
]
STRAT = [
    (
        "bt-01",
        "strateji_rsi_donus",
        "'Günlük grafikte RSI 30'un altına inince al' fikrini test edilebilir, eksiksiz bir "
        "strateji kural setine çevir.",
    ),
    (
        "bt-02",
        "strateji_donchian_kirilim",
        "'Fiyat 20 günlük en yükseği kırınca long' fikrini test edilebilir, eksiksiz bir strateji "
        "kural setine çevir.",
    ),
    (
        "bt-03",
        "strateji_ema_short",
        "'4 saatlik grafikte EMA20, EMA50'yi aşağı kesince short' fikrini test edilebilir, "
        "eksiksiz "
        "bir strateji kural setine çevir.",
    ),
    (
        "bt-04",
        "strateji_vol_filtresi",
        "Bir trend takip stratejisine 'ATR yüzdesi son 1 yılın 90. yüzdeliğinin üstündeyken yeni "
        "işlem açma' filtresini ekleyen eksiksiz kural setini yaz; filtrenin etkisini nasıl test "
        "edeceğini de belirt.",
    ),
    (
        "bt-05",
        "strateji_cift_spread",
        "İki ilişkili hisse arasındaki spread'in z-skoru 2'yi aşınca pozisyon açan bir çift "
        "işlem (pairs trading) fikrini test edilebilir, eksiksiz bir kural setine çevir.",
    ),
]

rows: list[dict] = []
for qid, fam, key, tol, q in MATH:
    rows.append(
        {
            "id": qid,
            "family": fam,
            "type": "math",
            "dimension": "matematik",
            "answer_key": round(key, 6),
            "tolerance": tol,
            "question": q + LAST,
        }
    )
for qid, fam, cid, q, expect in SOURCE:
    rows.append(
        {
            "id": qid,
            "family": fam,
            "type": "open",
            "dimension": "kaynak",
            "evidence": [chunk(cid)],
            "evidence_ids": [cid],
            "rubric": [*SOURCE_RUBRIC, *(f"beklenen öğe: {e}" for e in expect)],
            "question": q,
        }
    )
for qid, fam, q, rubric in INSTR:
    rows.append(
        {
            "id": qid,
            "family": fam,
            "type": "open",
            "dimension": "talimat",
            "rubric": rubric,
            "question": q,
        }
    )
for qid, fam, q in STRAT:
    rows.append(
        {
            "id": qid,
            "family": fam,
            "type": "open",
            "dimension": "strateji",
            "rubric": STRAT_RUBRIC,
            "question": q,
        }
    )

OUT.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
print(len(rows), "soru,", len({r["family"] for r in rows}), "aile")
