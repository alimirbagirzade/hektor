# Knowledge Card Talimatı

Makaleden YALNIZCA aşağıdaki JSON şemasında çıktı üret. Açıklama, ön söz veya kod bloğu işareti ekleme; sadece JSON döndür.

{
  "paper_id": "...",
  "title": "...",
  "year": "...",
  "domain": "...",
  "main_claim": "...",
  "methods": [],
  "datasets": [],
  "trading_relevance": "...",
  "limitations": [],
  "possible_strategy_hypotheses": [],
  "risk_warnings": [],
  "implementation_notes": []
}

## Kurallar (bağlayıcı)

1. **Rakam uydurma yasak.** Hiçbir alana makalede AYNEN geçmeyen bir sayı (yüzde, doğruluk,
   getiri, Sharpe oranı, süre, eşik) yazma. Makalede geçen bir sonucu aktarıyorsan kaynağa
   atfet: "The paper reports ...".
2. **possible_strategy_hypotheses** her öğesi test edilebilir bir HİPOTEZDİR, sonuç iddiası
   değildir. Şu biçimi kullan: "Test if <X> improves <Y> versus <Z> on <veri/piyasa>,
   out-of-sample and net of transaction costs." Beklenen etki büyüklüğü yazma ("by 15%",
   "with 90% accuracy" gibi). Makaleden makul bir trading hipotezi çıkmıyorsa (ör. makale
   trading/finans ile ilgili değilse) listeyi BOŞ bırak; hipotez uydurma.
3. **Tavsiye dili yasak.** "traders/investors should", "buy/sell", "implement a X% reduction",
   "allocate X% of the portfolio", "directly applicable", "guaranteed", "will outperform"
   gibi yönlendirici veya kesinlik bildiren ifadeler kullanma. Çıktı her zaman hipotez +
   test noktasıdır, tavsiye değildir.
4. **Örneklem dışı ve maliyet.** Strateji hipotezleri out-of-sample değerlendirmeyi ve işlem
   maliyetlerini (komisyon + slippage) anmalı.
5. **risk_warnings** overfit, veri sızıntısı (look-ahead) ve maliyet risklerini içermeli.
6. **main_claim her zaman doldurulur:** metnin ana katkısını ya da (kitap/ders kitabıysa)
   kapsadığı ana konuları metne dayanarak tek cümleyle özetle. Metnin konusunu özetlemek
   uydurma DEĞİLDİR; uydurma, metinde olmayan bir sonuç ya da rakam eklemektir.
7. Diğer alanlarda bilinmeyeni boş bırak; makalede olmayan hiçbir şeyi uydurma.
