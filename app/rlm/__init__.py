"""RLM — Recursive / Reasoning Language Model Controller.

Bu paket yeni bir bilgi deposu DEĞİLDİR; mevcut RAG retrieval ve doğrulama
modüllerini (app/memory, app/verification) çok-adımlı, kaynaklı ve denetimli bir
cevap akışında orkestre eden bir KONTROL katmanıdır.

Akış (özet):
    Kullanıcı sorusu
      → görev sınıflandırma (task_classifier)
      → reasoning plan
      → çok-turlu retrieval + sorgu yeniden-formülasyonu
      → kanıt yeterlilik skoru (evidence_builder)
      → taslak cevap (LLM)
      → iddia çıkarımı (claim_extractor) + atıf/dayanak doğrulama
      → çelişki kontrolü
      → güven skoru + çekimserlik
      → yapısal nihai cevap + run logları (rlm_store)

Mutlak kurallar (CLAUDE.md): canlı trading sinyali üretmez; desteklenmeyen
iddiayı nihai cevaba koymaz; eksik bağlamda uydurmaz; determinizm seed ile.
"""

from __future__ import annotations

from app.rlm.rlm_controller import RlmController, RlmResult

__all__ = ["RlmController", "RlmResult"]
